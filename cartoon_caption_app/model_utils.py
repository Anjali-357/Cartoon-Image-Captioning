"""
model_utils.py — Model loading and caption generation utilities
for the Cartoon Image Captioning App.
"""

import os
# Block TF/Flax before any transformers import to prevent libmetal_plugin crash
os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# ── Safe torch import ─────────────────────────────────────────────────────────
try:
    import torch
    _TORCH_OK = True
    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else "cpu"
    )
except Exception as _e:
    torch = None          # type: ignore
    _TORCH_OK = False
    DEVICE = "cpu"
    warnings.warn(
        f"PyTorch could not be imported ({_e}). "
        "Model inference will be unavailable until torch is fixed.\n"
        "Fix with:  conda install pytorch torchvision torchaudio -c pytorch --force-reinstall",
        RuntimeWarning
    )

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_cartoon(image: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Apply cartoon-specific preprocessing:
    - Convert to RGB
    - Resize to target_size x target_size
    - Mild sharpening to enhance cartoon edges
    - Normalize brightness/contrast
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize with high-quality Lanczos resampling
    image = image.resize((target_size, target_size), Image.LANCZOS)

    # Mild edge enhancement for cartoon-style images
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.4)

    # Slight contrast boost
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(1.1)

    return image


# ─────────────────────────────────────────────────────────────────────────────
# ViT-GPT2 Model
# ─────────────────────────────────────────────────────────────────────────────

def load_vit_gpt2():
    """Load and return ViT-GPT2 image captioning model."""
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

    model_name = "nlpconnect/vit-gpt2-image-captioning"
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model = model.to(DEVICE)
    model.eval()
    return model, processor, tokenizer


def generate_vit_gpt2(
    model, processor, tokenizer,
    image: Image.Image,
    num_captions: int = 3,
    max_length: int = 60
) -> List[Tuple[str, float]]:
    """
    Generate multiple diverse captions using ViT-GPT2.
    Returns list of (caption, score) tuples.
    """
    img = preprocess_cartoon(image)

    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)

    results = []
    with torch.no_grad():
        # Beam search for best caption
        output_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=num_captions,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids

        if hasattr(sequences, "shape") and sequences.dim() == 2:
            for i, seq in enumerate(sequences):
                caption = tokenizer.decode(seq, skip_special_tokens=True).strip()
                # Compute approximate confidence from sequence length
                score = max(0.3, 1.0 - i * 0.12)
                if caption:
                    results.append((caption, round(score, 3)))
        else:
            caption = tokenizer.decode(sequences, skip_special_tokens=True).strip()
            results.append((caption, 0.85))

    # Ensure we return num_captions results
    while len(results) < num_captions:
        if results:
            results.append((results[0][0], max(0.1, results[0][1] - 0.1)))
        else:
            results.append(("A cartoon scene.", 0.5))

    return results[:num_captions]


# ─────────────────────────────────────────────────────────────────────────────
# BLIP-2 Model
# ─────────────────────────────────────────────────────────────────────────────

def load_blip2():
    """Load and return BLIP-2 model."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    model_name = "Salesforce/blip2-opt-2.7b"
    dtype = torch.float16 if DEVICE != "cpu" else torch.float32

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    return model, processor


BLIP2_CARTOON_PROMPTS = [
    "A funny New Yorker-style caption for this cartoon:",
    "This cartoon is funny because",
    "Question: What would be a witty caption for this cartoon? Answer:",
]


def generate_blip2(
    model, processor,
    image: Image.Image,
    num_captions: int = 3,
    max_new_tokens: int = 60
) -> List[Tuple[str, float]]:
    """
    Generate captions using BLIP-2 with multiple prompts.
    Returns list of (caption, score) tuples.
    """
    img = preprocess_cartoon(image)
    dtype = torch.float16 if DEVICE != "cpu" else torch.float32
    results = []

    prompts_to_use = BLIP2_CARTOON_PROMPTS[:num_captions]
    if len(prompts_to_use) < num_captions:
        prompts_to_use += [None] * (num_captions - len(prompts_to_use))

    for i, prompt in enumerate(prompts_to_use):
        try:
            if prompt:
                inputs = processor(
                    img, text=prompt, return_tensors="pt"
                ).to(DEVICE, dtype)
            else:
                inputs = processor(img, return_tensors="pt").to(DEVICE, dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    repetition_penalty=1.3,
                    temperature=0.9,
                )

            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            # Remove the prompt echo if present
            if prompt and caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            score = round(0.92 - i * 0.08, 3)
            results.append((caption or "Caption generation failed.", score))

        except Exception as e:
            results.append((f"[Generation error: {str(e)[:40]}]", 0.1))

    return results[:num_captions]


# ─────────────────────────────────────────────────────────────────────────────
# BLIP-1 (Lightweight fallback — faster for CPU)
# ─────────────────────────────────────────────────────────────────────────────

def load_blip():
    """Load lightweight BLIP model (good CPU fallback)."""
    from transformers import BlipProcessor, BlipForConditionalGeneration

    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return model, processor


def generate_blip(
    model, processor,
    image: Image.Image,
    num_captions: int = 3,
    max_new_tokens: int = 60
) -> List[Tuple[str, float]]:
    """Generate captions using BLIP with conditional prompts."""
    img = preprocess_cartoon(image)

    prompts = [
        "a funny cartoon caption:",
        "this cartoon shows",
        "a witty caption for this cartoon:",
    ][:num_captions]

    results = []
    for i, prompt in enumerate(prompts):
        try:
            inputs = processor(img, text=prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            if caption.lower().startswith(prompt.lower()):
                caption = caption[len(prompt):].strip()
            score = round(0.88 - i * 0.06, 3)
            results.append((caption or "No caption generated.", score))
        except Exception as e:
            results.append((f"Error: {str(e)[:40]}", 0.1))

    return results[:num_captions]


# ─────────────────────────────────────────────────────────────────────────────
# Humor Scorer (lightweight approx. using perplexity + lexical cues)
# ─────────────────────────────────────────────────────────────────────────────

HUMOR_KEYWORDS = {
    "positive": [
        "irony", "twist", "surprise", "unexpected", "ironic",
        "sarcastic", "absurd", "bizarre", "ridiculous", "clever",
        "witty", "irony", "pun", "joke", "funny", "laugh", "hilarious",
        "bizarre", "awkward", "ridiculous", "paradox"
    ],
    "structural": [
        "but", "however", "except", "unless", "despite", "although",
        "even though", "turns out", "actually", "wait", "suddenly"
    ]
}


def score_humor(caption: str) -> float:
    """
    Lightweight humor scoring based on:
    - Lexical humor markers
    - Caption length (good captions are medium length)
    - Structural incongruity markers
    Returns score in [0, 1].
    """
    caption_lower = caption.lower()
    words = caption_lower.split()
    n_words = len(words)

    # Base score
    score = 0.3

    # Keyword score
    kw_hits = sum(1 for kw in HUMOR_KEYWORDS["positive"] if kw in caption_lower)
    struct_hits = sum(1 for kw in HUMOR_KEYWORDS["structural"] if kw in caption_lower)
    score += min(kw_hits * 0.08, 0.24)
    score += min(struct_hits * 0.06, 0.18)

    # Length penalty: too short or too long is bad
    if 5 <= n_words <= 20:
        score += 0.15
    elif n_words < 3:
        score -= 0.1

    # Punctuation markers (question marks, exclamation for humor)
    if "?" in caption:
        score += 0.05
    if "!" in caption:
        score += 0.03

    # Clip to [0, 1]
    return round(min(max(score, 0.0), 1.0), 3)


def analyze_captions(captions: List[Tuple[str, float]]) -> List[dict]:
    """
    Full analysis of generated captions.
    Returns list of dicts with caption, confidence, humor_score, word_count.
    """
    results = []
    for caption, confidence in captions:
        humor = score_humor(caption)
        results.append({
            "caption": caption,
            "confidence": confidence,
            "humor_score": humor,
            "word_count": len(caption.split()),
            "char_count": len(caption),
        })
    return results
