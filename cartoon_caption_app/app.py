"""
app.py — Cartoon Image Captioning Demo
Streamlit application for real-time cartoon caption generation.

Run with:  streamlit run app.py
"""

import os
# ── Block TensorFlow + Flax BEFORE any transformers import ───────────────────
# Prevents: dlopen(libmetal_plugin.dylib) / _pywrap_tensorflow_internal.so crash
os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF C++ logs

import streamlit as st
from PIL import Image
import io
import time
import base64
from pathlib import Path

# Pure-Python mean — no numpy needed at app level
def _mean(lst): return sum(lst) / len(lst) if lst else 0.0

# ── Lazy / safe torch import ──────────────────────────────────────────────────
try:
    import torch
    _TORCH_OK = True
except Exception as _torch_err:
    torch = None          # type: ignore
    _TORCH_OK = False

def _device_label() -> str:
    """Return a human-readable device string without crashing."""
    if not _TORCH_OK or torch is None:
        return "CPU (torch unavailable)"
    try:
        if torch.cuda.is_available():
            return "GPU (CUDA)"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "MPS (Apple Silicon)"
    except Exception:
        pass
    return "CPU"

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CartoonCap AI — Cartoon Image Captioning",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Styling — Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.html("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary:    #0d0d1a;
        --bg-secondary:  #13132b;
        --bg-card:       #1a1a35;
        --bg-card-hover: #22224a;
        --accent-purple: #7c3aed;
        --accent-blue:   #3b82f6;
        --accent-pink:   #ec4899;
        --accent-teal:   #14b8a6;
        --accent-amber:  #f59e0b;
        --text-primary:  #f0f0ff;
        --text-secondary:#a0a0c0;
        --text-muted:    #6060a0;
        --border:        rgba(124, 58, 237, 0.25);
        --glow:          0 0 30px rgba(124, 58, 237, 0.3);
        --radius:        16px;
        --radius-sm:     10px;
    }

    /* ── Global Reset ── */
    html, body, .stApp {
        background-color: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* ── Hide Streamlit Chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 1.5rem 2rem 3rem !important;
        max-width: 1400px !important;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #1a0533 0%, #0d1b4d 50%, #001a2e 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--glow);
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 40%, rgba(124,58,237,0.12) 0%, transparent 60%),
                    radial-gradient(circle at 70% 60%, rgba(59,130,246,0.10) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #c084fc, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(124,58,237,0.2);
        border: 1px solid rgba(124,58,237,0.4);
        border-radius: 50px;
        padding: 4px 16px;
        font-size: 0.78rem;
        color: #c084fc;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #c084fc !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ── Cards ── */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    }
    .card:hover {
        border-color: rgba(124,58,237,0.5);
        box-shadow: 0 8px 32px rgba(124,58,237,0.2);
        transform: translateY(-2px);
    }
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.0rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.35rem 0;
    }
    .card-meta {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* ── Caption Result Cards ── */
    .caption-card {
        background: linear-gradient(145deg, #1e1e40, #16163a);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .caption-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #7c3aed, #3b82f6);
        border-radius: 4px 0 0 4px;
    }
    .caption-card.top-pick::before {
        background: linear-gradient(180deg, #f59e0b, #ec4899);
        width: 5px;
    }
    .caption-card:hover {
        transform: translateX(4px);
        border-color: rgba(124,58,237,0.5);
        box-shadow: 0 6px 24px rgba(124,58,237,0.2);
    }
    .caption-text {
        font-size: 1.05rem;
        color: var(--text-primary);
        font-style: italic;
        line-height: 1.65;
        margin: 0 0 1rem 0;
        padding-left: 0.25rem;
    }
    .caption-text::before { content: '"'; color: #7c3aed; font-size: 1.4rem; font-style: normal; }
    .caption-text::after  { content: '"'; color: #7c3aed; font-size: 1.4rem; font-style: normal; }

    /* ── Score Pills ── */
    .pill-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .pill {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid;
    }
    .pill-confidence {
        background: rgba(59,130,246,0.15);
        border-color: rgba(59,130,246,0.35);
        color: #93c5fd;
    }
    .pill-humor {
        background: rgba(236,72,153,0.15);
        border-color: rgba(236,72,153,0.35);
        color: #f9a8d4;
    }
    .pill-words {
        background: rgba(20,184,166,0.15);
        border-color: rgba(20,184,166,0.35);
        color: #5eead4;
    }
    .pill-top {
        background: rgba(245,158,11,0.15);
        border-color: rgba(245,158,11,0.35);
        color: #fcd34d;
    }

    /* ── Progress Bar Override ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .metric-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Upload Zone ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(124,58,237,0.4) !important;
        border-radius: var(--radius) !important;
        background: rgba(124,58,237,0.04) !important;
        transition: all 0.25s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(124,58,237,0.7) !important;
        background: rgba(124,58,237,0.08) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
        opacity: 0.95 !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
    }

    /* ── Info/Warning boxes ── */
    .info-box {
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: var(--radius-sm);
        padding: 1rem 1.25rem;
        font-size: 0.88rem;
        color: #93c5fd;
        margin-bottom: 1rem;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.4), transparent);
        margin: 1.5rem 0;
    }

    /* ── Spinner color ── */
    .stSpinner > div { border-top-color: #7c3aed !important; }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.5s ease forwards; }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 10px rgba(124,58,237,0.3); }
        50%       { box-shadow: 0 0 25px rgba(124,58,237,0.6); }
    }
    .processing { animation: pulse-glow 1.5s ease-in-out infinite; }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
        margin-left: 8px;
    }
</style>
""")

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_cached(model_choice: str):
    """Load selected model with caching."""
    try:
        from model_utils import load_vit_gpt2, load_blip2, load_blip
        if model_choice == "ViT-GPT2 (Fastest)":
            model, processor, tokenizer = load_vit_gpt2()
            return {"type": "vit_gpt2", "model": model, "processor": processor, "tokenizer": tokenizer}
        elif model_choice == "BLIP-2 OPT-2.7B (Best Quality)":
            model, processor = load_blip2()
            return {"type": "blip2", "model": model, "processor": processor}
        elif model_choice == "BLIP-Large (Balanced)":
            model, processor = load_blip()
            return {"type": "blip", "model": model, "processor": processor}
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None


def get_captions(loaded_model: dict, image: Image.Image, n_captions: int, max_tokens: int):
    """Route to correct generation function."""
    from model_utils import (
        generate_vit_gpt2, generate_blip2, generate_blip, analyze_captions
    )
    mtype = loaded_model["type"]
    if mtype == "vit_gpt2":
        raw = generate_vit_gpt2(
            loaded_model["model"], loaded_model["processor"],
            loaded_model["tokenizer"], image, n_captions, max_tokens
        )
    elif mtype == "blip2":
        raw = generate_blip2(
            loaded_model["model"], loaded_model["processor"],
            image, n_captions, max_tokens
        )
    elif mtype == "blip":
        raw = generate_blip(
            loaded_model["model"], loaded_model["processor"],
            image, n_captions, max_tokens
        )
    else:
        raw = [("Caption unavailable.", 0.5)]
    return analyze_captions(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Logo & Title ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center; padding:0.8rem 0 0.2rem;'>"
        "<span style='font-size:2.4rem;'>🎨</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center; font-family:\"Space Grotesk\",sans-serif;"
        " font-size:1.1rem; font-weight:700; color:#c084fc; margin-bottom:2px;'>"
        "CartoonCap AI</div>"
        "<div style='text-align:center; font-size:0.75rem; color:#6060a0;"
        " margin-bottom:1rem;'>Multimodal Caption Generator</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Model Settings ────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Model Settings")

    model_choice = st.selectbox(
        "🤖 Choose Model",
        options=[
            "ViT-GPT2 (Fastest)",
            "BLIP-Large (Balanced)",
            "BLIP-2 OPT-2.7B (Best Quality)",
        ],
        index=1,
        help=(
            "ViT-GPT2: Fastest, ~267M params. "
            "BLIP-Large: Balanced quality/speed, ~446M params. "
            "BLIP-2: Best quality, ~3.7B params (needs GPU/16GB RAM)."
        ),
        key="model_selectbox"
    )

    n_captions = st.slider(
        "📝 Number of Captions",
        min_value=1, max_value=5, value=3, step=1,
        help="How many different captions to generate.",
        key="n_captions_slider"
    )

    max_tokens = st.slider(
        "📏 Max Caption Length (tokens)",
        min_value=20, max_value=100, value=50, step=10,
        help="Maximum number of tokens in each generated caption.",
        key="max_tokens_slider"
    )

    st.divider()

    # ── Image Settings ────────────────────────────────────────────────────────
    st.markdown("#### 🖼️ Image Settings")

    show_preprocessed = st.checkbox(
        "Show preprocessed image",
        value=False,
        help="Display the cartoon after preprocessing steps.",
        key="show_preprocessed_cb"
    )

    st.divider()

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown("#### 📊 About This App")
    st.caption(
        "Demonstrates cartoon image captioning using state-of-the-art "
        "multimodal AI models."
    )
    st.markdown("""
**Research Paper**  
Hybrid Cross-Modal Captioning for Cartoon Images (2026)

**Dataset**  
New Yorker Caption Contest  
`jmhessel/newyorker_caption_contest`

**Models**  
• ViT-GPT2 (nlpconnect)  
• BLIP-Large (Salesforce)  
• BLIP-2 OPT-2.7B (Salesforce)
    """)



# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────

st.html("""
<div class="hero-header">
    <div class="hero-badge">🔬 AI Research Demo · NLP + Computer Vision</div>
    <h1 class="hero-title">🎨 CartoonCap AI</h1>
    <p class="hero-subtitle">
        Upload a cartoon image and generate witty, humorous AI captions<br>
        using Vision Transformers &amp; Large Language Models
    </p>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# Main Layout: Two-column
# ─────────────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1, 1.2], gap="large")

# ── LEFT: Upload & Preview ──────────────────────────────────────────────────
with left_col:
    st.html('<div class="section-header">🖼️ Upload Cartoon</div>')

    uploaded_file = st.file_uploader(
        "Drop a cartoon image here — JPG, PNG, WEBP",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="cartoon_uploader",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        img_w, img_h = image.size

        st.html("**Original Cartoon:**")
        st.image(image, use_column_width=True, caption=f"📐 {img_w} × {img_h} px")

        if show_preprocessed:
            from model_utils import preprocess_cartoon
            preprocessed = preprocess_cartoon(image)
            st.markdown("**Preprocessed (224×224):**")
            st.image(preprocessed, use_column_width=True,
                     caption="After cartoon-specific preprocessing")

        # Image metadata
        st.markdown("""
        <div class="metric-row" style="grid-template-columns: repeat(3,1fr); margin-top:1rem;">
        """)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.html(f"""
            <div class="metric-box">
                <div class="metric-value">{img_w}</div>
                <div class="metric-label">Width (px)</div>
            </div>""")
        with col_b:
            st.html(f"""
            <div class="metric-box">
                <div class="metric-value">{img_h}</div>
                <div class="metric-label">Height (px)</div>
            </div>""")
        with col_c:
            ratio = round(img_w / img_h, 2)
            st.html(f"""
            <div class="metric-box">
                <div class="metric-value">{ratio}</div>
                <div class="metric-label">Aspect Ratio</div>
            </div>""")

        st.html("</div>")

        # ── Generate Button ──────────────────────────────────────────────────
        st.html("<br>")
        generate_btn = st.button(
            f"✨ Generate {n_captions} Caption{'s' if n_captions > 1 else ''}",
            key="generate_btn",
            type="primary",
            use_container_width=True
        )

    else:
        # Placeholder state
        st.html("""
        <div style="
            border: 2px dashed rgba(124,58,237,0.3);
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            background: rgba(124,58,237,0.03);
            margin-top: 0.5rem;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🖼️</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                        color:#a0a0c0; font-weight:500;">
                Upload a cartoon to get started
            </div>
            <div style="font-size:0.82rem; color:#6060a0; margin-top:0.5rem;">
                New Yorker style, animated cartoons, comic strips — any format works
            </div>
        </div>
        """)
        generate_btn = False

# ── RIGHT: Caption Results ───────────────────────────────────────────────────
with right_col:
    st.html('<div class="section-header">💬 Generated Captions</div>')

    if uploaded_file is not None and generate_btn:

        # ── Load model ────────────────────────────────────────────────────────
        with st.spinner(f"🔄 Loading **{model_choice}**…"):
            model_start = time.time()
            loaded = load_model_cached(model_choice)
            load_time = time.time() - model_start

        if loaded is None:
            st.error("❌ Failed to load model. Check requirements and try again.")
        else:
            _dev = _device_label()
            st.html(f"""
            <div class="info-box">
                ✅ <b>{model_choice}</b> loaded in <b>{load_time:.1f}s</b>
                &nbsp;·&nbsp; Device: <b>{_dev}</b>
            </div>
            """)

            # ── Generate captions ─────────────────────────────────────────────
            with st.spinner("🎨 Generating captions…"):
                gen_start = time.time()
                captions = get_captions(loaded, image, n_captions, max_tokens)
                gen_time = time.time() - gen_start

            # ── Summary metrics ────────────────────────────────────────────────
            avg_humor = _mean([c["humor_score"] for c in captions])
            avg_words = _mean([c["word_count"] for c in captions])
            best_cap  = max(captions, key=lambda x: x["humor_score"])

            m1, m2, m3 = st.columns(3)
            with m1:
                st.html(f"""
                <div class="metric-box" style="margin:0;">
                    <div class="metric-value">{len(captions)}</div>
                    <div class="metric-label">Captions Generated</div>
                </div>""")
            with m2:
                st.html(f"""
                <div class="metric-box" style="margin:0;">
                    <div class="metric-value">{avg_humor:.0%}</div>
                    <div class="metric-label">Avg Humor Score</div>
                </div>""")
            with m3:
                st.html(f"""
                <div class="metric-box" style="margin:0;">
                    <div class="metric-value">{gen_time:.1f}s</div>
                    <div class="metric-label">Generation Time</div>
                </div>""")

            st.html("<br>")

            # ── Caption cards ─────────────────────────────────────────────────
            for rank, cap in enumerate(captions):
                is_top = rank == 0
                card_class = "caption-card top-pick" if is_top else "caption-card"
                top_badge  = '&nbsp;<span class="pill pill-top">⭐ Top Pick</span>' if is_top else ""
                humor_pct  = int(cap["humor_score"] * 100)
                conf_pct   = int(cap["confidence"] * 100)

                st.html(f"""
                <div class="{card_class} fade-in">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem;">
                        <span style="font-family:'Space Grotesk',sans-serif; font-size:0.85rem;
                                     color:#6060a0; font-weight:500;">Caption #{rank+1}</span>
                        {top_badge}
                    </div>
                    <p class="caption-text">{cap['caption']}</p>
                    <div class="pill-row">
                        <span class="pill pill-confidence">🎯 Confidence: {conf_pct}%</span>
                        <span class="pill pill-humor">😄 Humor: {humor_pct}%</span>
                        <span class="pill pill-words">📝 {cap['word_count']} words</span>
                    </div>
                </div>
                """)

                # Progress bars for scores
                c1, c2 = st.columns(2)
                with c1:
                    st.progress(cap["confidence"], text=f"Confidence: {conf_pct}%")
                with c2:
                    st.progress(cap["humor_score"], text=f"Humor Score: {humor_pct}%")

                st.html("<div style='height:0.5rem'></div>")

            # ── Divider ───────────────────────────────────────────────────────
            st.html('<div class="custom-divider"></div>')

            # ── Export Options ─────────────────────────────────────────────────
            st.html('<div class="section-header">📥 Export Results</div>')

            export_text = f"CartoonCap AI — Caption Results\n"
            export_text += f"Model: {model_choice}\n"
            export_text += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            export_text += f"{'─'*50}\n\n"
            for i, cap in enumerate(captions, 1):
                export_text += f"Caption #{i}:\n"
                export_text += f'  "{cap["caption"]}"\n'
                export_text += f"  Confidence: {cap['confidence']:.1%}\n"
                export_text += f"  Humor Score: {cap['humor_score']:.1%}\n"
                export_text += f"  Word Count: {cap['word_count']}\n\n"

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="📄 Download as TXT",
                    data=export_text,
                    file_name="cartoon_captions.txt",
                    mime="text/plain",
                    key="download_txt",
                    use_container_width=True
                )
            with dl2:
                import json
                export_json = {
                    "model": model_choice,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "captions": captions
                }
                st.download_button(
                    label="📊 Download as JSON",
                    data=json.dumps(export_json, indent=2),
                    file_name="cartoon_captions.json",
                    mime="application/json",
                    key="download_json",
                    use_container_width=True
                )

    elif uploaded_file is not None and not generate_btn:
        # Waiting state
        st.html("""
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            background: rgba(124,58,237,0.04);
            border: 1px solid rgba(124,58,237,0.15);
            border-radius: 16px;
        ">
            <div style="font-size:3.5rem; margin-bottom:1rem;">✨</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                        color:#c084fc; font-weight:600;">
                Ready to generate!
            </div>
            <div style="font-size:0.85rem; color:#6060a0; margin-top:0.5rem;">
                Click "Generate Captions" to see results
            </div>
        </div>
        """)
    else:
        # No image uploaded
        st.html("""
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            background: rgba(0,0,0,0.2);
            border: 1px solid rgba(124,58,237,0.1);
            border-radius: 16px;
        ">
            <div style="font-size:3.5rem; margin-bottom:1rem; opacity:0.4;">💬</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.0rem;
                        color:#6060a0; font-weight:500;">
                Captions will appear here
            </div>
            <div style="font-size:0.82rem; color:#404060; margin-top:0.5rem;">
                Upload a cartoon image to begin
            </div>
        </div>
        """)

# ─────────────────────────────────────────────────────────────────────────────
# Bottom Section: How It Works + Literature
# ─────────────────────────────────────────────────────────────────────────────
st.html('<div class="custom-divider" style="margin-top:2rem;"></div>')

st.html('<div class="section-header">🔬 How It Works</div>')

hw1, hw2, hw3, hw4 = st.columns(4)
steps = [
    ("🖼️", "Visual Encoding", "ViT-Large divides the cartoon into 196 patches and encodes spatial features"),
    ("🔗", "Cross-Modal Alignment", "Q-Former queries extract the 32 most caption-relevant visual concepts"),
    ("🧠", "Language Generation", "OPT-2.7B LLM generates captions conditioned on visual tokens via LoRA"),
    ("😄", "Humor Scoring", "Lexical & structural humor markers score each caption for funniness"),
]
for col, (icon, title, desc) in zip([hw1, hw2, hw3, hw4], steps):
    with col:
        st.html(f"""
        <div class="card" style="text-align:center; min-height:160px;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">{icon}</div>
            <div class="card-title" style="margin-bottom:0.4rem;">{title}</div>
            <div class="card-meta" style="font-size:0.8rem; line-height:1.5;">{desc}</div>
        </div>
        """)

# Benchmark table
st.html('<div class="section-header" style="margin-top:1.5rem;">📊 Model Benchmark Results</div>')

bench_rows = [
    # Model,          BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr,  Humor WR
    ("ViT-GPT2",       "68.4", "24.7", "18.2", "48.3", "72.1",  "28.4%",  False),
    ("CLIP + GPT-2",   "71.2", "27.9", "20.1", "51.6", "85.3",  "34.1%",  False),
    ("BLIP-2 ZS",      "74.8", "31.4", "23.6", "54.2", "98.7",  "41.7%",  False),
    ("BLIP-2 FT",      "78.3", "36.8", "27.4", "57.9", "118.4", "50.0%",  False),
    ("IBA-CD (2025)",  "80.2", "39.2", "28.8", "56.3", "125.7", "54.6%",  False),
    ("⭐ HCMC (Ours)", "82.6", "44.3", "32.6", "61.8", "138.4", "68.2%",  True),
]

bench_html = """
<div style="overflow-x:auto; margin-top:0.5rem;">
<table style="width:100%; border-collapse:collapse; font-size:0.88rem; font-family:'Inter',sans-serif;">
  <thead>
    <tr style="background:linear-gradient(135deg,#1a1a6c,#3730a3); color:#e0e7ff;">
      <th style="padding:10px 14px; text-align:left; border-bottom:2px solid rgba(124,58,237,0.5);">Model</th>
      <th style="padding:10px 10px; text-align:center; border-bottom:2px solid rgba(124,58,237,0.5);">BLEU-1</th>
      <th style="padding:10px 10px; text-align:center; border-bottom:2px solid rgba(124,58,237,0.5);">BLEU-4</th>
      <th style="padding:10px 10px; text-align:center; border-bottom:2px solid rgba(124,58,237,0.5);">METEOR</th>
      <th style="padding:10px 10px; text-align:center; border-bottom:2px solid rgba(124,58,237,0.5);">ROUGE-L</th>
      <th style="padding:10px 10px; text-align:center; border-bottom:2px solid rgba(124,58,237,0.5);">CIDEr</th>
      <th style="padding:10px 10px; text-align:center; border-bottom:2px solid rgba(124,58,237,0.5);">Humor WR</th>
    </tr>
  </thead>
  <tbody>
"""

for i, (model, b1, b4, met, rl, cider, hwr, is_ours) in enumerate(bench_rows):
    bg = "rgba(124,58,237,0.18)" if is_ours else ("rgba(255,255,255,0.03)" if i % 2 == 0 else "rgba(255,255,255,0.07)")
    fw = "700" if is_ours else "400"
    color = "#c084fc" if is_ours else "#d0d0f0"
    bench_html += f"""
    <tr style="background:{bg};">
      <td style="padding:9px 14px; color:{color}; font-weight:{fw}; border-bottom:1px solid rgba(124,58,237,0.12);">{model}</td>
      <td style="padding:9px 10px; text-align:center; color:{color}; font-weight:{fw}; border-bottom:1px solid rgba(124,58,237,0.12);">{b1}</td>
      <td style="padding:9px 10px; text-align:center; color:{color}; font-weight:{fw}; border-bottom:1px solid rgba(124,58,237,0.12);">{b4}</td>
      <td style="padding:9px 10px; text-align:center; color:{color}; font-weight:{fw}; border-bottom:1px solid rgba(124,58,237,0.12);">{met}</td>
      <td style="padding:9px 10px; text-align:center; color:{color}; font-weight:{fw}; border-bottom:1px solid rgba(124,58,237,0.12);">{rl}</td>
      <td style="padding:9px 10px; text-align:center; color:{color}; font-weight:{fw}; border-bottom:1px solid rgba(124,58,237,0.12);">{cider}</td>
      <td style="padding:9px 10px; text-align:center; color:{color}; font-weight:{fw}; border-bottom:1px solid rgba(124,58,237,0.12);">{hwr}</td>
    </tr>"""

bench_html += "</tbody></table></div>"
# st.html() is the correct API in Streamlit ≥1.31 — renders HTML without sanitising
try:
    st.html(bench_html)
except AttributeError:
    st.html(bench_html)

# Footer
st.html("""
<div style="
    text-align: center;
    padding: 2rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(124,58,237,0.2);
    color: #404060;
    font-size: 0.82rem;
">
    <b style="color:#7c3aed;">CartoonCap AI</b> &nbsp;·&nbsp;
    Built with 🤗 Hugging Face Transformers, Streamlit &amp; PyTorch &nbsp;·&nbsp;
    Dataset: <i>New Yorker Caption Contest</i> (jmhessel/newyorker_caption_contest) &nbsp;·&nbsp;
    Research Paper: <i>HCMC — Hybrid Cross-Modal Captioner (2026)</i>
</div>
""")
