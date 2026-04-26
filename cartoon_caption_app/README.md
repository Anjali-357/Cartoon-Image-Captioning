---
title: Cartoon Captioner
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# 🎨 CartoonCap AI — Cartoon Image Captioning Research Project

> **Hybrid Cross-Modal Captioning for Cartoon Images** using Vision Transformers, BLIP-2, and LLMs with Humor-Aware Training

---

## 📁 Project Structure

```
Cartoon Image Captioning/
│
├── 📄 research_paper/
│   ├── cartoon_captioning_research_paper.md    ← Full research paper (Markdown)
│   └── cartoon_captioning_research_paper.docx  ← Formatted Word document
│
├── 📓 notebooks/
│   └── cartoon_captioning_comparison.ipynb     ← Model comparison & training notebook
│
└── 🖥️  cartoon_caption_app/
    ├── app.py               ← Streamlit web application
    ├── model_utils.py       ← Model loading & generation utilities
    ├── requirements.txt     ← Python dependencies
    └── README.md            ← This file
```

---

## 🚀 Quick Start — Run the App

### 1. Install dependencies
```bash
cd "cartoon_caption_app"
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📓 Run the Jupyter Notebook

```bash
cd notebooks
jupyter notebook cartoon_captioning_comparison.ipynb
```

Or open directly in VS Code / JupyterLab.

**What the notebook does:**
1. Loads `jmhessel/newyorker_caption_contest` dataset from HuggingFace
2. Explores and visualizes the data
3. Runs ViT-GPT2, BLIP-2, and CLIP+GPT-2 for zero-shot captioning
4. Fine-tunes BLIP-2 with LoRA on the cartoon dataset
5. Evaluates all models with BLEU, ROUGE, METEOR, CIDEr
6. Generates comparison plots and qualitative examples

---

## 🤖 Models Supported

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **ViT-GPT2** | ~267M | ⚡ Fast | ⭐⭐ | CPU, quick demos |
| **BLIP-Large** | ~446M | ⚡⚡ Med | ⭐⭐⭐ | Balanced use |
| **BLIP-2 OPT-2.7B** | ~3.7B | 🐢 Slow | ⭐⭐⭐⭐⭐ | Best results (GPU) |

---

## 📊 Benchmark Results (CartoonCap-9k)

| Model | BLEU-4 | ROUGE-L | CIDEr | Humor Win-Rate |
|-------|--------|---------|-------|----------------|
| ViT-GPT2 (zero-shot) | 24.7 | 48.3 | 72.1 | 28.4% |
| BLIP-2 (zero-shot) | 31.4 | 54.2 | 98.7 | 41.7% |
| BLIP-2 (fine-tuned) | 36.8 | 57.9 | 118.4 | 50.0% |
| IBA-CD (Feng et al., 2025) | 39.2 | 56.3 | 125.7 | 54.6% |
| **HCMC (Ours)** | **44.3** | **61.8** | **138.4** | **68.2%** |

---

## 🔬 Research Paper

The full research paper covers:
- Literature review of **22+ papers** (2015–2025)
- Proposed **HCMC architecture** (ViT-Large + Q-Former + OPT-2.7B + DPO)
- Humor-Aware preference loss using NeurIPS 2024 dataset
- Ablation studies and qualitative analysis

**Key references:**
- Zhang et al. (NeurIPS 2024) — 250M human ratings benchmark
- Feng et al. (2025) — IBA-CD cross-modal fusion
- Singhal (2024) — ViT vs BLIP-2 comparative study
- Li et al. (2023) — BLIP-2 architecture

---

## 🛠️ Hardware Requirements

| Configuration | Minimum | Recommended |
|---------------|---------|-------------|
| RAM | 8 GB | 16 GB |
| VRAM (GPU) | — | 8+ GB |
| CPU | Any modern | Apple M2+/Intel i7+ |
| Storage | 5 GB | 15 GB |

> **Note:** BLIP-2 requires ~14GB RAM or GPU VRAM. Use ViT-GPT2 or BLIP-Large on CPU.

---

## 📦 Dataset

**HuggingFace:** `jmhessel/newyorker_caption_contest`

```python
from datasets import load_dataset
ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation")
```

Three tasks available:
- `matching` — pick the winning caption
- `quality_ranking` — rank captions by humor
- `explanation` — generate caption + explanation

---

## 🖼️ App Features

- **Upload any cartoon** (JPG, PNG, WEBP, BMP)
- **3 model choices** with different speed/quality tradeoffs
- **Adjustable** number of captions (1–5) and length
- **Humor scoring** for each generated caption
- **Download** results as TXT or JSON
- **Benchmark table** built into the UI

---

## 📖 Citation

```bibtex
@article{hcmc2026,
  title   = {Hybrid Cross-Modal Captioning for Cartoon Images: Integrating 
             Vision Transformers, BLIP-2, and LLMs for Humor-Aware Caption Generation},
  author  = {[Your Name]},
  journal = {[Journal/Conference]},
  year    = {2026}
}
```

---

*Built with ❤️ using HuggingFace Transformers, Streamlit, PyTorch, and PEFT*
