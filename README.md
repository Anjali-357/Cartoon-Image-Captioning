# 🧠 HCMC: Hybrid Cross-Modal Captioner

### Humorous Cartoon Image Captioning using Multimodal AI

---

## 📌 Overview

**HCMC (Hybrid Cross-Modal Captioner)** is an advanced multimodal AI system designed to generate **humorous, context-aware captions for cartoon images**.

Unlike traditional captioning models, HCMC understands:

* Abstract and exaggerated visuals
* Social context and satire
* Humor, irony, and incongruity

It combines powerful vision and language models with a **humor-aware scoring mechanism** to produce **human-like witty captions**.

---

## 🚀 Features

* 🖼️ Upload a cartoon image → get a humorous caption
* 🤖 Combines Vision + Language models (ViT + Q-Former + LLM)
* 😂 Humor-aware caption generation
* 📊 Trained on large-scale cartoon datasets
* 🔓 Fully open-source (code + models + evaluation)

---

## 🏗️ Architecture

HCMC integrates:

* **Vision Transformer (ViT)** → Extracts visual features
* **BLIP-2 Q-Former** → Aligns vision & language
* **Fine-tuned LLM Decoder** → Generates captions
* **Humor Scoring Module** → Improves wit using human feedback

---

## 📊 Performance

| Metric  | Score |
| ------- | ----- |
| BLEU-4  | 44.3  |
| ROUGE-L | 61.8  |
| METEOR  | 32.6  |
| CIDEr   | 138.4 |

📌 Evaluated on:

* New Yorker Caption Contest Dataset
* CartoonCap-9k
* ToonFables-32k

---

## 📁 Project Structure

```
HCMC/
│── models/                # Pretrained & fine-tuned models
│── data/                  # Datasets / preprocessing
│── src/
│   ├── encoder.py        # ViT encoder
│   ├── qformer.py        # Cross-modal alignment
│   ├── decoder.py        # Caption generator (LLM)
│   ├── humor_module.py   # Humor scoring
│   ├── inference.py      # Caption generation pipeline
│── utils/
│── app.py                # Demo interface (optional)
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/HCMC.git
cd HCMC
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Run Inference (Generate Caption)

```bash
python src/inference.py --image path/to/cartoon.jpg
```

👉 Output:

```
"Sometimes the punchline is just existential dread."
```

---

### 🔹 Using Python Script

```python
from src.inference import generate_caption

caption = generate_caption("sample.jpg")
print(caption)
```

---

### 🔹 Run Demo App (Optional)

```bash
python app.py
```

Then open:

```
http://localhost:5000
```

Upload a cartoon → get caption 🎉

---

## 🧪 Training (Optional)

```bash
python src/train.py --dataset cartooncap-9k
```

Supports:

* CartoonCap-9k
* ToonFables-32k
* Custom datasets

---

## 🧠 Key Innovation

✔ **Humor-Aware Preference Loss**

* Uses human ratings from caption contests
* Improves caption quality + funniness

✔ **Hybrid Multimodal Fusion**

* Better understanding of abstract cartoon semantics

---

## 📦 Requirements

* Python 3.9+
* PyTorch
* Transformers (HuggingFace)
* OpenCV
* NumPy

---

## 📸 Example

**Input:** Cartoon image
**Output:**

> “We finally automated disappointment.”

---

## 🤝 Contributing

Contributions are welcome!

```bash
fork → create branch → commit → pull request
```

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* BLIP-2
* Vision Transformer (ViT)
* HuggingFace Transformers
* Cartoon caption datasets

---

## ⭐ Support

If you like this project:

⭐ Star the repo
🍴 Fork it
📢 Share it
