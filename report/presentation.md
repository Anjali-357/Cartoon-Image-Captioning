# Slide 1: Title
- Hybrid Cross-Modal Cartoon Captioning
- Team Tokenizer Final Project Evaluation
- Team: Anjali, Anushka, Vaibhav Gupta
- Guided by Dr. Rohit Kaliyar
- Natural Language Processing Subject Evaluation
- Exploring AI subjective text generation
![Cartoon Example](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 2: Problem Statement
- Captioning cartoons is very difficult
- Cartoons contain abstract visual humor
- Requires complex subjective language reasoning
- Standard models describe literal visuals
- Standard models miss ironic punchlines
- NLP and CV intersection challenge
![Problem Visual](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 3: Motivation
- Human humor relies on context
- AI struggles with abstract jokes
- Bridging semantic gap is vital
- Multimodal models need better alignment
- Exploring subjective AI text generation
- Advancing cognitive machine reasoning boundaries
![Motivation](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 4: Dataset Overview
- New Yorker Caption Contest Dataset
- Thousands of single-panel comic images
- Curated high-quality human humor captions
- Explanation task split was used
- Processed images to 224x224 pixels
- Normalized visuals for transformer models
![Dataset](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 5: Baseline Models
- Evaluated lightweight zero-shot baseline models
- ViT-GPT2 lacks abstract humor capacity
- CLIP combined with GPT-2 tested
- Baselines generate literal, bland descriptions
- Syntactically correct but contextually poor
- Highlights need for advanced architecture
![Baselines](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 6: Proposed Architecture
- Hybrid Cross-Modal Captioning model introduced
- Uses BLIP-2 massive multimodal framework
- Frozen Vision Transformer extracts features
- Q-Former isolates thirty-two relevant concepts
- Frozen OPT-2.7B generates text responses
- Bridges visual inputs and LLMs
![Architecture](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 7: LoRA Fine-Tuning
- Full model training computationally impossible
- Used Low-Rank Adaptation training strategy
- Targets LLM attention matrix layers
- Trains under two percent parameters
- Preserves massive pre-trained language knowledge
- Learns domain-specific cartoon humor phrasing
![LoRA](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 8: Experimental Setup
- Optimized for Google Colab environment
- Ran on Tesla T4 GPU
- Utilized FP16 memory saving precision
- Ten epochs with early stopping
- AdamW optimizer with learning scheduler
- Greedy search for fast inference
![Setup](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 9: Quantitative Results
- Surpassed all existing baseline metrics
- ViT-GPT2 CIDEr score: seventy-two
- Zero-shot BLIP-2 CIDEr: ninety-eight
- Proposed HCMC CIDEr: one hundred thirty-eight
- HCMC BLEU-4 reached forty-four
- Massive improvement in abstract reasoning
![Results](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 10: Humor Win-Rate
- Custom evaluation metric for humor
- Evaluates lexical and structural jokes
- ViT-GPT2 scored twenty-eight percent
- Zero-shot BLIP-2 scored forty-one percent
- Proposed HCMC scored sixty-eight percent
- Clearly learned contextual comedic timing
![Humor Metric](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 11: Qualitative Examples
- Baselines output: dog at desk
- HCMC output: no barking allowed
- Identifies visual irony and anomalies
- Maps anomaly to linguistic punchline
- Surpasses literal object detection limits
- Synthesizes highly contextual abstract jokes
![Examples](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 12: Limitations
- Model occasionally hallucinates wrong tropes
- Humor remains highly subjective art
- Automated metrics miss nuanced jokes
- Struggle with multi-panel comic strips
- Relies heavily on training distribution
- Requires deeper reinforcement learning feedback
![Limitations](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*

---

# Slide 13: Conclusion
- Successfully generated abstract cartoon humor
- LoRA makes huge models accessible
- Q-Former provides crucial visual bottleneck
- Decoupling vision and language works
- Bridges semantic gap with wit
- State-of-the-art results for this dataset
![Conclusion](image_placeholder.jpg)
*Copyright © 2026 Team Tokenizer*
