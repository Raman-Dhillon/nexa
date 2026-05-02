# 🌐 NEXA Adaptive Translation System

NEXA is an intelligent machine translation framework that dynamically selects the most suitable model between M2M100 and NLLB based on semantic evaluation metrics. The system improves translation quality by leveraging adaptive decision-making, back-translation, and embedding-based similarity.

## 🚀 Key Features

- 🔄 Adaptive model switching (M2M100 ↔ NLLB)
- 🧠 Semantic evaluation using BERTScore and cosine similarity
- 🔁 Back-translation validation (Punjabi → English)
- 📊 Multiple evaluation metrics: BLEU, BERTScore, Cosine
- ⚡ Streamlit-based interactive UI
- 📈 Visualization of performance metrics

## 🧠 Models Used

- M2M100 (LoRA fine-tuned)
- NLLB-200 (LoRA fine-tuned)

## ⚙️ How It Works

1. Input sentence is translated using M2M100
2. Translation is evaluated using:
   - BERTScore (semantic similarity)
   - Cosine similarity (embedding-based)
3. If quality is below threshold, system switches to NLLB
4. Back-translation is used for validation
5. Best output is selected

## 📊 Evaluation Metrics

- BLEU Score
- BERTScore
- Cosine Similarity

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
