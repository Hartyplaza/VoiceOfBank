# VoiceOfBank — Customer Voice Intelligence from UK Bank Reviews

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![RoBERTa](https://img.shields.io/badge/RoBERTa-Fine--tuned-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-93.6%25-brightgreen?style=flat-square)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.708-brightgreen?style=flat-square)
![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow?style=flat-square&logo=huggingface)

**End-to-end NLP pipeline analysing 30,000 UK bank app reviews**

[LinkedIn](https://linkedin.com/in/hart-ofigwe) · [Portfolio](https://hartyplaza.github.io) · [GitHub](https://github.com/Hartyplaza)

</div>

---

## Overview

VoiceOfBank is a customer voice intelligence system that extracts sentiment, topics, and complaint patterns from 30,000 Google Play reviews across six UK banks — Monzo, Starling, Barclays, HSBC, NatWest, and Lloyds.

The project answers three business questions:

1. Which UK bank has the most positive and most negative customer sentiment — and why?
2. What are customers actually complaining about across different banks?
3. Can we automatically classify a new review into Positive / Neutral / Negative in real time?

---

## Banks Analysed

| Bank | Type | Reviews | Negative Rate |
|------|------|---------|---------------|
| Starling | Challenger | 5,000 | 9.9% |
| Barclays | Traditional | 5,000 | 14.5% |
| Lloyds | Traditional | 5,000 | 13.7% |
| Monzo | Challenger | 5,000 | 27.2% |
| NatWest | Traditional | 5,000 | 39.9% |
| HSBC | Traditional | 5,000 | 48.3% |

---

## Results

### Sentiment Classification

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| VADER (zero-shot) | 0.7798 | 0.5435 |
| RoBERTa (zero-shot) | 0.8695 | 0.6372 |
| XGBoost + TF-IDF (tuned) | 0.8957 | 0.6069 |
| **RoBERTa (fine-tuned)** | **0.9361** | **0.7077** |

Fine-tuned RoBERTa is the best model. Every step of the pipeline improved on the previous — VADER → zero-shot RoBERTa → XGBoost → fine-tuned RoBERTa.

### Per-Class Performance — Fine-tuned RoBERTa

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Positive | 0.97 | 0.98 | 0.97 |
| Negative | 0.88 | 0.92 | 0.90 |
| Neutral | 0.33 | 0.20 | 0.25 |

---

## Key Findings

**Finding 1 — The challenger vs traditional divide is not clean**

The real segmentation by sentiment is:
```
Excellent  : Starling   (9.9% negative)
Good       : Barclays, Lloyds, Monzo  (13-27% negative)
Poor       : NatWest   (39.9% negative)
Very poor  : HSBC      (48.3% negative)
```
Barclays and Lloyds outperform Monzo despite being traditional banks. The real divide is not challenger vs traditional — it is banks that invested in mobile infrastructure vs those that did not.

**Finding 2 — Challenger banks fail on customer service, traditional banks fail on technical infrastructure**

LDA topic modelling on complaint reviews revealed:
- Monzo's top complaint topic is Customer Service at 33.6% — the highest concentration of any topic across any bank
- HSBC's top complaint topic is Login & Authentication at 21% — a direct consequence of complex legacy security architecture
- NatWest complaints concentrate on App Performance and chatbot (Cora) frustration

**Finding 3 — The Feb-Apr 2026 Barclays and Lloyds spike**

Review volume for Barclays and Lloyds spiked 10x in February-April 2026 (38 complaints before vs 385 during). Topic distribution was unchanged between periods — meaning the same issues amplified rather than a new specific incident. This points to broad app degradation from a major update rather than a single failure.

**Finding 4 — Fine-tuning captures domain-specific negative language**

Zero-shot RoBERTa achieved Negative F1 of 0.85. Fine-tuning on banking app reviews pushed this to 0.90. The model learned domain-specific patterns — "locked out", "keeps crashing", "waiting weeks" — that generic Twitter-trained RoBERTa did not associate strongly with negative sentiment.

**Finding 5 — BERTopic vs LDA on short informal text**

BERTopic found 5 topics with one dominant catch-all cluster (85.3% of non-outlier complaints). LDA's 8-topic structure was more granular and interpretable for this dataset. Short informal reviews — often multi-issue in a single sentence — resist fine-grained neural clustering. LDA's forced assignment produces more actionable topic labels.

---

## Pipeline

```
01_scraping.ipynb          LOCAL
  Google Play Scraper → 30,000 reviews (5,000 per bank)
  Banks: Monzo, Starling, Barclays, HSBC, NatWest, Lloyds
        │
        ▼
02_eda.ipynb               LOCAL
  Rating distributions · Review volume over time
  Challenger vs traditional comparison · Reply rate analysis
        │
        ▼
03_preprocessing.ipynb     LOCAL
  Text cleaning · Tokenisation · Lemmatisation
  TF-IDF matrix · Train/test split (80/20 stratified)
        │
        ▼
04_sentiment_vader.ipynb   LOCAL
  VADER rule-based sentiment · Aspect-level scoring
  5 aspects: Customer Service, App & Interface, Fees & Charges,
             Transfer & Speed, Security & Fraud
        │
        ▼
05_sentiment_bert.ipynb    COLAB (GPU)
  cardiffnlp/twitter-roberta-base-sentiment-latest
  Zero-shot inference on 27,142 reviews
  Accuracy: 86.95% | Macro F1: 0.637
        │
        ▼
06_topic_modelling.ipynb   COLAB (GPU)
  LDA (8 topics) + BERTopic (5 topics)
  Complaint topic analysis per bank
  Spike investigation: Feb-Apr 2026 Barclays/Lloyds
        │
        ▼
07_classification.ipynb    COLAB (GPU)
  XGBoost + TF-IDF (RandomizedSearchCV tuned)
  Fine-tuned RoBERTa (3 epochs, fp16)
  Accuracy: 93.61% | Macro F1: 0.708
```

---

## Project Structure

```
VoiceOfBank/
├── notebooks/
│   ├── 01_scraping.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_sentiment_vader.ipynb
│   ├── 05_sentiment_bert.ipynb
│   └── 06_topic_modelling.ipynb
│   └── 07_classification.ipynb
├── src/
│   ├── scraper/
│   ├── preprocessing/
│   ├── sentiment/
│   ├── topics/
│   ├── models/
│   └── api/
├── data/
│   ├── raw/                  reviews_raw.csv
│   ├── processed/            reviews_clean.csv · reviews_vader.csv
│   │                         reviews_bert.csv · topics_lda.csv
│   │                         topics_bert.csv · classification_results.csv
│   └── models/               xgb_sentiment.joblib · roberta_sentiment/
├── app.py                    Streamlit dashboard
└── requirements.txt
```

---

## Quickstart

> **Note:** Notebooks 01-04 include full cell outputs. Notebooks 05-07 (Colab) are saved without outputs to keep file sizes manageable. Run them on Google Colab with a T4 GPU to reproduce all results.

```bash
git clone https://github.com/Hartyplaza/VoiceOfBank
cd VoiceOfBank
pip install -r requirements.txt
```

**Run locally (notebooks 01-04):**
```bash
jupyter notebook notebooks/01_scraping.ipynb
```

**Run on Colab (notebooks 05-07):**

Notebooks 05, 06, and 07 use transformer models (RoBERTa, BERTopic, sentence-transformers) that require a GPU to run in reasonable time. If you do not have a local GPU, run these on Google Colab which provides a free T4 GPU.

- Upload the notebook to [Google Colab](https://colab.research.google.com)
- Go to `Runtime` → `Change runtime type` → select `T4 GPU`
- Upload your `data/processed/` files to Google Drive at `MyDrive/VoiceOfBank/data/processed/`
- Run all cells

> **Note:** Running notebooks 05-07 on CPU is possible but not recommended. RoBERTa inference on 27,000 reviews takes approximately 3 hours on CPU vs 8 minutes on a T4 GPU. BERTopic embedding generation takes approximately 45-90 minutes on CPU vs 3 minutes on GPU. Fine-tuning RoBERTa for 3 epochs takes approximately 4-6 hours on CPU vs 7 minutes on GPU.

---

## Why This Model — FinBERT vs RoBERTa

FinBERT (trained on financial news) was tested first but misclassified app reviews as Neutral — "I love this app" scored Neutral at 86% confidence. App reviews are short, informal, and opinionated — linguistically much closer to Twitter text than financial news. Switching to `cardiffnlp/twitter-roberta-base-sentiment-latest` (trained on 58 million tweets) immediately improved classification quality, confirmed by correctly predicting "I love this app" as Positive at 98.9% confidence.

---

## Stack

| Layer | Technology |
|-------|-----------|
| **Scraping** | google-play-scraper |
| **NLP preprocessing** | NLTK, scikit-learn |
| **Rule-based sentiment** | VADER (nltk) |
| **Transformer sentiment** | HuggingFace Transformers, RoBERTa |
| **Topic modelling** | LDA (sklearn), BERTopic, sentence-transformers |
| **Classification** | XGBoost, fine-tuned RoBERTa |
| **Hyperparameter tuning** | RandomizedSearchCV |
| **Visualisation** | matplotlib, seaborn |
| **Dashboard** | Streamlit, Plotly |

---

## Limitations

- Google Play reviews only — Trustpilot scraping blocked by Cloudflare
- Neutral class remains hard to classify (F1 0.25) — inherent ambiguity in 3-star language
- 5,000 reviews per bank may not capture rare complaint types
- Fine-tuned model trained on star rating labels — borderline 3-star reviews may have noisy labels
- No named entity recognition — cannot automatically identify specific product features being complained about

---

## Author

**Ofigwe Hart** — Data Scientist / ML Engineer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/spaces/Demerchanthart/VoiceOfBank)
[![Portfolio](https://img.shields.io/badge/Portfolio-hartyplaza.github.io-blue?style=flat-square)](https://hartyplaza.github.io)
[![GitHub](https://img.shields.io/badge/GitHub-Hartyplaza-181717?style=flat-square&logo=github)](https://github.com/Hartyplaza)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-hart--ofigwe-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/hart-ofigwe)
