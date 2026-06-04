# VoiceOfBank — FastAPI Backend
# Serves the fine-tuned RoBERTa model for real-time sentiment classification

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import numpy as np
import os

BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="VoiceOfBank API",
    description="Real-time sentiment classification for UK bank reviews",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "data/models/roberta_sentiment"))
XGB_PATH   = os.getenv("XGB_PATH",   str(BASE_DIR / "data/models/xgb_sentiment.joblib"))
TFIDF_PATH = os.getenv("TFIDF_PATH", str(BASE_DIR / "data/models/tfidf_vectorizer.joblib"))
LE_PATH    = os.getenv("LE_PATH",    str(BASE_DIR / "data/models/label_encoder.joblib"))

roberta_classifier = None
xgb_model          = None
tfidf_vectorizer    = None
label_encoder       = None


def load_models():
    global roberta_classifier, xgb_model, tfidf_vectorizer, label_encoder

    import joblib
    from transformers import pipeline

    print("Loading XGBoost model...")
    xgb_model       = joblib.load(XGB_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    label_encoder    = joblib.load(LE_PATH)
    print("XGBoost loaded.")

    print("Loading fine-tuned RoBERTa...")
    roberta_classifier = pipeline(
        "text-classification",
        model     = MODEL_PATH,
        tokenizer = MODEL_PATH,
        device    = -1,  # CPU for HF Spaces
        top_k     = None,
    )
    print("RoBERTa loaded.")


@app.on_event("startup")
async def startup_event():
    load_models()


# ── Schemas ───────────────────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str
    bank: Optional[str] = None


class SentimentResponse(BaseModel):
    text: str
    bank: Optional[str]
    roberta_label: str
    roberta_confidence: float
    roberta_prob_positive: float
    roberta_prob_neutral: float
    roberta_prob_negative: float
    xgb_label: str
    xgb_confidence: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name"   : "VoiceOfBank API",
        "version": "1.0.0",
        "status" : "running",
    }


@app.get("/health")
def health():
    return {
        "status"  : "healthy",
        "roberta" : roberta_classifier is not None,
        "xgboost" : xgb_model is not None,
    }


@app.post("/predict", response_model=SentimentResponse)
def predict(request: ReviewRequest):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Review text cannot be empty")

    text = request.text.strip()

    # RoBERTa prediction
    roberta_results = roberta_classifier(text, truncation=True, max_length=128)[0]
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

    probs = {label_map.get(r["label"], r["label"]): r["score"] for r in roberta_results}
    roberta_label = max(probs, key=probs.get)
    roberta_conf  = probs[roberta_label]

    # XGBoost prediction
    import re
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    lemmatizer = WordNetLemmatizer()
    text_clean = text.lower()
    text_clean = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text_clean)
    text_clean = re.sub(r"[^a-z\s]", " ", text_clean)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()
    tokens = word_tokenize(text_clean)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) >= 3]
    text_clean = " ".join(tokens)

    X = tfidf_vectorizer.transform([text_clean])
    xgb_pred   = xgb_model.predict(X)[0]
    xgb_proba  = xgb_model.predict_proba(X)[0]
    xgb_label  = label_encoder.inverse_transform([xgb_pred])[0]
    xgb_conf   = float(xgb_proba.max())

    return SentimentResponse(
        text                  = text,
        bank                  = request.bank,
        roberta_label         = roberta_label,
        roberta_confidence    = round(roberta_conf, 4),
        roberta_prob_positive = round(probs.get("Positive", 0), 4),
        roberta_prob_neutral  = round(probs.get("Neutral", 0), 4),
        roberta_prob_negative = round(probs.get("Negative", 0), 4),
        xgb_label             = xgb_label,
        xgb_confidence        = round(xgb_conf, 4),
    )
