# train.py
import pandas as pd
import joblib
import os
from utils import ensure_nltk, clean_text
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "models/sentiment_model.joblib"


# -----------------------------
# AUTO-DETECT TEXT & LABEL COLUMNS
# -----------------------------
def autodetect_columns(df):

    text_candidates = [
        "text", "Text", "tweet", "content", "sentence", "comment", "message"
    ]

    # **IMPORTANT: Added "hate_label" here**
    label_candidates = [
        "Label", "label", "sentiment", "target", "class", "output", "hate_label"
    ]

    text_col = None
    label_col = None

    for col in df.columns:
        col_clean = col.strip()
        if col_clean in text_candidates:
            text_col = col
        if col_clean in label_candidates:
            label_col = col

    if text_col is None:
        raise KeyError(
            f"No text column found. CSV columns = {list(df.columns)}"
        )

    if label_col is None:
        raise KeyError(
            f"No label column found. CSV columns = {list(df.columns)}"
        )

    return text_col, label_col


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(csv_path):
    ensure_nltk()

    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines='skip')

    # auto detect columns
    text_col, label_col = autodetect_columns(df)

    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df = df.dropna(subset=[text_col, label_col])

    X = df[text_col].tolist()
    y = df[label_col].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    acc = accuracy_score(y_val, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    return acc
