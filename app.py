# app.py
import os
import time
from typing import List, Dict, Any
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

# huggingface_hub Inference client
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "models/sentiment_model.joblib"  # your toxicity model

# 🛡️ Secure: Load HuggingFace token from environment variable
HF_TOKEN = os.getenv("HF_API_KEY")  # <-- NO TOKEN inside the file

# Cardiff sentiment model
HF_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)
CORS(app)

# toxicity model (scikit-learn/joblib)
model = None

# HF Inference client (only initialize if token exists)
client = None
if HF_TOKEN:
    client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
else:
    print("[WARNING] HF_API_KEY not found in environment variables. Sentiment may not work.")


# -------------------------
# Load toxicity model
# -------------------------
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("[app] Toxicity model loaded from", MODEL_PATH)
    else:
        model = None
        print("[app] No toxicity model found at", MODEL_PATH)


load_model()


# -------------------------
# Homepage
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------
# Label mapping for HF model
# -------------------------
HF_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive",
}


def normalize_label(raw_label: str) -> str:
    if not raw_label:
        return "neutral"
    k = raw_label.strip().upper()
    return HF_LABEL_MAP.get(k, "neutral")


# -------------------------
# HuggingFace Sentiment Function
# -------------------------
def hf_sentiment(texts: List[str], max_retries: int = 3, backoff: float = 1.5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if client is None:
        print("[hf_sentiment] No HF client initialized (missing HF_API_KEY).")
        return [{"text": t, "label": "neutral", "confidence": 0.0} for t in texts]

    for t in texts:
        attempts = 0
        success = False

        while attempts < max_retries and not success:
            try:
                res = client.text_classification(text=t, model=HF_MODEL_ID)

                if isinstance(res, list) and len(res) > 0:
                    best = max(res, key=lambda x: float(x.get("score", 0.0)))
                    raw_label = best.get("label")
                    label = normalize_label(raw_label)
                    confidence = float(best.get("score", 0.0))
                    results.append({"text": t, "label": label, "confidence": confidence})
                    success = True
                    break

                print(f"[hf_sentiment] Unexpected HF response for '{t}': {res}")
                break

            except Exception as e:
                attempts += 1
                sleep_time = backoff ** attempts
                print(f"[hf_sentiment] Error (attempt {attempts}) for '{t}': {e}")
                time.sleep(sleep_time)

        if not success:
            results.append({"text": t, "label": "neutral", "confidence": 0.0})

    return results


# -------------------------
# Toxicity Function
# -------------------------
def toxic_predict(texts: List[str]) -> List[Dict[str, Any]]:
    if model is None:
        return [{"text": t, "toxic": False} for t in texts]

    try:
        preds = model.predict(texts)
    except Exception as e:
        print("[toxic_predict] model.predict error:", e)
        preds = [0] * len(texts)

    return [{"text": t, "toxic": int(p) == 1} for t, p in zip(texts, preds)]


# -------------------------
# Combined Endpoint
# -------------------------
@app.route("/analyze_bulk", methods=["POST"])
def analyze_bulk():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])

    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Expected JSON: {\"texts\": [\"...\"]}"}), 400

    sentiments = hf_sentiment(texts)
    toxicity = toxic_predict(texts)

    final = []
    for i, text in enumerate(texts):
        s = sentiments[i]
        t = toxicity[i]

        label = s["label"]
        confidence = s["confidence"]

        if t["toxic"]:
            label = "negative"

        final.append({
            "text": text,
            "sentiment": label,
            "toxic": t["toxic"],
            "confidence": confidence
        })

    return jsonify({"results": final})


# -------------------------
# Old toxicity endpoint
# -------------------------
@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    if model is None:
        return jsonify({"error": "Train a model first"}), 400

    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "No texts received"}), 400

    preds = model.predict(texts)
    return jsonify({
        "predictions": [{"text": t, "label": int(p)} for t, p in zip(texts, preds)]
    })


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

