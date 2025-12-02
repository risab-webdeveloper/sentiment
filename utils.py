# utils.py
import re
import nltk
from nltk.corpus import stopwords

def ensure_nltk():
    try:
        stopwords.words("english")
    except:
        nltk.download("stopwords")
        nltk.download("punkt")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+","", text)
    text = re.sub(r"#","", text)
    text = re.sub(r"\s+"," ", text).strip()
    return text
