import re 
import pandas as pd
import unicodedata
from langdetect import detect
from datetime import datetime
from typing import Tuple

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def language_filter(text: str, lang="en") -> str:
    try:
        return detect(text)
    except:
        return "unknown"

def fix_date(value):
    if pd.isna(value):
        return None
    if isinstance(value, str) and "$date" in value:
        try:
            millis = int(re.search(r"\d+", value).group())
            return pd.to_datetime(millis, unit="ms")
        except Exception:
            return None
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None

def preprocess_dataset(df: pd.DataFrame, filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df[["id", "title", "text", "url", "authors", "source", "publish_date"]].copy()
    df["published_at"] = df["publish_date"].apply(fix_date)
    df["clean_text"] = df["text"].apply(clean_text)
    df["lang"] = df["clean_text"].apply(language_filter)

    news_df = df.rename(
        columns={
            "title": "title",
            "text": "text",
            "source": "source",
            "url": "url",
        }
    )[["id", "title", "text", "source", "published_at", "url", "lang"]]

    filename_lower = filename.lower()
    if "fake" in filename_lower:
        label_value = True
    elif "real" in filename_lower:
        label_value = False
    else:
        label_value = None

    label_df = pd.DataFrame({
        "news_id": df["id"],
        "label": label_value,
        "annotator": df["authors"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if isinstance(x, str) else "auto")),
        "confidence": 1.0,
        "timestamp": datetime.utcnow(),
        "predicted_label": None  # 🔹 нове поле
    })

    return news_df, label_df
