from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from typing import List
import pandas as pd
from app.preprocess import clean_text, preprocess_dataset
from app.db import save_news_and_labels, save_explanation
from app.model import FakeNewsClassifier, BertTinyClassifier
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI(title="FakeNews ML Service")

# ✅ Увімкнення CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Моделі ======
models = {
    "logreg": FakeNewsClassifier(),
    "bert-tiny": BertTinyClassifier()
}

model_status = {name: {"running": False, "metrics": None, "ready": False} for name in models.keys()}

def get_model(name: str):
    if name not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'")
    return models[name]

# ====== Pydantic Models ======
class PredictRequest(BaseModel):
    news_text: str
    model_name: str = Field("logreg", description="Назва моделі: logreg або bert-tiny")
    
    class Config:
        protected_namespaces = ()


class AnalyzeRequest(BaseModel):
    model_name: str = Field("logreg", description="Назва моделі: logreg або bert-tiny")
    test_size: float = Field(0.3, alias="testSize")
    max_iter: int = 10
    C: float = 1.0
    solver: str = "liblinear"

    class Config:
        allow_population_by_field_name = True
        protected_namespaces = ()


# ====== Health ======
@app.get("/health")
def health():
    return {"status": "ok"}


# ====== Preprocess ======
@app.post("/preprocess")
async def preprocess(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            df = pd.read_csv(file.file)
            news_df, label_df = preprocess_dataset(df, file.filename)
            save_news_and_labels(news_df, label_df)
            results.append({"filename": file.filename, "rows": len(df)})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    return {"status": "preprocessed", "files": results}


# ====== Training ======
@app.post("/analyze")
def analyze_all(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    ml_model = get_model(request.model_name)
    status = model_status.get(request.model_name)

    if status["running"]:
        return {"status": "already_running", "model": request.model_name}

    def background_train():
        model_status[request.model_name]["running"] = True
        try:
            # --- FakeNewsClassifier ---
            if hasattr(ml_model, "run_training"):
                ml_model.run_training(
                    test_size=getattr(request, "test_size", 0.3),
                    max_iter=getattr(request, "max_iter", 1000),
                    C=getattr(request, "C", 1.0),
                    solver=getattr(request, "solver", "liblinear")
                )
                metrics = getattr(ml_model, "train_metrics", None)

            # --- BertTinyClassifier ---
            elif hasattr(ml_model, "train"):
                from app.db import load_all_texts, load_all_labels, load_all_news_ids

                texts = load_all_texts()
                labels = load_all_labels()
                news_ids = load_all_news_ids()

                if not texts or not labels or not news_ids:
                    raise HTTPException(status_code=400, detail="Немає даних для тренування")

                metrics = ml_model.train(texts=texts, labels=labels, test_size=getattr(request, "test_size", 0.3))
                
                if hasattr(ml_model, "save_embeddings_and_predictions"):
                    ml_model.save_embeddings_and_predictions(news_ids=news_ids, texts=texts)

            else:
                raise HTTPException(status_code=400, detail=f"Модель '{request.model_name}' не підтримує тренування")

            # Оновлюємо статус
            model_status[request.model_name]["metrics"] = metrics
            model_status[request.model_name]["ready"] = True

        finally:
            model_status[request.model_name]["running"] = False

    # Запускаємо тренування у фоні
    background_tasks.add_task(background_train)

    return {
        "status": "training_started",
        "model": request.model_name,
        "params": {
            "test_size": getattr(request, "test_size", None),
            "max_iter": getattr(request, "max_iter", None),
            "C": getattr(request, "C", None),
            "solver": getattr(request, "solver", None),
        }
    }

@app.get("/analyze/status")
def analyze_status(model_name: str = Query("logreg", description="Назва моделі")):
    status = model_status.get(model_name)
    if not status:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")
    return status


# ====== Prediction ======
@app.post("/predict")
def predict(req: PredictRequest):
    ml_model = get_model(req.model_name)

    if not ml_model.is_trained():
        raise HTTPException(status_code=400, detail=f"Модель '{req.model_name}' ще не натренована.")

    clean = clean_text(req.news_text)
    label, prob = ml_model.predict(clean)
    return {"model": req.model_name, "label": label, "probability": prob}


@app.get("/random_predict")
def random_predict(model_name: str = Query("logreg", description="Назва моделі")):
    ml_model = get_model(model_name)

    if not ml_model.is_trained():
        raise HTTPException(status_code=400, detail=f"Модель '{model_name}' ще не натренована.")

    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")

    with engine.connect() as conn:
        news_ids = conn.execute(text("SELECT id FROM NewsItem")).fetchall()
        if not news_ids:
            return {"error": "База новин порожня"}
        random_id = random.choice(news_ids)[0]

        query = text("""
            SELECT n.text, l.predicted_label, l.confidence, l.label
            FROM NewsItem n
            JOIN Label l ON n.id = l.news_id
            WHERE n.id = :news_id
        """)
        row = conn.execute(query, {"news_id": random_id}).fetchone()

        if not row:
            return {"error": f"Не знайдено прогнозу для новини {random_id}"}

        predicted_label = "real" if row.predicted_label == 0 else "fake"
        true_label = "real" if row.label is False else "fake"

        return {
            "model": model_name,
            "id": random_id,
            "text": row.text,
            "prediction": {
                "predicted_label": predicted_label,
                "probability": float(row.confidence)
            },
            "true_label": true_label
        }

# ====== Interpretability ======
@app.post("/interpret/{method}")
def interpret(method: str, news_id: int, model_name: str = Query("logreg")):
    ml_model = get_model(model_name)
    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")

    df = pd.read_sql(f"SELECT id, text FROM NewsItem WHERE id={news_id}", engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="News not found")

    text_item = df.iloc[0]["text"]

    if method.upper() == "SHAP":
        values = ml_model.explain_shap([text_item])
    elif method.upper() == "IG":
        values = ml_model.explain_ig([text_item])
    elif method.upper() == "TCAV":
        values = ml_model.explain_tcav([text_item])
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method '{method}'")

    exp_id = save_explanation(news_id, method.upper(), values, fidelity=None)
    return {"news_id": news_id, "model": model_name, "method": method.upper(), "explanation_id": exp_id, "payload": values}


# ====== Visualization ======
@app.get("/visualize/{method}")
def visualize(method: str, model_name: str = Query("logreg")):
    method_clean = method.replace('"', '').strip().upper()

    if method_clean not in ["TSNE", "UMAP"]:
        return {"error": f"Unknown method '{method}'"}

    engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")

    query = text("""
        SELECT p.news_id, p.x, p.y, n.text, l.label, l.predicted_label
        FROM ProjectionPoint p
        JOIN NewsItem n ON p.news_id = n.id
        JOIN Label l ON n.id = l.news_id
        WHERE UPPER(TRIM(BOTH '"' FROM p.method)) = :method
    """)
    df_coords = pd.read_sql(query, engine, params={"method": method_clean})

    if df_coords.empty:
        return {"ids": [], "points": [], "labels": [], "predicted_labels": []}

    return {
        "model": model_name,
        "ids": df_coords["news_id"].tolist(),
        "points": df_coords[["x", "y"]].values.tolist(),
        "labels": df_coords["label"].tolist(),
        "predicted_labels": df_coords["predicted_label"].tolist()
    }
