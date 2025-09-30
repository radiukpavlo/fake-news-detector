import os
import json
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from train import train_model
from models import get_model

app = FastAPI(title="Fake News ML Service")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# In-memory store for training status
training_status = {}


class TrainRequest(BaseModel):
    model_name: str = "roberta-base"
    download_dataset: bool = False


class PredictRequest(BaseModel):
    model_name: str
    text: str


@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Starts a new training job in the background.
    """
    model_name = request.model_name
    if training_status.get(model_name) == "in_progress":
        raise HTTPException(
            status_code=400, detail=f"Training for model '{model_name}' is already in progress."
        )

    def run_training():
        training_status[model_name] = "in_progress"
        try:
            train_model(model_name=model_name, download_dataset=request.download_dataset)
            training_status[model_name] = "completed"
        except Exception as e:
            training_status[model_name] = f"failed: {str(e)}"

    background_tasks.add_task(run_training)
    return {"message": f"Training for model '{model_name}' started in the background."}


@app.get("/train/status/{model_name}")
async def get_training_status(model_name: str):
    """
    Gets the status of a training job.
    """
    status = training_status.get(model_name)
    if not status:
        raise HTTPException(status_code=404, detail=f"No training job found for model '{model_name}'.")
    return {"model_name": model_name, "status": status}


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Makes a prediction using a trained model.
    """
    model_name = request.model_name
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Please train it first.")

    model = get_model(model_name, output_dir=MODELS_DIR)
    prediction = model.predict(request.text)
    return prediction


@app.get("/metrics/{model_name}")
async def get_metrics(model_name: str):
    """
    Retrieves the evaluation metrics for a trained model.
    """
    metrics_path = os.path.join(OUTPUT_DIR, model_name, "classification_report.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404, detail=f"Metrics not found for model '{model_name}'. Please train the model first."
        )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics


@app.get("/metrics/plots/{model_name}/{plot_name}")
async def get_plot(model_name: str, plot_name: str):
    """
    Retrieves an evaluation plot for a trained model.
    """
    if plot_name not in ["confusion_matrix.png", "roc_curve.png"]:
        raise HTTPException(status_code=404, detail="Plot not found. Available plots: 'confusion_matrix.png', 'roc_curve.png'.")

    plot_path = os.path.join(OUTPUT_DIR, model_name, plot_name)
    if not os.path.exists(plot_path):
        raise HTTPException(
            status_code=404, detail=f"Plot '{plot_name}' not found for model '{model_name}'. Please train the model first."
        )

    return FileResponse(plot_path)