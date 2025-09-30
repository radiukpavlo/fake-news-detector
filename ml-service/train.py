import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

from dataset import load_dataset
from models import get_model

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def train_model(
    model_name: str = "roberta-base",
    download_dataset: bool = False,
    models_dir: Path = MODELS_DIR,
    force_rebuild: bool = False,
) -> Dict:
    """Trains a model and returns a summary of metrics."""
    model_output_dir = OUTPUT_DIR / model_name
    model_output_dir.mkdir(exist_ok=True, parents=True)

    if force_rebuild:
        shutil.rmtree(models_dir / model_name, ignore_errors=True)

    # Load data
    df = load_dataset(download=download_dataset)

    # Train and persist the model
    model = get_model(model_name, models_dir)
    trainer_metrics = model.train(df)

    # Evaluate and persist artefacts
    evaluation_metrics = evaluate(model, df, model_output_dir)

    summary = {
        "model_name": model_name,
        "trainer_metrics": trainer_metrics,
        "evaluation": evaluation_metrics,
    }

    with (model_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    return summary


def evaluate(model, df, output_dir: Path) -> Dict:
    """Evaluates the model and stores metrics and plots."""
    texts: List[str] = df["text"].tolist()
    labels: List[int] = df["label"].tolist()

    predictions = model.predict_many(texts)
    probabilities_real = [pred["probabilities"]["real"] for pred in predictions]
    predicted_labels = [1 if pred["label"] == "real" else 0 for pred in predictions]

    report = classification_report(
        labels,
        predicted_labels,
        target_names=["fake", "real"],
        output_dict=True,
        zero_division=0,
    )

    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)

    cm = confusion_matrix(labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["fake", "real"],
        yticklabels=["fake", "real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(labels, probabilities_real)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()

    return {
        "accuracy": report.get("accuracy"),
        "precision": report.get("real", {}).get("precision"),
        "recall": report.get("real", {}).get("recall"),
        "f1": report.get("real", {}).get("f1-score"),
        "roc_auc": roc_auc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fake news detection model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        choices=["roberta-base", "roberta-large"],
        help="The name of the model to train.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from Kaggle.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Remove any cached model files before training.",
    )
    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        download_dataset=args.download,
        force_rebuild=args.force_rebuild,
    )
