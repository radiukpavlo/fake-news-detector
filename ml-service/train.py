import argparse
import json
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from models import get_model
from dataset import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def train_model(model_name="roberta-base", download_dataset=False):
    """
    Main function to train a model.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Load data
    print("Loading dataset...")
    df = load_dataset(download=download_dataset)

    # Get model
    print(f"Loading model: {model_name}")
    model = get_model(model_name, output_dir=model_output_dir)

    # Train model
    print("Training model...")
    model.train(df)
    print("Training complete.")

    # Evaluate model
    print("Evaluating model...")
    evaluate(model, df, model_output_dir)
    print(f"Evaluation results saved to {model_output_dir}")


def evaluate(model, df, output_dir):
    """
    Evaluates the model and saves the metrics and plots.
    """
    y_true = df["label"].tolist()
    y_pred_probs = [model.predict(text)["confidence"] for text in df["text"]]
    y_pred = [1 if p > 0.5 else 0 for p in y_pred_probs]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["fake", "real"], output_dict=True)
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["fake", "real"], yticklabels=["fake", "real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()


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
    args = parser.parse_args()

    train_model(model_name=args.model_name, download_dataset=args.download)