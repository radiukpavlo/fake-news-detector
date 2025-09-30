import os
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class BaseModel:
    def __init__(self, model_name: str, storage_root: Path):
        self.model_name = model_name
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.storage_root / model_name

        if (self.model_dir / "config.json").exists():
            source = self.model_dir
        else:
            source = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=2
        )

    def train(self, df, test_size: float = 0.2):
        """Trains the model on the provided dataframe and returns trainer metrics."""
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)

        train_df, eval_df = train_test_split(df, test_size=test_size, random_state=42)

        train_encodings = self.tokenizer(
            train_df["text"].tolist(), truncation=True, padding=True
        )
        eval_encodings = self.tokenizer(
            eval_df["text"].tolist(), truncation=True, padding=True
        )

        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = TorchDataset(train_encodings, train_df["label"].tolist())
        eval_dataset = TorchDataset(eval_encodings, eval_df["label"].tolist())

        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "results"),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.model_dir / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary"
            )
            acc = accuracy_score(labels, preds)
            return {
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        metrics = trainer.train()
        trainer.save_model(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))

        return metrics.metrics

    def _predict_logits(self, tokenized_inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
        return outputs.logits

    def predict_proba(self, text: str) -> Tuple[float, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        logits = self._predict_logits(inputs)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        return probs[0].item(), probs[1].item()

    def predict(self, text: str) -> dict:
        prob_fake, prob_real = self.predict_proba(text)
        if prob_real >= prob_fake:
            label = "real"
            confidence = prob_real
        else:
            label = "fake"
            confidence = prob_fake

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "fake": prob_fake,
                "real": prob_real,
            },
        }

    def predict_many(self, texts: Iterable[str]) -> List[dict]:
        return [self.predict(text) for text in texts]


def get_model(model_name: str, storage_root: Path):
    """Factory to return a BaseModel stored under the given root."""
    return BaseModel(model_name, storage_root)
