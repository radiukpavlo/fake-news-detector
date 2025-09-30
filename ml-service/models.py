import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class BaseModel:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model_path = os.path.join(output_dir, model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    def train(self, df, test_size=0.2):
        """
        Trains the model on the given dataframe.
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        train_encodings = self.tokenizer(
            train_df["text"].tolist(), truncation=True, padding=True
        )
        test_encodings = self.tokenizer(
            test_df["text"].tolist(), truncation=True, padding=True
        )

        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {
                    key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()
                }
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = TorchDataset(
            train_encodings, train_df["label"].tolist()
        )
        test_dataset = TorchDataset(
            test_encodings, test_df["label"].tolist()
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_path, "results"),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.model_path, "logs"),
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
            return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

        return trainer.evaluate()

    def predict(self, text):
        """
        Makes a prediction on a single text.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()

        return {
            "prediction": "real" if prediction == 1 else "fake",
            "confidence": probs[0][prediction].item(),
        }

def get_model(model_name, output_dir="models"):
    """
    Factory function to get a model instance.
    """
    if model_name not in ["roberta-base", "roberta-large"]:
        raise ValueError("Unsupported model name. Choose 'roberta-base' or 'roberta-large'.")

    return BaseModel(model_name, output_dir)