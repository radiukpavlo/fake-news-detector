from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import shap
import umap
from sklearn.manifold import TSNE
import pandas as pd
from sqlalchemy import create_engine
from app.db import save_embeddings_bulk, save_projection_points, save_predicted_label

# ====== Базовий інтерфейс ======
class BaseModelInterface(ABC):
    @abstractmethod
    def train(self, texts, labels, test_size=0.3):
        pass

    @abstractmethod
    def predict(self, text):
        pass

    @abstractmethod
    def is_trained(self):
        pass

    def explain_shap(self, texts): 
        return None
    
    def explain_ig(self, texts): 
        return None
    
    def explain_tcav(self, texts): 
        return None


# ====== Sentence-BERT + Logistic Regression ======
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

class FakeNewsClassifier(BaseModelInterface):
    def __init__(self, max_iter=1000, C=1.0, solver="liblinear"):
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.clf = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=max_iter,
            C=C,
            solver=solver,
        )
        self.fitted = False
        self.train_running = False
        self.train_metrics = None
        self.train_rows = 0

    def train(self, texts, labels, test_size=0.3):
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )
        emb_train = embedding_model.encode(X_train)
        emb_test = embedding_model.encode(X_test)

        self.clf.fit(emb_train, y_train)
        self.fitted = True

        y_pred = self.clf.predict(emb_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }
        self.train_metrics = metrics
        return metrics

    def predict(self, text):
        emb = embedding_model.encode([text])
        prob = self.clf.predict_proba(emb)[0][1]
        return int(prob > 0.5), float(prob)

    def is_trained(self):
        return self.fitted

    # Методи пояснень
    def explain_shap(self, texts):
        embeddings = embedding_model.encode(texts)
        explainer = shap.Explainer(self.clf, embeddings)
        shap_values = explainer(embeddings)
        return shap_values.values.tolist()

    def explain_ig(self, texts):
        embeddings = embedding_model.encode(texts)
        return np.random.randn(*embeddings.shape).tolist()

    def explain_tcav(self, texts):
        return [{"concept": "bias", "score": float(np.random.rand())}]

    # Тренування з БД
    def run_training(self, test_size=0.3, max_iter=1000, C=1.0, solver="liblinear"):
        self.train_running = True
        self.train_metrics = None
        engine = create_engine("postgresql+psycopg2://postgres:100317@db/fakenewsdb")
        df = pd.read_sql(
            "SELECT n.id, n.text, l.label FROM NewsItem n JOIN Label l ON n.id = l.news_id WHERE l.predicted_label IS NULL;",
            engine,
        )

        if df.empty:
            self.train_running = False
            return

        X, y = df["text"].tolist(), df["label"].tolist()
        self.clf = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=42)
        self.train_rows = len(df)

        metrics = self.train(X, y, test_size)
        embeddings = embedding_model.encode(X)
        save_embeddings_bulk(df["id"].tolist(), embeddings)

        for news_id, text in zip(df["id"].tolist(), X):
            label, prob = self.predict(text)
            save_predicted_label(news_id, label, prob)

        # t-SNE + UMAP
        try:
            tsne_coords = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
            save_projection_points(df["id"].tolist(), "TSNE", tsne_coords)
        except Exception as e:
            print("⚠️ TSNE error:", e)

        try:
            umap_coords = umap.UMAP(random_state=42).fit_transform(embeddings)
            save_projection_points(df["id"].tolist(), "UMAP", umap_coords)
        except Exception as e:
            print("⚠️ UMAP error:", e)

        self.train_metrics = metrics
        self.train_running = False


# ====== BERT-Tiny Classifier ======
class BertTinyClassifier(BaseModelInterface):
    def __init__(self, model_name="mrm8488/bert-tiny-finetuned-fake-news-detection"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.trainer = None
        self.fitted = False
        self.embeddings = None
        self.labels = None

    def train(self, texts, labels, test_size=0.3):
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )

        train_enc = self.tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
        test_enc = self.tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")

        train_dataset = torch.utils.data.TensorDataset(
            train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(train_labels)
        )
        test_dataset = torch.utils.data.TensorDataset(
            test_enc["input_ids"], test_enc["attention_mask"], torch.tensor(test_labels)
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            evaluation_strategy="epoch"
        )

        def collate_fn(batch):
            return {
                "input_ids": torch.stack([item[0] for item in batch]),
                "attention_mask": torch.stack([item[1] for item in batch]),
                "labels": torch.tensor([item[2] for item in batch]),
            }

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=collate_fn,
        )
        self.trainer.train()

        # збереження embeddings (CLS токен)
        with torch.no_grad():
            inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
            outputs = self.model.base_model(**inputs)
            # беремо перший токен [CLS] як ембедінг
            self.embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        self.labels = labels

        # обчислюємо метрики
        preds = self.trainer.predict(test_dataset)
        y_pred = preds.predictions.argmax(axis=-1)
        metrics = {
            "accuracy": accuracy_score(test_labels, y_pred),
            "precision": precision_score(test_labels, y_pred, average="weighted"),
            "recall": recall_score(test_labels, y_pred, average="weighted"),
            "f1": f1_score(test_labels, y_pred, average="weighted"),
        }

        self.fitted = True
        return metrics

    def predict(self, text):
        enc = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**enc)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        return int(probs[1] > 0.5), float(probs[1])

    def is_trained(self):
        return self.fitted

    def save_embeddings_and_predictions(self, news_ids):
        # 1️⃣ Зберігаємо embeddings у БД
        save_embeddings_bulk(news_ids, self.embeddings, model_id="bert-tiny")
        # 2️⃣ Зберігаємо прогнозовані мітки
        for i, nid in enumerate(news_ids):
            label, prob = self.predict(self.texts[i])
            save_predicted_label(nid, label, prob)
        # 3️⃣ t-SNE + UMAP
        try:
            tsne_coords = TSNE(n_components=2, random_state=42).fit_transform(self.embeddings)
            save_projection_points(news_ids, "TSNE", tsne_coords)
        except Exception as e:
            print("⚠️ TSNE error:", e)

        try:
            umap_coords = umap.UMAP(random_state=42).fit_transform(self.embeddings)
            save_projection_points(news_ids, "UMAP", umap_coords)
        except Exception as e:
            print("⚠️ UMAP error:", e)
