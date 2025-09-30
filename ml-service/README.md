# Fake News Detection ML Service

This service provides a machine learning pipeline for training and evaluating fake news detection models. It is built using FastAPI and Hugging Face Transformers and is designed to work with the LIAR dataset.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Kaggle API Setup](#kaggle-api-setup)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Making Predictions](#making-predictions)
  - [Retrieving Metrics](#retrieving-metrics)
- [Models](#models)
- [Dataset](#dataset)
- [Evaluation](#evaluation)

---

## Project Structure

The `ml-service` is organized as follows:

```
ml-service/
├── data/                 # Data for the models (LIAR dataset)
├── models/               # Saved model artifacts
├── output/               # Training outputs (metrics, plots)
├── dataset.py            # Handles dataset downloading and loading
├── main.py               # FastAPI application
├── models.py             # Model definitions and training logic
├── train.py              # Training and evaluation script
├── Dockerfile            # Docker configuration
├── README.md             # This file
└── start.sh              # Script to start the service
```

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)

### Kaggle API Setup

To download the LIAR dataset, you need to configure your Kaggle API credentials.

1.  **Create a Kaggle Account**: If you don't have one, sign up at [kaggle.com](https://www.kaggle.com).
2.  **Generate API Token**: Go to your account settings and click "Create New API Token". This will download a `kaggle.json` file.
3.  **Place the Token**: Move the `kaggle.json` file to `~/.kaggle/` on your system.

    ```bash
    mkdir -p ~/.kaggle
    mv kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-url>/ml-service
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will be created in a later step.)*

---

## Usage

The service is run using `uvicorn`.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Training a Model

You can train a model by sending a POST request to the `/train` endpoint.

**Endpoint**: `POST /train`

**Request Body**:

```json
{
  "model_name": "roberta-base",
  "download_dataset": true
}
```

-   `model_name`: The model to train (`roberta-base` or `roberta-large`).
-   `download_dataset`: If `true`, the LIAR dataset will be downloaded from Kaggle.

This will start the training process in the background. You can check the status at `GET /train/status/{model_name}`.

### Making Predictions

Once a model is trained, you can make predictions using the `/predict` endpoint.

**Endpoint**: `POST /predict`

**Request Body**:

```json
{
  "model_name": "roberta-base",
  "text": "This is a news headline to classify."
}
```

**Response**:

```json
{
  "prediction": "fake",
  "confidence": 0.89
}
```

### Retrieving Metrics

You can retrieve the evaluation metrics for a trained model.

-   **Classification Report**: `GET /metrics/{model_name}`
-   **Confusion Matrix**: `GET /metrics/plots/{model_name}/confusion_matrix.png`
-   **ROC Curve**: `GET /metrics/plots/{model_name}/roc_curve.png`

---

## Models

The service supports two Transformer-based models from the Hugging Face library:

-   `roberta-base`: A powerful and widely used language model.
-   `roberta-large`: A larger, more powerful version of RoBERTa, which may yield higher accuracy at the cost of more resources.

The models are defined in `models.py` and are trained for binary classification (real/fake).

---

## Dataset

The service uses the [LIAR dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset), which contains short statements from political contexts, labeled for truthfulness.

The original 6-class labels are mapped to a binary classification as follows:

-   **Fake (0)**: `pants-fire`, `false`, `barely-true`, `half-true`
-   **Real (1)**: `mostly-true`, `true`

The dataset can be automatically downloaded by setting `download_dataset: true` in the training request.

---

## Evaluation

The models are evaluated on a held-out test set, and the following metrics are computed:

-   **Classification Report**: Precision, recall, and F1-score for each class.
-   **Confusion Matrix**: A visual representation of the model's performance.
-   **ROC Curve and AUC**: The Receiver Operating Characteristic curve and the Area Under the Curve, which measure the model's ability to distinguish between classes.

All evaluation artifacts are saved in the `output/{model_name}/` directory.