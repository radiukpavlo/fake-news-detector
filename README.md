# Fake News Detector

A containerised demo that trains and serves a transformer-based fake-news classifier. The stack is composed of:

- **frontend** – React SPA for training and inference
- **backend** – ASP.NET Core API that proxies requests to the ML service
- **ml-service** – FastAPI application wrapping Hugging Face models

## Prerequisites

- Docker Desktop (Compose v2)
- Optional: Kaggle API token at `ml-service/.kaggle/kaggle.json` if you want the service to download the LIAR dataset automatically.

## Quick start

```bash
# from the repository root
docker compose build
docker compose up -d
```

Services exposed locally:

| Service   | URL                     |
|-----------|-------------------------|
| Frontend  | http://localhost:3000   |
| Backend   | http://localhost:5000   |
| ML API    | http://localhost:8000   |

### Training a model

1. Open `http://localhost:3000`.
2. Choose a model (`roberta-base` or `roberta-large`).
3. Optionally tick “Download dataset from Kaggle” (requires valid credentials inside the container).
4. Click **Start Training** and monitor progress.

Metrics and plots are written to `ml-service/output/<model-name>/`.

### Predicting

Once training completes, type a news headline/snippet in the text box and click **Analyse** to see the model prediction and confidence.

## Development workflow

Run the backend and ML service without Docker:

```bash
# backend
cd backend/RealFakeNews
dotnet run

# ml-service
cd ml-service
uvicorn main:app --reload --port 8000
```

Then start the frontend:

```bash
cd frontend/my-app
npm install
npm start
```

Update `REACT_APP_API_URL` if the backend runs on a non-default host/port.

## Project structure

```
.
├── backend           # ASP.NET Core proxy API
├── frontend          # React single page app
├── ml-service        # FastAPI + Hugging Face models
└── docker-compose.yml
```

## Known limitations

- A GPU is not required but training large models on CPU can be slow.
- Kaggle downloads fail unless credentials are present in `ml-service/.kaggle/kaggle.json`.
- The sample React styles use basic CSS; feel free to replace with a design system of your choice.
