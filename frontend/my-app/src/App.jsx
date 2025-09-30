import React, { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = process.env.REACT_APP_API_URL ?? "http://localhost:5000";
const DEFAULT_MODELS = ["roberta-base", "roberta-large"];

const toDisplayMetric = (value) =>
  typeof value === "number" ? value.toFixed(3) : "-";

export default function App() {
  const [availableModels, setAvailableModels] = useState(DEFAULT_MODELS);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODELS[0]);
  const [downloadDataset, setDownloadDataset] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isPolling, setIsPolling] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [predictionText, setPredictionText] = useState("");
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const apiPath = (path) => `${API_BASE}/api/ml${path}`;

  const callApi = async (path, options = {}) => {
    const { headers, body, ...rest } = options;
    const requestInit = { ...rest };

    if (body !== undefined) {
      requestInit.body = body;
      requestInit.headers = {
        "Content-Type": "application/json",
        ...(headers ?? {}),
      };
    } else if (headers) {
      requestInit.headers = headers;
    }

    const response = await fetch(apiPath(path), requestInit);
    const contentType = response.headers.get("content-type");
    const isJson = contentType && contentType.includes("application/json");
    const payload = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      const message = typeof payload === "string" ? payload : payload?.detail ?? "Request failed";
      throw new Error(message);
    }

    return payload;
  };

  const fetchAvailableModels = async () => {
    try {
      const data = await callApi("/models");
      if (Array.isArray(data.models) && data.models.length) {
        setAvailableModels((prev) => {
          const merged = Array.from(new Set([...DEFAULT_MODELS, ...prev, ...data.models]));
          merged.sort();
          if (!merged.includes(selectedModel)) {
            setSelectedModel(merged[0]);
          }
          return merged;
        });
      }
    } catch (error) {
      console.warn("Failed to fetch model list", error);
    }
  };

  useEffect(() => {
    fetchAvailableModels();
  }, []);

  useEffect(() => {
    if (!isPolling) {
      return undefined;
    }

    let timerId;

    const poll = async () => {
      try {
        const status = await callApi(`/train/status/${selectedModel}`);
        setTrainingStatus(status);

        if (status.state === "in_progress") {
          timerId = setTimeout(poll, 2000);
        } else {
          setIsPolling(false);
          if (status.state === "completed" && status.summary?.evaluation) {
            setMetrics({
              model_name: selectedModel,
              report: {},
              summary: status.summary,
            });
            fetchAvailableModels();
          }
        }
      } catch (error) {
        setIsPolling(false);
        setErrorMessage(error.message);
      }
    };

    poll();
    return () => clearTimeout(timerId);
  }, [isPolling, selectedModel]);

  const startTraining = async () => {
    setErrorMessage(null);
    setTrainingStatus({ state: "starting" });
    setMetrics(null);

    try {
      await callApi("/train", {
        method: "POST",
        body: JSON.stringify({
          modelName: selectedModel,
          downloadDataset,
        }),
      });
      setIsPolling(true);
    } catch (error) {
      setErrorMessage(error.message);
      setTrainingStatus(null);
    }
  };

  const refreshStatus = async () => {
    setErrorMessage(null);
    try {
      const status = await callApi(`/train/status/${selectedModel}`);
      setTrainingStatus(status);
    } catch (error) {
      setErrorMessage(error.message);
    }
  };

  const loadMetrics = async () => {
    setErrorMessage(null);
    setMetrics(null);
    try {
      const data = await callApi(`/metrics/${selectedModel}`);
      setMetrics(data);
    } catch (error) {
      setErrorMessage(error.message);
    }
  };

  const runPrediction = async () => {
    setErrorMessage(null);
    setPredictionResult(null);

    if (!predictionText.trim()) {
      setErrorMessage("Please provide text to analyse.");
      return;
    }

    try {
      const result = await callApi("/predict", {
        method: "POST",
        body: JSON.stringify({
          modelName: selectedModel,
          text: predictionText,
        }),
      });
      setPredictionResult(result);
    } catch (error) {
      setErrorMessage(error.message);
    }
  };

  const aggregateMetrics = useMemo(() => {
    if (!metrics?.summary?.evaluation) {
      return null;
    }
    return metrics.summary.evaluation;
  }, [metrics]);

  return (
    <div className="app-shell">
      <header>
        <h1>Fake News Detector</h1>
        <p>Train a transformer-based classifier and evaluate news snippets.</p>
      </header>

      <section className="card">
        <h2>Model Configuration</h2>
        <div className="field">
          <label htmlFor="model-select">Model</label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
          >
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>

        <div className="checkbox">
          <input
            id="download"
            type="checkbox"
            checked={downloadDataset}
            onChange={(event) => setDownloadDataset(event.target.checked)}
          />
          <label htmlFor="download">Download dataset from Kaggle before training</label>
        </div>

        <div className="actions">
          <button onClick={startTraining} disabled={isPolling}>
            Start Training
          </button>
          <button onClick={refreshStatus}>Check Status</button>
          <button onClick={loadMetrics}>Load Metrics</button>
        </div>

        {trainingStatus && (
          <div className="status">
            <strong>Status:</strong> {trainingStatus.state}
          </div>
        )}

        {isPolling && <p className="info">Training in progress… polling status.</p>}
      </section>

      <section className="card">
        <h2>Evaluate Model</h2>
        {aggregateMetrics ? (
          <ul className="metrics">
            <li>
              <span>Accuracy</span>
              <span>{toDisplayMetric(aggregateMetrics.accuracy)}</span>
            </li>
            <li>
              <span>Precision</span>
              <span>{toDisplayMetric(aggregateMetrics.precision)}</span>
            </li>
            <li>
              <span>Recall</span>
              <span>{toDisplayMetric(aggregateMetrics.recall)}</span>
            </li>
            <li>
              <span>F1</span>
              <span>{toDisplayMetric(aggregateMetrics.f1)}</span>
            </li>
            <li>
              <span>ROC AUC</span>
              <span>{toDisplayMetric(aggregateMetrics.roc_auc)}</span>
            </li>
          </ul>
        ) : (
          <p className="info">Metrics will appear here once generated.</p>
        )}
      </section>

      <section className="card">
        <h2>Predict</h2>
        <textarea
          value={predictionText}
          onChange={(event) => setPredictionText(event.target.value)}
          placeholder="Enter a news headline or snippet"
        />
        <div className="actions">
          <button onClick={runPrediction}>Analyse</button>
        </div>
        {predictionResult && (
          <div className="prediction">
            <p>
              <strong>Prediction:</strong> {predictionResult.label}
            </p>
            <p>
              <strong>Confidence:</strong> {toDisplayMetric(predictionResult.confidence)}
            </p>
          </div>
        )}
      </section>

      {errorMessage && (
        <div className="error">Error: {errorMessage}</div>
      )}
    </div>
  );
}
