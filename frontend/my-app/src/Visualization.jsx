// src/Visualization.jsx
import React from "react";
import Plot from "react-plotly.js";

export default function Visualization({ data, labelType }) {
  if (!data) return null;

  // Підготовка підписів та кольорів
  let textLabels = [];
  let colorLabels = [];

  if (labelType === "predicted_label") {
    // Для прогнозованих: 0/1 → fake/real
    textLabels = (data.predicted_labels || []).map((v) => (v === 1 ? "fake" : "real"));
    colorLabels = textLabels.map((v) => (v === "fake" ? 1 : 0));
  } else {
    // Для справжніх: boolean → fake/real
    textLabels = (data.labels || []).map((v) => (v ? "fake" : "real"));
    colorLabels = textLabels.map((v) => (v === "fake" ? 1 : 0));
  }

  return (
    <Plot
      data={[
        {
          x: data.points.map((p) => p[0]),
          y: data.points.map((p) => p[1]),
          text: textLabels.map((label, i) => `ID: ${data.ids[i]}, ${label}`),
          mode: "markers",
          type: "scatter",
          marker: {
            color: colorLabels,
            colorscale: [
              [0, "green"], // real
              [1, "red"],   // fake
            ],
            showscale: true,
            colorbar: {
              tickvals: [0, 1],
              ticktext: ["real", "fake"],
            },
          },
        },
      ]}
      layout={{
        title:
          labelType === "predicted_label"
            ? "Прогнозовані мітки"
            : "Справжні мітки",
        autosize: true,
        height: 400,
      }}
      style={{ width: "100%", height: "400px" }}
    />
  );
}
