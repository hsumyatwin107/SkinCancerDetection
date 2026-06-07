import { usePrediction } from "../context/PredictionContext";

const RESULT_STYLES = {
  benign: {
    label: "Benign",
    badge: "bg-emerald-100 text-emerald-700",
    progress: "bg-emerald-500",
    ring: "ring-emerald-100",
  },
  malignant: {
    label: "Malignant",
    badge: "bg-rose-100 text-rose-700",
    progress: "bg-rose-500",
    ring: "ring-rose-100",
  },
  unknown: {
    label: "Unknown",
    badge: "bg-slate-100 text-slate-700",
    progress: "bg-slate-500",
    ring: "ring-slate-100",
  },
};

function ResultCard({
  prediction,
  confidence,
  modelUsed,
  cnnScore,
  xgbScore,
  // START VIT INTEGRATION
  cnnPrediction,
  xgbPrediction,
  hybridPrediction,
  vitPrediction,
  vitConfidence,
  vitScore,
  // END VIT INTEGRATION
}) {
  const { selectedModel } = usePrediction();

  const confidenceValue = Number(confidence) || 0;
  const cnnScoreValue = Number(cnnScore) || 0;
  const xgbScoreValue = Number(xgbScore) || 0;
  // START VIT INTEGRATION
  const vitConfidenceValue = Number(vitConfidence) || 0;
  const vitScoreValue = Number(vitScore) || 0;
  // END VIT INTEGRATION

  // START RESULT FILTERING
  const modelKey = String(selectedModel || "").trim().toLowerCase();
  const showCnnOnly = modelKey === "cnn";
  const showVitOnly = modelKey === "vit";
  const showHybridOnly = modelKey === "hybrid";

  let activePrediction = null;
  let activeConfidence = 0;
  let modelLabel = "N/A";

  if (showVitOnly) {
    activePrediction = vitPrediction;
    activeConfidence = vitConfidenceValue;
    modelLabel = "VIT";
  } else if (showCnnOnly) {
    activePrediction = cnnPrediction;
    const isMalignant = cnnPrediction === "malignant";
    activeConfidence = isMalignant ? cnnScoreValue : 1 - cnnScoreValue;
    modelLabel = "CNN";
  } else if (showHybridOnly) {
    activePrediction = hybridPrediction || prediction;
    activeConfidence = confidenceValue;
    modelLabel = "HYBRID";
  } else {
    activePrediction = prediction;
    activeConfidence = confidenceValue;
    modelLabel = modelUsed ? String(modelUsed).toUpperCase() : "N/A";
  }

  const hasActiveResult = Boolean(activePrediction);
  // END RESULT FILTERING

  const ui = RESULT_STYLES[activePrediction] || RESULT_STYLES.unknown;
  const percent = activeConfidence <= 1 ? activeConfidence * 100 : activeConfidence;
  const clampedPercent = Math.min(Math.max(percent, 0), 100);

  return (
    <section
      className={`rounded-2xl border border-slate-100 bg-white p-5 shadow-sm ring-4 transition-all duration-300 ${ui.ring}`}
    >
      <h2 className="text-lg font-semibold text-slate-800">Prediction Result</h2>

      {hasActiveResult ? (
        <div className="mt-4 space-y-4">
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-medium text-slate-600">Model Used</span>
            <span className="rounded-full bg-blue-100 px-3 py-1 text-sm font-semibold text-blue-700">
              {modelLabel}
            </span>
          </div>

          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-medium text-slate-600">Assessment</span>
            <span className={`rounded-full px-3 py-1 text-sm font-semibold ${ui.badge}`}>
              {ui.label}
            </span>
          </div>

          {/* START RESULT FILTERING */}
          {showCnnOnly ? (
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-medium text-slate-600">CNN Score</span>
              <span className="text-sm font-semibold text-slate-800">{cnnScoreValue.toFixed(4)}</span>
            </div>
          ) : null}

          {showHybridOnly ? (
            <>
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium text-slate-600">CNN Score</span>
                <span className="text-sm font-semibold text-slate-800">{cnnScoreValue.toFixed(4)}</span>
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium text-slate-600">XGBoost Score</span>
                <span className="text-sm font-semibold text-slate-800">{xgbScoreValue.toFixed(4)}</span>
              </div>
            </>
          ) : null}

          {showVitOnly ? (
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-medium text-slate-600">ViT Score</span>
              <span className="text-sm font-semibold text-slate-800">{vitScoreValue.toFixed(4)}</span>
            </div>
          ) : null}
          {/* END RESULT FILTERING */}

          <div>
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="font-medium text-slate-600">Confidence</span>
              <span className="font-semibold text-slate-800">{clampedPercent.toFixed(2)}%</span>
            </div>
            <div className="h-3 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className={`h-full rounded-full transition-all duration-700 ease-out ${ui.progress}`}
                style={{ width: `${clampedPercent}%` }}
                aria-hidden="true"
              />
            </div>
          </div>
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate-500">
          Your prediction will appear here after analysis.
        </p>
      )}
    </section>
  );
}

export default ResultCard;
