import { createContext, useContext, useMemo, useState } from "react";

const PredictionContext = createContext(null);

export function PredictionProvider({ children }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [modelUsed, setModelUsed] = useState("");
  const [cnnScore, setCnnScore] = useState(0);
  const [xgbScore, setXgbScore] = useState(0);
  // START VIT INTEGRATION
  const [cnnPrediction, setCnnPrediction] = useState(null);
  const [xgbPrediction, setXgbPrediction] = useState(null);
  const [hybridPrediction, setHybridPrediction] = useState(null);
  const [vitPrediction, setVitPrediction] = useState(null);
  const [vitConfidence, setVitConfidence] = useState(0);
  const [vitScore, setVitScore] = useState(0);
  // END VIT INTEGRATION

  const clearPrediction = () => {
    setPrediction(null);
    setConfidence(0);
    setModelUsed("");
    setCnnScore(0);
    setXgbScore(0);
    // START VIT INTEGRATION
    setCnnPrediction(null);
    setXgbPrediction(null);
    setHybridPrediction(null);
    setVitPrediction(null);
    setVitConfidence(0);
    setVitScore(0);
    // END VIT INTEGRATION
  };

  const clearAll = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl("");
    clearPrediction();
  };

  const value = useMemo(
    () => ({
      selectedFile,
      setSelectedFile,
      previewUrl,
      setPreviewUrl,
      selectedModel,
      setSelectedModel,
      prediction,
      setPrediction,
      confidence,
      setConfidence,
      modelUsed,
      setModelUsed,
      cnnScore,
      setCnnScore,
      xgbScore,
      setXgbScore,
      // START VIT INTEGRATION
      cnnPrediction,
      setCnnPrediction,
      xgbPrediction,
      setXgbPrediction,
      hybridPrediction,
      setHybridPrediction,
      vitPrediction,
      setVitPrediction,
      vitConfidence,
      setVitConfidence,
      vitScore,
      setVitScore,
      // END VIT INTEGRATION
      clearPrediction,
      clearAll,
    }),
    [
      selectedFile,
      previewUrl,
      selectedModel,
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
    ]
  );

  return <PredictionContext.Provider value={value}>{children}</PredictionContext.Provider>;
}

export function usePrediction() {
  const context = useContext(PredictionContext);
  if (!context) {
    throw new Error("usePrediction must be used inside PredictionProvider");
  }
  return context;
}
