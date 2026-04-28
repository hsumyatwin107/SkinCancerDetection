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

  const clearPrediction = () => {
    setPrediction(null);
    setConfidence(0);
    setModelUsed("");
    setCnnScore(0);
    setXgbScore(0);
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
      clearPrediction,
      clearAll,
    }),
    [selectedFile, previewUrl, selectedModel, prediction, confidence, modelUsed, cnnScore, xgbScore]
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
