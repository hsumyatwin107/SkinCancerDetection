import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, Upload } from "lucide-react";
import Loader from "../components/Loader";
import UploadCard from "../components/UploadCard";
import { usePrediction } from "../context/PredictionContext";

const API_URL = "http://127.0.0.1:8000/predict";
const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];
const MAX_FILE_SIZE_MB = 10;

function DetectPage() {
  const {
    selectedFile,
    setSelectedFile,
    previewUrl,
    setPreviewUrl,
    selectedModel,
    setSelectedModel,
    setPrediction,
    setConfidence,
    setModelUsed,
    setCnnScore,
    setXgbScore,
    clearPrediction,
  } = usePrediction();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Choose an image to begin.");
  const [cameraOn, setCameraOn] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();

  const stateModel = String(location.state?.model || "")
    .trim()
    .toLowerCase();
  const modelFromState = stateModel === "cnn" || stateModel === "vit" ? stateModel : "";
  const activeModel = modelFromState || selectedModel;

  const canPredict = useMemo(
    () => selectedFile && activeModel && !loading,
    [selectedFile, activeModel, loading]
  );

  const normalizePrediction = (rawPrediction) => {
    const value = String(rawPrediction || "").trim().toLowerCase();
    if (value.includes("malignant")) return "malignant";
    if (value.includes("benign")) return "benign";
    return "unknown";
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraOn(false);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  useEffect(() => {
    if (modelFromState && modelFromState !== selectedModel) {
      setSelectedModel(modelFromState);
    }
  }, [modelFromState, selectedModel, setSelectedModel]);

  const validateImageFile = (file) => {
    if (!file) return "No file selected.";
    if (!ALLOWED_TYPES.includes(file.type)) {
      return "Invalid file type. Please use JPG or PNG image files.";
    }
    const maxBytes = MAX_FILE_SIZE_MB * 1024 * 1024;
    if (file.size > maxBytes) {
      return `Image is too large. Please upload a file smaller than ${MAX_FILE_SIZE_MB}MB.`;
    }
    return "";
  };

  const setNewSelectedFile = (file) => {
    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      setStatus("Please choose a valid image file.");
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    clearPrediction();
    setError("");
    setStatus("Image ready. Click Analyze Image.");
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setNewSelectedFile(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    setNewSelectedFile(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => {
    setDragActive(false);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraOn(true);
      setError("");
      setStatus("Camera started. Capture an image when ready.");
    } catch {
      setError("Could not access camera. Please allow camera permission.");
      setStatus("Camera access failed.");
    }
  };

  const captureFromCamera = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError("Failed to capture image from camera.");
          setStatus("Capture failed. Please try again.");
          return;
        }
        const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
        setNewSelectedFile(file);
      },
      "image/jpeg",
      0.95
    );
  };

  const parseApiError = async (response) => {
    try {
      const data = await response.json();
      return data.detail || data.error || "Prediction request failed.";
    } catch {
      return "Server returned an invalid response.";
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please upload or capture an image first.");
      setStatus("No image selected.");
      return;
    }
    if (!activeModel) {
      setError("Please start from Home page and choose a model first.");
      setStatus("Model not selected.");
      return;
    }

    setLoading(true);
    setError("");
    setStatus(`Analyzing image with ${activeModel.toUpperCase()} model...`);

    try {
      const formData = new FormData();
      formData.append("image", selectedFile, selectedFile.name || "skin-image.jpg");
      formData.append("model_type", activeModel);

      const response = await fetch(API_URL, { method: "POST", body: formData });
      if (!response.ok) {
        const apiError = await parseApiError(response);
        throw new Error(apiError);
      }

      const data = await response.json();
      setPrediction(normalizePrediction(data.prediction));
      setConfidence(Number(data.confidence || 0));
      setModelUsed(String(data.model_used || activeModel).toLowerCase());
      setCnnScore(Number(data.cnn_score || 0));
      setXgbScore(Number(data.xgb_score || 0));
      setStatus("Prediction complete.");
      navigate("/result");
    } catch (err) {
      if (err instanceof TypeError) {
        setError("Cannot connect to server. Make sure FastAPI backend is running on port 8000.");
      } else {
        setError(err.message || "Unexpected error while predicting.");
      }
      setStatus("Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="fade-in mx-auto w-full max-w-6xl px-4 py-10 md:px-6">
      <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-6 shadow-medical md:p-9">
        <h1 className="mb-5 inline-flex items-center gap-2 text-2xl font-extrabold tracking-tight text-slate-800 md:text-3xl">
          <Upload className="h-6 w-6 text-blue-600" />
          Detection Workspace
        </h1>

        <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-blue-100 bg-blue-50/70 px-4 py-3">
          <p className="text-sm font-medium text-blue-800">{status}</p>
          {loading && <Loader text="Running AI prediction..." />}
        </div>

        <div className="mb-5 flex flex-wrap items-center gap-3">
          <p className="rounded-full bg-blue-50 px-4 py-2 text-sm font-semibold text-blue-800">
            {activeModel
              ? `Using ${activeModel.toUpperCase()} model`
              : "No model selected (go back to Home page)"}
          </p>
        </div>

        <UploadCard
          previewUrl={previewUrl}
          dragActive={dragActive}
          loading={loading}
          cameraOn={cameraOn}
          videoRef={videoRef}
          canvasRef={canvasRef}
          onFileInputChange={handleFileChange}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onStartCamera={startCamera}
          onStopCamera={stopCamera}
          onCaptureFromCamera={captureFromCamera}
        />

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-lg bg-slate-800 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-60"
            onClick={handlePredict}
            disabled={!canPredict}
            aria-label="Run skin cancer prediction"
          >
            {loading ? "Predicting..." : "Analyze Image"}
            {!loading && <ArrowRight className="h-4 w-4" />}
          </button>
          {error ? (
            <p
              role="alert"
              className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700"
            >
              {error}
            </p>
          ) : (
            <p className="text-sm text-slate-500">
              Supported: JPG/PNG, max {MAX_FILE_SIZE_MB}MB
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

export default DetectPage;
