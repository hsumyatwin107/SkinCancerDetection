import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, BrainCircuit, Cpu, Microscope, ShieldCheck, X } from "lucide-react";
import { usePrediction } from "../context/PredictionContext";

function HomePage() {
  const [showModelPicker, setShowModelPicker] = useState(false);
  const navigate = useNavigate();
  const { setSelectedModel } = usePrediction();

  const handleModelSelect = (modelType) => {
    setSelectedModel(modelType);
    setShowModelPicker(false);
    navigate("/detect", { state: { model: modelType } });
  };

  return (
    <section className="fade-in mx-auto w-full max-w-6xl px-4 py-12 md:px-6 md:py-14">
      <div className="grid items-center gap-10 rounded-3xl border border-blue-100/80 bg-white/90 p-7 shadow-medical md:grid-cols-2 md:p-12">
        <div>
          <p className="mb-4 inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold tracking-wide text-blue-700">
            <Microscope className="h-3.5 w-3.5" />
            Medical AI Screening
          </p>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-800 md:text-5xl">
            AI Skin Cancer Detection
          </h1>
          <p className="mt-5 text-sm leading-relaxed text-slate-600 md:text-base">
            Upload a skin image and receive a fast AI-powered screening result with confidence
            scoring to support early awareness.
          </p>
          <button
            type="button"
            onClick={() => setShowModelPicker(true)}
            className="mt-7 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-blue-700"
          >
            Start Detection
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>

        <div className="rounded-2xl border border-blue-100 bg-gradient-to-b from-blue-50 to-white p-7">
          <div className="mx-auto flex h-48 w-48 items-center justify-center rounded-full border border-blue-200 bg-blue-100 text-center shadow-inner">
            <div>
              <p className="text-4xl font-bold text-blue-700">AI</p>
              <p className="text-xs font-semibold text-blue-600">Medical Scan</p>
            </div>
          </div>
          <p className="mt-6 text-center text-sm text-slate-600">
            Professional, simple, and trustworthy interface for early skin risk screening.
          </p>
          <div className="mt-4 flex items-center justify-center gap-2 text-xs font-semibold text-emerald-700">
            <ShieldCheck className="h-4 w-4" />
            Secure and privacy-aware workflow
          </div>
        </div>
      </div>

      {showModelPicker && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 px-4">
          <div className="w-full max-w-xl rounded-2xl border border-slate-200 bg-white p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-bold text-slate-800">Choose Detection Model</h2>
              <button
                type="button"
                onClick={() => setShowModelPicker(false)}
                className="rounded-md border border-slate-200 p-1.5 text-slate-500 transition hover:bg-slate-50"
                aria-label="Close model selection modal"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <button
                type="button"
                onClick={() => handleModelSelect("cnn")}
                className="rounded-xl border border-slate-200 bg-white p-4 text-left transition hover:border-blue-300 hover:bg-blue-50/60"
              >
                <Cpu className="mb-2 h-5 w-5 text-blue-600" />
                <p className="font-semibold text-slate-800">CNN+XGBoost Model</p>
                <p className="mt-1 text-sm text-slate-600">
                  Classic convolutional neural network for lesion classification.
                </p>
              </button>

              <button
                type="button"
                onClick={() => handleModelSelect("vit")}
                className="rounded-xl border border-slate-200 bg-white p-4 text-left transition hover:border-blue-300 hover:bg-blue-50/60"
              >
                <BrainCircuit className="mb-2 h-5 w-5 text-violet-600" />
                <p className="font-semibold text-slate-800">ViT Model</p>
                <p className="mt-1 text-sm text-slate-600">
                  Vision Transformer option for transformer-based image understanding.
                </p>
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

export default HomePage;
