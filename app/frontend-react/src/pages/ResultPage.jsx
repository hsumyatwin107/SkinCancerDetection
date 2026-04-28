import { Link, Navigate } from "react-router-dom";
import { Repeat, ShieldAlert } from "lucide-react";
import ResultCard from "../components/ResultCard";
import { usePrediction } from "../context/PredictionContext";

function ResultPage() {
  const { prediction, confidence, modelUsed, cnnScore, xgbScore, clearAll } = usePrediction();

  if (!prediction) {
    return <Navigate to="/detect" replace />;
  }

  return (
    <section className="fade-in mx-auto w-full max-w-6xl px-4 py-10 md:px-6">
      <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-6 shadow-medical md:p-9">
        <h1 className="text-2xl font-extrabold tracking-tight text-slate-800 md:text-3xl">
          Screening Result
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-slate-600">
          This AI result supports awareness and should not replace professional medical advice.
        </p>

        <div className="mt-6">
          <ResultCard
            prediction={prediction}
            confidence={confidence}
            modelUsed={modelUsed}
            cnnScore={cnnScore}
            xgbScore={xgbScore}
          />
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            to="/detect"
            onClick={clearAll}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-blue-700"
          >
            <Repeat className="h-4 w-4" />
            Try another image
          </Link>
          <Link
            to="/about"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-blue-200 hover:bg-blue-50"
          >
            <ShieldAlert className="h-4 w-4" />
            Read disclaimer
          </Link>
        </div>
      </div>
    </section>
  );
}

export default ResultPage;
