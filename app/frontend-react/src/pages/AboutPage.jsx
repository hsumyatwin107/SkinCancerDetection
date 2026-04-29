import { Brain, Cpu, FlaskConical, ShieldAlert } from "lucide-react";

function AboutPage() {
  return (
    <section className="fade-in mx-auto w-full max-w-6xl px-4 py-10 md:px-6">
      <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-6 shadow-medical md:p-9">
        <h1 className="text-2xl font-extrabold tracking-tight text-slate-800 md:text-3xl">
          About This Application
        </h1>
        <p className="mt-4 text-sm leading-relaxed text-slate-600 md:text-base">
          Skin Cancer Detection helps users upload skin images and receive an AI-based screening
          prediction. It is designed to support early awareness through a clean and professional
          interface.
        </p>

        <div className="mt-7 grid gap-4 md:grid-cols-2">
          <article className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
            <h2 className="inline-flex items-center gap-2 text-lg font-semibold text-slate-800">
              <FlaskConical className="h-5 w-5 text-blue-600" />
              How it works
            </h2>
            <p className="mt-2 text-sm text-slate-600">
              The backend model analyzes uploaded images and returns a predicted class with
              confidence score so users can better understand potential risk levels.
            </p>
          </article>

          <article className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
            <h2 className="inline-flex items-center gap-2 text-lg font-semibold text-slate-800">
              <Cpu className="h-5 w-5 text-indigo-600" />
              Technology stack
            </h2>
            <p className="mt-2 text-sm text-slate-600">
              <span className="inline-flex items-center gap-1">
                <Brain className="h-4 w-4 text-violet-600" />
                VIT/CNN+XGBoost model
              </span>{" "}
              for image classification, FastAPI for model serving, React + Vite for frontend, and
              Tailwind CSS for modern, responsive UI.
            </p>
          </article>
        </div>

        <p className="mt-7 inline-flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-relaxed text-amber-800">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
          Disclaimer: This tool is for educational purposes only and not medical advice or a
          diagnosis. Please consult a qualified healthcare professional for clinical decisions.
        </p>
      </div>
    </section>
  );
}

export default AboutPage;
