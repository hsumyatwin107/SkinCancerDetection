import { HeartPulse } from "lucide-react";

function Footer() {
  return (
    <footer className="mt-auto border-t border-blue-100/70 bg-white/70">
      <div className="mx-auto flex w-full max-w-6xl flex-col items-center justify-between gap-3 px-4 py-5 text-center md:flex-row md:px-6 md:text-left">
        <p className="text-sm text-slate-600">
          © {new Date().getFullYear()} Skin Cancer Detection. Built for educational AI screening.
        </p>
        <p className="inline-flex items-center gap-2 text-sm text-slate-600">
          <HeartPulse className="h-4 w-4 text-blue-600" />
          Crafted with React, Tailwind, and FastAPI
        </p>
      </div>
    </footer>
  );
}

export default Footer;
