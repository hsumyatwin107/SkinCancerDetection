import { LoaderCircle, Sparkles } from "lucide-react";

function PageLoader() {
  return (
    <div className="flex min-h-[50vh] items-center justify-center">
      <div className="inline-flex items-center gap-3 rounded-full border border-blue-100 bg-white/90 px-5 py-3 text-sm font-semibold text-blue-700 shadow-sm">
        <LoaderCircle className="h-5 w-5 animate-spin" />
        Loading page
        <Sparkles className="h-4 w-4 text-indigo-500" />
      </div>
    </div>
  );
}

export default PageLoader;
