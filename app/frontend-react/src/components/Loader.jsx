function Loader({ text = "Analyzing image..." }) {
  return (
    <div
      className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1.5 text-sm font-medium text-blue-700"
      aria-live="polite"
    >
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-blue-200 border-t-blue-600" />
      <span>{text}</span>
    </div>
  );
}

export default Loader;
