function UploadCard({
  previewUrl,
  dragActive,
  loading,
  cameraOn,
  videoRef,
  canvasRef,
  onFileInputChange,
  onDrop,
  onDragOver,
  onDragLeave,
  onStartCamera,
  onStopCamera,
  onCaptureFromCamera,
}) {
  return (
    <section className="rounded-2xl border border-blue-100 bg-white p-5 shadow-sm transition-all duration-300 hover:shadow-md">
      <h2 className="text-lg font-semibold text-slate-800">Upload Skin Image</h2>
      <p className="mt-1 text-sm text-slate-600">
        Drag and drop an image here, or use the buttons below.
      </p>

      <div
        className={`mt-4 rounded-xl border-2 border-dashed p-4 text-center transition ${
          dragActive
            ? "border-blue-400 bg-blue-50"
            : "border-slate-200 bg-slate-50/70 hover:border-blue-300 hover:bg-blue-50/50"
        }`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
      >
        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Selected skin lesion preview"
            className="mx-auto max-h-64 w-full rounded-lg object-contain"
          />
        ) : (
          <p className="py-10 text-sm text-slate-500">
            Drop JPG or PNG image here (up to 10MB)
          </p>
        )}
      </div>

      <div className="mt-4 flex flex-wrap gap-3">
        <label className="inline-flex cursor-pointer items-center rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus-within:ring-2 focus-within:ring-blue-300 focus-within:ring-offset-2">
          <input
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            onChange={onFileInputChange}
            disabled={loading}
            className="hidden"
            aria-label="Upload image file"
          />
          Upload Image
        </label>

        {!cameraOn ? (
          <button
            type="button"
            onClick={onStartCamera}
            disabled={loading}
            className="rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-blue-200 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-60"
            aria-label="Start camera for capture"
          >
            Capture Photo
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={onCaptureFromCamera}
              disabled={loading}
              className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-2.5 text-sm font-semibold text-emerald-700 transition hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-60"
              aria-label="Take photo from camera"
            >
              Take Snapshot
            </button>
            <button
              type="button"
              onClick={onStopCamera}
              disabled={loading}
              className="rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
              aria-label="Stop camera"
            >
              Stop Camera
            </button>
          </>
        )}
      </div>

      {cameraOn && (
        <div className="mt-4">
          <video
            ref={videoRef}
            className="w-full rounded-lg border border-slate-200 bg-slate-100"
            autoPlay
            playsInline
            muted
          />
          <canvas ref={canvasRef} className="hidden" />
        </div>
      )}
    </section>
  );
}

export default UploadCard;
