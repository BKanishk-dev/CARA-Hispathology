import { useCallback, useRef, useState } from "react";
import { ImagePlus, X } from "lucide-react";

const ACCEPT = "image/png,image/jpeg,image/tiff,image/bmp,image/webp";

export default function Dropzone({ file, previewUrl, onFile, onClear, disabled }) {
  const [over, setOver] = useState(false);
  const inputRef = useRef(null);

  const pick = useCallback(
    (files) => {
      const f = files && files[0];
      if (!f) return;
      if (!f.type.startsWith("image/")) {
        alert("Please choose an image file (PNG, JPG, TIFF, BMP).");
        return;
      }
      onFile(f);
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) pick(e.dataTransfer.files);
      }}
      className={`relative flex min-h-[260px] flex-col items-center justify-center rounded-xl border-2 border-dashed p-4 text-center transition ${
        over
          ? "border-brand-500 bg-brand-50 dark:bg-brand-900/20"
          : "border-slate-300 bg-white hover:border-brand-400 dark:border-slate-700 dark:bg-slate-900"
      } ${disabled ? "opacity-60" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = "";
        }}
      />
      {previewUrl ? (
        <>
          <img
            src={previewUrl}
            alt="Selected histopathology preview"
            className="max-h-64 rounded-lg object-contain shadow-card"
          />
          <div className="mt-3 max-w-full truncate text-xs text-slate-500 dark:text-slate-400">
            {file?.name} · {(file?.size / 1024).toFixed(0)} KB
          </div>
          <button
            type="button"
            onClick={onClear}
            disabled={disabled}
            className="absolute right-3 top-3 rounded-full bg-slate-900/70 p-1.5 text-white hover:bg-slate-900"
            aria-label="Remove image"
          >
            <X size={14} />
          </button>
        </>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className="flex flex-col items-center gap-3 rounded-lg p-6 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-400"
        >
          <span className="flex h-14 w-14 items-center justify-center rounded-full bg-brand-50 text-brand-600 dark:bg-brand-900/40 dark:text-brand-200">
            <ImagePlus size={28} />
          </span>
          <span className="text-base font-semibold">Drag &amp; drop a histopathology image</span>
          <span className="text-sm text-slate-500 dark:text-slate-400">
            or click to browse · PNG, JPG, TIFF, BMP · centre-cropped and resized to 256 × 256
          </span>
        </button>
      )}
    </div>
  );
}
