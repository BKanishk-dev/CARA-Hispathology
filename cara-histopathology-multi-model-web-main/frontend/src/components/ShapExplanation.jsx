import { useState } from "react";
import { AlertCircle, Eye, Info, Sparkles } from "lucide-react";

export default function ShapExplanation({ shap, previewUrl, modelName }) {
  const [viewMode, setViewMode] = useState("overlay"); // "overlay" | "shap" | "original"
  const [opacity, setOpacity] = useState(0.75);
  const [showGuide, setShowGuide] = useState(false);

  if (!shap) return null;

  if (!shap.available) {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50/60 p-2.5 text-xs text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-300">
        <div className="flex items-start gap-1.5 font-semibold">
          <AlertCircle size={14} className="mt-0.5 shrink-0" />
          <span>SHAP attribution unavailable</span>
        </div>
        <p className="mt-1 text-[11px] leading-relaxed text-amber-700 dark:text-amber-400">
          {shap.error || "Gradients or runtime limits not supported for this architecture."}
        </p>
      </div>
    );
  }

  const { predicted_label, attribution_map, positive_pct, negative_pct } = shap;

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/80 p-2.5 text-xs dark:border-slate-800 dark:bg-slate-800/40">
      {/* Header */}
      <div className="flex items-center justify-between gap-1.5 pb-2 border-b border-slate-200/70 dark:border-slate-700/60">
        <div className="flex items-center gap-1.5 font-bold text-slate-800 dark:text-slate-100">
          <Sparkles size={13} className="text-brand-600 dark:text-brand-400" />
          <span>SHAP Feature Attribution</span>
        </div>
      </div>

      {/* Target Class Explanation Header */}
      <div className="mt-2 text-[11px] text-slate-600 dark:text-slate-300">
        Explaining predicted class: <strong className="text-slate-900 dark:text-white font-semibold">{predicted_label}</strong>
      </div>

      {/* Image Visualization Box */}
      <div className="relative mt-2 overflow-hidden rounded-md border border-slate-200 bg-slate-950 aspect-square dark:border-slate-700">
        {/* Base Uploaded Image */}
        {previewUrl && viewMode !== "shap" && (
          <img
            src={previewUrl}
            alt="Histopathology slice"
            className="absolute inset-0 h-full w-full object-cover"
          />
        )}

        {/* SHAP Attribution Heatmap Overlay */}
        {attribution_map && viewMode !== "original" && (
          <img
            src={attribution_map}
            alt={`SHAP heatmap for ${predicted_label}`}
            className="absolute inset-0 h-full w-full object-cover transition-opacity duration-150"
            style={{ opacity: viewMode === "shap" ? 1.0 : opacity }}
          />
        )}
      </div>

      {/* View Mode Controls */}
      <div className="mt-2 flex items-center justify-between gap-1 text-[11px]">
        <div className="inline-flex rounded-md bg-slate-200/70 p-0.5 dark:bg-slate-700/60">
          {[
            { id: "overlay", label: "Overlay" },
            { id: "shap", label: "SHAP" },
            { id: "original", label: "Image" },
          ].map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setViewMode(t.id)}
              className={`rounded px-2 py-0.5 font-medium transition ${
                viewMode === t.id
                  ? "bg-white text-slate-900 shadow-sm dark:bg-slate-900 dark:text-white"
                  : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {viewMode === "overlay" && (
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-slate-400">Opacity:</span>
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.05"
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="h-1.5 w-16 cursor-pointer accent-brand-600"
              aria-label="Attribution overlay opacity"
            />
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-2.5 space-y-1.5 rounded bg-white/70 p-2 text-[10px] dark:bg-slate-900/60 border border-slate-200/50 dark:border-slate-800">
        <div className="flex items-center justify-between text-slate-700 dark:text-slate-300 font-medium">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-rose-500" />
            <span>Positive (supports {predicted_label})</span>
          </span>
          {positive_pct != null && <span className="tabular-nums font-semibold">{positive_pct}%</span>}
        </div>
        <div className="flex items-center justify-between text-slate-700 dark:text-slate-300 font-medium">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-full bg-blue-500" />
            <span>Negative (opposes {predicted_label})</span>
          </span>
          {negative_pct != null && <span className="tabular-nums font-semibold">{negative_pct}%</span>}
        </div>
      </div>

      {/* How to Read / Clinical Disclaimer */}
      <div className="mt-2">
        <button
          type="button"
          onClick={() => setShowGuide((v) => !v)}
          className="flex items-center gap-1 text-[10px] font-semibold text-brand-600 hover:underline dark:text-brand-400"
        >
          <Info size={11} />
          <span>{showGuide ? "Hide interpretation guide" : "How to read this visualization"}</span>
        </button>

        {showGuide && (
          <div className="mt-1.5 rounded bg-slate-100 p-2 text-[10px] leading-relaxed text-slate-600 dark:bg-slate-800 dark:text-slate-300 border border-slate-200/60 dark:border-slate-700/60">
            <p>
              <strong>Red highlights</strong> identify tissue regions that pushed the model towards predicting <strong>{predicted_label}</strong>. <strong>Blue regions</strong> lowered that probability.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
