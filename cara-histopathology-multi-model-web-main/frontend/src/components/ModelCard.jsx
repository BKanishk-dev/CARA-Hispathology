import { AlertTriangle, Timer, Trophy } from "lucide-react";
import DecisionBadge from "./DecisionBadge";
import MeterBar from "./MeterBar";
import { MODEL_COLORS, ms, pct } from "../utils/format";
import ShapExplanation from "./ShapExplanation";

export default function ModelCard({ m, consensusLabel, previewUrl }) {
  return (
    <article
      className={`card flex flex-col overflow-hidden ${
        m.reliable ? "" : "ring-1 ring-rose-300/70 dark:ring-rose-800/70"
      }`}
    >
      <div className="h-1.5" style={{ background: MODEL_COLORS[m.key] }} />
      <div className="flex flex-1 flex-col gap-4 p-4">
        <header className="flex items-start justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold">{m.name}</h3>
            <span
              className={`badge mt-1 ${
                m.origin === "new"
                  ? "bg-brand-100 text-brand-800 dark:bg-brand-900/40 dark:text-brand-200"
                  : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300"
              }`}
            >
              {m.origin === "new" ? "New model" : "Original"}
            </span>
          </div>
          <div
            className="flex items-center gap-1 rounded-lg bg-slate-100 px-2 py-1 text-xs font-semibold dark:bg-slate-800"
            title="Rank by reliability, then CARA score"
          >
            <Trophy size={13} /> #{m.rank}
          </div>
        </header>

        <div>
          <div className="label">Predicted class</div>
          <div className="mt-0.5 text-xl font-bold leading-tight">{m.predicted_label}</div>
          <div className="mt-2">
            <DecisionBadge reliable={m.reliable} size="lg" />
          </div>
        </div>

        <div className="space-y-3">
          <MeterBar label="Softmax confidence" value={m.confidence} />
          <MeterBar label="Image quality" value={m.quality} />
          <MeterBar label="Class reliability" value={m.class_reliability} />
          <MeterBar
            label="CARA score"
            value={m.cara_score}
            threshold={m.threshold}
            hint={`Tick = class threshold ${pct(m.threshold)}`}
          />
        </div>

        <ol className="space-y-1 text-xs text-slate-600 dark:text-slate-300">
          {m.top3.map((t, i) => (
            <li key={t.class} className="flex justify-between gap-2">
              <span className="truncate">
                {i + 1}. {t.label}
              </span>
              <span className="tabular-nums text-slate-500 dark:text-slate-400">{pct(t.prob)}</span>
            </li>
          ))}
        </ol>

        <ShapExplanation shap={m.shap} previewUrl={previewUrl} modelName={m.name} />

        <footer className="mt-auto flex items-center justify-between border-t border-slate-100 pt-3 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
          <span className="flex items-center gap-1.5">
            <Timer size={13} /> Inference
          </span>
          <span className="font-semibold tabular-nums text-slate-700 dark:text-slate-200">{ms(m.inference_ms)}</span>
        </footer>

        {m.model_error && (
          <div className="flex gap-2 rounded-lg bg-amber-50 p-2 text-[11px] text-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
            <AlertTriangle size={14} className="mt-0.5 shrink-0" />
            <span>Network could not run (timing unavailable): {m.model_error}</span>
          </div>
        )}
      </div>
    </article>
  );
}
