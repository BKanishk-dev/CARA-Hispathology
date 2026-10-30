import { pct, scoreTone } from "../utils/format";

const FILL = {
  emerald: "bg-emerald-500",
  amber: "bg-amber-500",
  rose: "bg-rose-500",
};

/** Horizontal 0-1 meter with an optional threshold tick. */
export default function MeterBar({ label, value, threshold, hint }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="label">{label}</span>
        <span className="text-sm font-semibold tabular-nums">{pct(value)}</span>
      </div>
      <div
        className="relative h-2 w-full rounded-full bg-slate-200 dark:bg-slate-800"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={value}
      >
        <div
          className={`h-2 rounded-full ${FILL[scoreTone(value)]}`}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
        {threshold != null && (
          <div
            className="absolute -top-1 h-4 w-0.5 bg-slate-900 dark:bg-white"
            style={{ left: `${Math.max(0, Math.min(1, threshold)) * 100}%` }}
            title={`Class threshold ${pct(threshold)}`}
          />
        )}
      </div>
      {hint && <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">{hint}</div>}
    </div>
  );
}
