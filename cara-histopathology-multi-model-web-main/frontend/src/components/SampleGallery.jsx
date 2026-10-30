import { assetUrl } from "../api/client";

export default function SampleGallery({ samples, onPick, disabled, columns = "grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8" }) {
  if (!samples.length) {
    return <p className="text-sm text-slate-500 dark:text-slate-400">No sample images found in data/samples.</p>;
  }
  return (
    <div className={`grid gap-3 ${columns}`}>
      {samples.map((s) => (
        <button
          key={s.file}
          onClick={() => onPick(s)}
          disabled={disabled}
          className="group overflow-hidden rounded-lg border border-slate-200 bg-white text-left transition hover:border-brand-400 hover:shadow-card focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-400 disabled:opacity-60 dark:border-slate-800 dark:bg-slate-900"
          title={`Analyze ${s.label}`}
        >
          <img
            src={assetUrl(s.url)}
            alt={`${s.label} sample`}
            loading="lazy"
            className="aspect-square w-full object-cover transition group-hover:scale-105"
          />
          <div className="px-2 py-1.5">
            <div className="truncate text-xs font-semibold">{s.label}</div>
            <div className="truncate text-[10px] text-slate-500 dark:text-slate-400">
              {s.low_quality ? "sample · low quality" : "sample"}
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
