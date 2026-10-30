/** Row-normalised confusion matrix rendered as a CSS grid heat-map. */
export default function ConfusionMatrix({ matrix, labels }) {
  const n = matrix.length;
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  return (
    <div className="w-full overflow-x-auto">
      <div className="min-w-[500px] pb-2 pr-8">
        <div
          className="grid"
          style={{ gridTemplateColumns: `96px repeat(${n}, minmax(16px, 1fr))` }}
        >
          {/* Top-left corner */}
          <div className="h-20" />

          {/* X-axis column headers (Predicted class) */}
          {labels.map((l, j) => (
            <div key={j} className="relative h-20" title={`Predicted: ${l}`}>
              <div
                className="absolute bottom-1.5 left-1/2 origin-bottom-left whitespace-nowrap text-[10px] font-medium text-slate-500 select-none dark:text-slate-400"
                style={{ transform: "rotate(-45deg)" }}
              >
                {l}
              </div>
            </div>
          ))}

          {/* Y-axis row labels + Matrix cells */}
          {matrix.map((row, i) => (
            <div key={i} className="contents">
              <div
                className="flex items-center justify-end truncate pr-2.5 text-right text-[10px] font-medium text-slate-500 dark:text-slate-400"
                title={`True: ${labels[i]}`}
              >
                {labels[i]}
              </div>
              {row.map((v, j) => {
                const frac = rowSums[i] ? v / rowSums[i] : 0;
                return (
                  <div
                    key={j}
                    className="aspect-square border border-white/30 transition-transform hover:scale-125 hover:z-10 hover:shadow-md dark:border-slate-900/60"
                    style={{ background: `rgba(99,102,241,${v === 0 ? 0.05 : 0.15 + frac * 0.85})` }}
                    title={`True ${labels[i]} → predicted ${labels[j]}: ${v} (${(frac * 100).toFixed(0)}% of row)`}
                  />
                );
              })}
            </div>
          ))}
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
          <span>Rows = true class · Columns = predicted class</span>
          <span className="flex items-center gap-1.5">
            <span>0%</span>
            <span
              className="inline-block h-2 w-16 rounded"
              style={{
                background: "linear-gradient(to right, rgba(99,102,241,0.08), rgba(99,102,241,1))",
              }}
            />
            <span>100%</span>
          </span>
        </div>
      </div>
    </div>
  );
}
