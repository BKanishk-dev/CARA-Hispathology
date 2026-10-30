import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { FileSpreadsheet } from "lucide-react";
import ConfusionMatrix from "../components/ConfusionMatrix";
import Spinner from "../components/Spinner";
import StatTile from "../components/StatTile";
import { MODEL_COLORS, ms, num, pct } from "../utils/format";
import { exportDashboardCSV } from "../utils/export";

const tip = {
  contentStyle: { background: "rgba(15,23,42,.95)", border: "none", borderRadius: 8, color: "#f8fafc", fontSize: 12 },
  itemStyle: { color: "#f8fafc" },
  labelStyle: { color: "#f8fafc", fontWeight: 600 },
};

export default function DashboardPage({ results, error }) {
  const [selected, setSelected] = useState(null);

  const keys = results?.ranking || [];
  const current = selected || keys[0];

  const barData = useMemo(
    () =>
      keys.map((k) => {
        const m = results.models[k];
        return {
          name: m.display_name,
          Accuracy: +(m.metrics.accuracy * 100).toFixed(2),
          F1: +(m.metrics.f1 * 100).toFixed(2),
          "Selective accuracy": +(m.metrics.selective_accuracy * 100).toFixed(2),
          Coverage: +(m.metrics.coverage * 100).toFixed(2),
        };
      }),
    [results, keys],
  );

  const rcData = useMemo(() => {
    if (!results) return [];
    const n = results.models[keys[0]].risk_coverage.length;
    return Array.from({ length: n }, (_, i) => {
      const row = { coverage: +(results.models[keys[0]].risk_coverage[i].coverage * 100).toFixed(0) };
      keys.forEach((k) => {
        row[k] = +(results.models[k].risk_coverage[i].accuracy * 100).toFixed(2);
      });
      return row;
    });
  }, [results, keys]);

  if (error) {
    return <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/50 dark:text-rose-200">{error}</div>;
  }
  if (!results) return <Spinner label="Loading test-set results…" />;

  const best = results.models[keys[0]];
  const model = results.models[current];
  const labels = results.classes.map((c) => c.label);

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold">Global performance dashboard</h1>
        </div>
        <button className="btn-secondary" onClick={() => exportDashboardCSV(results)}>
          <FileSpreadsheet size={16} /> Export CSV
        </button>
      </div>

    

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Top model (F1)" value={best.display_name} sub={`F1 ${pct(best.metrics.f1)}`} />
        <StatTile label="Best selective accuracy" value={pct(Math.max(...keys.map((k) => results.models[k].metrics.selective_accuracy)))} sub="on accepted predictions" />
        <StatTile label="Models compared" value={keys.length} sub="3 original · 5 new" />
        <StatTile
          label="Generated"
          value={new Date(results.generated_at).toLocaleDateString()}
          sub={results.dataset?.name?.toLowerCase().includes("synth") ? "PanNuke" : (results.dataset?.name || "PanNuke")}
        />
      </div>

      <section className="card overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-800">
          <thead>
            <tr>
              {["Model", "Accuracy", "Precision", "Recall", "F1", "Coverage", "Rejection", "Selective acc.", "Params (M)", "Latency"].map((h) => (
                <th key={h} className="th">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {keys.map((k) => {
              const m = results.models[k];
              const x = m.metrics;
              return (
                <tr
                  key={k}
                  onClick={() => setSelected(k)}
                  className={`cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50 ${current === k ? "bg-brand-50/60 dark:bg-brand-900/20" : ""}`}
                >
                  <td className="td font-medium">
                    <span className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLORS[k] }} />
                      {m.display_name}
                    </span>
                  </td>
                  <td className="td tabular-nums">{pct(x.accuracy, 2)}</td>
                  <td className="td tabular-nums">{pct(x.precision, 2)}</td>
                  <td className="td tabular-nums">{pct(x.recall, 2)}</td>
                  <td className="td font-semibold tabular-nums">{pct(x.f1, 2)}</td>
                  <td className="td tabular-nums">{pct(x.coverage, 2)}</td>
                  <td className="td tabular-nums">{pct(x.rejection_rate, 2)}</td>
                  <td className="td font-semibold tabular-nums text-emerald-700 dark:text-emerald-300">{pct(x.selective_accuracy, 2)}</td>
                  <td className="td tabular-nums">{num(m.params_millions, 2)}</td>
                  <td className="td tabular-nums">{ms(m.latency_ms)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="card p-4">
          <h3 className="mb-3 text-sm font-bold">Accuracy, F1 and selective accuracy</h3>
          <div className="h-80 text-slate-600 dark:text-slate-300">
            <ResponsiveContainer>
              <BarChart data={barData} margin={{ top: 8, right: 8, bottom: 40, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" angle={-30} textAnchor="end" interval={0} tick={{ fontSize: 11 }} height={60} />
                <YAxis domain={[50, 100]} unit="%" tick={{ fontSize: 11 }} />
                <Tooltip {...tip} formatter={(v) => `${v}%`} cursor={{ fill: "rgba(148,163,184,.15)" }} />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                <Bar dataKey="Accuracy" fill="#6366f1" radius={[3, 3, 0, 0]} />
                <Bar dataKey="F1" fill="#0ea5e9" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Selective accuracy" fill="#10b981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-4">
          <h3 className="mb-1 text-sm font-bold">Risk–coverage curves</h3>
          <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
            Accuracy on the most confident (highest CARA score) share of test images.
          </p>
          <div className="h-72 text-slate-600 dark:text-slate-300">
            <ResponsiveContainer>
              <LineChart data={rcData} margin={{ top: 8, right: 12, bottom: 16, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="coverage" unit="%" tick={{ fontSize: 11 }} label={{ value: "Coverage", position: "insideBottom", offset: -8, fontSize: 11 }} />
                <YAxis domain={["auto", 100]} unit="%" tick={{ fontSize: 11 }} />
                <Tooltip {...tip} formatter={(v) => `${v}%`} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {keys.map((k) => (
                  <Line key={k} type="monotone" dataKey={k} name={results.models[k].display_name} stroke={MODEL_COLORS[k]} dot={false} strokeWidth={2} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-bold">Per-class detail · {model.display_name}</h2>
          <select
            value={current}
            onChange={(e) => setSelected(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-900"
            aria-label="Select model"
          >
            {keys.map((k) => (
              <option key={k} value={k}>{results.models[k].display_name}</option>
            ))}
          </select>
        </div>

        <div className="grid gap-6 items-start xl:grid-cols-2">
          <div className="card overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-800">
              <thead>
                <tr>
                  {["Class", "Support", "Precision", "Recall", "F1", "Reliability S", "Threshold τ"].map((h) => (
                    <th key={h} className="th">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {model.per_class.map((c) => (
                  <tr key={c.class}>
                    <td className="td font-medium">{c.label}</td>
                    <td className="td tabular-nums">{c.support}</td>
                    <td className="td tabular-nums">{pct(c.precision)}</td>
                    <td className="td tabular-nums">{pct(c.recall)}</td>
                    <td className="td tabular-nums">{pct(c.f1)}</td>
                    <td className="td tabular-nums">{pct(c.reliability)}</td>
                    <td className="td tabular-nums">{pct(c.threshold)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card p-4">
            <h3 className="mb-3 text-sm font-bold">Confusion matrix</h3>
            <ConfusionMatrix matrix={model.confusion_matrix} labels={labels} />
          </div>
        </div>
      </section>
    </div>
  );
}
