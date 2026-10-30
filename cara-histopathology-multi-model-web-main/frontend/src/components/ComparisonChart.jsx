import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const tip = {
  contentStyle: {
    background: "rgba(15,23,42,.95)",
    border: "none",
    borderRadius: 8,
    color: "#f8fafc",
    fontSize: 12,
  },
  itemStyle: { color: "#f8fafc" },
  labelStyle: { color: "#f8fafc", fontWeight: 600 },
};

/** Per-model confidence, CARA score and class threshold for the uploaded image. */
export default function ComparisonChart({ models }) {
  const data = models.map((m) => ({
    name: m.name,
    Confidence: +(m.confidence * 100).toFixed(1),
    "CARA score": +(m.cara_score * 100).toFixed(1),
    Threshold: +(m.threshold * 100).toFixed(1),
  }));
  return (
    <div className="card p-4">
      <h3 className="mb-1 text-sm font-bold">Confidence vs CARA score</h3>
      <p className="mb-3 text-xs text-slate-500 dark:text-slate-400">
        A prediction is reliable when its CARA score reaches the class-specific threshold.
      </p>
      <div className="h-72 w-full text-slate-600 dark:text-slate-300">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 40, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" angle={-30} textAnchor="end" interval={0} tick={{ fontSize: 11 }} height={60} />
            <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
            <Tooltip {...tip} formatter={(v) => `${v}%`} cursor={{ fill: "rgba(148,163,184,.15)" }} />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
            <Bar dataKey="Confidence" fill="#6366f1" radius={[3, 3, 0, 0]} />
            <Bar dataKey="CARA score" fill="#10b981" radius={[3, 3, 0, 0]} />
            <Bar dataKey="Threshold" fill="#94a3b8" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
