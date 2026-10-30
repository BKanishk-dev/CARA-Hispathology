import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import DecisionBadge from "./DecisionBadge";
import { MODEL_COLORS, ms, pct } from "../utils/format";

const COLUMNS = [
  { id: "rank", label: "Rank", get: (m) => m.rank },
  { id: "name", label: "Model", get: (m) => m.name },
  { id: "predicted_label", label: "Predicted class", get: (m) => m.predicted_label },
  { id: "confidence", label: "Confidence", get: (m) => m.confidence },
  { id: "quality", label: "Quality", get: (m) => m.quality },
  { id: "class_reliability", label: "Class reliability", get: (m) => m.class_reliability },
  { id: "cara_score", label: "CARA score", get: (m) => m.cara_score },
  { id: "decision", label: "Decision", get: (m) => (m.reliable ? 1 : 0) },
  { id: "inference_ms", label: "Time", get: (m) => m.inference_ms ?? Infinity },
];

export default function RankingTable({ models }) {
  const [sort, setSort] = useState({ id: "rank", dir: "asc" });

  const rows = useMemo(() => {
    const col = COLUMNS.find((c) => c.id === sort.id);
    const sorted = [...models].sort((a, b) => {
      const x = col.get(a);
      const y = col.get(b);
      const cmp = typeof x === "string" ? x.localeCompare(y) : x - y;
      return sort.dir === "asc" ? cmp : -cmp;
    });
    return sorted;
  }, [models, sort]);

  const toggle = (id) =>
    setSort((s) => (s.id === id ? { id, dir: s.dir === "asc" ? "desc" : "asc" } : { id, dir: "asc" }));

  return (
    <div className="card overflow-x-auto">
      <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-800">
        <thead>
          <tr>
            {COLUMNS.map((c) => (
              <th key={c.id} className="th" scope="col">
                <button className="flex items-center gap-1 hover:text-slate-800 dark:hover:text-white" onClick={() => toggle(c.id)}>
                  {c.label}
                  {sort.id === c.id && (sort.dir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
          {rows.map((m) => (
            <tr key={m.key} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
              <td className="td font-bold tabular-nums">#{m.rank}</td>
              <td className="td">
                <span className="flex items-center gap-2 font-medium">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: MODEL_COLORS[m.key] }} />
                  {m.name}
                </span>
              </td>
              <td className="td">{m.predicted_label}</td>
              <td className="td tabular-nums">{pct(m.confidence)}</td>
              <td className="td tabular-nums">{pct(m.quality)}</td>
              <td className="td tabular-nums">{pct(m.class_reliability)}</td>
              <td className="td font-semibold tabular-nums">{pct(m.cara_score)}</td>
              <td className="td">
                <DecisionBadge reliable={m.reliable} />
              </td>
              <td className="td tabular-nums">{ms(m.inference_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
