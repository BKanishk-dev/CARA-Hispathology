import { ShieldAlert, ShieldCheck } from "lucide-react";

export default function DecisionBadge({ reliable, size = "md" }) {
  const Icon = reliable ? ShieldCheck : ShieldAlert;
  const tone = reliable
    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200"
    : "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-200";
  return (
    <span className={`badge ${tone} ${size === "lg" ? "!px-3 !py-1 !text-sm" : ""}`}>
      <Icon size={size === "lg" ? 16 : 13} />
      {reliable ? "Reliable" : "Unreliable"}
    </span>
  );
}
