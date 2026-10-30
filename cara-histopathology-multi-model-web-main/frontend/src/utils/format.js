export const pct = (v, digits = 1) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`);
export const num = (v, digits = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(digits));
export const ms = (v) => (v == null ? "–" : `${Number(v).toFixed(1)} ms`);

// colour per model (stable across charts); chosen to stay distinguishable in light and dark mode
export const MODEL_COLORS = {
  resnet18: "#6366f1",
  mobilenetv3: "#0ea5e9",
  efficientnet: "#10b981",
  graphsage: "#f59e0b",
  gan_classifier: "#ef4444",
  swin_transformer: "#8b5cf6",
  capsnet: "#ec4899",
  vmamba: "#14b8a6",
};

export function scoreTone(v) {
  if (v >= 0.8) return "emerald";
  if (v >= 0.6) return "amber";
  return "rose";
}
