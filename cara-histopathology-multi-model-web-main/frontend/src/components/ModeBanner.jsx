import { AlertTriangle } from "lucide-react";

export default function ModeBanner({ error }) {
  if (!error) return null;
  return (
    <div className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/60 dark:text-rose-200">
      <div className="mx-auto flex max-w-7xl items-center gap-2">
        <AlertTriangle size={16} className="shrink-0" />
        <span>{error}</span>
      </div>
    </div>
  );
}