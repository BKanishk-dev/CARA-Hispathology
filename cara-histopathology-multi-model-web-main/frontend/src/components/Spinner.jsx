export default function Spinner({ label }) {
  return (
    <div className="flex items-center justify-center gap-3 py-10 text-sm text-slate-500 dark:text-slate-400" role="status">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-brand-600 dark:border-slate-700 dark:border-t-brand-400" />
      {label}
    </div>
  );
}
