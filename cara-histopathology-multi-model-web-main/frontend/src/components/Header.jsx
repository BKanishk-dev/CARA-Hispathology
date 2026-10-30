import { Microscope, Moon, Sun, LayoutDashboard, ScanSearch, Images } from "lucide-react";

const TABS = [
  { id: "analyze", label: "Analyze", icon: ScanSearch },
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "gallery", label: "Gallery", icon: Images },
];

export default function Header({ page, onNavigate, theme, onToggleTheme }) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/85 backdrop-blur dark:border-slate-800 dark:bg-slate-950/85">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-white">
            <Microscope size={20} />
          </span>
          <div className="leading-tight">
            <div className="text-sm font-bold tracking-tight">CARA Histopathology</div>
            <div className="hidden text-xs text-slate-500 dark:text-slate-400 sm:block">
              Confidence-aware analysis · 8 models
            </div>
          </div>
        </div>

        <nav className="ml-2 flex flex-1 items-center gap-1 overflow-x-auto" aria-label="Main">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              aria-current={page === id ? "page" : undefined}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                page === id
                  ? "bg-brand-50 text-brand-700 dark:bg-brand-900/40 dark:text-brand-200"
                  : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
              }`}
            >
              <Icon size={16} />
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </nav>

        <button
          onClick={onToggleTheme}
          className="btn-secondary !px-2.5"
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Light mode" : "Dark mode"}
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  );
}
