import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { CircleHelp, FolderOpen, LayoutDashboard, Menu, PenLine, RefreshCw, Scale, X } from "lucide-react";
import { Logo, StatusDot } from "./ui";
import { useModels } from "../context/Models";
import { modelLabel } from "../lib/models";

const NAV = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/editor", label: "Editor", icon: PenLine },
  { to: "/documents", label: "My documents", icon: FolderOpen },
  { to: "/compare", label: "Compare models", icon: Scale },
  { to: "/about", label: "Help", icon: CircleHelp },
];

function ApiStatus() {
  const { loading, error, models, reload } = useModels();
  if (loading && !models.length) return <p className="text-xs text-ink-300">Connecting to the server…</p>;
  if (error) {
    return (
      <div className="text-xs">
        <p className="flex items-center gap-2 font-semibold text-red-300"><StatusDot status="offline" /> Server offline</p>
        <p className="mt-1 text-ink-300">Start the backend, then retry.</p>
        <button onClick={reload} className="mt-2 inline-flex items-center gap-1.5 font-semibold text-white underline-offset-2 hover:underline">
          <RefreshCw size={12} aria-hidden /> Retry
        </button>
      </div>
    );
  }
  return (
    <ul className="space-y-1.5 text-xs text-ink-200">
      {models.map((m) => (
        <li key={m.id} className="flex items-center gap-2" title={m.reason || m.name}>
          <StatusDot status={m.status} />
          <span className="font-semibold text-white">{modelLabel(m.id)}</span>
          <span className="text-ink-300">{m.status === "ready" ? "ready" : "not trained"}</span>
        </li>
      ))}
    </ul>
  );
}

function SidebarContent({ onNavigate }) {
  return (
    <div className="flex h-full flex-col">
      <Link to="/" className="px-5 py-5" onClick={onNavigate} aria-label="PredictAI home">
        <Logo dark />
      </Link>
      <nav className="flex-1 space-y-1 px-3" aria-label="Main">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            onClick={onNavigate}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-semibold transition-colors ${
                isActive ? "bg-white/10 text-white" : "text-ink-200 hover:bg-white/5 hover:text-white"
              }`
            }
          >
            <Icon size={18} aria-hidden /> {label}
          </NavLink>
        ))}
      </nav>
      <div className="m-3 rounded-lg border border-white/10 bg-white/5 p-3">
        <ApiStatus />
      </div>
    </div>
  );
}

export default function AppShell() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => {
    setOpen(false);
  }, [pathname]);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <div className="min-h-screen lg:pl-64">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 bg-ink-900 lg:block">
        <SidebarContent />
      </aside>

      <header className="sticky top-0 z-20 flex items-center justify-between border-b border-ink-100 bg-white/90 px-4 py-3 backdrop-blur lg:hidden">
        <Link to="/" aria-label="PredictAI home"><Logo /></Link>
        <button onClick={() => setOpen(true)} className="btn btn-ghost !px-2.5" aria-label="Open menu" aria-expanded={open}>
          <Menu size={18} />
        </button>
      </header>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-ink-950/60" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 max-w-[85%] animate-pop bg-ink-900" role="dialog" aria-modal="true" aria-label="Menu">
            <button onClick={() => setOpen(false)} className="absolute right-3 top-4 text-ink-200 hover:text-white" aria-label="Close menu">
              <X size={20} />
            </button>
            <SidebarContent onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}

      <main className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <Outlet />
      </main>
    </div>
  );
}
