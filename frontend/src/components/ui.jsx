import { useEffect, useRef } from "react";
import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";

export function Logo({ dark = false, size = 28 }) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
        <rect width="32" height="32" rx="8" fill={dark ? "#ffffff" : "#0a1530"} />
        <path d="M9 22V10h6.2a3.8 3.8 0 0 1 0 7.6H9" fill="none" stroke={dark ? "#0a1530" : "#fff"} strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
        <rect x="19" y="14" width="5" height="3.6" rx="1.8" fill="#4f77ff" />
      </svg>
      <span className={`font-display text-lg font-bold tracking-tight ${dark ? "text-white" : "text-ink-900"}`}>PredictAI</span>
    </span>
  );
}

export function Spinner({ size = 16, className = "" }) {
  return <Loader2 size={size} className={`animate-spin ${className}`} aria-hidden />;
}

export function PageHeader({ title, subtitle, children }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-bold text-ink-900 sm:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1 max-w-2xl text-sm text-ink-500">{subtitle}</p>}
      </div>
      {children && <div className="flex flex-wrap gap-2">{children}</div>}
    </div>
  );
}

export function ErrorBanner({ error, onRetry, title = "Something went wrong" }) {
  if (!error) return null;
  return (
    <div role="alert" className="flex flex-wrap items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">
      <AlertTriangle size={18} className="mt-0.5 shrink-0 text-red-600" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="font-semibold">{title}</p>
        <p className="mt-0.5 break-words text-red-800">{error.message || String(error)}</p>
      </div>
      {onRetry && (
        <button onClick={onRetry} className="btn btn-ghost !py-1.5">
          <RefreshCw size={14} aria-hidden /> Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ icon: Icon, title, children, action }) {
  return (
    <div className="card flex flex-col items-center px-6 py-14 text-center">
      {Icon && (
        <span className="mb-4 grid h-12 w-12 place-items-center rounded-xl bg-brand-50 text-brand-600">
          <Icon size={22} aria-hidden />
        </span>
      )}
      <h2 className="text-lg font-bold">{title}</h2>
      {children && <p className="mt-1 max-w-md text-sm text-ink-500">{children}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function Skeleton({ className = "" }) {
  return <div className={`animate-pulse rounded-md bg-ink-100 ${className}`} aria-hidden />;
}

export function StatusDot({ status }) {
  const tone = status === "ready" ? "bg-emerald-500" : status === "pending" ? "bg-amber-500" : "bg-ink-300";
  return <span className={`inline-block h-2 w-2 rounded-full ${tone}`} aria-hidden />;
}

export function ConfirmDialog({ open, title, children, confirmLabel = "Delete", busy, onConfirm, onCancel }) {
  const cancelRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    cancelRef.current?.focus();
    const onKey = (e) => e.key === "Escape" && !busy && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, busy, onCancel]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-ink-950/60 p-4" onMouseDown={(e) => e.target === e.currentTarget && !busy && onCancel()}>
      <div role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" className="w-full max-w-sm animate-pop rounded-xl bg-white p-6 shadow-pop">
        <h2 id="confirm-title" className="text-lg font-bold">{title}</h2>
        <div className="mt-2 text-sm text-ink-500">{children}</div>
        <div className="mt-6 flex justify-end gap-2">
          <button ref={cancelRef} className="btn btn-ghost" onClick={onCancel} disabled={busy}>Cancel</button>
          <button className="btn btn-danger" onClick={onConfirm} disabled={busy}>
            {busy && <Spinner />} {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export function formatDate(iso) {
  try {
    return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(iso));
  } catch {
    return iso;
  }
}

export function timeAgo(iso) {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 30) return `${Math.floor(s / 86400)} d ago`;
  return formatDate(iso);
}
