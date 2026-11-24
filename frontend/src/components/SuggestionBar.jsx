import { RefreshCw } from "lucide-react";
import { modelLabel } from "../lib/models";
import { Spinner } from "./ui";

/**
 * Ranked suggestion chips. The bar behind each chip is the model probability relative to the top
 * suggestion, so users can see how confident the model is in each option.
 */
export default function SuggestionBar({ result, loading, error, onPick, onRetry, showShortcuts = true, dark = false }) {
  const preds = result?.predictions || [];
  const top = preds[0]?.score || 1;
  const chip = dark
    ? "border-white/15 bg-white/5 text-white hover:border-brand-400 hover:bg-white/10"
    : "border-ink-200 bg-white text-ink-900 hover:border-brand-500 hover:bg-brand-50";
  const muted = dark ? "text-ink-300" : "text-ink-500";

  return (
    <div>
      <div className={`mb-2 flex min-h-[20px] items-center gap-2 text-xs ${muted}`} aria-live="polite">
        {result?.mode === "completion" ? (
          <span>Completing “{result.prefix}”</span>
        ) : (
          <span>Next word</span>
        )}
        {loading && <Spinner size={12} />}
        {result && !error && <span className="ml-auto tabular-nums">{result.latency_ms.toFixed(1)} ms · {modelLabel(result.model)}</span>}
      </div>

      {error ? (
        <div className={`flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2.5 text-sm ${dark ? "border-red-400/40 bg-red-500/10 text-red-100" : "border-red-200 bg-red-50 text-red-900"}`} role="status">
          <span className="min-w-0 flex-1">Suggestions paused: {error.message} Your text is safe.</span>
          {onRetry && (
            <button onClick={onRetry} className={`btn !py-1 ${dark ? "btn-light" : "btn-ghost"}`}>
              <RefreshCw size={13} aria-hidden /> Retry
            </button>
          )}
        </div>
      ) : preds.length === 0 ? (
        <p className={`rounded-lg border border-dashed px-3 py-2.5 text-sm ${dark ? "border-white/20" : "border-ink-200"} ${muted}`}>
          {loading ? "Thinking…" : result?.mode === "completion" ? "No known word starts with that. Keep typing." : "Start typing to see suggestions."}
        </p>
      ) : (
        <ul className="flex gap-2 overflow-x-auto pb-1 sm:flex-wrap sm:overflow-visible" aria-label="Suggestions">
          {preds.map((p, i) => (
            <li key={p.word} className="shrink-0">
              <button
                onClick={() => onPick(p, i)}
                className={`relative flex items-center gap-2 overflow-hidden rounded-lg border px-3 py-1.5 text-sm font-semibold transition-colors ${chip}`}
                title={`${(p.score * 100).toFixed(1)}% probability${showShortcuts ? ` · Alt+${i + 1}` : ""}`}
              >
                <span
                  className={`absolute inset-y-0 left-0 ${dark ? "bg-brand-500/25" : "bg-brand-100"}`}
                  style={{ width: `${Math.max(6, (p.score / top) * 100)}%` }}
                  aria-hidden
                />
                <span className={`relative text-xs font-medium tabular-nums ${muted}`}>{i + 1}</span>
                <span className="relative">{p.word}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
