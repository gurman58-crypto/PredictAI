import { useModels } from "../context/Models";
import { MODEL_LABELS, MODEL_ORDER } from "../lib/models";
import { Spinner } from "./ui";

const LABELS = MODEL_LABELS;

/** Segmented control. Models that are not trained are disabled and say why. */
export default function ModelSelector({ value, onChange, className = "" }) {
  const { models, loading } = useModels();
  if (loading && !models.length) {
    return <span className={`inline-flex items-center gap-2 text-sm text-ink-500 ${className}`}><Spinner /> Loading models</span>;
  }
  const items = models.length ? models : MODEL_ORDER.map((id) => ({ id, status: "unavailable" }));
  return (
    <div role="radiogroup" aria-label="Prediction model" className={`inline-flex rounded-lg border border-ink-200 bg-white p-0.5 ${className}`}>
      {items.map((m) => {
        const ready = m.status === "ready";
        const active = value === m.id;
        return (
          <button
            key={m.id}
            role="radio"
            aria-checked={active}
            disabled={!ready}
            title={ready ? `${m.name} (${m.version})` : m.reason || "Not available"}
            onClick={() => onChange(m.id)}
            className={`rounded-md px-3 py-1.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed ${
              active ? "bg-ink-900 text-white" : ready ? "text-ink-700 hover:bg-ink-50" : "text-ink-300"
            }`}
          >
            {LABELS[m.id] || m.id}
            {!ready && <span className="ml-1.5 text-[11px] font-medium">not trained</span>}
          </button>
        );
      })}
    </div>
  );
}
