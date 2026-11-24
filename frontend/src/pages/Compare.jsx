import { useCallback, useEffect, useState } from "react";
import { Hourglass, Terminal } from "lucide-react";
import SuggestionBar from "../components/SuggestionBar";
import { ErrorBanner, PageHeader, Skeleton, StatusDot, formatDate } from "../components/ui";
import { useModels } from "../context/Models";
import { usePredictor } from "../hooks/usePredictor";
import { api } from "../lib/api";
import { MODEL_LABELS as NAMES, MODEL_ORDER } from "../lib/models";

// better: which direction wins. `get` reads the value from a model's result.
const METRICS = [
  { label: "Top-1 accuracy", unit: "%", better: "high", get: (m) => m.top1_accuracy, bar: true, hint: "Right word ranked first" },
  { label: "Top-5 accuracy", unit: "%", better: "high", get: (m) => m.top5_accuracy, bar: true, hint: "Right word among the five suggestions" },
  { label: "Perplexity", unit: "", better: "low", get: (m) => m.perplexity, hint: "Lower means less surprised by the test text" },
  { label: "Latency (median)", unit: " ms", better: "low", get: (m) => m.latency?.median_ms, hint: "One suggestion request, no network" },
  { label: "Latency (95th percentile)", unit: " ms", better: "low", get: (m) => m.latency?.p95_ms, hint: "Slowest 5% of requests" },
  { label: "Model size", unit: " MB", better: "low", get: (m) => m.model_size_mb, hint: "File size on disk" },
];

function winner(metric, models) {
  const vals = models.map((m) => (m?.status === "evaluated" ? metric.get(m) : null));
  const valid = vals.filter((v) => v != null);
  if (valid.length < 2) return -1;
  const ranked = [...valid].sort((a, b) => (metric.better === "high" ? b - a : a - b));
  const [best, runnerUp] = ranked;
  // a lead under 1% over the runner-up is treated as a tie, so no model is labelled "best" for noise
  if (Math.abs(best - runnerUp) <= Math.abs(best) * 0.01) return -1;
  return vals.indexOf(best);
}

function Pending({ report }) {
  return (
    <div className="card p-6">
      <div className="flex items-start gap-4">
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-amber-50 text-amber-600"><Hourglass size={22} aria-hidden /></span>
        <div className="min-w-0">
          <h2 className="text-lg font-bold">Evaluation pending</h2>
          <p className="mt-1 text-sm text-ink-500">{report?.message || "No evaluation has been run yet."} No numbers are shown until the models have been measured on the test split.</p>
          <p className="mt-4 flex items-center gap-2 text-sm font-semibold"><Terminal size={15} aria-hidden /> Run from the project root</p>
          <pre className="mt-2 overflow-x-auto rounded-lg bg-ink-900 p-3 text-sm text-ink-100">python -m ml_models.evaluation.evaluate</pre>
        </div>
      </div>
    </div>
  );
}

function ResultsTable({ report }) {
  const order = MODEL_ORDER;
  const rows = order.map((id) => report.models?.[id]);
  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-left text-sm">
          <thead className="border-b border-ink-100 bg-ink-50/70">
            <tr>
              <th scope="col" className="px-4 py-3 font-semibold text-ink-500">Measure</th>
              {order.map((id, i) => (
                <th key={id} scope="col" className="px-4 py-3 font-display text-base font-bold">
                  {NAMES[id]}
                  {rows[i]?.status !== "evaluated" && <span className="ml-2 inline-flex items-center gap-1 text-xs font-semibold text-amber-600"><StatusDot status="pending" /> pending</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-100">
            {METRICS.map((metric) => {
              const w = winner(metric, rows);
              return (
                <tr key={metric.label}>
                  <th scope="row" className="px-4 py-3 align-top font-semibold">
                    {metric.label}
                    <span className="block text-xs font-normal text-ink-500">{metric.hint}</span>
                  </th>
                  {rows.map((m, i) => {
                    const v = m?.status === "evaluated" ? metric.get(m) : null;
                    return (
                      <td key={order[i]} className="px-4 py-3 align-top">
                        {v == null ? (
                          <span className="text-ink-300">Pending</span>
                        ) : (
                          <div>
                            <span className={`tabular-nums ${w === i ? "font-bold text-brand-700" : ""}`}>
                              {v.toLocaleString(undefined, { maximumFractionDigits: 2 })}{metric.unit}
                            </span>
                            {w === i && <span className="ml-2 rounded bg-brand-50 px-1.5 py-0.5 text-[11px] font-semibold text-brand-700">best</span>}
                            {metric.bar && (
                              <div className="mt-1.5 h-1.5 w-full max-w-[180px] rounded-full bg-ink-100" role="img" aria-label={`${v}%`}>
                                <div className="h-full rounded-full bg-brand-600" style={{ width: `${Math.min(100, v)}%` }} />
                              </div>
                            )}
                          </div>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Variants({ variants }) {
  const entries = Object.entries(variants || {});
  if (!entries.length) return null;
  return (
    <section aria-labelledby="var-h">
      <h2 id="var-h" className="mb-3 text-lg font-bold">N-gram orders</h2>
      <div className="card overflow-x-auto">
        <table className="w-full min-w-[420px] text-left text-sm">
          <thead className="border-b border-ink-100 bg-ink-50/70 text-ink-500">
            <tr><th className="px-4 py-3 font-semibold">Order</th><th className="px-4 py-3 font-semibold">Top-1</th><th className="px-4 py-3 font-semibold">Top-5</th><th className="px-4 py-3 font-semibold">Perplexity</th></tr>
          </thead>
          <tbody className="divide-y divide-ink-100 tabular-nums">
            {entries.map(([name, m]) => (
              <tr key={name}>
                <th scope="row" className="px-4 py-2.5 font-semibold capitalize">{name}</th>
                <td className="px-4 py-2.5">{m.top1_accuracy}%</td><td className="px-4 py-2.5">{m.top5_accuracy}%</td><td className="px-4 py-2.5">{m.perplexity}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SideBySide() {
  const models = useModels();
  const [text, setText] = useState("The history of the");
  const ids = MODEL_ORDER;
  const preds = {
    ensemble: usePredictor(text, "ensemble", { enabled: models.ready.includes("ensemble") }),
    ngram: usePredictor(text, "ngram", { enabled: models.ready.includes("ngram") }),
    lstm: usePredictor(text, "lstm", { enabled: models.ready.includes("lstm") }),
    transformer: usePredictor(text, "transformer", { enabled: models.ready.includes("transformer") }),
  };

  return (
    <section aria-labelledby="live-h">
      <h2 id="live-h" className="mb-1 text-lg font-bold">Try both models on the same text</h2>
      <p className="mb-3 text-sm text-ink-500">Type a few words. Each model ranks its five best next words for the end of your text.</p>
      <div className="card p-4">
        <label htmlFor="cmp-text" className="sr-only">Text to predict from</label>
        <textarea id="cmp-text" className="input min-h-[88px] resize-y text-base" value={text} onChange={(e) => setText(e.target.value.slice(0, 1500))} />
        <div className="mt-4 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
          {ids.map((id) => (
            <div key={id}>
              <h3 className="mb-2 font-display font-bold">{NAMES[id]}</h3>
              {models.ready.includes(id) ? (
                <SuggestionBar result={preds[id].data} loading={preds[id].loading} error={preds[id].error} onPick={() => {}} onRetry={preds[id].retry} showShortcuts={false} />
              ) : (
                <p className="rounded-lg border border-dashed border-ink-200 px-3 py-2.5 text-sm text-ink-500">Not trained yet, so there are no suggestions from this model.</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Compare() {
  const models = useModels();
  const [state, setState] = useState({ loading: true, error: null, report: null });

  const load = useCallback(() => {
    setState((s) => ({ ...s, loading: true, error: null }));
    api.evaluation().then((report) => setState({ loading: false, error: null, report }), (error) => setState({ loading: false, error, report: null }));
  }, []);
  useEffect(() => {
    load();
  }, [load]);

  const { report } = state;
  const available = report?.status === "available";

  return (
    <div className="space-y-8">
      <PageHeader title="Compare models" subtitle="Measured results for both models on the same held-out test split. Nothing here is estimated." />

      {state.error && <ErrorBanner title="Could not load the evaluation" error={state.error} onRetry={load} />}
      {models.error && !state.error && <ErrorBanner title="Server not reachable" error={models.error} onRetry={models.reload} />}

      {state.loading ? (
        <Skeleton className="h-72" />
      ) : report && !available ? (
        <Pending report={report} />
      ) : available ? (
        <>
          <ResultsTable report={report} />
          <Variants variants={report.ngram_variants} />
          <section aria-labelledby="how-h" className="card p-5 text-sm text-ink-500">
            <h2 id="how-h" className="mb-2 text-base font-bold text-ink-900">How this was measured</h2>
            <ul className="list-disc space-y-1 pl-5">
              <li>{report.dataset.name}: {report.dataset.test_tokens.toLocaleString()} test tokens, {report.dataset.test_targets_evaluated.toLocaleString()} next-word targets, {report.dataset.vocab_size.toLocaleString()}-word vocabulary. Both models are scored on exactly the same targets.</li>
              <li>A model is marked “best” on a measure only when it beats the runner-up by more than 1%.</li>
              <li><strong>Accuracy:</strong> {report.protocol.accuracy}.</li>
              <li><strong>Perplexity:</strong> {report.protocol.perplexity}.</li>
              <li><strong>Latency:</strong> {report.protocol.latency}.</li>
              <li><strong>Context:</strong> {report.protocol.context}.</li>
              <li>About {(report.dataset.test_unk_rate * 100).toFixed(1)}% of test tokens are outside the vocabulary (<code>&lt;unk&gt;</code>); they count toward perplexity but are skipped for accuracy.</li>
              <li>Evaluated {formatDate(report.evaluated_at)} on {report.environment?.machine}. Latency depends on that machine.</li>
            </ul>
          </section>
        </>
      ) : null}

      <SideBySide />
    </div>
  );
}
