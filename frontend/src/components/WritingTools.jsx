import { useCallback, useEffect, useRef, useState } from "react";
import { CheckCheck, Eye, Loader2, RefreshCw, Wand2 } from "lucide-react";
import { Spinner } from "./ui";
import { api } from "../lib/api";
import { applyAll, applyIssue, SEVERITY_LABEL } from "../lib/writing";

const TABS = [
  { id: "grammar", label: "Grammar" },
  { id: "style", label: "Style" },
  { id: "paraphrase", label: "Paraphrase" },
];
const DOT = { error: "bg-red-500", warning: "bg-amber-500", info: "bg-brand-500" };
const MAX_PARAPHRASE = 1000;

function IssueList({ issues, onApply, onShow, empty = "Nothing to fix here." }) {
  if (issues.length === 0) return <p className="rounded-lg border border-dashed border-ink-200 px-3 py-3 text-sm text-ink-500">{empty}</p>;
  return (
    <ul className="divide-y divide-ink-100 rounded-lg border border-ink-100 bg-white">
      {issues.map((i) => (
        <li key={`${i.rule}-${i.start}-${i.end}`} className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2.5">
          <span className={`h-2 w-2 shrink-0 rounded-full ${DOT[i.severity] || DOT.info}`} title={SEVERITY_LABEL[i.severity]} aria-label={SEVERITY_LABEL[i.severity]} />
          <div className="min-w-0 flex-1 text-sm">
            <p className="text-ink-900">{i.message}</p>
            {i.original && (
              <p className="mt-0.5 truncate text-xs text-ink-500">
                “{i.original.length > 60 ? `${i.original.slice(0, 60)}…` : i.original}”
                {i.replacement != null && i.replacement !== "" && <> → <strong className="text-ink-700">“{i.replacement}”</strong></>}
                {i.replacement === "" && <> → <strong className="text-ink-700">remove</strong></>}
              </p>
            )}
          </div>
          <div className="flex shrink-0 gap-1.5">
            <button onClick={() => onShow(i)} className="btn btn-ghost !px-2 !py-1 text-xs" title="Select this text in the editor">
              <Eye size={13} aria-hidden /> Show
            </button>
            {i.replacement != null && (
              <button onClick={() => onApply(i)} className="btn btn-dark !px-2.5 !py-1 text-xs">Apply</button>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}

/**
 * Grammar, style and paraphrase tools. The editor owns the text; this panel asks it for the current text and
 * selection, and hands edits back through `onEdit` so that Ctrl+Z keeps working.
 */
export default function WritingTools({ text, live, settings, getTarget, onEdit, onShow, modelReady }) {
  const [tab, setTab] = useState("grammar");
  const [para, setPara] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const abort = useRef(null);

  useEffect(() => () => abort.current?.abort(), []);

  const doParaphrase = async () => {
    const target = getTarget();
    if (!target.text.trim()) return setError({ message: "Write or select a sentence first." });
    if (target.text.length > MAX_PARAPHRASE) return setError({ message: `Select a shorter passage (up to ${MAX_PARAPHRASE} characters).` });
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setBusy(true);
    setError(null);
    try {
      const data = await api.paraphrase({ text: target.text, variants: 3, model: modelReady?.includes("transformer") ? "transformer" : undefined }, controller.signal);
      setPara({ target, ...data });
    } catch (e) {
      if (!e.canceled) setError(e);
    } finally {
      setBusy(false);
    }
  };

  const usePara = (variant) => {
    const { target } = para;
    if (text.slice(target.start, target.end) !== target.text) return setError({ message: "The text changed since this suggestion was made. Run Paraphrase again." });
    onEdit({ text: text.slice(0, target.start) + variant.text + text.slice(target.end), cursor: target.start + variant.text.length, start: target.start, end: target.end, insert: variant.text });
    setPara(null);
  };

  const apply = (issue) => {
    const change = applyIssue(text, issue);
    if (change) onEdit(change);
  };
  const applyEvery = (issues) => {
    const { change, count } = applyAll(text, issues);
    if (count) onEdit(change);
  };

  const fixable = (list) => list.filter((i) => i.replacement != null).length;
  const status = live.error
    ? "Checking paused (server not reachable)."
    : !settings.liveOn && !live.checked
      ? "Live checking is off."
      : live.checking
        ? "Checking…"
        : live.checked
          ? "Up to date."
          : text.trim() ? "Starts after you pause typing." : "Start writing to see results.";

  return (
    <section className="card overflow-hidden" aria-labelledby="tools-h">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-100 px-4 py-2.5">
        <h2 id="tools-h" className="font-display text-base font-bold">Writing tools</h2>
        <div role="tablist" aria-label="Writing tools" className="inline-flex rounded-lg border border-ink-200 bg-white p-0.5">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              onClick={() => { setTab(t.id); setError(null); }}
              className={`rounded-md px-3 py-1.5 text-sm font-semibold ${tab === t.id ? "bg-ink-900 text-white" : "text-ink-700 hover:bg-ink-50"}`}
            >
              {t.label}
              {t.id === "grammar" && live.grammar.length > 0 && <span className="ml-1.5 rounded-full bg-red-100 px-1.5 text-xs text-red-800">{live.grammar.length}</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3 px-4 py-3" role="tabpanel">
        {error && <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-900" role="status">{error.message}</p>}

        {tab !== "paraphrase" && (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-ink-700">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={settings.liveOn} onChange={(e) => settings.setLiveOn(e.target.checked)} className="h-4 w-4 rounded border-ink-300 accent-brand-600" />
              Underline mistakes as I type
            </label>
            {tab === "style" && (
              <>
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={settings.showStyle} onChange={(e) => settings.setShowStyle(e.target.checked)} className="h-4 w-4 rounded border-ink-300 accent-brand-600" />
                  Also underline style suggestions
                </label>
                <label className="flex items-center gap-2">
                  Tone
                  <select value={settings.tone} onChange={(e) => settings.setTone(e.target.value)} className="rounded-lg border border-ink-200 bg-white px-2 py-1 text-sm">
                    <option value="neutral">Neutral</option>
                    <option value="formal">Formal</option>
                  </select>
                </label>
              </>
            )}
            <span className="inline-flex items-center gap-2 text-xs text-ink-500" aria-live="polite">
              {live.checking && <Spinner size={12} />}{status}
            </span>
            <button onClick={live.refresh} disabled={!text.trim() || live.checking} className="btn btn-ghost !px-2.5 !py-1 text-xs"><RefreshCw size={13} aria-hidden /> Check now</button>
          </div>
        )}

        {tab === "grammar" && (
          <>
            <p className="text-sm text-ink-500">
              Mistakes get a red wavy underline while you type. Click an underlined word (or press <span className="kbd">Alt</span> + <span className="kbd">Enter</span> on it) to pick the fix.
              It finds spelling errors and common grammar and punctuation mistakes, not everything.
            </p>
            {fixable(live.grammar) > 1 && (
              <button onClick={() => applyEvery(live.grammar)} className="btn btn-ghost"><CheckCheck size={15} aria-hidden /> Fix all ({fixable(live.grammar)})</button>
            )}
            <IssueList issues={live.grammar} onApply={apply} onShow={(i) => onShow(i.start, i.end)} empty={live.checked ? "No mistakes found." : "Nothing checked yet."} />
          </>
        )}

        {tab === "style" && (
          <>
            <p className="text-sm text-ink-500">Flags long sentences, passive voice, filler words and wordy phrases, and measures how easy the text is to read. These are suggestions, not errors.</p>
            {live.stats && live.stats.words > 0 && (
              <>
                <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {[
                    ["Reading level", live.stats.reading_level],
                    ["Ease score", live.stats.reading_ease != null ? `${live.stats.reading_ease} / 100` : "–"],
                    ["Words per sentence", live.stats.avg_sentence_words],
                    ["Reading time", live.stats.reading_time_seconds < 60 ? `${live.stats.reading_time_seconds} sec` : `${Math.round(live.stats.reading_time_seconds / 60)} min`],
                  ].map(([k, v]) => (
                    <div key={k} className="rounded-lg border border-ink-100 bg-ink-50/60 px-3 py-2">
                      <dt className="text-xs text-ink-500">{k}</dt>
                      <dd className="font-display text-lg font-bold capitalize tabular-nums">{v}</dd>
                    </div>
                  ))}
                </dl>
                {live.windowed && <p className="text-xs text-ink-500">Long document: results cover the part around your cursor.</p>}
              </>
            )}
            {fixable(live.style) > 1 && (
              <button onClick={() => applyEvery(live.style)} className="btn btn-ghost"><CheckCheck size={15} aria-hidden /> Apply all ({fixable(live.style)})</button>
            )}
            <IssueList issues={live.style} onApply={apply} onShow={(i) => onShow(i.start, i.end)} empty={live.checked ? "No style suggestions." : "Nothing checked yet."} />
          </>
        )}

        {tab === "paraphrase" && (
          <>
            <p className="text-sm text-ink-500">
              Select a sentence (or put the cursor in one) and get alternative wordings. It swaps words and phrases and keeps only changes the
              language model finds natural. It does not restructure sentences.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <button onClick={doParaphrase} disabled={!text.trim() || busy} className="btn btn-primary">
                {busy ? <Loader2 size={15} className="animate-spin" aria-hidden /> : <Wand2 size={15} aria-hidden />} Paraphrase
              </button>
              {para?.target && <span className="max-w-full truncate text-xs text-ink-500">Rewriting: “{para.target.text.length > 80 ? `${para.target.text.slice(0, 80)}…` : para.target.text}”</span>}
            </div>
            {para && para.variants.length === 0 && <p className="rounded-lg border border-dashed border-ink-200 px-3 py-3 text-sm text-ink-500">{para.message || "No safe rewrites were found."}</p>}
            {para && para.variants.length > 0 && (
              <ol className="space-y-2">
                {para.variants.map((v, idx) => (
                  <li key={v.text} className="rounded-lg border border-ink-100 bg-white p-3">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <p className="min-w-0 flex-1 text-sm leading-6 text-ink-900">{v.text}</p>
                      <button onClick={() => usePara(v)} className="btn btn-dark !px-2.5 !py-1 text-xs">Use this</button>
                    </div>
                    <p className="mt-1.5 text-xs text-ink-500">
                      Option {idx + 1} · {v.changes.length} {v.changes.length === 1 ? "change" : "changes"}:{" "}
                      {v.changes.slice(0, 6).map((c) => `${c.from} → ${c.to}`).join(", ")}
                    </p>
                  </li>
                ))}
              </ol>
            )}
            {para && <p className="text-xs text-ink-500">{para.scored ? `Fluency checked with the ${para.model} model.` : "No trained model was available, so the wording was not checked for fluency."}</p>}
          </>
        )}
      </div>
    </section>
  );
}
