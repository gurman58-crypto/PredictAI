import { useRef, useState } from "react";
import SuggestionBar from "./SuggestionBar";
import { useModels } from "../context/Models";
import { usePredictor } from "../hooks/usePredictor";
import { modelLabel } from "../lib/models";
import { applySuggestion } from "../lib/text";

/** Live demo on the landing page: it calls the real /predict endpoint (no canned answers). */
export default function HeroDemo() {
  const models = useModels();
  const ref = useRef(null);
  const [text, setText] = useState("Artificial intelligence is ");
  const [cursor, setCursor] = useState(text.length);
  const ready = models.ready.length > 0;
  const predictor = usePredictor(text.slice(0, cursor), models.model, { enabled: ready });

  const sync = () => ref.current && setCursor(ref.current.selectionStart);

  const pick = (p) => {
    const ta = ref.current;
    const pos = ta ? ta.selectionStart : text.length;
    const r = applySuggestion(text, pos, p.word, { mode: predictor.data?.mode, prefix: predictor.data?.prefix });
    setText(r.text);
    setCursor(r.cursor);
    requestAnimationFrame(() => {
      ta?.focus();
      ta?.setSelectionRange(r.cursor, r.cursor);
    });
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-ink-800/70 p-4 shadow-pop backdrop-blur sm:p-5">
      <div className="mb-3 flex items-center justify-between text-xs text-ink-300">
        <span className="font-semibold text-white">Try it: type or pick a word</span>
        {ready && models.ready.length > 1 ? (
          <div className="flex rounded-md border border-white/15 p-0.5" role="radiogroup" aria-label="Model">
            {models.ready.map((id) => (
              <button
                key={id}
                role="radio"
                aria-checked={models.model === id}
                onClick={() => models.setModel(id)}
                className={`rounded px-2.5 py-1 font-semibold ${models.model === id ? "bg-white text-ink-900" : "text-ink-200 hover:text-white"}`}
              >
                {modelLabel(id)}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <label htmlFor="hero-text" className="sr-only">Demo text</label>
      <textarea
        id="hero-text"
        ref={ref}
        value={text}
        onChange={(e) => {
          setText(e.target.value.slice(0, 600));
          sync();
        }}
        onSelect={sync}
        onKeyUp={sync}
        onClick={sync}
        rows={4}
        className="w-full resize-none rounded-lg border border-white/10 bg-ink-950/60 p-3 text-base leading-7 text-white placeholder:text-ink-300 focus:border-brand-400 focus:outline-none"
      />

      <div className="mt-3">
        {models.loading && !models.models.length ? (
          <p className="text-sm text-ink-300">Connecting to the model server…</p>
        ) : models.error ? (
          <div className="rounded-lg border border-red-400/40 bg-red-500/10 p-3 text-sm text-red-100" role="status">
            <p className="font-semibold">The model server is not running.</p>
            <p className="mt-1 text-red-200">
              Start it with <code className="rounded bg-black/30 px-1.5 py-0.5">uvicorn app.main:app --port 8000</code> in the backend folder, then{" "}
              <button onClick={models.reload} className="font-semibold underline">retry</button>.
            </p>
          </div>
        ) : !ready ? (
          <p className="rounded-lg border border-amber-300/40 bg-amber-400/10 p-3 text-sm text-amber-100">
            No model is trained yet. Run the training scripts from the README, then reload this page.
          </p>
        ) : (
          <SuggestionBar dark result={predictor.data} loading={predictor.loading} error={predictor.error} onPick={pick} onRetry={predictor.retry} showShortcuts={false} />
        )}
      </div>
    </div>
  );
}
