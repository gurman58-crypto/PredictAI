import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { FilePlus2, Save, Sparkles } from "lucide-react";
import ModelSelector from "../components/ModelSelector";
import SuggestionBar from "../components/SuggestionBar";
import LiveTextArea from "../components/LiveTextArea";
import WritingTools from "../components/WritingTools";
import { ErrorBanner, Skeleton, Spinner, timeAgo } from "../components/ui";
import { useModels } from "../context/Models";
import { useToast } from "../context/Toast";
import { useLiveChecks } from "../hooks/useLiveChecks";
import { usePredictor } from "../hooks/usePredictor";
import { api, toApiError } from "../lib/api";
import { applyContinuation, applySuggestion, contextBeforeCursor, countWords } from "../lib/text";
import { applyReplacement, isUnderlined, sentenceAround } from "../lib/writing";

const MAX_CHARS = 200_000;

function readSetting(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v == null ? fallback : JSON.parse(v);
  } catch {
    return fallback;
  }
}
function writeSetting(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: keep the setting in memory only */
  }
}

export default function Editor() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const models = useModels();
  const textareaRef = useRef(null);
  const loadedId = useRef(null);

  const [title, setTitle] = useState("Untitled document");
  const [text, setText] = useState("");
  const [saved, setSaved] = useState({ title: "Untitled document", text: "", at: null });
  const [docId, setDocId] = useState(id || null);
  const [loadState, setLoadState] = useState({ loading: !!id, error: null });
  const [saving, setSaving] = useState(false);
  const [completing, setCompleting] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [collapsed, setCollapsed] = useState(true);
  const [touched, setTouched] = useState(false);
  const completeAbort = useRef(null);

  // ---------------------------------------------------------------- live grammar / style checking
  const [liveOn, setLiveOnRaw] = useState(() => readSetting("predictai.live", true));
  const [showStyle, setShowStyleRaw] = useState(() => readSetting("predictai.showStyle", false));
  const [tone, setToneRaw] = useState(() => readSetting("predictai.tone", "neutral"));
  const [ignoredWords, setIgnoredWords] = useState(() => new Set(readSetting("predictai.dictionary", [])));
  const [ignoredOnce, setIgnoredOnce] = useState(() => new Set());
  const setLiveOn = (v) => { setLiveOnRaw(v); writeSetting("predictai.live", v); };
  const setShowStyle = (v) => { setShowStyleRaw(v); writeSetting("predictai.showStyle", v); };
  const setTone = (v) => { setToneRaw(v); writeSetting("predictai.tone", v); };

  const dirty = title !== saved.title || text !== saved.text;
  const ready = models.ready.length > 0;

  // ---------------------------------------------------------------- load / reset by route
  useEffect(() => {
    if (!id) {
      loadedId.current = null;
      setDocId(null);
      setTitle("Untitled document");
      setText("");
      setSaved({ title: "Untitled document", text: "", at: null });
      setLoadState({ loading: false, error: null });
      return undefined;
    }
    if (loadedId.current === id) return undefined; // we just created this document ourselves
    let cancelled = false;
    setLoadState({ loading: true, error: null });
    api
      .getDocument(id)
      .then((doc) => {
        if (cancelled) return;
        loadedId.current = id;
        setDocId(doc.id);
        setTitle(doc.title);
        setText(doc.content);
        setSaved({ title: doc.title, text: doc.content, at: doc.updated_at });
        setLoadState({ loading: false, error: null });
      })
      .catch((e) => !cancelled && setLoadState({ loading: false, error: e }));
    return () => {
      cancelled = true;
    };
  }, [id]);

  // warn before closing the tab with unsaved changes
  useEffect(() => {
    if (!dirty) return undefined;
    const handler = (e) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);

  // ---------------------------------------------------------------- predictions
  const context = useMemo(() => contextBeforeCursor(text, cursor), [text, cursor]);
  const predictor = usePredictor(context, models.model, {
    enabled: ready && touched && collapsed && !loadState.loading,
  });
  const result = predictor.data;

  const syncCursor = useCallback(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    setCursor(ta.selectionStart);
    setCollapsed(ta.selectionStart === ta.selectionEnd);
  }, []);

  const live = useLiveChecks(text, cursor, { enabled: liveOn && !loadState.loading, tone, ignoredWords, ignoredOnce });
  const underlined = useMemo(
    () => [
      ...live.grammar.filter(isUnderlined).map((i) => ({ ...i, kind: "grammar" })),
      ...(showStyle ? live.style.map((i) => ({ ...i, kind: "style" })) : []),
    ],
    [live.grammar, live.style, showStyle]
  );

  /** Apply an edit through the browser so Ctrl+Z still works; fall back to plain state. */
  const edit = useCallback((change) => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.focus();
    ta.setSelectionRange(change.start, change.end);
    let ok = false;
    try {
      ok = document.execCommand("insertText", false, change.insert);
    } catch {
      ok = false;
    }
    if (!ok || ta.value !== change.text) {
      setText(change.text);
      requestAnimationFrame(() => {
        ta.setSelectionRange(change.cursor, change.cursor);
        syncCursor();
      });
    } else {
      syncCursor();
    }
  }, [syncCursor]);

  /** What the paraphraser works on: the selection, otherwise the sentence around the cursor. */
  const getTarget = useCallback(() => {
    const ta = textareaRef.current;
    const t = ta ? ta.value : text;
    const s = ta ? ta.selectionStart : cursor;
    const e = ta ? ta.selectionEnd : cursor;
    if (e > s) return { start: s, end: e, text: t.slice(s, e) };
    return sentenceAround(t, s);
  }, [text, cursor]);

  const showRange = useCallback((start, end) => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.focus();
    ta.setSelectionRange(start, end);
    syncCursor();
  }, [syncCursor]);

  const applyIssueFix = useCallback((issue, replacement) => {
    const change = applyReplacement(text, issue, replacement);
    if (change) edit(change);
  }, [text, edit]);
  const ignoreIssue = useCallback((issue) => {
    setIgnoredOnce((s) => new Set(s).add(`${issue.rule}|${text.slice(issue.start, issue.end)}`));
  }, [text]);
  const addWord = useCallback((issue) => {
    const w = text.slice(issue.start, issue.end).toLowerCase();
    setIgnoredWords((s) => {
      const next = new Set(s).add(w);
      writeSetting("predictai.dictionary", [...next].slice(-500));
      return next;
    });
    toast.success(`“${w}” added to your dictionary.`);
  }, [text, toast]);

  const pick = useCallback(
    (prediction) => {
      if (!result) return;
      const ta = textareaRef.current;
      const pos = ta ? ta.selectionStart : cursor;
      edit(applySuggestion(text, pos, prediction.word, { mode: result.mode, prefix: result.prefix }));
      api.accepted(result.model);
    },
    [result, text, cursor, edit]
  );

  const continueSentence = useCallback(async () => {
    if (completing || !ready) return;
    const ta = textareaRef.current;
    const pos = ta ? ta.selectionStart : cursor;
    completeAbort.current?.abort();
    const controller = new AbortController();
    completeAbort.current = controller;
    setCompleting(true);
    try {
      const res = await api.complete(
        { text: contextBeforeCursor(text, pos), model: models.model, max_tokens: 20 },
        controller.signal
      );
      if (!res.continuation) {
        toast.info("The model had nothing to add here. Try typing a few more words.");
      } else {
        edit(applyContinuation(text, pos, res.continuation));
      }
    } catch (e) {
      if (!e.canceled) toast.error(`Could not continue the sentence. ${e.message} Your text was not changed.`);
    } finally {
      setCompleting(false);
    }
  }, [completing, ready, cursor, text, models.model, edit, toast]);

  // ---------------------------------------------------------------- saving
  const save = useCallback(async () => {
    if (saving) return;
    const cleanTitle = title.trim() || "Untitled document";
    setSaving(true);
    try {
      let doc;
      if (docId) {
        doc = await api.updateDocument(docId, { title: cleanTitle, content: text });
      } else {
        doc = await api.createDocument({ title: cleanTitle, content: text });
        loadedId.current = doc.id;
        setDocId(doc.id);
        navigate(`/editor/${doc.id}`, { replace: true });
      }
      setTitle(doc.title);
      setSaved({ title: doc.title, text, at: doc.updated_at });
      toast.success("Document saved.");
    } catch (e) {
      toast.error(`Could not save. ${e.message} Your text is still here.`);
    } finally {
      setSaving(false);
    }
  }, [saving, title, text, docId, navigate, toast]);

  // ---------------------------------------------------------------- keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        save();
      } else if (mod && e.key === "Enter") {
        e.preventDefault();
        continueSentence();
      } else if (e.altKey && /^Digit[1-5]$/.test(e.code) && result?.predictions?.length) {
        const p = result.predictions[Number(e.code.slice(5)) - 1];
        if (p) {
          e.preventDefault();
          pick(p);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [save, continueSentence, pick, result]);

  const newDocument = () => {
    if (dirty && !window.confirm("Discard unsaved changes and start a new document?")) return;
    navigate("/editor");
  };

  // ---------------------------------------------------------------- render
  if (loadState.loading) {
    return (
      <div className="space-y-4" aria-busy="true" aria-label="Loading document">
        <Skeleton className="h-10 w-72" />
        <Skeleton className="h-[420px] w-full" />
      </div>
    );
  }
  if (loadState.error) {
    return (
      <div className="space-y-4">
        <ErrorBanner
          title={loadState.error.status === 404 ? "Document not found" : "Could not open the document"}
          error={loadState.error}
          onRetry={loadState.error.status === 404 ? undefined : () => window.location.reload()}
        />
        <Link to="/documents" className="btn btn-ghost">Back to my documents</Link>
      </div>
    );
  }

  const chars = text.length;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          maxLength={200}
          aria-label="Document title"
          className="min-w-0 flex-1 rounded-lg border border-transparent bg-transparent px-2 py-1 font-display text-2xl font-bold hover:border-ink-200 focus:border-brand-500 focus:bg-white focus:outline-none"
        />
        <div className="flex items-center gap-2">
          <button onClick={newDocument} className="btn btn-ghost"><FilePlus2 size={16} aria-hidden /> New</button>
          <button onClick={save} disabled={saving || (!dirty && !!docId)} className="btn btn-primary">
            {saving ? <Spinner /> : <Save size={16} aria-hidden />} Save
          </button>
        </div>
      </div>

      {models.error && <ErrorBanner title="Server not reachable" error={models.error} onRetry={models.reload} />}
      {!models.error && !models.loading && !ready && (
        <ErrorBanner
          title="No trained model is available"
          error={{ message: "Train a model first (see the README), then reload. You can still write and save documents." }}
          onRetry={models.reload}
        />
      )}

      <div className="card overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-100 px-4 py-2.5">
          <div className="flex items-center gap-3">
            <ModelSelector value={models.model} onChange={models.setModel} />
          </div>
          <button onClick={continueSentence} disabled={!ready || completing} className="btn btn-dark" title="Ctrl+Enter">
            {completing ? <Spinner /> : <Sparkles size={16} aria-hidden />} Continue sentence
          </button>
        </div>

        <label htmlFor="editor-text" className="sr-only">Document text</label>
        <LiveTextArea
          ref={textareaRef}
          value={text}
          issues={underlined}
          onApply={applyIssueFix}
          onIgnore={ignoreIssue}
          onAddWord={addWord}
          textareaProps={{
            id: "editor-text",
            onChange: (e) => {
              setText(e.target.value.slice(0, MAX_CHARS));
              syncCursor();
            },
            onFocus: () => {
              setTouched(true);
              syncCursor();
            },
            onSelect: syncCursor,
            onKeyUp: syncCursor,
            onClick: syncCursor,
            placeholder: "Start writing. Suggestions appear as you type.",
            spellCheck: !liveOn,
          }}
        />

        <div className="border-t border-ink-100 bg-ink-50/60 px-4 py-3">
          {ready ? (
            collapsed ? (
              <SuggestionBar result={result} loading={predictor.loading} error={predictor.error} onPick={pick} onRetry={predictor.retry} />
            ) : (
              <p className="text-sm text-ink-500">Suggestions pause while text is selected.</p>
            )
          ) : (
            <p className="text-sm text-ink-500">Suggestions are unavailable until a model is trained.</p>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-1 border-t border-ink-100 px-4 py-2 text-xs text-ink-500">
          <span className="tabular-nums">{countWords(text).toLocaleString()} words · {chars.toLocaleString()} characters</span>
          <span aria-live="polite">
            {saving ? "Saving…" : dirty ? "Unsaved changes" : saved.at ? `Saved ${timeAgo(saved.at)}` : "Not saved yet"}
          </span>
        </div>
      </div>

      <WritingTools
        text={text}
        live={live}
        settings={{ liveOn, setLiveOn, showStyle, setShowStyle, tone, setTone }}
        getTarget={getTarget}
        onEdit={edit}
        onShow={showRange}
        modelReady={models.ready}
      />

      <p className="text-xs text-ink-500">
        <span className="kbd">Alt</span> + <span className="kbd">1</span>–<span className="kbd">5</span> insert a suggestion ·{" "}
        <span className="kbd">Ctrl</span> + <span className="kbd">Enter</span> continue the sentence ·{" "}
        <span className="kbd">Ctrl</span> + <span className="kbd">S</span> save · <span className="kbd">Ctrl</span> + <span className="kbd">Z</span> undo an insertion
      </p>
    </div>
  );
}
