import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { checkWindow, mapIssues, visibleIssues } from "../lib/writing";

const DEBOUNCE_MS = 700;
const EMPTY = { text: "", grammar: [], style: [], stats: null, windowed: false };

/**
 * Checks the document in the background shortly after the user stops typing (one request returns grammar +
 * style) and keeps the results aligned with the text while the user keeps typing.
 */
export function useLiveChecks(text, cursor, { enabled, tone, ignoredWords, ignoredOnce }) {
  const [res, setRes] = useState(EMPTY);
  const [status, setStatus] = useState({ checking: false, error: null, checked: false });
  const abort = useRef(null);
  const latest = useRef({ text, cursor, tone, ignoredWords });
  latest.current = { text, cursor, tone, ignoredWords };

  const run = useCallback(async () => {
    const { text: t, cursor: c, tone: tn, ignoredWords: iw } = latest.current;
    abort.current?.abort();
    if (!t.trim()) {
      setRes(EMPTY);
      setStatus({ checking: false, error: null, checked: false });
      return;
    }
    const controller = new AbortController();
    abort.current = controller;
    const win = checkWindow(t, c);
    setStatus((s) => ({ ...s, checking: true }));
    try {
      const data = await api.check({ text: win.text, tone: tn, ignore: [...iw].slice(0, 300) }, controller.signal);
      const shift = (list) => list.map((i) => ({ ...i, start: i.start + win.offset, end: i.end + win.offset }));
      setRes({ text: t, grammar: shift(data.grammar), style: shift(data.style), stats: data.stats, windowed: win.text.length < t.length });
      setStatus({ checking: false, error: null, checked: true });
    } catch (e) {
      if (!e.canceled) setStatus((s) => ({ ...s, checking: false, error: e }));
    }
  }, []);

  // very long documents: re-check when the caret moves to another part of the text
  const region = text.length > 5000 ? Math.floor(cursor / 1500) : 0;

  useEffect(() => {
    if (!enabled) return undefined;
    const timer = setTimeout(run, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [text, region, enabled, tone, run, ignoredWords]);

  useEffect(() => () => abort.current?.abort(), []);

  const grammar = useMemo(() => {
    const mapped = mapIssues(res.grammar, res.text, text);
    return visibleIssues(mapped, text, cursor, ignoredWords).filter((i) => !ignoredOnce.has(`${i.rule}|${text.slice(i.start, i.end)}`));
  }, [res, text, cursor, ignoredWords, ignoredOnce]);

  const style = useMemo(() => {
    const mapped = mapIssues(res.style, res.text, text);
    return visibleIssues(mapped, text, cursor, ignoredWords).filter((i) => !ignoredOnce.has(`${i.rule}|${text.slice(i.start, i.end)}`));
  }, [res, text, cursor, ignoredWords, ignoredOnce]);

  return { grammar, style, stats: res.stats, windowed: res.windowed, ...status, refresh: run };
}
