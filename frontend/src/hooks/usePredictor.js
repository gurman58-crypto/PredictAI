import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";

/**
 * Debounced next-word predictions for `text` (the text BEFORE the cursor).
 * - an in-flight request is aborted when newer input arrives, so stale answers never overwrite fresh ones;
 * - failures only set `error`: the caller's text is never touched.
 */
export function usePredictor(text, model, { enabled = true, delay = 250, topK = 5 } = {}) {
  const [state, setState] = useState({ data: null, loading: false, error: null });
  const [retryTick, setRetryTick] = useState(0);
  const lastKey = useRef("");

  useEffect(() => {
    if (!enabled || !model) {
      setState((s) => (s.loading ? { ...s, loading: false } : s));
      return undefined;
    }
    const key = `${model}|${text}`;
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true }));
    const timer = setTimeout(async () => {
      try {
        const data = await api.predict({ text, model, top_k: topK }, controller.signal);
        lastKey.current = key;
        setState({ data, loading: false, error: null });
      } catch (e) {
        if (e.canceled) return;
        setState((s) => ({ data: s.data, loading: false, error: e }));
      }
    }, delay);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [text, model, enabled, delay, topK, retryTick]);

  const retry = useCallback(() => setRetryTick((n) => n + 1), []);
  return { ...state, retry };
}
