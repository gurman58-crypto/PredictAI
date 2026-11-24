import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";

const ModelsContext = createContext(null);
const STORAGE_KEY = "predictai.model.v2";

function readStored() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function ModelsProvider({ children }) {
  const [state, setState] = useState({ loading: true, error: null, models: [], defaultModel: null });
  const [selected, setSelectedRaw] = useState(readStored());

  const load = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await api.models();
      setState({ loading: false, error: null, models: data.models, defaultModel: data.default_model });
    } catch (e) {
      setState((s) => ({ ...s, loading: false, error: e }));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const setSelected = useCallback((id) => {
    setSelectedRaw(id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* storage unavailable: keep the choice in memory only */
    }
  }, []);

  const value = useMemo(() => {
    const ready = state.models.filter((m) => m.status === "ready").map((m) => m.id);
    // never leave the user on a model that cannot answer
    const model = ready.includes(selected) ? selected : state.defaultModel || selected || "ngram";
    return { ...state, ready, model, setModel: setSelected, reload: load, byId: Object.fromEntries(state.models.map((m) => [m.id, m])) };
  }, [state, selected, setSelected, load]);

  return <ModelsContext.Provider value={value}>{children}</ModelsContext.Provider>;
}

export const useModels = () => useContext(ModelsContext);
