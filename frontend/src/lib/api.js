import axios from "axios";
import { getClientId } from "./clientId";

export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

const http = axios.create({ baseURL: API_URL, timeout: 8000 });
// Documents can take longer on a cold Render instance, so they get their own timeout.
const SLOW = { timeout: 30000 };

http.interceptors.request.use((config) => {
  config.headers["X-Client-Id"] = getClientId();
  return config;
});

/** Turns any axios failure into { message, code, status, canceled }. */
export function toApiError(err) {
  if (axios.isCancel(err) || err?.code === "ERR_CANCELED") return { canceled: true, message: "canceled" };
  const data = err?.response?.data?.error;
  if (data) return { code: data.code, message: data.message, status: err.response.status, details: data.details };
  if (err?.code === "ECONNABORTED") {
    return { code: "timeout", message: "The server took too long to respond.", status: 0 };
  }
  return {
    code: "network_error",
    message: `Cannot reach the PredictAI server at ${API_URL}. Start the backend or check VITE_API_URL.`,
    status: 0,
  };
}

async function call(promise) {
  try {
    return (await promise).data;
  } catch (err) {
    throw toApiError(err);
  }
}

export const api = {
  health: () => call(http.get("/", SLOW)),
  models: () => call(http.get("/models", SLOW)),
  evaluation: () => call(http.get("/evaluation", SLOW)),
  analytics: () => call(http.get("/analytics")),
  predict: (body, signal) => call(http.post("/predict", body, { signal, timeout: 6000 })),
  complete: (body, signal) => call(http.post("/complete", body, { signal, timeout: 10000 })),
  check: (body, signal) => call(http.post("/writing/check", body, { signal, timeout: 10000 })),
  grammar: (text, signal) => call(http.post("/writing/grammar", { text }, { signal, timeout: 10000 })),
  style: (text, tone, signal) => call(http.post("/writing/style", { text, tone }, { signal, timeout: 10000 })),
  paraphrase: (body, signal) => call(http.post("/writing/paraphrase", body, { signal, timeout: 30000 })),
  accepted: (model) => http.post("/analytics/accepted", { model }).catch(() => {}),
  listDocuments: (params) => call(http.get("/documents", { params: { limit: 100, ...params }, ...SLOW })),
  getDocument: (id) => call(http.get(`/documents/${encodeURIComponent(id)}`, SLOW)),
  createDocument: (body) => call(http.post("/documents", body, SLOW)),
  updateDocument: (id, body) => call(http.put(`/documents/${encodeURIComponent(id)}`, body, SLOW)),
  deleteDocument: (id) => call(http.delete(`/documents/${encodeURIComponent(id)}`, SLOW)),
};
