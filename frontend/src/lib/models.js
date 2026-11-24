// One place for the model catalogue shown in the UI (ids come from the backend's GET /models).
export const MODEL_ORDER = ["ensemble", "ngram", "lstm", "transformer"];
export const MODEL_LABELS = { ensemble: "Blend", ngram: "N-gram", lstm: "LSTM", transformer: "Transformer" };
export const modelLabel = (id) => MODEL_LABELS[id] || id;
