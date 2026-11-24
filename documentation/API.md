# API reference

Base URL (local): `http://localhost:8000`. Interactive docs: `/docs`, `/redoc`, raw schema `/openapi.json`.

## Conventions

- JSON in, JSON out. Time stamps are ISO-8601 UTC.
- **Client id.** Document routes require `X-Client-Id: <16-64 chars of [A-Za-z0-9_-]>`. The React app generates one per browser. `/predict`, `/complete`, `/analytics` accept it optionally (usage is then attributed to it).
- **Errors** always have the form
  ```json
  {"error": {"code": "validation_error", "message": "The request is invalid.", "details": [{"field": "top_k", "message": "..."}]}}
  ```
  The submitted text is never echoed in an error. Stack traces and paths are never returned.

| HTTP | `code` | When |
|---|---|---|
| 400 | `invalid_client_id` | malformed `X-Client-Id` |
| 401 | `missing_client_id` | document route without `X-Client-Id` |
| 404 | `not_found` | unknown route, or a document that does not exist **or belongs to someone else** |
| 405 | `method_not_allowed` | |
| 409 | `document_limit_reached` | more than `MAX_DOCUMENTS_PER_CLIENT` (200) |
| 422 | `validation_error`, `unknown_model` | bad body/params |
| 429 | `rate_limited` | per-IP limit on `/predict`, `/complete`; `Retry-After` header set |
| 503 | `model_unavailable`, `database_unavailable` | model not trained / MongoDB unreachable |
| 504 | `prediction_timeout` | inference exceeded `PREDICT_TIMEOUT_SECONDS` |
| 500 | `internal_error` | anything unexpected (generic message) |

## System

### `GET /`
```json
{"name":"PredictAI API","version":"1.0.0","status":"ok","environment":"development",
 "database":{"backend":"mongodb","connected":true},
 "models":{"ngram":"ready","lstm":"ready","transformer":"ready","ensemble":"ready"},"docs":"/docs"}
```
`status` is `degraded` when no model is ready. `database.backend` is `memory` when `MONGODB_URI` is not set.

### `GET /health` → `{"status":"ok"}`

### `GET /models`
```json
{"models":[{"id":"ngram","name":"N-gram (trigram)","description":"...","status":"ready","version":"v1",
            "reason":null,"vocab_size":10000,"details":{"order":3,"discount":0.95,"...":"..."}},
           {"id":"lstm","status":"unavailable","reason":"not trained yet - run training/train_lstm.py","...":"..."}],
 "default_model":"ngram"}
```

### `GET /evaluation`
`{"status":"pending","message":"..."}` until the evaluation script has been run; otherwise `{"status":"available", ...report}` (metrics per model, N-gram order variants, dataset facts, protocol).

## Prediction

### `POST /predict`
Request: `text` (string, ≤ 5000 chars, the text **before the cursor**; may be empty), `model` (`ensemble` | `ngram` | `lstm` | `transformer`, default `ngram`), `top_k` (1–10, default 5).

```json
{"text": "Artificial intelligence is", "model": "ngram", "top_k": 5}
```
Response:
```json
{"model":"ngram","model_version":"v1","mode":"next_word","prefix":null,
 "context":["artificial","intelligence","is"],
 "predictions":[{"word":"a","score":0.1234}, "..."],
 "latency_ms":0.6}
```
- `mode = "completion"` when the text ends inside a word (`"the cat sa"`): suggestions then start with `prefix`.
- `score` is the model probability `P(word | context)`, not a calibrated confidence.
- Only words are returned (no punctuation, `<unk>`, `<eos>`).
- Empty/unknown input falls back to sentence-start predictions; a prefix with no match returns `predictions: []`.

### `POST /complete`
Request: `text`, `model`, `max_tokens` (1–60, default 20), `temperature` (0.1–2, default 0.8), `stop_at_sentence` (default true), `seed` (optional, makes the output reproducible).
Response:
```json
{"model":"ngram","model_version":"v1","continuation":" the first half of the season.",
 "tokens":["the","first","half","of","the","season","."],
 "stopped_because":"sentence_end","latency_ms":3.1}
```
`continuation` already contains the leading space and capitalisation, so the client simply inserts it at the cursor. `stopped_because`: `max_tokens`, `sentence_end`, `end_of_text`, `no_candidates`.

## Writing tools

None of these endpoints log or store the submitted text. Offsets (`start`, `end`) are character positions in the submitted text.

### `POST /writing/check`
Grammar and style in one round trip (used for live checking). Request: `{"text": "...", "tone": "neutral"|"formal", "ignore": ["word", ...]}` (≤ 5000 characters; ≤ 300 ignored words). Response: `{"grammar":[issue,...],"style":[issue,...],"stats":{...},"latency_ms"}`.

### `POST /writing/grammar`
Request: `{"text": "...", "ignore": []}` (≤ 5000 characters). Each issue also has `suggestions` (alternatives, best first; up to 5 for spelling). Response: `{"issues":[{"rule","category","severity","start","end","original","message","replacement"}],"count","method":"rule-based","latency_ms"}`. `replacement` is the suggested fix (`""` = delete, `null` = no automatic fix).

### `POST /writing/style`
Request: `{"text": "...", "tone": "neutral" | "formal"}`. Response: `{"stats":{"words","sentences","avg_sentence_words","reading_ease","grade_level","reading_level","reading_time_seconds","unique_word_ratio"},"issues":[...],"method","latency_ms"}`.

### `POST /writing/paraphrase`
Request: `{"text": "...", "variants": 1-3, "model": "transformer" | ... | null, "seed": null}` (text ≤ 1000 characters). Response: `{"variants":[{"text","changes":[{"from","to"}],"fluency_delta"}],"scored":true,"model":"transformer","message":null,"latency_ms"}`. `scored` is `false` when no language model was available to check fluency. An explicitly requested model that is not available returns `503`; empty text returns `422`.

## Documents (require `X-Client-Id`)

| Call | Body | Success |
|---|---|---|
| `GET /documents?limit=50&skip=0` | | `{"documents":[{id,title,preview,word_count,char_count,created_at,updated_at}],"count":n,"total":n}` newest first |
| `POST /documents` | `{"title":"...", "content":"..."}` (title 1–200 chars, default "Untitled document"; content ≤ 200 000 chars) | `201` full document |
| `GET /documents/{id}` | | full document |
| `PUT /documents/{id}` | `{"title"?, "content"?}` (at least one) | updated document |
| `DELETE /documents/{id}` | | `{"deleted":true,"id":"..."}` |

A document that belongs to another client returns the same `404` as a missing one.

## Analytics

- `GET /analytics` → `{"scope":"client"|"global","predictions":n,"completions":n,"accepted":n,"acceptance_rate":0.31,"by_model":{"ngram":{"predictions":..,"completions":..,"accepted":..,"avg_latency_ms":..}}}`. Scoped to the caller when `X-Client-Id` is sent.
- `POST /analytics/accepted` `{"model":"ngram"}` → `204`. Records that a suggestion was accepted.

Only the event type, model, client id, latency, mode and time are stored, **never text**.
