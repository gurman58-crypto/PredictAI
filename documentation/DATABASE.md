# Database (MongoDB)

Database name: `MONGODB_DB` (default `predictai`). Driver: PyMongo. If `MONGODB_URI` is empty an in-memory store with the identical interface is used (for demos/tests; `GET /` reports `"backend": "memory"`).

## `documents`
| Field | Type | Notes |
|---|---|---|
| `_id` | ObjectId | exposed as `id` (24 hex chars) |
| `owner_id` | string | anonymous client id; **part of every query** |
| `title` | string | 1–200 chars |
| `content` | string | ≤ 200 000 chars |
| `word_count`, `char_count` | int | recomputed on every content change |
| `created_at`, `updated_at` | datetime (UTC, ms precision) | `updated_at` changes on every update |

Index: `(owner_id, updated_at desc)`.

## `prediction_events`
`{type: predict|complete|accept, model, owner_id, latency_ms, mode, timestamp}`. **No text is stored.** Index: `(owner_id, timestamp desc)`. Delete the collection at any time without affecting documents.

## Atlas setup
See README §10. Checklist: database user created · your IP allowed in Network Access · password URL-encoded in the URI · `GET /` shows `connected: true`.

## Backups / maintenance
Use Atlas backups or `mongodump --uri "$MONGODB_URI" --db predictai`. To delete one browser's data: `db.documents.deleteMany({owner_id: "<id>"})`.
