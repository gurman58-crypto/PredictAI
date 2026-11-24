# Security and privacy notes

## Ownership model (important)
The app has **no login**. On first visit the browser generates a random 128-bit identifier (stored in `localStorage`) and sends it as `X-Client-Id`. Every document query is filtered by it, so one browser cannot read, update or delete another browser's documents; foreign or missing ids both return `404`.

This is *ownership scoping*, not authentication: anyone who obtains the id can access that library, and clearing browser data loses access. Do not store sensitive text. If you need real accounts, add an auth provider (e.g. JWT) and replace `require_client_id` in `backend/app/security.py` with the authenticated user id; the store layer already takes an `owner_id`.

## Privacy of typed text
- `/predict` and `/complete` process the text in memory only.
- The `prediction_events` collection stores event type, model, client id, latency, mode and time. A test (`test_events_store_metadata_only_never_the_typed_text`) fails if any typed text leaks.
- Validation errors do not echo the submitted text; unexpected errors return a generic message.
- Text is persisted only through `POST/PUT /documents` (when the user presses Save).

## Hardening already in place
Pydantic validation and size limits on every input, per-IP rate limiting on prediction routes, server-side prediction timeout, CORS restricted to `CORS_ORIGINS`, document-count cap per client, indexes on `owner_id`, no secrets in the repository (`.env` is git-ignored, only `.env.example` ships), connection strings are never returned in errors.

## Before a public deployment
Use a strong Atlas password and restrict network access, set `CORS_ORIGINS` to your exact frontend origin, keep `ENVIRONMENT=production`, and consider an authentication provider and a shared rate limiter (the built-in limiter is per process).
