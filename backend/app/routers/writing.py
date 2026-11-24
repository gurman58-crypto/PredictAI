"""Grammar, style and paraphrase endpoints.

Privacy: like /predict, these endpoints never log or store the submitted text.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from app.errors import ApiError
from app.schemas import (
    CheckRequest,
    CheckResponse,
    GrammarRequest,
    GrammarResponse,
    ParaphraseRequest,
    ParaphraseResponse,
    ParaphraseVariant,
    StyleRequest,
    StyleResponse,
    WritingIssue,
)
from app.security import enforce_rate_limit
from ml_models.writing.grammar import check_grammar
from ml_models.writing.paraphrase import paraphrase
from ml_models.writing.style import check_style

router = APIRouter(prefix="/writing", tags=["writing tools"], dependencies=[Depends(enforce_rate_limit)])


async def _run(request: Request, fn, *args, **kwargs):
    timeout = request.app.state.settings.predict_timeout_seconds * 3
    try:
        return await asyncio.wait_for(run_in_threadpool(fn, *args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        raise ApiError(504, "writing_timeout", "The writing tool took too long to answer. Your text was not changed.")


def _engine_or_none(request: Request):
    engine = request.app.state.engine
    return engine if engine.any_ready() else None


@router.post("/check", response_model=CheckResponse, summary="Grammar + style in one call (live checking while typing)")
async def check(body: CheckRequest, request: Request):
    t0 = time.perf_counter()
    grammar_issues = await _run(request, check_grammar, body.text, _engine_or_none(request), body.ignore)
    stats, style_issues = await _run(request, check_style, body.text, body.tone)
    return CheckResponse(
        grammar=[WritingIssue(**i.to_dict(body.text)) for i in grammar_issues],
        style=[WritingIssue(**i.to_dict(body.text)) for i in style_issues],
        stats=stats,
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


@router.post("/grammar", response_model=GrammarResponse, summary="Rule-based grammar, spelling and punctuation check")
async def grammar(body: GrammarRequest, request: Request):
    t0 = time.perf_counter()
    issues = await _run(request, check_grammar, body.text, _engine_or_none(request), body.ignore)
    return GrammarResponse(
        issues=[WritingIssue(**i.to_dict(body.text)) for i in issues],
        count=len(issues),
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


@router.post("/style", response_model=StyleResponse, summary="Style suggestions and readability statistics")
async def style(body: StyleRequest, request: Request):
    t0 = time.perf_counter()
    stats, issues = await _run(request, check_style, body.text, body.tone)
    return StyleResponse(
        stats=stats,
        issues=[WritingIssue(**i.to_dict(body.text)) for i in issues],
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


@router.post("/paraphrase", response_model=ParaphraseResponse, summary="Rewrite wording with language-model-checked substitutions")
async def paraphrase_text(body: ParaphraseRequest, request: Request):
    t0 = time.perf_counter()
    engine = request.app.state.engine
    if not body.text.strip():
        raise ApiError(422, "empty_text", "Select or write some text to paraphrase.")
    model_id = body.model
    if model_id is not None:
        engine._get(model_id)            # raises a clear 503 if that model is not available
    elif not engine.any_ready():
        engine = None                    # still usable without a model, but unscored
    variants, scored = await _run(request, paraphrase, body.text, engine, model_id, body.variants, body.seed)
    used = None
    if scored:
        used = model_id or next((m for m in ("transformer", "ensemble", "lstm", "ngram") if engine.is_ready(m)), None)
    return ParaphraseResponse(
        variants=[ParaphraseVariant(text=v.text, changes=v.changes, fluency_delta=v.fluency_delta) for v in variants],
        scored=scored,
        model=used,
        message=None if variants else "No safe rewrites were found for this text. Try a longer or different sentence.",
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )
