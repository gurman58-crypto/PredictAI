"""Pydantic request / response models (also drive the OpenAPI / Swagger documentation)."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_TEXT_CHARS = 5_000
MAX_TITLE_CHARS = 200
MAX_DOC_CHARS = 200_000

ModelId = Literal["ensemble", "ngram", "lstm", "transformer"]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TITLE_CHARS)]


# ----------------------------------------------------------------- prediction
class PredictRequest(BaseModel):
    text: str = Field(
        default="",
        max_length=MAX_TEXT_CHARS,
        description="Text BEFORE the cursor (only its tail is used). Empty text returns sentence starters.",
    )
    model: ModelId = Field(default="ngram", description="Which trained model answers the request.")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of suggestions (1-10).")

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"text": "Artificial intelligence is", "model": "lstm", "top_k": 5}]}
    )


class Suggestion(BaseModel):
    word: str
    score: float = Field(description="Model probability P(word | context). Not a calibrated confidence.")


class PredictResponse(BaseModel):
    model: ModelId
    model_version: str
    mode: Literal["next_word", "completion"] = Field(
        description="'completion' when the cursor is inside a partly typed word (suggestions start with `prefix`)."
    )
    prefix: Optional[str] = None
    context: list[str] = Field(description="The last tokens the model actually used (after normalisation).")
    predictions: list[Suggestion]
    latency_ms: float


class CompleteRequest(BaseModel):
    text: str = Field(default="", max_length=MAX_TEXT_CHARS, description="Text BEFORE the cursor.")
    model: ModelId = "ngram"
    max_tokens: int = Field(default=20, ge=1, le=60, description="Hard upper bound on generated tokens.")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    stop_at_sentence: bool = Field(default=True, description="Stop after the first '.', '!' or '?'.")
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1, description="Fix for reproducible output.")


class CompleteResponse(BaseModel):
    model: ModelId
    model_version: str
    continuation: str = Field(description="Text to insert at the cursor (already includes any leading space).")
    tokens: list[str]
    stopped_because: Literal["max_tokens", "sentence_end", "end_of_text", "no_candidates"]
    latency_ms: float


# --------------------------------------------------------------------- models
class ModelInfo(BaseModel):
    id: ModelId
    name: str
    description: str
    status: Literal["ready", "unavailable"]
    version: Optional[str] = None
    reason: Optional[str] = Field(default=None, description="Why the model is unavailable.")
    vocab_size: Optional[int] = None
    details: dict = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_model: Optional[ModelId] = Field(description="First model that is ready, or null.")


# ------------------------------------------------------------------ documents
class DocumentCreate(BaseModel):
    title: Title = "Untitled document"
    content: str = Field(default="", max_length=MAX_DOC_CHARS)


class DocumentUpdate(BaseModel):
    title: Optional[Title] = None
    content: Optional[str] = Field(default=None, max_length=MAX_DOC_CHARS)

    @model_validator(mode="after")
    def _at_least_one(self):
        if self.title is None and self.content is None:
            raise ValueError("provide at least one of: title, content")
        return self


class DocumentOut(BaseModel):
    id: str
    title: str
    content: str
    word_count: int
    char_count: int
    created_at: datetime
    updated_at: datetime


class DocumentSummary(BaseModel):
    id: str
    title: str
    preview: str
    word_count: int
    char_count: int
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    documents: list[DocumentSummary]
    count: int = Field(description="Documents returned in this page.")
    total: int = Field(description="All documents owned by the client.")


class DeleteResponse(BaseModel):
    deleted: bool
    id: str


# ------------------------------------------------------------------ analytics
class AcceptedEvent(BaseModel):
    model: ModelId


class ModelUsage(BaseModel):
    predictions: int = 0
    completions: int = 0
    accepted: int = 0
    avg_latency_ms: Optional[float] = None


class AnalyticsResponse(BaseModel):
    scope: Literal["client", "global"]
    predictions: int
    completions: int
    accepted: int
    acceptance_rate: Optional[float] = Field(default=None, description="accepted / predictions (0-1)")
    by_model: dict[str, ModelUsage]
    note: str = "Aggregates only - typed text is never stored."


# -------------------------------------------------------------- writing tools
class WritingIssue(BaseModel):
    rule: str = Field(description="Stable rule identifier, e.g. 'repeated_word'.")
    category: str
    severity: Literal["error", "warning", "info"]
    start: int = Field(description="Start offset (characters) in the submitted text.")
    end: int
    original: str = Field(description="The text covered by [start, end).")
    message: str
    replacement: Optional[str] = Field(default=None, description="Suggested fix; '' means delete; null means no automatic fix.")
    suggestions: list[str] = Field(default_factory=list, description="Alternatives to offer, best first (spelling gives up to 5).")


IgnoreList = Annotated[list[Annotated[str, StringConstraints(max_length=40)]], Field(max_length=300)]


class GrammarRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    ignore: IgnoreList = Field(default_factory=list, description="Words the user chose to accept (not flagged as misspelled).")


class GrammarResponse(BaseModel):
    issues: list[WritingIssue]
    count: int
    method: str = "rule-based"
    latency_ms: float


class StyleRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    tone: Literal["neutral", "formal"] = Field(default="neutral", description="'formal' also flags contractions and slang.")


class StyleResponse(BaseModel):
    stats: dict
    issues: list[WritingIssue]
    method: str = "rule-based"
    latency_ms: float


class CheckRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    tone: Literal["neutral", "formal"] = "neutral"
    ignore: IgnoreList = Field(default_factory=list)


class CheckResponse(BaseModel):
    """Grammar + style in one round trip, used by the editor's live checking."""

    grammar: list[WritingIssue]
    style: list[WritingIssue]
    stats: dict
    latency_ms: float


MAX_PARAPHRASE_CHARS = 1_000


class ParaphraseRequest(BaseModel):
    text: str = Field(max_length=MAX_PARAPHRASE_CHARS, description="The sentence(s) to rewrite.")
    variants: int = Field(default=3, ge=1, le=3)
    model: Optional[ModelId] = Field(default=None, description="Language model used to check fluency (default: Transformer).")
    seed: Optional[int] = Field(default=None, ge=0, le=2**31 - 1)


class ParaphraseVariant(BaseModel):
    text: str
    changes: list[dict[str, str]] = Field(description="List of {from, to} substitutions made.")
    fluency_delta: Optional[float] = Field(default=None, description="Sum of log-likelihood change vs. the original wording (higher = more natural).")


class ParaphraseResponse(BaseModel):
    variants: list[ParaphraseVariant]
    scored: bool = Field(description="True when a language model checked the fluency of each substitution.")
    model: Optional[str] = None
    message: Optional[str] = None
    method: str = "substitution + language-model fluency filter"
    latency_ms: float
