"""
Core Pydantic schemas for every frozen interface in the RAG pipeline.

Design rules
------------
- Every schema uses ``model_config = ConfigDict(frozen=True)`` so instances
  are immutable and hashable after construction.
- Validation is strict: extra fields are forbidden so a typo in a field name
  raises immediately rather than passing silently.
- Each schema carries a ``model_json_schema()`` that downstream code can use
  to generate JSON Schema for guided decoding and gateway caching.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stable_hash(obj: Any) -> str:
    """Return a 16-character hex SHA-256 hash of the JSON-serialised object."""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Doc
# ---------------------------------------------------------------------------

class Doc(BaseModel):
    """A parsed source document with character-level metadata.

    ``source_path`` is the path relative to the corpus root.
    ``content`` is the full raw text of the document.
    ``char_start`` and ``char_end`` are always 0 and ``len(content)`` for the
    document root; they exist so the same span logic applies at every level.
    ``metadata`` holds arbitrary key/value pairs (title, version, timestamp …).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(..., description="Stable content-addressed identifier (SHA-256[:16] of path + content).")
    source_path: str = Field(..., description="Corpus-relative path to the source file.")
    content: str = Field(..., description="Full raw text of the document.")
    char_start: int = Field(0, ge=0, description="Always 0 for a document root.")
    char_end: int = Field(..., gt=0, description="len(content); set automatically.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Arbitrary document-level metadata.")

    @model_validator(mode="after")
    def _char_end_matches_content(self) -> "Doc":
        if self.char_end != len(self.content):
            raise ValueError(
                f"char_end ({self.char_end}) must equal len(content) ({len(self.content)})"
            )
        return self

    @classmethod
    def build(cls, source_path: str, content: str, metadata: dict[str, str] | None = None) -> "Doc":
        """Factory that derives ``doc_id`` and ``char_end`` automatically."""
        doc_id = _stable_hash({"path": source_path, "content": content})
        return cls(
            doc_id=doc_id,
            source_path=source_path,
            content=content,
            char_start=0,
            char_end=len(content),
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

class Span(BaseModel):
    """A character-level span inside a document.

    Spans are the atomic unit of gold evidence. ``raw[char_start:char_end]``
    must reproduce the original source text exactly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(..., description="Parent document identifier.")
    char_start: int = Field(..., ge=0, description="Inclusive start offset.")
    char_end: int = Field(..., gt=0, description="Exclusive end offset.")
    text: str = Field(..., description="The span text (doc.content[char_start:char_end]).")

    @model_validator(mode="after")
    def _start_before_end(self) -> "Span":
        if self.char_start >= self.char_end:
            raise ValueError(
                f"char_start ({self.char_start}) must be < char_end ({self.char_end})"
            )
        return self

    @classmethod
    def from_doc(cls, doc: Doc, char_start: int, char_end: int) -> "Span":
        """Factory that extracts and validates the text slice from a Doc."""
        text = doc.content[char_start:char_end]
        if not text:
            raise ValueError("Span text is empty; check offsets.")
        return cls(doc_id=doc.doc_id, char_start=char_start, char_end=char_end, text=text)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A retrieval unit derived from a document via a specific chunking variant.

    ``chunk_id`` is a stable hash of ``(doc_id, char_start, char_end, variant)``
    so that re-running the chunker with the same inputs produces identical IDs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., description="Stable hash of (doc_id, char_start, char_end, variant).")
    doc_id: str = Field(..., description="Parent document identifier.")
    char_start: int = Field(..., ge=0, description="Inclusive character start in the parent document.")
    char_end: int = Field(..., gt=0, description="Exclusive character end in the parent document.")
    variant: str = Field(..., description="Chunking strategy name, e.g. 'fixed-256', 'structure-aware'.")
    text: str = Field(..., description="The chunk text.")
    token_count: int | None = Field(None, ge=0, description="Token count under the embedder's tokenizer (populated later).")
    metadata: dict[str, str] = Field(default_factory=dict, description="Optional chunk-level metadata.")

    @model_validator(mode="after")
    def _start_before_end(self) -> "Chunk":
        if self.char_start >= self.char_end:
            raise ValueError(
                f"char_start ({self.char_start}) must be < char_end ({self.char_end})"
            )
        return self

    @classmethod
    def build(
        cls,
        doc: Doc,
        char_start: int,
        char_end: int,
        variant: str,
        metadata: dict[str, str] | None = None,
    ) -> "Chunk":
        """Factory that derives ``chunk_id`` and ``text`` from the parent Doc."""
        chunk_id = _stable_hash(
            {"doc_id": doc.doc_id, "start": char_start, "end": char_end, "variant": variant}
        )
        text = doc.content[char_start:char_end]
        if not text:
            raise ValueError("Chunk text is empty; check offsets.")
        return cls(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            char_start=char_start,
            char_end=char_end,
            variant=variant,
            text=text,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Qrel  (relevance judgement)
# ---------------------------------------------------------------------------

class Qrel(BaseModel):
    """A single relevance judgement in TREC qrels format.

    ``relevance`` follows the standard TREC graded scale:
        0 = not relevant
        1 = partially relevant
        2 = highly relevant
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(..., description="Unique query identifier.")
    doc_id: str = Field(..., description="Document or chunk identifier being judged.")
    relevance: int = Field(..., ge=0, le=2, description="Graded relevance: 0 not relevant, 1 partial, 2 highly relevant.")
    span: Span | None = Field(None, description="Gold evidence span, if available.")

    @field_validator("query_id", "doc_id")
    @classmethod
    def _non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ID fields must be non-empty strings.")
        return v


# ---------------------------------------------------------------------------
# Run  (retrieval result)
# ---------------------------------------------------------------------------

class ScoredDoc(BaseModel):
    """A single retrieved document with its retrieval score and rank."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(..., description="Chunk or document identifier.")
    score: float = Field(..., description="Retrieval score (higher = more relevant).")
    rank: int = Field(..., ge=1, description="1-based rank in the result list.")


class Run(BaseModel):
    """The complete output of a retrieval system for a single query.

    Maps directly to a TREC run file row: ``(query_id, doc_id, rank, score, run_id)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(..., description="Query identifier.")
    run_id: str = Field(..., description="Identifier for the retrieval system / experiment run.")
    results: tuple[ScoredDoc, ...] = Field(..., description="Ranked list of scored documents.")

    @field_validator("results", mode="after")
    @classmethod
    def _ranks_are_consecutive(cls, v: tuple[ScoredDoc, ...]) -> tuple[ScoredDoc, ...]:
        ranks = sorted(r.rank for r in v)
        if ranks and ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"Ranks must be consecutive starting from 1; got {ranks}.")
        return v

    @field_validator("query_id", "run_id")
    @classmethod
    def _non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ID fields must be non-empty strings.")
        return v


# ---------------------------------------------------------------------------
# TraceEvent  (gateway / pipeline observability)
# ---------------------------------------------------------------------------

class TraceEvent(BaseModel):
    """A single instrumented event emitted by the LLM gateway or pipeline stage.

    Every call that touches an LLM or external service emits a TraceEvent so
    that cost and latency are always attributable to a specific run_id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(..., description="Experiment run identifier this event belongs to.")
    stage: str = Field(..., description="Pipeline stage name, e.g. 'gateway', 'retriever', 'reranker'.")
    model: str | None = Field(None, description="Model name, if an LLM was called.")
    prompt_hash: str | None = Field(None, description="SHA-256[:16] of the serialised prompt, for cache keying.")
    input_tokens: int | None = Field(None, ge=0, description="Prompt token count.")
    output_tokens: int | None = Field(None, ge=0, description="Completion token count.")
    cost_usd: float | None = Field(None, ge=0.0, description="Estimated cost in USD.")
    latency_ms: float = Field(..., ge=0.0, description="Wall-clock latency of this call in milliseconds.")
    cache_hit: bool = Field(False, description="True if the response was served from the disk cache.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Arbitrary stage-specific metadata.")

    @field_validator("run_id", "stage")
    @classmethod
    def _non_empty_string(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ID/stage fields must be non-empty strings.")
        return v
