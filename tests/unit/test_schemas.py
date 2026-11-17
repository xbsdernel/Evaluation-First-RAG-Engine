"""
Tests for rag_engine.schemas.core (ISS-001).

Coverage targets:
- Round-trip serialisation for every schema (JSON → model → JSON → model).
- Validation rejects bad inputs loudly.
- Renaming any field (simulated by omitting a required field) breaks the test.
- Stable hash determinism: identical inputs produce identical IDs across runs.
- RNG seeding: file-hash output is identical on repeated invocations.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest
from pydantic import ValidationError

from rag_engine.schemas import (
    Chunk,
    Doc,
    Qrel,
    Run,
    ScoredDoc,
    Span,
    TraceEvent,
    _stable_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_doc() -> Doc:
    return Doc.build(
        source_path="docs/guide.md",
        content="Hello world. This is a test document.",
        metadata={"title": "Guide", "version": "1.0"},
    )


@pytest.fixture()
def sample_span(sample_doc: Doc) -> Span:
    # "Hello world"
    return Span.from_doc(sample_doc, char_start=0, char_end=11)


@pytest.fixture()
def sample_chunk(sample_doc: Doc) -> Chunk:
    return Chunk.build(
        doc=sample_doc,
        char_start=0,
        char_end=11,
        variant="fixed-256",
    )


@pytest.fixture()
def sample_qrel(sample_span: Span) -> Qrel:
    return Qrel(
        query_id="q-001",
        doc_id="chunk-abc",
        relevance=2,
        span=sample_span,
    )


@pytest.fixture()
def sample_run() -> Run:
    return Run(
        query_id="q-001",
        run_id="bm25-baseline",
        results=(
            ScoredDoc(doc_id="chunk-abc", score=3.14, rank=1),
            ScoredDoc(doc_id="chunk-def", score=2.71, rank=2),
        ),
    )


@pytest.fixture()
def sample_trace() -> TraceEvent:
    return TraceEvent(
        run_id="run-001",
        stage="gateway",
        model="gpt-4o-mini",
        prompt_hash="abc123def456",
        input_tokens=50,
        output_tokens=20,
        cost_usd=0.0003,
        latency_ms=312.5,
        cache_hit=False,
    )


# ---------------------------------------------------------------------------
# Doc
# ---------------------------------------------------------------------------

class TestDoc:
    def test_build_derives_doc_id_and_char_end(self, sample_doc: Doc) -> None:
        assert len(sample_doc.doc_id) == 16
        assert sample_doc.char_end == len(sample_doc.content)
        assert sample_doc.char_start == 0

    def test_round_trip_json(self, sample_doc: Doc) -> None:
        raw = sample_doc.model_dump_json()
        restored = Doc.model_validate_json(raw)
        assert restored == sample_doc

    def test_char_end_mismatch_raises(self) -> None:
        with pytest.raises(ValidationError, match="char_end"):
            Doc(
                doc_id="x" * 16,
                source_path="a.md",
                content="hello",
                char_start=0,
                char_end=99,  # wrong
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Doc.model_validate(
                {
                    "doc_id": "x" * 16,
                    "source_path": "a.md",
                    "content": "hi",
                    "char_start": 0,
                    "char_end": 2,
                    "unexpected_field": "boom",
                }
            )

    def test_missing_required_field_raises(self) -> None:
        """Renaming 'source_path' → 'path' must break validation."""
        with pytest.raises(ValidationError):
            Doc.model_validate(
                {
                    "doc_id": "x" * 16,
                    "path": "a.md",          # wrong field name
                    "content": "hi",
                    "char_start": 0,
                    "char_end": 2,
                }
            )

    def test_stable_hash_is_deterministic(self) -> None:
        d1 = Doc.build("a.md", "content")
        d2 = Doc.build("a.md", "content")
        assert d1.doc_id == d2.doc_id

    def test_different_paths_produce_different_ids(self) -> None:
        d1 = Doc.build("a.md", "content")
        d2 = Doc.build("b.md", "content")
        assert d1.doc_id != d2.doc_id

    def test_frozen_immutability(self, sample_doc: Doc) -> None:
        with pytest.raises((TypeError, ValidationError)):
            sample_doc.doc_id = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

class TestSpan:
    def test_text_matches_slice(self, sample_doc: Doc) -> None:
        span = Span.from_doc(sample_doc, 0, 11)
        assert span.text == sample_doc.content[0:11]

    def test_round_trip_json(self, sample_span: Span) -> None:
        raw = sample_span.model_dump_json()
        restored = Span.model_validate_json(raw)
        assert restored == sample_span

    def test_start_ge_end_raises(self, sample_doc: Doc) -> None:
        with pytest.raises(ValidationError, match="char_start"):
            Span(doc_id=sample_doc.doc_id, char_start=10, char_end=5, text="bad")

    def test_start_equals_end_raises(self, sample_doc: Doc) -> None:
        with pytest.raises(ValidationError, match="char_start"):
            Span(doc_id=sample_doc.doc_id, char_start=5, char_end=5, text="bad")

    def test_empty_offset_raises(self, sample_doc: Doc) -> None:
        with pytest.raises(ValueError, match="empty"):
            Span.from_doc(sample_doc, 5, 5)

    def test_missing_required_field_raises(self) -> None:
        """Renaming 'char_start' must break validation."""
        with pytest.raises(ValidationError):
            Span.model_validate(
                {"doc_id": "abc", "start": 0, "char_end": 5, "text": "hello"}
            )


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_build_derives_chunk_id_and_text(self, sample_doc: Doc) -> None:
        chunk = Chunk.build(sample_doc, 0, 11, "fixed-256")
        assert chunk.text == "Hello world"
        assert len(chunk.chunk_id) == 16

    def test_stable_id_across_builds(self, sample_doc: Doc) -> None:
        c1 = Chunk.build(sample_doc, 0, 11, "fixed-256")
        c2 = Chunk.build(sample_doc, 0, 11, "fixed-256")
        assert c1.chunk_id == c2.chunk_id

    def test_different_variant_different_id(self, sample_doc: Doc) -> None:
        c1 = Chunk.build(sample_doc, 0, 11, "fixed-256")
        c2 = Chunk.build(sample_doc, 0, 11, "structure-aware")
        assert c1.chunk_id != c2.chunk_id

    def test_round_trip_json(self, sample_chunk: Chunk) -> None:
        raw = sample_chunk.model_dump_json()
        restored = Chunk.model_validate_json(raw)
        assert restored == sample_chunk

    def test_start_ge_end_raises(self, sample_doc: Doc) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="x" * 16,
                doc_id=sample_doc.doc_id,
                char_start=10,
                char_end=5,
                variant="v",
                text="bad",
            )

    def test_missing_required_field_raises(self) -> None:
        """Renaming 'variant' must break validation."""
        with pytest.raises(ValidationError):
            Chunk.model_validate(
                {
                    "chunk_id": "x" * 16,
                    "doc_id": "y" * 16,
                    "char_start": 0,
                    "char_end": 5,
                    "chunking_strategy": "fixed-256",  # wrong field name
                    "text": "hello",
                }
            )


# ---------------------------------------------------------------------------
# Qrel
# ---------------------------------------------------------------------------

class TestQrel:
    def test_valid_construction(self, sample_qrel: Qrel) -> None:
        assert sample_qrel.relevance == 2

    def test_round_trip_json(self, sample_qrel: Qrel) -> None:
        raw = sample_qrel.model_dump_json()
        restored = Qrel.model_validate_json(raw)
        assert restored == sample_qrel

    def test_relevance_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            Qrel(query_id="q1", doc_id="d1", relevance=3)

    def test_negative_relevance_raises(self) -> None:
        with pytest.raises(ValidationError):
            Qrel(query_id="q1", doc_id="d1", relevance=-1)

    def test_empty_query_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            Qrel(query_id="   ", doc_id="d1", relevance=1)

    def test_missing_required_field_raises(self) -> None:
        """Renaming 'relevance' must break validation."""
        with pytest.raises(ValidationError):
            Qrel.model_validate({"query_id": "q1", "doc_id": "d1", "grade": 2})


# ---------------------------------------------------------------------------
# Run / ScoredDoc
# ---------------------------------------------------------------------------

class TestRun:
    def test_valid_construction(self, sample_run: Run) -> None:
        assert len(sample_run.results) == 2
        assert sample_run.results[0].rank == 1

    def test_round_trip_json(self, sample_run: Run) -> None:
        raw = sample_run.model_dump_json()
        restored = Run.model_validate_json(raw)
        assert restored == sample_run

    def test_non_consecutive_ranks_raise(self) -> None:
        with pytest.raises(ValidationError, match="consecutive"):
            Run(
                query_id="q1",
                run_id="sys",
                results=(
                    ScoredDoc(doc_id="a", score=1.0, rank=1),
                    ScoredDoc(doc_id="b", score=0.5, rank=3),  # gap: rank 2 missing
                ),
            )

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            Run(query_id="q1", run_id="  ", results=())

    def test_missing_required_field_raises(self) -> None:
        """Renaming 'results' must break validation."""
        with pytest.raises(ValidationError):
            Run.model_validate({"query_id": "q1", "run_id": "sys", "hits": []})


# ---------------------------------------------------------------------------
# TraceEvent
# ---------------------------------------------------------------------------

class TestTraceEvent:
    def test_valid_construction(self, sample_trace: TraceEvent) -> None:
        assert sample_trace.cache_hit is False
        assert sample_trace.latency_ms == 312.5

    def test_round_trip_json(self, sample_trace: TraceEvent) -> None:
        raw = sample_trace.model_dump_json()
        restored = TraceEvent.model_validate_json(raw)
        assert restored == sample_trace

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValidationError):
            TraceEvent(run_id="r1", stage="gateway", latency_ms=-1.0)

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValidationError):
            TraceEvent(run_id="r1", stage="gateway", latency_ms=10.0, cost_usd=-0.01)

    def test_cache_hit_no_cost(self) -> None:
        """A cache-hit event legitimately has zero cost."""
        t = TraceEvent(
            run_id="r1",
            stage="gateway",
            latency_ms=1.0,
            cost_usd=0.0,
            cache_hit=True,
        )
        assert t.cache_hit is True
        assert t.cost_usd == 0.0

    def test_missing_required_field_raises(self) -> None:
        """Renaming 'stage' must break validation."""
        with pytest.raises(ValidationError):
            TraceEvent.model_validate(
                {"run_id": "r1", "pipeline_stage": "gateway", "latency_ms": 1.0}
            )


# ---------------------------------------------------------------------------
# Stable hash helper
# ---------------------------------------------------------------------------

class TestStableHash:
    def test_deterministic(self) -> None:
        h1 = _stable_hash({"key": "value", "num": 42})
        h2 = _stable_hash({"key": "value", "num": 42})
        assert h1 == h2

    def test_key_order_independent(self) -> None:
        h1 = _stable_hash({"a": 1, "b": 2})
        h2 = _stable_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_inputs_different_hashes(self) -> None:
        h1 = _stable_hash({"x": 1})
        h2 = _stable_hash({"x": 2})
        assert h1 != h2

    def test_length_is_16(self) -> None:
        assert len(_stable_hash({})) == 16


# ---------------------------------------------------------------------------
# RNG seeding — deterministic output
# ---------------------------------------------------------------------------

class TestRNGSeeding:
    """
    Demonstrates that seeded random calls produce identical sequences across
    multiple invocations in the same test run.
    """

    def _generate_hashes(self, seed: int, n: int = 8) -> list[str]:
        random.seed(seed)
        values = [random.random() for _ in range(n)]
        raw = json.dumps(values, sort_keys=True)
        return [hashlib.sha256(raw.encode()).hexdigest()]

    def test_same_seed_same_output(self) -> None:
        hashes_a = self._generate_hashes(seed=42)
        hashes_b = self._generate_hashes(seed=42)
        assert hashes_a == hashes_b

    def test_different_seed_different_output(self) -> None:
        hashes_a = self._generate_hashes(seed=42)
        hashes_b = self._generate_hashes(seed=99)
        assert hashes_a != hashes_b
