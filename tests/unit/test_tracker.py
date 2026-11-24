"""
Unit tests for rag_engine.evaluation.tracker (ISS-002).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_engine.evaluation.tracker import (
    ExperimentRun,
    ExperimentTracker,
    compute_config_hash,
)


class TestConfigHash:
    def test_deterministic_hash(self) -> None:
        cfg1 = {"seed": 42, "model": "bm25", "k1": 1.2, "b": 0.75}
        cfg2 = {"b": 0.75, "k1": 1.2, "model": "bm25", "seed": 42}
        assert compute_config_hash(cfg1) == compute_config_hash(cfg2)
        assert len(compute_config_hash(cfg1)) == 16

    def test_different_configs_produce_different_hashes(self) -> None:
        cfg1 = {"k1": 1.2}
        cfg2 = {"k1": 1.5}
        assert compute_config_hash(cfg1) != compute_config_hash(cfg2)


class TestExperimentRun:
    def test_create_factory(self) -> None:
        run = ExperimentRun.create(
            run_id="run-001",
            parameters={"k1": 1.2, "b": 0.75},
            metrics={"ndcg@10": 0.654, "recall@10": 0.812},
            artifacts={"corpus": "data/raw.dvc"},
            notes="Initial baseline run",
            git_commit="b04a8a6",
        )
        assert run.run_id == "run-001"
        assert run.config_hash == compute_config_hash({"k1": 1.2, "b": 0.75})
        assert run.metrics["ndcg@10"] == 0.654
        assert run.git_commit == "b04a8a6"

    def test_frozen_immutable(self) -> None:
        run = ExperimentRun.create(
            run_id="run-001",
            parameters={"seed": 42},
        )
        with pytest.raises((TypeError, ValidationError)):
            run.run_id = "mutated"  # type: ignore[misc]

    def test_round_trip_json(self) -> None:
        run = ExperimentRun.create(
            run_id="run-002",
            parameters={"batch_size": 32},
            metrics={"mrr": 0.72},
        )
        dumped = run.model_dump_json()
        restored = ExperimentRun.model_validate_json(dumped)
        assert restored == run


class TestExperimentTracker:
    def test_log_and_get_run(self, tmp_path: Path) -> None:
        log_file = tmp_path / "runs.jsonl"
        tracker = ExperimentTracker(log_file)

        assert tracker.get_run("nonexistent") is None
        assert tracker.list_runs() == []

        run1 = ExperimentRun.create(
            run_id="dummy-run-001",
            parameters={"model": "dummy", "seed": 42},
            metrics={"ndcg@10": 0.5},
        )
        run2 = ExperimentRun.create(
            run_id="dummy-run-002",
            parameters={"model": "dummy", "seed": 43},
            metrics={"ndcg@10": 0.55},
        )

        tracker.log_run(run1)
        tracker.log_run(run2)

        retrieved1 = tracker.get_run("dummy-run-001")
        assert retrieved1 is not None
        assert retrieved1.run_id == "dummy-run-001"
        assert retrieved1.parameters["seed"] == 42
        assert retrieved1.metrics["ndcg@10"] == 0.5

        retrieved2 = tracker.get_run("dummy-run-002")
        assert retrieved2 is not None
        assert retrieved2.run_id == "dummy-run-002"

        runs = tracker.list_runs()
        assert len(runs) == 2
        assert runs[0].run_id == "dummy-run-001"
        assert runs[1].run_id == "dummy-run-002"

    def test_log_file_created_if_missing(self, tmp_path: Path) -> None:
        nested_log = tmp_path / "deep" / "dir" / "runs.jsonl"
        tracker = ExperimentTracker(nested_log)
        run = ExperimentRun.create(run_id="r1", parameters={})
        tracker.log_run(run)
        assert nested_log.exists()
