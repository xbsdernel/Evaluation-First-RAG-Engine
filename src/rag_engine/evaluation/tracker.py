"""
Experiment tracking module for the RAG engine.

Requirement: Every reported metric/number in the write-up links to a run ID
and a deterministic config hash. Runs are persisted as append-only JSON Lines (JSONL).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def compute_config_hash(config: dict[str, Any]) -> str:
    """Return a deterministic 16-character SHA-256 hash of a configuration dict."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def get_git_commit() -> str | None:
    """Return the current git commit short SHA if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


class ExperimentRun(BaseModel):
    """Record of an experiment run.

    Immutable once created. Extra fields forbidden.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(..., description="Unique experiment identifier.")
    config_hash: str = Field(..., description="Deterministic SHA-256[:16] hash of parameters.")
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        description="ISO-8601 UTC timestamp.",
    )
    git_commit: str | None = Field(default=None, description="Git commit SHA at run time.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Configuration & parameters.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Metric names and values.")
    artifacts: dict[str, str] = Field(default_factory=dict, description="Artifact paths or DVC hashes.")
    notes: str = Field(default="", description="Optional human-readable notes.")

    @classmethod
    def create(
        cls,
        run_id: str,
        parameters: dict[str, Any],
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, str] | None = None,
        notes: str = "",
        git_commit: str | None = None,
    ) -> ExperimentRun:
        """Factory that derives config_hash and git_commit automatically."""
        cfg_hash = compute_config_hash(parameters)
        commit = git_commit or get_git_commit()
        return cls(
            run_id=run_id,
            config_hash=cfg_hash,
            git_commit=commit,
            parameters=parameters,
            metrics=metrics or {},
            artifacts=artifacts or {},
            notes=notes,
        )


class ExperimentTracker:
    """Append-only JSONL experiment tracker."""

    def __init__(self, log_path: Path | str = "experiments/runs.jsonl") -> None:
        self.log_path = Path(log_path)

    def log_run(self, run: ExperimentRun) -> None:
        """Append an experiment run record to the JSONL log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(run.model_dump_json() + "\n")

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Retrieve an experiment run by its run_id."""
        if not self.log_path.exists():
            return None
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("run_id") == run_id:
                    return ExperimentRun.model_validate(data)
        return None

    def list_runs(self) -> list[ExperimentRun]:
        """List all logged runs in chronological order."""
        if not self.log_path.exists():
            return []
        runs = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                runs.append(ExperimentRun.model_validate_json(line))
        return runs
