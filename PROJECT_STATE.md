# PROJECT_STATE.md

> This file is updated after every completed Issue. It reflects the **actual** state of the codebase, not aspirational progress.

---

## Completed Issues

| Issue | Title | Branch | Commit | Status |
|---|---|---|---|---|
| ISS-001 | Project scaffold: package layout, CLI stub, and Pydantic schemas | `feat/001-project-scaffold` | `b04a8a6` | ✅ Done |

---

## Current Implementation State

### What exists

```
d:\rag-engine\
├── src/rag_engine/
│   ├── __init__.py
│   ├── cli.py                  # Click CLI, rag --help, --seed, evaluate, reproduce stubs
│   ├── schemas/
│   │   ├── __init__.py         # Public re-exports
│   │   └── core.py             # Doc, Chunk, Span, Qrel, Run, ScoredDoc, TraceEvent
│   ├── corpus/__init__.py      # stub
│   ├── retrieval/__init__.py   # stub
│   ├── reranker/__init__.py    # stub
│   ├── generation/__init__.py  # stub
│   ├── evaluation/__init__.py  # stub
│   ├── serving/__init__.py     # stub
│   └── gateway/__init__.py     # stub
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── __init__.py
│       └── test_schemas.py     # 43 tests — all passing
├── pyproject.toml              # setuptools.build_meta, pydantic>=2.7, click>=8.1, rich>=13.0
├── .gitignore
├── .gitattributes              # LF normalisation
└── README.md
```

### Test status
```
43 passed in 0.79s  (Python 3.12.3, pytest-9.2.0)
```

### Python environment
- Interpreter: `C:\Users\mohamed\AppData\Local\Programs\Python\Python312\python.exe` (3.12.3)
- Package installed as editable: `pip install -e ".[dev]"`
- Scripts not on PATH — invoke via `python -m rag_engine.cli` or `python -m pytest`

---

## Architectural Decisions Made

| Decision | Detail |
|---|---|
| `src/` layout | All importable code lives under `src/rag_engine/`; prevents accidental imports from the repo root |
| `frozen=True, extra="forbid"` on all schemas | Instances are immutable and hashable; unknown fields fail loudly at the boundary |
| `_stable_hash` = SHA-256[:16] of `json.dumps(…, sort_keys=True)` | Key-order-independent, deterministic, 16-char hex ID |
| Factory classmethods (`.build()`, `.from_doc()`) | Callers never compute IDs or slice text manually; derivation is centralised |
| `Run.results` is a `tuple` (not `list`) | Immutable; rank-consecutiveness validator runs once at construction |
| `setuptools.build_meta` (not `setuptools.backends.legacy:build`) | pip 24.0 on the host does not support the new backends API |
| RNG seeding via `random.seed(seed)` in CLI | NumPy / PyTorch seeded in their respective modules when first imported (ISS-003 / ISS-025) |

---

## Dependencies

| Package | Version constraint | Why |
|---|---|---|
| `pydantic` | `>=2.7` | Schema validation, JSON Schema export for guided decoding |
| `click` | `>=8.1` | CLI framework |
| `rich` | `>=13.0` | Console output formatting |
| `pytest` | `>=8.2` (dev) | Test runner |
| `pytest-cov` | `>=5.0` (dev) | Coverage |
| `ruff` | `>=0.4` (dev) | Linting |

---

## Known Limitations

- `data/` directory exists in the repo but is untracked (DVC not yet initialised — ISS-002).
- CLI `evaluate` and `reproduce` sub-commands are stubs; they exit 0 with a message.
- No GitHub Actions workflow yet (ISS-002).
- No Docker images yet (ISS-003).
- NumPy and PyTorch RNGs are not seeded yet (added in ISS-003 / ISS-025 when those libraries are imported).

---

## Next Implementation Step

**ISS-002 · Git hygiene, DVC data versioning, and experiment tracking**

- Branch: `feat/002-dvc-experiment-tracking` (from `main` once ISS-001 is merged)
- Key tasks:
  1. Merge `feat/001-project-scaffold` → `main` and tag `v0-scaffold`
  2. `dvc init` with a local remote; DVC-track `data/raw/`
  3. Experiment tracking module (plain JSONL + config hash, or MLflow)
  4. GitHub Actions workflow: lint (ruff) + pytest on push/PR
  5. Log one dummy run with a config hash

---

*Last updated after: ISS-001 — 2026-11-17*
