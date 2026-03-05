# IsalGraph Development Guide

## Setup

```bash
conda activate isalgraph
cd /home/mpascual/research/code/IsalGraph
python -m pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only (fast, no external deps)
python -m pytest tests/unit/ -v

# Integration tests (needs networkx, igraph)
python -m pytest tests/integration/ -v

# Property tests (hypothesis, may be slow)
python -m pytest tests/property/ -v

# With coverage
python -m pytest tests/ -v --cov=isalgraph --cov-report=term-missing
```

## Code Quality

```bash
# Lint
python -m ruff check src/ tests/
python -m ruff check --fix src/ tests/

# Format
python -m ruff format src/ tests/

# Type checking
python -m mypy src/isalgraph/
```

## Development Workflow

1. **Phase-gated development.** Phase N+1 cannot start until Phase N tests pass.
2. **Test-first.** Write the test, then implement the code.
3. **Round-trip is king.** Every code change must preserve round-trip correctness.
4. **No external deps in core.** `isalgraph.core` must never import external libraries.

## Phase Progression

| Phase | Description | Test location | Status |
|-------|-------------|---------------|--------|
| 1 | Short string round-trip | tests/unit/test_roundtrip.py | Not started |
| 2 | Massive random testing | benchmarks/random_roundtrip.py | Not started |
| 3 | Canonical string | benchmarks/canonical_invariance.py | Not started |

## Benchmarks

Benchmarks are NOT part of the test suite. Run separately:

```bash
python benchmarks/random_roundtrip.py
python benchmarks/string_length_analysis.py
python benchmarks/canonical_invariance.py
python benchmarks/levenshtein_vs_ged.py
```

## Git Conventions

- Commit messages: imperative mood, describe the "why"
- Branch naming: `feature/<description>`, `fix/<description>`, `phase-N/<description>`
- No force pushes to main
- Original code is read-only at `docs/original_code_and_files/`
