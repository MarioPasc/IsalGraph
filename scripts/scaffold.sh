#!/usr/bin/env bash
# scaffold.sh -- Generate IsalGraph project directory structure
# Safe to run multiple times (idempotent)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Scaffolding IsalGraph at: $REPO_ROOT"

# --- Directory tree ---
dirs=(
    "src/isalgraph"
    "src/isalgraph/core"
    "src/isalgraph/adapters"
    "tests"
    "tests/unit"
    "tests/integration"
    "tests/property"
    "benchmarks"
    "experiments/classification"
    "experiments/generation"
    "experiments/visualization"
    "experiments/datasets"
    "docs/references"
    "scripts"
    ".claude/hooks"
)

for d in "${dirs[@]}"; do
    mkdir -p "$REPO_ROOT/$d"
    echo "  [dir] $d"
done

# --- Helper: write file only if it doesn't exist ---
write_if_new() {
    local target="$1"
    local content="$2"
    if [ ! -f "$target" ]; then
        echo "$content" > "$target"
        echo "  [new] ${target#$REPO_ROOT/}"
    else
        echo "  [skip] ${target#$REPO_ROOT/} (exists)"
    fi
}

# --- __init__.py files ---
init_files=(
    "src/isalgraph/__init__.py"
    "src/isalgraph/core/__init__.py"
    "src/isalgraph/adapters/__init__.py"
)

for f in "${init_files[@]}"; do
    write_if_new "$REPO_ROOT/$f" '"""IsalGraph -- Representation of graph structure by instruction strings."""'
done

# --- Placeholder Python files (core) ---
declare -A core_files=(
    ["src/isalgraph/types.py"]="Shared type aliases and dataclasses for IsalGraph."
    ["src/isalgraph/errors.py"]="Custom exception hierarchy for IsalGraph."
    ["src/isalgraph/core/cdll.py"]="Array-backed Circular Doubly Linked List."
    ["src/isalgraph/core/sparse_graph.py"]="Adjacency-set sparse graph representation."
    ["src/isalgraph/core/string_to_graph.py"]="IsalGraph string to SparseGraph converter."
    ["src/isalgraph/core/graph_to_string.py"]="SparseGraph to IsalGraph string converter."
    ["src/isalgraph/core/canonical.py"]="Canonical string computation (Phase 3)."
)

for file in "${!core_files[@]}"; do
    write_if_new "$REPO_ROOT/$file" "\"\"\"${core_files[$file]}\"\"\""
done

# --- Placeholder Python files (adapters) ---
declare -A adapter_files=(
    ["src/isalgraph/adapters/base.py"]="Abstract adapter interface (ABC) for graph library bridges."
    ["src/isalgraph/adapters/networkx_adapter.py"]="NetworkX <-> SparseGraph adapter."
    ["src/isalgraph/adapters/igraph_adapter.py"]="igraph <-> SparseGraph adapter."
    ["src/isalgraph/adapters/pyg_adapter.py"]="PyTorch Geometric <-> SparseGraph adapter."
)

for file in "${!adapter_files[@]}"; do
    write_if_new "$REPO_ROOT/$file" "\"\"\"${adapter_files[$file]}\"\"\""
done

# --- Test files ---
declare -A test_files=(
    ["tests/conftest.py"]="Shared pytest fixtures for IsalGraph test suite."
    ["tests/unit/test_cdll.py"]="Unit tests for CircularDoublyLinkedList."
    ["tests/unit/test_sparse_graph.py"]="Unit tests for SparseGraph."
    ["tests/unit/test_string_to_graph.py"]="Unit tests for StringToGraph."
    ["tests/unit/test_graph_to_string.py"]="Unit tests for GraphToString."
    ["tests/unit/test_roundtrip.py"]="Round-trip correctness tests (Phase 1 + Phase 2)."
    ["tests/integration/test_networkx_adapter.py"]="Integration tests for NetworkX adapter."
    ["tests/integration/test_igraph_adapter.py"]="Integration tests for igraph adapter."
    ["tests/integration/test_pyg_adapter.py"]="Integration tests for PyG adapter."
    ["tests/property/test_roundtrip_property.py"]="Hypothesis-based property tests for round-trip."
)

for file in "${!test_files[@]}"; do
    write_if_new "$REPO_ROOT/$file" "\"\"\"${test_files[$file]}\"\"\""
done

# --- Benchmark files ---
declare -A bench_files=(
    ["benchmarks/random_roundtrip.py"]="Phase 2 -- massive random string round-trip testing."
    ["benchmarks/canonical_invariance.py"]="Phase 3 -- isomorphism invariance checks."
    ["benchmarks/string_length_analysis.py"]="Empirical string length vs graph size analysis."
    ["benchmarks/levenshtein_vs_ged.py"]="Levenshtein distance vs graph edit distance correlation."
)

for file in "${!bench_files[@]}"; do
    write_if_new "$REPO_ROOT/$file" "\"\"\"${bench_files[$file]}\"\"\""
done

# --- pyproject.toml ---
target="$REPO_ROOT/pyproject.toml"
if [ ! -f "$target" ]; then
    cat > "$target" << 'TOMLEOF'
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
name = "isalgraph"
version = "0.1.0"
description = "Graph representation by instruction strings"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Ezequiel Lopez-Rubio", email = "ezeqlr@lcc.uma.es"},
    {name = "Mario Pascual Gonzalez"},
]
# Core has zero dependencies
dependencies = []

[project.optional-dependencies]
networkx = ["networkx>=3.0"]
igraph = ["igraph>=0.11"]
pyg = ["torch>=2.0", "torch-geometric>=2.4"]
viz = ["matplotlib>=3.7", "python-pptx>=0.6.21"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "hypothesis>=6.0",
    "ruff>=0.4",
    "mypy>=1.0",
]
all = ["isalgraph[networkx,igraph,viz,dev]"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "unit: Unit tests (no external deps)",
    "integration: Integration tests (external graph libs)",
    "property: Property-based tests (hypothesis)",
    "slow: Long-running tests",
]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "ANN", "B", "SIM"]
TOMLEOF
    echo "  [new] pyproject.toml"
else
    echo "  [skip] pyproject.toml (exists)"
fi

# --- LICENSE (MIT) ---
target="$REPO_ROOT/LICENSE"
if [ ! -f "$target" ]; then
    YEAR=$(date +%Y)
    cat > "$target" << LICEOF
MIT License

Copyright (c) $YEAR Ezequiel Lopez-Rubio, Mario Pascual Gonzalez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICEOF
    echo "  [new] LICENSE"
else
    echo "  [skip] LICENSE (exists)"
fi

echo ""
echo "Scaffold complete. Next steps:"
echo "  1. Run: cd $REPO_ROOT && ~/.conda/envs/neuromf/bin/python -m pip install -e '.[dev]'"
echo "  2. Migrate code from docs/original_code_and_files/ to src/isalgraph/core/"
echo "  3. Fix known bugs (see .claude/CLAUDE.md critical invariants section)"
