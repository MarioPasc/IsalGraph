# IsalGraph: Agent Context Document

## 1. Overview

**IsalGraph** is a method for representing the structure of a graph as a string of
simple instructions. The tool is being developed by Ezequiel López-Rubio (University
of Málaga) and Mario Pascual González. It extends a prior tool called IsalChem
(designed for molecular graphs) to arbitrary graphs with unlabeled, indistinguishable
nodes and no degree restrictions.

The codebase implements two core conversions:

- **StringToGraph**: Converts an IsalGraph instruction string into a graph.
- **GraphToString**: Converts a graph back into an IsalGraph instruction string.

The primary debugging objective is to ensure **round-trip correctness**: for any valid
IsalGraph string $w$, the graph produced by StringToGraph and the graph produced by
running GraphToString on that output must be **isomorphic**.

The preprint `2512_10429v2.pdf` describes the original (adjacency-matrix-based)
method. The file `enhancedideaezequiel.pdf` describes the new IsalGraph variant
implemented in this codebase. Both are available in the project directory under
`docs/references/`.

---

## 2. Mathematical Foundation

### 2.1 Graph Definition

Let $G = (V, E)$ be a graph where $V = \{v_0, \ldots, v_{N-1}\}$ is the vertex set and
$E \subseteq V \times V$ is the edge set. The graph may be directed or undirected. Nodes
are unlabeled (indistinguishable), meaning the representation must ultimately be
invariant under graph isomorphism.

### 2.2 Data Structures

The algorithm maintains three structures during both conversions:

1. **A SparseGraph**: Adjacency-set representation. Nodes are identified by contiguous
   integer IDs $\{0, 1, \ldots\}$. Supports $O(1)$ average edge insertion and lookup.

2. **A Circular Doubly Linked List (CDLL)**: Array-backed CDLL that stores graph node
   indices as payloads. Each CDLL node has its own internal index (allocated from a
   free list) and stores a graph node index as its `_data` value. **These two index
   spaces (CDLL indices vs. graph node indices) are distinct and must not be
   conflated.**

3. **Two pointers** (primary and secondary): These are **CDLL node indices** (not
   graph node indices). To obtain the graph node that a pointer refers to, one must
   call `cdll.get_value(pointer)`.

### 2.3 Instruction Set

The IsalGraph language is defined over the alphabet
$\Sigma = \{N, n, P, p, V, v, C, c, W\}$.

| Instruction | Semantics |
|---|---|
| `N` | Move primary pointer to `cdll.next_node(primary_ptr)` |
| `P` | Move primary pointer to `cdll.prev_node(primary_ptr)` |
| `n` | Move secondary pointer to `cdll.next_node(secondary_ptr)` |
| `p` | Move secondary pointer to `cdll.prev_node(secondary_ptr)` |
| `V` | Create a new graph node, add edge from `cdll.get_value(primary_ptr)` to new node, insert new node into CDLL after the CDLL node pointed to by `primary_ptr` |
| `v` | Create a new graph node, add edge from `cdll.get_value(secondary_ptr)` to new node, insert new node into CDLL after the CDLL node pointed to by `secondary_ptr` |
| `C` | Add edge from `cdll.get_value(primary_ptr)` to `cdll.get_value(secondary_ptr)` |
| `c` | Add edge from `cdll.get_value(secondary_ptr)` to `cdll.get_value(primary_ptr)` |
| `W` | No operation |

**Critical semantic note on `V` and `v`:** The edge connects the graph node
referenced by the pointer to the new graph node. The new CDLL node is inserted after
the pointer's CDLL node. The pointer itself does NOT move after a `V`/`v` instruction.

**For undirected graphs:** `C` and `c` have identical effect because `add_edge`
inserts both $(u,v)$ and $(v,u)$.

### 2.4 Initial State

Both conversions begin from the same initial state:

- The graph contains exactly **one node** (node 0).
- The CDLL contains exactly **one node** whose payload is graph node 0.
- Both primary and secondary pointers point to this single CDLL node.

### 2.5 Round-Trip Correctness Property

Let $\text{S2G}: \Sigma^* \to \mathcal{G}$ denote StringToGraph and
$\text{G2S}: \mathcal{G} \times V \to \Sigma^*$ denote GraphToString (parameterized by
a starting node). The required property is:

$$\forall w \in \Sigma^*,\quad \text{S2G}(w) \cong \text{S2G}(\text{G2S}(\text{S2G}(w), v_0))$$

where $\cong$ denotes graph isomorphism and $v_0$ is any valid starting node. Note that
we do **not** require $w = \text{G2S}(\text{S2G}(w), v_0)$ — string equality is a stronger
condition that relates to canonical string selection (a later development step).

### 2.6 GraphToString Greedy Search Strategy

The GraphToString algorithm reconstructs a graph incrementally, seeking at each step
the pair of pointer displacements $(a, b) \in \{-M, \ldots, M\}^2$ (where $M$ is the
current node count) that requires the fewest total pointer movements $|a| + |b|$ and
enables either:

1. **Node insertion** (`V`/`v`): The pointer (after displacement) points to a graph
   node that has a neighbor in the input graph which has not yet been added to the
   output graph.
2. **Edge insertion** (`C`/`c`): After displacing both pointers, there exists an edge
   in the input graph between the nodes they reference that has not yet been added to
   the output graph.

Pairs are enumerated in increasing order of $|a| + |b|$ (implemented by
`generate_pairs_sorted_by_sum`). The algorithm stops searching as soon as the first
valid operation is found and emits the corresponding instruction sequence.

**Note on `generate_pairs_sorted_by_sum`:** The current implementation sorts by
$a + b$ (algebraic sum), but the design document specifies sorting by $|a| + |b|$
(total displacement cost). These are different orderings. The implementation should
match the specification: the number of pointer movement instructions emitted is
$|a| + |b|$, so minimizing this quantity is what produces shorter strings.

### 2.7 Index Space Mapping (GraphToString)

GraphToString maintains two dictionaries:

- `_i2o: Dict[int, int]` — maps input graph node indices to output graph node indices.
- `_o2i: Dict[int, int]` — maps output graph node indices to input graph node indices.

These are essential because the output graph is built incrementally with its own node
numbering, while the input graph has a fixed numbering. All neighbor lookups must
be translated through these dictionaries.

### 2.8 Future Step: Canonical String

Once round-trip is correct, the canonical string for a graph $G$ with $N$ nodes is:

$$w_G^* = \text{lexmin}\left\{ w \in \arg\min_{v \in V} |\text{G2S}(G, v)| \right\}$$

That is: run GraphToString starting from every node $v \in V$, collect all resulting
strings, filter to keep only those of minimum length, and among those select the
lexicographically smallest. This is $O(N)$ calls to GraphToString.

If implemented correctly, the canonical string is a **complete graph invariant**:
$w_G^* = w_{G'}^*$ if and only if $G \cong G'$.

---

## 3. Verification Strategy

### 3.1 Phase 1: Short Strings

Test round-trip correctness for short, manually inspectable strings. The current
`main.py` uses `my_string = "vvc"` with `directed = False`. Suggested test cases:

**Single-instruction strings:** `"V"`, `"v"` (these produce a 2-node, 1-edge graph)

**Two-instruction strings:** `"VV"`, `"Vv"`, `"vV"`, `"vv"`, `"VC"`, `"vC"`, `"Vc"`,
`"NV"`, `"nv"`, `"PV"`, `"pv"`

**Three-instruction strings:** `"VNV"`, `"VnC"`, `"vNv"`, `"vvc"`, `"VVN"`, `"VNC"`

For each string $w$:
1. Run `StringToGraph(w).run()` → obtain graph $G_1$.
2. Run `GraphToString(G_1).run(initial_node=0)` → obtain string $w'$ and graph $G_2$.
3. Assert `G_1.is_isomorphic(G_2)` returns `True`.
4. Optionally verify edge counts and node counts match.

### 3.2 Phase 2: Massive Random Testing

Generate hundreds of random valid IsalGraph strings and verify round-trip for each.
A random string generator should:

1. Choose a target length $L \sim \text{Uniform}(1, L_{\max})$.
2. At each position, choose an instruction from $\Sigma$ with appropriate constraints
   (e.g., `C`/`c` are only meaningful when the graph has $\geq 2$ nodes; movement
   instructions should not dominate excessively).
3. Run round-trip and assert isomorphism.

Test both `directed=True` and `directed=False` configurations.

### 3.3 Phase 3: Canonical String (post-debugging)

Once round-trip is verified, implement canonical string computation inside
`src/isalgraph/core/canonical.py`.

Verification: for pairs of isomorphic graphs (e.g., generated by random relabeling),
assert that `canonical_string(G) == canonical_string(G')`.

---

## 4. Package Architecture and Code Organization

### 4.1 Repository Layout

The project is structured as a proper Python package using the `src/` layout
convention (recommended by PyPA and the Python Packaging User Guide). The repository
serves a dual purpose: (1) distributable `isalgraph` library, and (2) reproducible
research experiments and benchmarks.

```
isalgraph/                              # Repository root
│
├── src/
│   └── isalgraph/                      # Installable Python package
│       ├── __init__.py                 # Package metadata, public API re-exports
│       ├── types.py                    # Shared type aliases and dataclasses
│       ├── errors.py                   # Custom exception hierarchy
│       │
│       ├── core/                       # Core math implementation (zero external deps)
│       │   ├── __init__.py
│       │   ├── cdll.py                 # CircularDoublyLinkedList
│       │   ├── sparse_graph.py         # SparseGraph (internal representation)
│       │   ├── string_to_graph.py      # StringToGraph converter
│       │   ├── graph_to_string.py      # GraphToString converter
│       │   └── canonical.py            # Canonical string computation (Phase 3)
│       │
│       └── adapters/                   # Bridges to external graph libraries
│           ├── __init__.py
│           ├── base.py                 # Abstract adapter interface (ABC)
│           ├── networkx_adapter.py     # NetworkX <-> SparseGraph
│           ├── igraph_adapter.py       # igraph <-> SparseGraph
│           └── pyg_adapter.py          # PyTorch Geometric <-> SparseGraph
│
├── tests/                              # pytest test suite
│   ├── conftest.py                     # Shared fixtures (sample graphs, generators)
│   ├── unit/
│   │   ├── test_cdll.py
│   │   ├── test_sparse_graph.py
│   │   ├── test_string_to_graph.py
│   │   ├── test_graph_to_string.py
│   │   └── test_roundtrip.py           # Phase 1 + Phase 2 round-trip tests
│   ├── integration/
│   │   ├── test_networkx_adapter.py
│   │   ├── test_igraph_adapter.py
│   │   └── test_pyg_adapter.py
│   └── property/                       # Hypothesis-based property tests (optional)
│       └── test_roundtrip_property.py
│
├── benchmarks/                         # Performance and correctness at scale
│   ├── random_roundtrip.py             # Phase 2: massive random string testing
│   ├── canonical_invariance.py         # Phase 3: isomorphism invariance checks
│   ├── string_length_analysis.py       # Empirical string length vs graph size
│   └── levenshtein_vs_ged.py           # Levenshtein distance vs graph edit distance
│
├── experiments/                        # Research experiments (not part of package)
│   ├── classification/                 # Transformer-based graph classification
│   ├── generation/                     # Autoregressive graph generation
│   ├── visualization/                  # Trace visualization (PPTX, matplotlib)
│   │   └── plotgraphlist.py            # Current PPTX trace renderer (moved here)
│   └── datasets/                       # Dataset generation scripts
│
├── docs/
│   ├── references/
│   │   ├── 2512_10429v2.pdf            # Original preprint
│   │   └── enhancedideaezequiel.pdf    # IsalGraph design notes
│   └── AGENT_CONTEXT.md               # This document
│
├── pyproject.toml                      # Package metadata, build config, deps
├── README.md
├── LICENSE
└── .gitignore
```

### 4.2 Dependency Separation

The package follows a strict dependency layering:

```
┌─────────────────────────────────────────────────────┐
│                    experiments/                       │  torch, torch_geometric,
│                    benchmarks/                        │  matplotlib, ...
├─────────────────────────────────────────────────────┤
│               isalgraph.adapters                     │  networkx, igraph,
│                                                      │  torch_geometric
├─────────────────────────────────────────────────────┤
│               isalgraph.core                         │  ZERO external dependencies
│               isalgraph.types                        │  (only Python stdlib + typing)
│               isalgraph.errors                       │
└─────────────────────────────────────────────────────┘
```

**`isalgraph.core`** must have **zero external dependencies**. It relies only on the
Python standard library (typing, collections, copy, etc.). This is critical: the core
mathematical implementation must be self-contained, portable, and testable in
isolation. The current codebase already satisfies this — it must remain so.

**`isalgraph.adapters`** introduces optional dependencies on external graph libraries.
Each adapter is importable independently, so a user who only has NetworkX installed
does not need igraph or PyG. This is handled via optional dependency groups in
`pyproject.toml`.

**`experiments/` and `benchmarks/`** may depend on anything (torch, matplotlib,
python-pptx, scipy, etc.) but are not part of the installable package.

### 4.3 `pyproject.toml` Skeleton

```toml
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
    {name = "Ezequiel López-Rubio", email = "ezeqlr@lcc.uma.es"},
    {name = "Mario Pascual González"},
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

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true

[tool.ruff]
target-version = "py310"
line-length = 100
```

### 4.4 Migration Plan from Current Flat Layout

The agent should restructure the existing code as follows:

| Current file                    | New location                                  |
|---------------------------------|-----------------------------------------------|
| `circulardoublylinkedlist.py`   | `src/isalgraph/core/cdll.py`                  |
| `sparsegraph.py`                | `src/isalgraph/core/sparse_graph.py`          |
| `stringtograph.py`              | `src/isalgraph/core/string_to_graph.py`       |
| `graphtostring.py`              | `src/isalgraph/core/graph_to_string.py`       |
| `plotgraphlist.py`              | `experiments/visualization/plotgraphlist.py`   |
| `main.py`                       | `tests/unit/test_roundtrip.py` (rewritten)    |
| `2512_10429v2.pdf`              | `docs/references/2512_10429v2.pdf`            |
| `enhancedideaezequiel.pdf`      | `docs/references/enhancedideaezequiel.pdf`    |

All internal imports must be updated to use the new package paths (e.g.,
`from isalgraph.core.sparse_graph import SparseGraph`).

---

## 5. External Library Selection and Adapter Design

### 5.1 Library Recommendations

Three external graph libraries are recommended, each serving a distinct role:

**NetworkX** (primary adapter, testing, visualization)
- Pure Python, the universal standard for graph manipulation in scientific computing.
- Built-in isomorphism testing via VF2 (`nx.is_isomorphic`), which should be used to
  cross-validate IsalGraph's own `is_isomorphic` and for round-trip testing.
- Rich graph generators: Erdos-Renyi (`gnp_random_graph`), Barabasi-Albert
  (`barabasi_albert_graph`), Watts-Strogatz (`watts_strogatz_graph`), grid graphs,
  tree generators, etc. Essential for Phase 2 testing.
- Native I/O for GML, GraphML, edge lists, adjacency lists, GEXF.
- Tight matplotlib integration for visualization.
- Reference: Hagberg, A., Schult, D., & Swart, P. (2008). "Exploring network
  structure, dynamics, and function using NetworkX." SciPy.

**igraph (python-igraph)** (performance benchmarking)
- C core with Python bindings, orders of magnitude faster than NetworkX for large
  graphs (>10k nodes).
- Isomorphism via Bliss and VF2 algorithms.
- Useful for benchmarking IsalGraph on large sparse graphs where NetworkX would be
  too slow.
- Reference: Csardi, G. & Nepusz, T. (2006). "The igraph software package for
  complex network research." InterJournal Complex Systems.

**PyTorch Geometric (PyG)** (downstream ML experiments)
- The standard library for GNN research in PyTorch.
- Provides access to benchmark graph datasets: TUDataset (MUTAG, PROTEINS, NCI1,
  COLLAB, IMDB, etc.), OGB, Planetoid.
- The graph classification experiments from the preprint (Section 3) should use PyG
  datasets for reproducibility and comparability with the GNN literature.
- Reference: Fey, M. & Lenssen, J. E. (2019). "Fast Graph Representation Learning
  with PyTorch Geometric." ICLR Workshop on Representation Learning on Graphs and
  Manifolds.

### 5.2 Adapter Architecture (Abstract Bridge Pattern)

The adapter layer translates between external library graph objects and IsalGraph's
internal `SparseGraph`. The design follows the **Bridge pattern** (Gamma et al., 1994),
with an abstract base class defining the interface:

```
                    ┌──────────────────────┐
                    │   External Graph     │
                    │   (nx.Graph,         │
                    │    igraph.Graph,     │
                    │    PyG Data)         │
                    └─────────┬────────────┘
                              │
                    ┌─────────▼────────────┐
                    │   GraphAdapter (ABC)  │
                    │                      │
                    │  from_external(G) ──►│── SparseGraph
                    │  to_external(sg)  ◄──│── SparseGraph
                    │  from_isalgraph(w) ─►│── External Graph
                    │  to_isalgraph(G)  ◄──│── str
                    └──────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──┐  ┌────────▼───┐  ┌────────▼───┐
    │ NetworkX   │  │  igraph     │  │  PyG       │
    │ Adapter    │  │  Adapter    │  │  Adapter   │
    └────────────┘  └────────────┘  └────────────┘
```

The abstract interface in `adapters/base.py`:

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from isalgraph.core.sparse_graph import SparseGraph

# Generic type for the external graph object
T = TypeVar("T")

class GraphAdapter(ABC, Generic[T]):
    """Abstract bridge between external graph libraries and IsalGraph core."""

    @abstractmethod
    def from_external(self, graph: T, directed: bool) -> SparseGraph:
        """Convert an external graph object to IsalGraph's SparseGraph."""
        ...

    @abstractmethod
    def to_external(self, sparse_graph: SparseGraph) -> T:
        """Convert IsalGraph's SparseGraph to an external graph object."""
        ...

    def to_isalgraph_string(self, graph: T, directed: bool,
                            initial_node: int = 0) -> str:
        """Convert an external graph to its IsalGraph string."""
        sg = self.from_external(graph, directed)
        gts = GraphToString(sg)
        string, _ = gts.run(initial_node=initial_node)
        return string

    def from_isalgraph_string(self, string: str, directed: bool) -> T:
        """Convert an IsalGraph string to an external graph object."""
        stg = StringToGraph(string, directed)
        sg, _ = stg.run()
        return self.to_external(sg)
```

**Key design principles:**

1. **Core never imports external libraries.** The adapters import their respective
   external library, but `isalgraph.core` never imports any external library.

2. **Node mapping preservation.** Each adapter must document and preserve the mapping
   between external node identifiers and SparseGraph integer indices. For NetworkX,
   this means maintaining a bidirectional mapping between arbitrary hashable node
   labels and contiguous integer IDs.

3. **Edge attribute handling.** IsalGraph represents pure structure (no edge weights,
   no node features). Adapters must clearly document that attributes are stripped
   during `from_external` and not restored during `to_external`. If the user needs
   attributed graphs, the adapter returns the structural skeleton and the user is
   responsible for re-attaching attributes using the node mapping.

4. **Lazy imports with clear errors.** Each adapter module imports its external library
   at the top of the file. If the library is not installed, the import fails with a
   clear `ImportError` message guiding the user to install the optional dependency
   (e.g., `pip install isalgraph[networkx]`).

### 5.3 NetworkX Adapter Sketch

```python
import networkx as nx
from isalgraph.adapters.base import GraphAdapter
from isalgraph.core.sparse_graph import SparseGraph

class NetworkXAdapter(GraphAdapter[nx.Graph]):
    """Bridge between NetworkX graphs and IsalGraph SparseGraph."""

    def from_external(self, graph: nx.Graph, directed: bool) -> SparseGraph:
        # Map nx node labels to contiguous integers
        node_list = sorted(graph.nodes())
        label_to_id = {label: i for i, label in enumerate(node_list)}

        sg = SparseGraph(max_nodes=len(node_list), directed_graph=directed)
        for _ in node_list:
            sg.add_node()
        for u, v in graph.edges():
            sg.add_edge(label_to_id[u], label_to_id[v])

        # Store mapping for later recovery
        self._label_to_id = label_to_id
        self._id_to_label = {v: k for k, v in label_to_id.items()}
        return sg

    def to_external(self, sparse_graph: SparseGraph) -> nx.Graph:
        G = nx.DiGraph() if sparse_graph.directed() else nx.Graph()
        for i in range(sparse_graph.node_count()):
            G.add_node(i)
        for u in range(sparse_graph.node_count()):
            for v in sparse_graph.neighbors(u):
                if sparse_graph.directed() or u < v:
                    G.add_edge(u, v)
        return G
```

### 5.4 Testing with External Libraries

The NetworkX adapter enables powerful testing strategies:

- **Round-trip via NetworkX isomorphism:** Convert IsalGraph string -> SparseGraph ->
  NetworkX graph $G_1$. Then GraphToString -> StringToGraph -> NetworkX graph $G_2$.
  Assert `nx.is_isomorphic(G_1, G_2)`. This cross-validates both IsalGraph's own
  `is_isomorphic` and the adapter correctness.

- **Diverse graph families:** Use NetworkX generators to produce graphs from known
  families (random, scale-free, small-world, trees, cycles, complete, bipartite,
  grid) and verify round-trip for each.

- **Benchmark datasets via PyG:** Load TUDataset graphs (MUTAG, PROTEINS, etc.),
  convert to SparseGraph via the PyG adapter, run GraphToString, and verify
  round-trip. This tests IsalGraph on real-world graph structures.

---

## 6. Key Contracts (Invariants)

- **CDLL indices != graph node indices.** CDLL nodes are allocated from an internal
  free list. Their payloads (accessed via `get_value`) are graph node indices.
- **Pointers are CDLL indices.** `primary_ptr` and `secondary_ptr` are always CDLL
  node indices. To get the graph node: `cdll.get_value(primary_ptr)`.
- **`insert_after(node, value)`** expects `node` to be a **CDLL node index**, and
  `value` to be the payload (graph node index) for the new CDLL node.
- **`SparseGraph.add_edge(source, target)`** expects graph node indices.
- **`SparseGraph.neighbors(node)`** returns a set of graph node indices.

---

## 7. Key References

- Lopez-Rubio, E. (2025). "Representation of the structure of graphs by sequences
  of instructions." arXiv:2512.10429v2.
- Garey, M. R., & Johnson, D. S. (1979). "Computers and Intractability: A Guide
  to the Theory of NP-Completeness." (bandwidth minimization, NP-hardness)
- Zeng, Z. et al. (2009). "Comparing Stars: On Approximating Graph Edit Distance."
  PVLDB. (graph edit distance complexity)
- You, J. et al. (2018). "GraphRNN: Generating Realistic Graphs with an
  Auto-Regressive Model." ICML. (autoregressive graph generation)
- Weisfeiler, B. & Leman, A. (1968). "The reduction of a graph to canonical form
  and the algebra which appears therein." (graph isomorphism testing)
- Hagberg, A., Schult, D., & Swart, P. (2008). "Exploring network structure,
  dynamics, and function using NetworkX." SciPy.
- Csardi, G. & Nepusz, T. (2006). "The igraph software package for complex network
  research." InterJournal Complex Systems.
- Fey, M. & Lenssen, J. E. (2019). "Fast Graph Representation Learning with
  PyTorch Geometric." ICLR Workshop on Representation Learning on Graphs and
  Manifolds.
- Gamma, E. et al. (1994). "Design Patterns: Elements of Reusable Object-Oriented
  Software." Addison-Wesley. (Bridge pattern)
