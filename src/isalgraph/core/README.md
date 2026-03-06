# isalgraph.core -- Complete Technical Reference

> **Zero external dependencies.** This package uses only the Python standard library.
> All mathematical foundations derive from Lopez-Rubio (2025),
> "Representation of the structure of graphs by sequences of instructions,"
> arXiv:2512.10429v2, and the companion design document `docs/references/Idea.pdf`.

---

## 1. Scientific Context and Motivation

### 1.1 The Problem

Deep learning language models (Transformers) are designed for sequential text.
Graphs -- the natural representation for social networks, molecules, knowledge
bases, and connectomics -- are not directly amenable to these models.
The dominant approach, Graph Neural Networks (GNNs), learns embeddings via
message passing (Gilmer et al., 2017; Kipf & Welling, 2017; Xu et al., 2019),
but these are bounded by the 1-WL isomorphism test and cannot easily leverage
the massive pre-trained language model ecosystem.

Lopez-Rubio (2025) proposes a fundamentally different strategy: represent the
*structure* of a graph as a string of simple instructions, so that any
Transformer architecture can process graphs as text. The transformation is:

- **Reversible**: given a graph, produce a string; given a string, reconstruct
  the graph (up to isomorphism).
- **Compact**: for large sparse graphs, the string length is much smaller than
  the binary adjacency matrix representation (N^2 bits).
- **Topology-preserving**: local changes in the graph produce local changes
  in the string, as measured by Levenshtein distance.

### 1.2 Relationship to the Preprint

The preprint (arXiv:2512.10429v2) describes the **matrix-pointer method**: a
single pointer traverses the N x N adjacency matrix, with five instructions
{U, D, L, R, E} (up, down, left, right, edge-insert). This codebase implements
the **IsalGraph variant** described in `Idea.pdf`, which replaces the matrix
pointer with:

- A **circular doubly linked list (CDLL)** containing all inserted graph nodes.
- **Two pointers** (primary and secondary) that traverse the CDLL.
- A richer **9-instruction alphabet** {N, n, P, p, V, v, C, c, W} that enables
  simultaneous node insertion and edge creation.

The IsalGraph variant was designed by Lopez-Rubio for general graphs with
unlabeled, indistinguishable nodes and no degree restrictions. It extends
the earlier IsalChem tool (designed for molecular graphs with valence
constraints) to arbitrary graphs.

### 1.3 Authors and Affiliation

- **Ezequiel Lopez-Rubio** (supervisor) -- Department of Computer Languages
  and Computer Science, University of Malaga. ezeqlr@lcc.uma.es
- **Mario Pascual Gonzalez** (PhD student) -- University of Malaga.

---

## 2. Mathematical Foundation

### 2.1 Graph Definition

Let G = (V, E) be a graph where V = {v_0, ..., v_{N-1}} is the vertex set and
E is the edge set. The graph may be directed or undirected. Nodes are
**unlabeled** (indistinguishable), meaning the string representation must
ultimately be invariant under graph isomorphism.

The adjacency matrix M_G has N x N elements (Lopez-Rubio, 2025, Eq. 1):

    M_G(i,j) = { 0  if (v_i, v_j) not in E
               { 1  if (v_i, v_j) in E

For undirected graphs M_G is symmetric. The standard binary string
representation flattens M_G row by row into a string of N^2 symbols
(Lopez-Rubio, 2025, Eq. 2):

    B_G = M_G(1,1) ... M_G(1,N) M_G(2,1) ... M_G(N,N)

### 2.2 The Matrix-Pointer Method (Preprint, Section 2.1)

Lopez-Rubio (2025) defines a string w over {U, D, L, R, E}* that builds M_G
from the null matrix. A pointer p = (p_1, p_2) indexes a cell of the matrix.
The instructions are:

| Instruction | Semantics (Lopez-Rubio, 2025, Eqs. 3-7) |
|-------------|------------------------------------------|
| U | p' = (p_1 - 1, p_2) if p_1 > 1, else p (boundary clamp) |
| D | p' = (p_1 + 1, p_2) if p_1 < N, else p |
| L | p' = (p_1, p_2 - 1) if p_2 > 1, else p |
| R | p' = (p_1, p_2 + 1) if p_2 < N, else p |
| E | Set M'(p_1, p_2) = 1 (and M'(p_2, p_1) = 1 if undirected) |

The canonical string I_G for this method is produced by a greedy algorithm
that iterates over nonzero cells of M_G in order of Manhattan distance from
the current pointer position, emitting movement instructions followed by E
for each cell (Lopez-Rubio, 2025, Section 2.1, steps 1-9).

### 2.3 String Length Asymptotics (Preprint, Section 2.2 and Appendix A)

Lopez-Rubio (2025) derives the expected length of the matrix-pointer canonical
string for large sparse random graphs. Let rho = |E| / N^2 be the edge density.
By modeling active cells as a homogeneous Poisson point process of intensity
lambda = rho on R^2, the mean nearest-neighbor L1 distance from a typical
active cell satisfies (Lopez-Rubio, 2025, Appendix A):

The survival function of the nearest-neighbor L1 distance S is:

    P(S > r) = exp(-2 * lambda * r^2),  r >= 0

The probability density is:

    f_S(r) = 4 * lambda * r * exp(-2 * lambda * r^2),  r >= 0

The mean nearest-neighbor distance is (Lopez-Rubio, 2025, Eq. 8):

    E[delta] ~ (sqrt(pi) / (2*sqrt(2))) * rho^{-1/2}  as N -> inf, rho -> 0

Each active cell requires approximately E[delta] + 1 instructions (delta
movement instructions plus one E instruction). With N^2 * rho active cells
total, the expected string length is (Lopez-Rubio, 2025, Eq. 9):

    E[|I_G|] ~ (sqrt(pi) / (2*sqrt(2))) * N^2 * sqrt(rho)  as N -> inf, rho -> 0

The upper bound is 2*N^2 - 1 (complete directed graph: every cell is active).
For large sparse graphs (rho -> 0), E[|I_G|] << N^2, achieving significant
compression over the binary representation.

**Important**: Equation 9 applies to the *matrix-pointer* method of the preprint,
NOT to the CDLL-based IsalGraph method implemented here. Our CDLL method uses a
different instruction set and traversal strategy, so its scaling may differ.
The benchmark `string_length_analysis` empirically characterizes the CDLL
method's scaling and uses Eq. 9 as a reference baseline.

### 2.4 Topological Properties (Preprint, Section 2.3)

Lopez-Rubio (2025) proves that the string representation preserves locality:
a minimal change in the adjacency matrix (flipping one cell) corresponds to
a bounded modification of the representing string. This follows from the
triangle inequality on Manhattan distance:

- If M(i,j) = 0 (cell currently absent), a detour of 2*delta + 1 instructions
  can be inserted to set it to 1 and return.
- If M(i,j) = 1 (cell currently present), the movement substring from the
  previous E instruction to the next can be replaced by a shortest-path
  movement, which by the triangle inequality is no longer.

Consequently, similar adjacency matrices map to similar strings as measured
by Levenshtein edit distance. This is a fundamental advantage for processing
with deep learning language models: local structural patterns in the graph
produce local patterns in the string.

### 2.5 The IsalGraph CDLL Method (Idea.pdf)

The IsalGraph variant replaces the matrix pointer with three data structures:

1. **SparseGraph**: An adjacency-set representation where nodes are contiguous
   integer IDs {0, 1, 2, ...}. Supports O(1) average edge insertion and
   membership testing.

2. **Circular Doubly Linked List (CDLL)**: An array-backed CDLL that stores
   graph node indices as payloads. Each CDLL node has its own internal index
   (allocated from a free-list stack) and stores a graph node index as its
   `_data` value.

   **Critical**: The CDLL index space and the graph node index space are
   **distinct**. A CDLL node at internal index `k` stores a graph node index
   `cdll.get_value(k)` as its payload. These two integers are NOT the same
   in general (though they may coincide when no CDLL nodes have been removed).

3. **Two pointers** (primary and secondary): These are **CDLL node indices**
   (not graph node indices). To obtain the graph node that a pointer refers to,
   one must call `cdll.get_value(pointer)`.

### 2.6 Instruction Set (Alphabet Sigma)

The IsalGraph instruction alphabet is Sigma = {N, n, P, p, V, v, C, c, W}.
All 9 instructions operate on the shared state (SparseGraph, CDLL, primary
pointer, secondary pointer):

| Instr | Full Semantics |
|-------|----------------|
| **N** | Move primary pointer to `cdll.next_node(primary_ptr)`. |
| **P** | Move primary pointer to `cdll.prev_node(primary_ptr)`. |
| **n** | Move secondary pointer to `cdll.next_node(secondary_ptr)`. |
| **p** | Move secondary pointer to `cdll.prev_node(secondary_ptr)`. |
| **V** | (1) Create new graph node `u`. (2) Add edge from `cdll.get_value(primary_ptr)` to `u`. (3) Insert `u` into CDLL after the CDLL node at `primary_ptr`. (4) The primary pointer does **NOT** move. |
| **v** | Same as V but using the secondary pointer: edge from `cdll.get_value(secondary_ptr)` to `u`, insert after `secondary_ptr`. The secondary pointer does **NOT** move. |
| **C** | Add edge from `cdll.get_value(primary_ptr)` to `cdll.get_value(secondary_ptr)`. For undirected graphs, the reverse edge is also added. |
| **c** | Add edge from `cdll.get_value(secondary_ptr)` to `cdll.get_value(primary_ptr)`. For undirected graphs, C and c have identical effect. For directed graphs, c creates the reverse direction of C. |
| **W** | No operation. |

**Pointer immobility after V/v**: This is a critical semantic detail from
Idea.pdf. After V (or v), the pointer that triggered the insertion stays at
its current CDLL node. The new node is inserted *after* it in the CDLL, but
the pointer does not advance to the new node. This means subsequent V
instructions from the same pointer position create a chain of nodes all
adjacent to the same source node (a star pattern), while N/P (or n/p)
instructions are needed to move to the newly inserted node.

**Why two pointers?** (from Idea.pdf): A single pointer suffices for creating
nodes and edges from the current position, but edge-only instructions (C/c)
require specifying *two* endpoints. With two independently movable pointers,
the algorithm can connect any pair of already-inserted nodes by navigating
each pointer to the desired endpoint. This is essential for graphs that are
not trees (i.e., graphs with cycles or multiple connectivity patterns).

### 2.7 Initial State

Both conversions (StringToGraph and GraphToString) begin from the same state:

- The graph contains exactly **one node** (node 0).
- The CDLL contains exactly **one node** whose payload is graph node 0.
- Both primary and secondary pointers point to this single CDLL node.

This is defined in `Idea.pdf`: "The initial state contains one node, no edges,
and the doubly linked circular list only contains the node."

### 2.8 StringToGraph (S2G)

Given a string w over Sigma*, the StringToGraph converter executes each
instruction left-to-right, mutating the shared state. The maximum number of
nodes is pre-computed as 1 + count('V') + count('v') (each V/v creates exactly
one node). The result is a SparseGraph.

**Code**: `string_to_graph.py`, class `StringToGraph`.

### 2.9 GraphToString (G2S) -- Greedy Algorithm

Given a SparseGraph and a starting node, the GraphToString converter
reconstructs the graph incrementally. At each step it searches for the
minimum-cost pair of pointer displacements (a, b) that enables either a
node insertion (V/v) or an edge insertion (C/c).

The search order is critical. From `Idea.pdf`: "In order to minimize the
number of pointer movement instructions (N/n/P/p), the pairs will be checked
in a sequence that visits all the possibilities (a,b) according to the lowest
sum a+b, which is the number of required pointer movement instructions."

**Bug B2 correction**: The original text says "a+b" but the *intended* metric
is |a| + |b| (the total displacement cost), since negative displacements
require the same number of movement instructions as positive ones. The number
of N/P instructions emitted for primary displacement `a` is |a|, and similarly
|b| for secondary. The function `generate_pairs_sorted_by_sum` sorts by
|a| + |b|, with tiebreaker (|a|, (a, b)) for determinism.

**Priority order**: When multiple operations are possible at the same
displacement cost, the algorithm uses a fixed priority: V > v > C > c.
This means: try node insertion via primary (V) first; if no uninserted
neighbor exists, try node insertion via secondary (v); if no uninserted
neighbors exist for either pointer, try edge insertion primary->secondary (C);
finally try edge insertion secondary->primary (c, directed only).

**Termination condition (Bug B3 correction)**: The loop continues while
`num_nodes_to_insert > 0 OR num_edges_to_insert > 0` (the original code
used AND, terminating as soon as either counter reached zero, leaving
remaining edges uninserted).

**Pointer update (Bug B4 correction)**: After emitting the N/P/n/p movement
instructions and the structural instruction (V/v/C/c), the actual pointer
fields are updated to the tentative positions. The original code failed to
update them, causing subsequent iterations to search from stale positions.

**Index space mapping**: G2S maintains two dictionaries:
- `_i2o: dict[int, int]` -- input graph node -> output graph node
- `_o2i: dict[int, int]` -- output graph node -> input graph node

These translate between the input graph's fixed numbering and the output
graph's incrementally-assigned numbering. All neighbor lookups must go
through these dictionaries.

**Code**: `graph_to_string.py`, class `GraphToString`.

### 2.10 Canonical String

The greedy G2S algorithm is **not** isomorphism-equivariant: its output
depends on the iteration order over Python sets, which in turn depends on
the integer node IDs. Two isomorphic graphs with different node labelings may
produce different greedy strings.

The **canonical string** for a graph G is defined as (Lopez-Rubio, 2025,
Section 2.1; Idea.pdf):

    w*_G = lexmin { w in argmin_{v in V, P in Paths(G,v)} |P| }

That is: consider all starting nodes v in V; for each starting node, consider
all possible execution paths P of the G2S algorithm (branching over all valid
neighbor choices at V/v steps); collect all resulting strings; filter to keep
only those of minimum length; among those, select the lexicographically
smallest. This is a **complete graph invariant**:

    w*_G = w*_H  if and only if  G is isomorphic to H

**Why exhaustive search is necessary**: The greedy G2S algorithm makes
deterministic but labeling-dependent choices at V/v steps (it picks the first
uninserted neighbor from a Python set iteration). Different labelings lead to
different neighbor iteration orders, hence different strings. The canonical
search must explore *all* valid neighbor choices at each V/v branch point to
find the true optimum.

**What is NOT searched over**: The V > v > C > c priority order and the
minimum-displacement pair sorting are **labeling-independent** properties of
the algorithm definition itself. They are preserved as constraints in the
canonical search, not as search dimensions. Only the *choice of which
uninserted neighbor to pick* at V/v steps is branched over.

**Implementation**: `canonical.py` uses in-place mutation with backtracking
(undo) rather than deep copies for performance. Each V/v branch:
1. Forward: add_node, add_edge, insert_after (mutate SparseGraph + CDLL).
2. Recurse: call _step with updated state.
3. Backward: remove (CDLL), undo_edge, undo_node (restore previous state).

**Complexity**: Exponential in the worst case (product of neighbor-choice
counts at each V/v step). Practical for graphs up to ~10-12 nodes. Dense
graphs (complete, high-density GNP) are much more expensive than sparse
graphs (trees, paths, stars) due to more branching at each step.

From `Idea.pdf`: "My graph-to-string conversion algorithm would have cubic
complexity, with N the number of nodes: N strings are computed (one per node),
and each string requires O(N^2) operations because at most there are N^2
edges." This refers to the *greedy* algorithm; the *exhaustive* canonical
search is exponential.

### 2.11 Graph Distance

From `Idea.pdf`: "The distance between graphs is the Levenshtein distance
between canonical strings. It allows knowing the changes needed to go from
one graph to another. It has distance properties invariant to graph
isomorphisms."

The Levenshtein edit distance between canonical strings provides:
- **Metric properties**: non-negativity, identity of indiscernibles (d=0 iff
  isomorphic), symmetry, triangle inequality.
- **Approximation to graph edit distance (GED)**: exact GED is NP-hard
  (Zeng et al., 2009). The Levenshtein distance on canonical strings is
  computable in O(|w*_G| * |w*_H|) and correlates with GED.

**Code**: `canonical.py`, functions `levenshtein()` and `graph_distance()`.

### 2.12 Round-Trip Correctness Property

Let S2G: Sigma* -> G denote StringToGraph and G2S: G x V -> Sigma* denote
GraphToString. The required property is:

    For all w in Sigma*:  S2G(w) ~ S2G(G2S(S2G(w), v_0))

where ~ denotes graph isomorphism and v_0 is any valid starting node. Note
that string equality (w = G2S(S2G(w), v_0)) is NOT required -- different
strings can produce isomorphic graphs. String equality holds only for the
canonical string.

---

## 3. Module-by-Module Reference

### 3.1 `cdll.py` -- CircularDoublyLinkedList

**Purpose**: Array-backed circular doubly linked list with integer payloads.
Provides O(1) insertion after a given node, O(1) removal, and O(1) traversal.

**Internal structure**:
- `_next[i]`, `_prev[i]`: successor/predecessor CDLL indices for node `i`.
- `_data[i]`: integer payload (graph node index) stored at CDLL node `i`.
- `_free`: stack of available CDLL indices. Pops in order 0, 1, 2, ...
- `_size`: number of active nodes.

**Key methods**:

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `insert_after(node, value)` | `(int, int) -> int` | Allocate new CDLL node, set payload to `value`, splice it after `node`. Returns new CDLL index. If list empty, `node` is ignored. |
| `remove(node)` | `(int) -> None` | Unlink `node` from list, return its index to free list. |
| `get_value(node)` | `(int) -> int` | Return `_data[node]` (the graph node index payload). |
| `set_value(node, value)` | `(int, int) -> None` | Overwrite `_data[node]`. |
| `next_node(node)` | `(int) -> int` | Return `_next[node]`. |
| `prev_node(node)` | `(int) -> int` | Return `_prev[node]`. |

**Circularity**: When the list has one node, `_next[node] == _prev[node] == node`
(self-loop in the CDLL structure, NOT in the graph).

**No known bugs** in the original code. Migrated as-is with type annotations.

### 3.2 `sparse_graph.py` -- SparseGraph

**Purpose**: Adjacency-set graph representation with contiguous integer node IDs.
Supports both directed and undirected semantics.

**Internal structure**:
- `_adjacency: list[set[int]]`: `_adjacency[u]` is the set of neighbors of node `u`.
- `_node_count`: current number of nodes (IDs are 0, 1, ..., _node_count-1).
- `_edge_count`: raw stored edge count. For undirected graphs, each logical
  edge is counted TWICE (once per direction). Use `logical_edge_count()` for
  the user-facing count.
- `_directed_graph`: whether edges are directed.
- `_max_nodes`: pre-allocated capacity.

**Key methods**:

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `add_node()` | `() -> int` | Increment `_node_count`, return new node ID. |
| `add_edge(source, target)` | `(int, int) -> None` | Add directed edge source->target. If undirected, also add target->source. **B9 guard**: if target already in `_adjacency[source]`, no-op. |
| `has_edge(source, target)` | `(int, int) -> bool` | Return `target in _adjacency[source]`. |
| `neighbors(node)` | `(int) -> set[int]` | Return `_adjacency[node]`. |
| `logical_edge_count()` | `() -> int` | Return `_edge_count` for directed, `_edge_count // 2` for undirected. |
| `is_isomorphic(other)` | `(SparseGraph) -> bool` | Backtracking isomorphism test (simple, for testing only). |

**Self-loop semantics**: `add_edge(x, x)` is valid. For undirected graphs,
the "reverse" edge is the same as the forward edge, so `_adjacency[x]` gets
one entry {x} but `_edge_count` is incremented by 2, giving
`logical_edge_count() = 1`. For directed graphs, `_edge_count` is incremented
by 1.

**Bug fixes applied**:
- **B1**: `_edge_count` was initialized to 1 in the original code. Fixed to 0.
- **B9**: `add_edge` did not guard against duplicate edges. If the same edge
  was added twice, `_edge_count` was incremented again. Fixed with an
  `if target not in _adjacency[source]` guard.

### 3.3 `string_to_graph.py` -- StringToGraph

**Purpose**: Execute an IsalGraph instruction string to produce a SparseGraph.

**Constructor**: Pre-computes `_max_nodes = 1 + count('V') + count('v')` to
allocate SparseGraph and CDLL with exact capacity. Validates that all
characters are in {N, n, P, p, V, v, C, c, W}.

**`run()` method**: Initializes the state (one node, one CDLL node, both
pointers on it), then iterates over instructions calling
`_execute_instruction()`. Optionally collects deep-copied snapshots for
debugging/visualization.

**Instruction dispatch** (`_execute_instruction`):

- `N`: `self._primary_ptr = self._cdll.next_node(self._primary_ptr)`
- `P`: `self._primary_ptr = self._cdll.prev_node(self._primary_ptr)`
- `n`: `self._secondary_ptr = self._cdll.next_node(self._secondary_ptr)`
- `p`: `self._secondary_ptr = self._cdll.prev_node(self._secondary_ptr)`
- `V`: Create new graph node `u`. Add edge from
  `cdll.get_value(primary_ptr)` to `u`. Insert `u` into CDLL after
  `primary_ptr`. Primary pointer does NOT move.
- `v`: Same as V but using secondary pointer.
- `C`: Add edge from `cdll.get_value(primary_ptr)` to
  `cdll.get_value(secondary_ptr)`.
- `c`: Add edge from `cdll.get_value(secondary_ptr)` to
  `cdll.get_value(primary_ptr)`.
- `W`: No-op.

**Bug fix applied**:
- **B6**: The original code passed `self._primary_ptr` (a CDLL node index)
  directly to `add_edge`, which expects graph node indices. This was a
  **latent** bug: the CDLL free-list pops indices 0, 1, 2, ... in order
  and graph nodes are also 0, 1, 2, ..., so the two index spaces coincide
  as long as no CDLL nodes are ever removed. The fix uses
  `cdll.get_value(ptr)` throughout.

### 3.4 `graph_to_string.py` -- GraphToString

**Purpose**: Convert a SparseGraph into an IsalGraph instruction string using
the greedy algorithm.

**`generate_pairs_sorted_by_sum(m)`**: Returns all (a, b) pairs with
a, b in [-m, m], sorted by |a| + |b| (total displacement cost). Within the
same cost, pairs are further sorted by (|a|, (a, b)) for determinism.
This is the spiral enumeration of Z^2 around the origin described in Idea.pdf.

**`run(initial_node)` method**: Initializes state, then iterates:
1. Compute `pairs = generate_pairs_sorted_by_sum(current_node_count)`.
2. For each (a, b) in pairs:
   a. Move primary pointer tentatively by `a` steps -> `tent_pri_ptr`.
   b. **V check**: Does `_o2i[cdll.get_value(tent_pri_ptr)]` have an
      uninserted neighbor in the input graph? If yes, emit N/P moves + "V",
      create node, create edge, update pointer. Break.
   c. Move secondary pointer tentatively by `b` steps -> `tent_sec_ptr`.
   d. **v check**: Same as V check but for secondary. Emit n/p moves + "v".
   e. **C check**: Is there an edge in the input graph between the nodes at
      `tent_pri_ptr` and `tent_sec_ptr` that is not yet in the output graph?
      If yes, emit N/P + n/p + "C". Break.
   f. **c check** (directed only): Same as C but reversed direction.
3. Repeat until all nodes and edges are inserted.

**Reachability check**: Before starting, verifies that all nodes are reachable
from `initial_node` via outgoing edges (DFS). For undirected graphs this means
the graph must be connected. For directed graphs, all nodes must be reachable
from the start node. Raises `ValueError` if not.

**Bug fixes applied**:
- **B2**: Pair sort used `a + b` (algebraic sum) instead of `|a| + |b|`.
- **B3**: While loop used `and` instead of `or`, terminating prematurely.
- **B4**: Pointers not updated after emitting movement instructions.
- **B5**: Debug `print()` left in main loop.
- **B7**: `insert_after` was passed a graph node index where a CDLL node
  index is required (same latent-bug pattern as B6).
- **B8**: V/v checked whether the *edge* existed in the output graph, rather
  than whether the *node* had been created. This caused duplicate node
  creation for nodes that existed but lacked a specific edge.

### 3.5 `canonical.py` -- Canonical String

**Purpose**: Compute the canonical IsalGraph string via exhaustive
backtracking search. Also provides Levenshtein distance and graph distance.

**Public API**:

| Function | Signature | Semantics |
|----------|-----------|-----------|
| `canonical_string(graph)` | `(SparseGraph) -> str` | Compute w*_G. Tries all starting nodes, returns shortest then lex-smallest. |
| `graph_distance(g1, g2)` | `(SparseGraph, SparseGraph) -> int` | Levenshtein distance between canonical strings. |
| `levenshtein(s, t)` | `(str, str) -> int` | Standard O(n*m) DP with O(min(n,m)) space. |

**Internal: `_canonical_g2s(input_graph, start_node)`**: Finds the shortest
then lex-smallest G2S string from a given starting node by exploring all
neighbor choices at V/v branch points.

**Internal: `_step(...)`**: Recursive backtracking. At each call:
1. If `nleft <= 0 and eleft <= 0`: return accumulated prefix (base case).
2. Generate displacement pairs sorted by |a| + |b|.
3. For each (a, b):
   - Compute tentative primary position.
   - **V branch**: If nleft > 0 and primary has uninserted neighbors,
     iterate over ALL candidates (not just the first). For each candidate:
     forward-mutate (add_node, add_edge, insert_after), recurse, then
     backward-undo (remove, undo_edge, undo_node). Keep the best result.
   - **v branch**: Same for secondary pointer.
   - **C branch**: Deterministic (no branching). If edge exists in input
     but not output, add it, recurse, undo.
   - **c branch**: Same for directed reverse edge.
4. Return best result found.

**Defensive `_undo_node`**: After decrementing `_node_count`, clears
`_adjacency[_node_count]` to prevent stale data from corrupting future
computations if the backtracking invariant is accidentally violated.

**Self-loop handling in `canonical_string`**: The function returns `""` for
0-node graphs, and for 1-node graphs only if they have no edges. A 1-node
graph with a self-loop proceeds to the main algorithm, which will emit "C"
at displacement (0,0).

---

## 4. Critical Invariants

Violating any of these invariants causes **silent corruption** -- incorrect
graphs with no error raised.

### Invariant 1: CDLL Index != Graph Node Index

Pointers (`primary_ptr`, `secondary_ptr`) are CDLL node indices. The graph
node is obtained via `cdll.get_value(pointer)`. These two integer spaces
are distinct. They coincide when no CDLL removals have occurred (because
the free-list pops 0, 1, 2, ... in order), but this is NOT guaranteed.

**Where this matters**: Every call to `add_edge`, `neighbors`, or `has_edge`
must use graph node indices (from `get_value`), not CDLL indices.

### Invariant 2: insert_after(cdll_index, graph_node)

The first argument is a CDLL node index (where to insert after). The second
argument is the payload (graph node index to store). Swapping these produces
a corrupt CDLL.

### Invariant 3: Pointer Immobility After V/v

After a V instruction, the primary pointer stays at its current CDLL node.
The new CDLL node is inserted *after* it, but the pointer does not advance.
This means:
- `VV` from node 0 creates nodes 1 and 2, both connected to node 0 (star).
- `VNV` from node 0 creates node 1 connected to 0, moves primary to node 1,
  then creates node 2 connected to node 1 (path).

### Invariant 4: Pair Sort by |a| + |b|

The displacement cost is the number of movement instructions emitted:
|a| N/P instructions for primary, |b| n/p instructions for secondary.
Sorting by algebraic sum `a + b` (Bug B2) would incorrectly rank the pair
(-3, 0) at cost 3 below the pair (2, 2) at cost 4, when in reality both
have cost 3 and 4 respectively under |a|+|b| but the algebraic sum gives
-3 and 4.

### Invariant 5: Loop Condition `or` (not `and`)

The G2S main loop must continue while `nodes_to_insert > 0 OR edges_to_insert > 0`.
Using AND causes premature termination: once all nodes are inserted but edges
remain, the loop exits without inserting the remaining edges.

---

## 5. Bug Fix History (B1-B9)

All bugs were found in the original advisor code preserved at
`docs/original_code_and_files/`. Each bug was verified against the
mathematical specification from Lopez-Rubio (2025) and Idea.pdf.

| Bug | Module | Original Code | Fix | Impact |
|-----|--------|---------------|-----|--------|
| **B1** | `sparse_graph.py` | `_edge_count = 1` | `_edge_count = 0` | All edge counts off by 1 |
| **B2** | `graph_to_string.py` | `sort(key=a+b)` | `sort(key=\|a\|+\|b\|)` | Wrong pair ordering, suboptimal strings |
| **B3** | `graph_to_string.py` | `while n>0 and e>0` | `while n>0 or e>0` | Edges dropped when all nodes already inserted |
| **B4** | `graph_to_string.py` | (no pointer update) | Update pointers after emission | Subsequent iterations searched from stale positions |
| **B5** | `graph_to_string.py` | `print(self._output_string)` | Removed | Debug output in production code |
| **B6** | `string_to_graph.py` | `add_edge(primary_ptr, ...)` | `add_edge(get_value(primary_ptr), ...)` | CDLL index used as graph node index (latent) |
| **B7** | `graph_to_string.py` | `insert_after(graph_node, ...)` | `insert_after(cdll_ptr, ...)` | Same latent bug pattern as B6 |
| **B8** | `graph_to_string.py` | Checked edge existence | Check node existence via `_i2o` | Duplicate node creation for existing nodes |
| **B9** | `sparse_graph.py` | No duplicate guard in `add_edge` | `if target not in _adjacency[source]` guard | Edge count inflated by duplicate insertions |

---

## 6. Verification Strategy

### 6.1 Phase 1: Short Deterministic Strings

Test round-trip for short, manually inspectable strings. For each string w:
1. `S2G(w)` -> graph G1
2. `G2S(G1, 0)` -> string w'
3. `S2G(w')` -> graph G2
4. Assert `G1.is_isomorphic(G2)`

Test cases from `docs/ISALGRAPH_AGENT_CONTEXT.md` Section 3.1:
- Single: "V", "v"
- Double: "VV", "Vv", "vV", "vv", "VC", "vC", "Vc", "NV", "nv", "PV", "pv"
- Triple: "VNV", "VnC", "vNv", "vvc", "VVN", "VNC"
- Self-loop: "C" (both pointers at node 0, creates self-loop)

**Code**: `tests/unit/test_roundtrip.py`

### 6.2 Phase 2: Massive Random Testing

Random valid strings (length 1-50), both directed and undirected. Also
NetworkX graph families: GNP, Barabasi-Albert, trees, cycles, complete,
grid, Watts-Strogatz, star, wheel, ladder, self-loop. Cross-validate with
`nx.is_isomorphic`.

**Code**: `benchmarks/random_roundtrip/random_roundtrip.py`

### 6.3 Phase 3: Canonical Invariance

For isomorphic graph pairs (created by random relabeling), assert that
`canonical_string(G) == canonical_string(G')`. For non-isomorphic pairs,
assert inequality. Test both undirected and directed graphs.

**Code**: `benchmarks/canonical_invariance/canonical_invariance.py`

---

## 7. File-to-Functionality Map

```
src/isalgraph/core/
  cdll.py             <- Section 3.1: CircularDoublyLinkedList
  sparse_graph.py     <- Section 3.2: SparseGraph (B1, B9 fixed)
  string_to_graph.py  <- Section 3.3: StringToGraph (B6 fixed)
  graph_to_string.py  <- Section 3.4: GraphToString (B2-B5, B7-B8 fixed)
  canonical.py        <- Section 3.5: canonical_string, graph_distance, levenshtein
  __init__.py         <- Re-exports public API
  README.md           <- This document
```

---

## 8. References

- Lopez-Rubio, E. (2025). "Representation of the structure of graphs by
  sequences of instructions." arXiv:2512.10429v2. `docs/references/2512_10429v2.pdf`
- Lopez-Rubio, E. (2025). IsalGraph design notes. `docs/references/Idea.pdf`
- Gilmer, J. et al. (2017). "Neural message passing for quantum chemistry." ICML.
- Kipf, T. N. & Welling, M. (2017). "Semi-supervised classification with graph
  convolutional networks." ICLR.
- Xu, K. et al. (2019). "How powerful are graph neural networks?" ICLR.
- Zeng, Z. et al. (2009). "Comparing Stars: On Approximating Graph Edit
  Distance." PVLDB.
- You, J. et al. (2018). "GraphRNN: Generating Realistic Graphs with an
  Auto-Regressive Model." ICML.
- Fey, M. & Lenssen, J. E. (2019). "Fast Graph Representation Learning with
  PyTorch Geometric." ICLR Workshop.
