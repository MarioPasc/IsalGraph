# ruff: noqa: N803, N806
"""Pure-math computation of message lengths for IsalGraph vs GED encodings.

Computes the number of bits needed to constructively encode a graph G
starting from the single-node graph G_0 = ({v_0}, emptyset), using either:

1. **IsalGraph** — a string w of length L over Sigma = {N,n,P,p,V,v,C,c,W}
   with |Sigma| = 9.  Uniform encoding: L * log_2(9) bits.
   Shannon-optimal:  L * H(w) bits.

2. **GED construction** — (N-1) node insertions + M edge insertions.
   Three schemes of increasing generosity to GED:
   - Generous: only encode edge endpoints (best case for GED)
   - Standard: 1-bit type per operation + endpoints
   - Full: log_2(6)-bit type per operation + endpoints

3. **Combinatorial lower bound** — log_2(C(N(N-1)/2, M))

All functions are stdlib-only (math + collections).
"""

from __future__ import annotations

import math
from collections import Counter

# IsalGraph alphabet size: {N, n, P, p, V, v, C, c, W}
SIGMA_SIZE: int = 9


# ---------------------------------------------------------------------------
# IsalGraph message length
# ---------------------------------------------------------------------------


def isalgraph_message_length_uniform(string_length: int) -> float:
    """Uniform encoding: L * log_2(|Sigma|) bits.

    Each symbol is one of 9 instruction types, encoded with a fixed-width
    code of log_2(9) ~ 3.17 bits per symbol.

    Args:
        string_length: Length L of the IsalGraph instruction string.

    Returns:
        Message length in bits.
    """
    if string_length <= 0:
        return 0.0
    return string_length * math.log2(SIGMA_SIZE)


def empirical_entropy(string: str) -> float:
    """Empirical Shannon entropy H(w) of a string.

    H(w) = -sum_{c in alphabet} f_c * log_2(f_c)

    where f_c = count(c) / len(w) is the relative frequency of character c.

    Args:
        string: The instruction string.

    Returns:
        Entropy in bits per symbol.  Returns 0.0 for empty or
        single-character-type strings.
    """
    L = len(string)
    if L == 0:
        return 0.0
    counts = Counter(string)
    if len(counts) <= 1:
        return 0.0
    H = 0.0
    for count in counts.values():
        f = count / L
        H -= f * math.log2(f)
    return H


def isalgraph_message_length_entropy(string: str) -> float:
    """Shannon-optimal encoding: L * H(w) bits.

    Uses the empirical symbol distribution of the string to compute the
    minimum achievable message length under an optimal prefix code.

    Args:
        string: The instruction string.

    Returns:
        Message length in bits.
    """
    L = len(string)
    if L == 0:
        return 0.0
    H = empirical_entropy(string)
    return L * H


# ---------------------------------------------------------------------------
# GED construction message length
# ---------------------------------------------------------------------------


def ged_construction_message_length(
    n_nodes: int,
    n_edges: int,
    scheme: str = "generous",
) -> float:
    """Message length for constructing graph G from G_0 via GED operations.

    Building G = (V, E) with N = |V|, M = |E| from G_0 = ({v_0}, {}):
    - Need (N-1) node insertions + M edge insertions.
    - Each edge insertion must specify two endpoint node indices, each
      requiring ceil(log_2(N)) bits.

    Three schemes:

    - **generous** (best case for GED): Receiver knows N.  Nodes inserted in
      a fixed order (no type bits needed, no node identity bits).  Only edge
      endpoints are encoded: M * 2 * ceil(log_2(N)) bits.

    - **standard**: 1 type bit per operation (node_ins vs edge_ins) +
      endpoint bits for edges: (N-1+M) * 1 + M * 2 * ceil(log_2(N))

    - **full**: All 6 GED operation types encoded with log_2(6) bits per
      operation + endpoints: (N-1+M) * log_2(6) + M * 2 * ceil(log_2(N))

    Args:
        n_nodes: Number of nodes N in the target graph.
        n_edges: Number of edges M in the target graph.
        scheme: One of "generous", "standard", "full".

    Returns:
        Message length in bits.

    Raises:
        ValueError: If scheme is not recognized.
    """
    N = n_nodes
    M = n_edges

    if N <= 1:
        return 0.0

    node_bits = math.ceil(math.log2(N))
    endpoint_bits = M * 2 * node_bits

    if scheme == "generous":
        return float(endpoint_bits)
    elif scheme == "standard":
        type_bits = (N - 1 + M) * 1
        return float(type_bits + endpoint_bits)
    elif scheme == "full":
        type_bits = (N - 1 + M) * math.log2(6)
        return float(type_bits + endpoint_bits)
    else:
        raise ValueError(f"Unknown GED scheme: {scheme!r}. Use 'generous', 'standard', or 'full'.")


# ---------------------------------------------------------------------------
# Combinatorial lower bound
# ---------------------------------------------------------------------------


def combinatorial_lower_bound(n_nodes: int, n_edges: int) -> float:
    """Information-theoretic minimum bits to specify which M edges exist.

    L_comb = log_2(C(N(N-1)/2, M))

    where C(n, k) is the binomial coefficient, computed via log-gamma
    for numerical stability.

    For undirected simple graphs, there are N(N-1)/2 possible edges,
    and we choose M of them.

    Args:
        n_nodes: Number of nodes N.
        n_edges: Number of edges M.

    Returns:
        Lower bound in bits.  Returns 0.0 when the graph is trivially
        determined (M=0, M=N(N-1)/2, or N<=1).
    """
    N = n_nodes
    M = n_edges

    if N <= 1 or M <= 0:
        return 0.0

    max_edges = N * (N - 1) // 2
    if max_edges <= M:
        return 0.0  # Complete graph: only one possibility

    # log_2(C(n, k)) = (lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)) / ln(2)
    log2_binom = (
        math.lgamma(max_edges + 1) - math.lgamma(M + 1) - math.lgamma(max_edges - M + 1)
    ) / math.log(2)

    return log2_binom


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------


def compression_ratio(isalgraph_bits: float, ged_bits: float) -> float:
    """Compression ratio: ged_bits / isalgraph_bits.

    Values > 1 mean IsalGraph is more compact.
    Values < 1 mean GED is more compact.

    Args:
        isalgraph_bits: IsalGraph message length in bits.
        ged_bits: GED message length in bits.

    Returns:
        Ratio, or float('inf') if isalgraph_bits is 0 and ged_bits > 0,
        or float('nan') if both are 0.
    """
    if isalgraph_bits == 0.0:
        if ged_bits == 0.0:
            return float("nan")
        return float("inf")
    return ged_bits / isalgraph_bits
