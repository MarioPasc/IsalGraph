"""T-13 paper figures and tables, all drawn from one design source.

T-13 answers R3.7d by replacing the manuscript's unqualified "exponential worst
case" with a **characterised** one.  The characterisation is a two-parameter
statement (``T-13-design.md`` 6.3):

    The unpruned canonical search's cost is governed by the degree sequence;
    the pruned search's cost is governed by the automorphism group.

The evidence is a set of *ladders* -- families holding ``n``, ``m`` and the
whole degree sequence exactly constant while ``|Aut|`` falls by orders of
magnitude -- so the figures in this package exist to make one contrast legible:
the search-free arms and ``isalgraph_exhaustive`` flat against ``log10|Aut|``,
``isalgraph_pruned`` and ``min_dfs`` rising steeply.

Modules:
    design: the representation registry and the style contract.  Unlike its
        T-06 counterpart it **raises** on an unregistered key.
    data: readers for the frozen ``t13.1`` shard files and the ``t13c.1``
        counter files; grouping and censoring-aware statistics, no plotting.
    fig_cost_law: the main-text figure -- seconds against ``log10|Aut|``, one
        panel per ladder.
    fig_resolution: partition resolution against the invariance ceiling.
    fig_operations: the four costed operations against ``n``.  Supplementary.
    tables: the LaTeX tables.

Entry points, one per module::

    python -m benchmarks.eval_t13_figures.fig_cost_law   --records '<glob>' --out-dir D
    python -m benchmarks.eval_t13_figures.fig_resolution --records '<glob>' --out-dir D
    python -m benchmarks.eval_t13_figures.fig_operations --counters '<glob>' --out-dir D
    python -m benchmarks.eval_t13_figures.tables         --records '<glob>' --out-dir D

Every module is importable with matplotlib absent: third-party imports live
inside function bodies, matching the ``isalgraph.viz`` contract.
"""

from __future__ import annotations

__all__ = ["data", "design", "fig_cost_law", "fig_operations", "fig_resolution", "tables"]
