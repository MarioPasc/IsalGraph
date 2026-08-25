"""T-06 paper figures and tables, all drawn from one design source.

``design.py`` owns every colour, font size, display name, draw order and
LaTeX macro. Nothing else in this package -- or in
``eval_size_profile/figures.py``, which imports from here -- may define one.

Modules:
    design: the registry and the style contract.
    data: readers for the T-06 archive; aggregation, no plotting.
    fig_ic: figure 4, information content (Claim A).
    tables: the LaTeX comparison tables.

Entry point:
    ``python -m benchmarks.real_data.eval_t06_figures --report <dir>``
"""

from __future__ import annotations

__all__ = ["data", "design", "fig_ic", "tables"]
