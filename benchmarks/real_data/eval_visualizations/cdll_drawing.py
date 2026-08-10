"""Compatibility shim -- the implementation now lives in ``isalgraph.viz``.

See :mod:`isalgraph.viz.cdll_view`, which keeps this module's
``draw_cdll_ring`` signature unchanged and adds
:func:`~isalgraph.viz.cdll_view.draw_cdll_ring_for_snapshot` for
trace-driven figures. New figure code should import from ``isalgraph.viz``
directly.
"""

from __future__ import annotations

from isalgraph.viz.cdll_view import (
    _DEFAULT_NODE_COLOR,
    NEW_NODE_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    _draw_pointer_arrow,
    draw_cdll_ring,
    get_legend_handles,
)

__all__ = [
    "NEW_NODE_COLOR",
    "PRIMARY_COLOR",
    "SECONDARY_COLOR",
    "_DEFAULT_NODE_COLOR",
    "_draw_pointer_arrow",
    "draw_cdll_ring",
    "get_legend_handles",
]
