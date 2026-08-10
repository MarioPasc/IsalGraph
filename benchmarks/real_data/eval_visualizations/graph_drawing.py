"""Compatibility shim -- the implementation now lives in ``isalgraph.viz``.

``draw_graph`` moved to :func:`isalgraph.viz.nx_view.draw_nx_graph`, where
it is renamed to leave the bare name ``draw_graph`` free for the
library-facing dispatcher :func:`isalgraph.viz.graph_view.draw_graph`,
which takes a :class:`~isalgraph.core.sparse_graph.SparseGraph` and routes
through the backend registry.

Behaviour is unchanged. New figure code should import from
``isalgraph.viz`` directly.
"""

from __future__ import annotations

from isalgraph.viz.nx_view import _DEFAULT_NODE_COLOR
from isalgraph.viz.nx_view import draw_nx_graph as draw_graph

__all__ = ["_DEFAULT_NODE_COLOR", "draw_graph"]
