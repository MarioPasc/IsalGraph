"""Tests for the T-13 complexity package.

Not collected by ``pytest`` at the repository root: ``pyproject.toml`` sets
``testpaths = ["tests"]``, so these run only when the path is given explicitly::

    $PY -m pytest benchmarks/real_data/eval_t13_complexity/tests/ -q

That is deliberate.  It keeps the repository's reference test count -- the
figure CLAUDE.md tracks and requires an explanation for any drop in -- a
property of ``tests/`` alone, so work under ``benchmarks/`` can neither inflate
nor deflate it.
"""

from __future__ import annotations
