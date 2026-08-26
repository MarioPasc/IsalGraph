"""Tests for T-13.

Present so that ``mypy`` resolves these modules under one name rather than two;
without it a namespace package makes ``measure.py`` visible as both ``measure``
and ``benchmarks.real_data.eval_t13_complexity.measure`` and type checking
aborts before it starts.
"""

from __future__ import annotations
