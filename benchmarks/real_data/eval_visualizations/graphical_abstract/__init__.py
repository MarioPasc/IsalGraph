"""Graphical abstract generation for the IsalGraph paper.

Generates publication-quality panels for the Elsevier graphical abstract:
  - Panel A: Bijective string <-> CDLL <-> graph encoding concept
  - Panel B: Three main results (message length, speedup, GED correlation)
  - Composite: Full graphical abstract at Elsevier dimensions (531x1328 px)

Results are presented in priority order:
  1. Message length compactness (strongest: beta=0.537, R^2=0.947, Wins=99.6%)
  2. Computational speedup (48x to 14,108x over exact GED)
  3. GED proxy quality (rho=0.691, formula-only display)
"""
