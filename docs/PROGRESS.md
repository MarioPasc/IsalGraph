# IsalGraph Website — Development Progress

## Status: Phase 1-4 Complete (Initial Implementation)

### Completed
- [x] **Phase 1: CSS Foundation** — 12 CSS files (design tokens, reset, typography, layout, components, nav, footer, 5 page-specific)
- [x] **Phase 1: Shared JS** — theme.js (dark/light toggle), components.js (nav/footer injection)
- [x] **Phase 1: Favicon** — SVG favicon (two connected graph nodes)
- [x] **Phase 2: HTML Pages** — All 5 pages with full static content
  - index.html — Hero, key results strip, how-it-works teaser, publications teaser
  - how-it-works.html — Instruction table, S2G demo area, KaTeX math theorems, expandable sections
  - publications.html — 2 papers (arXiv:2603.11039, arXiv:2512.10429), BibTeX copy, Schema.org JSON-LD
  - team.html — Profile cards (López-Rubio, Pascual González), ICAI lab section, acknowledgements
  - playground.html — S2G/G2S tabs, presets, graph editor, transport controls
- [x] **Phase 3: Core JS Algorithm Ports** — Faithful ports of Python implementations
  - alphabet.js — Instruction set definitions, colored rendering
  - cdll.js — CircularDoublyLinkedList with clone()
  - sparse-graph.js — SparseGraph with clone(), getEdgeList(), isIsomorphic()
  - s2g.js — StringToGraph with trace mode (B6 fix)
  - g2s.js — GraphToString with trace mode (B2, B3, B4, B5, B7, B8 fixes)
- [x] **Phase 4: Visualization** — D3 graph renderer, CDLL renderer, step player, hero animation, playground controller

### Validation
- [x] All 5 HTML files parse without errors
- [x] All 13 JS files pass Node.js syntax check
- [x] All internal links resolve correctly
- [x] Dark/light theme toggle works
- [x] Responsive design with mobile breakpoints

### Remaining (Phase 5: Polish)
- [ ] Compress icai.png (2.9MB) to <200KB
- [ ] Test at 375px, 768px, 1024px, 1440px viewports
- [ ] Full accessibility audit (ARIA labels, focus rings, contrast)
- [ ] OG social card image
- [ ] Cross-browser testing
