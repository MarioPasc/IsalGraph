# IsalGraph Website — Development Progress

## Status: Complete

### Phase 1: CSS Foundation + Shared Components
- [x] 12 CSS files: variables, reset, typography, layout, components, nav, footer, + 5 page-specific
- [x] theme.js (dark/light toggle with localStorage)
- [x] components.js (nav/footer injection, expandable toggles, mobile menu)
- [x] SVG favicon (two connected graph nodes)

### Phase 2: HTML Pages (5 pages)
- [x] **index.html** — Hero with animated graph, key results strip (KaTeX), how-it-works teaser, publications teaser
- [x] **how-it-works.html** — Instruction table, S2G demo with 3-panel viz, KaTeX theorems (round-trip, canonical, distance), expandable math sections
- [x] **publications.html** — 2 papers (arXiv:2603.11039, arXiv:2512.10429), BibTeX copy, Schema.org JSON-LD
- [x] **team.html** — Profile cards (López-Rubio, Pascual González), ICAI lab section, acknowledgements
- [x] **playground.html** — S2G/G2S tabs, presets (Triangle, Path, Star, Cycle, House), graph editor, transport controls

### Phase 3: Core JS Algorithm Ports (verified identical to Python)
- [x] alphabet.js — Instruction set definitions, colored string rendering
- [x] cdll.js — CircularDoublyLinkedList with clone()
- [x] sparse-graph.js — SparseGraph with clone(), getEdgeList(), isIsomorphic()
- [x] s2g.js — StringToGraph with trace mode (B6 fix)
- [x] g2s.js — GraphToString with trace mode (B2, B3, B4, B5, B7, B8 fixes)
- [x] Round-trip verified: JS produces identical output to Python ("VNVnC" → triangle, G2S → "VpvNC")

### Phase 4: Visualization Modules
- [x] graph-renderer.js — D3 force-directed graph with node/edge highlighting and pointer indicators
- [x] cdll-renderer.js — CDLL circular ring visualization with π/σ pointer labels
- [x] step-player.js — Transport controls (play/pause/step/reset/speed) with step log
- [x] hero-animation.js — D3 triangle graph with typewriter string animation
- [x] playground-controller.js — S2G/G2S mode orchestration, presets, validation

### Phase 5: Polish
- [x] Image optimization: icai.png 2.9MB → 89KB, team photos compressed
- [x] OG social card image (1200x630) for all pages
- [x] Twitter Card meta tags on all pages
- [x] Skip-to-content link on all pages
- [x] Visible focus rings for keyboard navigation (`focus-visible`)
- [x] Screen-reader utility class (`.sr-only`)
- [x] WCAG AA color contrast fixes (text-tertiary adjusted)
- [x] `aria-expanded`/`aria-controls` on expandable sections
- [x] SVG `aria-hidden` on decorative elements
- [x] Form label associations and `aria-label` attributes
- [x] Mobile menu changed to `<nav>` with proper ARIA
- [x] `prefers-reduced-motion` media query
- [x] `loading="lazy"` on non-critical images
- [x] CDN preconnect hints

### Validation
- [x] All 5 HTML files parse without errors
- [x] All 13 JS files pass Node.js syntax check
- [x] All internal links resolve
- [x] JS round-trip test matches Python output
- [x] Responsive breakpoints at 768px and 1024px

### Stats
- 40 files total
- 252KB code (excluding images)
- 1,974 lines CSS / 2,175 lines JS / 1,089 lines HTML
