# IsalGraph Website — Development Progress

## Status: Complete (All phases done)

### Phase 1-4: Initial Implementation (complete)
See git history for details.

### Phase 5: Polish (complete)
- [x] Image optimization (icai: 2.9MB→89KB, team photos compressed)
- [x] OG social card + Twitter Card meta on all pages
- [x] Skip-to-content, focus rings, ARIA, WCAG AA contrast
- [x] `prefers-reduced-motion` support

### Fixes Applied (feedback round)

#### Correctness
- [x] **Fixed all preset strings**: `VNVnC` was NOT a triangle (it produced a path + self-loop). Correct encodings verified via round-trip against Python:
  - Triangle K3: `VVPnC` (was `VNVnC`)
  - Path P3: `Vpv` (was `VNV`)
  - Cycle C4: `VVpvPpC` (was `VNVNVNVPPPnC`)
  - Star S4: `VVVV` (correct)
  - House: `VVpvpvPCnC` (was `VNVNVnCNC`)
- [x] Hero animation updated to use correct triangle string `VVPnC`
- [x] All 6 S2G presets and 5 G2S presets pass round-trip isomorphism test

#### How It Works page
- [x] Removed broken interactive S2G demo (JS was never wired to buttons)
- [x] Replaced with accurate step-by-step walkthrough table for `VVPnC` → K3
- [x] Added "Try it yourself" callout linking to Playground
- [x] Removed unnecessary D3/core JS script tags (page no longer needs them)

#### Playground
- [x] **G2S step-by-step**: Added trace visualization with action log, transport controls (play/pause/step/reset)
- [x] **S2G action log**: Added step-by-step action descriptions below graph output
- [x] G2S shows partial string being built at each step
- [x] Both modes show pointer positions (π/σ) on the graph

#### ICAI Logo
- [x] Fixed horizontal stretching: changed container width from 200px to 280px, added `height: auto`

#### Publications
- [x] Added IsalChem paper: Thurnhofer-Hemsi et al. (2025) "Representation of Molecules by Sequences of Instructions", J. Chem. Inf. Model.
- [x] Added Schema.org structured data for the new paper
- [x] BibTeX copy works for all 3 papers

### Validation
- [x] All 5 HTML files parse without errors
- [x] All 13 JS files pass Node.js syntax check
- [x] All internal links resolve correctly
- [x] 11/11 preset tests pass (6 S2G + 5 G2S round-trips)
- [x] JS output matches Python output exactly for all test cases
