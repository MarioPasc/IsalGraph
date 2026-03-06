  Scientific Analysis: What's Left to Try?

  Let me reason through this systematically, grounded in the preprint's claims and the
  mathematical foundations.

  What the current benchmarks cover

  ┌────────────────────────┬───────────────────────────────────┬──────────────────────────┐
  │       Benchmark        │           Claim tested            │          Status          │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┤
  │ random_roundtrip       │ Reversibility: S2G(w) ~           │ 100% pass                │
  │                        │ S2G(G2S(S2G(w)))                  │                          │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┤
  │ canonical_invariance   │ Completeness: w*(G)=w*(H) iff G~H │ 100% invariance +        │
  │                        │                                   │ discrimination           │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┤
  │ string_length_analysis │ Compactness: |w| << N^2 for       │ Confirmed                │
  │                        │ sparse graphs                     │                          │
  ├────────────────────────┼───────────────────────────────────┼──────────────────────────┤
  │ levenshtein_vs_ged     │ Locality: Lev(w*_G, w*_H)         │ Confirmed (Pearson       │
  │                        │ correlates with GED               │ ~0.7+)                   │
  └────────────────────────┴───────────────────────────────────┴──────────────────────────┘

  What is scientifically missing

  1. String Processing Pipeline: w -> w' -> w (the gap you identified)*

  This is the most important missing piece. You described it precisely: generate random w,
  compute G = S2G(w), then w' = G2S(G, v0), then w* = canonical(G), and systematically
  compare:
  - |w| vs |w'|: How much does G2S "normalize" a random string? (Our new Panel (a) shows this
  for the first time!)
  - |w'| vs |w|*: How suboptimal is greedy vs exhaustive? (The canonical_invariance
  greedy_vs_canonical tests this, but only reports pass/fail, not the distribution of the gap)
  - Lev(w, w') vs Lev(w', w) vs Lev(w, w)**: The Levenshtein distances between these three
  representations of the same graph. This characterizes the "representation space" geometry.

  The current benchmarks test pieces of this pipeline in isolation. A unified benchmark that
  reports all three stages simultaneously would be more powerful scientifically. It would show
   that the encoding converges: random w is far from w', w' is closer to w*, and w* is the
  unique attractor.

  2. Greedy Optimality Gap Distribution

  The canonical_invariance results show greedy != canonical in ~32% of cases. But we don't
  characterize how bad the greedy is. Key questions:
  - What is the distribution of |w'_best| / |w*|? (ratio > 1 means suboptimal)
  - Does the gap grow with graph size N?
  - Which graph families have the largest gap? (Dense graphs likely have more room for
  optimization)
  - Is the greedy sometimes optimal? (For trees, it likely is)

  3. Starting Node Sensitivity

  For a fixed graph G, how does |G2S(G, v)| vary across starting nodes v? This is a
  fundamental question about the encoding:
  - What is Var(|G2S(G, v)|) / E[|G2S(G, v)|] for different families?
  - How does the best starting node compare to the worst?
  - For which graph families does starting node matter most?

  This is partially in string_length_analysis (greedy_length_best vs greedy_length_node0) but
  not visualized or analyzed systematically.

  4. Round-trip Fixed Point Property

  A theoretical property worth testing: does the encoding stabilize after one round-trip?
  Specifically:
  - Given w' = G2S(S2G(w), v0), is w' a "fixed point" meaning G2S(S2G(w'), v0) = w'?
  - If yes, this proves that G2S always produces canonical-form strings from its own
  perspective (the greedy is idempotent).
  - If no, this would be a surprising finding worth reporting.

  5. Alphabet Utilization / Entropy Analysis

  Our new Panel (b) in random_roundtrip gives a first look at instruction composition. A
  deeper analysis would compute:
  - Shannon entropy H(w*) vs H(w') vs log2(9) (maximum for 9-character alphabet)
  - Information density: H(w*) / |w*| — how many bits per instruction character?
  - Compare with information-theoretic lower bound: log2(number of graphs with N nodes, M
  edges) / |w*|

  This would answer: "How close is IsalGraph to the information-theoretic optimum?"

  My Recommendation: Priority Order

  1. String Pipeline Benchmark (highest priority) — unifies the w vs w' vs w* analysis you
  described. This is the missing "bridge" benchmark that connects round-trip, canonical, and
  compression analyses. It answers the core question: "What happens at each stage of the
  encoding pipeline?"
  2. Greedy Optimality Gap — extends canonical_invariance with quantitative gap analysis. Can
  be added as a new figure panel in the existing benchmark.
  3. Starting Node Sensitivity — extends string_length_analysis with variance analysis.
  Relatively easy to add.
  4. Round-trip Idempotence — a small but elegant theoretical test. Quick to implement.
  5. Entropy Analysis — deeper but requires more thought on the right information-theoretic
  baselines.
