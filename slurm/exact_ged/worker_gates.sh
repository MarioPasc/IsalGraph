#!/usr/bin/env bash
# T-03 validation gates. BLOCKS every production job: the launcher chains the compute
# stages with --dependency=afterok, so a non-zero exit here stops the programme.
#
# Gate 0 is re-anchored on brute-force enumeration (design note, amendment 3):
# GraphEdX's published AIDS matrix turned out to be an approximate upper bound, so it
# cannot validate a solver. The GraphEdX comparison still runs, as a REPORT rather than
# a gate, because its result belongs in the response letter.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

echo "[gates] verifying the environment before spending anything"
"${PY}" - <<'EOF'
import importlib, sys
importlib.import_module("gklearn.gedlib.libraries_import")   # MUST precede gedlibpy_gxl
g = importlib.import_module("gklearn.gedlib.gedlibpy_gxl")
import networkx as nx
p4, c4 = nx.path_graph(4), nx.cycle_graph(4)
for G in (p4, c4):
    for n in G.nodes: G.nodes[n]["l"] = "1"
    for e in G.edges: G.edges[e]["l"] = "1"
env = g.GEDEnvGXL()
i0, i1 = env.add_nx_graph(p4, ""), env.add_nx_graph(c4, "")
env.set_edit_cost("CONSTANT", edit_cost_constant=[1, 1, 0, 1, 1, 0])
env.init(init_option="EAGER_WITHOUT_SHUFFLED_COPIES")
env.set_method("BRANCH_FAST", ""); env.init_method(); env.run_method(i0, i1)
lb = env.get_lower_bound(i0, i1)
assert 0 < lb < float("inf"), f"BRANCH_FAST lb={lb}: assert 0 < v < inf on EVERY read"
assert abs(lb - 1.0) < 1e-9, f"P4 vs C4 lb should be 1.00, got {lb}"
try:
    import isalgraph; sys.exit("FATAL: isalgraph is importable; T-03 must not depend on it")
except ImportError:
    pass
print(f"[gates] GEDLIB ok (P4/C4 BRANCH_FAST lb={lb}), isalgraph correctly absent")
EOF

echo "[gates] running all gates"
run_py benchmarks.real_data.eval_setup.ged_gates \
    --gate all \
    --input-dir "${DATA_DIR}" \
    --out "${MYLOCAL}/out/gates" \
    --seed 42 \
    --timeout "${TIMEOUT_PER_PAIR}" \
    --workers "${SLURM_CPUS_PER_TASK:-4}"
RC=$?
echo "[gates] exit=${RC}"
exit ${RC}
