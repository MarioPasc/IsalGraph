#!/usr/bin/env bash
# T-03 validation gates. BLOCKS every production job: the launcher chains the compute
# stages with --dependency=afterok, so a non-zero exit here stops the programme.
#
# Gate 0 is re-anchored on brute-force enumeration (design note, amendment 3):
# GraphEdX's published AIDS matrix turned out to be an approximate upper bound, so it
# cannot validate a solver. The GraphEdX comparison still runs, as a REPORT rather than
# a gate, because its result belongs in the response letter.
# SLURM copies the batch script into a per-job spool directory, so BASH_SOURCE
# resolves to /var/spool/slurmd/... and a sibling path finds nothing. Source from
# REPO_DIR, which the launcher exports. Keep set -euo here too: _env.sh sets it,
# so a failure to source it would otherwise silently disable it as well.
set -euo pipefail
source "${REPO_DIR:?REPO_DIR must be exported by the launcher}/slurm/exact_ged/_env.sh"

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

# Gate 0 is NOT run here, and that is a property of the data rather than an omission.
# It compares against GraphEdX's published matrix, which lives in .pt files that need
# torch to read -- and torch is deliberately absent from this cluster, which is the whole
# reason the datasets arrive pre-serialized as CONTRACT A. Gate 0 therefore runs on the
# workstation, where the .pt files and torch both are, and its report is carried here.
# Gates 1-3 are the ones that need GEDLIB, which exists only on this cluster.
GATE0="${OUT_DIR}/gates/gate0.json"
if [[ -f "${GATE0}" ]]; then
    echo "[gates] gate 0 report present (run on the workstation):"
    "${PY}" -c "import json,sys; d=json.load(open('${GATE0}')); print('   ', {k:v for k,v in d.items() if not isinstance(v,(list,dict))})"
else
    echo "[gates] WARNING: no gate 0 report at ${GATE0}."
    echo "[gates] Run it on the workstation before trusting production:"
    echo "  python -m benchmarks.real_data.eval_setup.ged_gates --gate 0 \\"
    echo "     --input-dir <exported> --source-dir <data/source> --out <dir>"
fi

echo "[gates] running the GEDLIB-dependent gates (1, 2, 3)"
RC=0
for G in probe 1 2 3; do
    echo "--- gate ${G} ---"
    if ! run_py benchmarks.real_data.eval_setup.ged_gates \
        --gate "${G}" \
        --input-dir "${DATA_DIR}" \
        --out "${MYLOCAL}/out/gates" \
        --backend gedlib \
        --seed 42 \
        --timeout "${TIMEOUT_PER_PAIR}" \
        --workers "${SLURM_CPUS_PER_TASK:-4}"; then
        echo "[gates] gate ${G} FAILED"
        RC=1
    fi
done
echo "[gates] exit=${RC}"
exit ${RC}
