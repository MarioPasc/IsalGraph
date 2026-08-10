# Synthetic validation suite

Launcher for the synthetic benchmarks in `benchmarks/synthetic_data/`. Formerly the top-level
`slurm/` directory.

**None of these benchmarks produces a figure or table in the submitted manuscript.** They verify
the mathematical properties asserted in `src/isalgraph/core/README.md` (round-trip correctness,
canonical invariance, string-length scaling, Levenshtein-GED agreement on synthetic pairs).

For anything that goes into the paper, use `experiments/paper_pipeline/`. See
`experiments/README.md` for the full artifact registry.

## Config schema

This suite reads `benchmarks.<name>` blocks from its own `config.yaml`. That is a **different
schema** from `paper_pipeline/config.yaml`, which uses `steps.<name>` and injects
`ISALGRAPH_RUN_DIR` into every worker. The two are not interchangeable.

## Superseded workers

Five workers here duplicate paper-pipeline steps and are older:

| Here (older) | Canonical replacement |
|---|---|
| `workers/eval_setup_slurm.sh` | `paper_pipeline/workers/step1_eval_setup.sh` |
| `workers/eval_correlation_slurm.sh` | `paper_pipeline/workers/step2a_eval_correlation.sh` |
| `workers/eval_computational_slurm.sh` | `paper_pipeline/workers/step2b_eval_computational.sh` |
| `workers/eval_encoding_slurm.sh` | `paper_pipeline/workers/step2c_eval_encoding.sh` |
| `workers/topology_complexity_figs_slurm.sh` | `paper_pipeline/workers/step3b_topology_figs.sh` |

They are kept so that run directories produced before the pipeline consolidation remain
interpretable. Do not launch paper steps from here.

`workers/eval_embedding_{slurm,login}.sh` has no pipeline equivalent because the embedding track
produces no paper artifact.

## Usage

```bash
bash experiments/synthetic_suite/launch.sh --dry-run
bash experiments/synthetic_suite/launch.sh                            # all enabled
bash experiments/synthetic_suite/launch.sh --benchmark canonical_invariance
```
