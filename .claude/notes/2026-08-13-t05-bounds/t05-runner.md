# Work log — t05-runner

## Identity

| Field | Value |
|---|---|
| Agent | `wave-t05-runner` |
| Wave | `2026-08-13-t05-bounds` |
| Model / effort | `claude-opus-5` / `xhigh` |
| Branch | `worktree-agent-ab10166d8f9bb07a0` |
| Worktree | `/home/mpascual/research/code/IsalGraph/.claude/worktrees/agent-ab10166d8f9bb07a0` |
| Base commit | `885d98d8e6b37dfeb98c4df741510fc28d4a8615` |
| Head commit | `<pending>` |
| Started / finished | `2026-08-13T00:00:00Z` / `<pending>` |
| Status | in progress |

## 1. Prompt as received

See `.claude/notes/2026-08-13-t05-bounds/agents/` for the wave brief; the verbatim delegation
prompt is reproduced at the end of this file (§12) to keep the operative sections readable.

## 2. Understanding and plan

**Restatement of the task in my own words:** T-27 proved that a GEDLIB method name without its
options string is not a specification, because upper bounds move on 91.5–93.6 % of pairs between
runs at library defaults. The production runner cannot currently express an options string per bound
end — one `_heuristic_options` feeds both. I must make `ged_backends.py`, `ged_exact_runner.py` and
`ged_merge_shards.py` express a per-end method+options specification, additively, then write
`approx_ged_crossfill.py` to join the separate role campaigns into a bracket, and prove the result
element-wise against T-27's recorded LINUX census.

**Approach chosen:** additive only. Every new constructor parameter and CLI flag defaults to the
value that reproduces T-03's behaviour byte-for-byte. `_heuristic_options` is kept as a derived
read-only property so no existing test that inspects it breaks; the two new slots `_lb_options` and
`_ub_options` carry the real strings.

**Plan as executed:**
1. Read CONTRACTS.md, T-05-design.md §1/§3.2/§4, and the three target modules.
2. `ged_backends.py`: `lb_options`/`ub_options`, `compute` mode, lazy `zero_ok`, accessor probe.
3. `ged_exact_runner.py`: the six new CLI flags, threaded into `BackendSpec.options` and shard meta.
4. `ged_merge_shards.py`: `--ged-from`, `--role`, `--seconds-role`, G4 zero-fraction check.
5. `approx_ged_crossfill.py`: new module, atomic three-file cross-fill.
6. Reproduction gate against T-27's `linux__*.npz` cells.

## 3. Changes made

(filled in as work proceeds)

## 4. Tests

(filled in as work proceeds)

## 5. Test results

(pending)

## 6. Verification beyond unit tests

(pending)

## 7. Decisions, assumptions, open questions

(pending)

## 8. Coordination

(pending)

## 9. Deliberately not done

(pending)

## 10. Risks and follow-ups

(pending)

## 11. Self-assessment against the definition of done

(pending)
