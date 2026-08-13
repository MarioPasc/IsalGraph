# Work log — <task-slug>

<!--
Location: <worktree>/.claude/notes/<WAVE_ID>/<task-slug>.md
Write this continuously while you work, not from memory at the end. Commit it as your final commit:
    docs(notes): <task-slug> work log
The orchestrator verifies every claim in this file against `git diff` and by re-running your tests,
so state what actually happened, including failures. An honest account of something that did not
work is more useful than a confident account of something that was never exercised.
-->

## Identity

| Field | Value |
|---|---|
| Agent | `<agent name>` |
| Wave | `<WAVE_ID>` |
| Model / effort | `<model>` / `<effort>` |
| Branch | `<git rev-parse --abbrev-ref HEAD>` |
| Worktree | `<git rev-parse --show-toplevel>` |
| Base commit | `<BASE_SHA>` |
| Head commit | `<final sha>` |
| Started / finished | `<ISO timestamp>` / `<ISO timestamp>` |
| Status | complete \| partial \| blocked |

## 1. Prompt as received

<!-- The delegation prompt, verbatim and complete. Do not summarise or tidy it: the orchestrator
compares what you were asked to do against what you did, and a paraphrase destroys that comparison. -->

```
<verbatim prompt>
```

## 2. Understanding and plan

**Restatement of the task in my own words:** <two or three sentences>

**Approach chosen:** <what you decided to build and how>

**Alternatives considered and rejected:**
- <alternative> — rejected because <reason>
- <alternative> — rejected because <reason>

**Plan as executed:**
1. <step>
2. <step>
3. <step>

**Deviations from the plan:** <what changed once you saw the code, and why. If none, say none.>

## 3. Changes made

**Created**
| Path | Purpose |
|---|---|
| `<path>` | <one line> |

**Modified**
| Path | Change | Reason |
|---|---|---|
| `<path>` | <what changed> | <why> |

**Removed**
| Path | Reason |
|---|---|
| `<path>` | <why it was safe to remove> |

**Commits**
| SHA | Message |
|---|---|
| `<short sha>` | `<subject>` |

<!-- This list must match `git diff --name-only <BASE_SHA>..HEAD` exactly. The orchestrator checks. -->

## 4. Tests

**Tests created or extended**
| Test | File | What it verifies | Why it matters |
|---|---|---|---|
| `<test name>` | `<path>` | <behaviour> | <the failure mode it catches> |

**Coverage of the behaviour that matters:** <which paths through the code are exercised, and which
deliberately are not>

**Not tested, and why:** <be explicit; an honest gap is manageable, a hidden one is not>

## 5. Test results

**Command:** `<exact command>`

```
<verbatim terminal output, or the summary line plus the full text of every failure>
```

**Result:** <N passed, M failed, K skipped> · **Duration:** <time> · **Run at:** `<sha>`

**Failures and their resolution:** <for each failure: cause, fix, and the output after the fix. If
anything still fails, say so plainly and explain why you could not resolve it.>

## 6. Verification beyond unit tests

<!-- Unit tests on fixtures routinely miss what real inputs expose: dtype, shape, orientation,
scale, missing fields, encoding, timing, memory. State concretely what you exercised, with numbers.
If you did none, write "None" and explain what would need to happen to do it. -->

| Circumstance | What was run | Evidence (paths, shapes, numbers, timings) | Outcome |
|---|---|---|---|
| Real data | <command> | <e.g. 3 volumes, 512×512×180, spacing 0.43/0.43/3.0, 2.1 s each> | <pass/fail + detail> |
| Edge cases | <inputs> | <observed behaviour> | |
| Failure paths | <how you induced the failure> | <error raised> | |
| Scale / performance | <size, hardware> | <time, peak memory> | |
| Environment | <OS, Python/Node version, GPU, package versions> | | |

## 7. Decisions, assumptions, open questions

**Decisions with a real trade-off:** <decision — what it costs, what it buys>

**Assumptions I proceeded on:** <assumption — what breaks if it is wrong. Every assumption here
should also have been messaged to `main` when made.>

**Open questions for the orchestrator:** <question — why it matters, what I did in the meantime>

## 8. Coordination

**Messages sent:** <to whom, about what, and the outcome>

**Messages received and how they changed the work:** <…>

**Contracts I depend on and confirmed unchanged:** <…>

## 9. Deliberately not done

<!-- Things a reader might expect to find and will not. Naming them prevents the orchestrator from
reading absence as oversight. -->

- <item> — <out of scope / deferred / blocked by X>

## 10. Risks and follow-ups

| Item | Severity | Detail | Suggested owner |
|---|---|---|---|
| <risk or follow-up> | low \| medium \| high | <what could go wrong, or what remains> | <next wave / orchestrator / user> |

## 11. Self-assessment against the definition of done

| # | Criterion | Met | Evidence |
|---|---|---|---|
| 1 | <criterion as given in the prompt> | yes \| partial \| no | <test name, command output, file> |
| 2 | | | |

**Overall:** <one paragraph: what you are confident about, what you are not, and what the
orchestrator should scrutinise first.>
