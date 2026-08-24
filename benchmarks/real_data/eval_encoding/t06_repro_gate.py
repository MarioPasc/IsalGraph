"""Reproduction gate: the shipped module must reproduce T-04a's corrected table.

Acceptance criterion 5. **No production distance matrix may be computed until
this passes at ``max |delta| = 0.0000``.**

The reference is ``corrected_rho_table.json``, produced by
``isalgraph.competitors.reproduce --mode table``: one recomputation, from one
script, on one seed-42 draw per dataset, under the frozen conventions ---
**column-wise** adjacency and the **shared-vocabulary** WL at ``h = 2``. It
supersedes ``competitors/README`` sections 4.1-4.2, which were a three-draw
composite differing from the single-draw artefact by up to 0.074.

The comparison is **structural and recursive**, not a hand-listed set of keys.
A hand-listed comparison silently passes when the producer stops emitting a
field --- the failure mode is indistinguishable from agreement --- so this walks
both trees, compares every numeric leaf, and reports any key present in one and
absent in the other as a violation in its own right.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Final

LOGGER: Final = logging.getLogger(__name__)

#: Exact reproduction. Not a tolerance: the same code on the same seed must
#: produce the same bits, and anything else is a behaviour change to diagnose.
MAX_ABS_DELTA: Final[float] = 0.0


class ReproGateError(Exception):
    """Raised when the reproduction gate cannot be evaluated."""


def _walk(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested JSON structure to ``path -> leaf``.

    Args:
        node: The structure to flatten.
        prefix: Accumulated path.

    Returns:
        Mapping from dotted path to leaf value.
    """
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(_walk(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.update(_walk(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = node
    return out


def compare(reference: dict[str, Any], produced: dict[str, Any]) -> dict[str, Any]:
    """Compare two corrected-table payloads leaf by leaf.

    Args:
        reference: The stored ``corrected_rho_table.json`` payload.
        produced: A fresh ``reproduce --mode table`` payload.

    Returns:
        A report with the worst delta, the offending paths, and the verdict.
    """
    ref_flat = _walk(reference.get("corrected_table", reference))
    got_flat = _walk(produced.get("corrected_table", produced))

    only_reference = sorted(set(ref_flat) - set(got_flat))
    only_produced = sorted(set(got_flat) - set(ref_flat))

    numeric_compared = 0
    worst_delta = 0.0
    worst_path = ""
    mismatches: list[dict[str, Any]] = []

    for path in sorted(set(ref_flat) & set(got_flat)):
        a, b = ref_flat[path], got_flat[path]
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                mismatches.append({"path": path, "reference": a, "produced": b, "delta": None})
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if math.isnan(float(a)) and math.isnan(float(b)):
                continue
            delta = abs(float(a) - float(b))
            numeric_compared += 1
            if delta > worst_delta:
                worst_delta, worst_path = delta, path
            if delta > MAX_ABS_DELTA:
                mismatches.append({"path": path, "reference": a, "produced": b, "delta": delta})
            continue
        if a != b:
            mismatches.append({"path": path, "reference": a, "produced": b, "delta": None})

    passed = not mismatches and not only_reference and not only_produced
    return {
        "gate": "T-06 reproduction gate (acceptance criterion 5)",
        "max_abs_delta_allowed": MAX_ABS_DELTA,
        "numeric_values_compared": numeric_compared,
        "max_abs_delta_observed": worst_delta,
        "max_abs_delta_path": worst_path,
        "keys_only_in_reference": only_reference,
        "keys_only_in_produced": only_produced,
        "mismatches": mismatches,
        "passed": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True, help="corrected_rho_table.json")
    ap.add_argument("--produced", type=Path, required=True, help="a fresh reproduce --mode table")
    ap.add_argument("--out", type=Path, required=True, help="gate report JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        0 when the gate passes, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = compare(json.loads(args.reference.read_text()), json.loads(args.produced.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"numeric values compared : {report['numeric_values_compared']}")
    print(f"max |delta| observed    : {report['max_abs_delta_observed']:.10f}")
    if report["max_abs_delta_path"]:
        print(f"  at                    : {report['max_abs_delta_path']}")
    for key in ("keys_only_in_reference", "keys_only_in_produced"):
        if report[key]:
            print(f"{key}: {report[key][:10]}")
    for m in report["mismatches"][:20]:
        print(
            f"  MISMATCH {m['path']}: ref={m['reference']} got={m['produced']} delta={m['delta']}"
        )
    print(f"\nGATE: {'PASS' if report['passed'] else 'FAIL'}  -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
