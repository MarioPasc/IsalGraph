"""Manifest and contract-conformance gate over the T-06 encoding campaign output.

Proves acceptance criterion A1 -- every cohort graph has an encoding under every
applicable representation, or is flagged censored with its fallback -- and emits
the per-``(representation, dataset)`` status breakdown that sets the
pre-registration's ``c`` term.

The tool *reports*; it does not decide. In particular it splits ``error_kind``
into scope refusals and budget failures and prints both, because summing them
would let a policy guard masquerade as a performance outcome -- the defect
diagnosed in ``T-06-design.md`` 11.1, where a completion rate measured through
the competitors registry was 100 % scope policy and 0 % budget. Applying the
reduction rule to those counts is the orchestrator's job, per
``preregistration.md`` 5.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

LOGGER: Final = logging.getLogger(__name__)

#: Cohort sizes frozen by the ticket. A mismatch is stop-and-ask condition 3.
EXPECTED_COHORT: Final[dict[str, int]] = {"suite1": 5_350, "suite2": 16_370}

#: ``error_kind`` values that denote a *scope refusal* -- the representation
#: declined to attempt an encode by policy, before doing any work. These must
#: never be summed with budget failures when computing completion rates.
SCOPE_ERROR_KINDS: Final[frozenset[str]] = frozenset({"SuiteScopeError"})

#: ``error_kind`` values that denote a genuine *budget* or resource failure --
#: the representation tried and could not finish. ``MinDfsBudgetExceeded`` and
#: ``WorkerExit`` were added 2026-08-23 after the production campaign surfaced
#: them as unclassified; the gate flagging them rather than silently bucketing
#: them is why they were noticed.
BUDGET_ERROR_KINDS: Final[frozenset[str]] = frozenset(
    {
        "AGMBudgetExceeded",
        "EncodeBudgetExceeded",
        "MinDfsBudgetExceeded",
        "TimeoutError",
        "MemoryError",
    }
)

#: ``error_kind`` values denoting an *infrastructure* failure -- the worker
#: process died rather than the algorithm declining. Deliberately kept out of
#: :data:`BUDGET_ERROR_KINDS`: a crashed worker is not evidence that a
#: representation cannot encode a graph, and folding it into the budget count
#: would let a harness defect be read as a capability limit. It still counts
#: against completion, because no encoding was produced either way.
INFRASTRUCTURE_ERROR_KINDS: Final[frozenset[str]] = frozenset({"WorkerExit"})


class ManifestError(Exception):
    """Raised when the encoding campaign output violates a frozen contract."""


@dataclass
class CellReport:
    """Conformance and accounting summary for one ``(suite, dataset, representation)`` cell.

    Attributes:
        suite: Suite key, ``suite1`` or ``suite2``.
        dataset: Dataset key.
        representation: Backend name.
        n_graphs: Number of rows in the file.
        n_ok: Rows with ``status == "ok"``.
        n_censored: Rows with ``status == "censored"`` (D14 fallback retained).
        n_error: Rows with ``status == "error"``.
        n_scope_errors: Error rows whose kind is a scope refusal.
        n_budget_errors: Error rows whose kind is a budget or resource failure.
        n_infra_errors: Error rows whose kind is an infrastructure failure.
        n_unclassified_errors: Error rows whose kind matches none of the sets.
        error_kinds: Raw ``error_kind`` histogram over error rows.
        symbol_sep: The frozen separator from ``metadata``.
        build_hash: ``isalgraph_build_hash`` from ``metadata``.
        src_commit: ``src_commit`` from ``metadata``.
        code_commit: ``code_commit`` from ``metadata``.
        encode_budget_s: Per-graph budget the cell ran under.
        violations: Contract violations found in this cell; empty means conformant.
    """

    suite: str
    dataset: str
    representation: str
    n_graphs: int
    n_ok: int
    n_censored: int
    n_error: int
    n_scope_errors: int
    n_budget_errors: int
    n_infra_errors: int
    n_unclassified_errors: int
    error_kinds: dict[str, int]
    symbol_sep: str
    build_hash: str | None
    src_commit: str | None
    code_commit: str | None
    encode_budget_s: float | None
    violations: list[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        """Fraction of graphs that produced a usable encoding.

        Censored rows count as completed: D14 retains them with a greedy-min
        fallback string, so a distance consumer has an operand. Only ``error``
        rows are missing.
        """
        if self.n_graphs == 0:
            return 0.0
        return (self.n_ok + self.n_censored) / self.n_graphs


def _classify_errors(kinds: Counter[str]) -> tuple[int, int, int, int]:
    """Split an ``error_kind`` histogram into scope, budget, infra and unknown counts.

    Args:
        kinds: Histogram of ``error_kind`` values over error rows.

    Returns:
        Quadruple ``(n_scope, n_budget, n_infra, n_unclassified)``.
    """
    n_scope = sum(v for k, v in kinds.items() if k in SCOPE_ERROR_KINDS)
    n_budget = sum(v for k, v in kinds.items() if k in BUDGET_ERROR_KINDS)
    n_infra = sum(v for k, v in kinds.items() if k in INFRASTRUCTURE_ERROR_KINDS)
    n_other = sum(kinds.values()) - n_scope - n_budget - n_infra
    return n_scope, n_budget, n_infra, n_other


def _check_status_invariants(
    status: np.ndarray,
    encoding: np.ndarray,
    length: np.ndarray,
    fallback: np.ndarray,
    error_kind: np.ndarray,
) -> list[str]:
    """Check the CONTRACTS 3.2 status table row by row.

    Args:
        status: Per-graph status strings.
        encoding: Per-graph encoded symbol sequences.
        length: Per-graph symbol counts.
        fallback: Per-graph fallback flags.
        error_kind: Per-graph exception class names.

    Returns:
        Human-readable violation descriptions; empty when conformant.
    """
    out: list[str] = []
    is_ok = status == "ok"
    is_cens = status == "censored"
    is_err = status == "error"

    unknown = ~(is_ok | is_cens | is_err)
    if unknown.any():
        bad = sorted(set(status[unknown].tolist()))
        out.append(f"unknown status values {bad} on {int(unknown.sum())} rows")

    if (fallback[is_ok]).any():
        out.append(f"{int(fallback[is_ok].sum())} ok rows have fallback_used=True")
    if is_cens.any() and not fallback[is_cens].all():
        n = int((~fallback[is_cens]).sum())
        out.append(f"{n} censored rows have fallback_used=False (CONTRACTS 3.2)")
    if is_cens.any() and (encoding[is_cens] == "").any():
        n = int((encoding[is_cens] == "").sum())
        out.append(f"{n} censored rows have an EMPTY encoding -- 3.2 requires the fallback string")
    if is_err.any():
        if (encoding[is_err] != "").any():
            out.append("error rows carry a non-empty encoding")
        if (length[is_err] != -1).any():
            out.append("error rows do not carry length == -1")
    if (length[is_ok | is_cens] < 0).any():
        out.append("ok/censored rows carry a negative length")

    # error_kind is defined as the exception class name when status == "error",
    # else ''. A non-empty kind on a non-error row would let a downstream
    # `error_kind != ''` test count censored rows as errors -- and that test is
    # how c gets computed.
    misplaced = (~is_err) & (error_kind != "")
    if misplaced.any():
        out.append(
            f"{int(misplaced.sum())} non-error rows carry a non-empty error_kind "
            f"{sorted(set(error_kind[misplaced].tolist()))}"
        )
    if is_err.any() and (error_kind[is_err] == "").any():
        n = int((error_kind[is_err] == "").sum())
        out.append(f"{n} error rows carry an EMPTY error_kind -- the scope/budget split needs it")
    return out


def _check_symbol_invariant(
    encoding: np.ndarray, length: np.ndarray, status: np.ndarray, sep: str
) -> list[str]:
    """Check CONTRACTS 3.1: ``length`` is the symbol count, never the character count.

    Args:
        encoding: Per-graph encoded symbol sequences.
        length: Per-graph symbol counts.
        status: Per-graph status strings.
        sep: The frozen ``symbol_sep`` for this representation.

    Returns:
        Human-readable violation descriptions; empty when conformant.
    """
    encoded = (status == "ok") | (status == "censored")
    if not encoded.any():
        return []
    texts = encoding[encoded]
    if sep:
        calc = np.fromiter((len(t.split(sep)) for t in texts), dtype=np.int64, count=len(texts))
    else:
        calc = np.fromiter((len(t) for t in texts), dtype=np.int64, count=len(texts))
    bad = int((calc != length[encoded]).sum())
    if bad:
        return [f"{bad} rows violate length == len(encoding.split({sep!r})) -- CONTRACTS 3.1"]
    return []


def verify_cell(path: Path) -> CellReport:
    """Read one encoding ``.npz`` and check it against the frozen contracts.

    Args:
        path: Path to ``{dataset}__{representation}.npz``.

    Returns:
        The cell's conformance and accounting report.

    Raises:
        ManifestError: If the file cannot be parsed as an encoding cell.
    """
    suite = path.parent.name
    stem = path.stem
    if "__" not in stem:
        raise ManifestError(f"{path} does not follow {{dataset}}__{{representation}}.npz")
    dataset, representation = stem.split("__", 1)

    with np.load(path, allow_pickle=True) as z:
        missing = {"graph_ids", "encoding", "length", "status", "error_kind", "metadata"} - set(
            z.files
        )
        if missing:
            raise ManifestError(f"{path} is missing required keys {sorted(missing)}")
        meta = json.loads(str(z["metadata"]))
        graph_ids = np.asarray(z["graph_ids"]).ravel()
        status = np.asarray(z["status"]).ravel().astype(str)
        encoding = np.asarray(z["encoding"]).ravel().astype(str)
        length = np.asarray(z["length"]).ravel().astype(np.int64)
        error_kind = np.asarray(z["error_kind"]).ravel().astype(str)
        fallback = (
            np.asarray(z["fallback_used"]).ravel().astype(bool)
            if "fallback_used" in z.files
            else np.zeros(len(status), dtype=bool)
        )

    sep = str(meta.get("symbol_sep", ""))
    violations: list[str] = []

    n = len(graph_ids)
    if len({len(status), len(encoding), len(length), len(error_kind), n}) != 1:
        violations.append("column lengths disagree")
    if len(set(graph_ids.tolist())) != n:
        violations.append(f"graph_ids are not unique ({n - len(set(graph_ids.tolist()))} repeats)")

    violations += _check_status_invariants(status, encoding, length, fallback, error_kind)
    violations += _check_symbol_invariant(encoding, length, status, sep)

    for key in ("isalgraph_build_hash", "src_commit"):
        if not meta.get(key):
            violations.append(
                f"metadata.{key} is absent -- CONTRACTS 5 calls this unusable evidence"
            )

    kinds = Counter(error_kind[status == "error"].tolist())
    n_scope, n_budget, n_infra, n_other = _classify_errors(kinds)

    return CellReport(
        suite=suite,
        dataset=dataset,
        representation=representation,
        n_graphs=n,
        n_ok=int((status == "ok").sum()),
        n_censored=int((status == "censored").sum()),
        n_error=int((status == "error").sum()),
        n_scope_errors=n_scope,
        n_budget_errors=n_budget,
        n_infra_errors=n_infra,
        n_unclassified_errors=n_other,
        error_kinds=dict(kinds),
        symbol_sep=sep,
        build_hash=meta.get("isalgraph_build_hash"),
        src_commit=meta.get("src_commit"),
        code_commit=meta.get("code_commit"),
        encode_budget_s=meta.get("encode_budget_s"),
        violations=violations,
    )


def build_manifest(root: Path) -> dict[str, object]:
    """Scan an ``encodings/`` tree and build the campaign manifest.

    Args:
        root: The ``encodings/`` directory holding ``suite1/`` and ``suite2/``.

    Returns:
        A JSON-serialisable manifest with per-cell reports and cohort checks.
    """
    cells: list[CellReport] = []
    for suite in sorted(EXPECTED_COHORT):
        d = root / suite
        if not d.is_dir():
            LOGGER.warning("suite directory absent: %s", d)
            continue
        for p in sorted(d.glob("*.npz")):
            cells.append(verify_cell(p))

    cohort: dict[str, dict[str, object]] = {}
    for suite, expected in EXPECTED_COHORT.items():
        per_rep: dict[str, int] = {}
        for c in cells:
            if c.suite == suite:
                per_rep[c.representation] = per_rep.get(c.representation, 0) + c.n_graphs
        mismatched = {r: n for r, n in per_rep.items() if n != expected}
        cohort[suite] = {
            "expected": expected,
            "graphs_by_representation": per_rep,
            "representations": len(per_rep),
            "mismatched": mismatched,
        }

    violations = {
        f"{c.suite}/{c.dataset}/{c.representation}": c.violations for c in cells if c.violations
    }
    return {
        "schema_version": "t06.1",
        "ticket": "T-06",
        "n_cells": len(cells),
        "cohort": cohort,
        "provenance": {
            "build_hash": sorted({c.build_hash for c in cells if c.build_hash}),
            "src_commit": sorted({c.src_commit for c in cells if c.src_commit}),
            "encode_budget_s": sorted({c.encode_budget_s for c in cells if c.encode_budget_s}),
        },
        "violations": violations,
        "cells": [asdict(c) | {"completion_rate": c.completion_rate} for c in cells],
    }


def _print_summary(manifest: dict[str, object]) -> None:
    """Print the human-readable gate summary.

    Args:
        manifest: The manifest returned by :func:`build_manifest`.
    """
    cohort = manifest["cohort"]
    assert isinstance(cohort, dict)
    for suite, info in cohort.items():
        assert isinstance(info, dict)
        bad = info["mismatched"]
        assert isinstance(bad, dict)
        flag = "OK" if not bad else "MISMATCH"
        print(
            f"[{flag}] {suite}: {info['representations']} representations, "
            f"expected {info['expected']}"
        )
        for rep, n in sorted(bad.items()):
            print(f"        {rep:22s} n={n} != {info['expected']}")

    cells = manifest["cells"]
    assert isinstance(cells, list)
    incomplete = [c for c in cells if c["completion_rate"] < 1.0]
    print(f"\ncells: {manifest['n_cells']}, incomplete: {len(incomplete)}")
    for c in sorted(incomplete, key=lambda x: float(x["completion_rate"])):
        print(
            f"  {c['suite']}/{c['dataset']}/{c['representation']:16s} "
            f"rate={c['completion_rate']:.4f} ok={c['n_ok']} cens={c['n_censored']} "
            f"err={c['n_error']} (scope={c['n_scope_errors']} budget={c['n_budget_errors']} "
            f"infra={c['n_infra_errors']} other={c['n_unclassified_errors']}) "
            f"{c['error_kinds'] or ''}"
        )

    prov = manifest["provenance"]
    assert isinstance(prov, dict)
    print(f"\nprovenance: build_hash={prov['build_hash']} budget={prov['encode_budget_s']}")
    print(f"            src_commit={prov['src_commit']}")

    viol = manifest["violations"]
    assert isinstance(viol, dict)
    if viol:
        print(f"\n🔴 CONTRACT VIOLATIONS in {len(viol)} cells:")
        for cell, msgs in sorted(viol.items()):
            for m in msgs:
                print(f"  {cell}: {m}")
    else:
        print("\ncontract conformance: 0 violations")


def main() -> int:
    """Entry point.

    Returns:
        Process exit status: 0 when the gate passes, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="the encodings/ directory")
    ap.add_argument("--out", type=Path, default=None, help="write manifest.json here")
    ap.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless every representation covers its full cohort",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = build_manifest(args.root)
    _print_summary(manifest)

    out = args.out if args.out is not None else args.root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nwrote {out}")

    viol = manifest["violations"]
    assert isinstance(viol, dict)
    rc = 1 if viol else 0
    if args.require_complete:
        cohort = manifest["cohort"]
        assert isinstance(cohort, dict)
        for info in cohort.values():
            assert isinstance(info, dict)
            if info["mismatched"]:
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
