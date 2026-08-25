"""T-06 encoding campaign: one ``.npz`` per ``(suite, dataset, representation)``.

Serves **AE.4a**, R3.6a and R1.1. Claim A's bit counts are computed directly
from what this writes, and Claim B's distance matrices are computed from its
encoding strings, so the schema in ``CONTRACTS.md`` §3 is the contract with the
rest of the wave.

Three things this module is responsible for that a plain loop is not.

**The budget is enforced by killing a process.** See
``t06_encode_worker``'s docstring: a signal-based timeout does not interrupt the
C++ engine, and the failure mode is a silent 25-minute hang. The driver reads
the worker's stdout with a per-line deadline and kills the child when a line is
late, so the wall clock is real. Restarting from the next index amortises the
interpreter start-up over a chunk instead of paying it per graph.

**D14 is applied here and only here.** A graph whose IsalGraph encoding exceeds
the budget is **not dropped**: it re-enters with its greedy-min string, with
``status="censored"``, ``fallback_used=True`` and a non-empty ``encoding``. The
graphs that time out are exactly those with the largest automorphism groups, so
dropping them would delete the hardest cases from the cohort the paper reports
on. Both a primary arm (fallback included) and a complete-case sensitivity arm
are recoverable from the output, because ``fallback_used`` marks the difference.

**One file per (suite, dataset, representation), never one per graph.** The
cluster's quota is a file count, not a size, and a per-graph layout would exhaust
it.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

import numpy as np

from benchmarks.real_data.eval_encoding.t06_cohort import Cohort, load_cohort
from benchmarks.real_data.eval_encoding.t06_encode_worker import (
    ISALGRAPH_ARMS,
    error_family,
    symbol_sep,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "t06.1"
TICKET = "T-06"
WAVE = "2026-08-16-t06-recompute"
SEED = 42
ENCODE_BUDGET_S = 300.0

#: The checkout ``import isalgraph`` actually resolves to. The editable finder
#: is path-pinned there and outranks ``PYTHONPATH``, so a run inside a worktree
#: still executes this tree's ``src/``. Recording its commit is the only way to
#: detect afterwards that a file was produced against another branch's code.
INSTALLED_SRC_CHECKOUT = Path("/home/mpascual/research/code/IsalGraph")

#: Seconds allowed on top of the budget for interpreter start-up and cohort
#: load before the first record. Generous on purpose: a false kill here would be
#: recorded as a censoring and would inflate the headline censoring rate.
STARTUP_GRACE_S = 90.0

#: Seconds allowed on top of the budget between two consecutive records.
LINE_GRACE_S = 15.0

#: The failure family that makes a graph *censored* rather than failed, and
#: therefore eligible for the D14 greedy-min fallback. An internal-cap failure,
#: a scope refusal or a missing dependency is a different fact about the
#: representation and must not be laundered into a censoring.
CENSORING_FAMILY = "wall_clock"

#: ``error_kind`` the driver writes for a graph it killed itself.
KILLED_KIND = "Killed"

REPRESENTATIONS: tuple[str, ...] = (
    "graph6",
    "sparse6",
    "nauty_graph6",
    "adjacency",
    "agm_cam",
    "min_dfs",
    "isalgraph_pruned",
    "isalgraph_canonical",
    "isalgraph_exhaustive",
    "isalgraph_greedy",
    "wl_subtree",
    "size_null",
    "sparse6_nauty",
)


class CampaignError(RuntimeError):
    """Raised when the campaign cannot produce a conforming file."""


@dataclass(frozen=True, slots=True)
class EncodeConfig:
    """Everything one campaign cell needs.

    Attributes:
        suite: ``"suite1"`` or ``"suite2"``.
        dataset: Suite-local dataset key.
        representation: Backend name.
        out_dir: Root of the output tree; the file lands under
            ``encodings/{suite}/``.
        cohort_root: Export root, or ``None`` for the default.
        budget_s: Per-graph wall clock, enforced by the kill.
        jobs: Concurrent worker processes.
        limit: Development cap on the number of graphs.
        agm_search_nodes: Override for AGM's branch-and-bound cap.
        min_dfs_max_projections: Override for min-DFS's memory cap.
    """

    suite: str
    dataset: str
    representation: str
    out_dir: Path
    cohort_root: Path | None = None
    budget_s: float = ENCODE_BUDGET_S
    jobs: int = 1
    limit: int | None = None
    agm_search_nodes: int | None = None
    min_dfs_max_projections: int | None = None


def _pump(stream: IO[str], sink: queue.Queue[str | None]) -> None:
    """Move lines from *stream* onto *sink*, terminating with ``None``."""
    try:
        for line in stream:
            sink.put(line)
    finally:
        sink.put(None)


def _worker_command(cfg: EncodeConfig, indices_file: Path, mode: str) -> list[str]:
    """Build the worker argv for one chunk.

    Args:
        cfg: The cell configuration.
        indices_file: JSON file holding the indices still to process.
        mode: ``"primary"`` or ``"fallback"``.

    Returns:
        The argv.
    """
    argv = [
        sys.executable,
        "-m",
        "benchmarks.real_data.eval_encoding.t06_encode_worker",
        "--suite",
        cfg.suite,
        "--dataset",
        cfg.dataset,
        "--representation",
        cfg.representation,
        "--mode",
        mode,
        "--budget-s",
        str(cfg.budget_s),
        "--indices-file",
        str(indices_file),
    ]
    if cfg.cohort_root is not None:
        argv += ["--cohort-root", str(cfg.cohort_root)]
    if cfg.limit is not None:
        argv += ["--limit", str(cfg.limit)]
    if cfg.agm_search_nodes is not None:
        argv += ["--agm-search-nodes", str(cfg.agm_search_nodes)]
    if cfg.min_dfs_max_projections is not None:
        argv += ["--min-dfs-max-projections", str(cfg.min_dfs_max_projections)]
    return argv


def _run_chunk(
    cfg: EncodeConfig, indices: Sequence[int], mode: str
) -> tuple[dict[int, dict[str, Any]], bool]:
    """Run one worker process over *indices*, killing it on a late record.

    Args:
        cfg: The cell configuration.
        indices: Indices still to process, in order.
        mode: ``"primary"`` or ``"fallback"``.

    Returns:
        ``(records, killed)``. ``killed`` is ``True`` when the deadline fired,
        which attributes the failure to the first index with no record.
    """
    with tempfile.TemporaryDirectory() as tmp:
        indices_file = Path(tmp) / "indices.json"
        indices_file.write_text(json.dumps(list(indices)))
        argv = _worker_command(cfg, indices_file, mode)
        return _consume(argv, cfg.budget_s)


def _consume(argv: Sequence[str], budget_s: float) -> tuple[dict[int, dict[str, Any]], bool]:
    """Read a worker's records under a per-line deadline.

    Args:
        argv: The worker argv.
        budget_s: Per-graph wall clock.

    Returns:
        ``(records, killed)``.
    """
    proc = subprocess.Popen(  # noqa: S603 - argv is built here, never from input
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    sink: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, sink), daemon=True).start()

    records: dict[int, dict[str, Any]] = {}
    deadline = budget_s + STARTUP_GRACE_S
    while True:
        try:
            line = sink.get(timeout=deadline)
        except queue.Empty:
            proc.kill()
            proc.wait()
            return records, True
        if line is None:
            proc.wait()
            return records, False
        record = json.loads(line)
        records[int(record["index"])] = record
        deadline = budget_s + LINE_GRACE_S


def _killed_record(cohort: Cohort, index: int, budget_s: float, mode: str) -> dict[str, Any]:
    """Build the record for the graph the parent killed.

    Args:
        cohort: The loaded cohort.
        index: Position in cohort order.
        budget_s: The budget it exceeded.
        mode: The mode it was killed in.

    Returns:
        A failed record with ``seconds == -1``, which is how a kill is told
        apart from a raised exception downstream.
    """
    return {
        "index": index,
        "graph_id": str(cohort.graph_ids[index]),
        "status": "error",
        "error_kind": KILLED_KIND,
        "encoding": "",
        "length": -1,
        "entropy_bits": None,
        "realised_bits": None,
        "fallback_used": False,
        "seconds": -1.0,
        "message": f"killed by the parent after {budget_s} s in {mode} mode",
    }


def _drive(
    cfg: EncodeConfig, cohort: Cohort, indices: Sequence[int], mode: str
) -> dict[int, dict[str, Any]]:
    """Process *indices*, restarting the worker past each killed graph.

    Args:
        cfg: The cell configuration.
        cohort: The loaded cohort.
        indices: Indices to process, in order.
        mode: ``"primary"`` or ``"fallback"``.

    Returns:
        One record per requested index.
    """
    records: dict[int, dict[str, Any]] = {}
    pending = list(indices)
    while pending:
        produced, killed = _run_chunk(cfg, pending, mode)
        records.update(produced)
        pending = [index for index in pending if index not in produced]
        if not pending:
            break
        head = pending.pop(0)
        records[head] = _blocked_record(cohort, head, cfg.budget_s, mode, killed)
        LOGGER.warning(
            "%s/%s/%s index %d: %s",
            cfg.suite,
            cfg.dataset,
            cfg.representation,
            head,
            records[head]["message"],
        )
    return records


def _blocked_record(
    cohort: Cohort, index: int, budget_s: float, mode: str, killed: bool
) -> dict[str, Any]:
    """Record for a graph the worker never reported on.

    Args:
        cohort: The loaded cohort.
        index: Position in cohort order.
        budget_s: The budget in force.
        mode: The mode it was attempted in.
        killed: ``True`` when the deadline fired, ``False`` when the worker
            died without emitting -- a crash or an out-of-memory kill, which is
            an ``other`` failure and gets no D14 fallback.

    Returns:
        The record.
    """
    record = _killed_record(cohort, index, budget_s, mode)
    if not killed:
        record["error_kind"] = "WorkerExit"
        record["message"] = "worker exited without emitting a record (crash or OOM)"
    return record


def _chunks(indices: Sequence[int], jobs: int) -> list[list[int]]:
    """Split *indices* into at most *jobs* contiguous slices."""
    if jobs <= 1 or len(indices) <= 1:
        return [list(indices)]
    size = -(-len(indices) // jobs)
    return [list(indices[i : i + size]) for i in range(0, len(indices), size)]


def _drive_parallel(
    cfg: EncodeConfig, cohort: Cohort, indices: Sequence[int], mode: str
) -> dict[int, dict[str, Any]]:
    """:func:`_drive` fanned out over ``cfg.jobs`` contiguous slices."""
    slices = _chunks(indices, cfg.jobs)
    if len(slices) == 1:
        return _drive(cfg, cohort, slices[0], mode)
    merged: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(slices)) as pool:
        for part in pool.map(lambda s: _drive(cfg, cohort, s, mode), slices):
            merged.update(part)
    return merged


def _apply_d14(
    cfg: EncodeConfig, cohort: Cohort, records: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Retain every censored IsalGraph graph with its greedy-min string.

    Args:
        cfg: The cell configuration.
        cohort: The loaded cohort.
        records: The primary-pass records, mutated in place.

    Returns:
        The same mapping, with censored graphs rewritten.
    """
    if cfg.representation not in ISALGRAPH_ARMS:
        return records
    censored = sorted(
        index
        for index, record in records.items()
        if record["status"] == "error" and error_family(record["error_kind"]) == CENSORING_FAMILY
    )
    if not censored:
        return records
    LOGGER.info("D14: %d censored graphs enter with the greedy-min string", len(censored))
    for index, fallback in _drive_parallel(cfg, cohort, censored, "fallback").items():
        _stamp_fallback(records[index], fallback)
    return records


def _stamp_fallback(target: dict[str, Any], fallback: dict[str, Any]) -> None:
    """Overwrite a censored record with its substitute result.

    A censored graph never leaves with an empty encoding; if the fallback itself
    failed, the record stays an ``error`` so the invariant
    ``censored => encoding != ''`` cannot be violated.

    The fallback's ``message`` is carried onto the target rather than dropped.
    It names the cascade tier that produced the string -- ``pruned`` or
    ``greedy`` -- and those are not the same datum: a pruned-tier row is still a
    canonical form and stays inside the completeness theorem, a greedy-tier row
    does not. A censoring rate that conflates them is not interpretable.

    Args:
        target: The censored primary record.
        fallback: The substitute record.
    """
    if fallback["status"] != "ok" or not fallback["encoding"]:
        target["message"] += f" | fallback cascade also failed: {fallback['message']}"
        return
    target.update(
        status="censored",
        error_kind="",
        fallback_used=True,
        encoding=fallback["encoding"],
        length=fallback["length"],
        entropy_bits=fallback["entropy_bits"],
        realised_bits=fallback["realised_bits"],
        message=fallback["message"],
    )


def _git_head(repo: Path) -> str:
    """Return ``git rev-parse HEAD`` for *repo*, or ``"unknown"``."""
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def build_metadata(cfg: EncodeConfig, n_graphs: int, notes: str = "") -> str:
    """Assemble the ``CONTRACTS.md`` §5 metadata block.

    ``isalgraph_build_hash`` and ``src_commit`` are not optional: they are the
    only way to detect afterwards that a run picked up another branch's ``src/``.

    Args:
        cfg: The cell configuration.
        n_graphs: Rows in the file.
        notes: Free text appended to the fixed note.

    Returns:
        The JSON string to store as the 0-d ``metadata`` array.
    """
    import isalgraph

    sep = symbol_sep(cfg.representation)
    fixed = (
        "encoding is symbol_sep.join(Encoding.symbols) per CONTRACTS 3.1; length "
        "is always the symbol count. sparse6 and sparse6_nauty therefore drop the "
        "leading ':' format marker, which is not a symbol -- prepend it to decode."
    )
    payload = {
        "symbol_sep": sep,
        "schema_version": SCHEMA_VERSION,
        "ticket": TICKET,
        "wave": WAVE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "suite": cfg.suite,
        "dataset": cfg.dataset,
        "representation": cfg.representation,
        "metric": None,
        "n_graphs": n_graphs,
        "isalgraph_engine": isalgraph.engine(),
        "isalgraph_build_hash": str(isalgraph.build_info().get("build_hash", "")),
        "code_commit": _git_head(Path(__file__).resolve().parents[3]),
        "src_commit": _git_head(INSTALLED_SRC_CHECKOUT),
        "encode_budget_s": cfg.budget_s,
        "notes": f"{fixed}. {notes}".strip(),
    }
    return json.dumps(payload, sort_keys=True)


def _column(records: Iterable[dict[str, Any]], key: str, default: Any) -> list[Any]:
    """Extract one field from every record, substituting *default* for ``None``."""
    return [default if record[key] is None else record[key] for record in records]


def fallback_tier_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Tally which cascade tier produced each substituted string.

    The tier cannot become a column: ``schema._require_keys`` rejects a file
    whose key set is not **exactly** ``ENCODINGS_KEYS``, so an extra array would
    make every distance cell for this arm unreadable. It goes into
    ``metadata.notes`` instead, which is free text and already in the contract.

    Args:
        records: Records in any order.

    Returns:
        Tier name -> count, over rows with ``fallback_used``. A pruned-tier row
        is still a canonical form; a greedy-tier row is not, so the two are
        reported apart rather than as one censoring rate.
    """
    out: dict[str, int] = {}
    for record in records:
        if not record.get("fallback_used"):
            continue
        message = str(record.get("message", ""))
        tier = message.split("fallback_tier=", 1)[1] if "fallback_tier=" in message else "greedy"
        out[tier] = out.get(tier, 0) + 1
    return out


def to_arrays(
    cfg: EncodeConfig, cohort: Cohort, records: dict[int, dict[str, Any]]
) -> dict[str, np.ndarray]:
    """Turn per-graph records into the ``CONTRACTS.md`` §3 arrays.

    Args:
        cfg: The cell configuration.
        cohort: The loaded cohort, which supplies cohort order and the carried
            node and edge counts.
        records: One record per index.

    Returns:
        The arrays, keyed exactly as §3 specifies.

    Raises:
        CampaignError: If a graph has no record, or a D14 invariant is broken.
    """
    missing = [index for index in range(len(cohort)) if index not in records]
    if missing:
        raise CampaignError(
            f"{len(missing)} graphs have no record (first {missing[:5]}); "
            f"refusing to write a partial file"
        )
    ordered = [records[index] for index in range(len(cohort))]
    _check_invariants(ordered, symbol_sep(cfg.representation))
    tiers = fallback_tier_counts(ordered)
    notes = f"fallback_tiers={json.dumps(tiers, sort_keys=True)}" if tiers else ""
    return {
        "graph_ids": cohort.graph_ids.astype("<U16"),
        "node_counts": cohort.node_counts.astype(np.int32),
        "edge_counts": cohort.edge_counts.astype(np.int32),
        "encoding": np.array(_column(ordered, "encoding", ""), dtype=np.str_),
        "length": np.array(_column(ordered, "length", -1), dtype=np.int32),
        "error_kind": np.array(_column(ordered, "error_kind", ""), dtype="<U32"),
        "entropy_bits": np.array(_column(ordered, "entropy_bits", np.nan), dtype=np.float64),
        "realised_bits": np.array(_column(ordered, "realised_bits", np.nan), dtype=np.float64),
        "status": np.array(_column(ordered, "status", "error"), dtype="<U12"),
        "fallback_used": np.array(_column(ordered, "fallback_used", False), dtype=bool),
        "seconds": np.array(_column(ordered, "seconds", -1.0), dtype=np.float32),
        "metadata": np.array(build_metadata(cfg, len(cohort), notes), dtype=np.str_),
    }


def _check_invariants(records: Sequence[dict[str, Any]], sep: str) -> None:
    """Assert the D14 and §3/§3.1 invariants before anything is written.

    Args:
        records: Records in cohort order.
        sep: The file's ``symbol_sep``.

    Raises:
        CampaignError: On any violation. Writing a file that breaks these is
            worse than failing, because the breakage is invisible downstream.
    """
    for record in records:
        _check_status(record)
        _check_symbols(record, sep)


def _check_status(record: dict[str, Any]) -> None:
    """Assert the status invariants for one record."""
    status = record["status"]
    if status not in ("ok", "censored", "fallback", "error"):
        raise CampaignError(f"index {record['index']} has unknown status {status!r}")
    if status == "censored" and not (record["fallback_used"] and record["encoding"]):
        raise CampaignError(
            f"D14 violated at index {record['index']}: a censored graph must carry "
            f"fallback_used=True and a non-empty greedy-min encoding, never be dropped"
        )
    if status == "ok" and record["length"] < 0:
        raise CampaignError(f"index {record['index']} is ok but has length < 0")
    if status != "error" and record["error_kind"]:
        raise CampaignError(f"index {record['index']} is {status} but carries an error_kind")


def _check_symbols(record: dict[str, Any], sep: str) -> None:
    """Assert §3.1: the encoding splits back into exactly ``length`` symbols."""
    if not record["encoding"]:
        return
    recovered = len(record["encoding"].split(sep)) if sep else len(record["encoding"])
    if recovered != record["length"]:
        raise CampaignError(
            f"index {record['index']}: encoding splits into {recovered} symbols but "
            f"length is {record['length']}; the consumer would silently disagree"
        )


def output_path(cfg: EncodeConfig) -> Path:
    """Where this cell's file lands."""
    return cfg.out_dir / "encodings" / cfg.suite / f"{cfg.dataset}__{cfg.representation}.npz"


def run_campaign(cfg: EncodeConfig) -> Path:
    """Encode one ``(suite, dataset, representation)`` cell and write its file.

    Args:
        cfg: The cell configuration.

    Returns:
        The path written.

    Raises:
        CampaignError: If the file would not conform.
    """
    cohort = load_cohort(cfg.suite, cfg.dataset, root=cfg.cohort_root, limit=cfg.limit)
    LOGGER.info(
        "%s/%s/%s: %d graphs, budget %.1f s, jobs %d",
        cfg.suite,
        cfg.dataset,
        cfg.representation,
        len(cohort),
        cfg.budget_s,
        cfg.jobs,
    )
    records = _drive_parallel(cfg, cohort, range(len(cohort)), "primary")
    records = _apply_d14(cfg, cohort, records)
    arrays = to_arrays(cfg, cohort, records)
    path = output_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    LOGGER.info(
        "wrote %s (ok=%d censored=%d error=%d)",
        path,
        int((arrays["status"] == "ok").sum()),
        int((arrays["status"] == "censored").sum()),
        int((arrays["status"] == "error").sum()),
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one T-06 encoding campaign cell.")
    parser.add_argument("--suite", required=True, choices=("suite1", "suite2"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cohort-root", type=Path, default=None)
    parser.add_argument("--budget-s", type=float, default=ENCODE_BUDGET_S)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--agm-search-nodes", type=int, default=None)
    parser.add_argument("--min-dfs-max-projections", type=int, default=None)
    parser.add_argument("--require-cpp", action="store_true", help="abort unless engine()=='cpp'")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.require_cpp:
        import isalgraph

        if isalgraph.engine() != "cpp":
            raise CampaignError(f"engine is {isalgraph.engine()!r}; refusing to run on Python")
    run_campaign(
        EncodeConfig(
            suite=args.suite,
            dataset=args.dataset,
            representation=args.representation,
            out_dir=args.out,
            cohort_root=args.cohort_root,
            budget_s=args.budget_s,
            jobs=args.jobs,
            limit=args.limit,
            agm_search_nodes=args.agm_search_nodes,
            min_dfs_max_projections=args.min_dfs_max_projections,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
