"""The T-13 LaTeX tables.

Three tables, and the caption of each says which rows it used.

``tab_t13_ladder_spearman``
    Spearman's rho between ``log10|Aut(G)|`` and ``seconds`` on each ladder,
    per representation, **over completed rows only**, with the completion rate
    printed beside it so the reader can see how much of the ladder the
    correlation actually covers.  A footer block sign-tests the sign of rho
    across ladders per representation, which is the across-ladder statement the
    response letter makes.

``tab_t13_scaling_exponent``
    The fitted exponent of ``T ~ n^alpha`` with a percentile-bootstrap CI.
    ``T-13-design.md`` 2.2 leg (iii) is explicit that this is **a property of
    the cohort, not a complexity result**: it measures how ``|Aut|`` happens to
    co-vary with ``n`` in the graphs that were measured, and reporting it as
    complexity is precisely the conflation R3.7d objects to.  The caption says
    so, in the table, where a reader who quotes the number will see it.

``tab_t13_completion``
    Completion, censoring, ``unsupported`` and ``error`` counts per
    representation per ladder, with the censoring mechanism named.  This is the
    denominator of the other two tables, and neither is readable without it.

Every cell here is derived through :mod:`data`, whose summaries are either
censoring-aware or named for the subset they use.  No cell pools a censored row
with a completed one.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from benchmarks.real_data.eval_t13_figures import data, design

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

LOGGER: Final = logging.getLogger(__name__)

#: Significance level quoted in the sign-test footer.
SIGN_TEST_ALPHA: Final[float] = 0.05

#: Bootstrap resamples for the exponent CI, and the seed that makes it
#: reproducible.  Frozen here so that a re-run of the tables reproduces the
#: interval exactly rather than a nearby one.
N_BOOT: Final[int] = 2000
BOOT_SEED: Final[int] = 13

#: Output file names.
FILES: Final[dict[str, str]] = {
    "spearman": "tab_t13_ladder_spearman.tex",
    "exponent": "tab_t13_scaling_exponent.tex",
    "completion": "tab_t13_completion.tex",
}

_DASH: Final[str] = "--"


def _tex_escape(text: str) -> str:
    """Escape the LaTeX specials that appear in ladder and family names."""
    out = text.replace("\\", r"\textbackslash{}")
    for char in ("_", "%", "&", "#", "$"):
        out = out.replace(char, "\\" + char)
    return out


def _ladder_name(ladder: data.Ladder) -> str:
    """Return a LaTeX-safe short name for one ladder."""
    return _tex_escape(f"{ladder.family}/{ladder.base}/n={ladder.n}")


def _fmt_rho(value: float | None) -> str:
    """Render Spearman's rho, or a dash when it is undefined."""
    return _DASH if value is None else f"{value:+.3f}"


def _fmt_pct(value: float | None) -> str:
    """Render a rate as a percentage, or a dash when undefined."""
    return _DASH if value is None else f"{100.0 * value:.0f}\\%"


def _fmt_p(value: float) -> str:
    """Render a p-value as a complete math expression."""
    if value < 1e-4:
        return r"$p < 10^{-4}$"
    return f"$p = {value:.4f}$"


def _fmt_seconds(value: float | None) -> str:
    """Render a duration in seconds, or a dash."""
    if value is None:
        return _DASH
    if value < 1e-3:
        return f"{value * 1e3:.2f}\\,ms"
    if value < 1.0:
        return f"{value * 1e3:.0f}\\,ms"
    return f"{value:.2f}\\,s"


def _selected(records: data.Records, focus_only: bool) -> tuple[design.Representation, ...]:
    """Return the representations to tabulate, in draw order.

    Args:
        records: The loaded campaign.
        focus_only: Restrict to the arms the 6.3 characterisation names.

    Returns:
        Registered representations present in the data.

    Raises:
        design.UnknownRepresentationError: If the data carries an unregistered
            backend.
    """
    present = design.present(records.representations)
    return tuple(r for r in present if r.is_focus) if focus_only else present


def ladder_spearman_rows(
    records: data.Records, *, arm: str = data.DEFAULT_ARM, focus_only: bool = False
) -> tuple[list[dict[str, Any]], dict[str, data.SignTest]]:
    """Compute the per-ladder correlations and the across-ladder sign tests.

    Args:
        records: The loaded campaign.
        arm: Engine arm to read.
        focus_only: Restrict to the arms the characterisation names.

    Returns:
        ``(rows, sign_tests)``.  Each row carries one ``(ladder,
        representation)`` cell; ``sign_tests`` maps a representation key to the
        sign test of its rho values across ladders.
    """
    reps = _selected(records, focus_only)
    rows: list[dict[str, Any]] = []
    rhos: dict[str, list[float]] = {r.key: [] for r in reps}

    for ladder in data.ladders(records, arm=arm):
        for rep in reps:
            pairs = ladder.series(rep.key)
            completed = [
                (graph.log10_aut, float(row["seconds"]))
                for graph, row in pairs
                if data.is_completed(row) and graph.log10_aut is not None
            ]
            rho = (
                data.spearman([p[0] for p in completed], [p[1] for p in completed])
                if len(completed) >= 3
                else None
            )
            if rho is not None:
                rhos[rep.key].append(rho)
            summary = data.summarise_times([row for _, row in pairs])
            rows.append(
                {
                    "ladder": ladder,
                    "representation": rep,
                    "n_rungs": len(ladder.rungs),
                    "rho": rho,
                    "summary": summary,
                }
            )
    return rows, {key: data.sign_test(values) for key, values in rhos.items()}


def ladder_spearman_table(
    records: data.Records, *, arm: str = data.DEFAULT_ARM, focus_only: bool = False
) -> str:
    """Emit the per-ladder Spearman table with its sign-test footer.

    Args:
        records: The loaded campaign.
        arm: Engine arm to read.
        focus_only: Restrict to the arms the characterisation names.

    Returns:
        A complete ``table*`` environment.
    """
    rows, tests = ladder_spearman_rows(records, arm=arm, focus_only=focus_only)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cost against symmetry on each ladder. Every ladder holds $n$, $m$ and the "
        r"whole degree sequence fixed, so $\rho$ is a correlation with $\log_{10}|\mathrm{Aut}"
        r"(G)|$ and nothing else. $\rho$ is computed over \emph{completed} encodings only, "
        r"because a censored row carries no completion time; the completion column is printed "
        r"beside it so that a correlation over a fifth of a ladder cannot be read as a "
        r"correlation over the ladder. The footer sign-tests the sign of $\rho$ across "
        rf"ladders at $\alpha = {SIGN_TEST_ALPHA}$. "
        rf"Arm: \texttt{{{_tex_escape(arm)}}}.}}",
        r"\label{tab:t13-ladder-spearman}",
        r"\footnotesize",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"ladder & representation & rungs & completed & censored & completion & $\rho$ \\",
        r"\midrule",
    ]
    previous: str | None = None
    for row in rows:
        ladder = row["ladder"]
        rep = row["representation"]
        summary = row["summary"]
        name = _ladder_name(ladder)
        shown = "" if name == previous else name
        previous = name
        tex = rf"\textbf{{{rep.tex}}}" if rep.is_ours else rep.tex
        lines.append(
            " & ".join(
                [
                    shown,
                    tex,
                    str(row["n_rungs"]),
                    str(summary.n_completed),
                    str(summary.n_censored),
                    _fmt_pct(summary.completion_rate),
                    _fmt_rho(row["rho"]),
                ]
            )
            + r" \\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{sign test on the sign of $\rho$ across ladders}} \\",
    ]
    for rep in _selected(records, focus_only):
        test = tests[rep.key]
        tex = rf"\textbf{{{rep.tex}}}" if rep.is_ours else rep.tex
        lines.append(
            " & ".join(
                [
                    "",
                    tex,
                    f"$n_+ = {test.n_positive}$",
                    f"$n_- = {test.n_negative}$",
                    f"ties {test.n_ties}",
                    "",
                    _fmt_p(test.p_value),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def scaling_exponent_table(
    records: data.Records, *, arm: str = data.DEFAULT_ARM, focus_only: bool = False
) -> str:
    """Emit the fitted ``T ~ n^alpha`` table with bootstrap CIs.

    Args:
        records: The loaded campaign.
        arm: Engine arm to read.
        focus_only: Restrict to the arms the characterisation names.

    Returns:
        A complete ``table`` environment.
    """
    reps = _selected(records, focus_only)
    rows = records.with_arm(arm)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical scaling of runtime with order, fitted as $\log T = \alpha \log n "
        r"+ c$ by ordinary least squares over \emph{completed} encodings, with a "
        rf"percentile bootstrap ({N_BOOT} resamples, seed {BOOT_SEED}) on $\alpha$. "
        r"\textbf{This is a property of the measured cohort, not a complexity result.} It "
        r"records how $|\mathrm{Aut}(G)|$ happens to co-vary with $n$ among the graphs that "
        r"finished, and it moves with the cohort; the complexity statement is the bound of "
        r"Section~2.1 and the $|\mathrm{Aut}|$ characterisation, not this exponent. The "
        r"completion column bounds how much of the cohort the fit saw: censored graphs "
        r"contribute no completion time and are absent from the fit, which biases $\alpha$ "
        rf"toward the tractable graphs. Arm: \texttt{{{_tex_escape(arm)}}}.}}",
        r"\label{tab:t13-scaling-exponent}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"representation & $\alpha$ & 95\% CI & fitted points & completion \\",
        r"\midrule",
    ]
    for rep in reps:
        subset = [r for r in rows if r["representation"] == rep.key]
        fit = data.fit_power_law_completions_only(subset, n_boot=N_BOOT, seed=BOOT_SEED)
        summary = data.summarise_times(subset)
        tex = rf"\textbf{{{rep.tex}}}" if rep.is_ours else rep.tex
        if fit is None:
            lines.append(
                " & ".join(
                    [tex, _DASH, _DASH, str(summary.n_completed), _fmt_pct(summary.completion_rate)]
                )
                + r" \\"
            )
            continue
        lines.append(
            " & ".join(
                [
                    tex,
                    f"{fit.alpha:.2f}",
                    f"[{fit.ci_low:.2f}, {fit.ci_high:.2f}]",
                    str(fit.n_points),
                    _fmt_pct(summary.completion_rate),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def completion_table(
    records: data.Records, *, arm: str = data.DEFAULT_ARM, focus_only: bool = False
) -> str:
    """Emit the completion-rate table, per representation per ladder.

    Args:
        records: The loaded campaign.
        arm: Engine arm to read.
        focus_only: Restrict to the arms the characterisation names.

    Returns:
        A complete ``table*`` environment.
    """
    reps = _selected(records, focus_only)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{What finished, and what stopped it. \emph{completed} timed an encoding; "
        r"\emph{censored} means a declared budget stopped it and the true time is greater "
        r"than the value recorded, so those rows enter no mean and no fit; "
        r"\emph{unsupported} is the backend declining the graph, a property of the "
        r"representation; \emph{error} is a fault. The mechanism column names which budget "
        r"fired, because a wall-clock kill at the full budget and an internal cap that fires "
        r"in milliseconds are different observations and must not be pooled. The "
        r"Kaplan--Meier median uses the censored rows as the right-censored observations "
        r"they are; \emph{not reached} means more than half the units were still running "
        r"when observation stopped, and the completions-only median beside it is then a "
        rf"statement about the tractable subset alone. Arm: \texttt{{{_tex_escape(arm)}}}.}}",
        r"\label{tab:t13-completion}",
        r"\footnotesize",
        r"\begin{tabular}{@{}llrrrrlrr@{}}",
        r"\toprule",
        r"ladder & representation & compl. & cens. & unsup. & err. & mechanism & "
        r"KM median & compl.-only \\",
        r"\midrule",
    ]
    previous: str | None = None
    for ladder in data.ladders(records, arm=arm):
        for rep in reps:
            rows = [row for _, row in ladder.series(rep.key)]
            summary = data.summarise_times(rows)
            name = _ladder_name(ladder)
            shown = "" if name == previous else name
            previous = name
            mechanism = (
                ", ".join(
                    f"{design.CENSORING_DISPLAY.get(kind, kind)} $\\times$ {count}"
                    for kind, count in summary.censoring_kinds
                )
                or _DASH
            )
            km = (
                _fmt_seconds(summary.km_median)
                if summary.km_median_reached
                else rf"not reached ($>{_fmt_seconds(summary.max_observed)}$)"
                if summary.max_observed is not None
                else _DASH
            )
            tex = rf"\textbf{{{rep.tex}}}" if rep.is_ours else rep.tex
            lines.append(
                " & ".join(
                    [
                        shown,
                        tex,
                        str(summary.n_completed),
                        str(summary.n_censored),
                        str(summary.n_unsupported),
                        str(summary.n_error),
                        mechanism,
                        km,
                        _fmt_seconds(summary.completions_only_median),
                    ]
                )
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def all_tables(
    records: data.Records, *, arm: str = data.DEFAULT_ARM, focus_only: bool = False
) -> dict[str, str]:
    """Return every T-13 table, keyed by output file name.

    Args:
        records: The loaded campaign.
        arm: Engine arm to read.
        focus_only: Restrict to the arms the characterisation names.

    Returns:
        ``{filename: latex}``.
    """
    return {
        FILES["spearman"]: ladder_spearman_table(records, arm=arm, focus_only=focus_only),
        FILES["exponent"]: scaling_exponent_table(records, arm=arm, focus_only=focus_only),
        FILES["completion"]: completion_table(records, arm=arm, focus_only=focus_only),
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--records",
        nargs="+",
        required=True,
        help="glob(s) or paths of records_*.jsonl shards",
    )
    ap.add_argument(
        "--counters",
        nargs="+",
        default=None,
        help="accepted for CLI uniformity across this package; not read by this module",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="LaTeX output directory")
    ap.add_argument("--arm", default=data.DEFAULT_ARM, help="engine arm to tabulate")
    ap.add_argument(
        "--focus-only",
        action="store_true",
        help="tabulate only the arms the 6.3 characterisation names",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.counters:
        LOGGER.info("--counters is not read by tables; ignoring %s", args.counters)
    records = data.load(args.records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in all_tables(records, arm=args.arm, focus_only=args.focus_only).items():
        (args.out_dir / name).write_text(text + "\n")
        LOGGER.info("%s -> %s", name, args.out_dir / name)
    return 0


__all__ = [
    "BOOT_SEED",
    "FILES",
    "N_BOOT",
    "SIGN_TEST_ALPHA",
    "all_tables",
    "build_parser",
    "completion_table",
    "ladder_spearman_rows",
    "ladder_spearman_table",
    "main",
    "scaling_exponent_table",
]


if __name__ == "__main__":
    raise SystemExit(main())
