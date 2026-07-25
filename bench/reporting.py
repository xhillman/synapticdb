"""Stable machine-readable and human-readable benchmark reports."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .contracts import MAX_RUN_ID_CHARS, MAX_SEED_COUNT
from .protocol import BenchmarkReport

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


_MEASUREMENT_CLAIMS = {
    "trajectory": "using the system improves it: warm associative hits >= cold",
    "diversity": "learning does not collapse the result set: repeat distinct >= first",
    "decay_direct_recall": "an aged graph still serves search: aged direct >= warm",
}


def _render_measurements(report: BenchmarkReport) -> list[str]:
    """Render the directional gates, each beside the claim it tests."""
    if not any(run.measurements for run in report.runs):
        return []
    lines = [
        "| Seed | Measurement | Before | After | Series | Gate | Claim |",
        "|---:|---|---:|---:|---|---|---|",
    ]
    for run in report.runs:
        for measurement in run.measurements:
            claim = _MEASUREMENT_CLAIMS.get(measurement.name, "")
            # The series separates a value that settles once from one that
            # keeps falling; the gate alone cannot tell them apart.
            series = " → ".join(str(value) for value in measurement.series) if measurement.series else "—"
            lines.append(
                f"| {run.seed} | {measurement.name} | {measurement.before} | {measurement.after} | "
                f"{series} | {'PASS' if measurement.passed else 'FAIL'} | {claim} |"
            )
    lines.append("")
    return lines


def render_markdown(report: BenchmarkReport) -> str:
    _validate_report(report)
    decision = "PASS" if report.passed else "FAIL"
    lines = [
        "# SynapticDB Benchmark",
        "",
        f"- Decision: **{decision}**",
        f"- Dataset: `{report.dataset_fingerprint}`",
        f"- Baseline: `{report.baseline_name}`",
        f"- Candidate: `{report.candidate_name}`",
        f"- Top-k: `{report.top_k}`",
        f"- Runtime: Python `{report.environment['python']}`",
        "",
        "| Seed | Baseline Direct | Candidate Direct | Baseline Assoc | Candidate Assoc | Unique Wins | Reproduced | Direct Parity | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.runs:
        lines.append(
            f"| {run.seed} | {run.baseline_direct_hits}/{report.direct_total} | "
            f"{run.candidate_direct_hits}/{report.direct_total} | "
            f"{run.baseline_associative_hits}/{report.associative_total} | "
            f"{run.candidate_associative_hits}/{report.associative_total} | "
            f"{run.associative_unique_wins}/{report.associative_total} | "
            f"{'Y' if run.baseline_reproduced else 'N'} | "
            f"{'Y' if run.direct_parity else 'N'} | {'PASS' if run.passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Gate: direct recall within `{report.direct_tolerance:.0%}` of baseline and "
            f"at least `{report.required_unique_wins}` path-backed associative unique wins on every seed.",
            "",
        ]
    )
    lines.extend(_render_measurements(report))
    if report.expected_baseline_hits is not None:
        direct, associative = report.expected_baseline_hits
        lines.insert(
            -1,
            f"Baseline reproduction target: `{direct}/{report.direct_total}` direct and "
            f"`{associative}/{report.associative_total}` associative hits.",
        )
    return "\n".join(lines)


def write_report(report: BenchmarkReport, output_dir: str | Path, *, run_id: str | None = None) -> tuple[Path, Path]:
    _validate_report(report)
    resolved = run_id if run_id is not None else datetime.now(timezone.utc).strftime("bench-%Y%m%dT%H%M%SZ")
    if len(resolved) > MAX_RUN_ID_CHARS or _RUN_ID.fullmatch(resolved) is None:
        raise ValueError("run_id must be a short filename-safe identifier")
    root = Path(output_dir) / resolved
    root.mkdir(parents=True, exist_ok=False)
    json_path = root / "report.json"
    markdown_path = root / "report.md"
    _write_text_exact(json_path, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    _write_text_exact(markdown_path, render_markdown(report))
    return json_path, markdown_path


def _validate_report(report: BenchmarkReport) -> None:
    if not 1 <= len(report.runs) <= MAX_SEED_COUNT:
        raise ValueError(f"benchmark report requires 1 to {MAX_SEED_COUNT} runs")
    if report.direct_total <= 0 or report.associative_total <= 0:
        raise ValueError("benchmark report totals must be positive")


def _write_text_exact(path: Path, payload: str) -> None:
    written = path.write_text(payload, encoding="utf-8")
    if written != len(payload):
        raise OSError(f"incomplete report write: {path}")
