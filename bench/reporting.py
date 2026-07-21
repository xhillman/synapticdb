"""Stable machine-readable and human-readable benchmark reports."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

from .protocol import BenchmarkReport


def render_markdown(report: BenchmarkReport) -> str:
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
    if report.expected_baseline_hits is not None:
        direct, associative = report.expected_baseline_hits
        lines.insert(
            -1,
            f"Baseline reproduction target: `{direct}/{report.direct_total}` direct and "
            f"`{associative}/{report.associative_total}` associative hits.",
        )
    return "\n".join(lines)


def write_report(report: BenchmarkReport, output_dir: str | Path, *, run_id: str | None = None) -> tuple[Path, Path]:
    resolved = run_id or datetime.now(UTC).strftime("bench-%Y%m%dT%H%M%SZ")
    root = Path(output_dir) / resolved
    root.mkdir(parents=True, exist_ok=False)
    json_path = root / "report.json"
    markdown_path = root / "report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
