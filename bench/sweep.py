"""Calibration sweep over one PRD section 9 parameter.

A developer tool, not a merge gate: it runs the full benchmark once per value
and prints a comparison table. Excluded from the wheel like the rest of bench/.

    python -m bench.sweep activation_blend_weight 0.45 0.7 0.9

Sweeping many parameters against 25 associative queries overfits. Prefer one
parameter at a time, prefer values with a mechanical reason to help, and record
the configuration of anything promoted.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synapticdb.learning import default_parameters

from .baseline import LockedBaseline
from .dataset import load_dataset
from .protocol import run_benchmark
from .retrievers import FixtureRetriever, SynapticRetriever, _fixture_embedding

ROOT = Path(__file__).resolve().parent
MAX_SWEEP_VALUES = 12


@dataclass(frozen=True, slots=True)
class SweepRow:
    value: object
    direct_hits: int
    associative_hits: int
    unique_wins: int
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter", help="One PRD section 9 parameter name")
    parser.add_argument("values", nargs="+", help="JSON values to try, e.g. 0.45 0.7 [600,3,0.4]")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def parse_values(raw_values: Sequence[str]) -> tuple[object, ...]:
    """Parse JSON sweep values, bounded so one command cannot run all day."""
    if not 1 <= len(raw_values) <= MAX_SWEEP_VALUES:
        raise ValueError(f"a sweep runs 1 to {MAX_SWEEP_VALUES} values")
    values: list[object] = []
    for raw in raw_values:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"sweep values must be JSON, got: {raw}") from error
        values.append(tuple(parsed) if isinstance(parsed, list) else parsed)
    return tuple(values)


def run_sweep(parameter: str, values: Sequence[object], profile: str, seed: int) -> tuple[SweepRow, ...]:
    """Run the benchmark once per value, holding everything else at default."""
    if parameter not in default_parameters():
        raise ValueError(f"unknown parameter: {parameter}")
    expected = (50, 5, 5) if profile == "smoke" else (500, 25, 25)
    dataset = load_dataset(ROOT / "data" / profile, expected_counts=expected)
    reference = FixtureRetriever if profile == "smoke" else LockedBaseline
    rows: list[SweepRow] = []
    for value in values:
        report = run_benchmark(
            dataset,
            baseline_factory=reference,
            candidate_factory=lambda value=value: _candidate(profile, parameter, value),  # type: ignore[misc]
            seeds=(seed,),
            required_unique_wins=10 if profile == "full" else 0,
        )
        run = report.runs[0]
        rows.append(
            SweepRow(
                value=value,
                direct_hits=run.candidate_direct_hits,
                associative_hits=run.candidate_associative_hits,
                unique_wins=run.associative_unique_wins,
                passed=run.passed,
            )
        )
    return tuple(rows)


def _candidate(profile: str, parameter: str, value: object) -> SynapticRetriever:
    overrides = {parameter: value}
    if profile == "smoke":
        return SynapticRetriever(_fixture_embedding, embedding_name="fixture", overrides=overrides)
    return SynapticRetriever(overrides=overrides)


def render(parameter: str, rows: Sequence[SweepRow], direct_total: int, associative_total: int) -> str:
    lines = [
        f"| {parameter} | direct | associative | unique wins | gate |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {json.dumps(list(row.value) if isinstance(row.value, tuple) else row.value)} "
            f"| {row.direct_hits}/{direct_total} "
            f"| {row.associative_hits}/{associative_total} "
            f"| {row.unique_wins}/{associative_total} "
            f"| {'PASS' if row.passed else 'FAIL'} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    values = parse_values(args.values)
    totals = (5, 5) if args.profile == "smoke" else (25, 25)
    rows = run_sweep(args.parameter, values, args.profile, args.seed)
    print(render(args.parameter, rows, *totals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
