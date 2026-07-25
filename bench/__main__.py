"""Command-line entry point for the SynapticDB benchmark."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from synapticdb.learning import ParameterValue, default_parameters

from .baseline import LockedBaseline
from .contracts import MAX_SEED_COUNT, MAX_SEED_TEXT_CHARS, MAX_TOP_K
from .dataset import load_dataset
from .protocol import KNOWN_MEASURES, Timeline, run_benchmark
from .reporting import render_markdown, write_report
from .retrievers import FixtureRetriever, Retriever, SynapticRetriever, _fixture_embedding

ROOT = Path(__file__).resolve().parent
MAX_PARAM_OVERRIDES = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--retriever", choices=("fixture", "baseline", "synaptic"), default="baseline")
    parser.add_argument("--seeds", default="1337", help="Comma-separated deterministic seeds")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--check", action="store_true", help="Return a nonzero status when the candidate gate fails")
    # Semantic seeding is off by default (disabled on benchmark evidence);
    # pass --semantic-threshold to re-enable it for a calibration run.
    parser.add_argument("--semantic-threshold", type=float, default=None)
    parser.add_argument("--semantic-max-links", type=int, default=3)
    parser.add_argument("--semantic-weight", type=float, default=0.25)
    parser.add_argument("--temporal-window", type=int, default=600)
    parser.add_argument("--temporal-max-links", type=int, default=3)
    parser.add_argument("--temporal-weight", type=float, default=0.2)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Override one PRD section 9 parameter, e.g. --param activation_blend_weight=0.9",
    )
    # Simulated time. All default to zero, which keeps the wall-clock behaviour
    # every record before the harness clock was produced with.
    parser.add_argument("--warmup-span-days", type=float, default=0.0)
    parser.add_argument("--query-offset-days", type=float, default=0.0)
    parser.add_argument("--decay-probe-days", type=float, default=0.0)
    parser.add_argument(
        "--measure",
        action="append",
        default=[],
        choices=sorted(KNOWN_MEASURES),
        help="Run a directional gate; repeatable",
    )
    return parser.parse_args()


def _parse_overrides(entries: Sequence[str]) -> dict[str, ParameterValue]:
    """Parse repeated KEY=JSON overrides against the known parameter budget."""
    if len(entries) > MAX_PARAM_OVERRIDES:
        raise ValueError(f"--param accepts at most {MAX_PARAM_OVERRIDES} overrides")
    known = default_parameters()
    overrides: dict[str, ParameterValue] = {}
    for entry in entries:
        key, separator, raw = entry.partition("=")
        if not separator or not key:
            raise ValueError(f"--param expects KEY=JSON, got: {entry}")
        if key not in known:
            raise ValueError(f"unknown parameter: {key}")
        if key in overrides:
            raise ValueError(f"parameter repeated: {key}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"--param {key} needs a JSON value, got: {raw}") from error
        # JSON has no tuples, so a group like temporal_link arrives as a list.
        overrides[key] = tuple(value) if isinstance(value, list) else value
    return overrides


def _parse_seeds(value: str) -> tuple[int, ...]:
    if len(value) > MAX_SEED_TEXT_CHARS:
        raise ValueError(f"--seeds accepts at most {MAX_SEED_TEXT_CHARS} characters")
    if value.count(",") >= MAX_SEED_COUNT:
        raise ValueError(f"--seeds accepts at most {MAX_SEED_COUNT} values")
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("--seeds requires at least one integer")
    return seeds


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not 1 <= args.top_k <= MAX_TOP_K:
        raise ValueError(f"--top-k must be between 1 and {MAX_TOP_K}")
    if args.profile == "full" and args.top_k != 10:
        raise ValueError("the locked full-profile reproduction requires --top-k 10")
    seeds = _parse_seeds(args.seeds)
    expected = (50, 5, 5) if args.profile == "smoke" else (500, 25, 25)
    dataset = load_dataset(ROOT / "data" / args.profile, expected_counts=expected)
    semantic_seed = (
        None
        if args.semantic_threshold is None
        else (args.semantic_threshold, args.semantic_max_links, args.semantic_weight)
    )
    temporal_link = (args.temporal_window, args.temporal_max_links, args.temporal_weight)
    overrides = _parse_overrides(args.param)
    factories: dict[str, Callable[[], Retriever]] = {
        "fixture": FixtureRetriever,
        "baseline": LockedBaseline,
        "synaptic": (
            (
                lambda: SynapticRetriever(
                    _fixture_embedding,
                    embedding_name="fixture",
                    semantic_seed=semantic_seed,
                    temporal_link=temporal_link,
                    overrides=overrides,
                )
            )
            if args.profile == "smoke"
            else (
                lambda: SynapticRetriever(
                    semantic_seed=semantic_seed,
                    temporal_link=temporal_link,
                    overrides=overrides,
                )
            )
        ),
    }
    selected = factories[args.retriever]
    reference = FixtureRetriever if args.profile == "smoke" else LockedBaseline
    candidate_factory = None if selected is reference else selected
    report = run_benchmark(
        dataset,
        baseline_factory=reference,
        candidate_factory=candidate_factory,
        seeds=seeds,
        top_k=args.top_k,
        required_unique_wins=10 if args.profile == "full" and args.retriever != "baseline" else 0,
        expected_baseline_hits=(25, 10) if args.profile == "full" else None,
        timeline=Timeline(args.warmup_span_days, args.query_offset_days, args.decay_probe_days),
        measures=frozenset(args.measure),
    )
    print(render_markdown(report))
    if not args.no_write:
        json_path, markdown_path = write_report(report, args.output_dir, run_id=args.run_id)
        print(f"JSON {json_path.resolve()}")
        print(f"MARKDOWN {markdown_path.resolve()}")
    return int(args.check and not report.passed)


if __name__ == "__main__":
    raise SystemExit(main())
