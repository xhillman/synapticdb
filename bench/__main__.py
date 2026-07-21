"""Command-line entry point for the SynapticDB benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from .baseline import LockedBaseline
from .dataset import load_dataset
from .protocol import run_benchmark
from .reporting import render_markdown, write_report
from .retrievers import FixtureRetriever, Retriever

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--retriever", choices=("fixture", "baseline"), default="baseline")
    parser.add_argument("--seeds", default="1337", help="Comma-separated deterministic seeds")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--check", action="store_true", help="Return a nonzero status when the candidate gate fails")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.profile == "full" and args.top_k != 10:
        raise ValueError("the locked full-profile reproduction requires --top-k 10")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    expected = (50, 5, 5) if args.profile == "smoke" else (500, 25, 25)
    dataset = load_dataset(ROOT / "data" / args.profile, expected_counts=expected)
    factories: dict[str, Callable[[], Retriever]] = {
        "fixture": FixtureRetriever,
        "baseline": LockedBaseline,
    }
    selected = factories[args.retriever]
    reference = FixtureRetriever if args.profile == "smoke" else LockedBaseline
    report = run_benchmark(
        dataset,
        baseline_factory=reference,
        candidate_factory=selected,
        seeds=seeds,
        top_k=args.top_k,
        required_unique_wins=10 if args.profile == "full" and args.retriever != "baseline" else 0,
        expected_baseline_hits=(25, 10) if args.profile == "full" else None,
    )
    print(render_markdown(report))
    if not args.no_write:
        json_path, markdown_path = write_report(report, args.output_dir, run_id=args.run_id)
        print(f"JSON {json_path.resolve()}")
        print(f"MARKDOWN {markdown_path.resolve()}")
    return int(args.check and not report.passed)


if __name__ == "__main__":
    raise SystemExit(main())
