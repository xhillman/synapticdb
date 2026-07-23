"""Command-line entry point for the SynapticDB benchmark."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from .baseline import LockedBaseline
from .contracts import MAX_SEED_COUNT, MAX_SEED_TEXT_CHARS, MAX_TOP_K
from .dataset import load_dataset
from .protocol import run_benchmark
from .reporting import render_markdown, write_report
from .retrievers import FixtureRetriever, Retriever, SynapticRetriever, _fixture_embedding

ROOT = Path(__file__).resolve().parent


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
    return parser.parse_args()


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
    factories: dict[str, Callable[[], Retriever]] = {
        "fixture": FixtureRetriever,
        "baseline": LockedBaseline,
        "synaptic": (
            (lambda: SynapticRetriever(_fixture_embedding, embedding_name="fixture"))
            if args.profile == "smoke"
            else SynapticRetriever
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
    )
    print(render_markdown(report))
    if not args.no_write:
        json_path, markdown_path = write_report(report, args.output_dir, run_id=args.run_id)
        print(f"JSON {json_path.resolve()}")
        print(f"MARKDOWN {markdown_path.resolve()}")
    return int(args.check and not report.passed)


if __name__ == "__main__":
    raise SystemExit(main())
