"""In-repository associative-recall benchmark harness."""

from .dataset import BenchmarkDataset, load_dataset
from .protocol import BenchmarkReport, run_benchmark

__all__ = ["BenchmarkDataset", "BenchmarkReport", "load_dataset", "run_benchmark"]
