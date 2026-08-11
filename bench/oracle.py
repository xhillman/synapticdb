"""Load immutable rankings captured from the locked baseline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from .baseline import BaselineConfig
from .contracts import MAX_QUERY_COUNT
from .dataset import BenchmarkDataset

MAX_ORACLE_FILE_BYTES = 131_072
_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_fingerprint",
        "baseline_name",
        "baseline_config",
        "top_k",
        "environment",
        "rankings_sha256",
        "rankings",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "commit",
        "python",
        "platform",
        "faiss-cpu",
        "numpy",
        "rank-bm25",
        "scikit-learn",
        "sentence-transformers",
        "tokenizers",
        "torch",
        "transformers",
    }
)


class OracleError(ValueError):
    """Raised when frozen baseline evidence violates its contract."""


@dataclass(frozen=True, slots=True)
class OracleRanking:
    query_id: str
    ranked_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BaselineOracle:
    dataset_fingerprint: str
    baseline_name: str
    baseline_config: BaselineConfig
    top_k: int
    environment: tuple[tuple[str, str], ...]
    rankings_sha256: str
    rankings: tuple[OracleRanking, ...]

    def ranked_ids(self, query_id: str) -> tuple[str, ...]:
        """Return one query's frozen ranking."""
        for ranking in self.rankings:
            if ranking.query_id == query_id:
                return ranking.ranked_ids
        raise OracleError(f"baseline oracle has no ranking for query: {query_id}")


def load_baseline_oracle(path: str | Path, dataset: BenchmarkDataset) -> BaselineOracle:
    """Load a frozen baseline result bound to ``dataset``."""
    root = _exact_object(_read_json(Path(path)), _ROOT_FIELDS, "baseline oracle")
    if _integer(root["schema_version"], "schema_version") != _SCHEMA_VERSION:
        raise OracleError(f"baseline oracle schema_version must be {_SCHEMA_VERSION}")
    fingerprint = _sha256(root["dataset_fingerprint"], "dataset_fingerprint")
    if fingerprint != dataset.fingerprint:
        raise OracleError("baseline oracle dataset fingerprint does not match the loaded dataset")
    name = _string(root["baseline_name"], "baseline_name", maximum=64)
    if name != "baseline":
        raise OracleError("baseline oracle baseline_name must be baseline")
    config = _baseline_config(root["baseline_config"])
    top_k = _integer(root["top_k"], "top_k")
    if top_k != config.final_top_k:
        raise OracleError(f"baseline oracle top_k must be {config.final_top_k}")
    environment = _environment(root["environment"])
    rankings, canonical = _rankings(root["rankings"], dataset, top_k)
    expected_digest = _sha256(root["rankings_sha256"], "rankings_sha256")
    actual_digest = hashlib.sha256(_canonical_json(canonical)).hexdigest()
    if actual_digest != expected_digest:
        raise OracleError("baseline oracle rankings digest does not match its contents")
    return BaselineOracle(fingerprint, name, config, top_k, environment, expected_digest, rankings)


def _read_json(path: Path) -> object:
    if not path.is_file():
        raise OracleError(f"missing baseline oracle: {path}")
    try:
        size = path.stat().st_size
        if not 1 <= size <= MAX_ORACLE_FILE_BYTES:
            raise OracleError(f"baseline oracle must contain 1 to {MAX_ORACLE_FILE_BYTES} bytes: {path}")
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"invalid baseline oracle JSON: {path}") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OracleError(f"baseline oracle contains duplicate field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise OracleError(f"baseline oracle contains invalid number: {value}")


def _exact_object(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise OracleError(f"{label} must be an object")
    row = cast(dict[str, object], value)
    if set(row) != fields:
        raise OracleError(f"{label} fields do not match the schema")
    return row


def _baseline_config(value: object) -> BaselineConfig:
    row = _exact_object(value, frozenset(BaselineConfig().to_dict()), "baseline configuration")
    if _canonical_json(row) != _canonical_json(BaselineConfig().to_dict()):
        raise OracleError("baseline oracle configuration does not match the locked baseline")
    return BaselineConfig()


def _environment(value: object) -> tuple[tuple[str, str], ...]:
    row = _exact_object(value, _ENVIRONMENT_FIELDS, "baseline environment")
    parsed = {
        key: _string(item, f"environment {key}", maximum=256 if key == "platform" else 64) for key, item in row.items()
    }
    if _COMMIT.fullmatch(parsed["commit"]) is None:
        raise OracleError("baseline environment commit must be a clean 40-character SHA")
    return tuple(sorted(parsed.items()))


def _rankings(
    value: object,
    dataset: BenchmarkDataset,
    top_k: int,
) -> tuple[tuple[OracleRanking, ...], list[dict[str, object]]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_QUERY_COUNT:
        raise OracleError(f"baseline oracle rankings must contain 1 to {MAX_QUERY_COUNT} rows")
    if len(value) != len(dataset.queries):
        raise OracleError("baseline oracle query coverage does not match the loaded dataset")
    known_memory_ids = {memory.benchmark_id for memory in dataset.memories}
    parsed: list[OracleRanking] = []
    canonical: list[dict[str, object]] = []
    for index, (raw, query) in enumerate(zip(value, dataset.queries, strict=True), 1):
        row = _exact_object(raw, frozenset({"query_id", "ranked_ids"}), f"baseline ranking {index}")
        query_id = _string(row["query_id"], f"baseline ranking {index} query_id", maximum=64)
        if query_id != query.query_id:
            raise OracleError("baseline oracle query IDs do not match the loaded dataset order")
        ranked_ids = _ranked_ids(row["ranked_ids"], known_memory_ids, top_k, index)
        parsed.append(OracleRanking(query_id, ranked_ids))
        canonical.append({"query_id": query_id, "ranked_ids": list(ranked_ids)})
    return tuple(parsed), canonical


def _ranked_ids(value: object, known: set[str], top_k: int, index: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != top_k:
        raise OracleError(f"baseline ranking {index} must contain exactly {top_k} memory IDs")
    if any(not isinstance(item, str) or not item for item in value):
        raise OracleError(f"baseline ranking {index} memory IDs must be non-empty strings")
    ranked_ids = tuple(cast(list[str], value))
    if len(set(ranked_ids)) != len(ranked_ids):
        raise OracleError(f"baseline ranking {index} memory IDs must be unique")
    if not set(ranked_ids) <= known:
        raise OracleError(f"baseline ranking {index} contains an unknown memory ID")
    return ranked_ids


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise OracleError(f"baseline oracle {label} must be an integer")
    return value


def _string(value: object, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise OracleError(f"baseline oracle {label} must contain 1 to {maximum} characters")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label, maximum=64)
    if _SHA256.fullmatch(text) is None:
        raise OracleError(f"baseline oracle {label} must be a lowercase SHA-256 digest")
    return text
