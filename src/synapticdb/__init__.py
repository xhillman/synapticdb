"""SynapticDB public package."""

from synapticdb.api import SynapticDB
from synapticdb.models import (
    EmbeddingError,
    InvalidArgumentError,
    Memory,
    NotFoundError,
    Recalled,
    RecallResult,
    Stats,
    SynapticError,
)

__version__ = "0.1.0"

__all__: tuple[str, ...] = (
    "EmbeddingError",
    "InvalidArgumentError",
    "Memory",
    "NotFoundError",
    "RecallResult",
    "Recalled",
    "Stats",
    "SynapticDB",
    "SynapticError",
)
