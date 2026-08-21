"""SynapticDB public package."""

from synapticdb.api import SynapticDB
from synapticdb.models import (
    EmbeddingError,
    InvalidArgumentError,
    Memory,
    NotFoundError,
    RecalledMemory,
    RecallResult,
    Stats,
    SynapticError,
)

__version__ = "0.1.1"

__all__: tuple[str, ...] = (
    "EmbeddingError",
    "InvalidArgumentError",
    "Memory",
    "NotFoundError",
    "RecallResult",
    "RecalledMemory",
    "Stats",
    "SynapticDB",
    "SynapticError",
)
