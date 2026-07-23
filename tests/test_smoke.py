import synapticdb


def test_import() -> None:
    assert synapticdb.__version__ == "0.1.0"
    assert synapticdb.__all__ == (
        "EmbeddingError",
        "InvalidArgumentError",
        "Memory",
        "NotFoundError",
        "RecallResult",
        "Recalled",
        "Stats",
        "Synaptic",
        "SynapticError",
    )
