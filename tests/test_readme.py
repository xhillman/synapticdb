import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
_PYTHON_BLOCK = re.compile(r"```python\n(?P<code>.*?)```", re.DOTALL)


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        assert model_name == "all-MiniLM-L6-v2"

    def encode(self, text: str, *, convert_to_numpy: bool) -> tuple[float, float]:
        assert text
        assert convert_to_numpy
        return (1.0, 0.0)


class FakeSentenceTransformersModule(ModuleType):
    SentenceTransformer = FakeSentenceTransformer


def test_readme_quickstart_runs_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = _PYTHON_BLOCK.search(readme)
    if match is None:
        raise AssertionError("README requires a Python quickstart block")
    fake_module = FakeSentenceTransformersModule("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    exec(compile(match.group("code"), "README.md", "exec"), {})
    output = capsys.readouterr().out
    assert output.strip() == "Client X requires SOC2 for vendor deployments"


def test_readme_confidence_example_runs_verbatim() -> None:
    """The filtering example is what an agent would copy, so it must work.

    Only the first block was ever executed, which left later snippets free to
    drift out of the API. This runs the confidence example against a real
    instance rather than trusting it by inspection.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = [match.group("code") for match in _PYTHON_BLOCK.finditer(readme)]
    snippet = next((block for block in blocks if "confidence" in block), None)
    assert snippet is not None, "README should show how to filter on confidence"

    from synapticdb import SynapticDB

    def embedding(text: str) -> tuple[float, float]:
        lowered = text.lower()
        return (float(lowered.count("client")), float(lowered.count("invoice")))

    with SynapticDB(":memory:", embedding_fn=embedding) as memories:
        memories.remember("Client X requires SOC2 for vendor deployments")
        namespace: dict[str, Any] = {"memories": memories}
        exec(compile(snippet, "README.md", "exec"), namespace)
        relevant = namespace["relevant"]
        assert all(item.confidence >= 0.6 for item in relevant)
