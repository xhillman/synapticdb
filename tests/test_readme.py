import re
import sys
from pathlib import Path
from types import ModuleType

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
