"""Examples must at least compile — so they can't rot silently."""

import py_compile
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


def test_examples_exist():
    assert len(EXAMPLES) >= 5


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "out.pyc"), doraise=True)


def test_offline_multitenant_example_runs(capsys):
    """04 is fully offline — actually execute it."""
    import runpy

    runpy.run_path(str(EXAMPLES[3]), run_name="__main__")
    out = capsys.readouterr().out
    assert "other tenant sees: []" in out
    assert "wiped user:tarik" in out
