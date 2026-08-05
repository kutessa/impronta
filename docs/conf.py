"""Sphinx configuration for the impronta documentation.

Build locally with:

    uv run --group docs sphinx-build -b html docs docs/_build/html

Heavy runtime deps (torch, speechbrain, faiss) are mocked so autodoc never
loads models and the macOS faiss+torch OpenMP conflict cannot bite. If you
remove the mocks, run the build with KMP_DUPLICATE_LIB_OK=TRUE.
"""

project = "impronta"
author = "Tarik Hastor"
copyright = "2026, Tarik Hastor"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

html_theme = "furo"
html_title = "impronta"

exclude_patterns = ["_build"]

myst_enable_extensions = ["colon_fence"]

autodoc_mock_imports = ["torch", "speechbrain", "faiss"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
