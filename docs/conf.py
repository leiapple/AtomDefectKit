from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

project = "AtomDefectKit"
author = "Lei Zhang"
copyright = "2026, Lei Zhang"

try:
    release = version("atomdefectkit")
except PackageNotFoundError:
    release = "0.1.0"

version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"
