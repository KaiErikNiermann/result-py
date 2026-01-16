from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "result_py"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google/Numpy docstrings
    "sphinx.ext.viewcode",
    "myst_parser",                # remove if not using Markdown
    "sphinx_autodoc_typehints",   # better typing rendering
]

project = 'result-py'
copyright = '2025, Kai Erik Niermann'
author = 'Kai Erik Niermann'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# --- Autodoc / typing / napoleon quality-of-life ---
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_typehints = "description"
typehints_fully_qualified = False

# extensions = ["breathe", "exhale"]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
