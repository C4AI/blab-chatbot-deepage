# flake8: noqa
#
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "BLAB Chatbot Client for DEEPAGÉ"
# noinspection PyShadowingBuiltins
copyright = "2023, C4AI"
author = "C4AI"
release = subprocess.run(
    ["git", "describe", "--tags"], capture_output=True
).stdout.decode("utf-8")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
]

autoapi_dirs = ["../src"]
autodoc_typehints = "description"

templates_path = ["_templates"]
html_css_files = [
    "override.css",
]
html_js_files = [
    "fix.js",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
add_module_names = False
autoapi_python_class_content = "both"
python_use_unqualified_type_names = True
