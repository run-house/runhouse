# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "Runhouse"
copyright = "2023, Runhouse Inc"
author = "the Runhouse team 🏃‍♀️🏠"

# The full version, including alpha/beta/rc tags
import runhouse

release = runhouse.__version__

# -- General configuration ---------------------------------------------------

needs_sphinx = "4.5.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_click.ext",
    "myst_parser",
    "sphinx_thebe",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "_ext.json_globaltoc",
]

autodoc_typehints_format = "short"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

markdown_http_base = "/examples/guide"
markdown_anchor_sections = True

if tags.has("json"):
    html_link_suffix = ""
    json_baseurl = "examples/"

sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto-examples',  # path to where to save gallery generated output
     'filename_pattern': r"\.py",
     'plot_gallery': 'False',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
if not tags.has("json"):
    html_theme = "sphinx_book_theme"

html_title = "Runhouse"
html_logo = "../docs/_static/rh_1.png"
html_favicon = "../docs/_static/favicon.ico"

html_theme_options = {
    "repository_url": "https://github.com/run-house/runhouse",
    "repository_branch": "stable",
    "path_to_docs": "examples/",  # Path to the documentation, relative to the repository root (e.g. docs/)
    "use_repository_button": True,
    "logo_only": True,
    "home_page_in_toc": True,
    "use_download_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
}

html_show_sourcelink = False

# pygments_style = None
# autosummary_generate = True
# napolean_use_rtype = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../docs/_static"]

# Cache the webpages for up to a week and make them immutable.
# This will allow Vercel’s Edge Network to cache the website’s content globally.
# For more info: https://vercel.com/docs/edge-network/caching
html_extra_headers = {"Cache-Control": "public, max-age=604800, immutable"}
