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
import datetime
import pathlib
import subprocess
from contextlib import suppress

sys.path.insert(0, os.path.abspath("."))
insertPaths = [x[0] for x in os.walk(r"../..") if (x[0][-1] != "_")]
for path in insertPaths:
    # print(path)
    sys.path.insert(0, os.path.abspath(path))

import spagat

# print(f"spagat: {spagat.__version__}, {spagat.__file__}")
# TODO: set spagat version

with suppress(ImportError):
    import matplotlib

    matplotlib.use("Agg")
allowed_failures = set()

print("python exec:", sys.executable)

# -- Project information -----------------------------------------------------

project = "SPAGAT"
current_year = datetime.datetime.now().year
copyright = f"2020-{current_year}, SPAGAT Developer Team"
author = "Robin Beer"

# The short X.Y version.
# version = spagat.__version__.split("+")[0]
# The full version, including alpha/beta/rc tags.
# release = spagat.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# TODO: the extension selection is inspired by both FINE and xarray and might include redundancies concerning functionality

extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    'sphinx.ext.autosectionlabel',
]

# recommonmark config
autosectionlabel_prefix_document = True
enable_eval_rst = True

# TODO: add jugit or github links to create issues or merge requests
# extlinks = {
#     "issue": ("https://github.com/pydata/xarray/issues/%s", "GH"),
#     "pull": ("https://github.com/pydata/xarray/pull/%s", "PR"),
# }

autosummary_generate = True
# autodoc_typehints = "none"

numpydoc_class_members_toctree = True
numpydoc_show_class_members = True

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

inheritance_graph_attrs = dict(
    rankdir="LR", size='"26.0, 8.0"', fontsize=14, ratio="compress"
)

inheritance_node_attrs = dict(
    shape="ellipse", fontsize=14, height=0.75, color="lightgray", style="filled"
)


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
today_fmt = "%Y-%m-%d"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"logo_only": True}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/spatial_layer_visualization.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = today_fmt

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    # "numpy": ("https://docs.scipy.org/doc/numpy", None),
    # "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}


source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}