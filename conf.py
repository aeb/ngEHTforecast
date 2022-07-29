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
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./ngEHTforecast/'))
sys.path.insert(0, os.path.abspath('./examples/'))
# sys.path.insert(0, os.path.abspath('./scripts/'))
print("Documentation generation searching", sys.path)

# -- Project information -----------------------------------------------------

project = 'ngEHTforecast'
copyright = '2022, Avery E. Broderick'
author = 'Avery E. Broderick'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#extensions = ['sphinx.ext.autodoc']
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.napoleon',
              'sphinxarg.ext',
              'sphinx.ext.autosectionlabel']

#source_suffix = {'.rst': 'restructuredtext', '.txt': 'restructuredtext'} #, '.md': 'markdown'}
#source_suffix = {'.rst': 'restructuredtext'} #, '.txt': 'restructuredtext'} #, '.md': 'markdown'}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['docs/_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

import versioneer
version = versioneer.get_version()
release = version
show_authors = True

master_doc = 'docs/src/index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "bizstyle"
# html_theme = "pyramid"
html_theme = "rtcat_sphinx_theme"
html_theme_path = ["./docs/_themes"]
# html_theme_options = {
#     "rightsidebar": "true",
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['docs/_static']


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {
#     # 'sticky_navigation': True  # Set to False to disable the sticky nav while scrolling.
#     # 'logo_only': True,  # if we have a html_logo below, this shows /only/ the logo with no title text
# }

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = ["../.."]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "./docs/icons/ngeht_medallion_white_on_white_small_512x512.png"
# html_logo = "./docs/icons/ngeht_construction.png"
html_logo = "./docs/icons/ngEHTforecast_logo_transparent.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = "./docs/icons/ngeht_construction.ico"
html_favicon = "./docs/icons/ngeht_blue2.ico"

# Example configuration for intersphinx: refer to the Python standard library.
#intersphinx_mapping = {'https://docs.python.org/': None}
intersphinx_mapping = {'python': ('https://docs.python.org', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('https://matplotlib.org/', None),
                       'ehtim': ('https://achael.github.io/eht-imaging/', None),
                       'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
                       'ngehtsim': ('https://smithsonian.github.io/ngehtsim/html/', None),
#                       'Themis': ('https://perimeterinstitute.github.io/Themis/html/', None)
#                       'Themis': ('/Users/abroderick/Research/Themis/Themis/docs/html',None)
                   }
