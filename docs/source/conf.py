# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import shutil

# Point to the src directory
sys.path.insert(0, os.path.abspath('../../src')) 

# --- Auto-copy examples folder ---
root_examples_dir = os.path.abspath('../../examples')
docs_examples_dir = os.path.abspath('./examples')

if os.path.exists(root_examples_dir):
    # Remove old copy if it exists to ensure fresh build
    if os.path.exists(docs_examples_dir):
        shutil.rmtree(docs_examples_dir)
    # Copy the master examples folder into docs/source/
    shutil.copytree(root_examples_dir, docs_examples_dir)
# ---------------------------------

project = 'anglepy'
copyright = '2026, Rajdeep Pathak, Archi Roy, and Tanujit Chakraborty'
author = 'Rajdeep Pathak, Archi Roy, and Tanujit Chakraborty'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.napoleon',
#     'sphinx.ext.mathjax'
# ]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_design',
    'myst_nb',
]
nb_execution_mode = "off"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_logo = '../../images/ANGLE_Logo.jpg'