# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..')) # Add project root to path
import safe_store # Import your package

project = 'safe_store'
copyright = '2025, ParisNeo' # Update year/author
author = 'ParisNeo'

# Get version from package
release = safe_store.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # For Google/NumPy style docstrings
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__', # Include __init__ methods
    'show-inheritance': True,
}

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    # Add others if needed (e.g., cryptography, sentence-transformers)
}
