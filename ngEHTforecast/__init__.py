"""
This is a set of tools for forecasting ngEHT science capabilities.

Tools may be accessed via

>>> import ngEHTforecast as nf

Available subpackages
---------------------
fisher
   Fisher matrix forecasting tools
data
   Data production and preprocessing tools
"""

__author__="Avery E. Broderick"
__bibtex__ = r"""@Article{Broderick \& Pesce, 2022,
  %%% Fill in from ADS!
}"""

__all__=['fisher', 'data']
from . import *

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


