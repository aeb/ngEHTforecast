Installation Guide
==============================

Getting ngEHTforecast
------------------------------
The latest version of ngEHTforecast can be obtained from https://github.com/aeb/ngEHTforecast.

The git repository contains a number of branches. By default, after cloning you will be in the `master` branch, which contains the latest stable release, and is probably where you want to be. The most up-to-date functionality can be found in `develop`, which will periodically be merged into `master`.  (See :ref:`Developer's Guide` for more details about the repository structure.)

Installing with pip
------------------------------
To install ngEHTforecast, in the ngEHTforecast directory,

::

  $ pip install [--upgrade] . [--user]
  

See pip help for more information.

Dependencies
------------------------------
ngEHTforecast requires with the following packages:

* `python <https://www.python.org/downloads>`_ >=3.3
* `scipy <https://www.scipy.org>`_ (pip install scipy)
* `numpy <https://numpy.org>`_ (pip install numpy)
* `matplotlib <https://matplotlib.org>`_ (pip install matplotlib)
* `ehtim <https://github.com/achael/eht-imaging>`_

Generating local version of the documation for ngEHTforecast requires:

* `Sphinx <https://www.sphinx-doc.org>`_ (pip install sphinx)
* `Sphinx-argparse <https://sphinx-argparse.readthedocs.io>`_ (pip install sphinx-argparse)
