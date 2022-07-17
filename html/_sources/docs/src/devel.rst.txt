Developer's Guide
==============================

Getting the latest version
------------------------------
The latest version of ngEHTforecast can be obtained from https://github.com/aeb/ngEHTforecast.

..
   ThemisPy uses the git-flow_ extensions to facilitate development.


Scope and goals
------------------------------
ngEHTforecast seeks to collect and distribute tools for performing science forecasts
for the ngEHT_.  It is dependent on the ability to specify an observation (i.e., a
UVFITS file), and is therefore closely integrated with the existing software infrastructure
for performing data simulation.


Backward compatibility
------------------------------
ngEHTforecast follows the `Semantic Versioning 2.0`_ versioning system.


Portability
------------------------------
ngEHTforecast aspires to be as universally portable as possible.  While library
interdependencies are neither unavoidable nor always undesirable, and ngEHTforecast does
depend on Numpy, SciPy, Matplotlib, and ehtim_, such dependencies should be kept to a
minimum.  No absolute rules are possible for this.  Nevertheless:

1. Consider if additional libraries are required, or if the goal can be
   achieved within the confines existing dependencies.
2. Where additional libraries are required to enable specific functionality
   consider wrapping the necessary import statements in ``try``, ``except``
   blocks with warnings.

For example, imagine that the python package foo is not included as a dependency, though
some functionality is dependent upon it.  In source files where such functions are present,
the following import block should be included,

::

   try:
       import foo
       foo_found = True
   except:
       warnings.warn("foo not found.", Warning)
       foo_found = False

with attendant checks around the functions themselves.       


Documenting with autodoc
------------------------------
All components should be documented with python docstrings that clearly specify the
purpose, function, inputs and outputs (if any).  For example, a new function might
be documented as:

::

   def myfunc(x,y) :
   """
   This function is an example.
   
   Args:
     - x (float): A first parameter.
     - y (float): A second parameter.

   Returns:
     - (float): The sum of the two inputs.
   """

   return x+y

If types are specified, they will be crosslinked with the relevant source documentation.

Mathematical espressions can be included inline using the ``:math:`` command:

::

   """
   :math:`x^2+y^2+\\alpha`
   """

and in displaystyle via:

::
   
   """
   .. math::

      f(x) = \\int_0^\\infty dx \\frac{\\cos(x^2)}{1+g(x)}
   """
   
AMSmath LaTex directives can be inserted, though escape characters must be
escaped themselves if they are imported via autodoc (i.e., the ``\\`` in ``\\alpha``).


Autodoc directives for new module files must be added to the appropriate documentation
file in docs/src, e.g.,

::

   .. automodule:: fisher.fisher_forecast
   :members:









 

.. _ehtim: https://achael.github.io/eht-imaging/array.html      
.. _`Semantic Versioning 2.0`: https://semver.org/
.. _git-flow: https://danielkummer.github.io/git-flow-cheatsheet/
.. _ngEHT: https://www.ngeht.org
