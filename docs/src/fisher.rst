.. module:: fisher


Fisher
=========================

Base Class
-------------------------
Definition of the FisherForecast interface and a number of general utility functions.


.. automodule:: fisher.fisher_forecast
   :members:

Station Gain Mitigation
-------------------------
A set of FisherForecast childe classes that facilitate including gain uncertainties.


.. automodule:: fisher.ff_complex_gains
   :members:

Models
-------------------------
Specific models implemented in ngEHTforecast as FisherForecast child classes.  Each inherits all of the functionality of FisherForecast, and may be used anywhere a FisherForecast object is required.


.. automodule:: fisher.ff_models
   :members:


Model Combinations
-------------------------
Classes that can be used to construct new FisherForecast objects from others.  The prototypical example is :class:`fisher.ff_metamodels.FF_Sum`, which constructs a new model by directly summing the intensity maps of many previous models. 


.. automodule:: fisher.ff_metamodels
   :members:
