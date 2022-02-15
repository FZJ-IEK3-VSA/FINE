################
FINE's News Feed
################

*********************
Release version 2.2.2
*********************

FINE release (2.2.2) provides new major functionalities: 

* Add netCDF compatibility for import and export of EnergySystemModel instances to store input and output data. 
* Add generic spatial aggregation and technology aggregation functions for complexity reduction of models with high spatial resolution

Black autoformatting was applied to make the source code easier to read and to simplify contributions. 
Additionally, the installation guide was revised to make the installation easier to handle.

*********************
Release version 2.2.1
*********************
FINE release (2.2.1) provides some changes in code including 

* compatibility to newer versions of pandas (bugs due to reading .xlsx files are fixed)
* correct zlabel description for plotLocationalColorMap in standardIO.py
* add some more documentation to functions

FINE release (2.2.1) fixes a bug in storage.py

* constraints for chargeOperationMax, chargeOperationFix, disChargeOperationMax and dischargeOperationFix should be set up without an error message. 

*********************
Release version 2.2.0
*********************
FINE release (2.2.0) provides some changes in code including bug fixes for 

* plotOperationColorMap (issubclass error should not occur anymore)
* default solver (default solver is changed to None; it is searched for an available solver if no solver is specified)
* transmission components: capacityMin and opexPerOperation can be given as a pandas DataFrame
* postprocessing: no ValueError occur if components are not chosen in the optimized solution
* postprocessing: optimizationSummary is ordered correctly s.t. properties are assigned to the corresponding component.

New features were included: 

* New keyword argument linkedQuantityID: The number of different components can be forced to be the same. 
* Enable time-dependent conversion factors (e.g. for modeling heat pumps)
* Add warning for simultaneous charge and discharge of storage components; users can check if and when simultaneous charge and discharge of storage components occur
* Add operation value for considered time horizon in the optimizationSummary 
* Add new attribute objectiveValue to EnergySystemModel class

*********************
Release version 2.1.1
*********************

FINE release (2.1.1) provides some minor changes in code including bug fixes for 

* Missing labels (for newer pandas versions) 
* setting operationRateFix or operationRateMax for transmission components

*******************
Release version 2.1
*******************

In FINE release (2.1) the following functionalities were included, for example: 

* New time series aggregation method: Segmentation of time series
* Bug Fix: TAC of transmission components within the optimization summary is fixed

*******************
Release version 2.0
*******************

In FINE release (2.0) the following functionalities were included, for example:

* Part load behavior (using piecewise linear functions (-> MILP))
* Robust discrete pipeline design under the consideration of pressure losses (see Robinius et al. (2019) https://link.springer.com/article/10.1007/s10589-019-00085-x and Reuß et al. 2019 (https://www.sciencedirect.com/science/article/abs/pii/S0360319919338625))
* A two-stage approach to reduce computation time of MILPs with time series aggregation, while providing a lower and upper error bound (Kannengießer et al. (2020) https://www.mdpi.com/1996-1073/12/14/2825)
* The option to model nonlinear investment cost functions (via a quadratic function, Lopion et al. (2019) https://www.mdpi.com/1996-1073/12/20/4006)
* A simple approach to model myopic foresight
* Ramping behavior of conversion components
* A beta version for modeling demand side management
