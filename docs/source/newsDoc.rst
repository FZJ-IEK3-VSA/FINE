FINE's News Feed
################

Since version 2.3.3 this news feed is not updated anymore. Please refer to the [release page](https://github.com/FZJ-IEK3-VSA/FINE/releases) for changelogs.

Release version 2.3.2
*********************

IMPORTANT: The name of the package folder has been changed from `FINE` to `fine` in this release. If you still see a `FINE` folder locally after pulling the latest change, you might need to clone the repository to a new folder.

Further, FINE release 2.3.2 provides changes in the requirements:

* Pin `GDAL` to version 3.4.3 because version 3.4.1 is not compatible with the latest Fiona versions.
* Change the repository of `gurobi-logtools` from pypi to conda-forge.

Release version 2.3.1
*********************

FINE release 2.3.1 provides the following changes:

* Adds a performance summary as attribute `EnergySystemModel.performanceSummary`. The performance summary includes Data about RAM usage (assesed by the `psutil` package), Gurobi values (extracted from gurobi log with the `grblogtools` package) and other various parameters such as model buildtime, runtime and time series aggregation paramerters.
* Fixes a bug in the stochastic optimization example.
* Makes subclass `conversionPartLoad` usable again. The `nSegments` parameter has to be set manually depending on the form of the non-linear function.
* Drops the constraint on the version of `pandas` to also work with versions lower than 2.

Release version 2.3.0
*********************

FINE release (2.3.0) provides new major functionalities:

* Representation of multiple investment periods in a transformation pathway (perfect foresight) or for single year optimization (stochastic optimization)
* Consideration of CO<sub>2</sub> budgets for the full transformation pathway
* Consideration of stock including techno-economic parameters depending on commissioning date
* Variable efficiencies for conversion components depending on commissioning date and operation time
* Additional or lowered costs for components which are not present for full investment periods

The ConversionPartLoad class is not supported in this release due to the deprecated package `GPyOpt`.
Also, the installation method has been changed from `setup.py` to `pyproject.toml`.

Release version 2.2.2
*********************

FINE release (2.2.2) provides new major functionalities: 

* Add netCDF compatibility for import and export of EnergySystemModel instances to store input and output data. 
* Add generic spatial aggregation and technology aggregation functions for complexity reduction of models with high spatial resolution

Black autoformatting was applied to make the source code easier to read and to simplify contributions. 
Additionally, the installation guide was revised to make the installation easier to handle.

Release version 2.2.1
*********************

FINE release (2.2.1) provides some changes in code including 

* compatibility to newer versions of pandas (bugs due to reading .xlsx files are fixed)
* correct zlabel description for plotLocationalColorMap in standardIO.py
* add some more documentation to functions

FINE release (2.2.1) fixes a bug in storage.py

* constraints for chargeOperationMax, chargeOperationFix, disChargeOperationMax and dischargeOperationFix should be set up without an error message. 

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

Release version 2.1.1
*********************

FINE release (2.1.1) provides some minor changes in code including bug fixes for 

* Missing labels (for newer pandas versions) 
* setting operationRateFix or operationRateMax for transmission components

Release version 2.1
*******************

In FINE release (2.1) the following functionalities were included, for example: 

* New time series aggregation method: Segmentation of time series
* Bug Fix: TAC of transmission components within the optimization summary is fixed

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
