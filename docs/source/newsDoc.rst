################
FINE's News Feed
################

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
