"""Conversion component with nonlinear part-load efficiency modeling.

This module provides the ``ConversionPartLoad`` class and its associated optimization
model ``ConversionPartLoadModel``. These extend the standard ``Conversion`` component
to capture load-dependent conversion efficiencies using piecewise linear approximations
with SOS2 (Special Ordered Sets of type 2) constraints.

Helper functions for piecewise linearization, input validation, and boundary correction
are also included.
"""

from fine.conversion import Conversion, ConversionModel
from fine.utils import checkDataFrameConversionFactor, checkCallableConversionFactor
from fine import utils
import pyomo.environ as pyomo
import pandas as pd
import numpy as np
import pwlf


def pieceWiseLinearization(functionOrRaw, xLowerBound, xUpperBound, nSegments):
    """Perform piecewise linear approximation of a nonlinear part-load efficiency curve.

    Uses the PWLF (piecewise linear fit) library to approximate a nonlinear function or
    raw data points with a piecewise linear function consisting of ``nSegments`` line segments.

    If the input is raw data (DataFrame) that does not cover the full [0, 1] range, the missing
    regions are extrapolated by holding the boundary values constant.

    Parameters
    ----------
    functionOrRaw : callable or pandas.DataFrame
        Either a callable function ``f(x) -> y`` mapping operation level to conversion factor,
        or a DataFrame with two columns: the first column contains operation levels (x) and the
        second contains corresponding conversion factors (y).
    xLowerBound : float
        Lower bound of the operation level range (typically 0).
    xUpperBound : float
        Upper bound of the operation level range (typically 1).
    nSegments : int or None
        Number of line segments for the piecewise linearization. If None, defaults to 5.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``'xSegments'`` (numpy.ndarray): breakpoint x-coordinates of the piecewise linear function.
        - ``'ySegments'`` (numpy.ndarray): breakpoint y-coordinates of the piecewise linear function.
        - ``'nSegments'`` (int): number of segments used.
        - ``'Rsquared'`` (float): overall R-squared value of the fit.
        - ``'R2values'`` (numpy.ndarray): per-segment R-squared values.
    """
    if callable(functionOrRaw):
        nPointsForInputData = 1000
        x = np.linspace(xLowerBound, xUpperBound, nPointsForInputData)
        y = np.array([functionOrRaw(x_i) for x_i in x])
    else:
        x = np.array(functionOrRaw.iloc[:, 0])
        y = np.array(functionOrRaw.iloc[:, 1])
        if 0.0 not in x:
            xMinDefined = np.amin(x)
            xMaxDefined = np.amax(x)
            lenIntervalDefined = xMaxDefined - xMinDefined
            lenIntervalUndefined = xMinDefined
            nPointsUndefined = lenIntervalUndefined * (x.size / lenIntervalDefined)
            xMinIndex = np.argmin(x)
            for i in range(int(nPointsUndefined)):
                x = np.append(x, [i / int(nPointsUndefined + 1) * lenIntervalUndefined])
                y = np.append(y, y[xMinIndex])
        if 1.0 not in x:
            xMinDefined = np.amin(x)
            xMaxDefined = np.amax(x)
            lenIntervalDefined = xMaxDefined - xMinDefined
            lenIntervalUndefined = 1.0 - xMaxDefined
            nPointsUndefined = lenIntervalUndefined * (x.size / lenIntervalDefined)
            xMaxIndex = np.argmax(x)
            for i in range(int(nPointsUndefined)):
                x = np.append(
                    x,
                    [
                        xMaxDefined
                        + (i + 1) / int(nPointsUndefined) * lenIntervalUndefined
                    ],
                )
                y = np.append(y, y[xMaxIndex])

    myPwlf = pwlf.PiecewiseLinFit(x, y)

    if nSegments is None:
        nSegments = 5

    xSegments = myPwlf.fit(nSegments)

    # Get the y segments
    ySegments = myPwlf.predict(xSegments)

    # Calcualte the R^2 value
    Rsquared = myPwlf.r_squared()

    # Calculate the piecewise R^2 value
    R2values = np.zeros(nSegments)
    for i in range(nSegments):
        # Segregate the data based on break point locations
        xMin = myPwlf.fit_breaks[i]
        xMax = myPwlf.fit_breaks[i + 1]
        xTemp = myPwlf.x_data
        yTemp = myPwlf.y_data
        indTemp = np.where(xTemp >= xMin)
        xTemp = myPwlf.x_data[indTemp]
        yTemp = myPwlf.y_data[indTemp]
        indTemp = np.where(xTemp <= xMax)
        xTemp = xTemp[indTemp]
        yTemp = yTemp[indTemp]

        # Predict for the new data
        yHatTemp = myPwlf.predict(xTemp)

        # Calcualte ssr
        e = yHatTemp - yTemp
        ssr = np.dot(e, e)

        # Calculate sst
        yBar = np.ones(yTemp.size) * np.mean(yTemp)
        ydiff = yTemp - yBar
        sst = np.dot(ydiff, ydiff)

        for j in range(nSegments):
            if sst == 0:
                R2values[j] = np.nan
            else:
                R2values[j] = 1.0 - (ssr / sst)

    return {
        "xSegments": xSegments,
        "ySegments": ySegments,
        "nSegments": nSegments,
        "Rsquared": Rsquared,
        "R2values": R2values,
    }


def getDiscretizedPartLoad(commodityConversionFactorsPartLoad, nSegments):
    """Preprocess commodity conversion factors into discretized piecewise linear representations.

    For each commodity in ``commodityConversionFactorsPartLoad``, this function either applies
    piecewise linearization (if the conversion factor is a callable or DataFrame) or assigns
    constant segment values (if the conversion factor is 1 or -1). After discretization, the
    non-function commodity's x-segments are aligned with the function commodity's breakpoints.

    Parameters
    ----------
    commodityConversionFactorsPartLoad : dict
        Dictionary mapping commodity names (str) to their part-load conversion factors.
        Exactly one commodity must have a callable or DataFrame value, and exactly one
        must have a value of 1 or -1.
    nSegments : int or None
        Number of line segments for piecewise linearization. If None, defaults to 5.

    Returns
    -------
    tuple of (dict, int)
        A tuple of ``(discretizedPartLoad, nSegments)`` where ``discretizedPartLoad`` maps
        each commodity to a dict with keys ``'xSegments'``, ``'ySegments'``, ``'nSegments'``,
        ``'Rsquared'``, and ``'R2values'``.

    """
    discretizedPartLoad = {
        commod: None for commod in commodityConversionFactorsPartLoad.keys()
    }
    functionOrRawCommod = None
    nonFunctionOrRawCommod = None
    for commod, conversionFactor in commodityConversionFactorsPartLoad.items():
        if (isinstance(conversionFactor, pd.DataFrame)) or (callable(conversionFactor)):
            discretizedPartLoad[commod] = pieceWiseLinearization(
                functionOrRaw=conversionFactor,
                xLowerBound=0,
                xUpperBound=1,
                nSegments=nSegments,
            )
            functionOrRawCommod = commod
            nSegments = discretizedPartLoad[commod]["nSegments"]
        elif conversionFactor in (1, -1):
            discretizedPartLoad[commod] = {
                "xSegments": None,
                "ySegments": None,
                "nSegments": None,
                "Rsquared": 1.0,
                "R2values": 1.0,
            }
            nonFunctionOrRawCommod = commod
    discretizedPartLoad[nonFunctionOrRawCommod]["xSegments"] = discretizedPartLoad[
        functionOrRawCommod
    ]["xSegments"]
    discretizedPartLoad[nonFunctionOrRawCommod]["ySegments"] = np.array(
        [commodityConversionFactorsPartLoad[nonFunctionOrRawCommod]] * (nSegments + 1)
    )
    discretizedPartLoad[nonFunctionOrRawCommod]["nSegments"] = nSegments
    checkAndCorrectDiscretizedPartloads(discretizedPartLoad)
    return discretizedPartLoad, nSegments


def checkAndCorrectDiscretizedPartloads(discretizedPartLoad):
    """Validate and correct discretized part-load breakpoints to the [0, 1] range.

    Checks that all x-segment values (operation levels) lie within [0, 1] and all y-segment
    values (conversion factors) are non-negative. If a single boundary point falls slightly
    outside the valid range, it is corrected by linear extrapolation from the two nearest
    breakpoints. Raises a ``ValueError`` if more than one point is out of range or if the
    out-of-range point is not at a boundary.

    Parameters
    ----------
    discretizedPartLoad : dict
        Dictionary mapping commodity names to their discretization dicts, each containing
        ``'xSegments'`` and ``'ySegments'`` arrays.

    Returns
    -------
    dict
        The corrected ``discretizedPartLoad`` dictionary (modified in-place).

    Raises
    ------
    ValueError
        If more than one breakpoint is out of the valid range, or if an interior breakpoint
        is out of range.

    """
    for commod, conversionFactor in discretizedPartLoad.items():
        # ySegments
        if not np.all(
            conversionFactor["ySegments"] == conversionFactor["ySegments"][0]
        ):
            if any(conversionFactor["ySegments"] < 0):
                if sum(conversionFactor["ySegments"] < 0) > 1:
                    raise ValueError(
                        "There is at least two partLoad efficiency values that are < 0. Please check your partLoadEfficiency data or function visually."
                    )
                # First element
                if np.where(conversionFactor["ySegments"] < 0)[0][0] == 0:
                    # Correct efficiency < 0 for index = 0 -> construct line
                    coefficients = np.polyfit(
                        conversionFactor["xSegments"][0:2],
                        conversionFactor["ySegments"][0:2],
                        1,
                    )
                    discretizedPartLoad[commod]["ySegments"][0] = 0
                    discretizedPartLoad[commod]["xSegments"][0] = (
                        -coefficients[1] / coefficients[0]
                    )

                # Last element
                elif (
                    np.where(conversionFactor["ySegments"] < 0)[0][0]
                    == len(conversionFactor["ySegments"]) - 1
                ):
                    # Correct efficiency < for index = 0 -> construct line
                    coefficients = np.polyfit(
                        conversionFactor["xSegments"][-2:],
                        conversionFactor["ySegments"][-2:],
                        1,
                    )
                    discretizedPartLoad[commod]["ySegments"][-1] = 0
                    discretizedPartLoad[commod]["xSegments"][-1] = (
                        -coefficients[1] / coefficients[0]
                    )
                else:
                    raise ValueError(
                        "PartLoad efficiency value < 0 detected where slope cannot be constructed. Please check your partLoadEfficiency data or function visually."
                    )
        # xSegments
        if any(conversionFactor["xSegments"] < 0):
            if sum(conversionFactor["xSegments"] < 0) > 1:
                raise ValueError(
                    "There is at least two partLoad efficiency values that are < 0. Please check your partLoadEfficiency data or function visually."
                )
            # First element
            if np.where(conversionFactor["xSegments"] < 0)[0][0] == 0:
                coefficients = np.polyfit(
                    conversionFactor["xSegments"][0:2],
                    conversionFactor["ySegments"][0:2],
                    1,
                )
                discretizedPartLoad[commod]["xSegments"][0] = 0
                discretizedPartLoad[commod]["ySegments"][0] = coefficients[1]
            else:
                raise ValueError(
                    "PartLoad efficiency value < 0 detected where slope cannot be constructed. Please check your partLoadEfficiency data or function visually."
                )
        if any(conversionFactor["xSegments"] > 1):
            if sum(conversionFactor["xSegments"] > 1) > 1:
                raise ValueError(
                    "There is at least two partLoad efficiency values that are > 1. Please check your partLoadEfficiency data or function visually."
                )
            # Last element
            if (
                np.where(conversionFactor["xSegments"] > 1)[0][0]
                == len(conversionFactor["xSegments"]) - 1
            ):
                coefficients = np.polyfit(
                    conversionFactor["xSegments"][-2:],
                    conversionFactor["ySegments"][-2:],
                    1,
                )
                discretizedPartLoad[commod]["xSegments"][0] = 1
                discretizedPartLoad[commod]["ySegments"][0] = (
                    coefficients[0] + coefficients[1]
                )
            else:
                raise ValueError(
                    "PartLoad efficiency value > 1 detected where slope cannot be constructed. Please check your partLoadEfficiency data or function visually."
                )

    return discretizedPartLoad


def checkCommodityConversionFactorsPartLoad(commodityConversionFactorsPartLoad):
    """Validate that part-load commodity conversion factors are correctly specified.

    Ensures that:

    - Exactly one commodity has a conversion factor of 1 or -1 (the nominal commodity).
    - Exactly one commodity has a callable function or a DataFrame describing the
      nonlinear part-load efficiency curve.

    The callable or DataFrame conversion factor is further validated via
    ``checkCallableConversionFactor`` or ``checkDataFrameConversionFactor`` from
    ``fine.utils``.

    Parameters
    ----------
    commodityConversionFactorsPartLoad : dict_values
        The values of the ``commodityConversionFactorsPartLoad`` dictionary, where each
        value is either ``1``, ``-1``, a callable, or a ``pandas.DataFrame``.

    Raises
    ------
    TypeError
        If no conversion factor equals 1 or -1, or if no conversion factor is a callable
        or DataFrame.

    """
    partLoadCommodPresent = False
    nonPartLoadCommodPresent = False

    for conversionFactor in commodityConversionFactorsPartLoad:
        if isinstance(conversionFactor, pd.DataFrame):
            checkDataFrameConversionFactor(conversionFactor)
            partLoadCommodPresent = True
        elif callable(conversionFactor):
            checkCallableConversionFactor(conversionFactor)
            partLoadCommodPresent = True
        elif conversionFactor in (1, -1):
            nonPartLoadCommodPresent = True

    if not nonPartLoadCommodPresent:
        raise TypeError("One conversion factor needs to be either 1 or -1.")
    if not partLoadCommodPresent:
        raise TypeError(
            "One conversion factor needs to be either a callable function or a list of two-dimensional data points."
        )


class ConversionPartLoad(Conversion):
    """A Conversion component with nonlinear part-load efficiency behavior.

    ``ConversionPartLoad`` extends the standard ``Conversion`` component to model the
    nonlinear relationship between a component's operation level and its conversion
    efficiency. Instead of using fixed (constant) conversion factors, this class
    approximates a nonlinear efficiency curve with a piecewise linear function using
    the `PWLF <https://github.com/cjekel/piecewise_linear_fit_py>`_ library.

    The piecewise linear approximation is formulated in the optimization model using
    Special Ordered Sets of type 2 (SOS2) constraints. This ensures that only two
    adjacent breakpoints of the piecewise linear function can be active simultaneously,
    correctly interpolating between them.

    .. note::
        When using ``ConversionPartLoad``, it is recommended to visually inspect the
        piecewise linearization to verify that it approximates the original curve
        with sufficient accuracy. See Example 11 (Partload) for guidance.

    """

    def __init__(
        self,
        esM,
        name,
        physicalUnit,
        commodityConversionFactors,
        commodityConversionFactorsPartLoad,
        nSegments=None,
        **kwargs,
    ):
        """Create a ConversionPartLoad instance.

        Capacities are given in the physical unit of the plants. The
        ``ConversionPartLoad``-specific parameters are described below. All other
        parameters are inherited from the ``Conversion`` class and the ``Component``
        base class.

        Parameters
        ----------
        esM : EnergySystemModel
            The energy system model to which the component is added.
        name : str
            Name of the component. Must be unique within the energy system model.
        physicalUnit : str
            Reference physical unit of the plants (e.g., ``'GW_el'``).
        commodityConversionFactors : dict
            Constant conversion factors (see ``Conversion`` class for details).
        commodityConversionFactorsPartLoad : dict or tuple
            Specifies the part-load efficiency behavior. Can be provided in two forms:

            **As a dict:** Maps commodity names (str) to their part-load conversion
            factors. Exactly one commodity must have a value of ``1`` or ``-1``
            (the nominal commodity), and exactly one must provide the nonlinear
            efficiency curve as either:

            - A **callable** ``f(x) -> y`` where ``x`` is the operation level in
              [0, 1] and ``y`` is the conversion factor at that operation level.
            - A **pandas.DataFrame** with two columns: operation levels (x) and
              corresponding conversion factors (y).

            A negative value indicates commodity consumption; a positive value
            indicates commodity production.

            **As a tuple:** A pre-computed ``(discretizedPartLoad, nSegments)`` tuple,
            useful for reusing previously computed discretizations.

            Example
            -------
            An electrolyzer converting electricity to hydrogen with load-dependent
            efficiency::

                operation_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                efficiency = [0.1, 0.15, 0.5, 0.7, 0.7, 0.65, 0.63, 0.62, 0.61, 0.60]
                partLoadData = pd.DataFrame({"x": operation_level, "y": efficiency})

                commodityConversionFactorsPartLoad = {
                    "electricity": -1,
                    "hydrogen": partLoadData,
                }

        nSegments : int or None, optional
            Number of line segments for the piecewise linearization. Determines the
            number of point variables (``nSegments + 1``) and segment variables
            (``nSegments``). Recommended values are between 3 and 7, as computational
            cost increases significantly with more segments.
            If None, defaults to 5.
        **kwargs
            All other keyword arguments accepted by the ``Conversion`` class
            (e.g., ``hasCapacityVariable``, ``investPerCapacity``, ``bigM``, etc.).

        """
        Conversion.__init__(
            self, esM, name, physicalUnit, commodityConversionFactors, **kwargs
        )

        self.modelingClass = ConversionPartLoadModel

        # TODO: Make compatible with conversion
        utils.checkNumberOfConversionFactors(commodityConversionFactors)

        if isinstance(commodityConversionFactorsPartLoad, dict):
            # TODO: Multiple conversionPartLoads
            utils.checkNumberOfConversionFactors(commodityConversionFactorsPartLoad)
            utils.checkCommodities(esM, set(commodityConversionFactorsPartLoad.keys()))
            checkCommodityConversionFactorsPartLoad(
                commodityConversionFactorsPartLoad.values()
            )
            self.commodityConversionFactorsPartLoad = commodityConversionFactorsPartLoad
            self.discretizedPartLoad, self.nSegments = getDiscretizedPartLoad(
                commodityConversionFactorsPartLoad, nSegments
            )

        elif isinstance(commodityConversionFactorsPartLoad, tuple):
            utils.checkNumberOfConversionFactors(
                commodityConversionFactorsPartLoad[0].keys()
            )
            self.discretizedPartLoad = commodityConversionFactorsPartLoad[0]
            self.nSegments = commodityConversionFactorsPartLoad[1]


class ConversionPartLoadModel(ConversionModel):
    """Optimization model for ConversionPartLoad components.

    This class extends ``ConversionModel`` with the additional sets, variables, and
    constraints needed to formulate the piecewise linear part-load efficiency model
    using SOS2 (Special Ordered Sets of type 2) constraints.

    The model introduces three types of additional decision variables per component,
    location, and time step:

    - **Discretization point variables** (continuous, non-negative): one per breakpoint
      of the piecewise linear function (``nSegments + 1`` variables).
    - **Discretization segment continuous variables** (continuous, non-negative): one per
      line segment (``nSegments`` variables), representing the weight of each segment.
    - **Discretization segment binary variables** (binary): one per line segment,
      indicating which segment is active.

    An instance is automatically created when a ``ConversionPartLoad`` component is
    initialized.

    """

    def __init__(self):
        super().__init__()
        self.abbrvName = "partLoad"
        self.dimension = "1dim"
        self._operationVariablesOptimum = {}
        self._discretizationPointVariablesOptimum = {}
        self._discretizationSegmentConVariablesOptimum = {}
        self._discretizationSegmentBinVariablesOptimum = {}

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def initDiscretizationPointVarSet(self, pyM):
        """Declare the discretization point variable index set.

        Creates a 3-dimensional Pyomo set indexed by ``(location, componentName,
        discreteStep)`` where ``discreteStep`` ranges from 0 to ``nSegments``
        (inclusive), representing all breakpoints of the piecewise linear function.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        # Set for operation variables
        def initDiscretizationPointVarSet(pyM):
            return (
                (loc, compName, discreteStep)
                for compName, comp in compDict.items()
                for loc in compDict[compName].processedLocationalEligibility.index
                if compDict[compName].processedLocationalEligibility[loc] == 1
                for discreteStep in range(compDict[compName].nSegments + 1)
            )

        setattr(
            pyM,
            "discretizationPointVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=initDiscretizationPointVarSet),
        )

    def initDiscretizationSegmentVarSet(self, pyM):
        """Declare the discretization segment variable index set.

        Creates a 3-dimensional Pyomo set indexed by ``(location, componentName,
        discreteStep)`` where ``discreteStep`` ranges from 0 to ``nSegments - 1``,
        representing each line segment of the piecewise linear function.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        # Set for operation variables
        def initDiscretizationSegmentVarSet(pyM):
            return (
                (loc, compName, discreteStep)
                for compName, comp in compDict.items()
                for loc in compDict[compName].processedLocationalEligibility.index
                if compDict[compName].processedLocationalEligibility[loc] == 1
                for discreteStep in range(compDict[compName].nSegments)
            )

        setattr(
            pyM,
            "discretizationSegmentVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=initDiscretizationSegmentVarSet),
        )

    def declareSets(self, esM, pyM):
        """Declare sets including discretization point and segment variable sets.

        Extends the parent ``ConversionModel.declareSets`` with the additional
        discretization point and segment index sets needed for the piecewise
        linear part-load formulation.

        Parameters
        ----------
        esM : EnergySystemModel
            The energy system model instance.
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        super().declareSets(esM, pyM)

        # Declare operation variable sets
        self.initDiscretizationPointVarSet(pyM)
        self.initDiscretizationSegmentVarSet(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareDiscretizationPointVariables(self, pyM):
        """Declare continuous non-negative discretization point variables.

        One variable per breakpoint of the piecewise linear function, indexed over
        the discretization point variable set and the time set.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        setattr(
            pyM,
            "discretizationPoint_" + self.abbrvName,
            pyomo.Var(
                getattr(pyM, "discretizationPointVarSet_" + self.abbrvName),
                pyM.timeSet,
                domain=pyomo.NonNegativeReals,
            ),
        )

    def declareDiscretizationSegmentBinVariables(self, pyM):
        """Declare binary discretization segment variables.

        One binary variable per line segment, indexed over the discretization
        segment variable set and the time set. Indicates which segment of the
        piecewise linear function is active.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        setattr(
            pyM,
            "discretizationSegmentBin_" + self.abbrvName,
            pyomo.Var(
                getattr(pyM, "discretizationSegmentVarSet_" + self.abbrvName),
                pyM.timeSet,
                domain=pyomo.Binary,
            ),
        )

    def declareDiscretizationSegmentConVariables(self, pyM):
        """Declare continuous non-negative discretization segment variables.

        One continuous variable per line segment, indexed over the discretization
        segment variable set and the time set. Represents the weight/contribution
        of each segment.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        setattr(
            pyM,
            "discretizationSegmentCon_" + self.abbrvName,
            pyomo.Var(
                getattr(pyM, "discretizationSegmentVarSet_" + self.abbrvName),
                pyM.timeSet,
                domain=pyomo.NonNegativeReals,
            ),
        )

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary, relevanceThreshold):
        """Declare all variables including discretization point and segment variables.

        Extends the parent ``ConversionModel.declareVariables`` with the three
        additional variable types for the piecewise linear formulation.

        Parameters
        ----------
        esM : EnergySystemModel
            The energy system model instance.
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.
        relaxIsBuiltBinary : bool
            If True, solve as relaxed LP to obtain a lower bound.
        relevanceThreshold : float or None
            Force operation parameters to 0 if below this threshold.

        """
        super().declareVariables(esM, pyM, relaxIsBuiltBinary, relevanceThreshold)

        # Operation of component [commodityUnit]
        self.declareDiscretizationPointVariables(pyM)
        # Operation of component [commodityUnit]
        self.declareDiscretizationSegmentBinVariables(pyM)
        # Operation of component [commodityUnit]
        self.declareDiscretizationSegmentConVariables(pyM)

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def segmentSOS1(self, pyM):
        """Enforce that exactly one segment is active at each time step (SOS1 constraint).

        The binary segment variables must sum to 1, ensuring that exactly one segment
        of the piecewise linear function is selected while all others are zero.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationSegmentBinVar = getattr(
            pyM, "discretizationSegmentBin_" + self.abbrvName
        )
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def segmentSOS1(pyM, loc, compName, ip, p, t):
            return (
                sum(
                    discretizationSegmentBinVar[loc, compName, discretStep, ip, p, t]
                    for discretStep in range(compDict[compName].nSegments)
                )
                == 1
            )

        setattr(
            pyM,
            "ConstrSegmentSOS1_" + abbrvName,
            pyomo.Constraint(opVarSet, pyM.intraYearTimeSet, rule=segmentSOS1),
        )

    def segmentBigM(self, pyM):
        """Link continuous segment variables to their binary counterparts via Big-M constraints.

        If a segment's binary variable is 0, the corresponding continuous variable is
        forced to 0. If the binary is 1, the continuous variable is bounded by ``bigM``.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationSegmentConVar = getattr(
            pyM, "discretizationSegmentCon_" + self.abbrvName
        )
        discretizationSegmentBinVar = getattr(
            pyM, "discretizationSegmentBin_" + self.abbrvName
        )
        discretizationSegmentVarSet = getattr(
            pyM, "discretizationSegmentVarSet_" + self.abbrvName
        )

        def segmentBigM(pyM, loc, compName, discretStep, ip, p, t):
            return (
                discretizationSegmentConVar[loc, compName, discretStep, ip, p, t]
                <= discretizationSegmentBinVar[loc, compName, discretStep, ip, p, t]
                * compDict[compName].bigM
            )

        setattr(
            pyM,
            "ConstrSegmentBigM_" + abbrvName,
            pyomo.Constraint(
                discretizationSegmentVarSet, pyM.timeSet, rule=segmentBigM
            ),
        )

    def segmentCapacityConstraint(self, pyM, esM):
        """Ensure segment variables sum to the installed capacity.

        The sum of all continuous segment variables must equal the component's installed
        capacity (scaled by ``hoursPerTimeStep`` or ``hoursPerSegment``), linking the
        piecewise linear formulation to the capacity decision.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.
        esM : EnergySystemModel
            The energy system model instance.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationSegmentConVar = getattr(
            pyM, "discretizationSegmentCon_" + self.abbrvName
        )
        capVar = getattr(pyM, "cap_" + abbrvName)
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)

        if not pyM.hasSegmentation:

            def segmentCapacityConstraint(pyM, loc, compName, ip, p, t):
                return (
                    sum(
                        discretizationSegmentConVar[
                            loc, compName, discretStep, ip, p, t
                        ]
                        for discretStep in range(compDict[compName].nSegments)
                    )
                    == esM.hoursPerTimeStep * capVar[loc, compName, ip]
                )

            setattr(
                pyM,
                "ConstrSegmentCapacity_" + abbrvName,
                pyomo.Constraint(
                    opVarSet, pyM.intraYearTimeSet, rule=segmentCapacityConstraint
                ),
            )
        else:

            def segmentCapacityConstraint(pyM, loc, compName, ip, p, t):
                return (
                    sum(
                        discretizationSegmentConVar[
                            loc, compName, discretStep, ip, p, t
                        ]
                        for discretStep in range(compDict[compName].nSegments)
                    )
                    == esM.hoursPerSegment.to_dict()[p, t] * capVar[loc, compName, ip]
                )

            setattr(
                pyM,
                "ConstrSegmentCapacity_" + abbrvName,
                pyomo.Constraint(opVarSet, pyM.timeSet, rule=segmentCapacityConstraint),
            )

            def segmentCapacityConstraint(pyM, loc, compName, p, t, ip):
                return (
                    sum(
                        discretizationSegmentConVar[loc, compName, discretStep, p, t]
                        for discretStep in range(compDict[compName].nSegments)
                    )
                    == esM.hoursPerSegment.to_dict()[p, t] * capVar[loc, compName, ip]
                )

            setattr(
                pyM,
                "ConstrSegmentCapacity_" + abbrvName,
                pyomo.Constraint(opVarSet, pyM.timeSet, rule=segmentCapacityConstraint),
            )

    def pointCapacityConstraint(self, pyM, esM):
        """Ensure point variables sum to the installed capacity.

        The sum of all discretization point variables must equal the component's installed
        capacity (scaled by ``hoursPerTimeStep`` or ``hoursPerSegment``).

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.
        esM : EnergySystemModel
            The energy system model instance.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(
            pyM, "discretizationPoint_" + self.abbrvName
        )
        capVar = getattr(pyM, "cap_" + abbrvName)
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)

        if not pyM.hasSegmentation:

            def pointCapacityConstraint(pyM, loc, compName, ip, p, t):
                nPoints = compDict[compName].nSegments + 1
                return (
                    sum(
                        discretizationPointConVar[loc, compName, discretStep, ip, p, t]
                        for discretStep in range(nPoints)
                    )
                    == esM.hoursPerTimeStep * capVar[loc, compName, ip]
                )

            setattr(
                pyM,
                "ConstrPointCapacity_" + abbrvName,
                pyomo.Constraint(
                    opVarSet, pyM.intraYearTimeSet, rule=pointCapacityConstraint
                ),
            )
        else:

            def pointCapacityConstraint(pyM, loc, compName, ip, p, t):
                nPoints = compDict[compName].nSegments + 1
                return (
                    sum(
                        discretizationPointConVar[loc, compName, discretStep, ip, p, t]
                        for discretStep in range(nPoints)
                    )
                    == esM.hoursPerSegment.to_dict()[p, t] * capVar[loc, compName, ip]
                )

            setattr(
                pyM,
                "ConstrPointCapacity_" + abbrvName,
                pyomo.Constraint(opVarSet, pyM.timeSet, rule=pointCapacityConstraint),
            )

    def pointSOS2(self, pyM):
        """Enforce the SOS2 adjacency condition on point variables.

        Ensures that at most two consecutive discretization point variables can be
        non-zero at any time step. Each point variable is bounded by the sum of the
        continuous segment variables of its adjacent segments. This is the core SOS2
        constraint that guarantees correct piecewise linear interpolation.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(
            pyM, "discretizationPoint_" + self.abbrvName
        )
        discretizationSegmentConVar = getattr(
            pyM, "discretizationSegmentCon_" + self.abbrvName
        )
        discretizationPointVarSet = getattr(
            pyM, "discretizationPointVarSet_" + self.abbrvName
        )

        def pointSOS2(pyM, loc, compName, discretStep, ip, p, t):
            points = list(range(compDict[compName].nSegments + 1))
            segments = list(range(compDict[compName].nSegments))

            if discretStep == points[0]:
                return (
                    discretizationPointConVar[loc, compName, points[0], ip, p, t]
                    <= discretizationSegmentConVar[loc, compName, segments[0], ip, p, t]
                )
            if discretStep == points[-1]:
                return (
                    discretizationPointConVar[loc, compName, points[-1], ip, p, t]
                    <= discretizationSegmentConVar[
                        loc, compName, segments[-1], ip, p, t
                    ]
                )
            return (
                discretizationPointConVar[loc, compName, discretStep, ip, p, t]
                <= discretizationSegmentConVar[loc, compName, discretStep - 1, ip, p, t]
                + discretizationSegmentConVar[loc, compName, discretStep, ip, p, t]
            )

        setattr(
            pyM,
            "ConstrPointSOS2_" + abbrvName,
            pyomo.Constraint(discretizationPointVarSet, pyM.timeSet, rule=pointSOS2),
        )

    def partLoadOperationOutput(self, pyM):
        """Link the operation variable to the piecewise linear part-load representation.

        Constrains the operation variable to equal the weighted sum of the discretization
        point variables multiplied by their corresponding x-segment breakpoint values.
        This maps the abstract piecewise linear formulation back to the physical operation
        level of the component.

        Parameters
        ----------
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(
            pyM, "discretizationPoint_" + self.abbrvName
        )
        opVar, opVarSet = (
            getattr(pyM, "op_" + abbrvName),
            getattr(pyM, "operationVarSet_" + abbrvName),
        )

        def partLoadOperationOutput(pyM, loc, compName, ip, p, t):
            nPoints = compDict[compName].nSegments + 1

            return opVar[loc, compName, ip, p, t] == sum(
                discretizationPointConVar[loc, compName, discretStep, ip, p, t]
                * compDict[compName].discretizedPartLoad[
                    list(compDict[compName].discretizedPartLoad.keys())[0]
                ]["xSegments"][discretStep]
                for discretStep in range(nPoints)
            )

        setattr(
            pyM,
            "ConstrpartLoadOperationOutput_" + abbrvName,
            pyomo.Constraint(
                opVarSet, pyM.intraYearTimeSet, rule=partLoadOperationOutput
            ),
        )

    def declareComponentConstraints(self, esM, pyM):
        """Declare all constraints including piecewise linear part-load constraints.

        Extends the parent ``ConversionModel.declareComponentConstraints`` with six
        additional constraints: SOS1 on segments, Big-M linking, segment and point
        capacity constraints, SOS2 adjacency, and the operation output mapping.

        Parameters
        ----------
        esM : EnergySystemModel
            The energy system model instance.
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        super().declareComponentConstraints(esM, pyM)

        ################################################################################################################
        #                                         Add piecewise linear part load efficiency constraints                                        #
        ################################################################################################################

        self.segmentSOS1(pyM)
        self.segmentBigM(pyM)
        self.segmentCapacityConstraint(pyM, esM)
        self.pointCapacityConstraint(pyM, esM)
        self.pointSOS2(pyM)
        self.partLoadOperationOutput(pyM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        """Check if operation variables exist for a given location and commodity.

        Parameters
        ----------
        esM : EnergySystemModel
            The energy system model instance.
        loc : str
            Name of the location.
        commod : str
            Name of the commodity.

        """
        return super().hasOpVariablesForLocationCommodity(esM, loc, commod)

    def getCommodityBalanceContribution(self, pyM, commod, loc, ip, p, t):
        """Get the part-load-aware contribution to a commodity balance.

        Computes the commodity flow as the weighted sum of discretization point
        variables multiplied by both the x-segment (operation level) and y-segment
        (conversion factor) breakpoint values. This replaces the simple
        ``conversionFactor * operation`` product used in the standard Conversion model.

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVarDict = getattr(pyM, "operationVarDict_" + abbrvName)
        discretizationPointConVar = getattr(
            pyM, "discretizationPoint_" + self.abbrvName
        )

        return sum(
            sum(
                discretizationPointConVar[loc, compName, discretStep, ip, p, t]
                * compDict[compName].discretizedPartLoad[commod]["xSegments"][
                    discretStep
                ]
                * compDict[compName].discretizedPartLoad[commod]["ySegments"][
                    discretStep
                ]
                for discretStep in range(compDict[compName].nSegments + 1)
            )
            for compName in opVarDict[ip][loc]
            if commod in compDict[compName].discretizedPartLoad
        )

    def getObjectiveFunctionContribution(self, esM, pyM):
        """Get contribution to the objective function.

        Parameters
        ----------
        esM : EnergySystemModel
            The energy system model instance.
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        return super().getObjectiveFunctionContribution(esM, pyM)

    def setOptimalValues(self, esM, pyM):
        """Extract and store optimal values after optimization.

        Extends the parent method to also extract the optimal values of the
        discretization point, segment continuous, and segment binary variables.

        Parameters
        ----------
        esM : EnergySystemModel
            The energy system model instance.
        pyM : pyomo.ConcreteModel
            Pyomo ConcreteModel storing the mathematical formulation.

        """
        super().setOptimalValues(esM, pyM)
        abbrvName = self.abbrvName
        discretizationPointVariables = getattr(pyM, "discretizationPoint_" + abbrvName)
        discretizationSegmentConVariables = getattr(
            pyM, "discretizationSegmentCon_" + abbrvName
        )
        discretizationSegmentBinVariables = getattr(
            pyM, "discretizationSegmentBin_" + abbrvName
        )

        for ip in esM.investmentPeriods:
            discretizationPointVariablesOptVal_ = utils.formatOptimizationOutput(
                discretizationPointVariables.get_values(),
                "operationVariables",
                "1dim",
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )
            discretizationSegmentConVariablesOptVal_ = utils.formatOptimizationOutput(
                discretizationSegmentConVariables.get_values(),
                "operationVariables",
                "1dim",
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )
            discretizationSegmentBinVariablesOptVal_ = utils.formatOptimizationOutput(
                discretizationSegmentBinVariables.get_values(),
                "operationVariables",
                "1dim",
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )

            self._discretizationPointVariablesOptimum[esM.investmentPeriodNames[ip]] = (
                discretizationPointVariablesOptVal_
            )
            self._discretizationSegmentConVariablesOptimum[
                esM.investmentPeriodNames[ip]
            ] = discretizationSegmentConVariablesOptVal_
            self._discretizationSegmentBinVariablesOptimum[
                esM.investmentPeriodNames[ip]
            ] = discretizationSegmentBinVariablesOptVal_

    def getOptimalValues(self, name="all", ip=0):
        """Return optimal values of the components.

        In addition to the standard capacity, isBuilt, and operation variables, this
        method can return the discretization point, segment continuous, and segment
        binary variable optima.

        Parameters
        ----------
        name : str, optional
            Name of the variable group to return. Options:

            - ``'capacityVariablesOptimum'``
            - ``'isBuiltVariablesOptimum'``
            - ``'operationVariablesOptimum'``
            - ``'discretizationPointVariablesOptimum'``
            - ``'discretizationSegmentConVariablesOptimum'``
            - ``'discretizationSegmentBinVariablesOptimum'``
            - ``'all'`` (default): returns all of the above.

        ip : int, optional
            Investment period index. Default is 0.

        Returns
        -------
        dict
            Dictionary with keys ``'values'``, ``'timeDependent'``, and ``'dimension'``.
            If ``name='all'``, returns a nested dict keyed by variable group name.

        """
        # return super().getOptimalValues(name)

        timeDependentMapping = {
            "capacityVariablesOptimum": False,
            "isBuiltVariablesOptimum": False,
            "operationVariablesOptimum": True,
            "discretizationPointVariablesOptimum": True,
            "discretizationSegmentConVariablesOptimum": True,
            "discretizationSegmentBinVariablesOptimum": True,
        }

        if name in timeDependentMapping:
            return {
                "values": getattr(self, f"_{name}")[ip],
                "timeDependent": timeDependentMapping[name],
                "dimension": self.dimension,
            }
        return {
            valName: {
                "values": getattr(self, f"_{valName}")[ip],
                "timeDependent": timeDependentMapping[valName],
                "dimension": self.dimension,
            }
            for valName in timeDependentMapping
        }
