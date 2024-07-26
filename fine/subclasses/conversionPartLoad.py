from fine.conversion import Conversion, ConversionModel
from fine.utils import checkDataFrameConversionFactor, checkCallableConversionFactor
from fine import utils
import pyomo.environ as pyomo
import pandas as pd
import numpy as np
import pwlf

# ruff: noqa

def pieceWiseLinearization(functionOrRaw, xLowerBound, xUpperBound, nSegments):
    """
    Determine xSegments, ySegments.
    If nSegments is not specified by the user it is either set (e.g. nSegments=5) or nSegements is determined by
    a bayesian optimization algorithm.
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

        R2values[i] = 1.0 - (ssr / sst)

    return {
        "xSegments": xSegments,
        "ySegments": ySegments,
        "nSegments": nSegments,
        "Rsquared": Rsquared,
        "R2values": R2values,
    }


def getDiscretizedPartLoad(commodityConversionFactorsPartLoad, nSegments):
    """Preprocess the conversion factors passed by the user"""
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
        elif conversionFactor == 1 or conversionFactor == -1:
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
    """Check if the discretized points are >=0 and <=100%"""

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
                else:
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
            else:
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
            else:
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
    """
    Check if one of the commodity conversion factors equals 1 and another is either a lambda function or a set of data points.
    Additionally check if the conversion factor that depicts part load behavior
        (1) covers part loads from 0 to 1 and
        (2) includes only conversion factors greater than 0 in the relevant part load range.
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
        elif conversionFactor == 1 or conversionFactor == -1:
            nonPartLoadCommodPresent = True

    if nonPartLoadCommodPresent == False:
        raise TypeError("One conversion factor needs to be either 1 or -1.")
    if partLoadCommodPresent == False:
        raise TypeError(
            "One conversion factor needs to be either a callable function or a list of two-dimensional data points."
        )


class ConversionPartLoad(Conversion):
    """
    A ConversionPartLoad component maps the (nonlinear) part-load behavior of a Conversion component.
    It uses the open source module PWLF to generate piecewise linear functions upon a continuous function or
    discrete data points.
    The formulation of the optimization is done by using special ordered sets (SOS) constraints.
    When using ConversionPartLoad it is recommended to check the piecewise linearization
    visually to verify that the accuracy meets the desired requirements.
    The ConversionPartLoad class inherits from the Conversion class.
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
        """
        Constructor for creating an ConversionPartLoad class instance. Capacities are given in the physical unit
        of the plants.
        The ConversionPartLoad component specific input arguments are described below.
        Other specific input arguments are described in the Conversion class
        and the general component input arguments are described in the Component class.

        **Required arguments:**

        :param commodityConversionFactorsPartLoad: A dictionary containing key-value pairs, where each key represents
        a commodity (e.g., "electricity", "hydrogen") and each value provides the conversion factors that vary with
        the operation load. These conversion factors dictate the efficiency or rate at which one commodity is transformed
        into another under different operational conditions. The (nonlinear) part load behavior, which is the relationship
        between the conversion factors (or efficiency) and the operational load, can be described either using a lambda function
        for a direct mathematical relationship or a Pandas DataFrame. If a Pandas DataFrame is used, it should contain two columns:
        one for the x-axis, which represents the operation level (nominal load), and one for the y-axis, which represents the
        corresponding conversion factor (efficiency) at the corresponding operation level. A negative value indicates that the
        commodity is consumed. A positive value indicates that the commodity is produced.

            Example:
                * An electrolyzer converts, simply put, electricity into hydrogen with an electrical efficiency
                    depending on the operation level. The physicalUnit is given as GW_electric, the unit for the 'electricity'
                    commodity isgiven in GW_electric and the 'hydrogen' commodity is given in GW_hydrogen_lowerHeatingValue.
                    Here, electricity consumption is represented by a negative value (-1), and hydrogen production efficiency
                    is detailed in a DataFrame with operation levels and corresponding efficiencies.

                    # Efficiency Curve of Electrolyzer
                    Operation_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                    Efficiency = [0.1, 0.15, 0.5, 0.7, 0.7, 0.65, 0.63, 0.62, 0.61, 0.60]
                    d = {"x": Operation_level, "y": Efficiency}
                    partLoadData = pd.DataFrame(d)

                    # Definition of commodityConversionFactorsPartLoad
                    -> the commodityConversionFactorsPartLoad are defined as {'electricity':-1,'hydrogen':partLoadData}.

        **Default arguments:**

        :param nSegments: Number of line segments used for piecewise linearization and generation of point variable (nSegment+1) and
            segment (nSegment) variable sets.
            By default, the nSegments is None. For this case, the number of line segments is set to 5.
            The user can set nSegments by choosing an integer (>=0). It is recommended to choose values between 3 and 7 since
            the computational cost rises dramatically with increasing nSegments.
            When specifying nSegements='optimizeSegmentNumbers', an optimal number of line segments is automatically chosen by a
            bayesian optimization algorithm.
            |br| * the default value is None
        :type nSegments: None or integer or string

        :param **kwargs: All other keyword arguments of the conversion class can be defined as well.
        :type **kwargs:
            * Check Conversion Class documentation.
        """

        Conversion.__init__(
            self, esM, name, physicalUnit, commodityConversionFactors, **kwargs
        )

        self.modelingClass = ConversionPartLoadModel

        # TODO: Make compatible with conversion
        utils.checkNumberOfConversionFactors(commodityConversionFactors)

        if type(commodityConversionFactorsPartLoad) == dict:
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

        elif type(commodityConversionFactorsPartLoad) == tuple:
            utils.checkNumberOfConversionFactors(
                commodityConversionFactorsPartLoad[0].keys()
            )
            self.discretizedPartLoad = commodityConversionFactorsPartLoad[0]
            self.nSegments = commodityConversionFactorsPartLoad[1]


class ConversionPartLoadModel(ConversionModel):
    """
    A ConversionPartLoad class instance will be instantly created if a ConversionPartLoad class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Conversion class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The ConversionPartLoad class inherits from the ConversionModel class.
    """

    def __init__(self):
        super().__init__()
        self.abbrvName = "partLoad"
        self.dimension = "1dim"
        self._operationVariablesOptimum = {}
        self.discretizationPointVariablesOptimun = {}
        self.discretizationSegmentConVariablesOptimun = {}
        self.discretizationSegmentBinVariablesOptimun = {}

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def initDiscretizationPointVarSet(self, pyM):
        """
        Declare discretization variable set of type 1 in the pyomo object for for each node.
        Type 1 represents every start, end, and intermediate point in the piecewise linear function.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Declare discretization variable set of type 2 in the pyomo object for for each node.
        Type 2 represents every segment in the piecewise linear function.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Declare sets and dictionaries: design variable sets, operation variable sets, operation mode sets and
        linked components dictionary.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        super().declareSets(esM, pyM)

        # Declare operation variable sets
        self.initDiscretizationPointVarSet(pyM)
        self.initDiscretizationSegmentVarSet(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareDiscretizationPointVariables(self, pyM):
        """
        Declare discretization point variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Declare discretization segment variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Declare discretization segment variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Declare design and operation variables.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model

        :param relaxIsBuiltBinary: states if the optimization problem should be solved as a relaxed LP to get the lower
            bound of the problem.
            |br| * the default value is False
        :type declaresOptimizationProblem: boolean

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None
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
        """
        Ensure that the binary segment variables are in sum equal to 1.
        Enforce that only one binary is set to 1, while all other are fixed 0.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Ensure that the continuous segment variables are zero if the respective binary variable is zero and unlimited otherwise.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Ensure that the continuous segment variables are in sum equal to the installed capacity of the component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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

            def segmentCapacityConstraint(pyM, loc, compName, p, t):
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
        """
        Ensure that the continuous point variables are in sum equal to the installed capacity of the component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Ensure that only two consecutive point variables are non-zero while all other point variables are fixed to zero.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
            elif discretStep == points[-1]:
                return (
                    discretizationPointConVar[loc, compName, points[-1], ip, p, t]
                    <= discretizationSegmentConVar[
                        loc, compName, segments[-1], ip, p, t
                    ]
                )
            else:
                return (
                    discretizationPointConVar[loc, compName, discretStep, ip, p, t]
                    <= discretizationSegmentConVar[
                        loc, compName, discretStep - 1, ip, p, t
                    ]
                    + discretizationSegmentConVar[loc, compName, discretStep, ip, p, t]
                )

        setattr(
            pyM,
            "ConstrPointSOS2_" + abbrvName,
            pyomo.Constraint(discretizationPointVarSet, pyM.timeSet, rule=pointSOS2),
        )

    def partLoadOperationOutput(self, pyM):
        """
        Set the required input of a conversion process dependent on the part load efficency.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Declare time independent and dependent constraints.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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
        """
        Check if the commodity´s transfer between a given location and the other locations of the energy system model
        is eligible.

        :param esM: EnergySystemModel in which the LinearOptimalPowerFlow components have been added to.
        :type esM: esM - EnergySystemModel class instance

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: Name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """
        return super().hasOpVariablesForLocationCommodity(esM, loc, commod)

    def getCommodityBalanceContribution(self, pyM, commod, loc, ip, p, t):
        """Get contribution to a commodity balance."""
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
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        return super().getObjectiveFunctionContribution(esM, pyM)

    def setOptimalValues(self, esM, pyM):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
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

            self.discretizationPointVariablesOptimun[esM.investmentPeriodNames[ip]] = (
                discretizationPointVariablesOptVal_
            )
            self.discretizationSegmentConVariablesOptimun[
                esM.investmentPeriodNames[ip]
            ] = discretizationSegmentConVariablesOptVal_
            self.discretizationSegmentBinVariablesOptimun[
                esM.investmentPeriodNames[ip]
            ] = discretizationSegmentBinVariablesOptVal_

    def getOptimalValues(self, name="all", ip=0):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariables',
            * 'isBuiltVariables',
            * '_operationVariablesOptimum',
            * 'all' or another input: all variables are returned.

        |br| * the default value is 'all'
        :type name: string

        :param ip: investment period
        |br| * the default value is 0
        :type ip: int

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        # return super().getOptimalValues(name)
        if name == "capacityVariablesOptimum":
            return {
                "values": self._capacityVariablesOptimum[ip],
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "isBuiltVariablesOptimum":
            return {
                "values": self._isBuiltVariablesOptimum[ip],
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "operationVariablesOptimum":
            return {
                "values": self._operationVariablesOptimum[ip],
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "discretizationPointVariablesOptimun":
            return {
                "values": self._discretizationPointVariablesOptimun[ip],
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "discretizationSegmentConVariablesOptimun":
            return {
                "values": self._discretizationSegmentConVariablesOptimun[ip],
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "discretizationSegmentBinVariablesOptimun":
            return {
                "values": self._discretizationSegmentBinVariablesOptimun[ip],
                "timeDependent": True,
                "dimension": self.dimension,
            }
        else:
            return {
                "capacityVariablesOptimum": {
                    "values": self._capacityVariablesOptimum[ip],
                    "timeDependent": False,
                    "dimension": self.dimension,
                },
                "isBuiltVariablesOptimum": {
                    "values": self._isBuiltVariablesOptimum[ip],
                    "timeDependent": False,
                    "dimension": self.dimension,
                },
                "operationVariablesOptimum": {
                    "values": self._operationVariablesOptimum[ip],
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "discretizationPointVariablesOptimun": {
                    "values": self._discretizationPointVariablesOptimun[ip],
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "discretizationSegmentConVariablesOptimun": {
                    "values": self._discretizationSegmentConVariablesOptimun[ip],
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "discretizationSegmentBinVariablesOptimun": {
                    "values": self._discretizationSegmentBinVariablesOptimun[ip],
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
            }
