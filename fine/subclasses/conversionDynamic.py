from fine.conversion import Conversion, ConversionModel
from fine import utils
import pyomo.environ as pyomo
import pandas as pd

import warnings

# ruff: noqa


class ConversionDynamic(Conversion):
    """
    Extension of the conversion class with more specific ramping behavior
    """

    def __init__(
        self,
        esM,
        name,
        physicalUnit,
        commodityConversionFactors,
        downTimeMin=None,
        upTimeMin=None,
        useTemporalCyclicConstraints=True,
        **kwargs,
    ):
        """
        Constructor for creating a ConversionDynamic class instance.
        The ConversionDynamic component specific input arguments are described below. The Conversion
        specific input arguments are described in the Conversion class and the general component
        input arguments are described in the Component class.

        **Default arguments:**

        :param downTimeMin: if specified, indicates minimal down time of the component [hours].
            |br| * the default value is None
        :type downTimeMin: None or integer value in range \]0,numberOfTimeSteps*hoursPerTimeStep\]

        :param upTimeMin: if specified, indicates minimal up time of the component [hours].
            |br| * the default value is None
        :type upTimeMin: None or integer value in range \[0,numberOfTimeSteps*hoursPerTimeStep\]

        :param rampUpMax: A maximum ramping rate to limit the increase in the operation of the component as share of the installed capacity.
            The maximum ramping is defined per hour and not per hoursPerTimeStep.
            |br| * the default value is None
        :type rampUpMax: None or float value in range \]0.0,1.0\]

        :param rampDownMax: A maximum ramping rate to limit the decrease in the operation of the component as share of the installed capacity.
            The maximum ramping is defined per hour and not per hoursPerTimeStep.
            |br| * the default value is None
        :type rampDownMax: None or float value in range \]0.0,1.0\]

        :param useTemporalCyclicConstraints: If True, the temporal cyclic constraints are used.
            This means that the operation of the first time steps are mathematically linked to the operation of the last time steps.
            |br| * the default value is True
        :type useTemporalCyclicConstraints: boolean

        :param \*\*kwargs: All other keyword arguments of the conversion class can be defined as well.
        :type \*\*kwargs: Check Conversion Class documentation.
        """
        Conversion.__init__(
            self, esM, name, physicalUnit, commodityConversionFactors, **kwargs
        )

        self.modelingClass = ConversionDynamicModel
        self.downTimeMin = downTimeMin
        self.upTimeMin = upTimeMin
        self.useTemporalCyclicConstraints = useTemporalCyclicConstraints
        utils.checkConversionDynamicSpecficDesignInputParams(self, esM)

        if self.isCommisDepending:
            raise ValueError(
                "Currently commissioning-depending constraints are not possible"
            )

    def setTimeSeriesData(self, hasTSA):
        """
        Function for setting the maximum operation rate and fixed operation rate depending on whether a time series
        analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        super().setTimeSeriesData(hasTSA)
        if hasTSA:
            if any(
                x is not None
                for x in [
                    self.upTimeMin,
                    self.downTimeMin,
                ]
            ):
                raise ValueError(
                    "Time series aggregation is not supported for conversion dynamic components."
                )
            # Information for refactoring:
            # Time series aggregation is currently not supported for conversion dynamic class.
            # The current constraints link the last time step / segment to the first time step / segment of the same TSA period.
            # The order of periods over a year is not considered.
            # For upTimeMin and downTimeMin, the constraint must consider the length of the time step / segments in future.


class ConversionDynamicModel(ConversionModel):
    """
    A ConversionDynamicModel class instance will be instantly created if a ConversionDynamic class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the ConversionDynamic
    class instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The ConversionDynamicModel class inherits from the ConversionModel class.
    """

    def __init__(self):
        super().__init__()
        self.abbrvName = "conv_dyn"
        self.dimension = "1dim"
        self._operationVariablesOptimum = {}

    def declareSets(self, esM, pyM):
        """
        Declare sets and dictionaries: design variable sets, operation variable set, operation mode sets and
        linked components dictionary.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        super().declareSets(esM, pyM)
        allBinaryParameters = [
            "partLoadMin",
            "downTimeMin",
            "upTimeMin",
        ]
        self.declareBinOpVarSet(
            esM,
            pyM,
            binaryOperationParameter=allBinaryParameters,
            binaryOperationSetName="operationBinVarSet",
        )
        self.declareBinOpVarSet(
            esM,
            pyM,
            binaryOperationParameter=["downTimeMin"],
            binaryOperationSetName="opConstrSet_downTimeMin",
        )
        self.declareBinOpVarSet(
            esM,
            pyM,
            binaryOperationParameter=["upTimeMin"],
            binaryOperationSetName="opConstrSet_upTimeMin",
        )

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary, relevanceThreshold):
        """
        Declare design and operation variables

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param relaxIsBuiltBinary: states if the optimization problem should be solved as a relaxed LP to get the lower
            bound of the problem.
            |br| * the default value is False
        :type declaresOptimizationProblem: boolean

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None
        """
        super().declareVariables(esM, pyM, relaxIsBuiltBinary, relevanceThreshold)

        hasTemporalRestrictions = any(
            x
            for x in self.componentsDict
            if esM.getComponent(x).upTimeMin is not None
            or esM.getComponent(x).downTimeMin is not None
        )

        if hasTemporalRestrictions:
            self.declareOperationBinaryVars(
                pyM, opVarBinName="startVariable", opBinSetName="operationVarSet"
            )
            self.declareOperationBinaryVars(
                pyM, opVarBinName="stopVariable", opBinSetName="operationVarSet"
            )

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def minimumTimeConstraints(self, pyM, esM, timeType):
        """
        Internal function to handle both minimum up and down time constraints.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param timeType: Type of time constraint to set up. Can be either "upTimeMin" or "downTimeMin"
            |br| * the default value is None.
        """
        if not timeType in ["upTimeMin", "downTimeMin"]:
            raise ValueError(
                f"Time type {timeType} is not valid. Please choose between upTimeMin and downTimeMin."
            )

        compDict, abbrvName = self.componentsDict, self.abbrvName

        # first check if the parameter and therefore the set is defined
        if not hasattr(pyM, f"opConstrSet_{timeType}_" + abbrvName):
            return

        # if set exists, set up the constraint
        opVarBin = getattr(pyM, "op_bin_" + abbrvName)
        opVarStartBin = getattr(pyM, "startVariable_" + abbrvName)
        opVarStopBin = getattr(pyM, "stopVariable_" + abbrvName)
        constrSetMinTime = getattr(pyM, f"opConstrSet_{timeType}_" + abbrvName)

        if not pyM.hasSegmentation:
            numberOfTimeSteps = len(esM.timeStepsPerPeriod)
        else:
            numberOfTimeSteps = len(esM.segmentsPerPeriod)

        def minimumTime1(pyM, loc, compName, ip, p, t):
            isCyclic = getattr(compDict[compName], "useTemporalCyclicConstraints")
            if t == 0 and not isCyclic:
                return pyomo.Constraint.Skip
            elif t == 0:
                return (
                    opVarBin[loc, compName, ip, p, t]
                    - opVarBin[loc, compName, ip, p, numberOfTimeSteps - 1]
                    - opVarStartBin[loc, compName, ip, p, t]
                    + opVarStopBin[loc, compName, ip, p, t]
                    == 0
                )
            else:
                return (
                    opVarBin[loc, compName, ip, p, t]
                    - opVarBin[loc, compName, ip, p, t - 1]
                    - opVarStartBin[loc, compName, ip, p, t]
                    + opVarStopBin[loc, compName, ip, p, t]
                    == 0
                )

        setattr(
            pyM,
            f"Constr{timeType}1_{abbrvName}",
            pyomo.Constraint(constrSetMinTime, pyM.intraYearTimeSet, rule=minimumTime1),
        )

        def minimumTime2(pyM, loc, compName, ip, p, t):
            # check if timeType is multiple of hoursPerTimeStep
            if getattr(compDict[compName], timeType) % esM.hoursPerTimeStep != 0:
                raise ValueError(
                    f"Time type {timeType} is not a multiple of hoursPerTimeStep."
                )

            timeMinTimeSteps = int(
                getattr(compDict[compName], timeType) / esM.hoursPerTimeStep
            )
            isCyclic = getattr(compDict[compName], "useTemporalCyclicConstraints")
            fromTimeStep = t - timeMinTimeSteps + 1
            toTimeStep = t
            # when cyclic -> previous time horizon
            fromTimeStepPrevious = numberOfTimeSteps - (timeMinTimeSteps - t)
            toTimeStepPrevious = numberOfTimeSteps

            if t < timeMinTimeSteps and not isCyclic:
                return pyomo.Constraint.Skip

            if timeType == "downTimeMin":
                if t >= timeMinTimeSteps:
                    return opVarBin[loc, compName, ip, p, t] <= 1 - pyomo.quicksum(
                        opVarStopBin[loc, compName, ip, p, t_down]
                        for t_down in range(fromTimeStep, toTimeStep)
                    )
                else:
                    return opVarBin[loc, compName, ip, p, t] <= 1 - pyomo.quicksum(
                        opVarStopBin[loc, compName, ip, p, t_down]
                        for t_down in range(0, t)
                    ) - pyomo.quicksum(
                        opVarStopBin[loc, compName, ip, p, t_down]
                        for t_down in range(fromTimeStepPrevious, toTimeStepPrevious)
                    )
            else:  # upTimeMin
                if t >= timeMinTimeSteps:
                    return opVarBin[loc, compName, ip, p, t] >= pyomo.quicksum(
                        opVarStartBin[loc, compName, ip, p, t_up]
                        for t_up in range(fromTimeStep, toTimeStep)
                    )
                else:
                    return opVarBin[loc, compName, ip, p, t] >= pyomo.quicksum(
                        opVarStartBin[loc, compName, ip, p, t_up]
                        for t_up in range(0, t)
                    ) + pyomo.quicksum(
                        opVarStartBin[loc, compName, ip, p, t_up]
                        for t_up in range(fromTimeStepPrevious, toTimeStepPrevious)
                    )

        setattr(
            pyM,
            f"Constr{timeType}2_{abbrvName}",
            pyomo.Constraint(constrSetMinTime, pyM.intraYearTimeSet, rule=minimumTime2),
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

        self.binaryOperation(
            pyM,
            "ConstrOperation",
            "operationBinVarSet",
            "",
            "op",
            "op_bin",
            isOperationCommisYearDepending=False,
        )

        ################################################################################################################
        #                                         Dynamic Constraints                                                  #
        ################################################################################################################
        self.minimumTimeConstraints(pyM, esM, timeType="downTimeMin")
        self.minimumTimeConstraints(pyM, esM, timeType="upTimeMin")
