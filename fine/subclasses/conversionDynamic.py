from fine.conversion import Conversion, ConversionModel
from fine.enums import ComponentAbbreviation, Dimension
from fine import utils
import pyomo.environ as pyomo


class ConversionDynamic(Conversion):
    """Extension of the conversion class with more specific ramping behavior."""

    def __init__(
        self,
        esM,
        name,
        physicalUnit,
        commodityConversionFactors,
        downTimeMin=None,
        upTimeMin=None,
        useTemporalCyclicConstraints=True,
        minimumDowntimeRequired=False,
        **kwargs,
    ):
        r"""Create a ConversionDynamic class instance.
        The ConversionDynamic component specific input arguments are described below. The Conversion
        specific input arguments are described in the Conversion class and the general component
        input arguments are described in the Component class.

        **Default arguments:**

        :param downTimeMin: if specified, indicates minimal down time of the component [hours].
            |br| * the default value is None
        :type downTimeMin: None or integer value in range [0,numberOfTimeSteps*hoursPerTimeStep]

        :param upTimeMin: if specified, indicates minimal up time of the component [hours].
            |br| * the default value is None
        :type upTimeMin: None or integer value in range [0,numberOfTimeSteps*hoursPerTimeStep]

        :param useTemporalCyclicConstraints: If True, the temporal cyclic constraints are used.
            This means that the operation of the first time steps are mathematically linked to the operation of the last time steps.
            |br| * the default value is True
        :type useTemporalCyclicConstraints: boolean

        :param minimumDowntimeRequired: If True, the component is required to be
            offline for at least ``downTimeMin`` hours in every eligible location
            and investment period. Any qualifying downtime can satisfy this
            requirement, irrespective of its cause. This option currently only
            supports full-resolution, unsegmented time series and temporal cyclic
            constraints.
            |br| * the default value is False
        :type minimumDowntimeRequired: boolean

        :param **kwargs: All other keyword arguments of the conversion class can be defined as well.
        :type **kwargs: Check Conversion Class documentation.
        """
        Conversion.__init__(
            self, esM, name, physicalUnit, commodityConversionFactors, **kwargs
        )

        self.modelingClass = ConversionDynamicModel
        self.downTimeMin = downTimeMin
        self.upTimeMin = upTimeMin
        self.useTemporalCyclicConstraints = useTemporalCyclicConstraints
        self.minimumDowntimeRequired = minimumDowntimeRequired
        utils.checkConversionDynamicSpecficDesignInputParams(self, esM)

        if self.isCommisDepending:
            raise ValueError(
                "Currently commissioning-depending constraints are not possible"
            )

    def setTimeSeriesData(self, hasTSA):
        """Set the maximum operation rate and fixed operation rate depending on whether a time series
        analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        super().setTimeSeriesData(hasTSA)


class ConversionDynamicModel(ConversionModel):
    """A ConversionDynamicModel class instance will be instantly created if a ConversionDynamic class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the ConversionDynamic
    class instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The ConversionDynamicModel class inherits from the ConversionModel class.
    """

    def __init__(self):
        super().__init__()
        self.abbrvName = ComponentAbbreviation.CONVERSION_DYNAMIC
        self.dimension = Dimension.ONE
        self._operationVariablesOptimum = {}

    def declareSets(self, esM, pyM):
        """Declare sets and dictionaries: design variable sets, operation variable set, operation mode sets and
        linked components dictionary.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        super().declareSets(esM, pyM)

        maintenanceComponents = [
            name
            for name, component in self.componentsDict.items()
            if component.minimumDowntimeRequired
        ]
        if maintenanceComponents and pyM.hasTSA:
            raise ValueError(
                "minimumDowntimeRequired currently does not support time series "
                "aggregation or segmentation."
            )

        operationVarSet = getattr(pyM, "operationVarSet_" + self.abbrvName)
        maintenanceSet = [
            (loc, compName, ip)
            for loc, compName, ip in operationVarSet
            if compName in maintenanceComponents
        ]
        if maintenanceSet:
            setattr(
                pyM,
                "minimumDowntimeRequiredSet_" + self.abbrvName,
                pyomo.Set(dimen=3, initialize=maintenanceSet),
            )
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
        """Declare design and operation variables.

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
        """Define minimum up and down time constraints.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param timeType: Type of time constraint to set up. Can be either "upTimeMin" or "downTimeMin"
            |br| * the default value is None.
        """
        if timeType not in ["upTimeMin", "downTimeMin"]:
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
            if t == 0:
                return (
                    opVarBin[loc, compName, ip, p, t]
                    - opVarBin[loc, compName, ip, p, numberOfTimeSteps - 1]
                    - opVarStartBin[loc, compName, ip, p, t]
                    + opVarStopBin[loc, compName, ip, p, t]
                    == 0
                )
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
                return opVarBin[loc, compName, ip, p, t] <= 1 - pyomo.quicksum(
                    opVarStopBin[loc, compName, ip, p, t_down] for t_down in range(0, t)
                ) - pyomo.quicksum(
                    opVarStopBin[loc, compName, ip, p, t_down]
                    for t_down in range(fromTimeStepPrevious, toTimeStepPrevious)
                )
            if t >= timeMinTimeSteps:  # upTimeMin
                return opVarBin[loc, compName, ip, p, t] >= pyomo.quicksum(
                    opVarStartBin[loc, compName, ip, p, t_up]
                    for t_up in range(fromTimeStep, toTimeStep)
                )
            return opVarBin[loc, compName, ip, p, t] >= pyomo.quicksum(
                opVarStartBin[loc, compName, ip, p, t_up] for t_up in range(0, t)
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
        """Declare time independent and dependent constraints.

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
        self.minimumDowntimeRequiredConstraint(pyM, esM)

    def minimumDowntimeRequiredConstraint(self, pyM, esM):
        """Require a minimum amount of offline time for selected components."""
        setName = "minimumDowntimeRequiredSet_" + self.abbrvName
        if not hasattr(pyM, setName):
            return

        opVarBin = getattr(pyM, "op_bin_" + self.abbrvName)
        maintenanceSet = getattr(pyM, setName)

        def minimumDowntimeRequired(pyM, loc, compName, ip):
            return (
                pyomo.quicksum(
                    esM.hoursPerTimeStep * (1 - opVarBin[loc, compName, ip, p, t])
                    for p, t in pyM.intraYearTimeSet
                )
                >= self.componentsDict[compName].downTimeMin
            )

        setattr(
            pyM,
            "ConstrMinimumDowntimeRequired_" + self.abbrvName,
            pyomo.Constraint(maintenanceSet, rule=minimumDowntimeRequired),
        )
