from fine.conversion import Conversion, ConversionModel
from fine.enums import ComponentAbbreviation, Dimension, VarType
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
        maintenanceTime=None,
        maintenanceOccurrences=None,
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

        :param maintenanceTime: Minimum duration of each scheduled maintenance
            window [hours]. The value can be specified uniformly, by location,
            by investment period, or by investment period and location. It must
            be specified together with ``maintenanceOccurrences``.
            |br| * the default value is None
        :type maintenanceTime: None, positive number, Pandas Series, or dict

        :param maintenanceOccurrences: Exact number of distinct maintenance
            windows required in each investment period. The value can be
            specified uniformly, by location, by investment period, or by
            investment period and location. Consecutive windows are separated
            by at least one non-maintenance time step. Maintenance is only
            scheduled when positive capacity is installed. Time series
            aggregation and segmentation are currently unsupported.
            |br| * the default value is None
        :type maintenanceOccurrences: None, non-negative integer, Pandas Series,
            or dict

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
        self.maintenanceTime = maintenanceTime
        self.maintenanceOccurrences = maintenanceOccurrences
        (
            self.processedMaintenanceTime,
            self.processedMaintenanceOccurrences,
        ) = utils.checkAndSetMaintenanceParameters(self, esM)
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
        self._maintenanceActiveVariablesOptimum = {}

    def declareMinimumDowntimeRequiredSet(self, esM, pyM):
        """Declare location-component-IP tuples subject to required downtime.

        The set contains every eligible ``(location, component, investmentPeriod)``
        tuple for which ``minimumDowntimeRequired`` is enabled. It indexes the
        aggregate minimum-downtime constraint.

        :param esM: EnergySystemModel containing the modeled temporal structure.
        :type esM: EnergySystemModel

        :param pyM: Pyomo model to which the set is attached.
        :type pyM: pyomo.ConcreteModel
        """
        components = [
            name
            for name, component in self.componentsDict.items()
            if component.minimumDowntimeRequired
        ]
        if components and pyM.hasTSA:
            raise ValueError(
                "minimumDowntimeRequired currently does not support time series "
                "aggregation or segmentation."
            )
        operationVarSet = getattr(pyM, "operationVarSet_" + self.abbrvName)
        maintenanceSet = [index for index in operationVarSet if index[1] in components]
        if maintenanceSet:
            setattr(
                pyM,
                "minimumDowntimeRequiredSet_" + self.abbrvName,
                pyomo.Set(dimen=3, initialize=maintenanceSet),
            )

    def declareScheduledMaintenanceSet(self, esM, pyM):
        """Declare location-component-IP tuples requiring maintenance.

        ``scheduledMaintenanceSet`` contains eligible location-component-IP tuples
        with a positive maintenance occurrence count.

        :param esM: EnergySystemModel containing the modeled temporal structure.
        :type esM: EnergySystemModel

        :param pyM: Pyomo model to which the sets are attached.
        :type pyM: pyomo.ConcreteModel
        """
        components = [
            name
            for name, component in self.componentsDict.items()
            if component.maintenanceTime is not None
        ]
        if components and pyM.hasTSA:
            raise ValueError(
                "maintenanceTime and maintenanceOccurrences currently do not "
                "support time series aggregation or segmentation."
            )
        operationVarSet = getattr(pyM, "operationVarSet_" + self.abbrvName)
        maintenanceSet = [
            (loc, compName, ip)
            for loc, compName, ip in operationVarSet
            if compName in components
            and self.componentsDict[compName].processedMaintenanceOccurrences[ip][loc]
            > 0
        ]
        if not maintenanceSet:
            return
        setattr(
            pyM,
            "scheduledMaintenanceSet_" + self.abbrvName,
            pyomo.Set(dimen=3, initialize=maintenanceSet),
        )

    def declareMaintenanceStartSet(self, esM, pyM):
        """Declare the valid starts of scheduled maintenance windows.

        ``maintenanceStartSet`` extends each scheduled-maintenance tuple with period
        and timestep indices. It includes only starts for which the complete minimum
        maintenance duration fits inside the modeled horizon.

        :param esM: EnergySystemModel containing the modeled temporal structure.
        :type esM: EnergySystemModel

        :param pyM: Pyomo model to which the set is attached.
        :type pyM: pyomo.ConcreteModel
        """
        setName = "scheduledMaintenanceSet_" + self.abbrvName
        if not hasattr(pyM, setName):
            return
        maintenanceSet = getattr(pyM, setName)
        startSet = []
        for loc, compName, ip in maintenanceSet:
            duration = int(
                self.componentsDict[compName].processedMaintenanceTime[ip][loc]
                / esM.hoursPerTimeStep
            )
            startSet.extend(
                (loc, compName, ip, p, t)
                for p, t in pyM.intraYearTimeSet
                if t + duration <= esM.numberOfTimeSteps
            )
        setattr(
            pyM,
            "maintenanceStartSet_" + self.abbrvName,
            pyomo.Set(dimen=5, initialize=startSet),
        )

    def declareSets(self, esM, pyM):
        """Declare sets and dictionaries: design variable sets, operation variable set, operation mode sets and
        linked components dictionary.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        super().declareSets(esM, pyM)
        self.declareMinimumDowntimeRequiredSet(esM, pyM)
        self.declareScheduledMaintenanceSet(esM, pyM)
        self.declareMaintenanceStartSet(esM, pyM)
        allBinaryParameters = [
            "partLoadMin",
            "downTimeMin",
            "upTimeMin",
            "maintenanceTime",
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

    def declareMaintenanceStartVariables(self, pyM):
        r"""Declare binary variables marking the start of maintenance windows.

        .. math::
            mStart_{loc,comp,ip,p,t} \in \{0,1\}

        A value of one means that a new scheduled maintenance window starts at
        timestep ``t``. Variables exist only for starts whose minimum duration fits
        completely within the horizon.

        :param pyM: Pyomo model to which the variable is attached.
        :type pyM: pyomo.ConcreteModel
        """
        if hasattr(pyM, "maintenanceStartSet_" + self.abbrvName):
            setattr(
                pyM,
                "maintenanceStart_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "maintenanceStartSet_" + self.abbrvName),
                    domain=pyomo.Binary,
                ),
            )

    def declareMaintenanceInstalledVariables(self, pyM):
        r"""Declare binary variables indicating positive installed capacity.

        .. math::
            mInstalled_{loc,comp,ip} \in \{0,1\}

        The variable activates the required number of maintenance windows only when
        the corresponding component has positive capacity.

        :param pyM: Pyomo model to which the variable is attached.
        :type pyM: pyomo.ConcreteModel
        """
        setName = "scheduledMaintenanceSet_" + self.abbrvName
        if hasattr(pyM, setName):
            setattr(
                pyM,
                "maintenanceInstalled_" + self.abbrvName,
                pyomo.Var(getattr(pyM, setName), domain=pyomo.Binary),
            )

    def declareMaintenanceActiveVariables(self, pyM):
        r"""Declare binary variables representing active maintenance.

        .. math::
            mActive_{loc,comp,ip,p,t} \in \{0,1\}

        A value of one means that the component is in a scheduled maintenance window
        and must therefore be offline at the corresponding timestep.

        :param pyM: Pyomo model to which the variable is attached.
        :type pyM: pyomo.ConcreteModel
        """
        setName = "scheduledMaintenanceSet_" + self.abbrvName
        if hasattr(pyM, setName):
            setattr(
                pyM,
                "maintenanceActive_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, setName),
                    pyM.intraYearTimeSet,
                    domain=pyomo.Binary,
                ),
            )

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
        self.declareMaintenanceStartVariables(pyM)
        self.declareMaintenanceInstalledVariables(pyM)
        self.declareMaintenanceActiveVariables(pyM)

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
            "ConstrDynamicOperation",
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
        self.scheduledMaintenanceConstraints(pyM, esM)

    def minimumDowntimeRequiredConstraint(self, pyM, esM):
        r"""Require a minimum amount of offline time for selected components.

        .. math::
            \sum_{p,t} \Delta t\,(1-opBin_{loc,comp,ip,p,t})
            \geq downTimeMin_{comp}

        This is the original aggregate downtime requirement. It remains independent
        from the explicit scheduled-maintenance formulation below.

        :param pyM: Pyomo model to which the constraint is attached.
        :type pyM: pyomo.ConcreteModel

        :param esM: EnergySystemModel containing the timestep duration.
        :type esM: EnergySystemModel
        """
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

    def scheduledMaintenanceConstraints(self, pyM, esM):
        r"""Create the independent scheduled-maintenance formulation.

        Installed capacity activates the maintenance requirement:

        .. math::
            cap_{loc,comp,ip} \leq M_{loc,comp,ip}\,mInstalled_{loc,comp,ip}

        .. math::
            cap_{loc,comp,ip} \geq \epsilon\,mInstalled_{loc,comp,ip}

        The exact requested number :math:`N` of starts is enforced by:

        .. math::
            \sum_{p,t}mStart_{loc,comp,ip,p,t}
            = N_{loc,comp,ip}\,mInstalled_{loc,comp,ip}

        A start activates at least :math:`D` consecutive maintenance timesteps:

        .. math::
            \sum_{\tau=t}^{t+D-1}mActive_{loc,comp,ip,p,\tau}
            \geq D\,mStart_{loc,comp,ip,p,t}

        Maintenance disables operation:

        .. math::
            opBin_{loc,comp,ip,p,t} \leq 1-mActive_{loc,comp,ip,p,t}

        Every transition into maintenance requires a start:

        .. math::
            mActive_t-mActive_{t-1} \leq mStart_t

        and a start is forbidden directly after an active timestep:

        .. math::
            mStart_t \leq 1-mActive_{t-1}

        Consequently, counted windows are distinct and separated by at least one
        timestep without scheduled maintenance. Maintenance can remain active beyond
        its minimum duration. Valid start indices ensure that the minimum-duration
        part of a window never crosses the modeled horizon boundary.

        :param pyM: Pyomo model to which the constraints are attached.
        :type pyM: pyomo.ConcreteModel

        :param esM: EnergySystemModel containing temporal and investment-period data.
        :type esM: EnergySystemModel
        """
        setName = "scheduledMaintenanceSet_" + self.abbrvName
        if not hasattr(pyM, setName):
            return

        abbrvName = self.abbrvName
        maintenanceSet = getattr(pyM, setName)
        maintenanceStartSet = getattr(pyM, "maintenanceStartSet_" + abbrvName)
        maintenanceStart = getattr(pyM, "maintenanceStart_" + abbrvName)
        maintenanceActive = getattr(pyM, "maintenanceActive_" + abbrvName)
        maintenanceInstalled = getattr(pyM, "maintenanceInstalled_" + abbrvName)
        opVarBin = getattr(pyM, "op_bin_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)

        def installedUpper(pyM, loc, compName, ip):
            component = self.componentsDict[compName]
            if component.processedCapacityFix[ip] is not None:
                capacityUpper = component.processedCapacityFix[ip][loc]
            elif component.processedCapacityMax[ip] is not None:
                capacityUpper = component.processedCapacityMax[ip][loc]
            else:
                capacityUpper = component.bigM
            return (
                capVar[loc, compName, ip]
                <= capacityUpper * maintenanceInstalled[loc, compName, ip]
            )

        def installedLower(pyM, loc, compName, ip):
            return capVar[loc, compName, ip] >= (
                1e-4 * maintenanceInstalled[loc, compName, ip]
            )

        def occurrenceCount(pyM, loc, compName, ip):
            occurrences = self.componentsDict[compName].processedMaintenanceOccurrences[
                ip
            ][loc]
            return (
                pyomo.quicksum(
                    maintenanceStart[index]
                    for index in maintenanceStartSet
                    if index[:3] == (loc, compName, ip)
                )
                == occurrences * maintenanceInstalled[loc, compName, ip]
            )

        def forceOffline(pyM, loc, compName, ip, p, t):
            return (
                opVarBin[loc, compName, ip, p, t]
                <= 1 - maintenanceActive[loc, compName, ip, p, t]
            )

        def activateMinimumDuration(pyM, loc, compName, ip, p, start):
            duration = int(
                self.componentsDict[compName].processedMaintenanceTime[ip][loc]
                / esM.hoursPerTimeStep
            )
            return (
                pyomo.quicksum(
                    maintenanceActive[loc, compName, ip, p, t]
                    for t in range(start, start + duration)
                )
                >= duration * maintenanceStart[loc, compName, ip, p, start]
            )

        def activeRiseRequiresStart(pyM, loc, compName, ip, p, t):
            start = (
                maintenanceStart[loc, compName, ip, p, t]
                if (loc, compName, ip, p, t) in maintenanceStartSet
                else 0
            )
            if t == 0:
                return maintenanceActive[loc, compName, ip, p, t] <= start
            return (
                maintenanceActive[loc, compName, ip, p, t]
                - maintenanceActive[loc, compName, ip, p, t - 1]
                <= start
            )

        def separateWindows(pyM, loc, compName, ip, p, start):
            if start == 0:
                return pyomo.Constraint.Skip
            return (
                maintenanceStart[loc, compName, ip, p, start]
                <= 1 - (maintenanceActive[loc, compName, ip, p, start - 1])
            )

        setattr(
            pyM,
            "ConstrMaintenanceInstalledUpper_" + abbrvName,
            pyomo.Constraint(maintenanceSet, rule=installedUpper),
        )
        setattr(
            pyM,
            "ConstrMaintenanceInstalledLower_" + abbrvName,
            pyomo.Constraint(maintenanceSet, rule=installedLower),
        )
        setattr(
            pyM,
            "ConstrMaintenanceOccurrences_" + abbrvName,
            pyomo.Constraint(maintenanceSet, rule=occurrenceCount),
        )
        setattr(
            pyM,
            "ConstrMaintenanceOffline_" + abbrvName,
            pyomo.Constraint(maintenanceSet, pyM.intraYearTimeSet, rule=forceOffline),
        )
        setattr(
            pyM,
            "ConstrMaintenanceMinimumDuration_" + abbrvName,
            pyomo.Constraint(maintenanceStartSet, rule=activateMinimumDuration),
        )
        setattr(
            pyM,
            "ConstrMaintenanceActiveRise_" + abbrvName,
            pyomo.Constraint(
                maintenanceSet, pyM.intraYearTimeSet, rule=activeRiseRequiresStart
            ),
        )
        setattr(
            pyM,
            "ConstrMaintenanceSeparation_" + abbrvName,
            pyomo.Constraint(maintenanceStartSet, rule=separateWindows),
        )

    def _extractSubclassRawResults(self, esM, pyM, rawResults):
        """Extract operation and active-maintenance time series after optimization.

        ``maintenanceActiveVariablesOptimum`` has the same component/location/time
        layout as other one-dimensional operation variables and is made available to
        the standard xarray and netCDF result exporters.

        :param esM: Optimized EnergySystemModel instance.
        :type esM: EnergySystemModel

        :param pyM: Solved Pyomo model containing the maintenance variables.
        :type pyM: pyomo.ConcreteModel

        :param rawResults: Result dictionary updated in place for each investment period.
        :type rawResults: dict
        """
        super()._extractSubclassRawResults(esM, pyM, rawResults)
        variableName = "maintenanceActive_" + self.abbrvName
        if not hasattr(pyM, variableName):
            return
        maintenanceActive = getattr(pyM, variableName)
        for ip in esM.investmentPeriods:
            ipName = esM.investmentPeriodNames[ip]
            optVal = utils.formatOptimizationOutput(
                maintenanceActive.get_values(),
                VarType.OPERATION,
                Dimension.ONE,
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )
            self._maintenanceActiveVariablesOptimum[ipName] = optVal
            rawResults[ipName]["maintenanceActive"] = optVal

    def _exportOptimumVarMap(self):
        """Add active maintenance to the standard optimum-result export map.

        :return: Base result mappings extended by the time-dependent,
            one-dimensional active-maintenance result.
        :rtype: list
        """
        return super()._exportOptimumVarMap() + [
            (
                "maintenanceActive",
                "maintenanceActiveVariablesOptimum",
                True,
                Dimension.ONE,
            )
        ]

    def _convertOptimalValueNames(self, esM):
        """Publish maintenance results after the standard result conversion.

        :param esM: EnergySystemModel defining whether results are unwrapped for a
            single investment period or retained as a dictionary.
        :type esM: EnergySystemModel
        """
        super()._convertOptimalValueNames(esM)
        if not self._maintenanceActiveVariablesOptimum:
            return
        if esM.numberOfInvestmentPeriods == 1:
            ipName = esM.investmentPeriodNames[0]
            self.maintenanceActiveVariablesOptimum = (
                self._maintenanceActiveVariablesOptimum[ipName]
            )
        else:
            self.maintenanceActiveVariablesOptimum = (
                self._maintenanceActiveVariablesOptimum
            )

    def getOptimalValues(self, name="all", ip=0):
        """Return standard optima and the scheduled-maintenance status.

        :param name: Requested optimum variable name, or ``"all"``.
        :type name: str

        :param ip: Investment-period name used as the internal result key.
        :type ip: int or str

        :return: Result metadata and values in the standard FINE export layout.
        :rtype: dict
        """
        maintenanceName = "maintenanceActiveVariablesOptimum"
        if name == maintenanceName:
            return {
                "values": self._maintenanceActiveVariablesOptimum[ip],
                "timeDependent": True,
                "dimension": Dimension.ONE,
            }
        values = super().getOptimalValues(name, ip=ip)
        if name not in ("all", maintenanceName):
            return values
        if ip in self._maintenanceActiveVariablesOptimum:
            values[maintenanceName] = {
                "values": self._maintenanceActiveVariablesOptimum[ip],
                "timeDependent": True,
                "dimension": Dimension.ONE,
            }
        return values
