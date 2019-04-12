from FINE.component import Component, ComponentModel
from FINE import utils
import pyomo.environ as pyomo
import warnings
import pandas as pd


class Storage(Component):
    """
    A Storage component can store a commodity and thus transfers it between time steps.
    """
    def __init__(self, esM, name, commodity, chargeRate=1, dischargeRate=1,
                 chargeEfficiency=1, dischargeEfficiency=1, selfDischarge=0, cyclicLifetime=None,
                 stateOfChargeMin=0, stateOfChargeMax=1,
                 hasCapacityVariable=True, capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                 hasIsBuiltBinaryVariable=False, bigM=None, doPreciseTsaModeling=False,
                 chargeOpRateMax=None, chargeOpRateFix=None, chargeTsaWeight=1,
                 dischargeOpRateMax=None, dischargeOpRateFix=None, dischargeTsaWeight=1,
                 isPeriodicalStorage=False,
                 locationalEligibility=None, capacityMin=None, capacityMax=None, sharedPotentialID=None,
                 capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerChargeOperation=0,
                 opexPerDischargeOperation=0, opexPerCapacity=0, opexIfBuilt=0, interestRate=0.08, economicLifetime=10):
        """
        Constructor for creating an Storage class instance.
        The Storage component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param commodity: to the component related commodity.
        :type commodity: string

        **Default arguments:**

        :param chargeRate: ratio of the maximum storage inflow (in commodityUnit/hour) to the
            storage capacity (in commodityUnit).
            Example:\n
            * A hydrogen salt cavern which can store 133 GWh_H2_LHV can be charged 0.45 GWh_H2_LHV during
              one hour. The chargeRate thus equals 0.45/133 1/h.\n
            |br| * the default value is 1
        :type chargeRate: 0 <= float <=1

        :param dischargeRate: ratio of the maximum storage outflow (in commodityUnit/hour) to
            the storage capacity (in commodityUnit).
            Example:\n
            * A hydrogen salt cavern which can store 133 GWh_H2_LHV can be discharged 0.45 GWh_H2_LHV during
              one hour. The dischargeRate thus equals 0.45/133.\n
            |br| * the default value is 1
        :type dischargeRate: 0 <= float <=1

        :param chargeEfficiency: defines the efficiency with which the storage can be charged (equals
            the percentage of the injected commodity that is transformed into stored commodity).
            Enter 0.98 for 98% etc.
            |br| * the default value is 1
        :type chargeEfficiency: 0 <= float <=1

        :param dischargeEfficiency: defines the efficiency with which the storage can be discharged
            (equals the percentage of the withdrawn commodity that is transformed into stored commodity).
            Enter 0.98 for 98% etc.
            |br| * the default value is 1
        :type dischargeEfficiency: 0 <= float <=1

        :param selfDischarge: percentage of self-discharge from the storage during one hour
            |br| * the default value is 0
        :type selfDischarge: 0 <= float <=1

        :param cyclicLifetime: if specified, the total number of full cycle equivalents that are supported
            by the technology.
            |br| * the default value is None
        :type cyclicLifetime: None or positive float

        :param stateOfChargeMin: threshold (percentage) that the state of charge can not drop under
            |br| * the default value is 0
        :type stateOfChargeMin: 0 <= float <=1

        :param stateOfChargeMax: threshold (percentage) that the state of charge can not exceed
            |br| * the default value is 1
        :type stateOfChargeMax: 0 <= float <=1

        :param doPreciseTsaModeling: determines whether the state of charge is limited precisely (True) or
            with a simplified method (False). The error is small if the selfDischarge is small.
            |br| * the default value is False
        :type doPreciseTsaModeling: boolean

        :param chargeOpRateMax: if specified, indicates a maximum charging rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the charged commodity (before multiplying the charging efficiency)
            during one time step.
            |br| * the default value is None
        :type chargeOpRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model  specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param chargeOpRateFix: if specified, indicates a fixed charging rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the charged commodity (before multiplying the charging efficiency)
            during one time step.
            |br| * the default value is None
        :type chargeOpRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param chargeTsaWeight: weight with which the chargeOpRate (max/fix) time series of the
            component should be considered when applying time series aggregation.
            |br| * the default value is 1
        :type chargeTsaWeight: positive (>= 0) float

        :param dischargeOpRateMax: if specified, indicates a maximum discharging rate for each location and each
            time step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the discharged commodity (after multiplying the discharging
            efficiency) during one time step.
            |br| * the default value is None
        :type dischargeOpRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model  specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param dischargeOpRateFix: if specified, indicates a fixed discharging rate for each location and each
            time step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the charged commodity (after multiplying the discharging
            efficiency) during one time step.
            |br| * the default value is None
        :type dischargeOpRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param dischargeTsaWeight: weight with which the dischargeOpRate (max/fix) time series of the
            component should be considered when applying time series aggregation.
            |br| * the default value is 1
        :type dischargeTsaWeight: positive (>= 0) float

        :param isPeriodicalStorage: indicates if the state of charge of the storage has to be at the same value
            after the end of each period. This is especially relevant when using daily periods where short term
            storage can be restrained to daily cycles. Benefits the run time of the model.
            |br| * the default value is False
        :type isPeriodicalStorage: boolean

        :param opexPerChargeOperation: describes the cost for one unit of the charge operation.
            The cost which is directly proportional to the charge operation of the
            component is obtained by multiplying the opexPerChargeOperation parameter with the annual sum of the
            operational time series of the components. The opexPerChargeOperation can either be given as a float
            or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerChargeOperation: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param opexPerDischargeOperation: describes the cost for one unit of the discharge operation.
            The cost which is directly proportional to the discharge operation
            of the component is obtained by multiplying the opexPerDischargeOperation parameter with the annual sum
            of the operational time series of the components. The opexPerDischargeOperation can either be given as
            a float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerDischargeOperation: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.
        """
        Component. __init__(self, esM, name, dimension='1dim', hasCapacityVariable=hasCapacityVariable,
                            capacityVariableDomain=capacityVariableDomain, capacityPerPlantUnit=capacityPerPlantUnit,
                            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable, bigM=bigM,
                            locationalEligibility=locationalEligibility, capacityMin=capacityMin,
                            capacityMax=capacityMax, sharedPotentialID=sharedPotentialID, capacityFix=capacityFix,
                            isBuiltFix=isBuiltFix, investPerCapacity=investPerCapacity, investIfBuilt=investIfBuilt,
                            opexPerCapacity=opexPerCapacity, opexIfBuilt=opexIfBuilt, interestRate=interestRate,
                            economicLifetime=economicLifetime)

        # Set general storage component data: chargeRate, dischargeRate, chargeEfficiency, dischargeEfficiency,
        # selfDischarge, cyclicLifetime, stateOfChargeMin, stateOfChargeMax, isPeriodicalStorage, doPreciseTsaModeling
        utils.checkCommodities(esM, {commodity})
        self.commodity, self.commodityUnit = commodity, esM.commodityUnitsDict[commodity]
        # TODO unit and type checks
        self.chargeRate, self.dischargeRate = chargeRate, dischargeRate
        self.chargeEfficiency, self.dischargeEfficiency = chargeEfficiency, dischargeEfficiency
        self.selfDischarge = selfDischarge
        self.cyclicLifetime = cyclicLifetime
        self.stateOfChargeMin, self.stateOfChargeMax = stateOfChargeMin, stateOfChargeMax
        self.isPeriodicalStorage = isPeriodicalStorage
        self.doPreciseTsaModeling = doPreciseTsaModeling
        self.modelingClass = StorageModel

        # Set additional economic data: opexPerChargeOperation, opexPerDischargeOperation
        self.opexPerChargeOperation = utils.checkAndSetCostParameter(esM, name, opexPerChargeOperation, '1dim',
                                                                     locationalEligibility)
        self.opexPerDischargeOperation = utils.checkAndSetCostParameter(esM, name, opexPerDischargeOperation, '1dim',
                                                                        locationalEligibility)

        # Set location-specific operation parameters (charging rate, discharging rate, state of charge rate)
        # and time series aggregation weighting factor
        if chargeOpRateMax is not None and chargeOpRateFix is not None:
            chargeOpRateMax = None
            if esM.verbose < 2:
                warnings.warn('If chargeOpRateFix is specified, the chargeOpRateMax parameter is not required.\n' +
                              'The chargeOpRateMax time series was set to None.')

        self.fullChargeOpRateMax = utils.checkAndSetTimeSeries(esM, chargeOpRateMax, locationalEligibility)
        self.aggregatedChargeOpRateMax, self.chargeOpRateMax = None, None

        self.fullChargeOpRateFix = utils.checkAndSetTimeSeries(esM, chargeOpRateFix, locationalEligibility)
        self.aggregatedChargeOpRateFix, self.chargeOpRateFix = None, None

        utils.isPositiveNumber(chargeTsaWeight)
        self.chargeTsaWeight = chargeTsaWeight

        if dischargeOpRateMax is not None and dischargeOpRateFix is not None:
            dischargeOpRateMax = None
            if esM.verbose < 2:
                warnings.warn('If dischargeOpRateFix is specified, the dischargeOpRateMax parameter is not required.\n'
                              + 'The dischargeOpRateMax time series was set to None.')

        self.fullDischargeOpRateMax = utils.checkAndSetTimeSeries(esM, dischargeOpRateMax, locationalEligibility)
        self.aggregatedDischargeOpRateMax, self.dischargeOpRateMax = None, None

        self.fullDischargeOpRateFix = utils.checkAndSetTimeSeries(esM, dischargeOpRateFix, locationalEligibility)
        self.aggregatedDischargeOpRateFix, self.dischargeOpRateFix = None, None

        utils.isPositiveNumber(dischargeTsaWeight)
        self.dischargeTsaWeight = dischargeTsaWeight

        # Set locational eligibility
        timeSeriesData = None
        tsNb = sum([0 if data is None else 1 for data in [self.fullChargeOpRateMax, self.fullChargeOpRateFix,
                                                          self.fullDischargeOpRateMax, self.fullDischargeOpRateFix]])
        if tsNb > 0:
            timeSeriesData = sum([data for data in [self.fullChargeOpRateMax, self.fullChargeOpRateFix,
                                 self.fullDischargeOpRateMax, self.fullDischargeOpRateFix] if data is not None])
        self.locationalEligibility = \
            utils.setLocationalEligibility(esM, self.locationalEligibility, self.capacityMax, self.capacityFix,
                                           self.isBuiltFix, self.hasCapacityVariable, timeSeriesData)

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a storage component to the given energy system model.

        :param esM: energy system model to which the storage component should be added.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)

    def setTimeSeriesData(self, hasTSA):
        """
        Function for setting the maximum operation rate and fixed operation rate for charging and discharging
        depending on whether a time series analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        self.chargeOpRateMax = self.aggregatedChargeOpRateMax if hasTSA else self.fullChargeOpRateMax
        self.chargeOpRateFix = self.aggregatedChargeOpRateFix if hasTSA else self.fullChargeOpRateFix
        self.dischargeOpRateMax = self.aggregatedDischargeOpRateMax if hasTSA else self.fullDischargeOpRateMax
        self.dischargeOpRateFix = self.aggregatedDischargeOpRateFix if hasTSA else self.fullDischargeOpRateFix

    def getDataForTimeSeriesAggregation(self):
        """ Function for getting the required data if a time series aggregation is requested. """
        weightDict, data = {}, []
        I = [(self.fullChargeOpRateFix, self.fullChargeOpRateMax, 'chargeRate_', self.chargeTsaWeight),
             (self.fullDischargeOpRateFix, self.fullDischargeOpRateMax, 'dischargeRate_', self.dischargeTsaWeight)]

        for rateFix, rateMax, rateName, rateWeight in I:
            weightDict, data = self.prepareTSAInput(rateFix, rateMax, rateName, rateWeight, weightDict, data)
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):
        """
        Function for determining the aggregated maximum rate and the aggregated fixed operation rate for charging
        and discharging.

        :param data: Pandas DataFrame with the clustered time series data of the source component
        :type data: Pandas DataFrame
        """
        self.aggregatedChargeOpRateFix = self.getTSAOutput(self.fullChargeOpRateFix, 'chargeRate_', data)
        self.aggregatedChargeOpRateMax = self.getTSAOutput(self.fullChargeOpRateMax, 'chargeRate_', data)

        self.aggregatedDischargeOpRateFix = self.getTSAOutput(self.fullDischargeOpRateFix, 'dischargeRate_', data)
        self.aggregatedDischargeOpRateMax = self.getTSAOutput(self.fullDischargeOpRateMax, 'dischargeRate_', data)


class StorageModel(ComponentModel):
    """
    A StorageModel class instance will be instantly created if a Storage class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Storage class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The StorageModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        """" Constructor for creating a StorageModel class instance """
        self.abbrvName = 'stor'
        self.dimension = '1dim'
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.chargeOperationVariablesOptimum, self.dischargeOperationVariablesOptimum = None, None
        self.stateOfChargeOperationVariablesOptimum = None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """
        Declare sets: design variable sets, operation variable set, operation mode sets.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        compDict = self.componentsDict

        # Declare design variable sets
        self.declareDesignVarSet(pyM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        if pyM.hasTSA:
            varSet = getattr(pyM, 'designDimensionVarSet_' + self.abbrvName)

            def initDesignVarSimpleTSASet(pyM):
                return ((loc, compName) for loc, compName in varSet if not compDict[compName].doPreciseTsaModeling)
            setattr(pyM, 'designDimensionVarSetSimple_' + self.abbrvName,
                    pyomo.Set(dimen=2, initialize=initDesignVarSimpleTSASet))

            def initDesignVarPreciseTSASet(pyM):
                return ((loc, compName) for loc, compName in varSet if compDict[compName].doPreciseTsaModeling)
            setattr(pyM, 'designDimensionVarSetPrecise_' + self.abbrvName,
                    pyomo.Set(dimen=2, initialize=initDesignVarPreciseTSASet))

        # Declare operation variable set
        self.declareOpVarSet(esM, pyM)

        # Declare sets for case differentiation of operating modes
        # * Charge operation
        self.declareOperationModeSets(pyM, 'chargeOpConstrSet', 'chargeOpRateMax', 'chargeOpRateFix')
        # * Discharge operation
        self.declareOperationModeSets(pyM, 'dischargeOpConstrSet', 'dischargeOpRateMax', 'dischargeOpRateFix')

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """
        Declare design and operation variables.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        # Capacity variables [commodityUnit*hour]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not
        self.declareBinaryDesignDecisionVars(pyM)
        # Energy amount injected into a storage (before injection efficiency losses) between two time steps
        self.declareOperationVars(pyM, 'chargeOp')
        # Energy amount delivered from a storage (after delivery efficiency losses) between two time steps
        self.declareOperationVars(pyM, 'dischargeOp')

        # Inventory of storage components [commodityUnit*hour]
        if not pyM.hasTSA:
            # Energy amount stored at the beginning of a time step during the (one) period (the i-th state of charge
            # refers to the state of charge at the beginning of the i-th time step, the last index is the state of
            # charge after the last time step)
            setattr(pyM, 'stateOfCharge_' + self.abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' +
                    self.abbrvName), pyM.interTimeStepsSet, domain=pyomo.NonNegativeReals))
        else:
            # (Virtual) energy amount stored during a period (the i-th state of charge refers to the state of charge at
            # the beginning of the i-th time step, the last index is the state of charge after the last time step)
            setattr(pyM, 'stateOfCharge_' + self.abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' +
                    self.abbrvName), pyM.interTimeStepsSet, domain=pyomo.Reals))
            # (Virtual) minimum amount of energy stored within a period
            setattr(pyM, 'stateOfChargeMin_' + self.abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' +
                    self.abbrvName), esM.typicalPeriods, domain=pyomo.Reals))
            # (Virtual) maximum amount of energy stored within a period
            setattr(pyM, 'stateOfChargeMax_' + self.abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' +
                    self.abbrvName), esM.typicalPeriods, domain=pyomo.Reals))
            # (Real) energy amount stored at the beginning of a period between periods(the i-th state of charge refers
            # to the state of charge at the beginning of the i-th period, the last index is the state of charge after
            # the last period)
            setattr(pyM, 'stateOfChargeInterPeriods_' + self.abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_'
                    + self.abbrvName), esM.interPeriodTimeSteps, domain=pyomo.NonNegativeReals))

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def connectSOCs(self, pyM, esM):
        """
        Declare the constraint for connecting the state of charge with the charge and discharge operation:
        the change in the state of charge between two points in time has to match the values of charging and
        discharging (considering the efficiencies of these processes) within the time step in between minus
        the self-discharge of the storage.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)
        chargeOp, dischargeOp = getattr(pyM, 'chargeOp_' + abbrvName), getattr(pyM, 'dischargeOp_' + abbrvName)
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def connectSOCs(pyM, loc, compName, p, t):
            return (SOC[loc, compName, p, t+1] - SOC[loc, compName, p, t] *
                    (1 - compDict[compName].selfDischarge) ** esM.hoursPerTimeStep ==
                    chargeOp[loc, compName, p, t] * compDict[compName].chargeEfficiency -
                    dischargeOp[loc, compName, p, t] / compDict[compName].dischargeEfficiency)
        setattr(pyM, 'ConstrConnectSOC_' + abbrvName, pyomo.Constraint(opVarSet, pyM.timeSet, rule=connectSOCs))

    def cyclicState(self, pyM, esM):
        """
        Declare the constraint for connecting the states of charge: the state of charge at the beginning of a period
        has to be the same as the state of charge in the end of that period.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)
        if not pyM.hasTSA:
            def cyclicState(pyM, loc, compName):
                return SOC[loc, compName, 0, 0] == SOC[loc, compName, 0, esM.timeStepsPerPeriod[-1] + 1]
        else:
            SOCInter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)
            def cyclicState(pyM, loc, compName):
                return SOCInter[loc, compName, 0] == SOCInter[loc, compName, esM.interPeriodTimeSteps[-1]]
        setattr(pyM, 'ConstrCyclicState_' + abbrvName, pyomo.Constraint(opVarSet, rule=cyclicState))

    def cyclicLifetime(self, pyM, esM):
        """
        Declare the constraint for limiting the number of full cycle equivalents to stay below cyclic lifetime.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        chargeOp, capVar = getattr(pyM, 'chargeOp_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        capVarSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        def cyclicLifetime(pyM, loc, compName):
            return (sum(chargeOp[loc, compName, p, t] * esM.periodOccurrences[p] for p, t in pyM.timeSet) /
                    esM.numberOfYears <= capVar[loc, compName] *
                    (compDict[compName].stateOfChargeMax - compDict[compName].stateOfChargeMin) *
                    compDict[compName].cyclicLifetime / compDict[compName].economicLifetime[loc]
                    if compDict[compName].cyclicLifetime is not None else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrCyclicLifetime_' + abbrvName, pyomo.Constraint(capVarSet, rule=cyclicLifetime))

    def connectInterPeriodSOC(self, pyM, esM):
        """
        Declare the constraint that the state of charge at the end of each period has to be equivalent to the state of
        charge of the period before it (minus its self discharge) plus the change in the state of charge which
        happened during the typical period which was assigned to that period.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)
        SOCInter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)

        def connectInterSOC(pyM, loc, compName, pInter):
            return SOCInter[loc, compName, pInter + 1] == \
                   SOCInter[loc, compName, pInter] * (1 - compDict[compName].selfDischarge) ** \
                   ((esM.timeStepsPerPeriod[-1] + 1) * esM.hoursPerTimeStep) + \
                   SOC[loc, compName, esM.periodsOrder[pInter], esM.timeStepsPerPeriod[-1] + 1]
        setattr(pyM, 'ConstrInterSOC_' + abbrvName, pyomo.Constraint(opVarSet, esM.periods, rule=connectInterSOC))

    def intraSOCstart(self, pyM, esM):
        """
        Declare the constraint that the (virtual) state of charge at the beginning of a typical period is zero.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        abbrvName = self.abbrvName
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)

        def intraSOCstart(pyM, loc, compName, p):
            return SOC[loc, compName, p, 0] == 0
        setattr(pyM, 'ConstrSOCPeriodStart_' + abbrvName,
                pyomo.Constraint(opVarSet, esM.typicalPeriods, rule=intraSOCstart))

    def equalInterSOC(self, pyM, esM):
        """
        Declare the constraint that, if periodic storage is selected, the states of charge between periods
        have the same value.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)
        SOCInter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)

        def equalInterSOC(pyM, loc, compName, pInter):
            return (SOCInter[loc, compName, pInter] == SOCInter[loc, compName, pInter + 1]
                    if compDict[compName].isPeriodicalStorage else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrEqualInterSOC_' + abbrvName, pyomo.Constraint(opVarSet, esM.periods, rule=equalInterSOC))

    def minSOC(self, pyM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] has to be larger than the
        installed capacity [commodityUnit*h] multiplied with the relative minimum state of charge.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVarSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)
        SOC, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)

        def SOCMin(pyM, loc, compName, p, t):
            return SOC[loc, compName, p, t] >= capVar[loc, compName] * compDict[compName].stateOfChargeMin
        setattr(pyM, 'ConstrSOCMin_' + abbrvName, pyomo.Constraint(capVarSet, pyM.timeSet, rule=SOCMin))

    def limitSOCwithSimpleTsa(self, pyM, esM):
        """
        Simplified version of the state of charge limitation control.
        The error compared to the precise version is small in cases of small selfDischarge.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVarSimpleSet = getattr(pyM, 'designDimensionVarSetSimple_' + abbrvName)
        SOC, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        SOCmax, SOCmin = getattr(pyM, 'stateOfChargeMax_' + abbrvName), getattr(pyM, 'stateOfChargeMin_' + abbrvName)
        SOCInter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)

        # The maximum (virtual) state of charge during a typical period is larger than all occurring (virtual)
        # states of charge in that period (the last time step is considered in the subsequent period for t=0).
        def SOCintraPeriodMax(pyM, loc, compName, p, t):
            return SOC[loc, compName, p, t] <= SOCmax[loc, compName, p]
        setattr(pyM, 'ConstSOCintraPeriodMax_' + abbrvName,
                pyomo.Constraint(capVarSimpleSet, pyM.timeSet, rule=SOCintraPeriodMax))

        # The minimum (virtual) state of charge during a typical period is smaller than all occurring (virtual)
        # states of charge in that period (the last time step is considered in the subsequent period for t=0).
        def SOCintraPeriodMin(pyM, loc, compName, p, t):
            return SOC[loc, compName, p, t] >= SOCmin[loc, compName, p]
        setattr(pyM, 'ConstSOCintraPeriodMin_' + abbrvName,
                pyomo.Constraint(capVarSimpleSet, pyM.timeSet, rule=SOCintraPeriodMin))

        # The state of charge at the beginning of one period plus the maximum (virtual) state of charge
        # during that period has to be smaller than the installed capacities multiplied with the relative maximum
        # state of charge.
        def SOCMaxSimple(pyM, loc, compName, pInter):
            return (SOCInter[loc, compName, pInter] + SOCmax[loc, compName, esM.periodsOrder[pInter]]
                    <= capVar[loc, compName] * compDict[compName].stateOfChargeMax)
        setattr(pyM, 'ConstrSOCMaxSimple_' + abbrvName,
                pyomo.Constraint(capVarSimpleSet, esM.periods, rule=SOCMaxSimple))

        # The state of charge at the beginning of one period plus the minimum (virtual) state of charge
        # during that period has to be larger than the installed capacities multiplied with the relative minimum
        # state of charge.
        def SOCMinSimple(pyM, loc, compName, pInter):
            return (SOCInter[loc, compName, pInter] * (1 - compDict[compName].selfDischarge) **
                    ((esM.timeStepsPerPeriod[-1] + 1) * esM.hoursPerTimeStep)
                    + SOCmin[loc, compName, esM.periodsOrder[pInter]]
                    >= capVar[loc, compName] * compDict[compName].stateOfChargeMin)
        setattr(pyM, 'ConstrSOCMinSimple_' + abbrvName,
                pyomo.Constraint(capVarSimpleSet, esM.periods, rule=SOCMinSimple))

    def operationModeSOC(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        [commodityUnit*h] and the relative maximum state of charge [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        # Operation [commodityUnit*h] limited by the installed capacity [commodityUnit*h] multiplied by the relative
        # maximum state of charge.
        def op(pyM, loc, compName, p, t):
            return (opVar[loc, compName, p, t] <=
                    compDict[compName].stateOfChargeMax * capVar[loc, compName])
        setattr(pyM, 'ConstrSOCMaxPrecise_' + abbrvName, pyomo.Constraint(constrSet, pyM.timeSet, rule=op))

    def operationModeSOCwithTSA(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        # [commodityUnit*h] and the relative maximum state of charge [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)
        SOC, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        def SOCMaxPrecise(pyM, loc, compName, pInter, t):
            if compDict[compName].doPreciseTsaModeling:
                return (SOCinter[loc, compName, pInter] *
                        ((1 - compDict[compName].selfDischarge) ** (t * esM.hoursPerTimeStep)) +
                        SOC[loc, compName, esM.periodsOrder[pInter], t]
                        <= capVar[loc, compName] * compDict[compName].stateOfChargeMax)
            else:
                return pyomo.Constraint.Skip
        setattr(pyM, 'ConstrSOCMaxPrecise_' + abbrvName,
                pyomo.Constraint(constrSet, esM.periods, esM.timeStepsPerPeriod, rule=SOCMaxPrecise))

    def minSOCwithTSAprecise(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] at each time step cannot be smaller
        than the installed capacity [commodityUnit*h] multiplied with the relative minimum state of charge [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)
        SOC, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        capVarPreciseSet = getattr(pyM, 'designDimensionVarSetPrecise_' + abbrvName)

        def SOCMinPrecise(pyM, loc, compName, pInter, t):
            return (SOCinter[loc, compName, pInter] * ((1 - compDict[compName].selfDischarge) **
                    (t * esM.hoursPerTimeStep)) + SOC[loc, compName, esM.periodsOrder[pInter], t]
                    >= capVar[loc, compName] * compDict[compName].stateOfChargeMin)
        setattr(pyM, 'ConstrSOCMinPrecise_' + abbrvName,
                pyomo.Constraint(capVarPreciseSet, esM.periods, esM.timeStepsPerPeriod, rule=SOCMinPrecise))

    def declareComponentConstraints(self, esM, pyM):
        """
        Declare time independent and dependent constraints.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        ################################################################################################################
        #                                    Declare time independent constraints                                      #
        ################################################################################################################

        # Determine the components' capacities from the number of installed units
        self.capToNbReal(pyM)
        # Determine the components' capacities from the number of installed units
        self.capToNbInt(pyM)
        # Enforce the consideration of the binary design variables of a component
        self.bigM(pyM)
        # Enforce the consideration of minimum capacities for components with design decision variables
        self.capacityMinDec(pyM)
        # Sets, if applicable, the installed capacities of a component
        self.capacityFix(pyM)
        # Sets, if applicable, the binary design variables of a component
        self.designBinFix(pyM)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Constraint for connecting the state of charge with the charge and discharge operation
        self.connectSOCs(pyM, esM)

        #                              Constraints for enforcing charging operation modes                              #

        # Charging of storage [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] multiplied by
        # the hours per time step [h] and the charging rate factor [1/h]
        self.operationMode1(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp', 'chargeRate')
        # Charging of storage [commodityUnit*h] is equal to the installed capacity [commodityUnit*h] multiplied by
        # the hours per time step [h] and the charging operation time series [1/h]
        self.operationMode2(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp', 'chargeOpRateFix')
        # Charging of storage [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] multiplied by
        # the hours per time step [h] and the charging operation time series [1/h]
        self.operationMode3(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp', 'chargeOpRateMax')
        # Operation [commodityUnit*h] is equal to the operation time series [commodityUnit*h]
        self.operationMode4(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp', 'chargeOpRateFix')
        # Operation [commodityUnit*h] is limited by the operation time series [commodityUnit*h]
        self.operationMode5(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp', 'chargeOpRateMax')

        #                             Constraints for enforcing discharging operation modes                            #

        # Discharging of storage [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] multiplied
        # by the hours per time step [h] and the discharging rate factor [1/h]
        self.operationMode1(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp', 'dischargeRate')
        # Discharging of storage [commodityUnit*h] is equal to the installed capacity [commodityUnit*h] multiplied
        # by the hours per time step [h] and the discharging operation time series [1/h]
        self.operationMode2(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp', 'dischargeOpRateFix')
        # Discharging of storage [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] multiplied
        # by the hours per time step [h] and the discharging operation time series [1/h]
        self.operationMode3(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp', 'dischargeOpRateMax')
        # Operation [commodityUnit*h] is equal to the operation time series [commodityUnit*h]
        self.operationMode4(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp', 'dischargeOpRateFix')
        # Operation [commodityUnit*h] is limited by the operation time series [commodityUnit*h]
        self.operationMode5(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp', 'dischargeOpRateMax')

        # Cyclic constraint enforcing that all storages have the same state of charge at the the beginning of the first
        # and the end of the last time step
        self.cyclicState(pyM, esM)

        # Constraint for limiting the number of full cycle equivalents to stay below cyclic lifetime
        self.cyclicLifetime(pyM, esM)

        if pyM.hasTSA:
            # The state of charge at the end of each period is equivalent to the state of charge of the period before it
            # (minus its self discharge) plus the change in the state of charge which happened during the typical
            # period which was assigned to that period
            self.connectInterPeriodSOC(pyM, esM)
            # The (virtual) state of charge at the beginning of a typical period is zero
            self.intraSOCstart(pyM, esM)
            # If periodic storage is selected, the states of charge between periods have the same value
            self.equalInterSOC(pyM, esM)

        # Ensure that the state of charge is within the operating limits of the installed capacities
        if not pyM.hasTSA:
            #              Constraints for enforcing a state of charge operation mode within given limits              #

            # State of charge [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] and the relative
            # maximum state of charge
            self.operationModeSOC(pyM, esM)

            # The state of charge [commodityUnit*h] has to be larger than the installed capacity [commodityUnit*h]
            # multiplied with the relative minimum state of charge
            self.minSOC(pyM)

        else:
            #                       Simplified version of the state of charge limitation control                       #
            #           (The error compared to the precise version is small in cases of small selfDischarge)           #
            self.limitSOCwithSimpleTsa(pyM, esM)

            #                        Precise version of the state of charge limitation control                         #

            # Constraints for enforcing a state of charge operation within given limits

            # State of charge [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] and the
            # relative maximum state of charge
            self.operationModeSOCwithTSA(pyM, esM)

            # The state of charge at each time step cannot be smaller than the installed capacity multiplied with the
            # relative minimum state of charge
            self.minSOCwithTSAprecise(pyM, esM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        """ Get contributions to shared location potential. """
        return super().getSharedPotentialContribution(pyM, key, loc)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        """
        Check if operation variables exist in the modeling class at a location which are connected to a commodity.
        
        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: Name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """
        return any([comp.commodity == commod and comp.locationalEligibility[loc] == 1
                    for comp in self.componentsDict.values()])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """ Get contribution to a commodity balance. """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        chargeOp, dischargeOp = getattr(pyM, 'chargeOp_' + abbrvName), getattr(pyM, 'dischargeOp_' + abbrvName)
        opVarDict = getattr(pyM, 'operationVarDict_' + abbrvName)
        return sum(dischargeOp[loc, compName, p, t] - chargeOp[loc, compName, p, t]
                   for compName in opVarDict[loc] if commod == self.componentsDict[compName].commodity)

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        capexCap = self.getEconomicsTI(pyM, ['investPerCapacity'], 'cap', 'CCF')
        capexDec = self.getEconomicsTI(pyM, ['investIfBuilt'], 'designBin', 'CCF')
        opexCap = self.getEconomicsTI(pyM, ['opexPerCapacity'], 'cap')
        opexDec = self.getEconomicsTI(pyM, ['opexIfBuilt'], 'designBin')
        opexOp1 = self.getEconomicsTD(pyM, esM, ['opexPerChargeOperation'], 'chargeOp', 'operationVarDict')
        opexOp2 = self.getEconomicsTD(pyM, esM, ['opexPerDischargeOperation'], 'dischargeOp', 'operationVarDict')

        return capexCap + capexDec + opexCap + opexDec + opexOp1 + opexOp2

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        chargeOp, dischargeOp = getattr(pyM, 'chargeOp_' + abbrvName), getattr(pyM, 'dischargeOp_' + abbrvName)
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(esM, pyM, esM.locations, 'commodityUnit', '*h')

        # Set optimal operation variables and append optimization summary
        props = ['operationCharge', 'operationDischarge', 'opexCharge', 'opexDischarge']
        units = ['[-]', '[-]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]].commodityUnit + '*h/a]')
                          if x[1] == 'operationCharge' else x, tuples))
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]].commodityUnit + '*h/a]')
                          if x[1] == 'operationDischarge' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM.locations)).sort_index()

        # * charge variables and contributions
        optVal = utils.formatOptimizationOutput(chargeOp.get_values(), 'operationVariables', '1dim', esM.periodsOrder)
        self.chargeOperationVariablesOptimum = optVal

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name].opexPerChargeOperation[op.index], axis=1)
            optSummary.loc[[(ix, 'operationCharge', '[' + compDict[ix].commodityUnit + '*h/a]')
                             for ix in opSum.index], opSum.columns] = opSum.values/esM.numberOfYears
            optSummary.loc[[(ix, 'opexCharge', '[' + esM.costUnit + '/a]') for ix in ox.index],
                            ox.columns] = ox.values/esM.numberOfYears

        # * discharge variables and contributions
        optVal = utils.formatOptimizationOutput(dischargeOp.get_values(), 'operationVariables', '1dim',
                                                esM.periodsOrder)
        self.dischargeOperationVariablesOptimum = optVal

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name].opexPerDischargeOperation[op.index], axis=1)
            optSummary.loc[[(ix, 'operationDischarge', '[' + compDict[ix].commodityUnit + '*h/a]')
                             for ix in opSum.index], opSum.columns] = opSum.values/esM.numberOfYears
            optSummary.loc[[(ix, 'opexDischarge', '[' + esM.costUnit + '/a]') for ix in ox.index],
                            ox.columns] = ox.values/esM.numberOfYears

        # * set state of charge variables
        if not pyM.hasTSA:
            optVal = utils.formatOptimizationOutput(SOC.get_values(), 'operationVariables', '1dim', esM.periodsOrder)
            # Remove the last column (by applying the cycle constraint, the first and the last columns are equal to each
            # other)
            optVal = optVal.loc[:, :len(optVal.columns) - 2]
            self.stateOfChargeOperationVariablesOptimum = optVal
            utils.setOptimalComponentVariables(optVal, '_stateOfChargeVariablesOptimum', compDict)
        else:
            SOCinter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)
            stateOfChargeIntra = SOC.get_values()
            stateOfChargeInter = SOCinter.get_values()
            if stateOfChargeIntra is not None:
                # Convert dictionary to DataFrame, transpose, put the period column first and sort the index
                # Results in a one dimensional DataFrame
                stateOfChargeIntra = pd.DataFrame(stateOfChargeIntra, index=[0]).T.swaplevel(i=0, j=-2).sort_index()
                stateOfChargeInter = pd.DataFrame(stateOfChargeInter, index=[0]).T.swaplevel(i=0, j=1).sort_index()
                # Unstack time steps (convert to a two dimensional DataFrame with the time indices being the columns)
                stateOfChargeIntra = stateOfChargeIntra.unstack(level=-1)
                stateOfChargeInter = stateOfChargeInter.unstack(level=-1)
                # Get rid of the unnecessary 0 level
                stateOfChargeIntra.columns = stateOfChargeIntra.columns.droplevel()
                stateOfChargeInter.columns = stateOfChargeInter.columns.droplevel()
                # Concat data
                data = []
                for count, p in enumerate(esM.periodsOrder):
                    data.append((stateOfChargeInter.loc[:, count] +
                                 stateOfChargeIntra.loc[p].loc[:, :esM.timeStepsPerPeriod[-1]].T).T)
                optVal = pd.concat(data, axis=1, ignore_index=True)
            else:
                optVal = None
            self.stateOfChargeOperationVariablesOptimum = optVal
            utils.setOptimalComponentVariables(optVal, '_stateOfChargeVariablesOptimum', compDict)

        # Append optimization summaries
        optSummary = optSummary.append(optSummaryBasic).sort_index()

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'TAC') |
                           (optSummary.index.get_level_values(1) == 'opexCharge') |
                           (optSummary.index.get_level_values(1) == 'opexDischarge')].groupby(level=0).sum().values

        self.optSummary = optSummary

    def getOptimalValues(self, name='all'):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:\n
        * 'capacityVariables',
        * 'isBuiltVariables',
        * 'chargeOperationVariablesOptimum',
        * 'dischargeOperationVariablesOptimum',
        * 'stateOfChargeOperationVariablesOptimum',
        * 'all' or another input: all variables are returned.\n
        |br| * the default value is 'all'
        :type name: string

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        if name == 'capacityVariablesOptimum':
            return {'values': self.capacityVariablesOptimum, 'timeDependent': False, 'dimension': self.dimension}
        elif name == 'isBuiltVariablesOptimum':
            return {'values': self.isBuiltVariablesOptimum, 'timeDependent': False, 'dimension': self.dimension}
        elif name == 'chargeOperationVariablesOptimum':
            return {'values': self.chargeOperationVariablesOptimum, 'timeDependent': True, 'dimension': self.dimension}
        elif name == 'dischargeOperationVariablesOptimum':
            return {'values': self.dischargeOperationVariablesOptimum, 'timeDependent': True, 'dimension':
                    self.dimension}
        elif name == 'stateOfChargeOperationVariablesOptimum':
            return {'values': self.stateOfChargeOperationVariablesOptimum, 'timeDependent': True, 'dimension':
                    self.dimension}
        else:
            return {'capacityVariablesOptimum': {'values': self.capacityVariablesOptimum, 'timeDependent': False,
                                                 'dimension': self.dimension},
                    'isBuiltVariablesOptimum': {'values': self.isBuiltVariablesOptimum, 'timeDependent': False,
                                                'dimension': self.dimension},
                    'chargeOperationVariablesOptimum': {'values': self.chargeOperationVariablesOptimum,
                                                        'timeDependent': True, 'dimension': self.dimension},
                    'dischargeOperationVariablesOptimum': {'values': self.dischargeOperationVariablesOptimum,
                                                           'timeDependent': True, 'dimension': self.dimension},
                    'stateOfChargeOperationVariablesOptimum': {'values': self.stateOfChargeOperationVariablesOptimum,
                                                               'timeDependent': True, 'dimension': self.dimension}}
