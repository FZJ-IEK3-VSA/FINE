from FINE.component import Component, ComponentModel
from FINE import utils
import pyomo.environ as pyomo
import warnings
import pandas as pd


class Storage(Component):
    """
    Doc
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

        :param chargeRate: ratio of the maximum storage inflow (in commodityUnit/hour) and the
            storage capacity (in commodityUnit).
            Example:
            * A hydrogen salt cavern which can store 133 GWh_H2_LHV can be charged 0.45 GWh_H2_LHV during
              one hour. The chargeRate thus equals 0.45/133.
            |br| * the default value is 1
        :type chargeRate: 0 <= float <=1

        :param dischargeRate: ratio of the maximum storage outflow (in commodityUnit/hour) and
            the storage capacity (in commodityUnit).
            Example:
            * A hydrogen salt cavern which can store 133 GWh_H2_LHV can be discharged 0.45 GWh_H2_LHV during
              one hour. The dischargeRate thus equals 0.45/133.
            |br| * the default value is 1
        :type chargeRate: 0 <= float <=1

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
        :type cyclicLifetime: positive float

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

        :param chargeOpRateMax: if specified indicates a maximum charging rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. in that case a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the charged commodity (before multiplying the charging efficiency)
            during one time step.
            |br| * the default value is None
        :type chargeOpRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model  specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param chargeOpRateFix: if specified indicates a fixed charging rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. in that case a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodity, referring to the charged commodity (before multiplying the charging efficiency)
            during one time step.
            |br| * the default value is None
        :type chargeOpRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param chargeTsaWeight: weight with which the chargeOpRate (max/fix) time series of the
            component should be considered when applying time series aggregation.
            |br| * the default value is 1
        :type chargeTsaWeight: positive (>= 0) float

        :param dischargeOpRateMax: if specified indicates a maximum discharging rate for each location and each
            time step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. in that case a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the discharged commodity (after multiplying the discharging
            efficiency) during one time step.
            |br| * the default value is None
        :type dischargeOpRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model  specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param dischargeOpRateFix: if specified indicates a fixed discharging rate for each location and each
            time step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. in that case a value of 1 indicates a utilization of 100% of the
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
        :type stateOfChargeTsaWeight: boolean

        :param opexPerChargeOperation: cost which is directly proportional to the charge operation of the
            component is obtained by multiplying the opexPerOperation parameter with the annual sum of the
            operational time series of the components. The opexPerOperation can either be given as a float
            or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerChargeOperation: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param opexPerDischargeOperation: cost which is directly proportional to the discharge operation
            of the component is obtained by multiplying the opexPerOperation parameter with the annual sum
            of the operational time series of the components. The opexPerOperation can either be given as
            a float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0

        :type opexPerDischargeOperation: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.
            component (in the physicalUnit of the component) and not of the specific operation itself are
            obtained by multiplying the capacity of the component at a location with the opexPerCapacity
            factor. The opexPerCapacity can either be given as a float or a Pandas Series with location
            specific values.
        """
        Component. __init__(self, esM, name, dimension='1dim', hasCapacityVariable=hasCapacityVariable,
                            capacityVariableDomain=capacityVariableDomain, capacityPerPlantUnit=capacityPerPlantUnit,
                            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable, bigM=bigM,
                            locationalEligibility=locationalEligibility, capacityMin=capacityMin,
                            capacityMax=capacityMax, sharedPotentialID=sharedPotentialID, capacityFix=capacityFix,
                            isBuiltFix=isBuiltFix, investPerCapacity=investPerCapacity, investIfBuilt=investIfBuilt,
                            opexPerCapacity=opexPerCapacity, opexIfBuilt=opexIfBuilt, interestRate=interestRate,
                            economicLifetime=economicLifetime)

        # Set general storage component data
        utils.checkCommodities(esM, {commodity})
        self._commodity, self._commodityUnit = commodity, esM._commoditiyUnitsDict[commodity]
        # TODO unit and type checks
        self._chargeRate, self._dischargeRate = chargeRate, dischargeRate
        self._chargeEfficiency, self._dischargeEfficiency = chargeEfficiency, dischargeEfficiency
        self._selfDischarge = selfDischarge
        self._cyclicLifetime = cyclicLifetime
        self._stateOfChargeMin, self._stateOfChargeMax = stateOfChargeMin, stateOfChargeMax
        self._isPeriodicalStorage = isPeriodicalStorage
        self._doPreciseTsaModeling = doPreciseTsaModeling
        self._modelingClass = StorageModel

        # Set additional economic data
        self._opexPerChargeOperation = utils.checkAndSetCostParameter(esM, name, opexPerChargeOperation, '1dim',
                                                                      locationalEligibility)
        self._opexPerDischargeOperation = utils.checkAndSetCostParameter(esM, name, opexPerDischargeOperation, '1dim',
                                                                      locationalEligibility)

        # Set location-specific operation parameters (Charging rate, discharging rate, state of charge rate)
        # and time series aggregation weighting factor
        if chargeOpRateMax is not None and chargeOpRateFix is not None:
            chargeOpRateMax = None
            warnings.warn('If chargeOpRateFix is specified, the chargeOpRateMax parameter is not required.\n' +
                          'The chargeOpRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, chargeOpRateMax, locationalEligibility)
        utils.checkOperationTimeSeriesInputParameters(esM, chargeOpRateFix, locationalEligibility)

        self._fullChargeOpRateMax = utils.setFormattedTimeSeries(chargeOpRateMax)
        self._aggregatedChargeOpRateMax = None
        self._chargeOpRateMax = None

        self._fullChargeOpRateFix = utils.setFormattedTimeSeries(chargeOpRateFix)
        self._aggregatedChargeOpRateFix = None
        self._chargeOpRateFix = None

        utils.isPositiveNumber(chargeTsaWeight)
        self._chargeTsaWeight = chargeTsaWeight

        if dischargeOpRateMax is not None and dischargeOpRateFix is not None:
            dischargeOpRateMax = None
            warnings.warn('If dischargeOpRateFix is specified, the dischargeOpRateMax parameter is not required.\n' +
                          'The dischargeOpRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, dischargeOpRateMax, locationalEligibility)
        utils.checkOperationTimeSeriesInputParameters(esM, dischargeOpRateFix, locationalEligibility)

        self._fullDischargeOpRateMax = utils.setFormattedTimeSeries(dischargeOpRateMax)
        self._aggregatedDischargeOpRateMax = None
        self._dischargeOpRateMax = None

        self._fullDischargeOpRateFix = utils.setFormattedTimeSeries(dischargeOpRateFix)
        self._aggregatedDischargeOpRateFix = None
        self._dischargeOpRateFix = None

        utils.isPositiveNumber(dischargeTsaWeight)
        self._dischargeTsaWeight = dischargeTsaWeight

        # Set locational eligibility
        timeSeriesData = None
        tsNb = sum([0 if data is None else 1 for data in [chargeOpRateMax, chargeOpRateFix, dischargeOpRateMax,
                    dischargeOpRateFix, ]])
        if tsNb > 0:
            timeSeriesData = sum([data for data in [chargeOpRateMax, chargeOpRateFix, dischargeOpRateMax,
                                  dischargeOpRateFix, ] if data is not None])
        self._locationalEligibility = \
            utils.setLocationalEligibility(esM, self._locationalEligibility, self._capacityMax, self._capacityFix,
                                           self._isBuiltFix, self._hasCapacityVariable, timeSeriesData)

    def addToEnergySystemModel(self, esM):
        super().addToEnergySystemModel(esM)

    def setTimeSeriesData(self, hasTSA):
        self._chargeOpRateMax = self._aggregatedChargeOpRateMax if hasTSA else self._fullChargeOpRateMax
        self._chargeOpRateFix = self._aggregatedChargeOpRateFix if hasTSA else self._fullChargeOpRateFix
        self._dischargeOpRateMax = self._aggregatedChargeOpRateMax if hasTSA else self._fullDischargeOpRateMax
        self._dischargeOpRateFix = self._aggregatedChargeOpRateFix if hasTSA else self._fullDischargeOpRateFix

    def getDataForTimeSeriesAggregation(self):
        weightDict, data = {}, []
        I = [(self._fullChargeOpRateFix, self._fullChargeOpRateMax, '_chargeRate_', self._chargeTsaWeight),
             (self._fullDischargeOpRateFix, self._fullDischargeOpRateMax, '_dischargeRate_', self._dischargeTsaWeight)]

        for rateFix, rateMax, rateName, rateWeight in I:
            weightDict, data = self.prepareTSAInput(rateFix, rateMax, rateName, rateWeight, weightDict, data)
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):

        self._aggregatedChargeOpRateFix = self.getTSAOutput(self._fullChargeOpRateFix, '_chargeRate_', data)
        self._aggregatedChargeOpRateMax = self.getTSAOutput(self._fullChargeOpRateMax, '_chargeRate_', data)

        self._aggregatedDischargeOpRateFix = self.getTSAOutput(self._fullDischargeOpRateFix, '_dischargeRate_', data)
        self._aggregatedDischargeOpRateMax = self.getTSAOutput(self._fullDischargeOpRateMax, '_dischargeRate_', data)


class StorageModel(ComponentModel):
    """ Doc """

    def __init__(self):
        self._abbrvName = 'stor'
        self._dimension = '1dim'
        self._componentsDict = {}
        self._capacityVariablesOptimum, self._isBuiltVariablesOptimum = None, None
        self._chargeOperationVariablesOptimum, self._dischargeOperationVariablesOptimum = None, None
        self._stateOfChargeOperationVariablesOptimum = None
        self._optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """ Declares sets and dictionaries """
        compDict = self._componentsDict

        # Declare design variable sets
        self.initDesignVarSet(pyM)
        self.initContinuousDesignVarSet(pyM)
        self.initDiscreteDesignVarSet(pyM)
        self.initDesignDecisionVarSet(pyM)

        if pyM.hasTSA:
            varSet = getattr(pyM, 'designDimensionVarSet_' + self._abbrvName)

            def initDesignVarSimpleTSASet(pyM):
                return ((loc, compName) for loc, compName in varSet if not compDict[compName]._doPreciseTsaModeling)
            setattr(pyM, 'designDimensionVarSetSimple_' + self._abbrvName,
                    pyomo.Set(dimen=2, initialize=initDesignVarSimpleTSASet))

            def initDesignVarPreciseTSASet(pyM):
                return ((loc, compName) for loc, compName in varSet if compDict[compName]._doPreciseTsaModeling)
            setattr(pyM, 'designDimensionVarSetPrecise_' + self._abbrvName,
                    pyomo.Set(dimen=2, initialize=initDesignVarPreciseTSASet))

        # Declare operation variable set
        self.initOpVarSet(esM, pyM)

        # Declare sets for case differentiation of operating modes
        # * Charge operation
        self.declareOperationModeSets(pyM, 'chargeOpConstrSet', '_chargeOpRateMax', '_chargeOpRateFix')
        # * Discharge operation
        self.declareOperationModeSets(pyM, 'dischargeOpConstrSet', '_dischargeOpRateMax', '_dischargeOpRateFix')

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """ Declares design and operation variables """

        # Capacity variables in [commodityUnit*hour]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components in [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components in [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not in [-]
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
            setattr(pyM, 'stateOfCharge_' + self._abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' +
                    self._abbrvName), pyM.interTimeStepsSet, domain=pyomo.NonNegativeReals))
        else:
            # (Virtual) energy amount stored during a period (the i-th state of charge refers to the state of charge at
            # the beginning of the i-th time step, the last index is the state of charge after the last time step)
            setattr(pyM, 'stateOfCharge_' + self._abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' +
                    self._abbrvName), pyM.interTimeStepsSet, domain=pyomo.Reals))
            # (Virtual) minimum amount of energy stored within a period
            setattr(pyM, 'stateOfChargeMin_' + self._abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' +
                    self._abbrvName), esM._typicalPeriods, domain=pyomo.Reals))
            # (Virtual) maximum amount of energy stored within a period
            setattr(pyM, 'stateOfChargeMax_' + self._abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' +
                    self._abbrvName), esM._typicalPeriods, domain=pyomo.Reals))
            # (Real) energy amount stored at the beginning of a period between periods(the i-th state of charge refers
            # to the state of charge at the beginning of the i-th period, the last index is the state of charge after
            # the last period)
            setattr(pyM, 'stateOfChargeInterPeriods_' + self._abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_'
                    + self._abbrvName), esM._interPeriodTimeSteps, domain=pyomo.NonNegativeReals))

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def connectSOCs(self, pyM, esM):
        """ Constraint for connecting the state of charge with the charge and discharge operation """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)
        chargeOp, dischargeOp = getattr(pyM, 'chargeOp_' + abbrvName), getattr(pyM, 'dischargeOp_' + abbrvName)
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def connectSOCs(pyM, loc, compName, p, t):
            return (SOC[loc, compName, p, t+1] - SOC[loc, compName, p, t] *
                    (1 - compDict[compName]._selfDischarge) ** esM._hoursPerTimeStep ==
                    chargeOp[loc, compName, p, t] * compDict[compName]._chargeEfficiency -
                    dischargeOp[loc, compName, p, t] / compDict[compName]._dischargeEfficiency)
        setattr(pyM, 'ConstrConnectSOC_' + abbrvName, pyomo.Constraint(opVarSet, pyM.timeSet, rule=connectSOCs))

    def cyclicState(self, pyM, esM):
        """ Constraint for connecting the state of charge with the charge and discharge operation """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)
        if not pyM.hasTSA:
            def cyclicState(pyM, loc, compName):
                return SOC[loc, compName, 0, 0] == SOC[loc, compName, 0, esM._timeStepsPerPeriod[-1] + 1]
        else:
            SOCInter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)
            def cyclicState(pyM, loc, compName):
                return SOCInter[loc, compName, 0] == SOCInter[loc, compName, esM._interPeriodTimeSteps[-1]]
        setattr(pyM, 'ConstrCyclicState_' + abbrvName, pyomo.Constraint(opVarSet, rule=cyclicState))

    def cyclicLifetime(self, pyM, esM):
        """ Constraint for limiting the number of full cycle equivalents to stay below cyclic lifetime """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        chargeOp, capVar = getattr(pyM, 'chargeOp_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        capVarSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        def cyclicLifetime(pyM, loc, compName):
            return (sum(chargeOp[loc, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet) /
                    esM._numberOfYears <= capVar[loc, compName] *
                    (compDict[compName]._stateOfChargeMax - compDict[compName]._stateOfChargeMin) *
                    compDict[compName]._cyclicLifetime / compDict[compName]._economicLifetime[loc]
                    if compDict[compName]._cyclicLifetime is not None else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrCyclicLifetime_' + abbrvName, pyomo.Constraint(capVarSet, rule=cyclicLifetime))

    def connectInterPeriodSOC(self, pyM, esM):
        """
        The state of charge at the end of each period is equivalent to the state of charge of the period
        before it (minus its self discharge) plus the change in the state of charge which happened during
        the typical period which was assigned to that period
        """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)
        SOCInter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)

        def connectInterSOC(pyM, loc, compName, pInter):
            return SOCInter[loc, compName, pInter + 1] == \
                   SOCInter[loc, compName, pInter] * (1 - compDict[compName]._selfDischarge) ** \
                   ((esM._timeStepsPerPeriod[-1] + 1) * esM._hoursPerTimeStep) + \
                   SOC[loc, compName, esM._periodsOrder[pInter], esM._timeStepsPerPeriod[-1] + 1]
        setattr(pyM, 'ConstrInterSOC_' + abbrvName, pyomo.Constraint(opVarSet, esM._periods, rule=connectInterSOC))

    def intraSOCstart(self, pyM, esM):
        """ The (virtual) state of charge at the beginning of a typical period is zero """
        abbrvName = self._abbrvName
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)

        def intraSOCstart(pyM, loc, compName, p):
            return SOC[loc, compName, p, 0] == 0
        setattr(pyM, 'ConstrSOCPeriodStart_' + abbrvName,
                pyomo.Constraint(opVarSet, esM._typicalPeriods, rule=intraSOCstart))

    def equalInterSOC(self, pyM, esM):
        """ If periodic storage is selected, the states of charge between periods have the same value """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)
        SOCInter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)

        def equalInterSOC(pyM, loc, compName, pInter):
            return (SOCInter[loc, compName, pInter] == SOCInter[loc, compName, pInter + 1]
                    if compDict[compName]._isPeriodicalStorage else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrEqualInterSOC_' + abbrvName, pyomo.Constraint(opVarSet, esM._periods, rule=equalInterSOC))

    def minSOC(self, pyM):
        """
        The state of charge [energyUnit] has to be larger than the installed capacity [energyUnit] multiplied
        with the relative minimum state of charge
        """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        capVarSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)
        SOC, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)

        def SOCMin(pyM, loc, compName, p, t):
            return SOC[loc, compName, p, t] >= capVar[loc, compName] * compDict[compName]._stateOfChargeMin
        setattr(pyM, 'ConstrSOCMin_' + abbrvName, pyomo.Constraint(capVarSet, pyM.timeSet, rule=SOCMin))

    def limitSOCwithSimpleTsa(self, pyM, esM):
        """
        Simplified version of the state of charge limitation control.
        The error compared to the precise version is small in cases of small selfDischarge.
        """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        capVarSimpleSet = getattr(pyM, 'designDimensionVarSetSimple_' + abbrvName)
        SOC, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        SOCmax, SOCmin = getattr(pyM, 'stateOfChargeMax_' + abbrvName), getattr(pyM, 'stateOfChargeMin_' + abbrvName)
        SOCInter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)

        # The maximum (virtual) state of charge during a typical period is larger than all occurring (virtual)
        # states of charge in that period (the last time step is considered in the subsequent period for t=0)
        def SOCintraPeriodMax(pyM, loc, compName, p, t):
            return SOC[loc, compName, p, t] <= SOCmax[loc, compName, p]
        setattr(pyM, 'ConstSOCintraPeriodMax_' + abbrvName,
                pyomo.Constraint(capVarSimpleSet, pyM.timeSet, rule=SOCintraPeriodMax))

        # The minimum (virtual) state of charge during a typical period is smaller than all occurring (virtual)
        # states of charge in that period (the last time step is considered in the subsequent period for t=0)
        def SOCintraPeriodMin(pyM, loc, compName, p, t):
            return SOC[loc, compName, p, t] >= SOCmin[loc, compName, p]
        setattr(pyM, 'ConstSOCintraPeriodMin_' + abbrvName,
                pyomo.Constraint(capVarSimpleSet, pyM.timeSet, rule=SOCintraPeriodMin))

        # The state of charge at the beginning of one period plus the maximum (virtual) state of charge
        # during that period has to be smaller than the installed capacities multiplied with the relative maximum
        # state of charge
        def SOCMaxSimple(pyM, loc, compName, pInter):
            return (SOCInter[loc, compName, pInter] + SOCmax[loc, compName, esM._periodsOrder[pInter]]
                    <= capVar[loc, compName] * compDict[compName]._stateOfChargeMax)
        setattr(pyM, 'ConstrSOCMaxSimple_' + abbrvName,
                pyomo.Constraint(capVarSimpleSet, esM._periods, rule=SOCMaxSimple))

        # The state of charge at the beginning of one period plus the minimum (virtual) state of charge
        # during that period has to be larger than the installed capacities multiplied with the relative minimum
        # state of charge
        def SOCMinSimple(pyM, loc, compName, pInter):
            return (SOCInter[loc, compName, pInter] * (1 - compDict[compName]._selfDischarge) **
                    ((esM._timeStepsPerPeriod[-1] + 1) * esM._hoursPerTimeStep)
                    + SOCmin[loc, compName, esM._periodsOrder[pInter]]
                    >= capVar[loc, compName] * compDict[compName]._stateOfChargeMin)
        setattr(pyM, 'ConstrSOCMinSimple_' + abbrvName,
                pyomo.Constraint(capVarSimpleSet, esM._periods, rule=SOCMinSimple))

    def operationModeSOC(self, pyM, esM):
        """
        State of charge [energyUnit] limited by the installed capacity [powerUnit] and the relative maximum
        state of charge
        """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVar, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        def op(pyM, loc, compName, p, t):
            return (opVar[loc, compName, p, t] <=
                    esM._hoursPerTimeStep * compDict[compName]._stateOfChargeMax * capVar[loc, compName])
        setattr(pyM, 'ConstrSOCMaxPrecise_' + abbrvName, pyomo.Constraint(constrSet, pyM.timeSet, rule=op))

    def operationModeSOCwithTSA(self, pyM, esM):
        """
        State of charge [energyUnit] limited by the installed capacity [powerUnit] and the relative maximum
        state of charge
        """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        SOCinter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)
        SOC, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        def SOCMaxPrecise(pyM, loc, compName, pInter, t):
            if compDict[compName]._doPreciseTsaModeling:
                return (SOCinter[loc, compName, pInter] *
                        ((1 - compDict[compName]._selfDischarge) ** (t * esM._hoursPerTimeStep)) +
                        SOC[loc, compName, esM._periodsOrder[pInter], t]
                        <= capVar[loc, compName] * compDict[compName]._stateOfChargeMax)
            else:
                return pyomo.Constraint.Skip
        setattr(pyM, 'ConstrSOCMaxPrecise_' + abbrvName,
                pyomo.Constraint(constrSet, esM._periods, esM._timeStepsPerPeriod, rule=SOCMaxPrecise))

    def minSOCwithTSAprecise(self, pyM, esM):
        """
        The state of charge at each time step cannot be smaller than the installed capacity multiplied with the
        relative minimum state of charge
        """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        SOCinter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)
        SOC, capVar = getattr(pyM, 'stateOfCharge_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        capVarPreciseSet = getattr(pyM, 'designDimensionVarSetPrecise_' + abbrvName)

        def SOCMinPrecise(pyM, loc, compName, pInter, t):
            return (SOCinter[loc, compName, pInter] * ((1 - compDict[compName]._selfDischarge) **
                    (t * esM._hoursPerTimeStep)) + SOC[loc, compName, esM._periodsOrder[pInter], t]
                    >= capVar[loc, compName] * compDict[compName]._stateOfChargeMin)
        setattr(pyM, 'ConstrSOCMinPrecise_' + abbrvName,
                pyomo.Constraint(capVarPreciseSet, esM._periods, esM._timeStepsPerPeriod, rule=SOCMinPrecise))

    def declareComponentConstraints(self, esM, pyM):
        """ Declares time independent and dependent constraints"""

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

        # Charging of storage [energyUnit] limited by the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging rate factor [powerUnit/energyUnit]
        self.operationMode1(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp', '_chargeRate')
        # Charging of storage [energyUnit] limited by the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging operation time series [powerUnit/energyUnit]
        self.operationMode2(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp')
        # Charging of storage [energyUnit] equal to the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging operation time series [powerUnit/energyUnit]
        self.operationMode3(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp')
        # Operation [energyUnit] limited by the operation time series [energyUnit]
        self.operationMode4(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp')
        # Operation [energyUnit] equal to the operation time series [energyUnit]
        self.operationMode5(pyM, esM, 'ConstrCharge', 'chargeOpConstrSet', 'chargeOp')

        #                             Constraints for enforcing discharging operation modes                            #

        # Discharging of storage [energyUnit] limited by the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the discharging rate factor [powerUnit/energyUnit]
        self.operationMode1(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp', '_dischargeRate')
        # Discharging of storage [energyUnit] limited by the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging operation time series [powerUnit/energyUnit]
        self.operationMode2(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp')
        # Discharging of storage [energyUnit] equal to the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging operation time series [powerUnit/energyUnit]
        self.operationMode3(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp')
        # Operation [energyUnit] limited by the operation time series [energyUnit]
        self.operationMode4(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp')
        # Operation [energyUnit] equal to the operation time series [energyUnit]
        self.operationMode5(pyM, esM, 'ConstrDischarge', 'dischargeOpConstrSet', 'dischargeOp')

        # Cyclic constraint enforcing that all storages have the same state of charge at the the beginning of the first
        # and the end of the last time step
        self.cyclicState(pyM, esM)

        # Constraint for limiting the number of full cycle equivalents to stay below cyclic lifetime
        self.cyclicLifetime(pyM, esM)

        if pyM.hasTSA:
            # The state of charge at the end of each period is equivalent to the state of charge of the period before it
            # (minus its self discharge) plus the change in the state of charge which happened during the typical
            # # period which was assigned to that period
            self.connectInterPeriodSOC(pyM, esM)
            # The (virtual) state of charge at the beginning of a typical period is zero
            self.intraSOCstart(pyM, esM)
            # If periodic storage is selected, the states of charge between periods have the same value
            self.equalInterSOC(pyM, esM)

        # Ensure that the state of charge is within the operating limits of the installed capacities
        if not pyM.hasTSA:
            #              Constraints for enforcing a state of charge operation mode within given limits              #

            # State of charge [energyUnit] limited by the installed capacity [energyUnit] and the relative maximum
            # state of charge
            self.operationModeSOC(pyM, esM)

            # The state of charge [energyUnit] has to be larger than the installed capacity [energyUnit] multiplied
            # with the relative minimum state of charge
            self.minSOC(pyM)

        else:
            #                       Simplified version of the state of charge limitation control                       #
            #           (The error compared to the precise version is small in cases of small selfDischarge)           #
            self.limitSOCwithSimpleTsa(pyM, esM)

            #                        Precise version of the state of charge limitation control                         #

            # Constraints for enforcing a state of charge operation within given limits

            # State of charge [energyUnit] limited by the installed capacity [energyUnit] and the relative maximum
            # state of charge
            self.operationModeSOCwithTSA(pyM, esM)

            # The state of charge at each time step cannot be smaller than the installed capacity multiplied with the
            # relative minimum state of charge
            self.minSOCwithTSAprecise(pyM, esM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        return super().getSharedPotentialContribution(pyM, key, loc)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([comp._commodity == commod and comp._locationalEligibility[loc] == 1
                    for comp in self._componentsDict.values()])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        compDict, abbrvName = self._componentsDict, self._abbrvName
        chargeOp, dischargeOp = getattr(pyM, 'chargeOp_' + abbrvName), getattr(pyM, 'dischargeOp_' + abbrvName)
        opVarDict = getattr(pyM, 'operationVarDict_' + abbrvName)
        return sum(dischargeOp[loc, compName, p, t] - chargeOp[loc, compName, p, t]
                   for compName in opVarDict[loc] if commod == self._componentsDict[compName]._commodity)

    def getObjectiveFunctionContribution(self, esM, pyM):

        capexCap = self.getEconomicsTI(pyM, ['_investPerCapacity'], 'cap', '_CCF')
        capexDec = self.getEconomicsTI(pyM, ['_investIfBuilt'], 'designBin', '_CCF')
        opexCap = self.getEconomicsTI(pyM, ['_opexPerCapacity'], 'cap')
        opexDec = self.getEconomicsTI(pyM, ['_opexIfBuilt'], 'designBin')
        opexOp1 = self.getEconomicsTD(pyM, esM, ['_opexPerChargeOperation'], 'chargeOp', 'operationVarDict')
        opexOp2 = self.getEconomicsTD(pyM, esM, ['_opexPerDischargeOperation'], 'dischargeOp', 'operationVarDict')

        return capexCap + capexDec + opexCap + opexDec + opexOp1 + opexOp2

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        compDict, abbrvName = self._componentsDict, self._abbrvName
        chargeOp, dischargeOp = getattr(pyM, 'chargeOp_' + abbrvName), getattr(pyM, 'dischargeOp_' + abbrvName)
        SOC = getattr(pyM, 'stateOfCharge_' + abbrvName)
        SOCinter = getattr(pyM, 'stateOfChargeInterPeriods_' + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(esM, pyM, esM._locations, '_commodityUnit', '*h')

        # Set optimal operation variables and append optimization summary
        props = ['operationCharge', 'operationDischarge', 'opexCharge', 'opexDischarge']
        units = ['[-]', '[-]', '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._commodityUnit + '*h/a]')
                          if x[1] == 'operationCharge' else x, tuples))
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._commodityUnit + '*h/a]')
                          if x[1] == 'operationDischarge' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM._locations)).sort_index()

        # * charge variables and contributions
        optVal = utils.formatOptimizationOutput(chargeOp.get_values(), 'operationVariables', '1dim', esM._periodsOrder)
        self._chargeOperationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_chargeOperationVariablesOptimum', compDict)

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name]._opexPerChargeOperation[op.index], axis=1)
            optSummary.loc[[(ix, 'operationCharge', '[' + compDict[ix]._commodityUnit + '*h/a]')
                             for ix in opSum.index], opSum.columns] = opSum.values/esM._numberOfYears
            optSummary.loc[[(ix, 'opexCharge', '[' + esM._costUnit + '/a]') for ix in ox.index],
                            ox.columns] = ox.values/esM._numberOfYears

        # * discharge variables and contributions
        optVal = utils.formatOptimizationOutput(dischargeOp.get_values(), 'operationVariables', '1dim',
                                                esM._periodsOrder)
        self._dischargeOperationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_dischargeOperationVariablesOptimum', compDict)

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name]._opexPerDischargeOperation[op.index], axis=1)
            optSummary.loc[[(ix, 'operationDischarge', '[' + compDict[ix]._commodityUnit + '*h/a]')
                             for ix in opSum.index], opSum.columns] = opSum.values/esM._numberOfYears
            optSummary.loc[[(ix, 'opexDischarge', '[' + esM._costUnit + '/a]') for ix in ox.index],
                            ox.columns] = ox.values/esM._numberOfYears

        # * set state of charge variables
        if not pyM.hasTSA:
            optVal = utils.formatOptimizationOutput(SOC.get_values(), 'operationVariables', '1dim', esM._periodsOrder)
            self._stateOfChargeOperationVariablesOptimum = optVal
            utils.setOptimalComponentVariables(optVal, '_stateOfChargeVariablesOptimum', compDict)
        else:
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
                for count, p in enumerate(esM._periodsOrder):
                    data.append((stateOfChargeInter.loc[:, count] +
                                 stateOfChargeIntra.loc[p].loc[:, :esM._timeStepsPerPeriod[-1]].T).T)
                optVal = pd.concat(data, axis=1, ignore_index=True)
            else:
                optVal = None
            self._stateOfChargeOperationVariablesOptimum = optVal
            utils.setOptimalComponentVariables(optVal, '_stateOfChargeVariablesOptimum', compDict)

        # Append optimization summaries
        optSummary = optSummary.append(optSummaryBasic).sort_index()

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'TAC') |
                           (optSummary.index.get_level_values(1) == 'opexCharge') |
                           (optSummary.index.get_level_values(1) == 'opexDischarge')].groupby(level=0).sum().values

        self._optSummary = optSummary

