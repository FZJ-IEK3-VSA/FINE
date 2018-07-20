from FINE.component import Component, ComponentModeling
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
                 stateOfChargeOpRateMax=None, stateOfChargeOpRateFix=None, stateOfChargeTsaWeight=1,
                 isPeriodicalStorage=False,
                 locationalEligibility=None, capacityMin=None, capacityMax=None, sharedPotentialID=None,
                 capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerChargeOperation=0,
                 opexPerDischargeOperation=0, opexPerCapacity=0, opexIfBuilt=0, interestRate=0.08, economicLifetime=10):
        # Set general component data
        utils.checkCommodities(esM, {commodity})
        self._name, self._commodity = name, commodity
        self._chargeRate, self._dischargeRate = chargeRate, dischargeRate
        self._chargeEfficiency, self._dischargeEfficiency = chargeEfficiency, dischargeEfficiency
        self._selfDischarge = selfDischarge
        self._cyclicLifetime = cyclicLifetime
        self._stateOfChargeMin, self._stateOfChargeMax = stateOfChargeMin, stateOfChargeMax
        self._isPeriodicalStorage = isPeriodicalStorage
        self._doPreciseTsaModeling = doPreciseTsaModeling

        # Set design variable modeling parameters
        utils.checkDesignVariableModelingParameters(capacityVariableDomain, hasCapacityVariable,
                                                    hasIsBuiltBinaryVariable, bigM)
        self._hasCapacityVariable = hasCapacityVariable
        self._hasCapacityVariable = hasCapacityVariable
        self._capacityVariableDomain = capacityVariableDomain
        self._capacityPerPlantUnit = capacityPerPlantUnit
        self._hasIsBuiltBinaryVariable = hasIsBuiltBinaryVariable
        self._bigM = bigM

        # Set economic data
        self._investPerCapacity = utils.checkAndSetCostParameter(esM, name, investPerCapacity)
        self._investIfBuilt = utils.checkAndSetCostParameter(esM, name, investIfBuilt)
        self._opexPerChargeOperation = utils.checkAndSetCostParameter(esM, name, opexPerChargeOperation)
        self._opexPerDischargeOperation = utils.checkAndSetCostParameter(esM, name, opexPerDischargeOperation)
        self._opexPerCapacity = utils.checkAndSetCostParameter(esM, name, opexPerCapacity)
        self._opexIfBuilt = utils.checkAndSetCostParameter(esM, name, opexIfBuilt)
        self._interestRate = utils.checkAndSetCostParameter(esM, name, interestRate)
        self._economicLifetime = utils.checkAndSetCostParameter(esM, name, economicLifetime)
        self._CCF = self.getCapitalChargeFactor()

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

        self._dischargeTsaWeight = dischargeTsaWeight

        # TODO describe that the state of charge refers to the SOC before the i-th time step
        if stateOfChargeOpRateMax is not None and stateOfChargeOpRateFix is not None:
            stateOfChargeOpRateMax = None
            warnings.warn('If stateOfChargeOpRateFix is specified, the stateOfChargeOpRateMax parameter is not +'
                          'required.\nThe stateOfChargeOpRateMax time series was set to None.')
        if (stateOfChargeOpRateMax is not None or stateOfChargeOpRateFix is not None) and not doPreciseTsaModeling:
            self._doPreciseTsaModeling = True
            warnings.warn('Warning only relevant when time series aggregation is used in optimization:\n' +
                          'If stateOfChargeOpRateFix or the stateOfChargeOpRateMax parameter are specified,\n' +
                          'the modeling is set to precise.')
        if stateOfChargeOpRateMax is not None:
            warnings.warn('Warning only relevant when time series aggregation is used in optimization:\n' +
                          'Setting the stateOfChargeOpRateMax parameter might lead to unwanted modeling behavior\n' +
                          'and should be handled with caution.')
        if stateOfChargeOpRateFix is not None and not isPeriodicalStorage:
            self._isPeriodicalStorage = True
            warnings.warn('Warning only relevant when time series aggregation is used in optimization:\n' +
                          'If the stateOfChargeOpRateFix parameter is specified, the storage\n' +
                          'is set to isPeriodicalStorage).')
        utils.checkOperationTimeSeriesInputParameters(esM, stateOfChargeOpRateMax, locationalEligibility)
        utils.checkOperationTimeSeriesInputParameters(esM, stateOfChargeOpRateFix, locationalEligibility)

        self._fullStateOfChargeOpRateMax = utils.setFormattedTimeSeries(stateOfChargeOpRateMax)
        self._aggregatedStateOfChargeOpRateMax = None
        self._stateOfChargeOpRateMax = None

        self._fullStateOfChargeOpRateFix = utils.setFormattedTimeSeries(stateOfChargeOpRateFix)
        self._aggregatedStateOfChargeOpRateFix = None
        self._stateOfChargeOpRateFix = None

        self._stateOfChargeTsaWeight = stateOfChargeTsaWeight

        # Set location-specific design parameters
        self._sharedPotentialID = sharedPotentialID
        utils.checkLocationSpecficDesignInputParams(esM, hasCapacityVariable, hasIsBuiltBinaryVariable,
                                                    capacityMin, capacityMax, capacityFix,
                                                    locationalEligibility, isBuiltFix, sharedPotentialID,
                                                    dimension='1dim')
        self._capacityMin, self._capacityMax, self._capacityFix = capacityMin, capacityMax, capacityFix
        self._isBuiltFix = isBuiltFix

        # Set locational eligibility
        timeSeriesData = None
        tsNb = sum([0 if data is None else 1 for data in [chargeOpRateMax, chargeOpRateFix, dischargeOpRateMax,
                    dischargeOpRateFix, stateOfChargeOpRateMax, stateOfChargeOpRateFix]])
        if tsNb > 0:
            timeSeriesData = sum([data for data in [chargeOpRateMax, chargeOpRateFix, dischargeOpRateMax,
                                  dischargeOpRateFix, stateOfChargeOpRateMax, stateOfChargeOpRateFix]
                                  if data is not None])
        self._locationalEligibility = utils.setLocationalEligibility(esM, locationalEligibility, capacityMax,
                                                                     capacityFix, isBuiltFix,
                                                                     hasCapacityVariable, timeSeriesData)

        # Variables at optimum (set after optimization)
        self._designDimensionVariablesOptimum = None
        self._designDecisionVariablesOptimum = None
        self._chargeOperationVariablesOptimum = None
        self._dischargeOperationVariablesOptimum = None
        self._stateOfChargeVariablesOptimum = None

    def getCapitalChargeFactor(self):
        """ Computes and returns capital charge factor (inverse of annuity factor) """
        return 1 / self._interestRate - 1 / (pow(1 + self._interestRate, self._economicLifetime) * self._interestRate)

    def addToEnergySystemModel(self, esM):
        esM._isTimeSeriesDataClustered = False
        if self._name in esM._componentNames:
            if esM._componentNames[self._name] == StorageModeling.__name__:
                warnings.warn(
                    'Component identifier ' + self._name + ' already exists. Data will be overwritten.')
            else:
                raise ValueError('Component name ' + self._name + ' is not unique.')
        else:
            esM._componentNames.update({self._name: StorageModeling.__name__})
        mdl = StorageModeling.__name__
        if mdl not in esM._componentModelingDict:
            esM._componentModelingDict.update({mdl: StorageModeling()})
        esM._componentModelingDict[mdl]._componentsDict.update({self._name: self})

    def setTimeSeriesData(self, hasTSA):
        self._chargeOpRateMax = self._aggregatedChargeOpRateMax if hasTSA else self._fullChargeOpRateMax
        self._chargeOpRateFix = self._aggregatedChargeOpRateFix if hasTSA else self._fullChargeOpRateFix
        self._dischargeOpRateMax = self._aggregatedChargeOpRateMax if hasTSA else self._fullDischargeOpRateMax
        self._dischargeOpRateFix = self._aggregatedChargeOpRateFix if hasTSA else self._fullDischargeOpRateFix
        self._stateOfChargeOpRateMax = self._aggregatedStateOfChargeOpRateMax if hasTSA \
            else self._fullStateOfChargeOpRateMax
        self._stateOfChargeOpRateFix = self._aggregatedStateOfChargeOpRateFix if hasTSA \
            else self._fullStateOfChargeOpRateFix

    def getDataForTimeSeriesAggregation(self):
        compDict = {}
        data = []

        dataC = self._fullChargeOpRateFix if self._fullChargeOpRateFix is not None else self._fullChargeOpRateMax
        if dataC is not None:
            dataC = dataC.copy()
            uniqueIdentifiers = [self._name + "_chargeOpRate_" + loc for loc in dataC.columns]
            dataC.rename(columns={loc: self._name + "_chargeOpRate_" + loc for loc in dataC.columns}, inplace=True)
            compDict.update({id: self._chargeTsaWeight for id in uniqueIdentifiers}), data.append(dataC)

        dataD = self._fullDischargeOpRateFix if self._fullDischargeOpRateFix is not None \
            else self._fullDischargeOpRateMax
        if dataD is not None:
            dataD = dataD.copy()
            uniqueIdentifiers = [self._name + "_dischargeOpRate_" + loc for loc in dataD.columns]
            dataD.rename(columns={loc: self._name + "_dischargeOpRate_" + loc for loc in dataD.columns}, inplace=True)
            compDict.update({id: self._chargeTsaWeight for id in uniqueIdentifiers}), data.append(dataD)

        dataSOC = self._stateOfChargeOpRateFix if self._stateOfChargeOpRateFix is not None \
            else self._stateOfChargeOpRateMax
        if dataSOC is not None:
            dataSOC = dataSOC.copy()
            uniqueIdentifiers = [self._name + "_SOCOpRate_" + loc for loc in dataD.columns]
            dataSOC.rename(columns={loc: self._name + "_SOCOpRate_" + loc for loc in dataSOC.columns}, inplace=True)
            compDict.update({id: self._chargeTsaWeight for id in uniqueIdentifiers}), data.append(dataSOC)

        return (pd.concat(data, axis=1), compDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):

        fullOperationRate = self._fullChargeOpRateFix if self._fullChargeOpRateFix is not None\
            else self._fullChargeOpRateMax
        if fullOperationRate is not None:
            uniqueIdentifiers = [self._name + "_chargeOpRate_" + loc for loc in fullOperationRate.columns]
            dataC = data[uniqueIdentifiers].copy()
            dataC.rename(columns={self._name + "_chargeOpRate_" + loc: loc for loc in dataC.columns}, inplace=True)
            if self._fullChargeOpRateFix is not None:
                self._aggregatedChargeOpRateFix = dataC
            else:
                self._aggregatedChargeOpRateMax = dataC

        fullOperationRate = self._fullDischargeOpRateFix if self._fullDischargeOpRateFix is not None \
            else self._fullDischargeOpRateMax
        if fullOperationRate is not None:
            uniqueIdentifiers = [self._name + "_dischargeOpRate_" + loc for loc in fullOperationRate.columns]
            dataD = data[uniqueIdentifiers].copy()
            dataD.rename(columns={self._name + "_dischargeOpRate_" + loc: loc for loc in dataD.columns}, inplace=True)
            if self._fullDischargeOpRateFix is not None:
                self._aggregatedDischargeOpRateFix = dataD
            else:
                self._aggregatedDischargeOpRateMax = dataD

        fullOperationRate = self._stateOfChargeOpRateFix if self._stateOfChargeOpRateFix is not None \
            else self._stateOfChargeOpRateMax
        if fullOperationRate is not None:
            uniqueIdentifiers = [self._name + "_SOCOpRate_" + loc for loc in fullOperationRate.columns]
            dataSOC = data[uniqueIdentifiers].copy()
            dataSOC.rename(columns={self._name + "_SOCOpRate_" + loc: loc for loc in dataSOC.columns}, inplace=True)
            if self._fullStateOfChargeOpRateFix is not None:
                self._aggregatedStateOfChargeOpRateFix = dataSOC
            else:
                self._aggregatedStateOfChargeOpRateMax = dataSOC


class StorageModeling(ComponentModeling):
    """ Doc """

    def __init__(self):
        self._componentsDict = {}
        self._designDimensionVariablesOptimum = None
        self._designDecisionVariablesOptimum = None
        self._chargeOperationVariablesOptimum = None
        self._dischargeOperationVariablesOptimum = None
        self._stateOfChargeVariablesOptimum = None


    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """ Declares sets and dictionaries """
        compDict = self._componentsDict

        ################################################################################################################
        #                                        Declare design variables sets                                         #
        ################################################################################################################

        def initDesignVarSet(pyM):
            return ((loc, compName) for loc in esM._locations for compName, comp in compDict.items()
                    if comp._locationalEligibility[loc] == 1 and comp._hasCapacityVariable)
        pyM.designDimensionVarSet_stor = pyomo.Set(dimen=2, initialize=initDesignVarSet)

        if pyM.hasTSA:
            def initDesignVarSimpleTSASet(pyM):
                return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_stor
                        if not compDict[compName]._doPreciseTsaModeling)
            pyM.designDimensionVarSetSimple_stor = pyomo.Set(dimen=2, initialize=initDesignVarSimpleTSASet)

            def initDesignVarPreciseTSASet(pyM):
                return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_stor
                        if compDict[compName]._doPreciseTsaModeling)
            pyM.designDimensionVarSetPrecise_stor = pyomo.Set(dimen=2, initialize=initDesignVarPreciseTSASet)

        def initContinuousDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_stor
                    if compDict[compName]._capacityVariableDomain == 'continuous')
        pyM.continuousDesignDimensionVarSet_stor = pyomo.Set(dimen=2, initialize=initContinuousDesignVarSet)

        def initDiscreteDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_stor
                    if compDict[compName]._capacityVariableDomain == 'discrete')
        pyM.discreteDesignDimensionVarSet_stor = pyomo.Set(dimen=2, initialize=initDiscreteDesignVarSet)

        def initDesignDecisionVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_stor
                    if compDict[compName]._hasIsBuiltBinaryVariable)
        pyM.designDecisionVarSet_stor = pyomo.Set(dimen=2, initialize=initDesignDecisionVarSet)

        ################################################################################################################
        #                                     Declare operation variables sets                                         #
        ################################################################################################################

        def initOpVarSet(pyM):
            return ((loc, compName) for loc in esM._locations for compName, comp in compDict.items()
                    if comp._locationalEligibility[loc] == 1)
        pyM.operationVarSet_stor = pyomo.Set(dimen=2, initialize=initOpVarSet)
        pyM.operationVarDict_stor = {loc: {compName for compName in compDict if (loc, compName)
                                           in pyM.operationVarSet_stor} for loc in esM._locations}

        ################################################################################################################
        #                           Declare sets for case differentiation of operating modes                           #
        ################################################################################################################

        # Charge operation
        def initChargeOpConstrSet1(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._chargeOpRateMax is None
                    and compDict[compName]._chargeOpRateFix is None)
        pyM.chargeOpConstrSet1_stor = pyomo.Set(dimen=2, initialize=initChargeOpConstrSet1)

        def initChargeOpConstrSet2(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._chargeOpRateFix is not None)
        pyM.chargeOpConstrSet2_stor = pyomo.Set(dimen=2, initialize=initChargeOpConstrSet2)

        def initChargeOpConstrSet3(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._chargeOpRateMax is not None)
        pyM.chargeOpConstrSet3_stor = pyomo.Set(dimen=2, initialize=initChargeOpConstrSet3)

        def initChargeOpConstrSet4(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    not compDict[compName]._hasCapacityVariable and
                    compDict[compName]._chargeOpRateFix is not None)
        pyM.chargeOpConstrSet4_stor = pyomo.Set(dimen=2, initialize=initChargeOpConstrSet4)

        def initChargeOpConstrSet5(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    not compDict[compName]._hasCapacityVariable and
                    compDict[compName]._chargeOpRateFix is not None)
        pyM.chargeOpConstrSet5_stor = pyomo.Set(dimen=2, initialize=initChargeOpConstrSet5)

        # Discharge operation
        def initDischargeOpConstrSet1(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._dischargeOpRateMax is None
                    and compDict[compName]._dischargeOpRateFix is None)
        pyM.dischargeOpConstrSet1_stor = pyomo.Set(dimen=2, initialize=initDischargeOpConstrSet1)

        def initDischargeOpConstrSet2(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and
                    compDict[compName]._dischargeOpRateFix is not None)
        pyM.dischargeOpConstrSet2_stor = pyomo.Set(dimen=2, initialize=initDischargeOpConstrSet2)

        def initDischargeOpConstrSet3(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and
                    compDict[compName]._dischargeOpRateMax is not None)
        pyM.dischargeOpConstrSet3_stor = pyomo.Set(dimen=2, initialize=initDischargeOpConstrSet3)

        def initDischargeOpConstrSet4(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    not compDict[compName]._hasCapacityVariable and
                    compDict[compName]._dischargeOpRateFix is not None)
        pyM.dischargeOpConstrSet4_stor = pyomo.Set(dimen=2, initialize=initDischargeOpConstrSet4)

        def initDischargeOpConstrSet5(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    not compDict[compName]._hasCapacityVariable and
                    compDict[compName]._dischargeOpRateMax is not None)
        pyM.dischargeOpConstrSet5_stor = pyomo.Set(dimen=2, initialize=initDischargeOpConstrSet5)

        # State of charge operation
        # TODO check if also applied for simple SOC modeling
        def initStateOfChargeOpConstrSet1(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and
                    compDict[compName]._stateOfChargeOpRateMax is None
                    and compDict[compName]._stateOfChargeOpRateFix is None
                    and (compDict[compName]._doPreciseTsaModeling if pyM.hasTSA else True))
        pyM.stateOfChargeOpConstrSet1_stor = pyomo.Set(dimen=2, initialize=initStateOfChargeOpConstrSet1)

        def initStateOfChargeOpConstrSet2(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and
                    compDict[compName]._stateOfChargeOpRateFix is not None
                    and (compDict[compName]._doPreciseTsaModeling if pyM.hasTSA else True))
        pyM.stateOfChargeOpConstrSet2_stor = pyomo.Set(dimen=2, initialize=initStateOfChargeOpConstrSet2)

        def initStateOfChargeOpConstrSet3(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    compDict[compName]._hasCapacityVariable and
                    compDict[compName]._stateOfChargeOpRateMax is not None
                    and (compDict[compName]._doPreciseTsaModeling if pyM.hasTSA else True))
        pyM.stateOfChargeOpConstrSet3_stor = pyomo.Set(dimen=2, initialize=initStateOfChargeOpConstrSet3)

        def initStateOfChargeOpConstrSet4(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    not compDict[compName]._hasCapacityVariable and
                    compDict[compName]._stateOfChargeOpRateFix is not None
                    and (compDict[compName]._doPreciseTsaModeling if pyM.hasTSA else True))
        pyM.stateOfChargeOpConstrSet4_stor = pyomo.Set(dimen=2, initialize=initStateOfChargeOpConstrSet4)

        def initStateOfChargeOpConstrSet5(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_stor if
                    not compDict[compName]._hasCapacityVariable and
                    compDict[compName]._stateOfChargeOpRateMax is not None
                    and (compDict[compName]._doPreciseTsaModeling if pyM.hasTSA else True))
        pyM.stateOfChargeOpConstrSet5_stor = pyomo.Set(dimen=2, initialize=initStateOfChargeOpConstrSet5)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """ Declares design and operation variables """

        # Function for setting lower and upper capacity bounds
        def capBounds(pyM, loc, compName):
            comp = self._componentsDict[compName]
            return (comp._capacityMin[loc] if (comp._capacityMin is not None and not comp._hasIsBuiltBinaryVariable)
                    else 0, comp._capacityMax[loc] if comp._capacityMax is not None else None)

        # Capacity of components [energyUnit]
        pyM.cap_stor = pyomo.Var(pyM.designDimensionVarSet_stor, domain=pyomo.NonNegativeReals, bounds=capBounds)
        # Number of components [-]
        pyM.nbReal_stor = pyomo.Var(pyM.continuousDesignDimensionVarSet_stor, domain=pyomo.NonNegativeReals)
        # Number of components [-]
        pyM.nbInt_stor = pyomo.Var(pyM.discreteDesignDimensionVarSet_stor, domain=pyomo.NonNegativeIntegers)
        # Binary variables [-], indicate if a component is considered at a location or not
        pyM.designBin_stor = pyomo.Var(pyM.designDecisionVarSet_stor, domain=pyomo.Binary)
        # Energy amount injected into a storage (before injection efficiency losses) between two time steps
        pyM.chargeOp = pyomo.Var(pyM.designDimensionVarSet_stor, pyM.timeSet, domain=pyomo.NonNegativeReals)
        # Energy amount delivered from a storage (after delivery efficiency losses) between two time steps
        pyM.dischargeOp = pyomo.Var(pyM.designDimensionVarSet_stor, pyM.timeSet, domain=pyomo.NonNegativeReals)

        # Inventory of storage components [energyUnit]
        if not pyM.hasTSA:
            # Energy amount stored at the beginning of a time step during the (one) period (the i-th state of charge
            # refers to the state of charge at the beginning of the i-th time step, the last index is the state of
            # charge after the last time step)
            pyM.stateOfCharge = pyomo.Var(pyM.designDimensionVarSet_stor, pyM.interTimeStepsSet,
                                          domain=pyomo.NonNegativeReals)
        else:
            # (Virtual) energy amount stored during a period (the i-th state of charge refers to the state of charge at
            # the beginning of the i-th time step, the last index is the state of charge after the last time step)
            pyM.stateOfCharge = pyomo.Var(pyM.designDimensionVarSet_stor, pyM.interTimeStepsSet,
                                          domain=pyomo.Reals)
            # (Virtual) minimum amount of energy stored within a period
            pyM.stateOfChargeMin = pyomo.Var(pyM.designDimensionVarSetSimple_stor, esM._typicalPeriods,
                                             domain=pyomo.Reals)
            # (Virtual) maximum amount of energy stored within a period
            pyM.stateOfChargeMax = pyomo.Var(pyM.designDimensionVarSetSimple_stor, esM._typicalPeriods,
                                             domain=pyomo.Reals)
            # (Real) energy amount stored at the beginning of a period between periods(the i-th state of charge refers
            # to the state of charge at the beginning of the i-th period, the last index is the state of charge after
            # the last period)
            pyM.stateOfChargeInterPeriods = pyomo.Var(pyM.designDimensionVarSet_stor, esM._interPeriodTimeSteps,
                                                      domain=pyomo.NonNegativeReals)

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def declareComponentConstraints(self, esM, pyM):
        """ Declares time independent and dependent constraints"""
        compDict = self._componentsDict

        ################################################################################################################
        #                                    Declare time independent constraints                                      #
        ################################################################################################################

        # Determine the components' capacities from the number of installed units
        def capToNbReal_stor(pyM, loc, compName):
            return pyM.cap_stor[loc, compName] == pyM.nbReal_stor[loc, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbReal_stor = pyomo.Constraint(pyM.continuousDesignDimensionVarSet_stor, rule=capToNbReal_stor)

        # Determine the components' capacities from the number of installed units
        def capToNbInt_stor(pyM, loc, compName):
            return pyM.cap_stor[loc, compName] == pyM.nbInt_stor[loc, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbInt_stor = pyomo.Constraint(pyM.discreteDesignDimensionVarSet_stor, rule=capToNbInt_stor)

        # Enforce the consideration of the binary design variables of a component
        def bigM_stor(pyM, loc, compName):
            return pyM.cap_stor[loc, compName] <= pyM.designBin_stor[loc, compName] * compDict[compName]._bigM
        pyM.ConstrBigM_stor = pyomo.Constraint(pyM.designDecisionVarSet_stor, rule=bigM_stor)

        # Enforce the consideration of minimum capacities for components with design decision variables
        def capacityMinDec_stor(pyM, loc, compName):
            return (pyM.cap_stor[loc, compName] >= compDict[compName]._capacityMin[loc] *
                    pyM.designBin_stor[loc, compName] if compDict[compName]._capacityMin is not None
                    else pyomo.Constraint.Skip)
        pyM.ConstrCapacityMinDec_stor = pyomo.Constraint(pyM.designDecisionVarSet_stor, rule=capacityMinDec_stor)

        # Sets, if applicable, the installed capacities of a component
        def capacityFix_stor(pyM, loc, compName):
            return pyM.cap_stor[loc, compName] == compDict[compName]._capacityFix[loc] \
                if compDict[compName]._capacityFix is not None else pyomo.Constraint.Skip
        pyM.ConstrCapacityFix_stor = pyomo.Constraint(pyM.designDimensionVarSet_stor, rule=capacityFix_stor)

        # Sets, if applicable, the binary design variables of a component
        def designBinFix_stor(pyM, loc, compName):
            return pyM.designBin_stor[loc, compName] == compDict[compName]._isBuiltFix[loc] \
                if compDict[compName]._isBuiltFix is not None else pyomo.Constraint.Skip
        pyM.ConstrDesignBinFix_stor = pyomo.Constraint(pyM.designDecisionVarSet_stor, rule=designBinFix_stor)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Constraint for connecting the state of charge with the charge and discharge operation
        def trans_stor(pyM, loc, compName, p, t):
            return (pyM.stateOfCharge[loc, compName, p, t+1] - pyM.stateOfCharge[loc, compName, p, t] *
                    (1 - compDict[compName]._selfDischarge) ** esM._hoursPerTimeStep ==
                    pyM.chargeOp[loc, compName, p, t] * compDict[compName]._chargeEfficiency -
                    pyM.dischargeOp[loc, compName, p, t] / compDict[compName]._dischargeEfficiency)
        pyM.ConstrTrans_stor = pyomo.Constraint(pyM.operationVarSet_stor, pyM.timeSet, rule=trans_stor)

        #                              Constraints for enforcing charging operation modes                              #

        # Charging of storage [energyUnit] limited by the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging rate factor [powerUnit/energyUnit]
        def charge1_stor(pyM, loc, compName, p, t):
            return pyM.chargeOp[loc, compName, p, t] <= \
                   compDict[compName]._chargeRate * esM._hoursPerTimeStep * pyM.cap_stor[loc, compName]
        pyM.ConstrCharge1_stor = pyomo.Constraint(pyM.chargeOpConstrSet1_stor, pyM.timeSet, rule=charge1_stor)

        # Charging of storage [energyUnit] limited by the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging operation time series [powerUnit/energyUnit]
        def charge2_stor(pyM, loc, compName, p, t):
            return pyM.chargeOp[loc, compName, p, t] == esM._hoursPerTimeStep * \
                   compDict[compName]._chargeOpRateFix[loc][p, t] * pyM.cap_stor[loc, compName]
        pyM.ConstrCharge2_stor = pyomo.Constraint(pyM.chargeOpConstrSet2_stor, pyM.timeSet, rule=charge2_stor)

        # Charging of storage [energyUnit] equal to the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging operation time series [powerUnit/energyUnit]
        def charge3_stor(pyM, loc, compName, p, t):
            return pyM.chargeOp[loc, compName, p, t] <= esM._hoursPerTimeStep * \
                   compDict[compName]._chargeOpRateMax[loc][p, t] * pyM.cap_stor[loc, compName]
        pyM.ConstrCharge3_stor = pyomo.Constraint(pyM.chargeOpConstrSet3_stor, pyM.timeSet, rule=charge3_stor)

        # Operation [energyUnit] limited by the operation time series [energyUnit]
        def charge4_stor(pyM, loc, compName, p, t):
            return pyM.chargeOp[loc, compName, p, t] == compDict[compName]._chargeOpRateFix[loc][p, t]
        pyM.ConstrCharge4_stor = pyomo.Constraint(pyM.chargeOpConstrSet4_stor, pyM.timeSet, rule=charge4_stor)

        # Operation [energyUnit] equal to the operation time series [energyUnit]
        def charge5_stor(pyM, loc, compName, p, t):
            return pyM.chargeOp[loc, compName, p, t] <= compDict[compName]._chargeOpRateMax[loc][p, t]
        pyM.ConstrCharge5_stor = pyomo.Constraint(pyM.chargeOpConstrSet5_stor, pyM.timeSet, rule=charge5_stor)

        #                            Constraints for enforcing discharging operation modes                             #

        # Discharging of storage [energyUnit] limited by the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the discharging rate factor [powerUnit/energyUnit]
        def discharge1_stor(pyM, loc, compName, p, t):
            return pyM.dischargeOp[loc, compName, p, t] <= \
                   compDict[compName]._dischargeRate * esM._hoursPerTimeStep * pyM.cap_stor[loc, compName]
        pyM.ConstrDischarge1_stor = pyomo.Constraint(pyM.dischargeOpConstrSet1_stor, pyM.timeSet, rule=discharge1_stor)

        # Discharging of storage [energyUnit] limited by the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging operation time series [powerUnit/energyUnit]
        def discharge2_stor(pyM, loc, compName, p, t):
            return pyM.dischargeOp[loc, compName, p, t] == esM._hoursPerTimeStep * \
                   compDict[compName]._dischargeOpRateFix[loc][p, t] * pyM.cap_stor[loc, compName]
        pyM.ConstrDischarge2_stor = pyomo.Constraint(pyM.dischargeOpConstrSet2_stor, pyM.timeSet, rule=discharge2_stor)

        # Discharging of storage [energyUnit] equal to the installed capacity [energyUnit] multiplied by the hours per
        # time step [h] and the charging operation time series [powerUnit/energyUnit]
        def discharge3_stor(pyM, loc, compName, p, t):
            return pyM.dischargeOp[loc, compName, p, t] <= esM._hoursPerTimeStep * \
                   compDict[compName]._dischargeOpRateMax[loc][p, t] * pyM.cap_stor[loc, compName]
        pyM.ConstrDischarge3_stor = pyomo.Constraint(pyM.dischargeOpConstrSet3_stor, pyM.timeSet, rule=discharge3_stor)

        # Operation [energyUnit] limited by the operation time series [energyUnit]
        def discharge4_stor(pyM, loc, compName, p, t):
            return pyM.dischargeOp[loc, compName, p, t] == compDict[compName]._dischargeOpRateFix[loc][p, t]
        pyM.ConstrDischarge4_stor = pyomo.Constraint(pyM.dischargeOpConstrSet4_stor, pyM.timeSet, rule=discharge4_stor)

        # Operation [energyUnit] equal to the operation time series [energyUnit]
        def discharge5_stor(pyM, loc, compName, p, t):
            return pyM.dischargeOp[loc, compName, p, t] <= compDict[compName]._dischargeOpRateMax[loc][p, t]
        pyM.ConstrDischarge5_stor = pyomo.Constraint(pyM.dischargeOpConstrSet5_stor, pyM.timeSet, rule=discharge5_stor)

        # Cyclic constraint enforcing that all storages have the same state of charge at the the beginning of the first
        # and the end of the last time step
        if not pyM.hasTSA:
            def cyclicState_stor(pyM, loc, compName):
                return pyM.stateOfCharge[loc, compName, 0, 0] == \
                       pyM.stateOfCharge[loc, compName, 0, esM._timeStepsPerPeriod[-1] + 1]
        else:
            def cyclicState_stor(pyM, loc, compName):
                return pyM.stateOfChargeInterPeriods[loc, compName, 0] == \
                       pyM.stateOfChargeInterPeriods[loc, compName, esM._interPeriodTimeSteps[-1]]
        pyM.ConstrCyclicState_stor = pyomo.Constraint(pyM.operationVarSet_stor, rule=cyclicState_stor)

        # Constraint for limiting the number of full cycle equivalents to stay below cyclic lifetime
        def cyclicLifetime_stor(pyM, loc, compName):
            return (sum(pyM.chargeOp[loc, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet) /
                    esM._numberOfYears <= pyM.cap_stor[loc, compName] *
                    (compDict[compName]._stateOfChargeMax - compDict[compName]._stateOfChargeMin) *
                    compDict[compName]._cyclicLifetime / compDict[compName]._economicLifetime[loc]
                    if compDict[compName]._cyclicLifetime is not None else pyomo.Constraint.Skip)
        pyM.ConstrCyclicLifetime_stor = pyomo.Constraint(pyM.designDimensionVarSet_stor, rule=cyclicLifetime_stor)

        if pyM.hasTSA:
            # The state of charge at the end of each period is equivalent to the state of charge of the period before it
            # (minus its self discharge) plus the change in the state of charge which happened during the typical
            # period which was assigned to that period
            def interSOC_stor(pyM, loc, compName, pInter):
                return pyM.stateOfChargeInterPeriods[loc, compName, pInter + 1] == \
                    pyM.stateOfChargeInterPeriods[loc, compName, pInter] * \
                    (1-compDict[compName]._selfDischarge)**((esM._timeStepsPerPeriod[-1]+1) * esM._hoursPerTimeStep) + \
                    pyM.stateOfCharge[loc, compName, esM._periodsOrder[pInter], esM._timeStepsPerPeriod[-1]+1]
            pyM.ConstrInterSOC_stor = \
                pyomo.Constraint(pyM.operationVarSet_stor, esM._periods, rule=interSOC_stor)

            # The (virtual) state of charge at the beginning of a typical period is zero
            def SOCPeriodStart_stor(pyM, loc, compName, p):
                return pyM.stateOfCharge[loc, compName, p, 0] == 0
            pyM.ConstrSOCPeriodStart_stor = \
                pyomo.Constraint(pyM.operationVarSet_stor, esM._typicalPeriods, rule=SOCPeriodStart_stor)

            # If periodic storage is selected, the states of charge between periods have the same value
            def equalInterSOC_stor(pyM, loc, compName, pInter):
                if compDict[compName]._isPeriodicalStorage:
                    return pyM.stateOfChargeInterPeriods[loc, compName, pInter] == \
                           pyM.stateOfChargeInterPeriods[loc, compName, pInter + 1]
                else:
                    return pyomo.Constraint.Skip
            pyM.ConstrEqualInterSOC_stor = pyomo.Constraint(pyM.operationVarSet_stor,
                                                            esM._periods, rule=equalInterSOC_stor)

        # Ensure that the state of charge is within the operating limits of the installed capacities
        if not pyM.hasTSA:
            #              Constraints for enforcing a state of charge operation mode within given limits              #

            # State of charge [energyUnit] limited by the installed capacity [energyUnit] and the relative maximum
            # state of charge
            def SOCMax1_stor(pyM, loc, compName, p, t):
                return pyM.stateOfCharge[loc, compName, p, t] <= \
                       pyM.cap_stor[loc, compName] * compDict[compName]._stateOfChargeMax
            pyM.ConstrSOCMax1_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet1_stor, pyM.timeSet,
                                                      rule=SOCMax1_stor)

            # State of charge [energyUnit] equal to the installed capacity [energyUnit] multiplied by state of charge
            # time series [energyUnit/energyUnit]
            def SOCMax2_stor(pyM, loc, compName, p, t):
                return pyM.stateOfCharge[loc, compName, p, t] == pyM.cap_stor[loc, compName] * \
                       compDict[compName]._stateOfChargeOpRateFix[loc][p, t]
            pyM.ConstrSOCMax2_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet2_stor, pyM.timeSet,
                                                      rule=SOCMax2_stor)

            # State of charge [energyUnit] limited by the installed capacity [energyUnit] multiplied by state of charge
            # time series [energyUnit/energyUnit]
            def SOCMax3_stor(pyM, loc, compName, p, t):
                return pyM.stateOfCharge[loc, compName, p, t] <= pyM.cap_stor[loc, compName] * \
                       compDict[compName]._stateOfChargeOpRateMax[loc][p, t]
            pyM.ConstrSOCMax3_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet3_stor, pyM.timeSet,
                                                      rule=SOCMax3_stor)

            # Operation [energyUnit] equal to the operation time series [energyUnit]
            def SOCMax4_stor(pyM, loc, compName, p, t):
                return pyM.stateOfCharge[loc, compName, p, t] == compDict[compName]._stateOfChargeOpRateFix[loc][p, t]
            pyM.ConstrSOCMax4_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet4_stor, pyM.timeSet,
                                                      rule=SOCMax4_stor)

            # Operation [energyUnit] limited by the operation time series [energyUnit]
            def SOCMax5_stor(pyM, loc, compName, p, t):
                return pyM.stateOfCharge[loc, compName, p, t] <= compDict[compName]._stateOfChargeOpRateMax[loc][p, t]
            pyM.ConstrSOCMax5_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet5_stor, pyM.timeSet,
                                                      rule=SOCMax5_stor)

            # The state of charge [energyUnit] has to be larger than the installed capacity [energyUnit] multiplied
            # with the relative minimum state of charge
            def SOCMin_stor(pyM, loc, compName, p, t):
                return pyM.stateOfCharge[loc, compName, p, t] >= \
                       pyM.cap_stor[loc, compName] * compDict[compName]._stateOfChargeMin
            pyM.ConstrSOCMin_stor = pyomo.Constraint(pyM.designDimensionVarSet_stor, pyM.timeSet, rule=SOCMin_stor)

        else:
            #                       Simplified version of the state of charge limitation control                       #
            #           (The error compared to the precise version is small in cases of small selfDischarge)           #

            # The maximum (virtual) state of charge during a typical period is larger than all occurring (virtual)
            # states of charge in that period (the last time step is considered in the subsequent period for t=0)
            def SOCintraPeriodMax_stor(pyM, loc, compName, p, t):
                return pyM.stateOfCharge[loc, compName, p, t] <= pyM.stateOfChargeMax[loc, compName, p]
            pyM.ConstSOCintraPeriodMax_stor = \
                pyomo.Constraint(pyM.designDimensionVarSetSimple_stor, pyM.timeSet, rule=SOCintraPeriodMax_stor)

            # The minimum (virtual) state of charge during a typical period is smaller than all occurring (virtual)
            # states of charge in that period (the last time step is considered in the subsequent period for t=0)
            def SOCintraPeriodMin_stor(pyM, loc, compName, p, t):
                return pyM.stateOfCharge[loc, compName, p, t] >= pyM.stateOfChargeMin[loc, compName, p]
            pyM.ConstSOCintraPeriodMin_stor = \
                pyomo.Constraint(pyM.designDimensionVarSetSimple_stor, pyM.timeSet, rule=SOCintraPeriodMin_stor)

            # The state of charge at the beginning of one period plus the maximum (virtual) state of charge
            # during that period has to be smaller than the installed capacities multiplied with the relative maximum
            # state of charge
            def SOCMaxSimple_stor(pyM, loc, compName, pInter):
                return (pyM.stateOfChargeInterPeriods[loc, compName, pInter] +
                        pyM.stateOfChargeMax[loc, compName, esM._periodsOrder[pInter]]
                        <= pyM.cap_stor[loc, compName] * compDict[compName]._stateOfChargeMax)
            pyM.ConstrSOCMaxSimple_stor = pyomo.Constraint(pyM.designDimensionVarSetSimple_stor,
                                                           esM._periods, rule=SOCMaxSimple_stor)

            # The state of charge at the beginning of one period plus the minimum (virtual) state of charge
            # during that period has to be larger than the installed capacities multiplied with the relative minimum
            # state of charge
            def SOCMinSimple_stor(pyM, loc, compName, pInter):
                return (pyM.stateOfChargeInterPeriods[loc, compName, pInter] *
                        (1-compDict[compName]._selfDischarge)**((esM._timeStepsPerPeriod[-1]+1)*esM._hoursPerTimeStep)
                        + pyM.stateOfChargeMin[loc, compName, esM._periodsOrder[pInter]]
                        >= pyM.cap_stor[loc, compName] * compDict[compName]._stateOfChargeMin)
            pyM.ConstrSOCMinSimple_stor = pyomo.Constraint(pyM.designDimensionVarSetSimple_stor,
                                                           esM._periods, rule=SOCMinSimple_stor)

            #                        Precise version of the state of charge limitation control                         #

            # Constraints for enforcing a state of charge operation within given limits

            # State of charge [energyUnit] limited by the installed capacity [powerUnit] and the relative maximum
            # state of charge
            def SOCMaxPrecise1_stor(pyM, loc, compName, pInter, t):
                return (pyM.stateOfChargeInterPeriods[loc, compName, pInter] *
                        ((1 - compDict[compName]._selfDischarge) ** (t * esM._hoursPerTimeStep)) +
                        pyM.stateOfCharge[loc, compName, esM._periodsOrder[pInter], t]
                        <= pyM.cap_stor[loc, compName] * compDict[compName]._stateOfChargeMax)
            pyM.ConstrSOCMaxPrecise1_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet1_stor,
                                                             esM._periods, esM._timeStepsPerPeriod,
                                                             rule=SOCMaxPrecise1_stor)

            def SOCMaxPrecise2_stor(pyM, loc, compName, pInter, t):
                return (pyM.stateOfChargeInterPeriods[loc, compName, pInter] *
                        ((1 - compDict[compName]._selfDischarge) ** (t * esM._hoursPerTimeStep)) +
                        pyM.stateOfCharge[loc, compName, esM._periodsOrder[pInter], t]
                        == pyM.cap_stor[loc, compName] *
                        compDict[compName]._stateOfChargeOpRateFix[loc][esM._periodsOrder[pInter], t])
            pyM.ConstrSOCMaxPrecise2_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet2_stor,
                                                             esM._periods, esM._timeStepsPerPeriod,
                                                             rule=SOCMaxPrecise2_stor)

            def SOCMaxPrecise3_stor(pyM, loc, compName, pInter, t):
                return (pyM.stateOfChargeInterPeriods[loc, compName, pInter] *
                        ((1 - compDict[compName]._selfDischarge) ** (t * esM._hoursPerTimeStep)) +
                        pyM.stateOfCharge[loc, compName, esM._periodsOrder[pInter], t]
                        <= pyM.cap_stor[loc, compName] *
                        compDict[compName]._stateOfChargeOpRateMax[loc][esM._periodsOrder[pInter], t])
            pyM.ConstrSOCMaxPrecise3_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet3_stor,
                                                             esM._periods, esM._timeStepsPerPeriod,
                                                             rule=SOCMaxPrecise3_stor)

            def SOCMaxPrecise4_stor(pyM, loc, compName, pInter, t):
                return (pyM.stateOfChargeInterPeriods[loc, compName, pInter] *
                        ((1 - compDict[compName]._selfDischarge) ** (t * esM._hoursPerTimeStep)) +
                        pyM.stateOfCharge[loc, compName, esM._periodsOrder[pInter], t]
                        == compDict[compName]._stateOfChargeOpRateFix[loc][esM._periodsOrder[pInter], t])
            pyM.ConstrSOCMaxPrecise4_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet4_stor,
                                                             esM._periods, esM._timeStepsPerPeriod,
                                                             rule=SOCMaxPrecise4_stor)

            def SOCMaxPrecise5_stor(pyM, loc, compName, pInter, t):
                return (pyM.stateOfChargeInterPeriods[loc, compName, pInter] *
                        ((1 - compDict[compName]._selfDischarge) ** (t * esM._hoursPerTimeStep)) +
                        pyM.stateOfCharge[loc, compName, esM._periodsOrder[pInter], t]
                        <= compDict[compName]._stateOfChargeOpRateMax[loc][esM._periodsOrder[pInter], t])
            pyM.ConstrSOCMaxPrecise5_stor = pyomo.Constraint(pyM.stateOfChargeOpConstrSet5_stor,
                                                             esM._periods, esM._timeStepsPerPeriod,
                                                             rule=SOCMaxPrecise5_stor)

            # The state of charge at each time step cannot be smaller than the installed capacity multiplied with the
            # relative minimum state of charge
            def SOCMinPrecise_stor(pyM, loc, compName, pInter, t):
                return (pyM.stateOfChargeInterPeriods[loc, compName, pInter] *
                        ((1 - compDict[compName]._selfDischarge) ** (t * esM._hoursPerTimeStep)) +
                        pyM.stateOfCharge[loc, compName, esM._periodsOrder[pInter], t]
                        >= pyM.cap_stor[loc, compName] * compDict[compName]._stateOfChargeMin)
            pyM.ConstrSOCMinPrecise_stor = pyomo.Constraint(pyM.designDimensionVarSetPrecise_stor,
                                                            esM._periods, esM._timeStepsPerPeriod,
                                                            rule=SOCMinPrecise_stor)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        return sum(pyM.cap_stor[loc, compName] / self._componentsDict[compName]._capacityMax[loc]
                   for compName in self._componentsDict if
                   self._componentsDict[compName]._sharedPotentialID == key and
                   (loc, compName) in pyM.designDimensionVarSet_stor)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([comp._commodity == commod and comp._locationalEligibility[loc] == 1
                    for comp in self._componentsDict.values()])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        return sum(pyM.dischargeOp[loc, compName, p, t] - pyM.chargeOp[loc, compName, p, t]
                   for compName in pyM.operationVarDict_stor[loc]
                   if commod == self._componentsDict[compName]._commodity)

    def getObjectiveFunctionContribution(self, esM, pyM):
        compDict = self._componentsDict

        capexDim = sum(compDict[compName]._investPerCapacity[loc] * pyM.cap_stor[loc, compName] /
                       compDict[compName]._CCF[loc] for loc, compName in pyM.cap_stor)

        capexDec = sum(compDict[compName]._investIfBuilt[loc] * pyM.designBin_stor[loc, compName] /
                       compDict[compName]._CCF[loc] for loc, compName in pyM.designBin_stor)

        opexDim = sum(compDict[compName]._opexPerCapacity[loc] * pyM.cap_stor[loc, compName]
                      for loc, compName in pyM.cap_stor)

        opexDec = sum(compDict[compName]._opexIfBuilt[loc] * pyM.designBin_stor[loc, compName]
                      for loc, compName in pyM.designBin_stor)

        opexOp = sum(compDict[compName]._opexPerChargeOperation[loc] *
                     sum(pyM.chargeOp[loc, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet) +
                     compDict[compName]._opexPerDischargeOperation[loc] *
                     sum(pyM.dischargeOp[loc, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet)
                     for loc, compNames in pyM.operationVarDict_stor.items() for compName in
                     compNames) / esM._numberOfYears

        return capexDim + capexDec + opexDim + opexDec + opexOp

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        optVal = utils.formatOptimizationOutput(pyM.cap_stor.get_values(), 'designVariables', '1dim')
        self._designDimensionVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_designDimensionVariablesOptimum', self._componentsDict)

        optVal = utils.formatOptimizationOutput(pyM.designBin_stor.get_values(), 'designVariables', '1dim')
        self._designDecisionVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_designDecisionVariablesOptimum', self._componentsDict)

        optVal = utils.formatOptimizationOutput(pyM.chargeOp.get_values(), 'operationVariables', '1dim',
                                                esM._periodsOrder)
        self._chargeOperationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_chargeOperationVariablesOptimum', self._componentsDict)

        optVal = utils.formatOptimizationOutput(pyM.dischargeOp.get_values(), 'operationVariables', '1dim',
                                                esM._periodsOrder)
        self._dischargeOperationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_dischargeOperationVariablesOptimum', self._componentsDict)

        if not pyM.hasTSA:
            optVal = utils.formatOptimizationOutput(pyM.stateOfCharge.get_values(), 'operationVariables', '1dim',
                                                    esM._periodsOrder)
            self._stateOfChargeOperationVariablesOptimum = optVal
            utils.setOptimalComponentVariables(optVal, '_stateOfChargeVariablesOptimum', self._componentsDict)
        else:
            stateOfChargeIntra = pyM.stateOfCharge.get_values()
            stateOfChargeInter = pyM.stateOfChargeInterPeriods.get_values()
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
            utils.setOptimalComponentVariables(optVal, '_stateOfChargeVariablesOptimum', self._componentsDict)



        #TODO state of charge

    def getOptimalCapacities(self):
        return self._capacitiesOpt