from FINE.component import Component, ComponentModel
from FINE import utils
import warnings
import pandas as pd


class Conversion(Component):
    #TODO
    """
    Conversion class

    The functionality of the the Conversion class is fourfold:
    * ...

    The parameter which are stored in an instance of the class refer to:
    * ...

    Instances of this class provide function for
    * ...

    Last edited: July 27, 2018
    |br| @author: Lara Welder
    """
    def __init__(self, esM, name, physicalUnit, commodityConversionFactors, hasCapacityVariable=True,
                 capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                 hasIsBuiltBinaryVariable=False, bigM=None,
                 operationRateMax=None, operationRateFix=None, tsaWeight=1,
                 locationalEligibility=None, capacityMin=None, capacityMax=None, sharedPotentialID=None,
                 capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerOperation=0, opexPerCapacity=0,
                 opexIfBuilt=0, interestRate=0.08, economicLifetime=10):
        # TODO: allow that the time series data or min/max/fixCapacity/eligibility is only specified for
        # TODO: eligible locations
        """
        Constructor for creating an Conversion class instance. Capacities are given in the plants
        physicalUnit.
        The Conversion component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param physicalUnit: reference physical unit of the plants to which maximum capacity limitations,
            cost parameters and the operation time series refer to.
        :type physicalUnit: string

        :param commodityConversionFactors: conversion factors with which commodities are converted into each
            other with one unit of operation (dictionary). Each commodity which is converted in this component
            is indicated by a string in this dictionary. The conversion factor related to this commodity is
            given as a float. A negative value indicates that the commodity is consumed. A positive value
            indicates that the commodity is produced. Check unit consistency when specifying this parameter!
            Examples:
            (a) An electrolyzer converts, simply put, electricity into hydrogen with an electrical efficiency
                of 70%. The physicalUnit is given as GW_electric, the unit for the 'electricity' commodity is
                given in GW_electric and the 'hydrogen' commodity is given in GW_hydrogen_lowerHeatingValue
                -> the commodityConversionFactors are defined as {'electricity':-1,'hydrogen':0.7}.
            (b) An fuel cell converts, simply put, hydrogen into electricity with an efficiency of 60%.
                The physicalUnit is given as GW_electric, the unit for the 'electricity' commodity is given in
                GW_electric and the 'hydrogen' commodity is given in GW_hydrogen_lowerHeatingValue -> the
                commodityConversionFactors are defined as {'electricity':1,'hydrogen':-1/0.6}.
        :type commodityConversionFactors: dictionary, assigns commodities (string) to a conversion factors
            (float)

        **Default arguments:**

        :param operationRateMax: if specified indicates a maximum operation rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. in that case a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the physicalUnit of the plant for each time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param operationRateFix: if specified indicates a fixed operation rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. in that case a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the physicalUnit of the plant for each time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param opexPerOperation: cost which is directly proportional to the operation of the component
            is obtained by multiplying the opexPerOperation parameter with the annual sum of the
            operational time series of the components. The opexPerOperation can either be given as a
            float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerOperation: positive (>=0) float or Pandas Series with positive (>=0) values.
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

        # Set general conversion data
        utils.checkCommodities(esM, set(commodityConversionFactors.keys()))
        self._commodityConversionFactors = commodityConversionFactors
        self._physicalUnit = physicalUnit
        self._modelingClass = ConversionModel

        # Set additional economic data
        self._opexPerOperation = utils.checkAndSetCostParameter(esM, name, opexPerOperation, '1dim',
                                                                locationalEligibility)

        # Set location-specific operation parameters
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            warnings.warn('If operationRateFix is specified, the operationRateMax parameter is not required.\n' +
                          'The operationRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateMax, locationalEligibility)
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateFix, locationalEligibility)

        self._fullOperationRateMax = utils.setFormattedTimeSeries(operationRateMax)
        self._aggregatedOperationRateMax = None
        self._operationRateMax = None

        self._fullOperationRateFix = utils.setFormattedTimeSeries(operationRateFix)
        self._aggregatedOperationRateFix = None
        self._operationRateFix = None

        utils.isPositiveNumber(tsaWeight)
        self._tsaWeight = tsaWeight

        # Set locational eligibility
        operationTimeSeries = operationRateFix if operationRateFix is not None else operationRateMax
        self._locationalEligibility = \
            utils.setLocationalEligibility(esM, self._locationalEligibility, self._capacityMax, self._capacityFix,
                                           self._isBuiltFix, self._hasCapacityVariable, operationTimeSeries)

    def addToEnergySystemModel(self, esM):
        super().addToEnergySystemModel(esM)

    def setTimeSeriesData(self, hasTSA):
        self._operationRateMax = self._aggregatedOperationRateMax if hasTSA else self._fullOperationRateMax
        self._operationRateFix = self._aggregatedOperationRateFix if hasTSA else self._fullOperationRateFix

    def getDataForTimeSeriesAggregation(self):
        weightDict, data = {}, []
        weightDict, data = self.prepareTSAInput(self._fullOperationRateFix, self._fullOperationRateMax,
                                                '_operationRate_', self._tsaWeight, weightDict, data)
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):
        self._aggregatedOperationRateFix = self.getTSAOutput(self._fullOperationRateFix, '_operationRate_', data)
        self._aggregatedOperationRateMax = self.getTSAOutput(self._fullOperationRateMax, '_operationRate_', data)


class ConversionModel(ComponentModel):
    """ Doc """
    def __init__(self):
        self._abbrvName = 'conv'
        self._dimension = '1dim'
        self._componentsDict = {}
        self._capacityVariablesOptimum, self._isBuiltVariablesOptimum = None, None
        self._operationVariablesOptimum = None
        self._optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """ Declares sets and dictionaries """

        # Declare design variable sets
        self.initDesignVarSet(pyM)
        self.initContinuousDesignVarSet(pyM)
        self.initDiscreteDesignVarSet(pyM)
        self.initDesignDecisionVarSet(pyM)

        # Declare operation variable set
        self.initOpVarSet(esM, pyM)

        # Declare operation variable set
        self.declareOperationModeSets(pyM, 'opConstrSet', '_operationRateMax', '_operationRateFix')

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """ Declares design and operation variables """

        # Capacity variables in [physicalUnit]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components in [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components in [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not in [-]
        self.declareBinaryDesignDecisionVars(pyM)
        # Operation of component [physicalUnit*hour]
        self.declareOperationVars(pyM, 'op')

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

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

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        self.operationMode1(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
        # Operation [energyUnit] equal to the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        self.operationMode2(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        self.operationMode3(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
        # Operation [energyUnit] equal to the operation time series [energyUnit]
        self.operationMode4(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
        # Operation [energyUnit] limited by the operation time series [energyUnit]
        self.operationMode5(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        return super().getSharedPotentialContribution(pyM, key, loc)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([(commod in comp._commodityConversionFactors and comp._commodityConversionFactors[commod] != 0)
                    and comp._locationalEligibility[loc] == 1 for comp in self._componentsDict.values()])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVar, opVarDict = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'operationVarDict_' + abbrvName)
        return sum(opVar[loc, compName, p, t] * compDict[compName]._commodityConversionFactors[commod]
                   for compName in opVarDict[loc] if commod in compDict[compName]._commodityConversionFactors)

    def getObjectiveFunctionContribution(self, esM, pyM):

        capexCap = self.getEconomicsTI(pyM, ['_investPerCapacity'], 'cap', '_CCF')
        capexDec = self.getEconomicsTI(pyM, ['_investIfBuilt'], 'designBin', '_CCF')
        opexCap = self.getEconomicsTI(pyM, ['_opexPerCapacity'], 'cap')
        opexDec = self.getEconomicsTI(pyM, ['_opexIfBuilt'], 'designBin')
        opexOp = self.getEconomicsTD(pyM, esM, ['_opexPerOperation'], 'op', 'operationVarDict')

        return capexCap + capexDec + opexCap + opexDec + opexOp

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVar = getattr(pyM, 'op_' + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        # optVal_ = utils.formatOptimizationOutput(capVar.get_values(), 'designVariables', '1dim')
        # self._capacityVariablesOptimum = optVal_
        # utils.setOptimalComponentVariables(optVal_, '_capacityVariablesOptimum', compDict)
        #
        # optVal_ = utils.formatOptimizationOutput(binVar.get_values(), 'designVariables', '1dim')
        # self._isBuiltVariablesOptimum = optVal_
        # utils.setOptimalComponentVariables(optVal_, '_isBuiltVariablesOptimum', compDict)

        optSummaryBasic = super().setOptimalValues(esM, pyM, esM._locations, '_physicalUnit')

        # Set optimal operation variables and append optimization summary
        optVal = utils.formatOptimizationOutput(opVar.get_values(), 'operationVariables', '1dim', esM._periodsOrder)
        self._operationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_operationVariablesOptimum', compDict)

        props = ['operation', 'opexOp']
        units = ['[-]', '[' + esM._costUnit + '/a]']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._physicalUnit + '*h/a]')
                          if x[1] == 'operation' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM._locations)).sort_index()

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name]._opexPerOperation[op.index], axis=1)
            optSummary.loc[[(ix, 'operation', '[' + compDict[ix]._physicalUnit + '*h/a]') for ix in opSum.index],
                            opSum.columns] = opSum.values/esM._numberOfYears
            optSummary.loc[[(ix, 'opexOp', '[' + esM._costUnit + '/a]') for ix in ox.index], ox.columns] = \
                ox.values/esM._numberOfYears

        optSummary = optSummary.append(optSummaryBasic).sort_index()

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'TAC') |
                           (optSummary.index.get_level_values(1) == 'opexOp')].groupby(level=0).sum().values

        self._optSummary = optSummary
