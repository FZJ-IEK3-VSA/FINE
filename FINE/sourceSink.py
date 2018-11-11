from FINE.component import Component, ComponentModel
from FINE import utils
import pandas as pd
import pyomo.environ as pyomo
import warnings


class Source(Component):
    # TODO
    """
    Doc
    """
    def __init__(self, esM, name, commodity, hasCapacityVariable,
                 capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                 hasIsBuiltBinaryVariable=False, bigM=None,
                 operationRateMax=None, operationRateFix=None, tsaWeight=1, commodityLimitID=None,
                 yearlyLimit=None, locationalEligibility=None, capacityMin=None, capacityMax=None,
                 sharedPotentialID=None, capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerOperation=0, commodityCost=0,
                 commodityRevenue=0, opexPerCapacity=0, opexIfBuilt=0, interestRate=0.08, economicLifetime=10):
        # TODO: allow that the time series data or min/max/fixCapacity/eligibility is only specified for
        # TODO: eligible locations
        """
        Constructor for creating an Source class instance.
        The Source component specific input arguments are described below. The general component
        input arguments are described in the Component class.
        Note: the Sink class inherits from the Source class and is initialized with the same parameter set

        **Required arguments:**

        :param commodity: to the component related commodity.
        :type commodity: string

        :param hasCapacityVariable: specifies if the component should be modeled with a capacity or not.
            Examples:\n
            * A wind turbine has a capacity given in GW_electric -> hasCapacityVariable is True.
            * Emitting CO2 into the environment is not per se limited by a capacity ->
              hasCapaityVariable is False.\n
        :type hasCapacityVariable: boolean

        **Default arguments:**

        :param operationRateMax: if specified indicates a maximum operation rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. in that case a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit for each time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param operationRateFix: if specified indicates a fixed operation rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. in that case a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit for each time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param commodityLimitID: can be specified to limit an annual commodity import/export over the
            energySystemModel's boundaries for one or multiple Source/Sink components. If the same ID
            is used in multiple components, the sum of all imports and exports is considered. If a
            commoditiyLimitID is specified, the yearlyLimit parameters has to be set as well.
            |br| * the default value is None
        :type commodityLimitID: string

        :param yearlyLimit: if specified, yearly import/export commodity limit for all components with
            the same commodityLimitID. If positive, the commodity flow leaving the energySystemModel is
            limited. If negative, the commodity flow entering the energySystemModel is limited. If a
            yearlyLimit is specified, the commoditiyLimitID parameters has to be set as well. Examples:\n
            * CO2 can be emitted in power plants by burning natural gas or coal. The CO2 which goes into
              the atmosphere over the energy system's boundaries is modelled as a Sink. CO2 can also be a
              Source, taken directly from the atmosphere (over the energy system's boundaries) for a
              methanation process. The commodityUnit for CO2 is tonnes_CO2. Overall, +XY tonnes_CO2 are
              allowed to be emitted during the year. All Sources/Sinks producing or consuming CO2 over the
              energy system's boundaries have the same commodityLimitID and the same yearlyLimit of +XY.
            * The maximum annual import of a certain chemical (commodityUnit tonnes_chem) is limited to
              XY tonnes_chem. The Source component modeling this import has a commodityLimitID
              "chemicalComponentLimitID" and a yearlyLimit of -XY.\n
            |br| * the default value is None
        :type yearlyLimit: float

        :param opexPerOperation: cost which is directly proportional to the operation of the component
            is obtained by multiplying the opexPerOperation parameter with the annual sum of the
            operational time series of the components. The opexPerOperation can either be given as a
            float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerOperation: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param commodityCost: cost which is directly proportional to the operation of the component
            is obtained by multiplying the commodityCost parameter with the annual sum of the
            time series of the components. The commodityCost can either be given as a
            float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            Example:\n
            * In a national energy system, natural gas could be purchased from another country with a
              certain cost.\n
            |br| * the default value is 0
        :type commodityCost: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param commodityRevenue: revenue which is directly proportional to the operation of the component
            is obtained by multiplying the commodityRevenue parameter with the annual sum of the
            time series of the components. The commodityRevenue can either be given as a
            float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro). Example:\n
            * Modeling a PV electricity feed-in tariff for a household\n
            |br| * the default value is 0
        :type commodityRevenue: positive (>=0) float or Pandas Series with positive (>=0) values.
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

        # Set general source/sink data
        utils.isEnergySystemModelInstance(esM), utils.checkCommodities(esM, {commodity})
        self.commodity, self.commodityUnit = commodity, esM.commodityUnitsDict[commodity]
        # TODO check value and type correctness
        self.commodityLimitID, self.yearlyLimit = commodityLimitID, yearlyLimit
        self.sign = 1
        self.modelingClass = SourceSinkModel

        # Set additional economic data
        self.opexPerOperation = utils.checkAndSetCostParameter(esM, name, opexPerOperation, '1dim',
                                                               locationalEligibility)
        self.commodityCost = utils.checkAndSetCostParameter(esM, name, commodityCost, '1dim',
                                                            locationalEligibility)
        self.commodityRevenue = utils.checkAndSetCostParameter(esM, name, commodityRevenue, '1dim',
                                                               locationalEligibility)

        # Set location-specific operation parameters
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            if esM.verbose < 2:
                warnings.warn('If operationRateFix is specified, the operationRateMax parameter is not required.\n' +
                              'The operationRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateMax, locationalEligibility)
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateFix, locationalEligibility)

        self.fullOperationRateMax = utils.setFormattedTimeSeries(operationRateMax)
        self.aggregatedOperationRateMax = None
        self.operationRateMax = utils.setFormattedTimeSeries(operationRateMax)

        self.fullOperationRateFix = utils.setFormattedTimeSeries(operationRateFix)
        self.aggregatedOperationRateFix = None
        self.operationRateFix = utils.setFormattedTimeSeries(operationRateFix)

        utils.isPositiveNumber(tsaWeight)
        self.tsaWeight = tsaWeight

        # Set locational eligibility
        operationTimeSeries = operationRateFix if operationRateFix is not None else operationRateMax
        self.locationalEligibility = \
            utils.setLocationalEligibility(esM, self.locationalEligibility, self.capacityMax, self.capacityFix,
                                           self.isBuiltFix, self.hasCapacityVariable, operationTimeSeries)

    def addToEnergySystemModel(self, esM):
        super().addToEnergySystemModel(esM)

    def setTimeSeriesData(self, hasTSA):
        self.operationRateMax = self.aggregatedOperationRateMax if hasTSA else self.fullOperationRateMax
        self.operationRateFix = self.aggregatedOperationRateFix if hasTSA else self.fullOperationRateFix

    def getDataForTimeSeriesAggregation(self):
        weightDict, data = {}, []
        weightDict, data = self.prepareTSAInput(self.fullOperationRateFix, self.fullOperationRateMax,
                                                '_operationRate_', self.tsaWeight, weightDict, data)
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):
        self.aggregatedOperationRateFix = self.getTSAOutput(self.fullOperationRateFix, '_operationRate_', data)
        self.aggregatedOperationRateMax = self.getTSAOutput(self.fullOperationRateMax, '_operationRate_', data)


class Sink(Source):
    def __init__(self, esM, name, commodity, hasCapacityVariable,
                 capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                 hasIsBuiltBinaryVariable=False, bigM=None,
                 operationRateMax=None, operationRateFix=None, tsamWeight=1, commodityLimitID=None,
                 yearlyLimit=None, locationalEligibility=None, capacityMin=None, capacityMax=None,
                 sharedPotentialID=None, capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerOperation=0, commodityCost=0,
                 commodityRevenue=0, opexPerCapacity=0, opexIfBuilt=0, interestRate=0.08,
                 economicLifetime=10):
        """
        Constructor for creating an Sink class instance.

        The Sink class inherits from the Source class. They coincide with there input parameters
        (see Source class for the parameter description) and differentiate themselves by the sign
        parameters, which is equal to -1 for Sink objects and +1 for Source objects.
        """
        Source.__init__(self, esM, name, commodity, hasCapacityVariable, capacityVariableDomain,
                        capacityPerPlantUnit, hasIsBuiltBinaryVariable, bigM, operationRateMax, operationRateFix,
                        tsamWeight, commodityLimitID, yearlyLimit, locationalEligibility, capacityMin,
                        capacityMax, sharedPotentialID, capacityFix, isBuiltFix, investPerCapacity,
                        investIfBuilt, opexPerOperation, commodityCost, commodityRevenue,
                        opexPerCapacity, opexIfBuilt, interestRate, economicLifetime)

        self.sign = -1


class SourceSinkModel(ComponentModel):
    """ Doc """
    def __init__(self):
        self.abbrvName = 'srcSnk'
        self.dimension = '1dim'
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareYearlyCommodityLimitationDict(self, pyM):
        yearlyCommodityLimitationDict = {}
        for compName, comp in self.componentsDict.items():
            if comp.commodityLimitID is not None:
                ID, limit = comp.commodityLimitID, comp.yearlyLimit
                if ID in yearlyCommodityLimitationDict and limit != yearlyCommodityLimitationDict[ID][0]:
                    raise ValueError('yearlyLimitationIDs with different upper limits detected.')
                yearlyCommodityLimitationDict.setdefault(ID, (limit, []))[1].append(compName)
        setattr(pyM, 'yearlyCommodityLimitationDict_' + self.abbrvName, yearlyCommodityLimitationDict)

    def declareSets(self, esM, pyM):
        """ Declares sets and dictionaries """

        # Declare design variable sets
        self.initDesignVarSet(pyM)
        self.initContinuousDesignVarSet(pyM)
        self.initDiscreteDesignVarSet(pyM)
        self.initDesignDecisionVarSet(pyM)

        # Declare operation variable set
        self.initOpVarSet(esM, pyM)

        # Declare sets for case differentiation of operating modes
        self.declareOperationModeSets(pyM, 'opConstrSet', 'operationRateMax', 'operationRateFix')

        # Declare commodity limitation dictionary
        self.declareYearlyCommodityLimitationDict(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """ Declares design and operation variables """

        # Capacity variables in [commodityUnit]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components in [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components in [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not in [-]
        self.declareBinaryDesignDecisionVars(pyM)
        # Operation of component [commodityUnit*hour]
        self.declareOperationVars(pyM, 'op')

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def yearlyLimitationConstraint(self, pyM, esM):
        """
        Limits annual commodity imports/exports over the energySystemModel's boundaries for one or multiple
        Source/Sink components.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, 'op_' + abbrvName)
        limitDict = getattr(pyM, 'yearlyCommodityLimitationDict_' + abbrvName)

        def yearlyLimitationConstraint(pyM, key):
            sumEx = -sum(opVar[loc, compName, p, t] * compDict[compName].sign *
                         esM.periodOccurrences[p]/esM.numberOfYears
                         for loc, compName, p, t in opVar if compName in limitDict[key][1])
            sign = limitDict[key][0]/abs(limitDict[key][0]) if limitDict[key][0] != 0 else 1
            return sign * sumEx <= sign * limitDict[key][0]
        setattr(pyM, 'ConstrYearlyLimitation_' + abbrvName,
                pyomo.Constraint(limitDict.keys(), rule=yearlyLimitationConstraint))

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

        self.yearlyLimitationConstraint(pyM, esM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        """ Gets contributions to shared location potential """
        return super().getSharedPotentialContribution(pyM, key, loc)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([comp.commodity == commod and comp.locationalEligibility[loc] == 1
                    for comp in self.componentsDict.values()])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """ Gets contribution to a commodity balance """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDict = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'operationVarDict_' + abbrvName)
        return sum(opVar[loc, compName, p, t] * compDict[compName].sign
                   for compName in opVarDict[loc] if compDict[compName].commodity == commod)

    def getObjectiveFunctionContribution(self, esM, pyM):
        """ Gets contribution to the objective function """

        capexCap = self.getEconomicsTI(pyM, ['investPerCapacity'], 'cap', 'CCF')
        capexDec = self.getEconomicsTI(pyM, ['investIfBuilt'], 'designBin', 'CCF')
        opexCap = self.getEconomicsTI(pyM, ['opexPerCapacity'], 'cap')
        opexDec = self.getEconomicsTI(pyM, ['opexIfBuilt'], 'designBin')
        opexOp = self.getEconomicsTD(pyM, esM, ['opexPerOperation'], 'op', 'operationVarDict')
        commodCost = self.getEconomicsTD(pyM, esM, ['commodityCost'], 'op', 'operationVarDict')
        commodRevenue = self.getEconomicsTD(pyM, esM, ['commodityRevenue'], 'op', 'operationVarDict')

        return capexCap + capexDec + opexCap + opexDec + opexOp + commodCost - commodRevenue

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, 'op_' + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(esM, pyM, esM.locations, 'commodityUnit')

        # Set optimal operation variables and append optimization summary
        optVal = utils.formatOptimizationOutput(opVar.get_values(), 'operationVariables', '1dim', esM.periodsOrder)
        self.operationVariablesOptimum = optVal

        props = ['operation', 'opexOp', 'commodCosts']
        units = ['[-]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]].commodityUnit + '*h/a]')
                          if x[1] == 'operation' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM.locations)).sort_index()

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name].opexPerOperation[op.index], axis=1)
            cCost = opSum.apply(lambda op: op * compDict[op.name].commodityCost[op.index], axis=1)
            cRevenue = opSum.apply(lambda op: op * compDict[op.name].commodityRevenue[op.index], axis=1)
            optSummary.loc[[(ix, 'operation', '[' + compDict[ix].commodityUnit + '*h/a]') for ix in opSum.index],
                            opSum.columns] = opSum.values/esM.numberOfYears
            optSummary.loc[[(ix, 'opexOp', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                ox.values/esM.numberOfYears
            optSummary.loc[[(ix, 'commodCosts', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                (cCost-cRevenue).values/esM.numberOfYears

        optSummary = optSummary.append(optSummaryBasic).sort_index()

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'TAC') |
                           (optSummary.index.get_level_values(1) == 'opexOp') |
                           (optSummary.index.get_level_values(1) == 'commodCosts')].groupby(level=0).sum().values

        self.optSummary = optSummary

    def getOptimalValues(self):
        return super().getOptimalValues()
