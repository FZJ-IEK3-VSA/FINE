from FINE.component import Component, ComponentModel
from FINE import utils
import warnings
import pyomo.environ as pyomo
import pandas as pd


class Transmission(Component):
    """
    Doc
    """
    def __init__(self, esM, name, commodity, losses=0, distances=None,
                 hasCapacityVariable=True, capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                 hasIsBuiltBinaryVariable=False, bigM=None,
                 operationRateMax=None, operationRateFix=None, tsaWeight=1,
                 locationalEligibility=None, capacityMin=None, capacityMax=None, sharedPotentialID=None,
                 capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerOperation=0, opexPerCapacity=0,
                 opexIfBuilt=0, interestRate=0.08, economicLifetime=10):
        """
        Constructor for creating an Conversion class instance.
        The Transmission component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param commodity: to the component related commodity.
        :type commodity: string

        **Default arguments:**

        :param losses: losses per lengthUnit (lengthUnit as specified in the energy system model). This loss
            factor can capture simple linear losses trans_in_ij=(1-losses*distance)*trans_out_ij (with trans
            being the commodity flow at a certain point in time and i and j being locations in the energy
            system). The losses can either be given as a float or a Pandas DataFrame with location specific
            values.
            |br| * the default value is 0
        :type losses: positive float (0 <= float <= 1) or Pandas DataFrame with positive values
            (0 <= float <= 1). The row and column indices of the DataFrame have to equal the in the energy
            system model specified locations.

        :param distances: distances between locations, given in the lengthUnit (lengthUnit as specified in
            the energy system model).
            |br| * the default value is None
        :type distances: positive float (>= 0) or Pandas DataFrame with positive values (>= 0). The row and
            column indices of the DataFrame have to equal the in the energy system model specified locations.

        :param operationRateMax: if specified, indicates a maximum operation rate for all possible connections
            (both directions) of the transmission component at each time step by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            in that case a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices are combinations
            of locations (as defined in the energy system model), separated by a underscore (i.e.
            "location1_location2"). The first location indicates where the commodity is coming from. The second
            one location indicates where the commodity is going too. If a flow is specified from location i to
            location j, it also has to be specified from j to i.

        :param operationRateFix: if specified, indicates a fixed operation rate for all possible connections
            (both directions) of the transmission component at each time step by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            in that case a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices are combinations
            of locations (as defined in the energy system model), separated by a underscore (i.e.
            "location1_location2"). The first location indicates where the commodity is coming from. The second
            one location indicates where the commodity is going too. If a flow is specified from location i to
            location j, it also has to be specified from j to i.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param opexPerOperation: cost which is directly proportional to the operation of the component
            is obtained by multiplying the opexPerOperation parameter with the annual sum of the
            operational time series of the components. The opexPerOperation can either be given as a
            float or a Pandas DataFrame with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerOperation: positive (>=0) float or Pandas DataFrame with positive (>=0) values.
            The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.
        """
        # TODO add unit checks
        # Preprocess two-dimensional data
        self._locationalEligibility = utils.preprocess2dimData(locationalEligibility)
        self._capacityMax = utils.preprocess2dimData(capacityMax)
        self._capacityFix = utils.preprocess2dimData(capacityFix)
        self._isBuiltFix = utils.preprocess2dimData(isBuiltFix)

        # Set locational eligibility
        operationTimeSeries = operationRateFix if operationRateFix is not None else operationRateMax
        self._locationalEligibility = \
            utils.setLocationalEligibility(esM, self._locationalEligibility, self._capacityMax, self._capacityFix,
                                           self._isBuiltFix, hasCapacityVariable, operationTimeSeries, '2dim')

        self._mapC, self._mapL, self._mapI = {}, {}, {}
        for loc1 in esM._locations:
            for loc2 in esM._locations:
                if loc1 + '_' + loc2 in self._locationalEligibility.index:
                    if self._locationalEligibility[loc1 + '_' + loc2] == 0:
                        self._locationalEligibility[loc1 + '_' + loc2].drop(inplace=True)
                    self._mapC.update({loc1 + '_' + loc2: (loc1, loc2)})
                    self._mapL.setdefault(loc1, {}).update({loc2: loc1 + '_' + loc2})
                    self._mapI.update({loc1 + '_' + loc2: loc2 + '_' + loc1})

        self._capacityMin = utils.preprocess2dimData(capacityMin, self._mapC)
        self._investPerCapacity = utils.preprocess2dimData(investPerCapacity, self._mapC)
        self._investIfBuilt = utils.preprocess2dimData(investIfBuilt, self._mapC)
        self._opexPerCapacity = utils.preprocess2dimData(opexPerCapacity, self._mapC)
        self._opexIfBuilt = utils.preprocess2dimData(opexIfBuilt, self._mapC)
        self._interestRate = utils.preprocess2dimData(interestRate, self._mapC)
        self._economicLifetime = utils.preprocess2dimData(economicLifetime, self._mapC)

        Component. __init__(self, esM, name, dimension='2dim', hasCapacityVariable=hasCapacityVariable,
                            capacityVariableDomain=capacityVariableDomain, capacityPerPlantUnit=capacityPerPlantUnit,
                            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable, bigM=bigM,
                            locationalEligibility=self._locationalEligibility, capacityMin=self._capacityMin,
                            capacityMax=self._capacityMax, sharedPotentialID=sharedPotentialID,
                            capacityFix=self._capacityFix, isBuiltFix=self._isBuiltFix,
                            investPerCapacity=self._investPerCapacity, investIfBuilt=self._investIfBuilt,
                            opexPerCapacity=self._opexPerCapacity, opexIfBuilt=self._opexIfBuilt,
                            interestRate=self._interestRate, economicLifetime=self._economicLifetime)

        # Set general component data
        utils.checkCommodities(esM, {commodity})
        self._commodity, self._commodityUnit = commodity, esM._commoditiyUnitsDict[commodity]
        # TODO flatten distances, losses with respect to elig
        self._distances = utils.preprocess2dimData(distances, self._mapC)
        self._losses = utils.preprocess2dimData(losses, self._mapC)
        self._distances = utils.checkAndSetDistances(self._distances, self._locationalEligibility)
        self._losses = utils.checkAndSetTransmissionLosses(self._losses, self._distances, self._locationalEligibility)
        self._modelingClass = TransmissionModel

        # Set additional economic data
        self._opexPerOperation = utils.checkAndSetCostParameter(esM, name, opexPerOperation, '2dim',
                                                                self._locationalEligibility)

        # Set location-specific operation parameters
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            warnings.warn('If operationRateFix is specified, the operationRateMax parameter is not required.\n' +
                          'The operationRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateMax, self._locationalEligibility, '2dim')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateFix, self._locationalEligibility, '2dim')

        self._fullOperationRateMax = utils.setFormattedTimeSeries(operationRateMax)
        self._aggregatedOperationRateMax = None
        self._operationRateMax = utils.setFormattedTimeSeries(operationRateMax)

        self._fullOperationRateFix = utils.setFormattedTimeSeries(operationRateFix)
        self._aggregatedOperationRateFix = None
        self._operationRateFix = utils.setFormattedTimeSeries(operationRateFix)

        utils.isPositiveNumber(tsaWeight)
        self._tsaWeight = tsaWeight

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


class TransmissionModel(ComponentModel):
    """ Doc """
    def __init__(self):
        self._abbrvName = 'trans'
        self._dimension = '2dim'
        self._componentsDict = {}
        self._capacityVariablesOptimum, self._isBuiltVariablesOptimum = None, None
        self._operationVariablesOptimum = None
        self._optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """ Declares sets and dictionaries """

        # # Declare design variable sets
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

        # Capacity variables in [commodityUnit]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components in [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components in [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not in [-]
        self.declareBinaryDesignDecisionVars(pyM)
        # Operation of component [commodityUnit]
        self.declareOperationVars(pyM, 'op')

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def symmetricalCapacity(self, pyM):
        """
        Enforces that the capacity between location_1 and location_2 is the same as the one
        between location_2 and location_1
        """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        capVar, capVarSet = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        def symmetricalCapacity(pyM, loc, compName):
            return capVar[loc, compName] == capVar[compDict[compName]._mapI[loc], compName]
        setattr(pyM, 'ConstrSymmetricalCapacity_' + abbrvName,  pyomo.Constraint(capVarSet, rule=symmetricalCapacity))

    def operationMode1_2dim(self, pyM, esM, constrName, constrSetName, opVarName):
        """
        Operation [commodityUnit*hour] limited by the installed capacity [commodityUnit] multiplied by the hours
        per time step.
        Since the flow should either go in one direction or the other, the limitation can be enforced on the sum
        of the forward and backward flow over the line. This leads to one of the flow variables being set to zero
        if a basic solution is obtained during optimization.
        """
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet1 = getattr(pyM, constrSetName + '1_' + abbrvName)

        def op1(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] + opVar[compDict[compName]._mapI[loc], compName, p, t] <= \
                   capVar[loc, compName] * esM._hoursPerTimeStep
        setattr(pyM, constrName + '_' + abbrvName, pyomo.Constraint(constrSet1, pyM.timeSet, rule=op1))

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
        # Enforce that the capacity cap_loc1_loc2 is the same as cap_loc2_loc1
        self.symmetricalCapacity(pyM)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        self.operationMode1_2dim(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
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
        return any([comp._commodity == commod and
                    loc + '_' + loc_ in comp._locationalEligibility.index or
                    loc_ + '_' + loc in comp._locationalEligibility.index
                    for comp in self._componentsDict.values() for loc_ in esM._locations])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVar, opVarDictIn = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'operationVarDictIn_' + abbrvName)
        opVarDictOut = getattr(pyM, 'operationVarDictOut_' + abbrvName)
        return sum(opVar[loc_ + '_' + loc, compName, p, t] *
                   (1 - compDict[compName]._losses[loc_ + '_' + loc] * compDict[compName]._distances[loc_ + '_' + loc])
                   for loc_ in opVarDictIn[loc].keys()
                   for compName in opVarDictIn[loc][loc_]
                   if commod in compDict[compName]._commodity) - \
               sum(opVar[loc + '_' + loc_, compName, p, t]
                   for loc_ in opVarDictOut[loc].keys()
                   for compName in opVarDictOut[loc][loc_]
                   if commod in compDict[compName]._commodity)

    def getObjectiveFunctionContribution(self, esM, pyM):

        capexCap = self.getEconomicsTI(pyM, ['_investPerCapacity', '_distances'], 'cap', '_CCF') * 0.5
        capexDec = self.getEconomicsTI(pyM, ['_investIfBuilt', '_distances'], 'designBin', '_CCF') * 0.5
        opexCap = self.getEconomicsTI(pyM, ['_opexPerCapacity', '_distances'], 'cap') * 0.5
        opexDec = self.getEconomicsTI(pyM, ['_opexIfBuilt', '_distances'], 'designBin') * 0.5
        opexOp = self.getEconomicsTD(pyM, esM, ['_opexPerOperation'], 'op', 'operationVarDictOut')

        return capexCap + capexDec + opexCap + opexDec + opexOp

    def setOptimalValues(self, esM, pyM):
        compDict, abbrvName = self._componentsDict, self._abbrvName
        opVar = getattr(pyM, 'op_' + abbrvName)
        mapC = {loc1 + '_' + loc2: (loc1, loc2) for loc1 in esM._locations for loc2 in esM._locations}

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(esM, pyM, mapC.keys(), '_commodityUnit', costApp=0.5)

        # Set optimal operation variables and append optimization summary
        optVal = utils.formatOptimizationOutput(opVar.get_values(), 'operationVariables', '1dim', esM._periodsOrder)
        optVal_ = utils.formatOptimizationOutput(opVar.get_values(), 'operationVariables', '2dim', esM._periodsOrder,
                                                 compDict=compDict)
        self._operationVariablesOptimum = optVal_
        utils.setOptimalComponentVariables(optVal_, '_operationVariablesOptimum', compDict)

        props = ['operation', 'opexOp']
        units = ['[-]', '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._commodityUnit + '*h/a]')
                          if x[1] == 'operation' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(mapC.keys())).sort_index()

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name]._opexPerOperation[op.index], axis=1)
            optSummary.loc[[(ix, 'operation', '[' + compDict[ix]._commodityUnit + '*h/a]') for ix in opSum.index],
                            opSum.columns] = opSum.values/esM._numberOfYears
            optSummary.loc[[(ix, 'opexOp', '[' + esM._costUnit + '/a]') for ix in ox.index], ox.columns] = \
                ox.values/esM._numberOfYears * 0.5

        optSummary = optSummary.append(optSummaryBasic).sort_index()

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'TAC') |
                           (optSummary.index.get_level_values(1) == 'opexOp')].groupby(level=0).sum().values

        # Split connection indices to two location indices
        optSummary = optSummary.stack()
        indexNew = []
        for tup in optSummary.index.tolist():
            loc1, loc2 = mapC[tup[3]]
            indexNew.append((tup[0], tup[1], tup[2], loc1, loc2))
        optSummary.index = pd.MultiIndex.from_tuples(indexNew)
        optSummary = optSummary.unstack(level=-1)

        self._optSummary = optSummary