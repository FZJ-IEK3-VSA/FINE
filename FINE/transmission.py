from FINE.component import Component, ComponentModel
from FINE import utils
import warnings
import pyomo.environ as pyomo
import pandas as pd


class Transmission(Component):
    """
    A Transmission component can transmit a commodity between locations of the energy system.

    Last edited: November 28, 2018
    |br| @author: Lara Welder
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
        Constructor for creating an Transmission class instance.
        The Transmission component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param commodity: to the component related commodity.
        :type commodity: string

        **Default arguments:**

        :param losses: relative losses per lengthUnit (lengthUnit as specified in the energy system model) in
            percentage of the commodity flow. This loss factor can capture simple linear losses
            trans_in_ij=(1-losses*distance)*trans_out_ij (with trans being the commodity flow at a certain point in
            time and i and j being locations in the energy system). The losses can either be given as a float or a
            Pandas DataFrame with location specific values.
            |br| * the default value is 0
        :type losses: positive float (0 <= float <= 1) or Pandas DataFrame with positive values
            (0 <= float <= 1). The row and column indices of the DataFrame have to equal the in the energy
            system model specified locations.

        :param distances: distances between locations given in the lengthUnit (lengthUnit as specified in
            the energy system model).
            |br| * the default value is None
        :type distances: positive float (>= 0) or Pandas DataFrame with positive values (>= 0). The row and
            column indices of the DataFrame have to equal the in the energy system model specified locations.

        :param operationRateMax: if specified, indicates a maximum operation rate for all possible connections
            (both directions) of the transmission component at each time step by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices are combinations
            of locations (as defined in the energy system model), separated by a underscore (e.g.
            "location1_location2"). The first location indicates where the commodity is coming from. The second
            one location indicates where the commodity is going too. If a flow is specified from location i to
            location j, it also has to be specified from j to i.

        :param operationRateFix: if specified, indicates a fixed operation rate for all possible connections
            (both directions) of the transmission component at each time step by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices are combinations
            of locations (as defined in the energy system model), separated by a underscore (e.g.
            "location1_location2"). The first location indicates where the commodity is coming from. The second
            one location indicates where the commodity is going too. If a flow is specified from location i to
            location j, it also has to be specified from j to i.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param opexPerOperation: describes the cost for one unit of the operation.
            The cost which is directly proportional to the operation of the component is obtained by multiplying
            the opexPerOperation parameter with the annual sum of the operational time series of the components.
            The opexPerOperation can either be given as a float or a Pandas DataFrame with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerOperation: positive (>=0) float or Pandas DataFrame with positive (>=0) values.
            The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.
        """
        # TODO add unit checks
        # Preprocess two-dimensional data
        self.locationalEligibility = utils.preprocess2dimData(locationalEligibility)
        self.capacityMax = utils.preprocess2dimData(capacityMax)
        self.capacityFix = utils.preprocess2dimData(capacityFix)
        self.isBuiltFix = utils.preprocess2dimData(isBuiltFix)

        # Set locational eligibility
        operationTimeSeries = operationRateFix if operationRateFix is not None else operationRateMax
        self.locationalEligibility = \
            utils.setLocationalEligibility(esM, self.locationalEligibility, self.capacityMax, self.capacityFix,
                                           self.isBuiltFix, hasCapacityVariable, operationTimeSeries, '2dim')

        self._mapC, self._mapL, self._mapI = {}, {}, {}
        for loc1 in esM.locations:
            for loc2 in esM.locations:
                if loc1 + '_' + loc2 in self.locationalEligibility.index:
                    if self.locationalEligibility[loc1 + '_' + loc2] == 0:
                        self.locationalEligibility[loc1 + '_' + loc2].drop(inplace=True)
                    self._mapC.update({loc1 + '_' + loc2: (loc1, loc2)})
                    self._mapL.setdefault(loc1, {}).update({loc2: loc1 + '_' + loc2})
                    self._mapI.update({loc1 + '_' + loc2: loc2 + '_' + loc1})

        self.capacityMin = utils.preprocess2dimData(capacityMin, self._mapC)
        self.investPerCapacity = utils.preprocess2dimData(investPerCapacity, self._mapC)
        self.investIfBuilt = utils.preprocess2dimData(investIfBuilt, self._mapC)
        self.opexPerCapacity = utils.preprocess2dimData(opexPerCapacity, self._mapC)
        self.opexIfBuilt = utils.preprocess2dimData(opexIfBuilt, self._mapC)
        self.interestRate = utils.preprocess2dimData(interestRate, self._mapC)
        self.economicLifetime = utils.preprocess2dimData(economicLifetime, self._mapC)

        Component. __init__(self, esM, name, dimension='2dim', hasCapacityVariable=hasCapacityVariable,
                            capacityVariableDomain=capacityVariableDomain, capacityPerPlantUnit=capacityPerPlantUnit,
                            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable, bigM=bigM,
                            locationalEligibility=self.locationalEligibility, capacityMin=self.capacityMin,
                            capacityMax=self.capacityMax, sharedPotentialID=sharedPotentialID,
                            capacityFix=self.capacityFix, isBuiltFix=self.isBuiltFix,
                            investPerCapacity=self.investPerCapacity, investIfBuilt=self.investIfBuilt,
                            opexPerCapacity=self.opexPerCapacity, opexIfBuilt=self.opexIfBuilt,
                            interestRate=self.interestRate, economicLifetime=self.economicLifetime)

        # Set general component data
        utils.checkCommodities(esM, {commodity})
        self.commodity, self.commodityUnit = commodity, esM.commodityUnitsDict[commodity]
        self.distances = utils.preprocess2dimData(distances, self._mapC)
        self.losses = utils.preprocess2dimData(losses, self._mapC)
        self.distances = utils.checkAndSetDistances(self.distances, self.locationalEligibility, esM)
        self.losses = utils.checkAndSetTransmissionLosses(self.losses, self.distances, self.locationalEligibility)
        self.modelingClass = TransmissionModel

        # Set additional economic data
        self.opexPerOperation = utils.checkAndSetCostParameter(esM, name, opexPerOperation, '2dim',
                                                               self.locationalEligibility)

        # Set location-specific operation parameters
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            if esM.verbose < 2:
                warnings.warn('If operationRateFix is specified, the operationRateMax parameter is not required.\n' +
                              'The operationRateMax time series was set to None.')

        self.fullOperationRateMax = utils.checkAndSetTimeSeries(esM, operationRateMax, self.locationalEligibility)
        self.aggregatedOperationRateMax, self.operationRateMax = None, None

        self.fullOperationRateFix = utils.checkAndSetTimeSeries(esM, operationRateFix, self.locationalEligibility)
        self.aggregatedOperationRateFix, self.operationRateFix = None, None

        utils.isPositiveNumber(tsaWeight)
        self.tsaWeight = tsaWeight

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a transmission component to the given energy system model.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)

    def setTimeSeriesData(self, hasTSA):
        """
        Function for setting the maximum operation rate and fixed operation rate depending on whether a time series
        analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        self.operationRateMax = self.aggregatedOperationRateMax if hasTSA else self.fullOperationRateMax
        self.operationRateFix = self.aggregatedOperationRateFix if hasTSA else self.fullOperationRateFix

    def getDataForTimeSeriesAggregation(self):
        """ Function for getting the required data if a time series aggregation is requested. """
        weightDict, data = {}, []
        weightDict, data = self.prepareTSAInput(self.fullOperationRateFix, self.fullOperationRateMax,
                                                '_operationRate_', self.tsaWeight, weightDict, data)
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):
        """
        Function for determining the aggregated maximum rate and the aggregated fixed operation rate.

        :param data: Pandas DataFrame with the clustered time series data of the conversion component
        :type data: Pandas DataFrame
        """
        self.aggregatedOperationRateFix = self.getTSAOutput(self.fullOperationRateFix, '_operationRate_', data)
        self.aggregatedOperationRateMax = self.getTSAOutput(self.fullOperationRateMax, '_operationRate_', data)


class TransmissionModel(ComponentModel):
    """
    A TransmissionModel class instance will be instantly created if a Transmission class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Transmission class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The TransmissionModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        """" Constructor for creating a TransmissionModel class instance """
        self.abbrvName = 'trans'
        self.dimension = '2dim'
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """
        Declare sets: design variable sets, operation variable set and operation mode sets.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        # # Declare design variable sets
        self.declareDesignVarSet(pyM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        # Declare operation variable set
        self.declareOpVarSet(esM, pyM)

        # Declare operation mode sets
        self.declareOperationModeSets(pyM, 'opConstrSet', 'operationRateMax', 'operationRateFix')

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """
        Declare design and operation variables

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        # Capacity variables [commodityUnit]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not
        self.declareBinaryDesignDecisionVars(pyM)
        # Operation of component [commodityUnit]
        self.declareOperationVars(pyM, 'op')

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def symmetricalCapacity(self, pyM):
        """
        Ensure that the capacity between location_1 and location_2 is the same as the one
        between location_2 and location_1.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, capVarSet = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        def symmetricalCapacity(pyM, loc, compName):
            return capVar[loc, compName] == capVar[compDict[compName]._mapI[loc], compName]
        setattr(pyM, 'ConstrSymmetricalCapacity_' + abbrvName,  pyomo.Constraint(capVarSet, rule=symmetricalCapacity))

    def operationMode1_2dim(self, pyM, esM, constrName, constrSetName, opVarName):
        """
        Declare the constraint that the operation [commodityUnit*hour] is limited by the installed
        capacity [commodityUnit] multiplied by the hours per time step.
        Since the flow should either go in one direction or the other, the limitation can be enforced on the sum
        of the forward and backward flow over the line. This leads to one of the flow variables being set to zero
        if a basic solution is obtained during optimization.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet1 = getattr(pyM, constrSetName + '1_' + abbrvName)

        def op1(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] + opVar[compDict[compName]._mapI[loc], compName, p, t] <= \
                   capVar[loc, compName] * esM.hoursPerTimeStep
        setattr(pyM, constrName + '_' + abbrvName, pyomo.Constraint(constrSet1, pyM.timeSet, rule=op1))

    def declareComponentConstraints(self, esM, pyM):
        """
        Declare time independent and dependent constraints

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
        # Set, if applicable, the installed capacities of a component
        self.capacityFix(pyM)
        # Set, if applicable, the binary design variables of a component
        self.designBinFix(pyM)
        # Enforce the equality of the capacities cap_loc1_loc2 and cap_loc2_loc1
        self.symmetricalCapacity(pyM)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [commodityUnit*h] is limited by the installed capacity [commodityUnit] multiplied by the hours per
        # time step [h]
        self.operationMode1_2dim(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
        # Operation [commodityUnit*h] is equal to the installed capacity [commodityUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode2(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
        # Operation [commodityUnit*h] is limited by the installed capacity [commodityUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode3(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
        # Operation [commodityUnit*h] is equal to the operation time series [commodityUnit*h]
        self.operationMode4(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')
        # Operation [commodityUnit*h] is limited by the operation time series [commodityUnit*h]
        self.operationMode5(pyM, esM, 'ConstrOperation', 'opConstrSet', 'op')

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        """ Get contributions to shared location potential. """
        return super().getSharedPotentialContribution(pyM, key, loc)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        """
        Check if the commodity´s transfer between a given location and the other locations of the energy system model
        is eligible.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: Name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """

        return any([comp.commodity == commod and
                    (loc + '_' + loc_ in comp.locationalEligibility.index or
                     loc_ + '_' + loc in comp.locationalEligibility.index)
                    for comp in self.componentsDict.values() for loc_ in esM.locations])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """ Get contribution to a commodity balance. """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDictIn = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'operationVarDictIn_' + abbrvName)
        opVarDictOut = getattr(pyM, 'operationVarDictOut_' + abbrvName)
        return sum(opVar[loc_ + '_' + loc, compName, p, t] *
                   (1 - compDict[compName].losses[loc_ + '_' + loc] * compDict[compName].distances[loc_ + '_' + loc])
                   for loc_ in opVarDictIn[loc].keys()
                   for compName in opVarDictIn[loc][loc_]
                   if commod in compDict[compName].commodity) - \
               sum(opVar[loc + '_' + loc_, compName, p, t]
                   for loc_ in opVarDictOut[loc].keys()
                   for compName in opVarDictOut[loc][loc_]
                   if commod in compDict[compName].commodity)

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        capexCap = self.getEconomicsTI(pyM, ['investPerCapacity', 'distances'], 'cap', 'CCF') * 0.5
        capexDec = self.getEconomicsTI(pyM, ['investIfBuilt', 'distances'], 'designBin', 'CCF') * 0.5
        opexCap = self.getEconomicsTI(pyM, ['opexPerCapacity', 'distances'], 'cap') * 0.5
        opexDec = self.getEconomicsTI(pyM, ['opexIfBuilt', 'distances'], 'designBin') * 0.5
        opexOp = self.getEconomicsTD(pyM, esM, ['opexPerOperation'], 'op', 'operationVarDictOut')

        return capexCap + capexDec + opexCap + opexDec + opexOp

    def setOptimalValues(self, esM, pyM):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, 'op_' + abbrvName)
        mapC = {loc1 + '_' + loc2: (loc1, loc2) for loc1 in esM.locations for loc2 in esM.locations}

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(esM, pyM, mapC.keys(), 'commodityUnit')
        for compName, comp in compDict.items():
            for cost in ['invest', 'capexCap', 'capexIfBuilt', 'opexCap', 'opexIfBuilt']:
                data = optSummaryBasic.loc[compName, cost]
                optSummaryBasic.loc[compName, cost] = (0.5 * data * comp.distances).values

        # Set optimal operation variables and append optimization summary
        optVal = utils.formatOptimizationOutput(opVar.get_values(), 'operationVariables', '1dim', esM.periodsOrder)
        optVal_ = utils.formatOptimizationOutput(opVar.get_values(), 'operationVariables', '2dim', esM.periodsOrder,
                                                 compDict=compDict)
        self.operationVariablesOptimum = optVal_

        props = ['operation', 'opexOp']
        units = ['[-]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]].commodityUnit + '*h/a]')
                          if x[1] == 'operation' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(mapC.keys())).sort_index()

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name].opexPerOperation[op.index], axis=1)
            optSummary.loc[[(ix, 'operation', '[' + compDict[ix].commodityUnit + '*h/a]') for ix in opSum.index],
                            opSum.columns] = opSum.values/esM.numberOfYears
            optSummary.loc[[(ix, 'opexOp', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                ox.values/esM.numberOfYears * 0.5

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

        self.optSummary = optSummary

    def getOptimalValues(self, name='all'):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:\n
        * 'capacityVariables',
        * 'isBuiltVariables',
        * 'operationVariablesOptimum',
        * 'all' or another input: all variables are returned.\n
        |br| * the default value is 'all'
        :type name: string

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        return super().getOptimalValues(name)
