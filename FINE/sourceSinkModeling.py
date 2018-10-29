from FINE.component import Component, ComponentModeling
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
        Note: the Sink class inherits from the Source class and is initialized with the same parameter set

        **Required arguments:**

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param name: name of the component. Has to be unique (i.e. no other components with that name can
        already exist in the EnergySystemModel instance to which the component is added).
        :type name: string

        :param commodity: to the component related commodity.
        :type commodity: string

        :param hasCapacityVariable: specifies if the component should be modeled with a capacity or not.
            Examples:
            (a) A wind turbine has a capacity given in GW_electric -> hasCapacityVariable is True.
            (b) Emitting CO2 into the environment is not per se limited by a capacity ->
                hasCapacityVariable is False.
            |br| * the default value is True
        :type hasCapacityVariable: boolean

        **Default arguments:**

        :param capacityVariableDomain: the mathematical domain of the capacity variables, if they are specified.
            By default, the domain is specified as 'continuous' and thus declares the variables as positive
            (>=0) real values. The second input option that is available for this parameter is 'discrete', which
            declares the variables as positive (>=0) integer values.
            |br| * the default value is 'continuous'
        :type capacityVariableDomain: string ('continuous' or 'discrete')

        :param capacityPerPlantUnit: capacity of one plant of the component (in the specified physicalUnit of
            the plant). The default is 1, thus the number of plants is equal to the installed capacity.
            This parameter should be specified when using a 'discrete' capacityVariableDomain.
            It can be specified when using a 'continuous' variable domain.
            |br| * the default value is 1
        :type capacityPerPlantUnit: strictly positive float

        :param hasIsBuiltBinaryVariable: specifies if binary decision variables should be declared for each
            eligible location of the component, which indicate if the component is built at that location or
            not. The binary variables can be used to enforce one-time investment cost or capacity-independent
            annual operation cost. If a minimum capacity is specified and this parameter is set to True,
            the minimum capacities are only considered if a component is built (i.e. if a component is built
            at that location, it has to be built with a minimum capacity of XY GW, otherwise it is set to 0 GW).
            |br| * the default value is False
        :type hasIsBuiltBinaryVariable: boolean

        :param bigM: the bigM parameter is only required when the hasIsBuiltBinaryVariable parameter is set to
            True. In that case, it is set as a strictly positive float, otherwise it can remain a None value.
            If not None and the ifBuiltBinaryVariables parameter is set to True, the parameter enforces an
            artificial upper bound on the maximum capacities which should, however, never be reached. The value
            should be chosen as small as possible but as large as necessary so that the optimal values of the
            designed capacities are well below this value after the optimization.
            |br| * the default value is None
        :type bigM: None or strictly positive float

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
            yearlyLimit is specified, the commoditiyLimitID parameters has to be set as well.
            Examples:
            * CO2 can be emitted in power plants by burning natural gas or coal. The CO2 which goes into
              the atmosphere over the energy system's boundaries is modelled as a Sink. CO2 can also be a
              Source, taken directly from the atmosphere (over the energy system's boundaries) for a
              methanation process. The commodityUnit for CO2 is tonnes_CO2. Overall, +XY tonnes_CO2 are
              allowed to be emitted during the year. All Sources/Sinks producing or consuming CO2 over the
              energy system's boundaries have the same commodityLimitID and the same yearlyLimit of +XY.
            * The maximum annual import of a certain chemical (commodityUnit tonnes_chem) is limited to
              XY tonnes_chem. The Source component modeling this import has a commodityLimitID
              "chemicalComponentLimitID" and a yearlyLimit of -XY.
            |br| * the default value is None
        :type yearlyLimit: float

        :param locationalEligibility: Pandas Series that indicates if a component can be built at a location
            (=1) or not (=0). If not specified and a maximum or fixed capacity or time series is given, the
            parameter will be set based on these inputs. If the parameter is specified, a consistency check
            is done to ensure that the parameters indicate the same locational eligibility. If the parameter
            is not specified and also no other of the parameters is specified it is assumed that the
            component is eligible in each location and all values are set to 1.
            This parameter is key for ensuring small built times of the optimization problem by avoiding the
            declaration of unnecessary variables and constraints.
            |br| * the default value is None
        :type locationalEligibility: None or Pandas Series with values equal to 0 and 1. The indices of the
            series have to equal the in the energy system model specified locations.

        :param capacityMin: if specified, Pandas Series indicating minimum capacities (in the plant's
            physicalUnit) else None. If binary decision variables are declared, which indicate if a
            component is built at a location or not, the minimum capacity is only enforced if the component
            is built (i.e. if a component is built in that location, it has to be built with a minimum
            capacity of XY GW, otherwise it is set to 0 GW).
            |br| * the default value is None
        :type capacityMin: None or Pandas Series with positive (>=0) values. The indices of the series
            have to equal the in the energy system model specified locations.

        :param capacityMax: if specified, Pandas Series indicating maximum capacities (in the plants
            physicalUnit) else None.
            |br| * the default value is None
        :type capacityMax: None or Pandas Series with positive (>=0) values. The indices of the series
            have to equal the in the energy system model specified locations.

        :param sharedPotentialID: if specified, indicates that the component has to share its maximum
            potential capacity with other components (i.e. due to space limitations). The shares of how
            much of the maximum potential is used have to add up to less then 100%.
            |br| * the default value is None
        :type sharedPotentialID: string

        :param capacityFix: if specified, Pandas Series indicating fixed capacities (in the plants
            physicalUnit) else None.
            |br| * the default value is None
        :type capacityFix: None or Pandas Series with positive (>=0) values. The indices of the series
            have to equal the in the energy system model specified locations.

        :param isBuiltFix: if specified, Pandas Series indicating fixed decisions in which locations the
            component is built else None (i.e. sets the isBuilt binary variables).
            |br| * the default value is None
        :type isBuiltFix: None or Pandas Series with values equal to 0 and 1. The indices of the series
            have to equal the in the energy system model specified locations.

        :param investPerCapacity: the invest of a component is obtained by multiplying the capacity of the
            component (in the physicalUnit of the component) at that location with the investPerCapacity
            factor. The investPerCapacity can either be given as a float or a Pandas Series with location
            specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type investPerCapacity: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param investIfBuilt: a capacity-independent invest which only arises in a location if a component
            is built at that location. The investIfBuilt can either be given as a float or a Pandas Series
            with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type investIfBuilt: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

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
            Example:
            * In a national energy system, natural gas could be purchased from another country with a
              certain cost.
            |br| * the default value is 0
        :type commodityCost: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param commodityRevenue: revenue which is directly proportional to the operation of the component
            is obtained by multiplying the commodityRevenue parameter with the annual sum of the
            time series of the components. The commodityRevenue can either be given as a
            float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            Example:
            * Modeling a PV electricity feed-in tariff for a household
            |br| * the default value is 0
        :type commodityRevenue: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param opexPerCapacity: annual operational cost which are only a function of the capacity of the
            component (in the physicalUnit of the component) and not of the specific operation itself are
            obtained by multiplying the capacity of the component at a location with the opexPerCapacity
            factor. The opexPerCapacity can either be given as a float or a Pandas Series with location
            specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerCapacity: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param opexIfBuilt: a capacity-independent annual operational cost which only arises in a location
            if a component is built at that location. The opexIfBuilt can either be given as a float or a
            Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexIfBuilt: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param interestRate: interest rate which is considered for computing the annuities of the invest
            of the component (depreciates the invests over the economic lifetime).
            A value of 0.08 corresponds to an interest rate of 8%.
            |br| * the default value is 0.08
        :type interestRate: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param economicLifetime: economic lifetime of the component which is considered for computing the
            annuities of the invest of the component (aka depreciation time).
            |br| * the default value is 10
        :type economicLifetime: strictly-positive (>0) float or Pandas Series with strictly-positive (>=0)
            values. The indices of the series have to equal the in the energy system model specified locations.
        """
        # Set general component data
        utils.isEnergySystemModelInstance(esM), utils.checkCommodities(esM, {commodity})
        self._name, self._commodity, self._commodityUnit = name, commodity, esM._commoditiyUnitsDict[commodity]
        # TODO check value and type correctness
        self._commodityLimitID, self._yearlyLimit = commodityLimitID, yearlyLimit
        self._sign = 1

        # Set design variable modeling parameters
        utils.checkDesignVariableModelingParameters(capacityVariableDomain, hasCapacityVariable, capacityPerPlantUnit,
                                                    hasIsBuiltBinaryVariable, bigM)
        self._hasCapacityVariable = hasCapacityVariable
        self._capacityVariableDomain = capacityVariableDomain
        self._capacityPerPlantUnit = capacityPerPlantUnit
        self._hasIsBuiltBinaryVariable = hasIsBuiltBinaryVariable
        self._bigM = bigM

        # Set economic data
        self._investPerCapacity = utils.checkAndSetCostParameter(esM, name, investPerCapacity)
        self._investIfBuilt = utils.checkAndSetCostParameter(esM, name, investIfBuilt)
        self._opexPerOperation = utils.checkAndSetCostParameter(esM, name, opexPerOperation)
        self._opexPerCapacity = utils.checkAndSetCostParameter(esM, name, opexPerCapacity)
        self._opexIfBuilt = utils.checkAndSetCostParameter(esM, name, opexIfBuilt)
        self._commodityCost = utils.checkAndSetCostParameter(esM, name, commodityCost)
        self._commodityRevenue = utils.checkAndSetCostParameter(esM, name, commodityRevenue)
        self._interestRate = utils.checkAndSetCostParameter(esM, name, interestRate)
        self._economicLifetime = utils.checkAndSetCostParameter(esM, name, economicLifetime)
        self._CCF = utils.getCapitalChargeFactor(self._interestRate, self._economicLifetime)

        # Set location-specific operation parameters
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            warnings.warn('If operationRateFix is specified, the operationRateMax parameter is not required.\n' +
                          'The operationRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateMax, locationalEligibility)
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateFix, locationalEligibility)

        self._fullOperationRateMax = utils.setFormattedTimeSeries(operationRateMax)
        self._aggregatedOperationRateMax = None
        self._operationRateMax = utils.setFormattedTimeSeries(operationRateMax)

        self._fullOperationRateFix = utils.setFormattedTimeSeries(operationRateFix)
        self._aggregatedOperationRateFix = None
        self._operationRateFix = utils.setFormattedTimeSeries(operationRateFix)

        utils.isPositiveNumber(tsaWeight)
        self._tsaWeight = tsaWeight

        # Set location-specific design parameters
        self._sharedPotentialID = sharedPotentialID
        utils.checkLocationSpecficDesignInputParams(esM, hasCapacityVariable, hasIsBuiltBinaryVariable,
                                                    capacityMin, capacityMax, capacityFix,
                                                    locationalEligibility, isBuiltFix, sharedPotentialID,
                                                    dimension='1dim')
        self._capacityMin, self._capacityMax, self._capacityFix = capacityMin, capacityMax, capacityFix
        self._isBuiltFix = isBuiltFix

        # Set locational eligibility
        operationTimeSeries = operationRateFix if operationRateFix is not None else operationRateMax
        self._locationalEligibility = utils.setLocationalEligibility(esM, locationalEligibility, capacityMax,
                                                                     capacityFix, isBuiltFix,
                                                                     hasCapacityVariable, operationTimeSeries)

        # Variables at optimum (set after optimization)
        self._capacityVariablesOptimum = None
        self._isBuiltVariablesOptimum = None
        self._operationVariablesOptimum = None

    def addToEnergySystemModel(self, esM):
        esM._isTimeSeriesDataClustered = False
        if self._name in esM._componentNames:
            if esM._componentNames[self._name] == SourceSinkModeling.__name__:
                warnings.warn('Component identifier ' + self._name + ' already exists. Data will be overwritten.')
            else:
                raise ValueError('Component name ' + self._name + ' is not unique.')
        else:
            esM._componentNames.update({self._name: SourceSinkModeling.__name__})
        mdl = SourceSinkModeling.__name__
        if mdl not in esM._componentModelingDict:
            esM._componentModelingDict.update({mdl: SourceSinkModeling()})
        esM._componentModelingDict[mdl]._componentsDict.update({self._name: self})

    def setTimeSeriesData(self, hasTSA):
        self._operationRateMax = self._aggregatedOperationRateMax if hasTSA else self._fullOperationRateMax
        self._operationRateFix = self._aggregatedOperationRateFix if hasTSA else self._fullOperationRateFix

    def getDataForTimeSeriesAggregation(self):
        data = self._fullOperationRateFix if self._fullOperationRateFix is not None else self._fullOperationRateMax
        if data is not None:
            data, compDict = data.copy(), {}
            for location in data.columns:
                uniqueIdentifier = self._name + "_operationRate_" + location
                data[uniqueIdentifier] = data.pop(location)
                compDict.update({uniqueIdentifier: self._tsaWeight})
            return data, compDict
        else:
            return None, {}

    def setAggregatedTimeSeriesData(self, data):
        fullOperationRate = self._fullOperationRateFix if self._fullOperationRateFix is not None \
            else self._fullOperationRateMax
        if fullOperationRate is not None:
            uniqueIdentifiers = [self._name + "_operationRate_" + location for location in fullOperationRate.columns]
            compData = data[uniqueIdentifiers].copy()
            compData = compData.rename(columns={self._name + "_operationRate_" + location: location
                                                for location in fullOperationRate.columns})
            if self._fullOperationRateFix is not None:
                self._aggregatedOperationRateFix = compData
            else:
                self._aggregatedOperationRateMax = compData


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
        (see Source class for the parameter description) and differentiate themselves by the _sign
        parameters, which is equal to -1 for Sink objects and +1 for Source objects.
        """
        Source.__init__(self, esM, name, commodity, hasCapacityVariable, capacityVariableDomain,
                        capacityPerPlantUnit, hasIsBuiltBinaryVariable, bigM, operationRateMax, operationRateFix,
                        tsamWeight, commodityLimitID, yearlyLimit, locationalEligibility, capacityMin,
                        capacityMax, sharedPotentialID, capacityFix, isBuiltFix, investPerCapacity,
                        investIfBuilt, opexPerOperation, commodityCost, commodityRevenue,
                        opexPerCapacity, opexIfBuilt, interestRate, economicLifetime)

        self._sign = -1


class SourceSinkModeling(ComponentModeling):
    """ Doc """
    def __init__(self):
        self._componentsDict = {}
        self._capacityVariablesOptimum, self._isBuiltVariablesOptimum = None, None
        self._operationVariablesOptimum = None
        self._optSummary = None

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
        pyM.designDimensionVarSet_srcSnk = pyomo.Set(dimen=2, initialize=initDesignVarSet)

        def initContinuousDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_srcSnk
                    if compDict[compName]._capacityVariableDomain == 'continuous')
        pyM.continuousDesignDimensionVarSet_srcSnk = pyomo.Set(dimen=2, initialize=initContinuousDesignVarSet)

        def initDiscreteDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_srcSnk
                    if compDict[compName]._capacityVariableDomain == 'discrete')
        pyM.discreteDesignDimensionVarSet_srcSnk = pyomo.Set(dimen=2, initialize=initDiscreteDesignVarSet)

        def initDesignDecisionVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_srcSnk
                    if compDict[compName]._hasIsBuiltBinaryVariable)
        pyM.designDecisionVarSet_srcSnk = pyomo.Set(dimen=2, initialize=initDesignDecisionVarSet)

        ################################################################################################################
        #                                     Declare operation variables sets                                         #
        ################################################################################################################

        def initOpVarSet(pyM):
            return ((loc, compName) for loc in esM._locations for compName, comp in compDict.items()
                    if comp._locationalEligibility[loc] == 1)
        pyM.operationVarSet_srcSnk = pyomo.Set(dimen=2, initialize=initOpVarSet)
        pyM.operationVarDict_srcSnk = {loc: {compName for compName in compDict if (loc, compName)
                                             in pyM.operationVarSet_srcSnk} for loc in esM._locations}

        ################################################################################################################
        #                           Declare sets for case differentiation of operating modes                           #
        ################################################################################################################

        def initOpConstrSet1(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_srcSnk if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is None
                    and compDict[compName]._operationRateFix is None)
        pyM.opConstrSet1_srcSnk = pyomo.Set(dimen=2, initialize=initOpConstrSet1)

        def initOpConstrSet2(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_srcSnk if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet2_srcSnk = pyomo.Set(dimen=2, initialize=initOpConstrSet2)

        def initOpConstrSet3(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_srcSnk if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet3_srcSnk = pyomo.Set(dimen=2, initialize=initOpConstrSet3)

        def initOpConstrSet4(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_srcSnk if not
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet4_srcSnk = pyomo.Set(dimen=2, initialize=initOpConstrSet4)

        def initOpConstrSet5(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_srcSnk if not
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet5_srcSnk = pyomo.Set(dimen=2, initialize=initOpConstrSet5)

        yearlyCommodityLimitationDict = {}
        for compName, comp in self._componentsDict.items():
            if comp._commodityLimitID is not None:
                ID, limit = comp._commodityLimitID, comp._yearlyLimit
                if ID in yearlyCommodityLimitationDict and limit != yearlyCommodityLimitationDict[ID][0]:
                    raise ValueError('yearlyLimitationIDs with different upper limits detected.')
                yearlyCommodityLimitationDict.setdefault(ID, (limit, []))[1].append(compName)
        pyM.yearlyCommodityLimitationDict = yearlyCommodityLimitationDict

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

        # Capacity of components [powerUnit]
        pyM.cap_srcSnk = pyomo.Var(pyM.designDimensionVarSet_srcSnk, domain=pyomo.NonNegativeReals, bounds=capBounds)
        # Number of components [-]
        pyM.nbReal_srcSnk = pyomo.Var(pyM.continuousDesignDimensionVarSet_srcSnk, domain=pyomo.NonNegativeReals)
        # Number of components [-]
        pyM.nbInt_srcSnk = pyomo.Var(pyM.discreteDesignDimensionVarSet_srcSnk, domain=pyomo.NonNegativeIntegers)
        # Binary variables [-], indicate if a component is considered at a location or not
        pyM.designBin_srcSnk = pyomo.Var(pyM.designDecisionVarSet_srcSnk, domain=pyomo.Binary)
        # Operation of component [energyUnit]
        pyM.op_srcSnk = pyomo.Var(pyM.operationVarSet_srcSnk, pyM.timeSet, domain=pyomo.NonNegativeReals)

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
        def capToNbReal_srcSnk(pyM, loc, compName):
            return pyM.cap_srcSnk[loc, compName] == \
                   pyM.nbReal_srcSnk[loc, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbReal_srcSnk = pyomo.Constraint(pyM.continuousDesignDimensionVarSet_srcSnk,
                                                         rule=capToNbReal_srcSnk)

        # Determine the components' capacities from the number of installed units
        def capToNbInt_srcSnk(pyM, loc, compName):
            return pyM.cap_srcSnk[loc, compName] == \
                   pyM.nbInt_srcSnk[loc, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbInt_srcSnk = pyomo.Constraint(pyM.discreteDesignDimensionVarSet_srcSnk,
                                                       rule=capToNbInt_srcSnk)

        # Enforce the consideration of the binary design variables of a component
        def bigM_srcSnk(pyM, loc, compName):
            return pyM.cap_srcSnk[loc, compName] <= compDict[compName]._bigM * pyM.designBin_srcSnk[loc, compName]
        pyM.ConstrBigM_srcSnk = pyomo.Constraint(pyM.designDecisionVarSet_srcSnk, rule=bigM_srcSnk)

        # Enforce the consideration of minimum capacities for components with design decision variables
        def capacityMinDec_srcSnk(pyM, loc, compName):
            return (pyM.cap_srcSnk[loc, compName] >= compDict[compName]._capacityMin[loc] *
                    pyM.designBin_srcSnk[loc, compName] if compDict[compName]._capacityMin is not None
                    else pyomo.Constraint.Skip)
        pyM.ConstrCapacityMinDec_srcSnk = pyomo.Constraint(pyM.designDecisionVarSet_srcSnk, rule=capacityMinDec_srcSnk)

        # Sets, if applicable, the installed capacities of a component
        def capacityFix_srcSnk(pyM, loc, compName):
            return (pyM.cap_srcSnk[loc, compName] == compDict[compName]._capacityFix[loc]
                    if compDict[compName]._capacityFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrCapacityFix_srcSnk = pyomo.Constraint(pyM.designDimensionVarSet_srcSnk, rule=capacityFix_srcSnk)

        # Sets, if applicable, the binary design variables of a component
        def designBinFix_srcSnk(pyM, loc, compName):
            return (pyM.designBin_srcSnk[loc, compName] == compDict[compName]._isBuiltFix[loc]
                    if compDict[compName]._isBuiltFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrDesignBinFix_srcSnk = pyomo.Constraint(pyM.designDecisionVarSet_srcSnk, rule=designBinFix_srcSnk)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        def op1_srcSnk(pyM, loc, compName, p, t):
            return pyM.op_srcSnk[loc, compName, p, t] <= pyM.cap_srcSnk[loc, compName] * esM._hoursPerTimeStep
        pyM.ConstrOperation1_srcSnk = pyomo.Constraint(pyM.opConstrSet1_srcSnk, pyM.timeSet, rule=op1_srcSnk)

        # Operation [energyUnit] equal to the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op2_srcSnk(pyM, loc, compName, p, t):
            return pyM.op_srcSnk[loc, compName, p, t] == pyM.cap_srcSnk[loc, compName] * \
                   compDict[compName]._operationRateFix[loc][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation2_srcSnk = pyomo.Constraint(pyM.opConstrSet2_srcSnk, pyM.timeSet, rule=op2_srcSnk)

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op3_srcSnk(pyM, loc, compName, p, t):
            return pyM.op_srcSnk[loc, compName, p, t] <= pyM.cap_srcSnk[loc, compName] * \
                   compDict[compName]._operationRateMax[loc][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation3_srcSnk = pyomo.Constraint(pyM.opConstrSet3_srcSnk, pyM.timeSet, rule=op3_srcSnk)

        # Operation [energyUnit] equal to the operation time series [energyUnit]
        def op4_srcSnk(pyM, loc, compName, p, t):
            return pyM.op_srcSnk[loc, compName, p, t] == compDict[compName]._operationRateFix[loc][p, t]
        pyM.ConstrOperation4_srcSnk = pyomo.Constraint(pyM.opConstrSet4_srcSnk, pyM.timeSet, rule=op4_srcSnk)

        # Operation [energyUnit] limited by the operation time series [energyUnit]
        def op5_srcSnk(pyM, loc, compName, p, t):
            return pyM.op_srcSnk[loc, compName, p, t] <= compDict[compName]._operationRateMax[loc][p, t]
        pyM.ConstrOperation5_srcSnk = pyomo.Constraint(pyM.opConstrSet5_srcSnk, pyM.timeSet, rule=op5_srcSnk)

        def yearlyLimitationConstraint(pyM, key):
            limitDict = pyM.yearlyCommodityLimitationDict
            sumEx = -sum(pyM.op_srcSnk[loc, compName, p, t] * self._componentsDict[compName]._sign *
                         esM._periodOccurrences[p]/esM._numberOfYears
                         for loc, compName, p, t in pyM.op_srcSnk if compName in limitDict[key][1])
            sign = limitDict[key][0]/abs(limitDict[key][0]) if limitDict[key][0] != 0 else 1
            return sign * sumEx <= sign * limitDict[key][0]
        pyM.ConstryearlyLimitation = \
            pyomo.Constraint(pyM.yearlyCommodityLimitationDict.keys(), rule=yearlyLimitationConstraint)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        return sum(pyM.cap_srcSnk[loc, compName] / self._componentsDict[compName]._capacityMax[loc]
                   for compName in self._componentsDict if self._componentsDict[compName]._sharedPotentialID == key and
                   (loc, compName) in pyM.designDimensionVarSet_srcSnk)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([comp._commodity == commod and comp._locationalEligibility[loc] == 1
                    for comp in self._componentsDict.values()])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        return sum(pyM.op_srcSnk[loc, compName, p, t] * self._componentsDict[compName]._sign
                   for compName in pyM.operationVarDict_srcSnk[loc]
                   if self._componentsDict[compName]._commodity == commod)

    def getObjectiveFunctionContribution(self, esM, pyM):
        compDict = self._componentsDict

        capexCap = sum(compDict[compName]._investPerCapacity[loc] * pyM.cap_srcSnk[loc, compName] /
                       compDict[compName]._CCF[loc] for loc, compName in pyM.cap_srcSnk)

        capexDec = sum(compDict[compName]._investIfBuilt[loc] * pyM.designBin_srcSnk[loc, compName] /
                       compDict[compName]._CCF[loc] for loc, compName in pyM.designBin_srcSnk)

        opexCap = sum(compDict[compName]._opexPerCapacity[loc] * pyM.cap_srcSnk[loc, compName]
                      for loc, compName in pyM.cap_srcSnk)

        opexDec = sum(compDict[compName]._opexIfBuilt[loc] * pyM.designBin_srcSnk[loc, compName]
                      for loc, compName in pyM.designBin_srcSnk)

        opexOp = sum(compDict[compName]._opexPerOperation[loc] *
                     sum(pyM.op_srcSnk[loc, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet)
                     for loc, compNames in pyM.operationVarDict_srcSnk.items()
                     for compName in compNames) / esM._numberOfYears

        commodCosts = sum((compDict[compName]._commodityCost[loc] - compDict[compName]._commodityRevenue[loc]) *
                          sum(pyM.op_srcSnk[loc, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet)
                          for loc, compNames in pyM.operationVarDict_srcSnk.items()
                          for compName in compNames) / esM._numberOfYears

        return capexCap + capexDec + opexCap + opexDec + opexOp + commodCosts

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        compDict = self._componentsDict
        props = ['capacity', 'isBuilt', 'operation', 'capexCap', 'capexIfBuilt', 'opexCap', 'opexIfBuilt',
                 'opexOp', 'commodCosts', 'TAC', 'invest']
        units = ['[-]', '[-]', '[-]', '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]',
                 '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]',
                 '[' + esM._costUnit + '/a]', '[' + esM._costUnit + ']', '[' + esM._costUnit + ']']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._commodityUnit + ']') if x[1] == 'capacity'
                          else x, tuples))
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._commodityUnit + '*h/a]') if x[1] == 'operation'
                          else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM._locations)).sort_index()

        # Get optimal variable values and contributions to the total annual cost and invest
        optVal = utils.formatOptimizationOutput(pyM.cap_srcSnk.get_values(), 'designVariables', '1dim')
        self._capacityVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_capacityVariablesOptimum', compDict)

        if optVal is not None:
            i = optVal.apply(lambda cap: cap * compDict[cap.name]._investPerCapacity[cap.index], axis=1)
            cx = optVal.apply(lambda cap: cap * compDict[cap.name]._investPerCapacity[cap.index] /
                              compDict[cap.name]._CCF[cap.index], axis=1)
            ox = optVal.apply(lambda cap: cap*compDict[cap.name]._opexPerCapacity[cap.index], axis=1)
            optSummary.loc[[(ix, 'capacity', '[' + compDict[ix]._commodityUnit + ']') for ix in optVal.index],
                            optVal.columns] = optVal.values
            optSummary.loc[[(ix, 'invest', '[' + esM._costUnit + ']') for ix in i.index], i.columns] = i.values
            optSummary.loc[[(ix, 'capexCap', '[' + esM._costUnit + '/a]') for ix in cx.index], cx.columns] = cx.values
            optSummary.loc[[(ix, 'opexCap', '[' + esM._costUnit + '/a]') for ix in ox.index], ox.columns] = ox.values

        optVal = utils.formatOptimizationOutput(pyM.designBin_srcSnk.get_values(), 'designVariables', '1dim')
        self._isBuiltVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_isBuiltVariablesOptimum', compDict)

        if optVal is not None:
            i = optVal.apply(lambda dec: dec * compDict[dec.name]._investIfBuilt[dec.index], axis=1)
            cx = optVal.apply(lambda dec: dec * compDict[dec.name]._investIfBuilt[dec.index] /
                              compDict[dec.name]._CCF[dec.index], axis=1)
            ox = optVal.apply(lambda dec: dec * compDict[dec.name]._opexIfBuilt[dec.index], axis=1)
            optSummary.loc[[(ix, 'isBuilt', '[-]') for ix in optVal.index], optVal.columns] = optVal.values
            optSummary.loc[[(ix, 'invest', '[' + esM._costUnit + ']') for ix in cx.index], cx.columns] += i.values
            optSummary.loc[[(ix, 'capexIfBuilt', '[' + esM._costUnit + '/a]') for ix in cx.index],
                            cx.columns] = cx.values
            optSummary.loc[[(ix, 'opexIfBuilt', '[' + esM._costUnit + '/a]') for ix in ox.index],
                            ox.columns] = ox.values

        optVal = utils.formatOptimizationOutput(pyM.op_srcSnk.get_values(), 'operationVariables', '1dim',
                                                esM._periodsOrder)
        self._operationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_operationVariablesOptimum', compDict)

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name]._opexPerOperation[op.index], axis=1)
            cCost = opSum.apply(lambda op: op * compDict[op.name]._commodityCost[op.index], axis=1)
            cRevenue = opSum.apply(lambda op: op * compDict[op.name]._commodityRevenue[op.index], axis=1)
            optSummary.loc[[(ix, 'operation', '[' + compDict[ix]._commodityUnit + '*h/a]') for ix in opSum.index],
                            opSum.columns] = opSum.values
            optSummary.loc[[(ix, 'opexOp', '[' + esM._costUnit + '/a]') for ix in ox.index], ox.columns] = ox.values
            optSummary.loc[[(ix, 'commodCosts', '[' + esM._costUnit + '/a]') for ix in ox.index], ox.columns] = \
                (cCost-cRevenue).values

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'capexCap') |
                            (optSummary.index.get_level_values(1) == 'opexCap') |
                            (optSummary.index.get_level_values(1) == 'capexIfBuilt') |
                            (optSummary.index.get_level_values(1) == 'opexIfBuilt') |
                            (optSummary.index.get_level_values(1) == 'opexOp') |
                            (optSummary.index.get_level_values(1) == 'commodCosts') ].groupby(level=0).sum().values

        self._optSummary = optSummary

    def getOptimalCapacities(self):
        return self._capacitiesOpt