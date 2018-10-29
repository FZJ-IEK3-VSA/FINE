from FINE.component import Component, ComponentModeling
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

        **Required arguments:**

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param name: name of the component. Has to be unique (i.e. no other components with that name can
        already exist in the EnergySystemModel instance to which the component is added).
        :type name: string

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

        :param hasCapacityVariable: specifies if the component should be modeled with a capacity or not.
            Examples:
            (a) A electricity cable has a capacity given in GW_electric -> hasCapacityVariable is True.
            (b) If the transmission capacity of the component is unlimited hasCapacityVariable is False.
            |br| * the default value is True
        :type hasCapacityVariable: boolean

        :param capacityVariableDomain: the mathematical domain of the capacity variables, if they are specified.
            By default, the domain is specified as 'continuous' and thus declares the variables as positive
            (>=0) real values. The second input option that is available for this parameter is 'discrete', which
            declares the variables as positive (>=0) integer values.
            |br| * the default value is 'continuous'
        :type capacityVariableDomain: string ('continuous' or 'discrete')

        :param capacityPerPlantUnit: capacity of one plant of the component (in the respective commodityUnit
            of the plant). The default is 1, thus the number of plants is equal to the installed capacity.
            This parameter should be specified when using a 'discrete' capacityVariableDomain.
            It can be specified when using a 'continuous' variable domain.
            |br| * the default value is 1
        :type capacityPerPlantUnit: strictly positive float

        :param hasIsBuiltBinaryVariable: specifies if binary decision variables should be declared for each
            eligible connection of the transmission component, which indicate if the component is built
            between two locations or not. The binary variables can be used to enforce one-time investment cost
            or capacity-independent annual operation cost. If a minimum capacity is specified and this parameter
            is set to True, the minimum capacities are only considered if a component is built (i.e. if a
            component is built between two locations, it has to be built with a minimum capacity of XY GW,
            otherwise it is set to 0 GW).
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

        :param operationRateMax: if specified, indicates a maximum operation rate for all possible connections
            (both directions) of the transmission component at each time step by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            in that case a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The DataFrame has to have two index
            columns. The first one indicates the location the commodity is coming from. The second one indicates
            the location the commodity is going too. These indices have to match the in the energy system model
            specified locations. If a flow is specified from location i to location j, it also has to be
            specified from j to i.

        :param operationRateFix: if specified, indicates a fixed operation rate for all possible connections
            (both directions) of the transmission component at each time step by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            in that case a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The DataFrame has to have two index
            columns. The first one indicates the location the commodity is coming from. The second one indicates
            the location the commodity is going too. These indices have to match the in the energy system model
            specified locations. If a flow is specified from location i to location j, it also has to be
            specified from j to i.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param locationalEligibility: Pandas DataFrame that indicates if a component can be built between two
            locations (=1) or not (=0). If not specified and a maximum or fixed capacity or time series is given,
            the parameter will be set based on these inputs. If the parameter is specified, a consistency check
            is done to ensure that the parameters indicate the same locational eligibility. If the parameter is
            not specified and also no other of the parameters is specified it is assumed that the component is
            eligible in each location and all values are set to 1.
            This parameter is key for ensuring small built times of the optimization problem by avoiding the
            declaration of unnecessary variables and constraints.
            |br| * the default value is None
        :type locationalEligibility: None or Pandas DataFrame with values equal to 0 and 1. The indices of the
            DataFrame have to equal the in the energy system model specified locations.

        :param capacityMin: if specified, Pandas DataFrame indicating minimum capacities (in the respective
            commodityUnit) else None. If binary decision variables are declared, which indicate if a component
            is built between two locations or not, the minimum capacity is only enforced if the component is
            built (i.e. if a component is built between two locations, it has to be built with a minimum
            capacity of XY GW, otherwise it is set to 0 GW).
            |br| * the default value is None
        :type capacityMin: None or Pandas DataFrame with positive (>=0) values. The row and column indices of
            the DataFrame have to equal the in the energy system model specified locations.

        :param capacityMax: if specified, Pandas Series indicating maximum capacities (in the respective
            commodityUnit) else None.
            |br| * the default value is None
        :type capacityMax: None or Pandas DataFrame with positive (>=0) values. The indices of the DataFrame
            have to equal the in the energy system model specified locations.

        :param sharedPotentialID: if specified, indicates that the component has to share its maximum
            potential capacity with other components (i.e. due to space limitations). The shares of how
            much of the maximum potential is used have to add up to less then 100%.
            |br| * the default value is None
        :type sharedPotentialID: string

        :param capacityFix: if specified, Pandas Series indicating fixed capacities (in the respective
            commodityUnit) else None.
            |br| * the default value is None
        :type capacityFix: None or Pandas Series with positive (>=0) values. The row and column indices of
            the series have to equal the in the energy system model specified locations.

        :param isBuiltFix: if specified, Pandas DataFrame indicating fixed decisions between which locations
            the component is built else None (i.e. sets the isBuilt binary variables).
            |br| * the default value is None
        :type isBuiltFix: None or Pandas DataFrame with values equal to 0 and 1. The row and column indices
            of the DataFrame have to equal the in the energy system model specified locations.

        :param investPerCapacity: the invest of a component is obtained by multiplying the capacity of the
            component (in the physicalUnit of the component) between two locations with the
            investPerCapacity factor. The investPerCapacity can either be given as a float or a Pandas
            DataFrame with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro) divided by the in the energy system specified
            lengthUnit.
            |br| * the default value is 0
        :type investPerCapacity: positive (>=0) float or Pandas DataFrame with positive (>=0) values.
            The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.

        :param investIfBuilt: a capacity-independent invest which only arises between two locations if a
            component is built between those locations. The investIfBuilt can either be given as a float
            or a Pandas DataFrame with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro) divided by the in the energy system specified
            lengthUnit.
            |br| * the default value is 0
        :type investIfBuilt: positive (>=0) float or Pandas DataFrame with positive (>=0) values.
            The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.

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

        :param opexPerCapacity: annual operational cost which are only a function of the capacity of the
            component (in the physicalUnit of the component) and not of the specific operation itself are
            obtained by multiplying the capacity of the component at between two locations with the
            opexPerCapacity factor. The opexPerCapacity can either be given as a float or a Pandas
            DataFrame with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro) divided by the in the energy system specified
            lengthUnit.
            |br| * the default value is 0
        :type opexPerCapacity: positive (>=0) float or Pandas DataFrame with positive (>=0) values.
            The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.

        :param opexIfBuilt: a capacity-independent annual operational cost which only arises between two
            locations if a component is built at that location. The opexIfBuilt can either be given as a
            float or a Pandas DataFrame with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (i.e. Euro, Dollar, 1e6 Euro) divided by the in the energy system specified
            lengthUnit.
            |br| * the default value is 0
        :type opexIfBuilt: positive (>=0) float or Pandas DataFrame with positive (>=0) values.
            The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.

        :param interestRate: interest rate which is considered for computing the annuities of the invest
            of the component (depreciates the invests over the economic lifetime). It can either be given
            as a float or a Pandas DataFrame with location specific values.
            A value of 0.08 corresponds to an interest rate of 8%.
            |br| * the default value is 0.08
        :type interestRate: positive (>=0) float or Pandas DataFrame with positive (>=0) values.
            The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.

        :param economicLifetime: economic lifetime of the component which is considered for computing the
            annuities of the invest of the component (aka depreciation time). It can either be given as a
            float or a Pandas DataFrame with location specific values.
            |br| * the default value is 10
        :type economicLifetime: strictly-positive (>0) float or Pandas DataFrame with strictly-positive (>=0)
            values. The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.
        """

        # TODO add unit checks
        # Set general component data
        utils.isEnergySystemModelInstance(esM), utils.checkCommodities(esM, {commodity})
        self._name, self._commodity, self._commodityUnit = name, commodity, esM._commoditiyUnitsDict[commodity]
        self._distances = utils.checkAndSetDistances(esM, distances)
        self._losses = utils.checkAndSetTransmissionLosses(esM, losses, distances)

        # Set design variable modeling parameters
        utils.checkDesignVariableModelingParameters(capacityVariableDomain, hasCapacityVariable, capacityPerPlantUnit,
                                                    hasIsBuiltBinaryVariable, bigM)
        self._hasCapacityVariable = hasCapacityVariable
        self._capacityVariableDomain = capacityVariableDomain
        self._capacityPerPlantUnit = capacityPerPlantUnit
        self._hasIsBuiltBinaryVariable = hasIsBuiltBinaryVariable
        self._bigM = bigM

        # Set economic data
        self._investPerCapacity = utils.checkAndSetCostParameter(esM, name, investPerCapacity, '2dim')
        self._investIfBuilt = utils.checkAndSetCostParameter(esM, name, investIfBuilt, '2dim')
        self._opexPerOperation = utils.checkAndSetCostParameter(esM, name, opexPerOperation, '2dim')
        self._opexPerCapacity = utils.checkAndSetCostParameter(esM, name, opexPerCapacity, '2dim')
        self._opexIfBuilt = utils.checkAndSetCostParameter(esM, name, opexIfBuilt, '2dim')
        self._interestRate = utils.checkAndSetCostParameter(esM, name, interestRate, '2dim')
        self._economicLifetime = utils.checkAndSetCostParameter(esM, name, economicLifetime, '2dim')
        self._CCF = utils.getCapitalChargeFactor(self._interestRate, self._economicLifetime)

        # Set location-specific operation parameters
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            warnings.warn('If operationRateFix is specified, the operationRateMax parameter is not required.\n' +
                          'The operationRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateMax, locationalEligibility, '2dim')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateFix, locationalEligibility, '2dim')

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
                                                    '2dim')
        self._capacityMin, self._capacityMax, self._capacityFix = capacityMin, capacityMax, capacityFix
        self._isBuiltFix = isBuiltFix

        # Set locational eligibility
        operationTimeSeries = operationRateFix if operationRateFix is not None else operationRateMax
        self._locationalEligibility = utils.setLocationalEligibility(esM, locationalEligibility, capacityMax,
                                                                     capacityFix, isBuiltFix,
                                                                     hasCapacityVariable, operationTimeSeries,
                                                                     '2dim')

        # Variables at optimum (set after optimization)
        self._capacityVariablesOptimum = None
        self._isBuiltVariablesOptimum = None
        self._operationVariablesOptimum = None

    def addToEnergySystemModel(self, esM):
        esM._isTimeSeriesDataClustered = False
        if self._name in esM._componentNames:
            if esM._componentNames[self._name] == TransmissionModeling.__name__:
                warnings.warn('Component identifier ' + self._name + ' already exists. Data will be overwritten.')
            else:
                raise ValueError('Component name ' + self._name + ' is not unique.')
        else:
            esM._componentNames.update({self._name: TransmissionModeling.__name__})
        mdl = TransmissionModeling.__name__
        if mdl not in esM._componentModelingDict:
            esM._componentModelingDict.update({mdl: TransmissionModeling()})
        esM._componentModelingDict[mdl]._componentsDict.update({self._name: self})

    def setTimeSeriesData(self, hasTSA):
        self._operationRateMax = self._aggregatedOperationRateMax if hasTSA else self._fullOperationRateMax
        self._operationRateFix = self._aggregatedOperationRateFix if hasTSA else self._fullOperationRateFix

    def getDataForTimeSeriesAggregation(self):
        fullOperationRate = self._fullOperationRateFix if self._fullOperationRateFix is not None \
            else self._fullOperationRateMax
        if fullOperationRate is not None:
            fullOperationRate = fullOperationRate.copy()
            uniqueIdentifiers = [self._name + "_operationRate_" + locationIn + '_' + locationOut
                                 for locationIn, locationOut in fullOperationRate.columns]
            compData = pd.DataFrame(index=fullOperationRate.index, columns=uniqueIdentifiers)
            compDict = {}
            for locationIn, locationOut in fullOperationRate.columns:
                uniqueIdentifier = self._name + "_operationRate_" + locationIn + '_' + locationOut
                compData[uniqueIdentifier] = fullOperationRate.pop((locationIn, locationOut))
                compDict.update({uniqueIdentifier: self._tsaWeight})
            return compData, compDict
        else:
            return None, {}

    def setAggregatedTimeSeriesData(self, data):
        fullOperationRate = self._fullOperationRateFix if self._fullOperationRateFix is not None \
            else self._fullOperationRateMax
        if fullOperationRate is not None:
            uniqueIdentifiers = [self._name + "_operationRate_" + locationIn + '_' + locationOut
                                 for locationIn, locationOut in fullOperationRate.columns]
            compData = data[uniqueIdentifiers].copy()
            compData = pd.DataFrame(index=data.index, columns=fullOperationRate.columns)
            for locationIn, locationOut in compData.columns:
                compData.loc[:, (locationIn, locationOut)] = \
                    data.loc[:, self._name + "_operationRate_" + locationIn + '_' + locationOut]
            if self._fullOperationRateFix is not None:
                self._aggregatedOperationRateFix = compData
            else:
                self._aggregatedOperationRateMax = compData


class TransmissionModeling(ComponentModeling):
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
            return ((loc, loc_, compName) for loc in esM._locations for loc_ in esM._locations
                    for compName, comp in compDict.items()
                    if comp._locationalEligibility[loc][loc_] == 1 and comp._hasCapacityVariable)
        pyM.designDimensionVarSet_trans = pyomo.Set(dimen=3, initialize=initDesignVarSet)

        def initContinuousDesignVarSet(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName, in pyM.designDimensionVarSet_trans
                    if compDict[compName]._capacityVariableDomain == 'continuous')
        pyM.continuousDesignDimensionVarSet_trans = pyomo.Set(dimen=3, initialize=initContinuousDesignVarSet)

        def initDiscreteDesignVarSet(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.designDimensionVarSet_trans
                    if compDict[compName]._capacityVariableDomain == 'discrete')
        pyM.discreteDesignDimensionVarSet_trans = pyomo.Set(dimen=3, initialize=initDiscreteDesignVarSet)

        def initDesignDecisionVarSet(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.designDimensionVarSet_trans
                    if compDict[compName]._hasIsBuiltBinaryVariable)
        pyM.designDecisionVarSet_trans = pyomo.Set(dimen=3, initialize=initDesignDecisionVarSet)

        ################################################################################################################
        #                                     Declare operation variables sets                                         #
        ################################################################################################################

        def initOpVarSet(pyM):
            return ((loc, loc_, compName) for loc in esM._locations for loc_ in esM._locations
                    for compName, comp in compDict.items() if comp._locationalEligibility[loc][loc_] == 1)
        pyM.operationVarSet_trans = pyomo.Set(dimen=3, initialize=initOpVarSet)
        pyM.operationVarDict_transOut = {loc: {loc_: {compName for compName in compDict
                                                      if (loc, loc_, compName) in pyM.operationVarSet_trans}
                                               for loc_ in esM._locations} for loc in esM._locations}
        pyM.operationVarDict_transIn = {loc: {loc_: {compName for compName in compDict
                                                     if (loc_, loc, compName) in pyM.operationVarSet_trans}
                                              for loc_ in esM._locations} for loc in esM._locations}

        ################################################################################################################
        #                           Declare sets for case differentiation of operating modes                           #
        ################################################################################################################

        def initOpConstrSet1(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is None
                    and compDict[compName]._operationRateFix is None)
        pyM.opConstrSet1_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet1)

        def initOpConstrSet2(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet2_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet2)

        def initOpConstrSet3(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet3_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet3)

        def initOpConstrSet4(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if not
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet4_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet4)

        def initOpConstrSet5(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if not
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet5_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet5)

        potentialDict = {} # TODO adapt for 2dim components
        for compName, comp in compDict.items():
            if comp._sharedPotentialID is not None:
                potentialDict.setdefault(comp._sharedPotentialID, []).append(compName)
        pyM.sharedPotentialTransmissionDict = potentialDict

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """ Declares design and operation variables """
        # Function for setting lower and upper capacity bounds
        def capBounds(pyM, loc, loc_, compName):
            comp = self._componentsDict[compName]
            return (comp._capacityMin[loc][loc_]
                    if (comp._capacityMin is not None and not comp._hasIsBuiltBinaryVariable) else 0,
                    comp._capacityMax[loc][loc_] if comp._capacityMax is not None else None)

        # Capacity of components [powerUnit]
        pyM.cap_trans = pyomo.Var(pyM.designDimensionVarSet_trans, domain=pyomo.NonNegativeReals, bounds=capBounds)
        # Number of components [-]
        pyM.nbReal_trans = pyomo.Var(pyM.continuousDesignDimensionVarSet_trans, domain=pyomo.NonNegativeReals)
        # Number of components [-]
        pyM.nbInt_trans = pyomo.Var(pyM.discreteDesignDimensionVarSet_trans, domain=pyomo.NonNegativeIntegers)
        # Binary variables [-], indicate if a component is considered at a location or not
        pyM.designBin_trans = pyomo.Var(pyM.designDecisionVarSet_trans, domain=pyomo.Binary)
        # Operation of component [energyUnit]
        pyM.op_trans = pyomo.Var(pyM.operationVarSet_trans, pyM.timeSet, domain=pyomo.NonNegativeReals)

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
        def capToNbReal_trans(pyM, loc, loc_, compName):
            return pyM.cap_trans[loc, loc_, compName] == \
                   pyM.nbReal_trans[loc, loc_, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbReal_trans = pyomo.Constraint(pyM.continuousDesignDimensionVarSet_trans,
                                                       rule=capToNbReal_trans)

        # Determine the components' capacities from the number of installed units
        def capToNbInt_trans(pyM, loc, loc_, compName):
            return pyM.cap_trans[loc, loc_, compName] == \
                   pyM.nbInt_trans[loc, loc_, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbInt_trans = pyomo.Constraint(pyM.discreteDesignDimensionVarSet_trans,
                                                      rule=capToNbInt_trans)

        # Enforce the consideration of the binary design variables of a component
        def bigM_trans(pyM, loc, loc_, compName):
            return pyM.cap_trans[loc, loc_, compName] <= \
                   compDict[compName]._bigM * pyM.designBin_trans[loc, loc_, compName]
        pyM.ConstrBigM_trans = pyomo.Constraint(pyM.designDecisionVarSet_trans, rule=bigM_trans)

        # Enforce the consideration of minimum capacities for components with design decision variables
        def capacityMinDec_trans(pyM, loc, loc_, compName):
            return (pyM.cap_trans[loc, loc_, compName] >= compDict[compName]._capacityMin[loc][loc_] *
                    pyM.designBin_trans[loc, loc_, compName] if compDict[compName]._capacityMin is not None
                    else pyomo.Constraint.Skip)
        pyM.ConstrCapacityMinDec_trans = pyomo.Constraint(pyM.designDecisionVarSet_trans, rule=capacityMinDec_trans)

        # Sets, if applicable, the installed capacities of a component
        def capacityFix_trans(pyM, loc, loc_, compName):
            return (pyM.cap_trans[loc, loc_, compName] == compDict[compName]._capacityFix[loc][loc_]
                    if compDict[compName]._capacityFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrCapacityFix_trans = pyomo.Constraint(pyM.designDimensionVarSet_trans, rule=capacityFix_trans)

        # Sets, if applicable, the binary design variables of a component
        def designBinFix_trans(pyM, loc, loc_, compName):
            return (pyM.designBin_trans[loc, loc_, compName] == compDict[compName]._isBuiltFix[loc][loc_]
                    if compDict[compName]._isBuiltFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrDesignBinFix_trans = pyomo.Constraint(pyM.designDecisionVarSet_trans, rule=designBinFix_trans)

        def sharedPotentialTransmission(pyM, key, loc, loc_):
            return sum(pyM.cap_trans[loc, loc_, compName] / compDict[compName].capacityMax[loc][loc_]
                       for compName in compDict if compDict[compName]._sharedPotentialID == key
                       and (loc, loc_, compName) in pyM.designDimensionVarSet_trans)
        pyM.ConstSharedPotential_trans = \
            pyomo.Constraint(pyM.sharedPotentialTransmissionDict.keys(), esM._locations, esM._locations,
                             rule=sharedPotentialTransmission)

        def symmetricalCapacity_trans(pyM, loc, loc_, compName):
            return pyM.cap_trans[loc, loc_, compName] == pyM.cap_trans[loc_, loc, compName]
        pyM.ConstrSymmetricalCapacity_trans = \
            pyomo.Constraint(pyM.designDimensionVarSet_trans, rule=symmetricalCapacity_trans)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        # Since the flow should either go in one direction or the other, the limitation can be enforced on the sum
        # of the forward and backward flow over the line. This leads to one of the flow variables being set to zero
        # if a basic solution is obtained during optimization.
        def op1_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] + pyM.op_trans[loc_, loc, compName, p, t] <= \
                   pyM.cap_trans[loc, loc_, compName] * esM._hoursPerTimeStep
        pyM.ConstrOperation1_trans = pyomo.Constraint(pyM.opConstrSet1_trans, pyM.timeSet, rule=op1_trans)

        # Operation [energyUnit] equal to the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op2_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] == pyM.cap_trans[loc, loc_, compName] * \
                   compDict[compName]._operationRateFix[loc, loc_][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation2_trans = pyomo.Constraint(pyM.opConstrSet2_trans, pyM.timeSet, rule=op2_trans)

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op3_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] <= pyM.cap_trans[loc, loc_, compName] * \
                   compDict[compName]._operationRateMax[loc, loc_][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation3_trans = pyomo.Constraint(pyM.opConstrSet3_trans, pyM.timeSet, rule=op3_trans)

        # Operation [energyUnit] equal to the operation time series [energyUnit]
        def op4_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] == compDict[compName]._operationRateFix[loc, loc_][p, t]
        pyM.ConstrOperation4_trans = pyomo.Constraint(pyM.opConstrSet4_trans, pyM.timeSet, rule=op4_trans)

        # Operation [energyUnit] limited by the operation time series [energyUnit]
        def op5_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] <= compDict[compName]._operationRateMax[loc, loc_][p, t]
        pyM.ConstrOperation5_trans = pyomo.Constraint(pyM.opConstrSet5_trans, pyM.timeSet, rule=op5_trans)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        return 0

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([comp._commodity == commod and
                    (comp._locationalEligibility[loc][loc_] == 1 or comp._locationalEligibility[loc_][loc] == 1)
                    for comp in self._componentsDict.values() for loc_ in esM._locations])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        return sum(pyM.op_trans[loc_, loc, compName, p, t] *
                   (1 - self._componentsDict[compName]._losses[loc_][loc] *
                    self._componentsDict[compName]._distances[loc_][loc])
                   for loc_ in pyM.operationVarDict_transIn[loc].keys()
                   for compName in pyM.operationVarDict_transIn[loc][loc_]
                   if commod in self._componentsDict[compName]._commodity) - \
               sum(pyM.op_trans[loc, loc_, compName, p, t]
                   for loc_ in pyM.operationVarDict_transOut[loc].keys()
                   for compName in pyM.operationVarDict_transOut[loc][loc_]
                   if commod in self._componentsDict[compName]._commodity)

    def getObjectiveFunctionContribution(self, esM, pyM):
        compDict = self._componentsDict

        capexDim = sum(compDict[compName]._investPerCapacity[loc][loc_] * pyM.cap_trans[loc, loc_, compName] *
                       compDict[compName]._distances[loc][loc_] /
                       compDict[compName]._CCF[loc][loc_] for loc, loc_, compName in pyM.cap_trans) * 0.5

        capexDec = sum(compDict[compName]._investIfBuilt[loc][loc_] *
                       pyM.designBin_trans[loc, loc_, compName] * compDict[compName]._distances[loc][loc_] /
                       compDict[compName]._CCF[loc][loc_] for loc, loc_, compName in pyM.designBin_trans) * 0.5

        opexDim = sum(compDict[compName]._opexPerCapacity[loc][loc_] * pyM.cap_trans[loc, loc_, compName] *
                      compDict[compName]._distances[loc][loc_] for loc, loc_, compName in pyM.cap_trans) * 0.5

        opexDec = sum(compDict[compName]._opexIfBuilt[loc][loc_] * pyM.designBin_trans[loc, loc_, compName] *
                      compDict[compName]._distances[loc][loc_] for loc, loc_, compName in pyM.designBin_trans) * 0.5

        opexOp = sum(compDict[compName]._opexPerOperation[loc][loc_] *
                     sum(pyM.op_trans[loc, loc_, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet)
                     for loc, subDict in pyM.operationVarDict_transOut.items()
                     for loc_, compNames in subDict.items()
                     for compName in compNames) / esM._numberOfYears

        return capexDim + capexDec + opexDim + opexDec + opexOp

    def setOptimalValues(self, esM, pyM):
        compDict = esM._componentModelingDict['TransmissionModeling']._componentsDict

        props = ['capacity', 'isBuilt', 'operation', 'capexCap', 'capexIfBuilt', 'opexCap', 'opexIfBuilt',
                 'opexOp', 'TAC', 'invest']
        units = ['[-]', '[-]', '[-]', '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]',
                 '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]',
                 '[' + esM._costUnit + '/a]', '[' + esM._costUnit + ']']
        tuples = [(compName, prop, unit, location) for compName in compDict.keys() for prop, unit in zip(props, units)
                  for location in sorted(esM._locations)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._commodityUnit + ']', x[3])
                          if x[1] == 'capacity' else x, tuples))
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._commodityUnit + '*h/a]', x[3])
                          if x[1] == 'operation' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit', 'Location'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM._locations)).sort_index()

        # Get optimal variable values and contributions to the total annual cost and invest
        optVal = utils.formatOptimizationOutput(pyM.cap_trans.get_values(), 'designVariables', '2dim')
        self._capacityVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_capacityVariablesOptimum', self._componentsDict)

        if optVal is not None:
            i = optVal.apply(lambda cap: cap * compDict[cap.name[0]]._investPerCapacity[cap.name[1]][cap.index] *
                              compDict[cap.name[0]]._distances[cap.name[1]][cap.index], axis=1).fillna(0)*0.5
            cx = optVal.apply(lambda cap: cap * compDict[cap.name[0]]._investPerCapacity[cap.name[1]][cap.index] *
                              compDict[cap.name[0]]._distances[cap.name[1]][cap.index] /
                              compDict[cap.name[0]]._CCF[cap.name[1]][cap.index], axis=1).fillna(0)*0.5
            ox = optVal.apply(lambda cap: cap * compDict[cap.name[0]]._opexPerCapacity[cap.name[1]][cap.index] *
                              compDict[cap.name[0]]._distances[cap.name[1]][cap.index], axis=1).fillna(0)*0.5
            optSummary.loc[[(ix1, 'capacity', '[' + compDict[ix1]._commodityUnit + ']', ix2)
                             for ix1, ix2 in optVal.index], optVal.columns] = optVal.values
            optSummary.loc[[(ix1, 'invest', '[' + esM._costUnit + ']', ix2) for ix1, ix2 in i.index],
                            i.columns] = i.values
            optSummary.loc[[(ix1, 'capexCap', '[' + esM._costUnit + '/a]', ix2) for ix1, ix2 in cx.index],
                            cx.columns] = cx.values
            optSummary.loc[[(ix1, 'opexCap', '[' + esM._costUnit + '/a]', ix2) for ix1, ix2 in ox.index],
                            ox.columns] = ox.values

        optVal = utils.formatOptimizationOutput(pyM.designBin_trans.get_values(), 'designVariables', '2dim')
        self._isBuiltVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_isBuiltVariablesOptimum', self._componentsDict)

        if optVal is not None:
            i = optVal.apply(lambda dec: dec * compDict[dec.name[0]]._investIfBuilt[dec.name[1]][dec.index] *
                              compDict[dec.name[0]]._distances[dec.name[1]][dec.index], axis=1).fillna(0)*0.5
            cx = optVal.apply(lambda dec: dec * compDict[dec.name[0]]._investIfBuilt[dec.name[1]][dec.index] *
                              compDict[dec.name[0]]._distances[dec.name[1]][dec.index] /
                              compDict[dec.name[0]]._CCF[dec.name[1]][dec.index], axis=1).fillna(0)*0.5
            ox = optVal.apply(lambda dec: dec * compDict[dec.name[0]]._opexIfBuilt[dec.name[1]][dec.index] *
                              compDict[dec.name[0]]._distances[dec.name[1]][dec.index], axis=1).fillna(0)*0.5
            optSummary.loc[[(ix1, 'isBuilt', '[-]', ix2) for ix1, ix2 in optVal.index], optVal.columns] = optVal.values
            optSummary.loc[[(ix1, 'invest', '[' + esM._costUnit + ']', ix2) for ix1, ix2 in i.index],
                            i.columns] += i.values
            optSummary.loc[[(ix1, 'capexIfBuilt', '[' + esM._costUnit + '/a]', ix2) for ix1, ix2 in cx.index],
                            cx.columns] = cx.values
            optSummary.loc[[(ix1, 'opexIfBuilt', '[' + esM._costUnit + '/a]', ix2) for ix1, ix2 in ox.index],
                            ox.columns] = ox.values

        optVal = utils.formatOptimizationOutput(pyM.op_trans.get_values(), 'operationVariables', '2dim',
                                                esM._periodsOrder)
        self._operationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_operationVariablesOptimum', self._componentsDict)

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name[0]]._opexPerOperation[op.name[1]][op.index], axis=1)*0.5
            optSummary.loc[[(ix1, 'operation', '[' + compDict[ix1]._commodityUnit + '*h/a]', ix2)
                             for ix1, ix2 in opSum.index], opSum.columns] = opSum.values
            optSummary.loc[[(ix1, 'opexOp', '[' + esM._costUnit + '/a]', ix2) for ix1, ix2 in ox.index],
                            ox.columns] = ox.values

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'capexCap') |
                            (optSummary.index.get_level_values(1) == 'opexCap') |
                            (optSummary.index.get_level_values(1) == 'capexIfBuilt') |
                            (optSummary.index.get_level_values(1) == 'opexIfBuilt') |
                            (optSummary.index.get_level_values(1) == 'opexOp')].groupby(level=[0, 3]).sum().values

        self._optSummary = optSummary

    def getOptimalCapacities(self):
        return self._capacitiesOpt