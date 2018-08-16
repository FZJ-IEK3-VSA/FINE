from FINE.component import Component, ComponentModeling
from FINE import utils
import pyomo.environ as pyomo
import warnings
import pandas as pd


class Conversion(Component):
    #TODO
    """
    Conversion class

    The functionality of the the EnergySystemModel class is fourfold:
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
        """
        Constructor for creating an Conversion class instance

        **Required arguments:**

        :param esM: energy system model to which the Conversion component should be added.
            Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param name: name of the component. Has to be a unique (i.e. not other components
            with that name can already exist in the energy system model to which the
            component is added).
        :type name: string

        :param physicalUnit: reference physical unit of the plants to which maximum capacity
            limitations and cost parameters refer to.
        :type physicalUnit: string

        :param commodityConversionFactors: conversion factors with which commodities are
            converted into each other with one unit of operation (dictionary). Each
            commodity which is converted in this component is indicated by a string in this
            dictionary. The conversion factor related to this commodity is given as a float.
            A negative value indicates that the commoditiy is consumed. A positive value
            indicates that the commodity is produced. Check unit consistency when specifying
            this parameter!
            Examples:
            (a) An electrolyzer converts, simply put, electricty into hydrogen with an
                efficiency of 70%. The cost are given in GW_electric (physicalUnit of the
                plant), the unit for the 'electricity' commodity is given in GW_electric
                and the 'hydrogen' commodity is given in
                GW_hydrogen_lowerHeatingValue -> the commodityConversionFactors are defined
                as {'electricity':-1,'hydrogen':0.7}.
            (b) An fuel cell converts, simply put, hydrogen into electricty with an
                efficiency of 60%. The cost are given in GW_electric (physicalUnit of the
                plant), the unit for the 'electricity' commodity is given in GW_electric and
                the 'hydrogen' commodity is given in
                GW_hydrogen_lowerHeatingValue -> the commodityConversionFactors are defined
                as {'electricity':1,'hydrogen':-1/0.6}.
        :type commodityConversionFactors: dictionary, assigns commodities (string)
            conversion factors (float)

        **Default arguments:**

        :param hasCapacityVariable: specifies if the component should be modeled with a
            capacity or not.
            Examples:
            (a) An electrolyzer has a capacity given in GW_electrict -> hasCapacityVariable
                is True.
            (b) In the energy system, biogas can, from a model perspective, be converted
                into methane (and then used in conventional power plants which emit CO2) by
                getting CO2 from the enviroment. Thus, using biogas in conventional power
                plants from a balance perspective CO2 free. This conversion is purely
                theoretical and does not require a capacity -> hasCapacityVariable is False.
            |br| * the default value is True
        :type hasCapacityVariable: boolean

        :param capacityVariableDomain: the mathematical domain of the capacity variables, if
            they are specfied. By default, the domain is specified as 'continuous' and thus
            declares the variables as non-negative (>=0) real values. The second input
            option that is available for this parameter is 'discrete', which declares the
            variables as non-negative (>=0) integer values.
            |br| * the default value is 'continuous'
        :type capacityVariableDomain: string

        :param capacityPerPlantUnit: capacity of one plant of the component (in the unit in
            which the cost parameters are specified). The default is 1, thus the number of
            plants is equal to the installed capacity. This parameter should be specified
            when using a 'discrete' capacityVariableDomain. It can be specified when using
            a 'continuous' variable domain.
            |br| * the default value is 1
        :type capacityPerPlantUnit: strictly positive float

        :param hasIsBuiltBinaryVariable: specifies if binary decision variables should be
            declared for each eligible location of the component which indicate if the
            component is built at that location or not. The binary variables can be used to
            enforce one-time invest cost or one capacity-independent annual operation cost.
            If a minimum capacity is specified and this parameter is set to True, the
            minimum capacities are only considered if a component is built (i.e. if a
            component is built in that location it has to be built with a minimum capacity
            of XY GW, otherwise it is set to 0 GW).
            |br| * the default value is False
        :type hasIsBuiltBinaryVariable: boolean

        :param bigM: the bigM parameter is only required when the ifBuiltBinaryVariables
            parameter is set to True. In that case, it is set as a strictly positive float,
            otherwise it can remain a None value. If not None and the ifBuiltBinaryVariables
            parameter is set to True, the parameter enforces an artificial upper bound on the
            maximum capacities which should, however, never be reached. The value should be
            chosen as small as possible but as large as necessary so that the optimal values
            of the designed capacities are in the end well below this value.
            |br| * the default value is None
        :type bigM: None or strictly positive float

        :param operationRateMax: if specified indicates a maximum operation rate for each
            location and each time step by a non-negative float. If hasCapacityVariable is
            set to True, the values are given relative to the installed capacities (i.e. in
            that case a value of 1 indicates a utilization of 100% of the capacity). If
            hasCapacityVariable is set to False, the values are given as absolute values in
            form of an energy or mass unit, referring to the converted energy or mass (after
            multiplying the conversion factors for each commodity) during one time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with non-negative entries. The row
            indices have to match the specified time steps. The column indices have to match
            the specified locations.

        :param operationRateFix: if specified indicates a fixed operation rate for each
            location and each time step by a non-negative float. If hasCapacityVariable
            is set to True, the values are given relative to the installed capacities (i.e.
            in that case a value of 1 indicates a utilization of 100% of the capacity). If
            hasCapacityVariable is set to False, the values are given as absolute values in
            form of an energy or mass unit, referring to the converted energy or mass (after
            multiplying the conversion factors for each commodity) during one time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with non-negative entries. The
            row indices have to match the in the energy system model specified time steps.
            The column indices have to match the in the energy system model specified
            locations.

        :param tsaWeight: weight with which the time series of the component should be
            considered when applying time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: non-negative (>= 0) float

        :param locationalEligibility: Pandas Series that indicates if a component can be
            built at a location (=1) or not (=0). If not specified and a maximum or fixed
            capacity or time series is given, the parameter will be set based on these
            inputs. If the parameter is specified, a consistency check is done to ensure
            that the parameters indicate the same location eligibility. If the parameter is
            not specified and also no other of the parameters is specified it is assumed
            that the component is eligible in each location and all values are set to 1.
            This parameter is key for ensuring small built times of the optimization problem
            by avoiding the declaration of unnecessary variables and constraints.
            |br| * the default value is None
        :type locationalEligibility: None or Pandas Series with values equal to 0 and 1. The
            indices of the series have to equal the in the energy system model specified
            locations.

        :param capacityMin: if specified, Pandas Series indicating minimum capacities (in
            the plants physicalUnit) else None.
            If binary decision variables are declared which indicate if a component is built
            at a location or not, the minimum capacity is only enforced if the component is
            built (i.e. if a component is built in that location it has to be built with a
            minimum capacity of XY GW, otherwise it is set to 0 GW).
            |br| * the default value is None
        :type capacityMin: None or Pandas Series with non-negative (>=0) values. The indices
            of the series have to equal the in the energy system model specified locations.

        :param capacityMax:  if specified, Pandas Series indicating maximum capacities (in
            the plants physicalUnit) else None.
            |br| * the default value is None
        :type capacityMax: None or Pandas Series with non-negative (>=0) values. The
            indices of the series have to equal the in the energy system model specified
            locations.

        :param sharedPotentialID: if specified, indicates that the component has to share
            its maximum potential capacity with other components (i.e. due to space
            limitations).
            |br| * the default value is None
        :type sharedPotentialID: string

        :param capacityFix: if specified, Pandas Series indicating fixed capacities
            (in the plants physicalUnit) else None.
            |br| * the default value is None
        :type capacityFix: None or Pandas Series with non-negative (>=0) values. The indices
            of the series have to equal the in the energy system model specified locations.

        :param isBuiltFix: if specified, Pandas Series indicating fixed decisions in which
            locations the component is built else None.
            |br| * the default value is None
        :type isBuiltFix: None or Pandas Series with values equal to 0 and 1. The indices
            of the series have to equal the in the energy system model specified locations.

        :param investPerCapacity: the invest of a component is obtained by multiplying the
            capacity of the component (in the physicalUnit of the component) at that
            location with the investPerCapacity factor. The investPerCapacity can either be
            given as a float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in
            the energy system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type investPerCapacity: non-negative (>=0) float or Pandas Series with non-negative
            (>=0) values. The indices of the series have to equal the in the energy system
            model specified locations.

        :param investIfBuilt: a capacity-independent invest which only arises in a location
            if a component is built at that location. The investIfBuilt can either be given
            as a float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in
            the energy system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type investIfBuilt: non-negative (>=0) float or Pandas Series with non-negative
            (>=0) values. The indices of the series have to equal the in the energy system
            model specified locations.

        :param opexPerOperation: cost which is directly proportional to the operation of the
            component is obtained by multiplying the opexPerOperation parameter with the
            annual sum of the operational time series of the components. The
            opexPerOperation can either be given as a float or a Pandas Series with location
            specific values. The cost unit in which the parameter is given has to match the
            one specified in the energy system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerOperation: non-negative (>=0) float or Pandas Series with non-negative
            (>=0) values. The indices of the series have to equal the in the energy system
            model specified locations.

        :param opexPerCapacity: annual operational cost which are only a function of the
            capacity of the component (in the physicalUnit of the component) and not of the
            specific operation itself are obtained by multiplying the capacity of the
            component at a location with the opexPerCapacity factor. The opexPerCapacity can
            either be given as a float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in
            the energy system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerCapacity: non-negative (>=0) float or Pandas Series with non-negative
            (>=0) values. The indices of the series have to equal the in the energy system
            model specified locations.

        :param opexIfBuilt: a capacity-independent annual operational cost which only arises
            in a location if a component is built at that location. The opexIfBuilt can
            either be given as a float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in
            the energy system model (i.e. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexIfBuilt: non-negative (>=0) float or Pandas Series with non-negative (>=0)
            values. The indices of the series have to equal the in the energy system model
            specified locations.

        :param interestRate: interest rate which is considered for computing the annuities
            of the invest of the component (in particular to depreciate the invests over
            the economic lifetime). A value of 0.08 corresponds to an interest rate of 8%.
            |br| * the default value is 0.08
        :type interestRate: non-negative (>=0) float or Pandas Series with non-negative (>=0)
            values. The indices of the series have to equal the in the energy system model
            specified locations.

        :param economicLifetime: economic lifetime of the component which is considered for
            computing the annuities of the invest of the component (aka depreciation time).
            |br| * the default value is 10
        :type economicLifetime: strictly-positive (>0) float or Pandas Series with
            strictly-positive (>0) values. The indices of the series have to equal the in
            the energy system model specified locations.

        Last edited: July 27, 2018
        |br| @author: Lara Welder
        """
        # Set general component data
        utils.isEnergySystemModelInstance(esM), utils.checkCommodities(esM, set(commodityConversionFactors.keys()))
        self._name, self._commodityConversionFactors = name, commodityConversionFactors
        self._physicalUnit = physicalUnit

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
        self._operationRateMax = None

        self._fullOperationRateFix = utils.setFormattedTimeSeries(operationRateFix)
        self._aggregatedOperationRateFix = None
        self._operationRateFix = None

        utils.isPositiveNumber(tsaWeight)
        self._tsaWeight = tsaWeight

        # Set location-specific design parameters
        utils.checkLocationSpecficDesignInputParams(esM, hasCapacityVariable, hasIsBuiltBinaryVariable,
                                                    capacityMin, capacityMax, capacityFix,
                                                    locationalEligibility, isBuiltFix, sharedPotentialID,
                                                    dimension='1dim')
        self._sharedPotentialID = sharedPotentialID
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
            if esM._componentNames[self._name] == ConversionModeling.__name__:
                warnings.warn('Component identifier ' + self._name + ' already exists. Data will be overwritten.')
            else:
                raise ValueError('Component name ' + self._name + ' is not unique.')
        else:
            esM._componentNames.update({self._name: ConversionModeling.__name__})
        mdl = ConversionModeling.__name__
        if mdl not in esM._componentModelingDict:
            esM._componentModelingDict.update({mdl: ConversionModeling()})
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


class ConversionModeling(ComponentModeling):
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
        pyM.designDimensionVarSet_conv = pyomo.Set(dimen=2, initialize=initDesignVarSet)

        def initContinuousDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_conv
                    if compDict[compName]._capacityVariableDomain == 'continuous')
        pyM.continuousDesignDimensionVarSet_conv = pyomo.Set(dimen=2, initialize=initContinuousDesignVarSet)

        def initDiscreteDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_conv
                    if compDict[compName]._capacityVariableDomain == 'discrete')
        pyM.discreteDesignDimensionVarSet_conv = pyomo.Set(dimen=2, initialize=initDiscreteDesignVarSet)

        def initDesignDecisionVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_conv
                    if compDict[compName]._hasIsBuiltBinaryVariable)
        pyM.designDecisionVarSet_conv = pyomo.Set(dimen=2, initialize=initDesignDecisionVarSet)

        ################################################################################################################
        #                                     Declare operation variables sets                                         #
        ################################################################################################################

        def initOpVarSet(pyM):
            return ((loc, compName) for loc in esM._locations for compName, comp in compDict.items()
                    if comp._locationalEligibility[loc] == 1)
        pyM.operationVarSet_conv = pyomo.Set(dimen=2, initialize=initOpVarSet)
        pyM.operationVarDict_conv = {loc: {compName for compName in compDict if (loc, compName)
                                           in pyM.operationVarSet_conv} for loc in esM._locations}

        ################################################################################################################
        #                           Declare sets for case differentiation of operating modes                           #
        ################################################################################################################

        def initOpConstrSet1(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is None
                    and compDict[compName]._operationRateFix is None)
        pyM.opConstrSet1_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet1)

        def initOpConstrSet2(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet2_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet2)

        def initOpConstrSet3(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet3_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet3)

        def initOpConstrSet4(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if not
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet4_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet4)

        def initOpConstrSet5(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if not
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet5_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet5)

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
        pyM.cap_conv = pyomo.Var(pyM.designDimensionVarSet_conv, domain=pyomo.NonNegativeReals, bounds=capBounds)
        # Number of components [-]
        pyM.nbReal_conv = pyomo.Var(pyM.continuousDesignDimensionVarSet_conv, domain=pyomo.NonNegativeReals)
        # Number of components [-]
        pyM.nbInt_conv = pyomo.Var(pyM.discreteDesignDimensionVarSet_conv, domain=pyomo.NonNegativeIntegers)
        # Binary variables [-], indicate if a component is considered at a location or not
        pyM.designBin_conv = pyomo.Var(pyM.designDecisionVarSet_conv, domain=pyomo.Binary)
        # Operation of component [energyUnit]
        pyM.op_conv = pyomo.Var(pyM.operationVarSet_conv, pyM.timeSet, domain=pyomo.NonNegativeReals)

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
        def capToNbReal_conv(pyM, loc, compName):
            return pyM.cap_conv[loc, compName] == pyM.nbReal_conv[loc, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbReal_conv = pyomo.Constraint(pyM.continuousDesignDimensionVarSet_conv, rule=capToNbReal_conv)

        # Determine the components' capacities from the number of installed units
        def capToNbInt_conv(pyM, loc, compName):
            return pyM.cap_conv[loc, compName] == pyM.nbInt_conv[loc, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbInt_conv = pyomo.Constraint(pyM.discreteDesignDimensionVarSet_conv, rule=capToNbInt_conv)

        # Enforce the consideration of the binary design variables of a component
        def bigM_conv(pyM, loc, compName):
            return pyM.cap_conv[loc, compName] <= pyM.designBin_conv[loc, compName] * compDict[compName]._bigM
        pyM.ConstrBigM_conv = pyomo.Constraint(pyM.designDecisionVarSet_conv, rule=bigM_conv)

        # Enforce the consideration of minimum capacities for components with design decision variables
        def capacityMinDec_conv(pyM, loc, compName):
            return (pyM.cap_conv[loc, compName] >= compDict[compName]._capacityMin[loc] *
                    pyM.designBin_conv[loc, compName] if compDict[compName]._capacityMin is not None
                    else pyomo.Constraint.Skip)
        pyM.ConstrCapacityMinDec_conv = pyomo.Constraint(pyM.designDecisionVarSet_conv, rule=capacityMinDec_conv)

        # Sets, if applicable, the installed capacities of a component
        def capacityFix_conv(pyM, loc, compName):
            return (pyM.cap_conv[loc, compName] == compDict[compName]._capacityFix[loc]
                    if compDict[compName]._capacityFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrCapacityFix_conv = pyomo.Constraint(pyM.designDimensionVarSet_conv, rule=capacityFix_conv)

        # Sets, if applicable, the binary design variables of a component
        def designBinFix_conv(pyM, loc, compName):
            return (pyM.designBin_conv[loc, compName] == compDict[compName]._isBuiltFix[loc]
                    if compDict[compName]._isBuiltFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrDesignBinFix_conv = pyomo.Constraint(pyM.designDecisionVarSet_conv, rule=designBinFix_conv)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        def op1_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] <= pyM.cap_conv[loc, compName] * esM._hoursPerTimeStep
        pyM.ConstrOperation1_conv = pyomo.Constraint(pyM.opConstrSet1_conv, pyM.timeSet, rule=op1_conv)

        # Operation [energyUnit] equal to the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op2_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] == pyM.cap_conv[loc, compName] * \
                   compDict[compName]._operationRateFix[loc][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation2_conv = pyomo.Constraint(pyM.opConstrSet2_conv, pyM.timeSet, rule=op2_conv)

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op3_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] <= pyM.cap_conv[loc, compName] * \
                   compDict[compName]._operationRateMax[loc][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation3_conv = pyomo.Constraint(pyM.opConstrSet3_conv, pyM.timeSet, rule=op3_conv)

        # Operation [energyUnit] equal to the operation time series [energyUnit]
        def op4_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] == compDict[compName]._operationRateFix[loc][p, t]
        pyM.ConstrOperation4_conv = pyomo.Constraint(pyM.opConstrSet4_conv, pyM.timeSet, rule=op4_conv)

        # Operation [energyUnit] limited by the operation time series [energyUnit]
        def op5_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] <= compDict[compName]._operationRateMax[loc][p, t]
        pyM.ConstrOperation5_conv = pyomo.Constraint(pyM.opConstrSet5_conv, pyM.timeSet, rule=op5_conv)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        return sum(pyM.cap_conv[loc, compName] / self._componentsDict[compName]._capacityMax[loc]
                   for compName in self._componentsDict if self._componentsDict[compName]._sharedPotentialID == key and
                   (loc, compName) in pyM.designDimensionVarSet_conv)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([(commod in comp._commodityConversionFactors and comp._commodityConversionFactors[commod] != 0)
                    and comp._locationalEligibility[loc] == 1 for comp in self._componentsDict.values()])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        return sum(pyM.op_conv[loc, compName, p, t] * self._componentsDict[compName]._commodityConversionFactors[commod]
                   for compName in pyM.operationVarDict_conv[loc]
                   if commod in self._componentsDict[compName]._commodityConversionFactors)

    def getObjectiveFunctionContribution(self, esM, pyM):
        compDict = self._componentsDict

        capexCap = sum(compDict[compName]._investPerCapacity[loc] * pyM.cap_conv[loc, compName] /
                       compDict[compName]._CCF[loc] for loc, compName in pyM.cap_conv)

        capexDec = sum(compDict[compName]._investIfBuilt[loc] * pyM.designBin_conv[loc, compName] /
                       compDict[compName]._CCF[loc] for loc, compName in pyM.designBin_conv)

        opexCap = sum(compDict[compName]._opexPerCapacity[loc] * pyM.cap_conv[loc, compName]
                      for loc, compName in pyM.cap_conv)

        opexDec = sum(compDict[compName]._opexIfBuilt[loc] * pyM.designBin_conv[loc, compName]
                      for loc, compName in pyM.designBin_conv)

        opexOp = sum(compDict[compName]._opexPerOperation[loc] *
                     sum(pyM.op_conv[loc, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet)
                     for loc, compNames in pyM.operationVarDict_conv.items()
                     for compName in compNames) / esM._numberOfYears

        return capexCap + capexDec + opexCap + opexDec + opexOp

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        compDict = self._componentsDict
        props = ['capacity', 'isBuilt', 'operation', 'capexCap', 'capexIfBuilt', 'opexCap', 'opexIfBuilt',
                 'opexOp', 'TAC', 'invest']
        units = ['[-]', '[-]', '[-]', '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]',
                 '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]', '[' + esM._costUnit + '/a]',
                 '[' + esM._costUnit + '/a]', '[' + esM._costUnit + ']']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._physicalUnit + ']') if x[1] == 'capacity'
                          else x, tuples))
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]]._physicalUnit + '*h/a]') if x[1] == 'operation'
                          else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM._locations)).sort_index()

        # Get optimal variable values and contributions to the total annual cost and invest
        optVal = utils.formatOptimizationOutput(pyM.cap_conv.get_values(), 'designVariables', '1dim')
        self._capacityVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_capacityVariablesOptimum', compDict)

        if optVal is not None:
            i = optVal.apply(lambda cap: cap * compDict[cap.name]._investPerCapacity[cap.index], axis=1)
            cx = optVal.apply(lambda cap: cap * compDict[cap.name]._investPerCapacity[cap.index] /
                              compDict[cap.name]._CCF[cap.index], axis=1)
            ox = optVal.apply(lambda cap: cap*compDict[cap.name]._opexPerCapacity[cap.index], axis=1)
            optSummary.loc[[(ix, 'capacity', '[' + compDict[ix]._physicalUnit + ']') for ix in optVal.index],
                            optVal.columns] = optVal.values
            optSummary.loc[[(ix, 'invest', '[' + esM._costUnit + ']') for ix in i.index], i.columns] = i.values
            optSummary.loc[[(ix, 'capexCap', '[' + esM._costUnit + '/a]') for ix in cx.index], cx.columns] = cx.values
            optSummary.loc[[(ix, 'opexCap', '[' + esM._costUnit + '/a]') for ix in ox.index], ox.columns] = ox.values

        optVal = utils.formatOptimizationOutput(pyM.designBin_conv.get_values(), 'designVariables', '1dim')
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

        optVal = utils.formatOptimizationOutput(pyM.op_conv.get_values(), 'operationVariables', '1dim',
                                                esM._periodsOrder)
        self._operationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_operationVariablesOptimum', compDict)

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name]._opexPerOperation[op.index], axis=1)
            optSummary.loc[[(ix, 'operation', '[' + compDict[ix]._physicalUnit + '*h/a]') for ix in opSum.index],
                            opSum.columns] = opSum.values
            optSummary.loc[[(ix, 'opexOp', '[' + esM._costUnit + '/a]') for ix in ox.index], ox.columns] = ox.values

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'capexCap') |
                            (optSummary.index.get_level_values(1) == 'opexCap') |
                            (optSummary.index.get_level_values(1) == 'capexIfBuilt') |
                            (optSummary.index.get_level_values(1) == 'opexIfBuilt') |
                            (optSummary.index.get_level_values(1) == 'opexOp')].groupby(level=0).sum().values

        self._optSummary = optSummary

    def getOptimalCapacities(self):
        return self._capacitiesOpt
