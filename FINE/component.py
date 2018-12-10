from abc import ABCMeta, abstractmethod
from FINE import utils
import warnings
import pyomo.environ as pyomo
import pandas as pd


class Component(metaclass=ABCMeta):
    """
    The Component class includes the general methods and arguments for the components which are add-able to
    the energy system model (e.g. storage component, source component, transmission component). Every of these
    components inherits from the Component class. 

    """
    def __init__(self, esM, name, dimension,
                 hasCapacityVariable, capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                 hasIsBuiltBinaryVariable=False, bigM=None, locationalEligibility=None,
                 capacityMin=None, capacityMax=None, sharedPotentialID=None, capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerCapacity=0, opexIfBuilt=0,
                 interestRate=0.08, economicLifetime=10):
        """
        Constructor for creating an Component class instance.

        **Required arguments:**

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param name: name of the component. Has to be unique (i.e. no other components with that name can
            already exist in the EnergySystemModel instance to which the component is added).
        :type name: string

        :param hasCapacityVariable: specifies if the component should be modeled with a capacity or not. Examples:\n
            * An electrolyzer has a capacity given in GW_electric -> hasCapacityVariable is True.
            * In the energy system, biogas can, from a model perspective, be converted into methane (and then
              used in conventional power plants which emit CO2) by getting CO2 from the environment. Thus,
              using biogas in conventional power plants is, from a balance perspective, CO2 free. This
              conversion is purely theoretical and does not require a capacity -> hasCapacityVariable
              is False.
            * A electricity cable has a capacity given in GW_electric -> hasCapacityVariable is True.
            * If the transmission capacity of a component is unlimited -> hasCapacityVariable is False.
            * A wind turbine has a capacity given in GW_electric -> hasCapacityVariable is True.
            * Emitting CO2 into the environment is not per se limited by a capacity ->
              hasCapacityVariable is False.\n
        :type hasCapacityVariable: boolean

        **Default arguments:**

        :param capacityVariableDomain: describes the mathematical domain of the capacity variables, if they are
            specified. By default, the domain is specified as 'continuous' and thus declares the variables as positive
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

        :param hasIsBuiltBinaryVariable: specifies if binary decision variables should be declared for\n
            * each eligible location of the component, which indicates if the component is built at that location or
              not (dimension=1dim).
            * each eligible connection of the transmission component, which indicates if the component is built
              between two locations or not (dimension=2dim).\n
            The binary variables can be used to enforce one-time investment cost or capacity-independent
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

        :param locationalEligibility:\n
            * Pandas Series that indicates if a component can be built at a location (=1) or not (=0)
              (dimension=1dim) or
            * Pandas Series or DataFrame that indicates if a component can be built between two
              locations (=1) or not (=0) (dimension=2dim).\n
            If not specified and a maximum or fixed capacity or time series is given, the parameter will be
            set based on these inputs. If the parameter is specified, a consistency check is done to ensure
            that the parameters indicate the same locational eligibility. If the parameter is not specified,
            and also no other of the parameters is specified, it is assumed that the component is eligible in
            each location and all values are set to 1.
            This parameter is the key part for ensuring small built times of the optimization problem by avoiding the
            declaration of unnecessary variables and constraints.
            |br| * the default value is None
        :type locationalEligibility:\n
            * None or
            * Pandas Series with values equal to 0 and 1. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with values equal to 0 and 1. The column and row indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param capacityMin: if specified, indicates the minimum capacities. The type of this parameter depends on the
            dimension of the component: If dimension=1dim, it has to be a Pandas Series. If dimension=2dim, it has to
            to be a Pandas Series or DataFrame. If binary decision variables are declared, capacityMin is only used
            if the component is built.
            |br| * the default value is None
        :type capacityMin:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param capacityMax: if specified, indicates the maximum capacities. The type of this parameter depends on the
            dimension of the component: If dimension=1dim, it has to be a Pandas Series. If dimension=2dim, it has to
            to be a Pandas Series or DataFrame.
            |br| * the default value is None
        :type capacityMax:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param sharedPotentialID: if specified, indicates that the component has to share its maximum
            potential capacity with other components (e.g. due to space limitations). The shares of how
            much of the maximum potential is used have to add up to less then 100%.
            |br| * the default value is None
        :type sharedPotentialID: string

        :param capacityFix: if specified, indicates the fixed capacities. The type of this parameter
            depends on the dimension of the component: If dimension=1dim, it has to be a Pandas Series.
            If dimension=2dim, it has to be a Pandas Series or DataFrame.
            |br| * the default value is None
        :type capacityFix:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param isBuiltFix: if specified, indicates fixed decisions in which or between which locations the component is
            built (i.e. sets the isBuilt binary variables). The type of this parameter
            depends on the dimension of the component: If dimension=1dim, it has to be a Pandas Series.
            If dimension=2dim, it has to be a Pandas Series or DataFrame.
            |br| * the default value is None
        :type isBuiltFix:
            * None or
            * Pandas Series with values equal to 0 and 1. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with values equal to 0 and 1. The row and column indices of the DataFrame
              have to equal the in the energy system model specified locations.

        :param investPerCapacity: describes the investment costs for one unit of the capacity. The
            invest of a component is obtained by multiplying the built capacities
            of the component (in the physicalUnit of the component) with the investPerCapacity factor.
            The investPerCapacity can either be given as\n
            * a float or a Pandas Series with location specific values (dimension=1dim). The cost unit in which the
              parameter is given has to match the one specified in the energy system model (e.g. Euro, Dollar,
              1e6 Euro) or
            * a float or a Pandas Series or DataFrame with location specific values (dimension=2dim). The cost unit
              in which the parameter is given has to match the one specified in the energy system model divided by
              the specified lengthUnit (e.g. Euro/m, Dollar/m, 1e6 Euro/km).\n
            |br| * the default value is 0
        :type investPerCapacity:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param investIfBuilt: a capacity-independent invest which only arises in a location if a component
            is built at that location. The investIfBuilt can either be given as\n
            * a float or a Pandas Series with location specific values (dimension=1dim). The cost unit in which
              the parameter is given has to match the one specified in the energy system model (e.g. Euro, Dollar,
              1e6 Euro) or
            * a float or a Pandas Series or DataFrame with location specific values (dimension=2dim). The cost unit
              in which the parameter is given has to match the one specified in the energy system model divided by
              the specified lengthUnit (e.g. Euro/m, Dollar/m, 1e6 Euro/km)\n
            |br| * the default value is 0
        :type investIfBuilt:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param opexPerCapacity: describes the operational cost for one unit of capacity. The annual operational cost,
            which are only a function of the capacity of the component (in the physicalUnit of the component) and not
            of the specific operation itself, are obtained by multiplying the capacity of the component at a location
            with the opexPerCapacity factor. The opexPerCapacity factor can either be given as\n
            * a float or a Pandas Series with location specific values (dimension=1dim). The cost unit in which the
              parameter is given has to match the one specified in the energy system model (e.g. Euro, Dollar,
              1e6 Euro) or
            * a float or a Pandas Series or DataFrame with location specific values (dimension=2dim). The cost unit
              in which the parameter is given has to match the one specified in the energy system model divided by
              the specified lengthUnit (e.g. Euro/m, Dollar/m, 1e6 Euro/km)\n
            |br| * the default value is 0
        :type opexPerCapacity:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param opexIfBuilt: a capacity-independent annual operational cost which only arises in a location
            if a component is built at that location. The opexIfBuilt can either be given as\n
            * a float or a Pandas Series with location specific values (dimension=1dim) . The cost unit in which
              the parameter is given has to match the one specified in the energy system model (e.g. Euro, Dollar,
              1e6 Euro)or
            * a float or a Pandas Series or DataFrame with location specific values (dimension=2dim). The cost unit
              in which the parameter is given has to match the one specified in the energy system model divided by
              the specified lengthUnit (e.g. Euro/m, Dollar/m, 1e6 Euro/km)\n
            |br| * the default value is 0
        :type opexIfBuilt:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param interestRate: interest rate which is considered for computing the annuities of the invest
            of the component (depreciates the invests over the economic lifetime).
            A value of 0.08 corresponds to an interest rate of 8%.
            |br| * the default value is 0.08
        :type interestRate:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param economicLifetime: economic lifetime of the component which is considered for computing the
            annuities of the invest of the component (aka depreciation time).
            |br| * the default value is 10
        :type economicLifetime:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param modelingClass: to the Component connected modeling class.
            |br| * the default value is ModelingClass
        :type modelingClass: a class inherting from ComponentModeling
        """
        # Set general component data
        utils.isEnergySystemModelInstance(esM)
        self.name = name
        self.dimension = dimension
        self.modelingClass = ComponentModel

        # Set design variable modeling parameters
        utils.checkDesignVariableModelingParameters(capacityVariableDomain, hasCapacityVariable, capacityPerPlantUnit,
                                                    hasIsBuiltBinaryVariable, bigM)
        self.hasCapacityVariable = hasCapacityVariable
        self.capacityVariableDomain = capacityVariableDomain
        self.capacityPerPlantUnit = capacityPerPlantUnit
        self.hasIsBuiltBinaryVariable = hasIsBuiltBinaryVariable
        self.bigM = bigM

        # Set economic data
        elig = locationalEligibility
        self.investPerCapacity = utils.checkAndSetCostParameter(esM, name, investPerCapacity, dimension, elig)
        self.investIfBuilt = utils.checkAndSetCostParameter(esM, name, investIfBuilt, dimension, elig)
        self.opexPerCapacity = utils.checkAndSetCostParameter(esM, name, opexPerCapacity, dimension, elig)
        self.opexIfBuilt = utils.checkAndSetCostParameter(esM, name, opexIfBuilt, dimension, elig)
        self.interestRate = utils.checkAndSetCostParameter(esM, name, interestRate, dimension, elig)
        self.economicLifetime = utils.checkAndSetCostParameter(esM, name, economicLifetime, dimension, elig)
        self.CCF = utils.getCapitalChargeFactor(self.interestRate, self.economicLifetime)

        # Set location-specific design parameters
        self.locationalEligibility = locationalEligibility
        self.sharedPotentialID = sharedPotentialID
        self.capacityMin, self.capacityMax, self.capacityFix = capacityMin, capacityMax, capacityFix
        self.isBuiltFix = isBuiltFix
        utils.checkLocationSpecficDesignInputParams(self, esM)
        #
        # # Variables at optimum (set after optimization)
        # self.capacityVariablesOptimum = None
        # self.isBuiltVariablesOptimum = None
        # self.operationVariablesOptimum = {}

    def addToEnergySystemModel(self, esM):
        """
        Add the component to an EnergySystemModel instance (esM). If the respective component class is not already in
        the esM, it is added as well.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance
        """
        esM.isTimeSeriesDataClustered = False
        if self.name in esM.componentNames:
            if esM.componentNames[self.name] == self.modelingClass.__name__ and esM.verbose < 2:
                warnings.warn('Component identifier ' + self.name + ' already exists. Data will be overwritten.')
            elif esM.componentNames[self.name] != self.modelingClass.__name__ :
                raise ValueError('Component name ' + self.name + ' is not unique.')
        else:
            esM.componentNames.update({self.name: self.modelingClass.__name__})
        mdl = self.modelingClass.__name__
        if mdl not in esM.componentModelingDict:
            esM.componentModelingDict.update({mdl: self.modelingClass()})
        esM.componentModelingDict[mdl].componentsDict.update({self.name: self})

    def prepareTSAInput(self, rateFix, rateMax, rateName, rateWeight, weightDict, data):
        """
        Format the time series data of a component to fit the requirements of the time series aggregation package and
        return a list of formatted data.

        :param rateFix: a fixed operation time series or None
        :type rateFix: Pandas DataFrame or None

        :param rateMax: a maximum operation time series or None
        :type rateMax: Pandas DataFrame of None

        :param rateName: name of the time series (to ensure uniqueness if a component has multiple relevant time series)
        :type rateName: string

        :param rateWeight: weight of the time series in the clustering process
        :type rateWeight: positive float (>=0)

        :param weightDict: dictionary to which the weight is added
        :type weightDict: dict

        :param data: list to which the formatted data is added
        :type data: list of Pandas DataFrames

        :return: data
        :rtype: Pandas DataFrame
        """
        data_ = rateFix if rateFix is not None else rateMax
        if data_ is not None:
            data_ = data_.copy()
            uniqueIdentifiers = [self.name + rateName + loc for loc in data_.columns]
            data_.rename(columns={loc: self.name + rateName + loc for loc in data_.columns}, inplace=True)
            weightDict.update({id: rateWeight for id in uniqueIdentifiers}), data.append(data_)
        return weightDict, data

    def getTSAOutput(self, rate, rateName, data):
        """
        Return a reformatted time series data after applying time series aggregation, if the original time series
        data is not None.

        :param rate: Full (unclustered) time series data or None
        :type rate: Pandas DataFrame or None

        :param rateName: name of the time series (to ensure uniqueness if a component has multiple relevant time series)
        :type rateName: string

        :param data: Pandas DataFrame with the clustered time series data of all components in the energy system
        :type data: Pandas DataFrame

        :return: reformatted data or None
        :rtype: Pandas DataFrame
        """
        if rate is not None:
            uniqueIdentifiers = [self.name + rateName + loc for loc in rate.columns]
            data_ = data[uniqueIdentifiers].copy()
            data_.rename(columns={self.name + rateName + loc: loc for loc in rate.columns}, inplace=True)
            return data_
        else:
            return None

    @abstractmethod
    def setTimeSeriesData(self, hasTSA):
        """
        Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises). Sets
        the time series data of a component (either the full time series if hasTSA is false or the aggregated
        time series if hasTSA is True).

        :param hasTSA: indicates if time series aggregation should be considered for modeling
        :type hasTSA: boolean
        """
        raise NotImplementedError

    @abstractmethod
    def getDataForTimeSeriesAggregation(self):
        """
        Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises). Get
        all time series data of a component for time series aggregation.
        """
        raise NotImplementedError

    @abstractmethod
    def setAggregatedTimeSeriesData(self, data):
        """
        Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises). Set
        aggregated time series data after applying time series aggregation.

        :param data: time series data
        :type data: Pandas DataFrame
        """
        raise NotImplementedError


class ComponentModel(metaclass=ABCMeta):
    """
    The ComponentModel class provides the general methods used for modeling the components.
    Every model class of the several component technologies inherits from the ComponentModel class.
    Within the ComponentModel class, general valid sets, variables and constraints are declared.
    """
    def __init__(self):
        """ Constructor for creating a ComponentModel class instance. """
        self.abbrvName = ''
        self.dimension = ''
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = {}
        self.optSummary = None

    ####################################################################################################################
    #                           Functions for declaring design and operation variables sets                            #
    ####################################################################################################################

    def declareDesignVarSet(self, pyM):
        """
        Declare set for capacity variables in the pyomo object for a modeling class.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName

        def declareDesignVarSet(pyM):
            return ((loc, compName) for compName, comp in compDict.items()
                    for loc in comp.locationalEligibility.index
                    if comp.locationalEligibility[loc] == 1 and comp.hasCapacityVariable)
        setattr(pyM, 'designDimensionVarSet_' + abbrvName, pyomo.Set(dimen=2, initialize=declareDesignVarSet))

    def declareContinuousDesignVarSet(self, pyM):
        """
        Declare set for continuous number of installed components in the pyomo object for a modeling class.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def declareContinuousDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in getattr(pyM, 'designDimensionVarSet_' + abbrvName)
                    if compDict[compName].capacityVariableDomain == 'continuous')
        setattr(pyM, 'continuousDesignDimensionVarSet_' + abbrvName,
                pyomo.Set(dimen=2, initialize=declareContinuousDesignVarSet))

    def declareDiscreteDesignVarSet(self, pyM):
        """ 
        Declare set for discrete number of installed components in the pyomo object for a modeling class. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel        
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def declareDiscreteDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in getattr(pyM, 'designDimensionVarSet_' + abbrvName)
                    if compDict[compName].capacityVariableDomain == 'discrete')
        setattr(pyM, 'discreteDesignDimensionVarSet_' + abbrvName,
                pyomo.Set(dimen=2, initialize=declareDiscreteDesignVarSet))

    def declareDesignDecisionVarSet(self, pyM):
        """ 
        Declare set for design decision variables in the pyomo object for a modeling class.
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel   
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        def declareDesignDecisionVarSet(pyM):
            return ((loc, compName) for loc, compName in getattr(pyM, 'designDimensionVarSet_' + abbrvName)
                    if compDict[compName].hasIsBuiltBinaryVariable)
        setattr(pyM, 'designDecisionVarSet_' + abbrvName, pyomo.Set(dimen=2, initialize=declareDesignDecisionVarSet))

    def declareOpVarSet(self, esM, pyM):
        """
        Declare operation related sets (operation variables and mapping sets) in the pyomo object for a
        modeling class.
        
        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance
                
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel  
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        # Set for operation variables
        def declareOpVarSet(pyM):
            return ((loc, compName) for compName, comp in compDict.items()
                    for loc in comp.locationalEligibility.index if comp.locationalEligibility[loc] == 1)
        setattr(pyM, 'operationVarSet_' + abbrvName, pyomo.Set(dimen=2, initialize=declareOpVarSet))

        if self.dimension == '1dim':
            # Dictionary which lists all components of the modeling class at one location
            setattr(pyM, 'operationVarDict_' + abbrvName,
                    {loc: {compName for compName in compDict
                           if (loc, compName) in getattr(pyM, 'operationVarSet_' + abbrvName)}
                     for loc in esM.locations})
        elif self.dimension == '2dim':
            # Dictionaries which list all outgoing and incoming components at a location
            setattr(pyM, 'operationVarDictOut_' + abbrvName,
                    {loc: {loc_: {compName for compName in compDict
                                  if (loc + '_' + loc_, compName) in getattr(pyM, 'operationVarSet_' + abbrvName)}
                           for loc_ in esM.locations} for loc in esM.locations})
            setattr(pyM, 'operationVarDictIn_' + abbrvName,
                    {loc: {loc_: {compName for compName in compDict
                                  if (loc_ + '_' + loc, compName) in getattr(pyM, 'operationVarSet_' + abbrvName)}
                           for loc_ in esM.locations} for loc in esM.locations})

    ####################################################################################################################
    #                                   Functions for declaring operation mode sets                                    #
    ####################################################################################################################

    def declareOpConstrSet1(self, pyM, constrSetName, rateMax, rateFix):
        """
        Declare set of locations and components for which hasCapacityVariable is set to True and neither the
        maximum nor the fixed operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def declareOpConstrSet1(pyM):
            return ((loc, compName) for loc, compName in varSet if compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateMax) is None
                    and getattr(compDict[compName], rateFix) is None)

        setattr(pyM, constrSetName + '1_' + abbrvName, pyomo.Set(dimen=2, initialize=declareOpConstrSet1))

    def declareOpConstrSet2(self, pyM, constrSetName, rateFix):
        """
        Declare set of locations and components for which hasCapacityVariable is set to True and a fixed
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def declareOpConstrSet2(pyM):
            return ((loc, compName) for loc, compName in varSet if compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateFix) is not None)

        setattr(pyM, constrSetName + '2_' + abbrvName, pyomo.Set(dimen=2, initialize=declareOpConstrSet2))

    def declareOpConstrSet3(self, pyM, constrSetName, rateMax):
        """
        Declare set of locations and components for which  hasCapacityVariable is set to True and a maximum
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def declareOpConstrSet3(pyM):
            return ((loc, compName) for loc, compName in varSet if compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateMax) is not None)

        setattr(pyM, constrSetName + '3_' + abbrvName, pyomo.Set(dimen=2, initialize=declareOpConstrSet3))

    def declareOpConstrSet4(self, pyM, constrSetName, rateFix):
        """
        Declare set of locations and components for which hasCapacityVariable is set to False and a fixed
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def declareOpConstrSet4(pyM):
            return ((loc, compName) for loc, compName in varSet if not compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateFix) is not None)

        setattr(pyM, constrSetName + '4_' + abbrvName, pyomo.Set(dimen=2, initialize=declareOpConstrSet4))

    def declareOpConstrSet5(self, pyM, constrSetName, rateMax):
        """
        Declare set of locations and components for which hasCapacityVariable is set to False and a maximum
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def declareOpConstrSet5(pyM):
            return ((loc, compName) for loc, compName in varSet if not compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateMax) is not None)

        setattr(pyM, constrSetName + '5_' + abbrvName, pyomo.Set(dimen=2, initialize=declareOpConstrSet5))

    def declareOperationModeSets(self, pyM, constrSetName, rateMax, rateFix):
        """
        Declare operating mode sets.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param constrSetName: name of the constraint set.
        :type constrSetName: string

        :param rateMax: attribute of the considered component which stores the maximum operation rate data.
        :type rateMax: string

        :param rateFix: attribute of the considered component which stores the fixed operation rate data.
        :type rateFix: string
        """
        self.declareOpConstrSet1(pyM, constrSetName, rateMax, rateFix)
        self.declareOpConstrSet2(pyM, constrSetName, rateFix)
        self.declareOpConstrSet3(pyM, constrSetName, rateMax)
        self.declareOpConstrSet4(pyM, constrSetName, rateFix)
        self.declareOpConstrSet5(pyM, constrSetName, rateMax)

    ####################################################################################################################
    #                                         Functions for declaring variables                                        #
    ####################################################################################################################

    def declareCapacityVars(self, pyM):
        """ 
        Declare capacity variables.
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel 
        """
        abbrvName = self.abbrvName

        def capBounds(pyM, loc, compName):
            """ Function for setting lower and upper capacity bounds. """
            comp = self.componentsDict[compName]
            return (comp.capacityMin[loc] if (comp.capacityMin is not None and not comp.hasIsBuiltBinaryVariable)
                    else 0,
                    comp.capacityMax[loc] if comp.capacityMax is not None else None)
        setattr(pyM, 'cap_' + abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' + abbrvName),
                domain=pyomo.NonNegativeReals, bounds=capBounds))

    def declareRealNumbersVars(self, pyM):
        """ 
        Declare variables representing the (continuous) number of installed components [-]. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName = self.abbrvName
        setattr(pyM, 'nbReal_' + abbrvName, pyomo.Var(getattr(pyM, 'continuousDesignDimensionVarSet_' + abbrvName),
                domain=pyomo.NonNegativeReals))

    def declareIntNumbersVars(self, pyM):
        """ 
        Declare variables representing the (discrete/integer) number of installed components [-]. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName = self.abbrvName
        setattr(pyM, 'nbInt_' + abbrvName, pyomo.Var(getattr(pyM, 'discreteDesignDimensionVarSet_' + abbrvName),
                domain=pyomo.NonNegativeIntegers))

    def declareBinaryDesignDecisionVars(self, pyM):
        """ 
        Declare binary variables [-] indicating if a component is considered at a location or not [-]. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName = self.abbrvName
        setattr(pyM, 'designBin_' + abbrvName, pyomo.Var(getattr(pyM, 'designDecisionVarSet_' + abbrvName),
                domain=pyomo.Binary))

    def declareOperationVars(self, pyM, opVarName):
        """ 
        Declare operation variables.
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName = self.abbrvName
        setattr(pyM, opVarName + '_' + abbrvName,
                pyomo.Var(getattr(pyM, 'operationVarSet_' + abbrvName), pyM.timeSet, domain=pyomo.NonNegativeReals))

    ####################################################################################################################
    #                              Functions for declaring time independent constraints                                #
    ####################################################################################################################

    def capToNbReal(self, pyM):
        """ 
        Determine the components' capacities from the number of installed units.
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel 
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, nbRealVar = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'nbReal_' + abbrvName)
        nbRealVarSet = getattr(pyM, 'continuousDesignDimensionVarSet_' + abbrvName)

        def capToNbReal(pyM, loc, compName):
            return capVar[loc, compName] == nbRealVar[loc, compName] * compDict[compName].capacityPerPlantUnit
        setattr(pyM, 'ConstrCapToNbReal_' + abbrvName, pyomo.Constraint(nbRealVarSet, rule=capToNbReal))

    def capToNbInt(self, pyM):
        """ 
        Determine the components' capacities from the number of installed units. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, nbIntVar = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'nbInt_' + abbrvName)
        nbIntVarSet = getattr(pyM, 'discreteDesignDimensionVarSet_' + abbrvName)

        def capToNbInt(pyM, loc, compName):
            return capVar[loc, compName] == nbIntVar[loc, compName] * compDict[compName].capacityPerPlantUnit
        setattr(pyM, 'ConstrCapToNbInt_' + abbrvName, pyomo.Constraint(nbIntVarSet, rule=capToNbInt))

    def bigM(self, pyM):
        """ 
        Enforce the consideration of the binary design variables of a component. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, designBinVar = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'designBin_' + abbrvName)
        designBinVarSet = getattr(pyM, 'designDecisionVarSet_' + abbrvName)

        def bigM(pyM, loc, compName):
            return capVar[loc, compName] <= designBinVar[loc, compName] * compDict[compName].bigM
        setattr(pyM, 'ConstrBigM_' + abbrvName, pyomo.Constraint(designBinVarSet, rule=bigM))

    def capacityMinDec(self, pyM):
        """ 
        Enforce the consideration of minimum capacities for components with design decision variables. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName, dim = self.componentsDict, self.abbrvName, self.dimension
        capVar, designBinVar = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'designBin_' + abbrvName)
        designBinVarSet = getattr(pyM, 'designDecisionVarSet_' + abbrvName)

        def capacityMinDec(pyM, loc, compName):
            return (capVar[loc, compName] >= compDict[compName].capacityMin[loc] * designBinVar[loc, compName]
                    if compDict[compName].capacityMin is not None else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrCapacityMinDec_' + abbrvName, pyomo.Constraint(designBinVarSet, rule=capacityMinDec))

    def capacityFix(self, pyM):
        """ 
        Set, if applicable, the installed capacities of a component. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName, dim = self.componentsDict, self.abbrvName, self.dimension
        capVar = getattr(pyM, 'cap_' + abbrvName)
        capVarSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        def capacityFix(pyM, loc, compName):
            return (capVar[loc, compName] == compDict[compName].capacityFix[loc]
                    if compDict[compName].capacityFix is not None else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrCapacityFix_' + abbrvName, pyomo.Constraint(capVarSet, rule=capacityFix))

    def designBinFix(self, pyM):
        """ 
        Set, if applicable, the installed capacities of a component. 
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName, dim = self.componentsDict, self.abbrvName, self.dimension
        designBinVar = getattr(pyM, 'designBin_' + abbrvName)
        designBinVarSet = getattr(pyM, 'designDecisionVarSet_' + abbrvName)

        def designBinFix(pyM, loc, compName):
            return (designBinVar[loc, compName] == compDict[compName].isBuiltFix[loc]
                    if compDict[compName].isBuiltFix is not None else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrDesignBinFix_' + abbrvName, pyomo.Constraint(designBinVarSet, rule=designBinFix))

    ####################################################################################################################
    #                               Functions for declaring time dependent constraints                                 #
    ####################################################################################################################

    def operationMode1(self, pyM, esM, constrName, constrSetName, opVarName, factorName=None, isStateOfCharge=False):
        """
        Define operation mode 1. The operation [commodityUnit*h] is limited by the installed capacity in:\n
        * [commodityUnit*h] (for storages) or in
        * [commodityUnit] multiplied by the hours per time step (else).\n
        An additional factor can limited the operation further.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet1 = getattr(pyM, constrSetName + '1_' + abbrvName)
        factor1 = 1 if isStateOfCharge else esM.hoursPerTimeStep

        def op1(pyM, loc, compName, p, t):
            factor2 = 1 if factorName is None else getattr(compDict[compName], factorName)
            return opVar[loc, compName, p, t] <= factor1 * factor2 * capVar[loc, compName]
        setattr(pyM, constrName + '1_' + abbrvName, pyomo.Constraint(constrSet1, pyM.timeSet, rule=op1))

    def operationMode2(self, pyM, esM, constrName, constrSetName, opVarName, isStateOfCharge=False):
        """
        Define operation mode 2. The operation [commodityUnit*h] is equal to the installed capacity multiplied
        with a time series in:\n
        * [commodityUnit*h] (for storages) or in
        * [commodityUnit] multiplied by the hours per time step (else).\n
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet2 = getattr(pyM, constrSetName + '2_' + abbrvName)
        factor = 1 if isStateOfCharge else esM.hoursPerTimeStep

        def op2(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] == capVar[loc, compName] * \
                   compDict[compName].operationRateFix[loc][p, t] * factor
        setattr(pyM, constrName + '2_' + abbrvName, pyomo.Constraint(constrSet2, pyM.timeSet, rule=op2))

    def operationMode3(self, pyM, esM, constrName, constrSetName, opVarName, isStateOfCharge=False):
        """
        Define operation mode 3. The operation [commodityUnit*h] is limited by an installed capacity multiplied
        with a time series in:\n
        * [commodityUnit*h] (for storages) or in
        * [commodityUnit] multiplied by the hours per time step (else).\n
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet3 = getattr(pyM, constrSetName + '3_' + abbrvName)
        factor = 1 if isStateOfCharge else esM.hoursPerTimeStep

        def op3(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] <= capVar[loc, compName] * \
                   compDict[compName].operationRateMax[loc][p, t] * factor
        setattr(pyM, constrName + '3_' + abbrvName, pyomo.Constraint(constrSet3, pyM.timeSet, rule=op3))

    def operationMode4(self, pyM, esM, constrName, constrSetName, opVarName):
        """
        Define operation mode 4. The operation [commodityUnit*h] is equal to a time series in.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, opVarName + '_' + abbrvName)
        constrSet4 = getattr(pyM, constrSetName + '4_' + abbrvName)

        def op4(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] == compDict[compName].operationRateFix[loc][p, t]
        setattr(pyM, constrName + '4_' + abbrvName, pyomo.Constraint(constrSet4, pyM.timeSet, rule=op4))

    def operationMode5(self, pyM, esM, constrName, constrSetName, opVarName):
        """
        Define operation mode 4. The operation  [commodityUnit*h] is limited by a time series.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, opVarName + '_' + abbrvName)
        constrSet5 = getattr(pyM, constrSetName + '5_' + abbrvName)

        def op5(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] <= compDict[compName].operationRateMax[loc][p, t]
        setattr(pyM, constrName + '5_' + abbrvName, pyomo.Constraint(constrSet5, pyM.timeSet, rule=op5))

    ####################################################################################################################
    #  Functions for declaring component contributions to basic energy system constraints and the objective function   #
    ####################################################################################################################

    @abstractmethod
    def declareSets(self, esM, pyM):
        """
        Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Declare sets of components and constraints in the componentModel class.
        
        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        raise NotImplementedError

    @abstractmethod
    def declareVariables(self, esM, pyM):
        """
        Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Declare variables of components in the componentModel class.
        
        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        raise NotImplementedError

    @abstractmethod
    def declareComponentConstraints(self, esM, pyM):
        """
        Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Declare constraints of components in the componentModel class.
        
        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        raise NotImplementedError

    @abstractmethod
    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        """
        Check if operation variables exist in the modeling class at a location which are connected to a commodity.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param loc: name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """
        
        raise NotImplementedError

    @abstractmethod
    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """
        Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Get contribution to a commodity balance.
        """
        raise NotImplementedError

    @abstractmethod
    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Get contribution to the objective function.
        
        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        raise NotImplementedError

    def getSharedPotentialContribution(self, pyM, key, loc):
        """
        Get the share which the components of the modeling class have on a shared maximum potential at a location.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar = getattr(pyM, 'cap_' + abbrvName)
        capVarSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        return sum(capVar[loc, compName] / compDict[compName].capacityMax[loc] for compName in compDict
                   if compDict[compName].sharedPotentialID == key and (loc, compName) in capVarSet)

    def getLocEconomicsTD(self, pyM, esM, factorNames, varName, loc, compName, getOptValue=False):
        var = getattr(pyM, varName + '_' + self.abbrvName)
        factors = [getattr(self.componentsDict[compName], factorName)[loc] for factorName in factorNames]
        factor = 1.
        for factor_ in factors:
            factor *= factor_
        if not getOptValue:
            return (factor * sum(var[loc, compName, p, t] * esM.periodOccurrences[p]
                                 for p, t in pyM.timeSet)/esM.numberOfYears)
        else:
            return (factor * sum(var[loc, compName, p, t].value * esM.periodOccurrences[p]
                                 for p, t in pyM.timeSet)/esM.numberOfYears)

    def getLocEconomicsTI(self, pyM, factorNames, varName, loc, compName, divisorName='', getOptValue=False):
        var = getattr(pyM, varName + '_' + self.abbrvName)
        factors = [getattr(self.componentsDict[compName], factorName)[loc] for factorName in factorNames]
        divisor = getattr(self.componentsDict[compName], divisorName)[loc] if not divisorName == '' else 1
        factor = 1./divisor
        for factor_ in factors:
            factor *= factor_
        if not getOptValue:
            return factor * var[loc, compName]
        else:
            return factor * var[loc, compName].value

    def getEconomicsTI(self, pyM, factorNames, varName, divisorName='', getOptValue=False):
        var = getattr(pyM, varName + '_' + self.abbrvName)
        return sum(self.getLocEconomicsTI(pyM, factorNames, varName, loc, compName, divisorName, getOptValue)
                   for loc, compName in var)

    def getEconomicsTD(self, pyM, esM, factorNames, varName, dictName, getOptValue=False):
        indices = getattr(pyM, dictName + '_' + self.abbrvName).items()
        if self.dimension == '1dim':
            return sum(self.getLocEconomicsTD(pyM, esM, factorNames, varName, loc, compName, getOptValue)
                       for loc, compNames in indices for compName in compNames)
        else:
            return sum(self.getLocEconomicsTD(pyM, esM, factorNames, varName, loc + '_' + loc_, compName, getOptValue)
                       for loc, subDict in indices
                       for loc_, compNames in subDict.items()
                       for compName in compNames)

    def setOptimalValues(self, esM, pyM, indexColumns, plantUnit, unitApp=''):
        """
        Set the optimal values for the considered components and return a summary of them.

        **Required arguments**

        :param esM: EnergySystemModel instance representing the energy system in which the components are modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param indexColumns: set of strings with the columns indices of the summary. The indices represent the locations
            or connections between the locations are used to call the optimal values of the variables of the components
            in the model class.
        :type indexColumns: set

        :param plantUnit: attribute of the component that describes the unit of the plants to which maximum capacity
            limitations, cost parameters and the operation time series refer to. Depending on the considered component,
            possible inputs are "commodityUnit" (e.g. for transmission components) or "physicalUnit" (e.g. for
            conversion components).
        :type plantUnit: string

        **Default arguments**

        :param unitApp: string which appends the capacity unit in the optimization summary.
            For example, for the StorageModel class, the parameter is set to '*h'.
            |br| * the default value is ''.
        :type unitApp: string

        :return: summary of the optimized values.
        :rtype: pandas DataFrame
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar = getattr(esM.pyM, 'cap_' + abbrvName)
        binVar = getattr(esM.pyM, 'designBin_' + abbrvName)

        props = ['capacity', 'isBuilt', 'capexCap', 'capexIfBuilt', 'opexCap', 'opexIfBuilt', 'TAC',
                 'invest']
        units = ['[-]', '[-]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]',
                 '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + ']']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + getattr(compDict[x[0]], plantUnit) + unitApp + ']')
                          if x[1] == 'capacity' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(indexColumns)).sort_index()

        # Get and set optimal variable values and contributions to the total annual cost and invest
        values = capVar.get_values()
        optVal = utils.formatOptimizationOutput(values, 'designVariables', '1dim')
        optVal_ = utils.formatOptimizationOutput(values, 'designVariables', self.dimension, compDict=compDict)
        self.capacityVariablesOptimum = optVal_

        # Check if the installed capacities are close to a bigM value for components with design decision variables
        for compName, comp in compDict.items():
            if comp.hasIsBuiltBinaryVariable and optVal.loc[compName].max().max() >= comp.bigM * 0.9 \
               and esM.verbose < 2:
                warnings.warn('the capacity of component ' + compName + ' is in one or more locations close or equal '
                              'to the chosen Big M. Consider rerunning the simulation with a higher Big M.')

        if optVal is not None:
            i = optVal.apply(lambda cap: cap * compDict[cap.name].investPerCapacity[cap.index], axis=1)
            cx = optVal.apply(lambda cap: cap * compDict[cap.name].investPerCapacity[cap.index] /
                                          compDict[cap.name].CCF[cap.index], axis=1)
            ox = optVal.apply(lambda cap: cap * compDict[cap.name].opexPerCapacity[cap.index], axis=1)
            optSummary.loc[
                [(ix, 'capacity', '[' + getattr(compDict[ix], plantUnit) + unitApp + ']') for ix in optVal.index],
                optVal.columns] = optVal.values
            optSummary.loc[[(ix, 'invest', '[' + esM.costUnit + ']') for ix in i.index], i.columns] = \
                i.values
            optSummary.loc[[(ix, 'capexCap', '[' + esM.costUnit + '/a]') for ix in cx.index], cx.columns] = \
                cx.values
            optSummary.loc[[(ix, 'opexCap', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                ox.values

        values = binVar.get_values()
        optVal = utils.formatOptimizationOutput(values, 'designVariables', '1dim')
        optVal_ = utils.formatOptimizationOutput(values, 'designVariables', self.dimension, compDict=compDict)
        self.isBuiltVariablesOptimum = optVal_

        if optVal is not None:
            i = optVal.apply(lambda dec: dec * compDict[dec.name].investIfBuilt[dec.index], axis=1)
            cx = optVal.apply(lambda dec: dec * compDict[dec.name].investIfBuilt[dec.index] /
                              compDict[dec.name].CCF[dec.index], axis=1)
            ox = optVal.apply(lambda dec: dec * compDict[dec.name].opexIfBuilt[dec.index], axis=1)
            optSummary.loc[[(ix, 'isBuilt', '[-]') for ix in optVal.index], optVal.columns] = optVal.values
            optSummary.loc[[(ix, 'invest', '[' + esM.costUnit + ']') for ix in cx.index], cx.columns] += \
                i.values
            optSummary.loc[[(ix, 'capexIfBuilt', '[' + esM.costUnit + '/a]') for ix in cx.index],
                           cx.columns] = cx.values
            optSummary.loc[[(ix, 'opexIfBuilt', '[' + esM.costUnit + '/a]') for ix in ox.index],
                           ox.columns] = ox.values

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'capexCap') |
                           (optSummary.index.get_level_values(1) == 'opexCap') |
                           (optSummary.index.get_level_values(1) == 'capexIfBuilt')].groupby(level=0).sum().values

        return optSummary

    def getOptimalValues(self, name='all'):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:\n
        * 'capacityVariables',
        * 'isBuiltVariables',
        * 'operationVariablesOptimum',
        * 'all' or another input: all variables are returned.\n
        :type name: string
        """
        if name == 'capacityVariablesOptimum':
            return {'values': self.capacityVariablesOptimum, 'timeDependent': False, 'dimension': self.dimension}
        elif name == 'isBuiltVariablesOptimum':
            return {'values': self.isBuiltVariablesOptimum, 'timeDependent': False, 'dimension': self.dimension}
        elif name == 'operationVariablesOptimum':
            return {'values': self.operationVariablesOptimum, 'timeDependent': True, 'dimension': self.dimension}
        else:
            return {'capacityVariablesOptimum': {'values': self.capacityVariablesOptimum, 'timeDependent': False,
                                                 'dimension': self.dimension},
                    'isBuiltVariablesOptimum': {'values': self.isBuiltVariablesOptimum, 'timeDependent': False,
                                                'dimension': self.dimension},
                    'operationVariablesOptimum': {'values': self.operationVariablesOptimum, 'timeDependent': True,
                                                  'dimension': self.dimension}}
