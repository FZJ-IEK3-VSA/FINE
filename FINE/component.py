from abc import ABCMeta, abstractmethod
from FINE import utils
import warnings
import pyomo.environ as pyomo
import pandas as pd


class Component(metaclass=ABCMeta):
    """
    Doc
    """
    def __init__(self, esM, name, dimension,
                 hasCapacityVariable, capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                 hasIsBuiltBinaryVariable=False, bigM=None, locationalEligibility=None,
                 capacityMin=None, capacityMax=None, sharedPotentialID=None, capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerCapacity=0, opexIfBuilt=0,
                 interestRate=0.08, economicLifetime=10):
        """
        Constructor for creating an Conversion class instance.

        **Required arguments:**

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param name: name of the component. Has to be unique (i.e. no other components with that name can
        already exist in the EnergySystemModel instance to which the component is added).
        :type name: string

        :param hasCapacityVariable: specifies if the component should be modeled with a capacity or not.
            Examples:
            (a) An electrolyzer has a capacity given in GW_electric -> hasCapacityVariable is True.
            (b) In the energy system, biogas can, from a model perspective, be converted into methane (and then
                used in conventional power plants which emit CO2) by getting CO2 from the environment. Thus,
                using biogas in conventional power plants is, from a balance perspective, CO2 free. This
                conversion is purely theoretical and does not require a capacity -> hasCapacityVariable
                is False.
            (c) A electricity cable has a capacity given in GW_electric -> hasCapacityVariable is True.
            (d) If the transmission capacity of a component is unlimited hasCapacityVariable is False.
            (e) A wind turbine has a capacity given in GW_electric -> hasCapacityVariable is True.
            (f) Emitting CO2 into the environment is not per se limited by a capacity ->
                hasCapacityVariable is False.
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

        :param hasIsBuiltBinaryVariable: specifies if binary decision variables should be declared for
            * each eligible location of the component, which indicate if the component is built at that location or
              not (dimension=1dim).
            * each eligible connection of the transmission component, which indicate if the component is built
              between two locations or not (dimension=2dim).
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

        :param locationalEligibility: Pandas
            * Series that indicates if a component can be built at a location (=1) or not (=0)
              (dimension=1dim) or
            * Pandas DataFrame that indicates if a component can be built between two locations
              (=1) or not (=0) (dimension=2dim).
            If not specified and a maximum or fixed capacity or time series is given, the parameter will be
            set based on these inputs. If the parameter is specified, a consistency check is done to ensure
            that the parameters indicate the same locational eligibility. If the parameter is not specified
            and also no other of the parameters is specified it is assumed that the component is eligible in
            each location and all values are set to 1.
            This parameter is key for ensuring small built times of the optimization problem by avoiding the
            declaration of unnecessary variables and constraints.
            |br| * the default value is None
        :type locationalEligibility:
            * None or
            * Pandas Series with values equal to 0 and 1. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with values equal to 0 and 1. The column and row indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param capacityMin: if specified, Pandas Series (dimension=1dim) or Pandas DataFrame (dimension=2dim)
            indicating minimum capacities else None. If binary decision variables are declared the minimum
            capacity is only enforced if the component is built .
            |br| * the default value is None
        :type capacityMin:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param capacityMax: if specified, Pandas Series (dimension=1dim) or Pandas DataFrame (dimension=2dim)
            indicating maximum capacities else None.
            |br| * the default value is None
        :type capacityMax:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param sharedPotentialID: if specified, indicates that the component has to share its maximum
            potential capacity with other components (i.e. due to space limitations). The shares of how
            much of the maximum potential is used have to add up to less then 100%.
            |br| * the default value is None
        :type sharedPotentialID: string

        :param capacityFix: if specified, Pandas Series (dimension=1dim) or Pandas DataFrame (dimension=2dim)
            indicating fixed capacities else None.
            |br| * the default value is None
        :type capacityFix:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param isBuiltFix: if specified, Pandas Series (dimension=1dim) or Pandas DataFrame (dimension=2dim)
            indicating fixed decisions in which or between which locations the component is built (i.e. sets
            the isBuilt binary variables) else None.
            |br| * the default value is None
        :type isBuiltFix:
            * None or
            * Pandas Series with with values equal to 0 and 1. The indices of the series have to equal the
              in the energy system model specified locations or
            * Pandas DataFrame with values equal to 0 and 1. The row and column indices of the DataFrame
              have to equal the in the energy system model specified locations.

        :param investPerCapacity: the invest of a component is obtained by multiplying the built capacities
            of the component (in the physicalUnit of the component) with the investPerCapacity factor.
            The investPerCapacity can either be given as
            * a float or a Pandas Series with location specific values. The cost unit in which the parameter
              is given has to match the one specified in the energy system model (i.e. Euro, Dollar,
              1e6 Euro) (dimension=1dim) or
            * a float or a Pandas DataFrame with location specific values. The cost unit in which the
              parameter is given has to match the one specified in the energy system model divided by
              the there specified lengthUnit (i.e. Euro/m, Dollar/m, 1e6 Euro/km) (dimension=2dim)
            |br| * the default value is 0
        :type investPerCapacity:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param investIfBuilt: a capacity-independent invest which only arises in a location if a component
            is built at that location. The investIfBuilt can either be given as
            * a float or a Pandas Series with location specific values. The cost unit in which the parameter
              is given has to match the one specified in the energy system model (i.e. Euro, Dollar,
              1e6 Euro) (dimension=1dim) or
            * a float or a Pandas DataFrame with location specific values. The cost unit in which the
              parameter is given has to match the one specified in the energy system model divided by
              the there specified lengthUnit (i.e. Euro/m, Dollar/m, 1e6 Euro/km) (dimension=2dim)
            |br| * the default value is 0
        :type investIfBuilt:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param opexPerCapacity: annual operational cost which are only a function of the capacity of the
            component (in the physicalUnit of the component) and not of the specific operation itself are
            obtained by multiplying the capacity of the component at a location with the opexPerCapacity
            factor. The opexPerCapacity can either be given as
            * a float or a Pandas Series with location specific values. The cost unit in which the parameter
              is given has to match the one specified in the energy system model (i.e. Euro, Dollar,
              1e6 Euro) (dimension=1dim) or
            * a float or a Pandas DataFrame with location specific values. The cost unit in which the
              parameter is given has to match the one specified in the energy system model divided by
              the there specified lengthUnit (i.e. Euro/m, Dollar/m, 1e6 Euro/km) (dimension=2dim)
            |br| * the default value is 0
        :type opexPerCapacity:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param opexIfBuilt: a capacity-independent annual operational cost which only arises in a location
            if a component is built at that location. The opexIfBuilt can either be given as
            * a float or a Pandas Series with location specific values. The cost unit in which the parameter
              is given has to match the one specified in the energy system model (i.e. Euro, Dollar,
              1e6 Euro) (dimension=1dim) or
            * a float or a Pandas DataFrame with location specific values. The cost unit in which the
              parameter is given has to match the one specified in the energy system model divided by
              the there specified lengthUnit (i.e. Euro/m, Dollar/m, 1e6 Euro/km) (dimension=2dim)
            |br| * the default value is 0
        :type opexIfBuilt:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param interestRate: interest rate which is considered for computing the annuities of the invest
            of the component (depreciates the invests over the economic lifetime).
            A value of 0.08 corresponds to an interest rate of 8%.
            |br| * the default value is 0.08
        :type interestRate:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param economicLifetime: economic lifetime of the component which is considered for computing the
            annuities of the invest of the component (aka depreciation time).
            |br| * the default value is 10
        :type economicLifetime:
            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations or
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
        utils.checkLocationSpecficDesignInputParams(esM, hasCapacityVariable, hasIsBuiltBinaryVariable,
                                                    capacityMin, capacityMax, capacityFix,
                                                    locationalEligibility, isBuiltFix, sharedPotentialID,
                                                    dimension=dimension)
        self.locationalEligibility = locationalEligibility
        self.sharedPotentialID = sharedPotentialID
        self.capacityMin, self.capacityMax, self.capacityFix = capacityMin, capacityMax, capacityFix
        self.isBuiltFix = isBuiltFix
        #
        # # Variables at optimum (set after optimization)
        # self.capacityVariablesOptimum = None
        # self.isBuiltVariablesOptimum = None
        # self.operationVariablesOptimum = {}

    def addToEnergySystemModel(self, esM):
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
        data_ = rateFix if rateFix is not None else rateMax
        if data_ is not None:
            data_ = data_.copy()
            uniqueIdentifiers = [self.name + rateName + loc for loc in data_.columns]
            data_.rename(columns={loc: self.name + rateName + loc for loc in data_.columns}, inplace=True)
            weightDict.update({id: rateWeight for id in uniqueIdentifiers}), data.append(data_)
        return weightDict, data

    def getTSAOutput(self, rate, rateName, data):
        if rate is not None:
            uniqueIdentifiers = [self.name + rateName + loc for loc in rate.columns]
            data_ = data[uniqueIdentifiers].copy()
            data_.rename(columns={self.name + rateName + loc: loc for loc in rate.columns}, inplace=True)
            return data_
        else:
            return None

    @abstractmethod
    def setTimeSeriesData(self, hasTSA):
        raise NotImplementedError

    @abstractmethod
    def getDataForTimeSeriesAggregation(self):
        raise NotImplementedError

    @abstractmethod
    def setAggregatedTimeSeriesData(self, data):
        raise NotImplementedError


class ComponentModel(metaclass=ABCMeta):
    """
    Doc
    """
    def __init__(self):
        self.abbrvName = ''
        self.dimension = ''
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = {}
        self.optSummary = None

    ####################################################################################################################
    #                           Functions for declaring design and operation variables sets                            #
    ####################################################################################################################

    def initDesignVarSet(self, pyM):
        """ Declares set for capacity variables in the pyomo object for a modeling class """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def initDesignVarSet(pyM):
            return ((loc, compName) for compName, comp in compDict.items()
                    for loc in comp.locationalEligibility.index
                    if comp.locationalEligibility[loc] == 1 and comp.hasCapacityVariable)
        setattr(pyM, 'designDimensionVarSet_' + abbrvName, pyomo.Set(dimen=2, initialize=initDesignVarSet))

    def initContinuousDesignVarSet(self, pyM):
        """ Declares set for continuous number of installed components in the pyomo object for a modeling class """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def initContinuousDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in getattr(pyM, 'designDimensionVarSet_' + abbrvName)
                    if compDict[compName].capacityVariableDomain == 'continuous')
        setattr(pyM, 'continuousDesignDimensionVarSet_' + abbrvName,
                pyomo.Set(dimen=2, initialize=initContinuousDesignVarSet))

    def initDiscreteDesignVarSet(self, pyM):
        """ Declares set for discrete number of installed components in the pyomo object for a modeling class """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def initDiscreteDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in getattr(pyM, 'designDimensionVarSet_' + abbrvName)
                    if compDict[compName].capacityVariableDomain == 'discrete')
        setattr(pyM, 'discreteDesignDimensionVarSet_' + abbrvName,
                pyomo.Set(dimen=2, initialize=initDiscreteDesignVarSet))

    def initDesignDecisionVarSet(self, pyM):
        """ Declares set for design decision variables in the pyomo object for a modeling class """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        def initDesignDecisionVarSet(pyM):
            return ((loc, compName) for loc, compName in getattr(pyM, 'designDimensionVarSet_' + abbrvName)
                    if compDict[compName].hasIsBuiltBinaryVariable)
        setattr(pyM, 'designDecisionVarSet_' + abbrvName, pyomo.Set(dimen=2, initialize=initDesignDecisionVarSet))

    def initOpVarSet(self, esM, pyM):
        """
        Declares operation related sets (operation variables and mapping sets) in the pyomo object for a
        modeling class
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        # Set for operation variables
        def initOpVarSet(pyM):
            return ((loc, compName) for compName, comp in compDict.items()
                    for loc in comp.locationalEligibility.index if comp.locationalEligibility[loc] == 1)
        setattr(pyM, 'operationVarSet_' + abbrvName, pyomo.Set(dimen=2, initialize=initOpVarSet))

        # TODO more generic formulation?
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

    def declareOperationModeSets(self, pyM, constrSetName, rateMax, rateFix):
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def initOpConstrSet1(pyM):
            return ((loc, compName) for loc, compName in varSet if compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateMax) is None
                    and getattr(compDict[compName], rateFix) is None)
        setattr(pyM, constrSetName + '1_' + abbrvName, pyomo.Set(dimen=2, initialize=initOpConstrSet1))

        def initOpConstrSet2(pyM):
            return ((loc, compName) for loc, compName in varSet if compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateFix) is not None)
        setattr(pyM, constrSetName + '2_' + abbrvName, pyomo.Set(dimen=2, initialize=initOpConstrSet2))

        def initOpConstrSet3(pyM):
            return ((loc, compName) for loc, compName in varSet if compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateMax) is not None)
        setattr(pyM, constrSetName + '3_' + abbrvName, pyomo.Set(dimen=2, initialize=initOpConstrSet3))

        def initOpConstrSet4(pyM):
            return ((loc, compName) for loc, compName in varSet if not compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateFix) is not None)
        setattr(pyM, constrSetName + '4_' + abbrvName, pyomo.Set(dimen=2, initialize=initOpConstrSet4))

        def initOpConstrSet5(pyM):
            return ((loc, compName) for loc, compName in varSet if not compDict[compName].hasCapacityVariable
                    and getattr(compDict[compName], rateMax) is not None)
        setattr(pyM, constrSetName + '5_' + abbrvName, pyomo.Set(dimen=2, initialize=initOpConstrSet5))

    ####################################################################################################################
    #                                         Functions for declaring variables                                        #
    ####################################################################################################################

    def declareCapacityVars(self, pyM):
        """ Declares capacity variables """
        abbrvName = self.abbrvName

        def capBounds(pyM, loc, compName):
            """ Function for setting lower and upper capacity bounds """
            comp = self.componentsDict[compName]
            return (comp.capacityMin[loc] if (comp.capacityMin is not None and not comp.hasIsBuiltBinaryVariable)
                    else 0,
                    comp.capacityMax[loc] if comp.capacityMax is not None else None)
        setattr(pyM, 'cap_' + abbrvName, pyomo.Var(getattr(pyM, 'designDimensionVarSet_' + abbrvName),
                domain=pyomo.NonNegativeReals, bounds=capBounds))

    def declareRealNumbersVars(self, pyM):
        """ Declares variables representing the (continuous) number of installed components [-] """
        abbrvName = self.abbrvName
        setattr(pyM, 'nbReal_' + abbrvName, pyomo.Var(getattr(pyM, 'continuousDesignDimensionVarSet_' + abbrvName),
                domain=pyomo.NonNegativeReals))

    def declareIntNumbersVars(self, pyM):
        """ Declares variables representing the (discrete/integer) number of installed components [-] """
        abbrvName = self.abbrvName
        setattr(pyM, 'nbInt_' + abbrvName, pyomo.Var(getattr(pyM, 'discreteDesignDimensionVarSet_' + abbrvName),
                domain=pyomo.NonNegativeIntegers))

    def declareBinaryDesignDecisionVars(self, pyM):
        """ Declares binary variables [-] indicating if a component is considered at a location or not [-] """
        abbrvName = self.abbrvName
        setattr(pyM, 'designBin_' + abbrvName, pyomo.Var(getattr(pyM, 'designDecisionVarSet_' + abbrvName),
                domain=pyomo.Binary))

    def declareOperationVars(self, pyM, opVarName):
        """ Declares operation variables """
        abbrvName = self.abbrvName
        setattr(pyM, opVarName + '_' + abbrvName,
                pyomo.Var(getattr(pyM, 'operationVarSet_' + abbrvName), pyM.timeSet, domain=pyomo.NonNegativeReals))

    ####################################################################################################################
    #                              Functions for declaring time independent constraints                                #
    ####################################################################################################################

    def capToNbReal(self, pyM):
        """ Determine the components' capacities from the number of installed units """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, nbRealVar = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'nbReal_' + abbrvName)
        nbRealVarSet = getattr(pyM, 'continuousDesignDimensionVarSet_' + abbrvName)

        def capToNbReal(pyM, loc, compName):
            return capVar[loc, compName] == nbRealVar[loc, compName] * compDict[compName].capacityPerPlantUnit
        setattr(pyM, 'ConstrCapToNbReal_' + abbrvName, pyomo.Constraint(nbRealVarSet, rule=capToNbReal))

    def capToNbInt(self, pyM):
        """ Determine the components' capacities from the number of installed units """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, nbIntVar = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'nbInt_' + abbrvName)
        nbIntVarSet = getattr(pyM, 'discreteDesignDimensionVarSet_' + abbrvName)

        def capToNbInt(pyM, loc, compName):
            return capVar[loc, compName] == nbIntVar[loc, compName] * compDict[compName].capacityPerPlantUnit
        setattr(pyM, 'ConstrCapToNbInt_' + abbrvName, pyomo.Constraint(nbIntVarSet, rule=capToNbInt))

    def bigM(self, pyM):
        """ Enforce the consideration of the binary design variables of a component """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, designBinVar = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'designBin_' + abbrvName)
        designBinVarSet = getattr(pyM, 'designDecisionVarSet_' + abbrvName)

        def bigM(pyM, loc, compName):
            return capVar[loc, compName] <= designBinVar[loc, compName] * compDict[compName].bigM
        setattr(pyM, 'ConstrBigM_' + abbrvName, pyomo.Constraint(designBinVarSet, rule=bigM))

    def capacityMinDec(self, pyM):
        """ Enforce the consideration of minimum capacities for components with design decision variables """
        compDict, abbrvName, dim = self.componentsDict, self.abbrvName, self.dimension
        capVar, designBinVar = getattr(pyM, 'cap_' + abbrvName), getattr(pyM, 'designBin_' + abbrvName)
        designBinVarSet = getattr(pyM, 'designDecisionVarSet_' + abbrvName)

        def capacityMinDec(pyM, loc, compName):
            return (capVar[loc, compName] >= compDict[compName].capacityMin[loc] * designBinVar[loc, compName]
                    if compDict[compName].capacityMin is not None else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrCapacityMinDec_' + abbrvName, pyomo.Constraint(designBinVarSet, rule=capacityMinDec))

    def capacityFix(self, pyM):
        """ Sets, if applicable, the installed capacities of a component """
        compDict, abbrvName, dim = self.componentsDict, self.abbrvName, self.dimension
        capVar = getattr(pyM, 'cap_' + abbrvName)
        capVarSet = getattr(pyM, 'designDimensionVarSet_' + abbrvName)

        def capacityFix(pyM, loc, compName):
            return (capVar[loc, compName] == compDict[compName].capacityFix[loc]
                    if compDict[compName].capacityFix is not None else pyomo.Constraint.Skip)
        setattr(pyM, 'ConstrCapacityFix_' + abbrvName, pyomo.Constraint(capVarSet, rule=capacityFix))

    def designBinFix(self, pyM):
        """ Sets, if applicable, the installed capacities of a component """
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
        """ Defines operation modes """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet1 = getattr(pyM, constrSetName + '1_' + abbrvName)
        factor1 = 1 if isStateOfCharge else esM.hoursPerTimeStep

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        def op1(pyM, loc, compName, p, t):
            factor2 = 1 if factorName is None else getattr(compDict[compName], factorName)
            return opVar[loc, compName, p, t] <= factor1 * factor2 * capVar[loc, compName]
        setattr(pyM, constrName + '1_' + abbrvName, pyomo.Constraint(constrSet1, pyM.timeSet, rule=op1))

    def operationMode2(self, pyM, esM, constrName, constrSetName, opVarName, isStateOfCharge=False):
        """ Defines operation modes """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet2 = getattr(pyM, constrSetName + '2_' + abbrvName)
        factor = 1 if isStateOfCharge else esM.hoursPerTimeStep

        # Operation [energyUnit] equal to the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op2(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] == capVar[loc, compName] * \
                   compDict[compName].operationRateFix[loc][p, t] * factor
        setattr(pyM, constrName + '2_' + abbrvName, pyomo.Constraint(constrSet2, pyM.timeSet, rule=op2))

    def operationMode3(self, pyM, esM, constrName, constrSetName, opVarName, isStateOfCharge=False):
        """ Defines operation modes """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet3 = getattr(pyM, constrSetName + '3_' + abbrvName)
        factor = 1 if isStateOfCharge else esM.hoursPerTimeStep

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op3(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] <= capVar[loc, compName] * \
                   compDict[compName].operationRateMax[loc][p, t] * factor
        setattr(pyM, constrName + '3_' + abbrvName, pyomo.Constraint(constrSet3, pyM.timeSet, rule=op3))

    def operationMode4(self, pyM, esM, constrName, constrSetName, opVarName):
        """ Defines operation modes """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet4 = getattr(pyM, constrSetName + '4_' + abbrvName)

        # Operation [energyUnit] equal to the operation time series [energyUnit]
        def op4(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] == compDict[compName].operationRateFix[loc][p, t]
        setattr(pyM, constrName + '4_' + abbrvName, pyomo.Constraint(constrSet4, pyM.timeSet, rule=op4))

    def operationMode5(self, pyM, esM, constrName, constrSetName, opVarName):
        """ Defines operation modes """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + '_' + abbrvName), getattr(pyM, 'cap_' + abbrvName)
        constrSet5 = getattr(pyM, constrSetName + '5_' + abbrvName)

        # Operation [energyUnit] limited by the operation time series [energyUnit]
        def op5(pyM, loc, compName, p, t):
            return opVar[loc, compName, p, t] <= compDict[compName].operationRateMax[loc][p, t]
        setattr(pyM, constrName + '5_' + abbrvName, pyomo.Constraint(constrSet5, pyM.timeSet, rule=op5))

    ####################################################################################################################
    #  Functions for declaring component contributions to basic energy system constraints and the objective function   #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        """
        Gets the share which the components of the modeling class have on a shared maximum potential at a location
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

    def setOptimalValues(self, esM, pyM, indexColumns, plantUnit, unitApp='', costApp=1):
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

        if optVal is not None:
            i = optVal.apply(lambda cap: cap * compDict[cap.name].investPerCapacity[cap.index], axis=1)
            cx = optVal.apply(lambda cap: cap * compDict[cap.name].investPerCapacity[cap.index] /
                                          compDict[cap.name].CCF[cap.index], axis=1)
            ox = optVal.apply(lambda cap: cap * compDict[cap.name].opexPerCapacity[cap.index], axis=1)
            optSummary.loc[
                [(ix, 'capacity', '[' + getattr(compDict[ix], plantUnit) + unitApp + ']') for ix in optVal.index],
                optVal.columns] = optVal.values
            optSummary.loc[[(ix, 'invest', '[' + esM.costUnit + ']') for ix in i.index], i.columns] = \
                i.values * costApp
            optSummary.loc[[(ix, 'capexCap', '[' + esM.costUnit + '/a]') for ix in cx.index], cx.columns] = \
                cx.values * costApp
            optSummary.loc[[(ix, 'opexCap', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                ox.values * costApp

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
                i.values * costApp
            optSummary.loc[[(ix, 'capexIfBuilt', '[' + esM.costUnit + '/a]') for ix in cx.index],
                           cx.columns] = cx.values * costApp
            optSummary.loc[[(ix, 'opexIfBuilt', '[' + esM.costUnit + '/a]') for ix in ox.index],
                           ox.columns] = ox.values * costApp

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'capexCap') |
                           (optSummary.index.get_level_values(1) == 'opexCap') |
                           (optSummary.index.get_level_values(1) == 'capexIfBuilt')].groupby(level=0).sum().values

        return optSummary

    def getOptimalValues(self):
        return {'capacityVariables': {'values': self.capacityVariablesOptimum, 'timeDependent': False,
                                      'dimension': self.dimension},
                'isBuiltVariables': {'values': self.isBuiltVariablesOptimum, 'timeDependent': False,
                                     'dimension': self.dimension},
                'operationVariablesOptimum': {'values': self.operationVariablesOptimum, 'timeDependent': True,
                                              'dimension': self.dimension}}

    @abstractmethod
    def declareVariables(self, esM, pyM):
        raise NotImplementedError

    @abstractmethod
    def declareComponentConstraints(self, esM, pyM):
        raise NotImplementedError

    @abstractmethod
    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        raise NotImplementedError

    @abstractmethod
    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        raise NotImplementedError

    @abstractmethod
    def getObjectiveFunctionContribution(self, esM, pyM):
        raise NotImplementedError
