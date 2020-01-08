from FINE.conversion import Conversion, ConversionModel
from FINE import utils
import pyomo.environ as pyomo
import pandas as pd


class ConversionFancy(Conversion):
    """
    ToDo
    """
    def __init__(self, esM, name, physicalUnit, commodityConversionFactors, 
                 commodityConversionFactorsPartLoad, hasCapacityVariable=True,
                 capacityVariableDomain='continuous', capacityPerPlantUnit=1, linkedConversionCapacityID=None,
                 hasIsBuiltBinaryVariable=False, bigM=None,
                 operationRateMax=None, operationRateFix=None, tsaWeight=1,
                 locationalEligibility=None, capacityMin=None, capacityMax=None, sharedPotentialID=None,
                 capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerOperation=0, opexPerCapacity=0,
                 opexIfBuilt=0, interestRate=0.08, economicLifetime=10):
        # TODO: allow for data sets as input for the (nonlinear) part load behavoir
        # TODO: make segments for discretization adjustable

        """
        Constructor for creating an Conversion class instance. Capacities are given in the physical unit
        of the plants.
        The Conversion component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param commodityConversionFactorsPartLoad: function or data set describing (nonlinear) part load behavior.
        :type commodityConversionFactorsPartLoad: Lambda function. Initially this will be a function, 
            later also data sets will be possible.
        """
        Conversion.__init__(self, esM, name, physicalUnit, commodityConversionFactors, hasCapacityVariable,
                 capacityVariableDomain, capacityPerPlantUnit, linkedConversionCapacityID,
                 hasIsBuiltBinaryVariable, bigM,
                 operationRateMax, operationRateFix, tsaWeight,
                 locationalEligibility, capacityMin, capacityMax, sharedPotentialID,
                 capacityFix, isBuiltFix,
                 investPerCapacity, investIfBuilt, opexPerOperation, opexPerCapacity,
                 opexIfBuilt, interestRate, economicLifetime)

        self.n_segments = 3
        self.modelingClass = ConversionFancyModel

        utils.checkCommodities(esM, set(commodityConversionFactorsPartLoad.keys()))
        utils.checkCommodityConversionFactorsPartLoad(commodityConversionFactorsPartLoad.values())
        self.commodityConversionFactorsPartLoad = commodityConversionFactorsPartLoad
        self.discretizedPartLoad = utils.getdiscretizedPartLoad(commodityConversionFactorsPartLoad, self.n_segments)
        # {commod: None for commod in commodityConversionFactorsPartLoad.keys()}
     
        # lambda_commod = None
        # non_lambda_commod = None
        # for commod, conversionFactor in commodityConversionFactorsPartLoad.items():
        #     if conversionFactor != 1 and conversionFactor != -1:
        #         ### TODO Test if object is lambda function
        #         self.discretizedPartLoad[commod] = utils.piece_wise_linearization(function=conversionFactor, x_min=0, x_max=1, n_segments=self.n_segments)
        #         lambda_commod = commod
        #     else:
        #         self.discretizedPartLoad[commod] = {
        #             'x_segments': None,
        #             'y_segments': [conversionFactor]*(self.n_segments+1), 
        #             'Rsquared': 1.0, 
        #             'R2values': 1.0
        #             }
        #         non_lambda_commod = commod
        
        # self.discretizedPartLoad[non_lambda_commod]['x_segments'] = self.discretizedPartLoad[lambda_commod]['x_segments']
        # print(self.discretizedPartLoad)

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a LinearOptimalPowerFlow component to the given energy system model.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)


class ConversionFancyModel(ConversionModel):

    """
    A ConversionModel class instance will be instantly created if a Conversion class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Conversion class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The ConversionModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        self.abbrvName = 'fancy'
        self.dimension = '1dim'
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def initDiscretizationPointVarSet(self, pyM):
        """
        Declare discretization variable set of type 1 in the pyomo object for for each node.
        Type 1 represents every start, end, and intermediate point in the piecewise linear function.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        # Set for operation variables
        def initDiscretizationPointVarSet(pyM):
            return ((loc, compName, discreteStep) for compName, comp in compDict.items() \
                    for loc in compDict[compName].locationalEligibility.index if compDict[compName].locationalEligibility[loc] == 1 \
                    for discreteStep in range(compDict[compName].n_segments+1))
        setattr(pyM, 'discretizationPointVarSet_' + abbrvName, pyomo.Set(dimen=3, initialize=initDiscretizationPointVarSet))

    def initDiscretizationSegmentVarSet(self, pyM):
        """
        Declare discretization variable set of type 2 in the pyomo object for for each node.
        Type 2 represents every segment in the piecewise linear function.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        # Set for operation variables
        def initDiscretizationSegmentVarSet(pyM):
            return ((loc, compName, discreteStep) for compName, comp in compDict.items() \
                    for loc in compDict[compName].locationalEligibility.index if compDict[compName].locationalEligibility[loc] == 1 \
                    for discreteStep in range(compDict[compName].n_segments))
        setattr(pyM, 'discretizationSegmentVarSet_' + abbrvName, pyomo.Set(dimen=3, initialize=initDiscretizationSegmentVarSet))

    def declareSets(self, esM, pyM):
        """
        Declare sets and dictionaries: design variable sets, operation variable sets, operation mode sets and
        linked components dictionary.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        # # Declare design variable sets
        self.declareDesignVarSet(pyM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        # Declare operation variable sets
        self.declareOpVarSet(esM, pyM)
        self.initDiscretizationPointVarSet(pyM)
        self.initDiscretizationSegmentVarSet(pyM)

        # Declare operation variable set
        self.declareOperationModeSets(pyM, 'opConstrSet', 'operationRateMax', 'operationRateFix')

        # Declare linked components dictionary
        self.declareLinkedCapacityDict(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareDiscretizationPointVariables(self, pyM):
        """
        Declare discretization point variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        setattr(pyM, 'discretizationPoint_' + self.abbrvName,
                pyomo.Var(getattr(pyM, 'discretizationPointVarSet_' + self.abbrvName), pyM.timeSet, domain=pyomo.NonNegativeReals))


    def declareDiscretizationSegmentBinVariables(self, pyM):
        """
        Declare discretization segment variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        setattr(pyM, 'discretizationSegmentBin_' + self.abbrvName,
                pyomo.Var(getattr(pyM, 'discretizationSegmentVarSet_' + self.abbrvName), pyM.timeSet, domain=pyomo.Binary))


    def declareDiscretizationSegmentConVariables(self, pyM):
        """
        Declare discretization segment variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        setattr(pyM, 'discretizationSegmentCon_' + self.abbrvName,
                pyomo.Var(getattr(pyM, 'discretizationSegmentVarSet_' + self.abbrvName), pyM.timeSet, domain=pyomo.NonNegativeReals))


    def declareVariables(self, esM, pyM):
        """
        Declare design and operation variables.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        # Capacity variables in [commodityUnit]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not [-]
        self.declareBinaryDesignDecisionVars(pyM)
        # Flow over the edges of the components [commodityUnit]
        self.declareOperationVars(pyM, 'op')
        # Operation of component [commodityUnit]
        self.declareDiscretizationPointVariables(pyM)
        # Operation of component [commodityUnit]
        self.declareDiscretizationSegmentBinVariables(pyM)
        # Operation of component [commodityUnit]
        self.declareDiscretizationSegmentConVariables(pyM)

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def segmentSOS1(self, pyM):
        """
        Ensure that the binary segment variables are in sum equal to 1.
        Enforce that only one binary is set to 1, while all other are fixed 0.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationSegmentBinVar = getattr(pyM, 'discretizationSegmentBin_' + self.abbrvName)
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def segmentSOS1(pyM, loc, compName, p, t):
            return sum(discretizationSegmentBinVar[loc, compName, discretStep, p, t] for discretStep in range(compDict[compName].n_segments)) == 1
        setattr(pyM, 'ConstrSegmentSOS1_' + abbrvName,  pyomo.Constraint(opVarSet, pyM.timeSet, rule=segmentSOS1))


    def segmentBigM(self, pyM):
        """
        Ensure that the continuous segment variables are zero if the respective binary variable is zero and unlimited otherwise.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationSegmentConVar = getattr(pyM, 'discretizationSegmentCon_' + self.abbrvName)
        discretizationSegmentBinVar = getattr(pyM, 'discretizationSegmentBin_' + self.abbrvName)
        discretizationSegmentVarSet = getattr(pyM, 'discretizationSegmentVarSet_' + self.abbrvName)

        def segmentBigM(pyM, loc, compName, discretStep, p, t):
            return discretizationSegmentConVar[loc, compName, discretStep, p, t] <= discretizationSegmentBinVar[loc, compName, discretStep, p, t] * compDict[compName].bigM
        setattr(pyM, 'ConstrSegmentBigM_' + abbrvName,  pyomo.Constraint(discretizationSegmentVarSet, pyM.timeSet, rule=segmentBigM))


    def segmentCapacityConstraint(self, pyM, esM):
        """
        Ensure that the continuous segment variables are in sum equal to the installed capacity of the component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationSegmentConVar = getattr(pyM, 'discretizationSegmentCon_' + self.abbrvName)
        capVar = getattr(pyM, 'cap_' + abbrvName)
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def segmentCapacityConstraint(pyM, loc, compName, p, t):
            return sum(discretizationSegmentConVar[loc, compName, discretStep, p, t] for discretStep in range(compDict[compName].n_segments)) == esM.hoursPerTimeStep * capVar[loc, compName]
        setattr(pyM, 'ConstrSegmentCapacity_' + abbrvName,  pyomo.Constraint(opVarSet, pyM.timeSet, rule=segmentCapacityConstraint))


    def pointCapacityConstraint(self, pyM, esM):
        """
        Ensure that the continuous point variables are in sum equal to the installed capacity of the component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(pyM, 'discretizationPoint_' + self.abbrvName)
        capVar = getattr(pyM, 'cap_' + abbrvName)
        opVarSet = getattr(pyM, 'operationVarSet_' + abbrvName)

        def pointCapacityConstraint(pyM, loc, compName, p, t):
            n_points = compDict[compName].n_segments+1
            return sum(discretizationPointConVar[loc, compName, discretStep, p, t] for discretStep in range(n_points)) == esM.hoursPerTimeStep * capVar[loc, compName]
        setattr(pyM, 'ConstrPointCapacity_' + abbrvName,  pyomo.Constraint(opVarSet, pyM.timeSet, rule=pointCapacityConstraint))


    def pointSOS2(self, pyM):
        """
        Ensure that only two consecutive point variables are non-zero while all other point variables are fixed to zero.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(pyM, 'discretizationPoint_' + self.abbrvName)
        discretizationSegmentConVar = getattr(pyM, 'discretizationSegmentCon_' + self.abbrvName)
        discretizationPointVarSet = getattr(pyM, 'discretizationPointVarSet_' + self.abbrvName)

        def pointSOS2(pyM, loc, compName, discretStep, p, t):
            points = list(range(compDict[compName].n_segments+1))
            segments = list(range(compDict[compName].n_segments))

            if discretStep == points[0]:
                return discretizationPointConVar[loc, compName, points[0], p, t] <= discretizationSegmentConVar[loc, compName, segments[0], p, t]
            elif discretStep == points[-1]:
                return discretizationPointConVar[loc, compName, points[-1], p, t] <= discretizationSegmentConVar[loc, compName, segments[-1], p, t]
            else:
                return discretizationPointConVar[loc, compName, discretStep, p, t] <= discretizationSegmentConVar[loc, compName, discretStep-1, p, t] + discretizationSegmentConVar[loc, compName, discretStep, p, t]

        setattr(pyM, 'ConstrPointSOS2_' + abbrvName,  pyomo.Constraint(discretizationPointVarSet, pyM.timeSet, rule=pointSOS2))


    def partLoadOperationOutput(self, pyM):
        """
        Set the required input of a conversion process dependent on the part load efficency.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(pyM, 'discretizationPoint_' + self.abbrvName)
        opVar, opVarSet = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'operationVarSet_' + abbrvName)

        def partLoadOperationOutput(pyM, loc, compName, p, t):        
            n_points = compDict[compName].n_segments+1
            ### TODO Store the part load levels seperately and do not use 
            # print(list(compDict[compName].discretizedPartLoad.keys()))
            return opVar[loc, compName, p, t] == sum(discretizationPointConVar[loc, compName, discretStep, p, t] * \
                                                 compDict[compName].discretizedPartLoad[list(compDict[compName].discretizedPartLoad.keys())[0]]['x_segments'][discretStep] \
                                                 for discretStep in range(n_points))
        setattr(pyM, 'ConstrpartLoadOperationOutput_' + abbrvName,  pyomo.Constraint(opVarSet, pyM.timeSet, rule=partLoadOperationOutput))


    def declareComponentConstraints(self, esM, pyM):
        """
        Declare time independent and dependent constraints.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        super().declareComponentConstraints(esM, pyM)

        ################################################################################################################
        #                                         Add piecewise linear part load efficiency constraints                                        #
        ################################################################################################################

        self.segmentSOS1(pyM)
        self.segmentBigM(pyM)
        self.segmentCapacityConstraint(pyM, esM)
        self.pointCapacityConstraint(pyM, esM)
        self.pointSOS2(pyM)
        self.partLoadOperationOutput(pyM)

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

        :param esM: EnergySystemModel in which the LinearOptimalPowerFlow components have been added to.
        :type esM: esM - EnergySystemModel class instance

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: Name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """
        return super().hasOpVariablesForLocationCommodity(esM, loc, commod)

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """ Get contribution to a commodity balance. """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDict = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'operationVarDict_' + abbrvName)
        discretizationPointConVar = getattr(pyM, 'discretizationPoint_' + self.abbrvName)
        
        return sum(sum(discretizationPointConVar[loc, compName, discretStep, p, t] * \
                       compDict[compName].discretizedPartLoad[commod]['x_segments'][discretStep] * \
                       compDict[compName].discretizedPartLoad[commod]['y_segments'][discretStep] for discretStep in range(compDict[compName].n_segments+1)) \
                   for compName in opVarDict[loc] if commod in compDict[compName].discretizedPartLoad)

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        return super().getObjectiveFunctionContribution(esM, pyM)

    def setOptimalValues(self, esM, pyM):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        super().setOptimalValues(esM, pyM)

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointVariables = getattr(pyM, 'discretizationPoint_' + abbrvName)
        discretizationSegmentConVariables = getattr(pyM, 'discretizationSegmentCon_' + abbrvName)
        discretizationSegmentBinVariables = getattr(pyM, 'discretizationSegmentBin_' + abbrvName)

        discretizationPointVariablesOptVal_ = utils.formatOptimizationOutput(discretizationPointVariables.get_values(), 'operationVariables', '1dim',
                                                 esM.periodsOrder)
        discretizationSegmentConVariablesOptVal_ = utils.formatOptimizationOutput(discretizationSegmentConVariables.get_values(), 'operationVariables', '1dim',
                                                 esM.periodsOrder)
        discretizationSegmentBinVariablesOptVal_ = utils.formatOptimizationOutput(discretizationSegmentBinVariables.get_values(), 'operationVariables', '1dim',
                                                 esM.periodsOrder)

        self.discretizationPointVariablesOptimun = discretizationPointVariablesOptVal_
        self.discretizationSegmentConVariablesOptimun = discretizationSegmentConVariablesOptVal_
        self.discretizationSegmentBinVariablesOptimun = discretizationSegmentBinVariablesOptVal_

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
        # return super().getOptimalValues(name)
        if name == 'capacityVariablesOptimum':
            return {'values': self.capacityVariablesOptimum, 'timeDependent': False, 'dimension': self.dimension}
        elif name == 'isBuiltVariablesOptimum':
            return {'values': self.isBuiltVariablesOptimum, 'timeDependent': False, 'dimension': self.dimension}
        elif name == 'operationVariablesOptimum':
            return {'values': self.operationVariablesOptimum, 'timeDependent': True, 'dimension': self.dimension}
        elif name == 'discretizationPointVariablesOptimun':
            return {'values': self.discretizationPointVariablesOptimun, 'timeDependent': True, 'dimension': self.dimension}
        elif name == 'discretizationSegmentConVariablesOptimun':
            return {'values': self.discretizationSegmentConVariablesOptimun, 'timeDependent': True, 'dimension': self.dimension}
        elif name == 'discretizationSegmentBinVariablesOptimun':
            return {'values': self.discretizationSegmentBinVariablesOptimun, 'timeDependent': True, 'dimension': self.dimension}
        else:
            return {'capacityVariablesOptimum': {'values': self.capacityVariablesOptimum, 'timeDependent': False,
                                                 'dimension': self.dimension},
                    'isBuiltVariablesOptimum': {'values': self.isBuiltVariablesOptimum, 'timeDependent': False,
                                                'dimension': self.dimension},
                    'operationVariablesOptimum': {'values': self.operationVariablesOptimum, 'timeDependent': True,
                                                  'dimension': self.dimension},
                    'discretizationPointVariablesOptimun': {'values': self.discretizationPointVariablesOptimun, 'timeDependent': True,
                                                  'dimension': self.dimension},
                    'discretizationSegmentConVariablesOptimun': {'values': self.discretizationSegmentConVariablesOptimun, 'timeDependent': True,
                                                  'dimension': self.dimension},
                    'discretizationSegmentBinVariablesOptimun': {'values': self.discretizationSegmentBinVariablesOptimun, 'timeDependent': True,
                                                  'dimension': self.dimension}}