from FINE.conversion import Conversion, ConversionModel
from FINE import utils
import pyomo.environ as pyomo
import pandas as pd



class ConversionFancy(Conversion):
    """
    Extension of the conversion class with more specific ramping behavior
    """
    def __init__(self, esM, name, physicalUnit, commodityConversionFactors, hasCapacityVariable=True,
                 capacityVariableDomain='continuous', capacityPerPlantUnit=1, linkedConversionCapacityID=None,
                 hasIsBuiltBinaryVariable=False, bigM=None,
                 operationRateMax=None, operationRateFix=None, tsaWeight=1,
                 locationalEligibility=None, capacityMin=None, capacityMax=None, partLoadMin=None, downTimeMin=None, sharedPotentialID=None,
                 capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerOperation=0, opexPerCapacity=0,
                 opexIfBuilt=0, interestRate=0.08, economicLifetime=10):
        """
        Constructor for creating a ConversionFancy class instance.
        The ConversionFancy component specific input arguments are described below. The Conversion
        specific input arguments are described in the Conversion class and the general component
        input arguments are described in the Component class.

        **Required arguments:**
        :param downTimeMin: if specified, indicates minimal down time of the component [number of time steps]. 
        :type downTimeMin:
            * None or
            * Integer value in range [0;numberOfTimeSteps]
        """

        
        Conversion. __init__(self, esM, name, physicalUnit, commodityConversionFactors, hasCapacityVariable, capacityVariableDomain, capacityPerPlantUnit, linkedConversionCapacityID,
                             hasIsBuiltBinaryVariable, bigM, operationRateMax, operationRateFix, tsaWeight,
                             locationalEligibility, capacityMin, capacityMax, partLoadMin, sharedPotentialID,
                             capacityFix, isBuiltFix, investPerCapacity, investIfBuilt, opexPerOperation, opexPerCapacity,
                             opexIfBuilt, interestRate, economicLifetime)


        self.modelingClass = ConversionFancyModel
        self.downTimeMin = downTimeMin
        utils.checkConversionFancySpecficDesignInputParams(self, esM)     

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a ConversionFancy component to the given energy system model.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)


class ConversionFancyModel(ConversionModel):

    """
    A ConversionFancyModel class instance will be instantly created if a ConversionFancy class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the ConversionFancy
    class instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The ConversionFancyModel class inherits from the ConversionModel class. """

    def __init__(self):
        self.abbrvName = 'conv_fancy'
        self.dimension = '1dim'
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = None
        self.optSummary = None
        
    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareOperationStartStopBinarySet(self, pyM):
        """
        Declare operation related sets for binary decicion variables (operation variables) in the pyomo object for a
        modeling class. This reflects the starting or stopping state of the conversion component.        
        
        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel  
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        def declareOperationBinarySet(pyM):
            return ((loc, compName) for compName, comp in compDict.items() 
                for loc in comp.locationalEligibility.index if comp.locationalEligibility[loc] == 1)
        setattr(pyM, 'operationVarStartStopSetBin_' + abbrvName, pyomo.Set(dimen=2, initialize=declareOperationBinarySet))
        
        
    def declareOpConstrSetMinDownTime(self, pyM, constrSetName):
        """
        Declare set of locations and components for which downTimeMin is not NONE.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, 'operationVarStartStopSetBin_' + abbrvName)

        def declareOpConstrSetMinDownTime(pyM):
            return ((loc, compName) for loc, compName in varSet if getattr(compDict[compName], 'downTimeMin') is not None)

        setattr(pyM, constrSetName + 'downTimeMin_' + abbrvName, pyomo.Set(dimen=2, initialize=declareOpConstrSetMinDownTime))
    
    def declareSets(self, esM, pyM):
        """
        Declare sets and dictionaries: design variable sets, operation variable set, operation mode sets and
        linked components dictionary.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        # Declare design variable sets
        self.declareDesignVarSet(pyM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        # Declare operation variable sets
        self.declareOpVarSet(esM, pyM)
        self.declareOperationBinarySet(pyM)
        self.declareOperationStartStopBinarySet(pyM)

        # Declare operation mode sets
        self.declareOperationModeSets(pyM, 'opConstrSet', 'operationRateMax', 'operationRateFix')
        self.declareOpConstrSetMinDownTime(pyM, 'opConstrSet')

        # Declare linked components dictionary
        self.declareLinkedCapacityDict(pyM)
        
   
    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareStartStopVariables(self, pyM):
        """
        Declare start stop variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        setattr(pyM, 'startVariable_' + self.abbrvName,
                pyomo.Var(getattr(pyM, 'operationVarStartStopSetBin_' + self.abbrvName), pyM.timeSet, domain=pyomo.Binary))
        
        setattr(pyM, 'stopVariable_' + self.abbrvName,
                pyomo.Var(getattr(pyM, 'operationVarStartStopSetBin_' + self.abbrvName), pyM.timeSet, domain=pyomo.Binary))
        

    def declareVariables(self, esM, pyM):
        """
        Declare design and operation variables

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        super().declareVariables(esM, pyM)
               
        self.declareStartStopVariables(pyM)
        

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def minimumDownTime(self, pyM):
        """
        Ensure that conversion unit is not ramping up and down too often by implementing a minimum down time after ramping down.
        

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        
        opVarBin= getattr(pyM, 'op_bin_' + abbrvName)
        opVarStartBin, opVarStopBin = getattr(pyM, 'startVariable_' + abbrvName), getattr(pyM, 'stopVariable_' + abbrvName)
        constrSetMinDownTime = getattr(pyM,'opConstrSet' + 'downTimeMin_' + abbrvName)
        

        def minimumDownTime1(pyM, loc, compName, p, t):
            if t>=1:
                return (opVarBin[loc, compName, p, t]-opVarBin[loc, compName, p, t-1]-opVarStartBin[loc, compName, p, t]+opVarStopBin[loc, compName, p, t] == 0)
            else:
                return pyomo.Constraint.Skip
        setattr(pyM, 'ConstrMinDownTime1_' + abbrvName, pyomo.Constraint(constrSetMinDownTime, pyM.timeSet, rule=minimumDownTime1))
          
        def minimumDownTime2(pyM, loc, compName, p, t):
            downTimeMin = getattr(compDict[compName], 'downTimeMin')
            if t >= downTimeMin:
                return opVarBin[loc, compName, p, t] <= 1 -pyomo.quicksum(opVarStopBin[loc, compName, p, t_down] for t_down in range(t-downTimeMin+1, t))
            else:
                return opVarBin[loc, compName, p, t] <= 1 -pyomo.quicksum(opVarStopBin[loc, compName, p, t_down] for t_down in range(0, t))

        setattr(pyM, 'ConstrMinDownTime2_' + abbrvName, pyomo.Constraint(constrSetMinDownTime, pyM.timeSet, rule=minimumDownTime2))          
                    

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
        #                                         Fancy Constraints                                        #
        ################################################################################################################

        self.minimumDownTime(pyM)
        
        

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
        return super().getCommodityBalanceContribution(pyM, commod, loc, p, t)

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
                    