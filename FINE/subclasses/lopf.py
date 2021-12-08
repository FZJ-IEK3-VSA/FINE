from FINE.transmission import Transmission, TransmissionModel
from FINE import utils
import pyomo.environ as pyomo
import pandas as pd


class LinearOptimalPowerFlow(Transmission):
    """
    A LinearOptimalPowerFlow component shows the behavior of a Transmission component but additionally models a
    linearized power flow (i.e. for AC lines). The LinearOptimalPowerFlow class inherits from the Transmission
    class.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
        reactances,
        losses=0,
        distances=None,
        hasCapacityVariable=True,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=1,
        hasIsBuiltBinaryVariable=False,
        bigM=None,
        operationRateMax=None,
        operationRateFix=None,
        tsaWeight=1,
        locationalEligibility=None,
        capacityMin=None,
        capacityMax=None,
        partLoadMin=None,
        sharedPotentialID=None,
        capacityFix=None,
        isBuiltFix=None,
        investPerCapacity=0,
        investIfBuilt=0,
        opexPerOperation=0,
        opexPerCapacity=0,
        opexIfBuilt=0,
        QPcostScale=0,
        interestRate=0.08,
        economicLifetime=10,
        technicalLifetime=None,
    ):
        """
        Constructor for creating an LinearOptimalPowerFlow class instance.
        The LinearOptimalPowerFlow component specific input arguments are described below. The Transmission
        component specific input arguments are described in the Transmission class and the general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param reactances: reactances for DC power flow modeling (of AC lines) given as a Pandas DataFrame. The row and column indices of the DataFrame have to equal
            the in the energy system model specified locations.
        :type reactances: Pandas DataFrame.
        """
        Transmission.__init__(
            self,
            esM,
            name,
            commodity,
            losses=losses,
            distances=distances,
            hasCapacityVariable=hasCapacityVariable,
            capacityVariableDomain=capacityVariableDomain,
            capacityPerPlantUnit=capacityPerPlantUnit,
            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable,
            bigM=bigM,
            operationRateMax=operationRateMax,
            operationRateFix=operationRateFix,
            tsaWeight=tsaWeight,
            locationalEligibility=locationalEligibility,
            capacityMin=capacityMin,
            capacityMax=capacityMax,
            partLoadMin=partLoadMin,
            sharedPotentialID=sharedPotentialID,
            capacityFix=capacityFix,
            isBuiltFix=isBuiltFix,
            investPerCapacity=investPerCapacity,
            investIfBuilt=investIfBuilt,
            opexPerOperation=opexPerOperation,
            opexPerCapacity=opexPerCapacity,
            opexIfBuilt=opexIfBuilt,
            QPcostScale=QPcostScale,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
            technicalLifetime=technicalLifetime,
        )

        self.modelingClass = LOPFModel

        self.reactances2dim = reactances

        try:
            self.reactances = pd.Series(self._mapC).apply(
                lambda loc: self.reactances2dim[loc[0]][loc[1]]
            )
        except:
            self.reactances = utils.preprocess2dimData(self.reactances2dim)

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a LinearOptimalPowerFlow component to the given energy system model.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)


class LOPFModel(TransmissionModel):

    """
    A LOPFModel class instance will be instantly created if a LinearOptimalPowerFlow class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the LinearOptimalPowerFlow
    class instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The LOPFModel class inherits from the TransmissionModel class.
    """

    def __init__(self):
        self.abbrvName = "lopf"
        self.dimension = "2dim"
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum, self.phaseAngleVariablesOptimum = None, None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def initPhaseAngleVarSet(self, pyM):
        """
        Declare phase angle variable set in the pyomo object for for each node.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        # Set for operation variables
        def initPhaseAngleVarSet(pyM):
            return (
                (loc, compName)
                for compName, comp in compDict.items()
                for loc in compDict[compName]._mapL.keys()
            )

        setattr(
            pyM,
            "phaseAngleVarSet_" + abbrvName,
            pyomo.Set(dimen=2, initialize=initPhaseAngleVarSet),
        )

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
        self.initPhaseAngleVarSet(pyM)
        self.declareOperationBinarySet(pyM)

        # Declare operation variable set
        self.declareOperationModeSets(
            pyM, "opConstrSet", "operationRateMax", "operationRateFix"
        )

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declarePhaseAngleVariables(self, pyM):
        """
        Declare phase angle variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        setattr(
            pyM,
            "phaseAngle_" + self.abbrvName,
            pyomo.Var(
                getattr(pyM, "phaseAngleVarSet_" + self.abbrvName),
                pyM.timeSet,
                domain=pyomo.Reals,
            ),
        )

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary):
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
        self.declareBinaryDesignDecisionVars(pyM, relaxIsBuiltBinary)
        # Flow over the edges of the components [commodityUnit]
        self.declareOperationVars(pyM, "op")
        # Operation of component as binary [1/0]
        self.declareOperationBinaryVars(pyM, "op_bin")
        # Operation of component [commodityUnit]
        self.declarePhaseAngleVariables(pyM)

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def powerFlowDC(self, pyM):
        """
        Ensure that the flow between two locations is equal to the difference between the phase angle variables at
        these locations divided by the reactance of the line between these locations.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        phaseAngleVar = getattr(pyM, "phaseAngle_" + self.abbrvName)
        opVar, opVarSet = getattr(pyM, "op_" + abbrvName), getattr(
            pyM, "operationVarSet_" + abbrvName
        )

        def powerFlowDC(pyM, loc, compName, p, t):
            node1, node2 = compDict[compName]._mapC[loc]
            return (
                opVar[loc, compName, p, t]
                - opVar[compDict[compName]._mapI[loc], compName, p, t]
                == (
                    phaseAngleVar[node1, compName, p, t]
                    - phaseAngleVar[node2, compName, p, t]
                )
                / compDict[compName].reactances[loc]
            )

        setattr(
            pyM,
            "ConstrpowerFlowDC_" + abbrvName,
            pyomo.Constraint(opVarSet, pyM.timeSet, rule=powerFlowDC),
        )

    def basePhaseAngle(self, pyM):
        """
        Declare the constraint that the reference phase angle is set to zero for all time steps.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        phaseAngleVar = getattr(pyM, "phaseAngle_" + self.abbrvName)

        def basePhaseAngle(pyM, compName, p, t):
            node0 = sorted(compDict[compName]._mapL)[0]
            return phaseAngleVar[node0, compName, p, t] == 0

        setattr(
            pyM,
            "ConstrBasePhaseAngle_" + abbrvName,
            pyomo.Constraint(compDict.keys(), pyM.timeSet, rule=basePhaseAngle),
        )

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
        #                                         Add DC power flow constraints                                        #
        ################################################################################################################

        self.powerFlowDC(pyM)
        self.basePhaseAngle(pyM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

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
        phaseAngleVar = getattr(pyM, "phaseAngle_" + abbrvName)

        optVal_ = utils.formatOptimizationOutput(
            phaseAngleVar.get_values(),
            "operationVariables",
            "1dim",
            esM.periodsOrder,
            esM=esM,
        )
        self.phaseAngleVariablesOptimum = optVal_

    def getOptimalValues(self, name="all"):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariables',
            * 'isBuiltVariables',
            * 'operationVariablesOptimum',
            * 'phaseAngleVariablesOptimum',
            * 'all' or another input: all variables are returned.

        :type name: string
        """
        if name == "capacityVariablesOptimum":
            return {
                "values": self.capacityVariablesOptimum,
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "isBuiltVariablesOptimum":
            return {
                "values": self.isBuiltVariablesOptimum,
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "operationVariablesOptimum":
            return {
                "values": self.operationVariablesOptimum,
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "phaseAngleVariablesOptimum":
            return {
                "values": self.phaseAngleVariablesOptimum,
                "timeDependent": True,
                "dimension": "1dim",
            }
        else:
            return {
                "capacityVariablesOptimum": {
                    "values": self.capacityVariablesOptimum,
                    "timeDependent": False,
                    "dimension": self.dimension,
                },
                "isBuiltVariablesOptimum": {
                    "values": self.isBuiltVariablesOptimum,
                    "timeDependent": False,
                    "dimension": self.dimension,
                },
                "operationVariablesOptimum": {
                    "values": self.operationVariablesOptimum,
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "phaseAngleVariablesOptimum": {
                    "values": self.phaseAngleVariablesOptimum,
                    "timeDependent": True,
                    "dimension": "1dim",
                },
            }
