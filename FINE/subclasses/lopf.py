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
        stockCommissioning=None,
        floorTechnicalLifetime=True,
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
            floorTechnicalLifetime=floorTechnicalLifetime,
            stockCommissioning=stockCommissioning,
        )

        self.modelingClass = LOPFModel

        self.reactances2dim = reactances

        try:
            self.reactances = pd.Series(self._mapC).apply(
                lambda loc: self.reactances2dim[loc[0]][loc[1]]
            )
        except:
            self.reactances = utils.preprocess2dimData(self.reactances2dim)


class LOPFModel(TransmissionModel):

    """
    A LOPFModel class instance will be instantly created if a LinearOptimalPowerFlow class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the LinearOptimalPowerFlow
    class instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The LOPFModel class inherits from the TransmissionModel class.
    """

    def __init__(self):
        super().__init__()
        self.abbrvName = "lopf"
        self.dimension = "2dim"
        self._operationVariablesOptimum = {}
        self._phaseAngleVariablesOptimum = {}

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
        self.declareDesignVarSet(pyM, esM)
        self.declareCommissioningVarSet(pyM, esM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        # Declare design pathway sets
        self.declarePathwaySets(pyM, esM)
        self.declareLocationComponentSet(pyM)

        # Declare operation variable sets
        self.declareOpVarSet(esM, pyM)
        self.initPhaseAngleVarSet(pyM)

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
        compDict = self.componentsDict

        def phaseAngleBounds(pyM, compName, loc, ip, p, t):
            if compName in compDict.keys():
                # reference phase angle is set to zero for all time steps (bounded by (0,0))
                node0 = sorted(compDict[compName]._mapL)[0]
                if loc == node0:
                    return (0, 0)
                else:
                    return (None, None)
            else:
                return (None, None)

        setattr(
            pyM,
            "phaseAngle_" + self.abbrvName,
            pyomo.Var(
                getattr(pyM, "phaseAngleVarSet_" + self.abbrvName),
                pyM.timeSet,
                domain=pyomo.Reals,
                bounds=phaseAngleBounds,
            ),
        )

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary, relevanceThreshold):
        """
        Declare design and operation variables.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model

        :param relaxIsBuiltBinary: states if the optimization problem should be solved as a relaxed LP to get the lower
            bound of the problem.
            |br| * the default value is False
        :type declaresOptimizationProblem: boolean

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None
        """

        # Call the declareVariables function of transmission model class
        super().declareVariables(esM, pyM, relaxIsBuiltBinary, relevanceThreshold)

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
        opVar, opVarSet = (
            getattr(pyM, "op_" + abbrvName),
            getattr(pyM, "operationVarSet_" + abbrvName),
        )

        def powerFlowDC(pyM, loc, compName, ip, p, t):
            node1, node2 = compDict[compName]._mapC[loc]
            return (
                opVar[loc, compName, ip, p, t]
                - opVar[compDict[compName]._mapI[loc], compName, ip, p, t]
                == (
                    phaseAngleVar[node1, compName, ip, p, t]
                    - phaseAngleVar[node2, compName, ip, p, t]
                )
                / compDict[compName].reactances[loc]
            )

        setattr(
            pyM,
            "ConstrpowerFlowDC_" + abbrvName,
            pyomo.Constraint(opVarSet, pyM.intraYearTimeSet, rule=powerFlowDC),
        )

    def basePhaseAngle(self, pyM):
        # TODO Check if this function is still required due to new added bounds
        """
        Declare the constraint that the reference phase angle is set to zero for all time steps.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        phaseAngleVar = getattr(pyM, "phaseAngle_" + self.abbrvName)

        def basePhaseAngle(pyM, compName, ip, p, t):
            node0 = sorted(compDict[compName]._mapL)[0]
            return phaseAngleVar[node0, compName, ip, p, t] == 0

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
        for ip in esM.investmentPeriods:
            compDict, abbrvName = self.componentsDict, self.abbrvName
            phaseAngleVar = getattr(pyM, "phaseAngle_" + abbrvName)

            optVal_ = utils.formatOptimizationOutput(
                phaseAngleVar.get_values(),
                "operationVariables",
                "1dim",
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )
            self._phaseAngleVariablesOptimum[esM.investmentPeriodNames[ip]] = optVal_

    def getOptimalValues(self, name="all", ip=0):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariables',
            * 'isBuiltVariables',
            * '_operationVariablesOptimum',
            * 'phaseAngleVariablesOptimum',
            * 'all' or another input: all variables are returned.

        :type name: string
        """
        if name == "capacityVariablesOptimum":
            return {
                "values": self._capacityVariablesOptimum[ip],
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "isBuiltVariablesOptimum":
            return {
                "values": self._isBuiltVariablesOptimum[ip],
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "operationVariablesOptimum":
            return {
                "values": self._operationVariablesOptimum[ip],
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "phaseAngleVariablesOptimum":
            return {
                "values": self._phaseAngleVariablesOptimum[ip],
                "timeDependent": True,
                "dimension": "1dim",
            }
        else:
            return {
                "capacityVariablesOptimum": {
                    "values": self._capacityVariablesOptimum[ip],
                    "timeDependent": False,
                    "dimension": self.dimension,
                },
                "isBuiltVariablesOptimum": {
                    "values": self._isBuiltVariablesOptimum[ip],
                    "timeDependent": False,
                    "dimension": self.dimension,
                },
                "operationVariablesOptimum": {
                    "values": self._operationVariablesOptimum[ip],
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "phaseAngleVariablesOptimum": {
                    "values": self._phaseAngleVariablesOptimum[ip],
                    "timeDependent": True,
                    "dimension": "1dim",
                },
            }
