from FINE.conversion import Conversion, ConversionModel
from FINE import utils
import pyomo.environ as pyomo
import pandas as pd


class ConversionPartLoad(Conversion):
    """
    A ConversionPartLoad component maps the (nonlinear) part-load behavior of a Conversion component.
    It uses the open source module PWLF to generate piecewise linear functions upon a continuous function or
    discrete data points.
    The formulation of the optimization is done by using special ordered sets (SOS) constraints.
    When using ConversionPartLoad it is recommended to check the piecewise linearization
    visually to verify that the accuracy meets the desired requirements.
    The ConversionPartLoad class inherits from the Conversion class.
    """

    def __init__(
        self,
        esM,
        name,
        physicalUnit,
        commodityConversionFactors,
        commodityConversionFactorsPartLoad,
        nSegments=None,
        **kwargs
    ):

        """
        Constructor for creating an ConversionPartLoad class instance. Capacities are given in the physical unit
        of the plants.
        The ConversionPartLoad component specific input arguments are described below.
        Other specific input arguments are described in the Conversion class
        and the general component input arguments are described in the Component class.

        **Required arguments:**

        :param commodityConversionFactorsPartLoad: Function or data set describing (nonlinear) part load behavior.
        :type commodityConversionFactorsPartLoad: Lambda function or Pandas DataFrame with two columns for the x-axis
            and the y-axis.

        **Default arguments:**

        :param nSegments: Number of line segments used for piecewise linearization and generation of point variable (nSegment+1) and
            segment (nSegment) variable sets.
            By default, the nSegments is None. For this case, the number of line segments is set to 5.
            The user can set nSegments by choosing an integer (>=0). It is recommended to choose values between 3 and 7 since
            the computational cost rises dramatically with increasing nSegments.
            When specifying nSegements='optimizeSegmentNumbers', an optimal number of line segments is automatically chosen by a
            bayesian optimization algorithm.
            |br| * the default value is None
        :type nSegments: None or integer or string

        :param **kwargs: All other keyword arguments of the conversion class can be defined as well.
        :type **kwargs:
            * Check Conversion Class documentation.
        """
        Conversion.__init__(
            self, esM, name, physicalUnit, commodityConversionFactors, **kwargs
        )

        self.modelingClass = ConversionPartLoadModel

        # TODO: Make compatible with conversion
        utils.checkNumberOfConversionFactors(commodityConversionFactors)

        if type(commodityConversionFactorsPartLoad) == dict:
            # TODO: Multiple conversionPartLoads
            utils.checkNumberOfConversionFactors(commodityConversionFactorsPartLoad)
            utils.checkCommodities(esM, set(commodityConversionFactorsPartLoad.keys()))
            utils.checkCommodityConversionFactorsPartLoad(
                commodityConversionFactorsPartLoad.values()
            )
            self.commodityConversionFactorsPartLoad = commodityConversionFactorsPartLoad
            self.discretizedPartLoad, self.nSegments = utils.getDiscretizedPartLoad(
                commodityConversionFactorsPartLoad, nSegments
            )

        elif type(commodityConversionFactorsPartLoad) == tuple:
            utils.checkNumberOfConversionFactors(
                commodityConversionFactorsPartLoad[0].keys()
            )
            self.discretizedPartLoad = commodityConversionFactorsPartLoad[0]
            self.nSegments = commodityConversionFactorsPartLoad[1]

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a ConversionPartLoad component to the given energy system model.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)


class ConversionPartLoadModel(ConversionModel):

    """
    A ConversionPartLoad class instance will be instantly created if a ConversionPartLoad class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Conversion class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The ConversionPartLoad class inherits from the ConversionModel class.
    """

    def __init__(self):
        self.abbrvName = "partLoad"
        self.dimension = "1dim"
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
            return (
                (loc, compName, discreteStep)
                for compName, comp in compDict.items()
                for loc in compDict[compName].locationalEligibility.index
                if compDict[compName].locationalEligibility[loc] == 1
                for discreteStep in range(compDict[compName].nSegments + 1)
            )

        setattr(
            pyM,
            "discretizationPointVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=initDiscretizationPointVarSet),
        )

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
            return (
                (loc, compName, discreteStep)
                for compName, comp in compDict.items()
                for loc in compDict[compName].locationalEligibility.index
                if compDict[compName].locationalEligibility[loc] == 1
                for discreteStep in range(compDict[compName].nSegments)
            )

        setattr(
            pyM,
            "discretizationSegmentVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=initDiscretizationSegmentVarSet),
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

        super().declareSets(esM, pyM)

        # Declare operation variable sets
        self.initDiscretizationPointVarSet(pyM)
        self.initDiscretizationSegmentVarSet(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareDiscretizationPointVariables(self, pyM):
        """
        Declare discretization point variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        setattr(
            pyM,
            "discretizationPoint_" + self.abbrvName,
            pyomo.Var(
                getattr(pyM, "discretizationPointVarSet_" + self.abbrvName),
                pyM.timeSet,
                domain=pyomo.NonNegativeReals,
            ),
        )

    def declareDiscretizationSegmentBinVariables(self, pyM):
        """
        Declare discretization segment variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        setattr(
            pyM,
            "discretizationSegmentBin_" + self.abbrvName,
            pyomo.Var(
                getattr(pyM, "discretizationSegmentVarSet_" + self.abbrvName),
                pyM.timeSet,
                domain=pyomo.Binary,
            ),
        )

    def declareDiscretizationSegmentConVariables(self, pyM):
        """
        Declare discretization segment variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """
        setattr(
            pyM,
            "discretizationSegmentCon_" + self.abbrvName,
            pyomo.Var(
                getattr(pyM, "discretizationSegmentVarSet_" + self.abbrvName),
                pyM.timeSet,
                domain=pyomo.NonNegativeReals,
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
        super().declareVariables(esM, pyM, relaxIsBuiltBinary)

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
        discretizationSegmentBinVar = getattr(
            pyM, "discretizationSegmentBin_" + self.abbrvName
        )
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def segmentSOS1(pyM, loc, compName, p, t):
            return (
                sum(
                    discretizationSegmentBinVar[loc, compName, discretStep, p, t]
                    for discretStep in range(compDict[compName].nSegments)
                )
                == 1
            )

        setattr(
            pyM,
            "ConstrSegmentSOS1_" + abbrvName,
            pyomo.Constraint(opVarSet, pyM.timeSet, rule=segmentSOS1),
        )

    def segmentBigM(self, pyM):
        """
        Ensure that the continuous segment variables are zero if the respective binary variable is zero and unlimited otherwise.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationSegmentConVar = getattr(
            pyM, "discretizationSegmentCon_" + self.abbrvName
        )
        discretizationSegmentBinVar = getattr(
            pyM, "discretizationSegmentBin_" + self.abbrvName
        )
        discretizationSegmentVarSet = getattr(
            pyM, "discretizationSegmentVarSet_" + self.abbrvName
        )

        def segmentBigM(pyM, loc, compName, discretStep, p, t):
            return (
                discretizationSegmentConVar[loc, compName, discretStep, p, t]
                <= discretizationSegmentBinVar[loc, compName, discretStep, p, t]
                * compDict[compName].bigM
            )

        setattr(
            pyM,
            "ConstrSegmentBigM_" + abbrvName,
            pyomo.Constraint(
                discretizationSegmentVarSet, pyM.timeSet, rule=segmentBigM
            ),
        )

    def segmentCapacityConstraint(self, pyM, esM):
        """
        Ensure that the continuous segment variables are in sum equal to the installed capacity of the component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationSegmentConVar = getattr(
            pyM, "discretizationSegmentCon_" + self.abbrvName
        )
        capVar = getattr(pyM, "cap_" + abbrvName)
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)

        if not pyM.hasSegmentation:

            def segmentCapacityConstraint(pyM, loc, compName, p, t):
                return (
                    sum(
                        discretizationSegmentConVar[loc, compName, discretStep, p, t]
                        for discretStep in range(compDict[compName].nSegments)
                    )
                    == esM.hoursPerTimeStep * capVar[loc, compName]
                )

            setattr(
                pyM,
                "ConstrSegmentCapacity_" + abbrvName,
                pyomo.Constraint(opVarSet, pyM.timeSet, rule=segmentCapacityConstraint),
            )
        else:

            def segmentCapacityConstraint(pyM, loc, compName, p, t):
                return (
                    sum(
                        discretizationSegmentConVar[loc, compName, discretStep, p, t]
                        for discretStep in range(compDict[compName].nSegments)
                    )
                    == esM.hoursPerSegment.to_dict()[p, t] * capVar[loc, compName]
                )

            setattr(
                pyM,
                "ConstrSegmentCapacity_" + abbrvName,
                pyomo.Constraint(opVarSet, pyM.timeSet, rule=segmentCapacityConstraint),
            )

    def pointCapacityConstraint(self, pyM, esM):
        """
        Ensure that the continuous point variables are in sum equal to the installed capacity of the component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(
            pyM, "discretizationPoint_" + self.abbrvName
        )
        capVar = getattr(pyM, "cap_" + abbrvName)
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)

        if not pyM.hasSegmentation:

            def pointCapacityConstraint(pyM, loc, compName, p, t):
                nPoints = compDict[compName].nSegments + 1
                return (
                    sum(
                        discretizationPointConVar[loc, compName, discretStep, p, t]
                        for discretStep in range(nPoints)
                    )
                    == esM.hoursPerTimeStep * capVar[loc, compName]
                )

            setattr(
                pyM,
                "ConstrPointCapacity_" + abbrvName,
                pyomo.Constraint(opVarSet, pyM.timeSet, rule=pointCapacityConstraint),
            )
        else:

            def pointCapacityConstraint(pyM, loc, compName, p, t):
                nPoints = compDict[compName].nSegments + 1
                return (
                    sum(
                        discretizationPointConVar[loc, compName, discretStep, p, t]
                        for discretStep in range(nPoints)
                    )
                    == esM.hoursPerSegment.to_dict()[p, t] * capVar[loc, compName]
                )

            setattr(
                pyM,
                "ConstrPointCapacity_" + abbrvName,
                pyomo.Constraint(opVarSet, pyM.timeSet, rule=pointCapacityConstraint),
            )

    def declareOpConstrSetMinPartLoad(self, pyM, constrSetName):
        """
        Declare set of locations and components for which partLoadMin is not None.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSetBin_" + abbrvName)

        def declareOpConstrSetMinPartLoad(pyM):
            return (
                (loc, compName)
                for loc, compName in varSet
                if getattr(compDict[compName], "partLoadMin") is not None
            )

        setattr(
            pyM,
            constrSetName + "partLoadMin_" + abbrvName,
            pyomo.Set(dimen=2, initialize=declareOpConstrSetMinPartLoad),
        )

    def pointSOS2(self, pyM):
        """
        Ensure that only two consecutive point variables are non-zero while all other point variables are fixed to zero.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(
            pyM, "discretizationPoint_" + self.abbrvName
        )
        discretizationSegmentConVar = getattr(
            pyM, "discretizationSegmentCon_" + self.abbrvName
        )
        discretizationPointVarSet = getattr(
            pyM, "discretizationPointVarSet_" + self.abbrvName
        )

        def pointSOS2(pyM, loc, compName, discretStep, p, t):
            points = list(range(compDict[compName].nSegments + 1))
            segments = list(range(compDict[compName].nSegments))

            if discretStep == points[0]:
                return (
                    discretizationPointConVar[loc, compName, points[0], p, t]
                    <= discretizationSegmentConVar[loc, compName, segments[0], p, t]
                )
            elif discretStep == points[-1]:
                return (
                    discretizationPointConVar[loc, compName, points[-1], p, t]
                    <= discretizationSegmentConVar[loc, compName, segments[-1], p, t]
                )
            else:
                return (
                    discretizationPointConVar[loc, compName, discretStep, p, t]
                    <= discretizationSegmentConVar[loc, compName, discretStep - 1, p, t]
                    + discretizationSegmentConVar[loc, compName, discretStep, p, t]
                )

        setattr(
            pyM,
            "ConstrPointSOS2_" + abbrvName,
            pyomo.Constraint(discretizationPointVarSet, pyM.timeSet, rule=pointSOS2),
        )

    def partLoadOperationOutput(self, pyM):
        """
        Set the required input of a conversion process dependent on the part load efficency.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo Concrete Model
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        discretizationPointConVar = getattr(
            pyM, "discretizationPoint_" + self.abbrvName
        )
        opVar, opVarSet = getattr(pyM, "op_" + abbrvName), getattr(
            pyM, "operationVarSet_" + abbrvName
        )

        def partLoadOperationOutput(pyM, loc, compName, p, t):
            nPoints = compDict[compName].nSegments + 1

            return opVar[loc, compName, p, t] == sum(
                discretizationPointConVar[loc, compName, discretStep, p, t]
                * compDict[compName].discretizedPartLoad[
                    list(compDict[compName].discretizedPartLoad.keys())[0]
                ]["xSegments"][discretStep]
                for discretStep in range(nPoints)
            )

        setattr(
            pyM,
            "ConstrpartLoadOperationOutput_" + abbrvName,
            pyomo.Constraint(opVarSet, pyM.timeSet, rule=partLoadOperationOutput),
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
        """Get contributions to shared location potential."""
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
        """Get contribution to a commodity balance."""
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVarDict = getattr(pyM, "operationVarDict_" + abbrvName)
        discretizationPointConVar = getattr(
            pyM, "discretizationPoint_" + self.abbrvName
        )

        return sum(
            sum(
                discretizationPointConVar[loc, compName, discretStep, p, t]
                * compDict[compName].discretizedPartLoad[commod]["xSegments"][
                    discretStep
                ]
                * compDict[compName].discretizedPartLoad[commod]["ySegments"][
                    discretStep
                ]
                for discretStep in range(compDict[compName].nSegments + 1)
            )
            for compName in opVarDict[loc]
            if commod in compDict[compName].discretizedPartLoad
        )

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

        abbrvName = self.abbrvName
        discretizationPointVariables = getattr(pyM, "discretizationPoint_" + abbrvName)
        discretizationSegmentConVariables = getattr(
            pyM, "discretizationSegmentCon_" + abbrvName
        )
        discretizationSegmentBinVariables = getattr(
            pyM, "discretizationSegmentBin_" + abbrvName
        )

        discretizationPointVariablesOptVal_ = utils.formatOptimizationOutput(
            discretizationPointVariables.get_values(),
            "operationVariables",
            "1dim",
            esM.periodsOrder,
            esM=esM,
        )
        discretizationSegmentConVariablesOptVal_ = utils.formatOptimizationOutput(
            discretizationSegmentConVariables.get_values(),
            "operationVariables",
            "1dim",
            esM.periodsOrder,
            esM=esM,
        )
        discretizationSegmentBinVariablesOptVal_ = utils.formatOptimizationOutput(
            discretizationSegmentBinVariables.get_values(),
            "operationVariables",
            "1dim",
            esM.periodsOrder,
            esM=esM,
        )

        self.discretizationPointVariablesOptimun = discretizationPointVariablesOptVal_
        self.discretizationSegmentConVariablesOptimun = (
            discretizationSegmentConVariablesOptVal_
        )
        self.discretizationSegmentBinVariablesOptimun = (
            discretizationSegmentBinVariablesOptVal_
        )

    def getOptimalValues(self, name="all"):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariables',
            * 'isBuiltVariables',
            * 'operationVariablesOptimum',
            * 'all' or another input: all variables are returned.

        |br| * the default value is 'all'
        :type name: string

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        # return super().getOptimalValues(name)
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
        elif name == "discretizationPointVariablesOptimun":
            return {
                "values": self.discretizationPointVariablesOptimun,
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "discretizationSegmentConVariablesOptimun":
            return {
                "values": self.discretizationSegmentConVariablesOptimun,
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "discretizationSegmentBinVariablesOptimun":
            return {
                "values": self.discretizationSegmentBinVariablesOptimun,
                "timeDependent": True,
                "dimension": self.dimension,
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
                "discretizationPointVariablesOptimun": {
                    "values": self.discretizationPointVariablesOptimun,
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "discretizationSegmentConVariablesOptimun": {
                    "values": self.discretizationSegmentConVariablesOptimun,
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "discretizationSegmentBinVariablesOptimun": {
                    "values": self.discretizationSegmentBinVariablesOptimun,
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
            }
