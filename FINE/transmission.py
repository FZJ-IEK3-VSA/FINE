from FINE.component import Component, ComponentModel
from FINE import utils
import warnings
import pyomo.environ as pyomo
import pandas as pd


class Transmission(Component):
    """
    A Transmission component can transmit a commodity between locations of the energy system.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
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
        linkedQuantityID=None,
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
        balanceLimitID=None,
    ):
        """
        Constructor for creating an Transmission class instance.
        The Transmission component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param commodity: to the component related commodity.
        :type commodity: string

        **Default arguments:**

        :param losses: relative losses per lengthUnit (lengthUnit as specified in the energy system model) in
            percentage of the commodity flow. This loss factor can capture simple linear losses

            .. math::
                trans_{in, ij} = (1 - \\text{losses} \\cdot \\text{distances}) \\cdot trans_{out, ij}

            (with trans being the commodity flow at a certain point in
            time and i and j being locations in the energy system). The losses can either be given as a float or a
            Pandas DataFrame with location specific values.
            |br| * the default value is 0
        :type losses: positive float (0 <= float <= 1) or Pandas DataFrame with positive values
            (0 <= float <= 1). The row and column indices of the DataFrame have to equal the in the energy
            system model specified locations.

        :param distances: distances between locations given in the lengthUnit (lengthUnit as specified in
            the energy system model).
            |br| * the default value is None
        :type distances: positive float (>= 0) or Pandas DataFrame with positive values (>= 0). The row and
            column indices of the DataFrame have to equal the in the energy system model specified locations.

        :param operationRateMax: if specified, indicates a maximum operation rate for all possible connections
            (both directions) of the transmission component at each time step by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices are combinations
            of locations (as defined in the energy system model), separated by a underscore (e.g.
            "location1_location2"). The first location indicates where the commodity is coming from. The second
            location indicates where the commodity is going too. If a flow is specified from location i to
            location j, it also has to be specified from j to i.

        :param operationRateFix: if specified, indicates a fixed operation rate for all possible connections
            (both directions) of the transmission component at each time step by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices are combinations
            of locations (as defined in the energy system model), separated by a underscore (e.g.
            "location1_location2"). The first location indicates where the commodity is coming from. The second
            one location indicates where the commodity is going too. If a flow is specified from location i to
            location j, it also has to be specified from j to i.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param opexPerOperation: describes the cost for one unit of the operation.
            The cost which is directly proportional to the operation of the component is obtained by multiplying
            the opexPerOperation parameter with the annual sum of the operational time series of the components.
            The opexPerOperation can either be given as a float or a Pandas DataFrame with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro). The value has to match the unit costUnit/operationUnit
            (e.g. Euro/kWh, Dollar/kWh).
            |br| * the default value is 0
        :type opexPerOperation: positive (>=0) float or Pandas DataFrame with positive (>=0) values.
            The row and column indices of the DataFrame have to equal the in the energy system model
            specified locations.

        :param balanceLimitID: ID for the respective balance limit (out of the balance limits introduced in the esM).
            Should be specified if the respective component of the TransmissionModel is supposed to be included in
            the balance analysis. If the commodity is transported out of the region, it is counted as a negative, if
            it is imported into the region it is considered positive.
            |br| * the default value is None
        :type balanceLimitID: string
        """
        # TODO add unit checks
        # Preprocess two-dimensional data
        self.locationalEligibility = utils.preprocess2dimData(locationalEligibility)
        self.capacityMax = utils.preprocess2dimData(
            capacityMax, locationalEligibility=locationalEligibility
        )
        self.capacityFix = utils.preprocess2dimData(
            capacityFix, locationalEligibility=locationalEligibility
        )
        self.isBuiltFix = utils.preprocess2dimData(
            isBuiltFix, locationalEligibility=locationalEligibility
        )

        # Set locational eligibility
        operationTimeSeries = (
            operationRateFix if operationRateFix is not None else operationRateMax
        )
        self.locationalEligibility = utils.setLocationalEligibility(
            esM,
            self.locationalEligibility,
            self.capacityMax,
            self.capacityFix,
            self.isBuiltFix,
            hasCapacityVariable,
            operationTimeSeries,
            "2dim",
        )

        self._mapC, self._mapL, self._mapI = {}, {}, {}
        for loc1 in esM.locations:
            for loc2 in esM.locations:
                if loc1 + "_" + loc2 in self.locationalEligibility.index:
                    if self.locationalEligibility[loc1 + "_" + loc2] == 0:
                        self.locationalEligibility.drop(
                            labels=loc1 + "_" + loc2, inplace=True
                        )
                    self._mapC.update({loc1 + "_" + loc2: (loc1, loc2)})
                    self._mapL.setdefault(loc1, {}).update({loc2: loc1 + "_" + loc2})
                    self._mapI.update({loc1 + "_" + loc2: loc2 + "_" + loc1})

        self.capacityMax = utils.preprocess2dimData(
            capacityMax, self._mapC, locationalEligibility=self.locationalEligibility
        )
        self.capacityFix = utils.preprocess2dimData(
            capacityFix, self._mapC, locationalEligibility=self.locationalEligibility
        )
        self.capacityMin = utils.preprocess2dimData(
            capacityMin, self._mapC, locationalEligibility=self.locationalEligibility
        )
        self.investPerCapacity = utils.preprocess2dimData(investPerCapacity, self._mapC)
        self.investIfBuilt = utils.preprocess2dimData(investIfBuilt, self._mapC)
        self.isBuiltFix = utils.preprocess2dimData(
            isBuiltFix, self._mapC, locationalEligibility=self.locationalEligibility
        )
        self.opexPerCapacity = utils.preprocess2dimData(opexPerCapacity, self._mapC)
        self.opexIfBuilt = utils.preprocess2dimData(opexIfBuilt, self._mapC)
        self.interestRate = utils.preprocess2dimData(interestRate, self._mapC)
        self.economicLifetime = utils.preprocess2dimData(economicLifetime, self._mapC)
        self.technicalLifetime = utils.preprocess2dimData(technicalLifetime, self._mapC)
        self.balanceLimitID = balanceLimitID

        Component.__init__(
            self,
            esM,
            name,
            dimension="2dim",
            hasCapacityVariable=hasCapacityVariable,
            capacityVariableDomain=capacityVariableDomain,
            capacityPerPlantUnit=capacityPerPlantUnit,
            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable,
            bigM=bigM,
            locationalEligibility=self.locationalEligibility,
            capacityMin=self.capacityMin,
            capacityMax=self.capacityMax,
            partLoadMin=partLoadMin,
            sharedPotentialID=sharedPotentialID,
            linkedQuantityID=linkedQuantityID,
            capacityFix=self.capacityFix,
            isBuiltFix=self.isBuiltFix,
            investPerCapacity=self.investPerCapacity,
            investIfBuilt=self.investIfBuilt,
            opexPerCapacity=self.opexPerCapacity,
            opexIfBuilt=self.opexIfBuilt,
            interestRate=self.interestRate,
            QPcostScale=QPcostScale,
            economicLifetime=self.economicLifetime,
            technicalLifetime=self.technicalLifetime,
        )

        # Set general component data
        utils.checkCommodities(esM, {commodity})
        self.commodity, self.commodityUnit = (
            commodity,
            esM.commodityUnitsDict[commodity],
        )
        self.distances = utils.preprocess2dimData(
            distances, self._mapC, locationalEligibility=self.locationalEligibility
        )
        self.losses = utils.preprocess2dimData(losses, self._mapC)
        self.distances = utils.checkAndSetDistances(
            self.distances, self.locationalEligibility, esM
        )
        self.losses = utils.checkAndSetTransmissionLosses(
            self.losses, self.distances, self.locationalEligibility
        )
        self.modelingClass = TransmissionModel

        # Set distance related costs data
        self.processedInvestPerCapacity = self.investPerCapacity * self.distances * 0.5
        self.processedInvestIfBuilt = self.investIfBuilt * self.distances * 0.5
        self.processedOpexPerCapacity = self.opexPerCapacity * self.distances * 0.5
        self.processedOpexIfBuilt = self.opexIfBuilt * self.distances * 0.5

        # Set additional economic data
        self.opexPerOperation = utils.preprocess2dimData(opexPerOperation, self._mapC)
        self.opexPerOperation = utils.checkAndSetCostParameter(
            esM, name, self.opexPerOperation, "2dim", self.locationalEligibility
        )

        self.operationRateMax = operationRateMax
        self.operationRateFix = operationRateFix

        self.fullOperationRateMax = utils.checkAndSetTimeSeries(
            esM, name, operationRateMax, self.locationalEligibility, self.dimension
        )
        self.aggregatedOperationRateMax, self.processedOperationRateMax = None, None

        self.fullOperationRateFix = utils.checkAndSetTimeSeries(
            esM, name, operationRateFix, self.locationalEligibility, self.dimension
        )
        self.aggregatedOperationRateFix, self.processedOperationRateFix = None, None

        # Set location-specific operation parameters
        if (
            self.fullOperationRateMax is not None
            and self.fullOperationRateFix is not None
        ):
            self.fullOperationRateMax = None
            if esM.verbose < 2:
                warnings.warn(
                    "If operationRateFix is specified, the operationRateMax parameter is not required.\n"
                    + "The operationRateMax time series was set to None."
                )

        if self.partLoadMin is not None:
            if self.fullOperationRateMax is not None:
                if (
                    (
                        (self.fullOperationRateMax > 0)
                        & (self.fullOperationRateMax < self.partLoadMin)
                    )
                    .any()
                    .any()
                ):
                    raise ValueError(
                        '"fullOperationRateMax" needs to be higher than "partLoadMin" or 0 for component '
                        + name
                    )
            if self.fullOperationRateFix is not None:
                if (
                    (
                        (self.fullOperationRateFix > 0)
                        & (self.fullOperationRateFix < self.partLoadMin)
                    )
                    .any()
                    .any()
                ):
                    raise ValueError(
                        '"fullOperationRateFix" needs to be higher than "partLoadMin" or 0 for component '
                        + name
                    )

        utils.isPositiveNumber(tsaWeight)
        self.tsaWeight = tsaWeight

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a transmission component to the given energy system model.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)

    def setTimeSeriesData(self, hasTSA):
        """
        Function for setting the maximum operation rate and fixed operation rate depending on whether a time series
        analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        self.processedOperationRateMax = (
            self.aggregatedOperationRateMax if hasTSA else self.fullOperationRateMax
        )
        self.processedOperationRateFix = (
            self.aggregatedOperationRateFix if hasTSA else self.fullOperationRateFix
        )

    def getDataForTimeSeriesAggregation(self):
        """Function for getting the required data if a time series aggregation is requested."""
        weightDict, data = {}, []
        weightDict, data = self.prepareTSAInput(
            self.fullOperationRateFix,
            self.fullOperationRateMax,
            "_operationRate_",
            self.tsaWeight,
            weightDict,
            data,
        )
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):
        """
        Function for determining the aggregated maximum rate and the aggregated fixed operation rate.

        :param data: Pandas DataFrame with the clustered time series data of the conversion component
        :type data: Pandas DataFrame
        """
        self.aggregatedOperationRateFix = self.getTSAOutput(
            self.fullOperationRateFix, "_operationRate_", data
        )
        self.aggregatedOperationRateMax = self.getTSAOutput(
            self.fullOperationRateMax, "_operationRate_", data
        )


class TransmissionModel(ComponentModel):
    """
    A TransmissionModel class instance will be instantly created if a Transmission class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Transmission class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The TransmissionModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        """ " Constructor for creating a TransmissionModel class instance"""
        self.abbrvName = "trans"
        self.dimension = "2dim"
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """
        Declare sets: design variable sets, operation variable set and operation mode sets.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        # # Declare design variable sets
        self.declareDesignVarSet(pyM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        # Declare operation variable set
        self.declareOpVarSet(esM, pyM)
        self.declareOperationBinarySet(pyM)

        # Declare operation mode sets
        self.declareOperationModeSets(
            pyM, "opConstrSet", "operationRateMax", "operationRateFix"
        )

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary):
        """
        Declare design and operation variables

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        # Capacity variables [commodityUnit]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not
        self.declareBinaryDesignDecisionVars(pyM, relaxIsBuiltBinary)
        # Operation of component [commodityUnit]
        self.declareOperationVars(pyM, "op")
        # Operation of component as binary [1/0]
        self.declareOperationBinaryVars(pyM, "op_bin")

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def symmetricalCapacity(self, pyM):
        """
        Ensure that the capacity between location_1 and location_2 is the same as the one
        between location_2 and location_1.

        .. math::

            cap^{comp}_{(loc_1,loc_2)} = cap^{comp}_{(loc_2,loc_1)}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, capVarSet = getattr(pyM, "cap_" + abbrvName), getattr(
            pyM, "designDimensionVarSet_" + abbrvName
        )

        def symmetricalCapacity(pyM, loc, compName):
            return (
                capVar[loc, compName] == capVar[compDict[compName]._mapI[loc], compName]
            )

        setattr(
            pyM,
            "ConstrSymmetricalCapacity_" + abbrvName,
            pyomo.Constraint(capVarSet, rule=symmetricalCapacity),
        )

    def operationMode1_2dim(self, pyM, esM, constrName, constrSetName, opVarName):
        """
        Declare the constraint that the operation [commodityUnit*hour] is limited by the installed
        capacity [commodityUnit] multiplied by the hours per time step.
        Since the flow should either go in one direction or the other, the limitation can be enforced on the sum
        of the forward and backward flow over the line. This leads to one of the flow variables being set to zero
        if a basic solution is obtained during optimization.

        .. math::

            op^{comp,op}_{(loc_1,loc_2),p,t} + op^{op}_{(loc_2,loc_1),p,t} \leq \\tau^{hours} \cdot \\text{cap}^{comp}_{(loc_{in},loc_{out})}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = getattr(pyM, opVarName + "_" + abbrvName), getattr(
            pyM, "cap_" + abbrvName
        )
        constrSet1 = getattr(pyM, constrSetName + "1_" + abbrvName)

        if not pyM.hasSegmentation:

            def op1(pyM, loc, compName, p, t):
                return (
                    opVar[loc, compName, p, t]
                    + opVar[compDict[compName]._mapI[loc], compName, p, t]
                    <= capVar[loc, compName] * esM.hoursPerTimeStep
                )

            setattr(
                pyM,
                constrName + "_" + abbrvName,
                pyomo.Constraint(constrSet1, pyM.timeSet, rule=op1),
            )
        else:

            def op1(pyM, loc, compName, p, t):
                return (
                    opVar[loc, compName, p, t]
                    + opVar[compDict[compName]._mapI[loc], compName, p, t]
                    <= capVar[loc, compName] * esM.hoursPerSegment.to_dict()[p, t]
                )

            setattr(
                pyM,
                constrName + "_" + abbrvName,
                pyomo.Constraint(constrSet1, pyM.timeSet, rule=op1),
            )

    def declareComponentConstraints(self, esM, pyM):
        """
        Declare time independent and dependent constraints

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        ################################################################################################################
        #                                    Declare time independent constraints                                      #
        ################################################################################################################

        # Determine the components' capacities from the number of installed units
        self.capToNbReal(pyM)
        # Determine the components' capacities from the number of installed units
        self.capToNbInt(pyM)
        # Enforce the consideration of the binary design variables of a component
        self.bigM(pyM)
        # Enforce the consideration of minimum capacities for components with design decision variables
        self.capacityMinDec(pyM)
        # Set, if applicable, the installed capacities of a component
        self.capacityFix(pyM)
        # Set, if applicable, the binary design variables of a component
        self.designBinFix(pyM)
        # Enforce the equality of the capacities cap_loc1_loc2 and cap_loc2_loc1
        self.symmetricalCapacity(pyM)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [commodityUnit*h] is limited by the installed capacity [commodityUnit] multiplied by the hours per
        # time step [h]
        self.operationMode1_2dim(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] is equal to the installed capacity [commodityUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode2(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] is limited by the installed capacity [commodityUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode3(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] is equal to the operation time series [commodityUnit*h]
        self.operationMode4(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] is limited by the operation time series [commodityUnit*h]
        self.operationMode5(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [physicalUnit*h] is limited by minimum part Load
        self.additionalMinPartLoad(
            pyM, esM, "ConstrOperation", "opConstrSet", "op", "op_bin", "cap"
        )

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

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: Name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """

        return any(
            [
                comp.commodity == commod
                and (
                    loc + "_" + loc_ in comp.locationalEligibility.index
                    or loc_ + "_" + loc in comp.locationalEligibility.index
                )
                for comp in self.componentsDict.values()
                for loc_ in esM.locations
            ]
        )

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """ 
        Get contribution to a commodity balance. 
        
        .. math::
            :nowrap:

            \\begin{eqnarray*}
            \\text{C}^{comp,comm}_{loc,p,t} = & & \\underset{\substack{(loc_{in},loc_{out}) \in \\ \mathcal{L}^{tans}: loc_{in}=loc}}{ \sum } \left(1-\eta_{(loc_{in},loc_{out})} \cdot I_{(loc_{in},loc_{out})} \\right) \cdot op^{comp,op}_{(loc_{in},loc_{out}),p,t} \\\\
            & - & \\underset{\substack{(loc_{in},loc_{out}) \in \\ \mathcal{L}^{tans}:loc_{out}=loc}}{ \sum } op^{comp,op}_{(loc_{in},loc_{out}),p,t}
            \\end{eqnarray*}
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDictIn = getattr(pyM, "op_" + abbrvName), getattr(
            pyM, "operationVarDictIn_" + abbrvName
        )
        opVarDictOut = getattr(pyM, "operationVarDictOut_" + abbrvName)
        return sum(
            opVar[loc_ + "_" + loc, compName, p, t]
            * (
                1
                - compDict[compName].losses[loc_ + "_" + loc]
                * compDict[compName].distances[loc_ + "_" + loc]
            )
            for loc_ in opVarDictIn[loc].keys()
            for compName in opVarDictIn[loc][loc_]
            if commod in compDict[compName].commodity
        ) - sum(
            opVar[loc + "_" + loc_, compName, p, t]
            for loc_ in opVarDictOut[loc].keys()
            for compName in opVarDictOut[loc][loc_]
            if commod in compDict[compName].commodity
        )

    def getBalanceLimitContribution(self, esM, pyM, ID, loc, timeSeriesAggregation):
        """
        Get contribution to balanceLimitConstraint (Further read in EnergySystemModel).
        Sum of the operation time series of a Transmission component is used as the balanceLimit contribution:

        - If commodity is transferred out of region a negative sign is used.
        - If commodity is transferred into region a positive sign is used and losses are considered.

        Sum of the operation time series of a Transmission component is used as the balanceLimit contribution:

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pym: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pym: pyomo ConcreteModel

        :param ID: ID of the regarded balanceLimitConstraint
        :param ID: string

        :param timeSeriesAggregation: states if the optimization of the energy system model should be done with

            (a) the full time series (False) or
            (b) clustered time series data (True).

        :type timeSeriesAggregation: boolean

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDictIn = getattr(pyM, "op_" + abbrvName), getattr(
            pyM, "operationVarDictIn_" + abbrvName
        )
        opVarDictOut = getattr(pyM, "operationVarDictOut_" + abbrvName)
        limitDict = getattr(pyM, "balanceLimitDict")
        if timeSeriesAggregation:
            periods = esM.typicalPeriods
            timeSteps = esM.timeStepsPerPeriod
        else:
            periods = esM.periods
            timeSteps = esM.totalTimeSteps
        aut = sum(
            opVar[loc_ + "_" + loc, compName, p, t]
            * (
                1
                - compDict[compName].losses[loc_ + "_" + loc]
                * compDict[compName].distances[loc_ + "_" + loc]
            )
            * esM.periodOccurrences[p]
            for loc_ in opVarDictIn[loc].keys()
            for compName in opVarDictIn[loc][loc_]
            if compName in limitDict[(ID, loc)]
            for p in periods
            for t in timeSteps
        ) - sum(
            opVar[loc + "_" + loc_, compName, p, t] * esM.periodOccurrences[p]
            for loc_ in opVarDictOut[loc].keys()
            for compName in opVarDictOut[loc][loc_]
            if compName in limitDict[(ID, loc)]
            for p in periods
            for t in timeSteps
        )
        return aut

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        opexOp = self.getEconomicsTD(
            pyM, esM, ["opexPerOperation"], "op", "operationVarDictOut"
        )

        capexCap = self.getEconomicsTI(
            pyM,
            factorNames=["processedInvestPerCapacity", "QPcostDev"],
            QPfactorNames=["QPcostScale", "processedInvestPerCapacity"],
            varName="cap",
            divisorName="CCF",
            QPdivisorNames=["QPbound", "CCF"],
        )
        capexDec = self.getEconomicsTI(
            pyM, ["processedInvestIfBuilt"], "designBin", "CCF"
        )
        opexCap = self.getEconomicsTI(
            pyM,
            factorNames=["processedOpexPerCapacity", "QPcostDev"],
            QPfactorNames=["QPcostScale", "processedOpexPerCapacity"],
            varName="cap",
            QPdivisorNames=["QPbound"],
        )
        opexDec = self.getEconomicsTI(pyM, ["processedOpexIfBuilt"], "designBin")

        return opexOp + capexCap + capexDec + opexCap + opexDec

    def setOptimalValues(self, esM, pyM):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)
        mapC = {
            loc1 + "_" + loc2: (loc1, loc2)
            for loc1 in esM.locations
            for loc2 in esM.locations
        }

        def _setOptimalValues(self, esM, pyM, indexColumns, plantUnit, unitApp=""):

            compDict, abbrvName = self.componentsDict, self.abbrvName
            capVar = getattr(esM.pyM, "cap_" + abbrvName)
            binVar = getattr(esM.pyM, "designBin_" + abbrvName)

            props = [
                "capacity",
                "isBuilt",
                "capexCap",
                "capexIfBuilt",
                "opexCap",
                "opexIfBuilt",
                "TAC",
                "invest",
            ]
            units = [
                "[-]",
                "[-]",
                "[" + esM.costUnit + "/a]",
                "[" + esM.costUnit + "/a]",
                "[" + esM.costUnit + "/a]",
                "[" + esM.costUnit + "/a]",
                "[" + esM.costUnit + "/a]",
                "[" + esM.costUnit + "]",
            ]
            tuples = [
                (compName, prop, unit)
                for compName in compDict.keys()
                for prop, unit in zip(props, units)
            ]
            tuples = list(
                map(
                    lambda x: (
                        x[0],
                        x[1],
                        "[" + getattr(compDict[x[0]], plantUnit) + unitApp + "]",
                    )
                    if x[1] == "capacity"
                    else x,
                    tuples,
                )
            )
            mIndex = pd.MultiIndex.from_tuples(
                tuples, names=["Component", "Property", "Unit"]
            )
            optSummary = pd.DataFrame(
                index=mIndex, columns=sorted(indexColumns)
            ).sort_index()

            # Get and set optimal variable values for expanded capacities
            values = capVar.get_values()
            optVal = utils.formatOptimizationOutput(values, "designVariables", "1dim")
            optVal_ = utils.formatOptimizationOutput(
                values, "designVariables", self.dimension, compDict=compDict
            )
            self.capacityVariablesOptimum = optVal_

            if optVal is not None:
                # Check if the installed capacities are close to a bigM value for components with design decision variables but
                # ignores cases where bigM was substituted by capacityMax parameter (see bigM constraint)
                for compName, comp in compDict.items():
                    if (
                        comp.hasIsBuiltBinaryVariable
                        and (comp.capacityMax is None)
                        and optVal.loc[compName].max() >= comp.bigM * 0.9
                        and esM.verbose < 2
                    ):  # and comp.capacityMax is None
                        warnings.warn(
                            "the capacity of component "
                            + compName
                            + " is in one or more locations close "
                            + "or equal to the chosen Big M. Consider rerunning the simulation with a higher"
                            + " Big M."
                        )

                # Calculate the investment costs i (proportional to capacity expansion)
                i = optVal.apply(
                    lambda cap: cap
                    * compDict[cap.name].processedInvestPerCapacity
                    * compDict[cap.name].QPcostDev
                    + (
                        compDict[cap.name].processedInvestPerCapacity
                        * compDict[cap.name].QPcostScale
                        / (compDict[cap.name].QPbound)
                        * cap
                        * cap
                    ),
                    axis=1,
                )
                # Calculate the annualized investment costs cx (CAPEX)
                cx = optVal.apply(
                    lambda cap: (
                        cap
                        * compDict[cap.name].processedInvestPerCapacity
                        * compDict[cap.name].QPcostDev
                        / compDict[cap.name].CCF
                    )
                    + (
                        compDict[cap.name].processedInvestPerCapacity
                        / compDict[cap.name].CCF
                        * compDict[cap.name].QPcostScale
                        / (compDict[cap.name].QPbound)
                        * cap
                        * cap
                    ),
                    axis=1,
                )
                # Calculate the annualized operational costs ox (OPEX)
                ox = optVal.apply(
                    lambda cap: cap
                    * compDict[cap.name].processedOpexPerCapacity
                    * compDict[cap.name].QPcostDev
                    + (
                        compDict[cap.name].processedOpexPerCapacity
                        * compDict[cap.name].QPcostScale
                        / (compDict[cap.name].QPbound)
                        * cap
                        * cap
                    ),
                    axis=1,
                )

                # Fill the optimization summary with the calculated values for invest, CAPEX and OPEX
                # (due to capacity expansion).
                optSummary.loc[
                    [
                        (
                            ix,
                            "capacity",
                            "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                        )
                        for ix in optVal.index
                    ],
                    optVal.columns,
                ] = optVal.values
                optSummary.loc[
                    [(ix, "invest", "[" + esM.costUnit + "]") for ix in i.index],
                    i.columns,
                ] = i.values
                optSummary.loc[
                    [(ix, "capexCap", "[" + esM.costUnit + "/a]") for ix in cx.index],
                    cx.columns,
                ] = cx.values
                optSummary.loc[
                    [(ix, "opexCap", "[" + esM.costUnit + "/a]") for ix in ox.index],
                    ox.columns,
                ] = ox.values

            # Get and set optimal variable values for binary investment decisions (isBuiltBinary).
            values = binVar.get_values()
            optVal = utils.formatOptimizationOutput(values, "designVariables", "1dim")
            optVal_ = utils.formatOptimizationOutput(
                values, "designVariables", self.dimension, compDict=compDict
            )
            self.isBuiltVariablesOptimum = optVal_

            if optVal is not None:
                # Calculate the investment costs i (fix value if component is built)
                i = optVal.apply(
                    lambda dec: dec * compDict[dec.name].processedInvestIfBuilt, axis=1
                )
                # Calculate the annualized investment costs cx (fix value if component is built)
                cx = optVal.apply(
                    lambda dec: dec
                    * compDict[dec.name].processedInvestIfBuilt
                    / compDict[dec.name].CCF,
                    axis=1,
                )
                # Calculate the annualized operational costs ox (fix value if component is built)
                ox = optVal.apply(
                    lambda dec: dec * compDict[dec.name].processedOpexIfBuilt, axis=1
                )

                # Fill the optimization summary with the calculated values for invest, CAPEX and OPEX
                # (due to isBuilt decisions).
                optSummary.loc[
                    [(ix, "isBuilt", "[-]") for ix in optVal.index], optVal.columns
                ] = optVal.values
                optSummary.loc[
                    [(ix, "invest", "[" + esM.costUnit + "]") for ix in cx.index],
                    cx.columns,
                ] += i.values
                optSummary.loc[
                    [
                        (ix, "capexIfBuilt", "[" + esM.costUnit + "/a]")
                        for ix in cx.index
                    ],
                    cx.columns,
                ] = cx.values
                optSummary.loc[
                    [
                        (ix, "opexIfBuilt", "[" + esM.costUnit + "/a]")
                        for ix in ox.index
                    ],
                    ox.columns,
                ] = ox.values

            # Summarize all annualized contributions to the total annual cost
            optSummary.loc[optSummary.index.get_level_values(1) == "TAC"] = (
                optSummary.loc[
                    (optSummary.index.get_level_values(1) == "capexCap")
                    | (optSummary.index.get_level_values(1) == "opexCap")
                    | (optSummary.index.get_level_values(1) == "capexIfBuilt")
                    | (optSummary.index.get_level_values(1) == "opexIfBuilt")
                ]
                .groupby(level=0)
                .sum()
                .values
            )

            return optSummary

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = _setOptimalValues(
            self, esM, pyM, mapC.keys(), "commodityUnit"
        )

        for compName, comp in compDict.items():
            for cost in [
                "invest",
                "capexCap",
                "capexIfBuilt",
                "opexCap",
                "opexIfBuilt",
                "TAC",
            ]:
                data = optSummaryBasic.loc[compName, cost]
                optSummaryBasic.loc[compName, cost] = (data).values

        # Set optimal operation variables and append optimization summary
        optVal = utils.formatOptimizationOutput(
            opVar.get_values(), "operationVariables", "1dim", esM.periodsOrder, esM=esM
        )
        optVal_ = utils.formatOptimizationOutput(
            opVar.get_values(),
            "operationVariables",
            "2dim",
            esM.periodsOrder,
            compDict=compDict,
            esM=esM,
        )
        self.operationVariablesOptimum = optVal_

        props = ["operation", "opexOp"]
        # Unit dict: Specify units for props
        units = {props[0]: ["[-*h]", "[-*h/a]"], props[1]: ["[" + esM.costUnit + "/a]"]}
        # Create tuples for the optSummary's multiIndex. Combine component with the respective properties and units.
        tuples = [
            (compName, prop, unit)
            for compName in compDict.keys()
            for prop in props
            for unit in units[prop]
        ]
        # Replace placeholder with correct unit of component
        tuples = list(
            map(
                lambda x: (x[0], x[1], x[2].replace("-", compDict[x[0]].commodityUnit))
                if x[1] == "operation"
                else x,
                tuples,
            )
        )
        mIndex = pd.MultiIndex.from_tuples(
            tuples, names=["Component", "Property", "Unit"]
        )
        optSummary = pd.DataFrame(
            index=mIndex, columns=sorted(mapC.keys())
        ).sort_index()

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name].opexPerOperation, axis=1)
            optSummary.loc[
                [
                    (ix, "operation", "[" + compDict[ix].commodityUnit + "*h/a]")
                    for ix in opSum.index
                ],
                opSum.columns,
            ] = (
                opSum.values / esM.numberOfYears
            )
            optSummary.loc[
                [
                    (ix, "operation", "[" + compDict[ix].commodityUnit + "*h]")
                    for ix in opSum.index
                ],
                opSum.columns,
            ] = opSum.values
            optSummary.loc[
                [(ix, "opexOp", "[" + esM.costUnit + "/a]") for ix in ox.index],
                ox.columns,
            ] = (
                ox.values / esM.numberOfYears * 0.5
            )

        optSummary = optSummary.append(optSummaryBasic).sort_index()

        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == "TAC"] = (
            optSummary.loc[
                (optSummary.index.get_level_values(1) == "TAC")
                | (optSummary.index.get_level_values(1) == "opexOp")
            ]
            .groupby(level=0)
            .sum()
            .values
        )

        # Split connection indices to two location indices
        optSummary = optSummary.stack()
        indexNew = []
        for tup in optSummary.index.tolist():
            loc1, loc2 = mapC[tup[3]]
            indexNew.append((tup[0], tup[1], tup[2], loc1, loc2))
        optSummary.index = pd.MultiIndex.from_tuples(indexNew)
        optSummary = optSummary.unstack(level=-1)
        names = list(optSummaryBasic.index.names)
        names.append("LocationIn")
        optSummary.index.set_names(names, inplace=True)

        self.optSummary = optSummary

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
        return super().getOptimalValues(name)
