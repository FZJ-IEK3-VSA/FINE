from FINE.component import Component, ComponentModel
from FINE import utils
import pandas as pd
import pyomo.environ as pyomo
import warnings


class Source(Component):
    """
    A Source component can transfer a commodity over the energy system boundary into the system.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
        hasCapacityVariable,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=1,
        hasIsBuiltBinaryVariable=False,
        bigM=None,
        operationRateMax=None,
        operationRateFix=None,
        tsaWeight=1,
        commodityLimitID=None,
        yearlyLimit=None,
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
        commodityCost=0,
        commodityRevenue=0,
        commodityCostTimeSeries=None,
        commodityRevenueTimeSeries=None,
        opexPerCapacity=0,
        opexIfBuilt=0,
        QPcostScale=0,
        interestRate=0.08,
        economicLifetime=10,
        technicalLifetime=None,
        yearlyFullLoadHoursMin=None,
        yearlyFullLoadHoursMax=None,
        balanceLimitID=None,
    ):

        """
        Constructor for creating an Source class instance.
        The Source component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        .. note::
            The Sink class inherits from the Source class and is initialized with the same parameter set.

        **Required arguments:**

        :param commodity: to the component related commodity.
        :type commodity: string

        :param hasCapacityVariable: specifies if the component should be modeled with a capacity or not.
            Examples:

            * A wind turbine has a capacity given in GW_electric -> hasCapacityVariable is True.
            * Emitting CO2 into the environment is not per se limited by a capacity ->
              hasCapaityVariable is False.

        :type hasCapacityVariable: boolean

        **Default arguments:**

        :param operationRateMax: if specified, indicates a maximum operation rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit for each time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to equal the
            in the energy system model specified locations. The data in ineligible locations are set to zero.

        :param operationRateFix: if specified, indicates a fixed operation rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit for each time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to equal the
            in the energy system model specified locations. The data in ineligible locations are set to zero.

        :param commodityCostTimeSeries: if specified, indicates commodity cost rates for each location and each
            time step by a positive float. The values are given as specific values relative to the commodityUnit
            for each time step.
            |br| * the default value is None
        :type commodityCostTimeSeries: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to equal the
            in the energy system model specified locations. The data in ineligible locations are set to zero.

        :param commodityRevenueTimeSeries:  if specified, indicates commodity revenue rate for each location and
            each time step by a positive float. The values are given as specific values relative to the
            commodityUnit for each time step.
            |br| * the default value is None
        :type commodityRevenueTimeSeries: None or Pandas DataFrame with positive (>= 0) entries. The row indices
            have to match the in the energy system model specified time steps. The column indices have to equal
            the in the energy system model specified locations. The data in ineligible locations are set to zero.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param commodityLimitID: can be specified to limit an annual commodity import/export over the
            energySystemModel's boundaries for one or multiple Source/Sink components. If the same ID
            is used in multiple components, the sum of all imports and exports is considered. If a
            commoditiyLimitID is specified, the yearlyLimit parameters has to be set as well.
            |br| * the default value is None
        :type commodityLimitID: string

        :param yearlyLimit: if specified, indicates a yearly import/export commodity limit for all components with
            the same commodityLimitID. If positive, the commodity flow leaving the energySystemModel is
            limited. If negative, the commodity flow entering the energySystemModel is limited. If a
            yearlyLimit is specified, the commoditiyLimitID parameters has to be set as well.
            Examples:

            * CO2 can be emitted in power plants by burning natural gas or coal. The CO2 which goes into
              the atmosphere over the energy system's boundaries is modelled as a Sink. CO2 can also be a
              Source taken directly from the atmosphere (over the energy system's boundaries) for a
              methanation process. The commodityUnit for CO2 is tonnes_CO2. Overall, +XY tonnes_CO2 are
              allowed to be emitted during the year. All Sources/Sinks producing or consuming CO2 over the
              energy system's boundaries have the same commodityLimitID and the same yearlyLimit of +XY.
            * The maximum annual import of a certain chemical (commodityUnit tonnes_chem) is limited to
              XY tonnes_chem. The Source component modeling this import has a commodityLimitID
              "chemicalComponentLimitID" and a yearlyLimit of -XY.

            |br| * the default value is None
        :type yearlyLimit: float

        :param opexPerOperation: describes the cost for one unit of the operation. The cost which is directly
            proportional to the operation of the component is obtained by multiplying the opexPerOperation parameter
            with the annual sum of the operational time series of the components.
            The opexPerOperation can either be given as a float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerOperation: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param commodityCost: describes the cost value of one operation´s unit of the component.
            The cost which is directly proportional to the operation of the component
            is obtained by multiplying the commodityCost parameter with the annual sum of the
            time series of the components. The commodityCost can either be given as a
            float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            Example:

            * In a national energy system, natural gas could be purchased from another country with a
              certain cost.

            |br| * the default value is 0
        :type commodityCost: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param commodityRevenue: describes the revenue of one operation´s unit of the component.
            The revenue which is directly proportional to the operation of the component
            is obtained by multiplying the commodityRevenue parameter with the annual sum of the
            time series of the components. The commodityRevenue can either be given as a
            float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            Example:

            * Modeling a PV electricity feed-in tariff for a household

            |br| * the default value is 0
        :type commodityRevenue: positive (>=0) float or Pandas Series with positive (>=0) values.
            The indices of the series have to equal the in the energy system model specified locations.

        :param balanceLimitID: ID for the respective balance limit (out of the balance limits introduced in the esM).
            Should be specified if the respective component of the SourceSinkModel is supposed to be included in
            the balance analysis. If the commodity is transported out of the region, it is counted as a negative, if
            it is imported into the region it is considered positive.
            |br| * the default value is None
        :type balanceLimitID: string
        """

        Component.__init__(
            self,
            esM,
            name,
            dimension="1dim",
            hasCapacityVariable=hasCapacityVariable,
            capacityVariableDomain=capacityVariableDomain,
            capacityPerPlantUnit=capacityPerPlantUnit,
            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable,
            bigM=bigM,
            locationalEligibility=locationalEligibility,
            capacityMin=capacityMin,
            capacityMax=capacityMax,
            partLoadMin=partLoadMin,
            sharedPotentialID=sharedPotentialID,
            linkedQuantityID=linkedQuantityID,
            capacityFix=capacityFix,
            isBuiltFix=isBuiltFix,
            investPerCapacity=investPerCapacity,
            investIfBuilt=investIfBuilt,
            opexPerCapacity=opexPerCapacity,
            opexIfBuilt=opexIfBuilt,
            QPcostScale=QPcostScale,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
            technicalLifetime=None,
            yearlyFullLoadHoursMin=yearlyFullLoadHoursMin,
            yearlyFullLoadHoursMax=yearlyFullLoadHoursMax,
        )

        # Set general source/sink data: ID and yearly limit
        utils.isEnergySystemModelInstance(esM), utils.checkCommodities(esM, {commodity})
        self.commodity, self.commodityUnit = (
            commodity,
            esM.commodityUnitsDict[commodity],
        )
        # TODO check value and type correctness
        self.commodityLimitID, self.yearlyLimit = commodityLimitID, yearlyLimit
        self.balanceLimitID = balanceLimitID
        self.sign = 1
        self.modelingClass = SourceSinkModel

        # Set additional economic data: opexPerOperation, commodityCost, commodityRevenue
        self.opexPerOperation = utils.checkAndSetCostParameter(
            esM, name, opexPerOperation, "1dim", locationalEligibility
        )
        self.commodityCost = utils.checkAndSetCostParameter(
            esM, name, commodityCost, "1dim", locationalEligibility
        )

        self.commodityRevenue = utils.checkAndSetCostParameter(
            esM, name, commodityRevenue, "1dim", locationalEligibility
        )

        self.commodityCostTimeSeries = commodityCostTimeSeries
        self.fullCommodityCostTimeSeries = utils.checkAndSetTimeSeries(
            esM, name, commodityCostTimeSeries, locationalEligibility
        )
        (
            self.aggregatedCommodityCostTimeSeries,
            self.processedCommodityCostTimeSeries,
        ) = (None, None)

        self.commodityRevenueTimeSeries = commodityRevenueTimeSeries
        self.fullCommodityRevenueTimeSeries = utils.checkAndSetTimeSeries(
            esM, name, commodityRevenueTimeSeries, locationalEligibility
        )
        (
            self.aggregatedCommodityRevenueTimeSeries,
            self.processedCommodityRevenueTimeSeries,
        ) = (None, None)

        self.operationRateMax = operationRateMax
        self.operationRateFix = operationRateFix

        self.fullOperationRateMax = utils.checkAndSetTimeSeries(
            esM, name, operationRateMax, locationalEligibility
        )
        self.aggregatedOperationRateMax, self.processedOperationRateMax = None, None

        self.fullOperationRateFix = utils.checkAndSetTimeSeries(
            esM, name, operationRateFix, locationalEligibility
        )
        self.aggregatedOperationRateFix, self.processedOperationRateFix = None, None

        # Set location-specific operation parameters: operationRateMax or operationRateFix, tsaweight
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

        # Set locational eligibility
        operationTimeSeries = (
            self.fullOperationRateFix
            if self.fullOperationRateFix is not None
            else self.fullOperationRateMax
        )
        self.locationalEligibility = utils.setLocationalEligibility(
            esM,
            self.locationalEligibility,
            self.capacityMax,
            self.capacityFix,
            self.isBuiltFix,
            self.hasCapacityVariable,
            operationTimeSeries,
        )

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a source component to the given energy system model.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)

    def setTimeSeriesData(self, hasTSA):
        """
        Function for setting the maximum operation rate, fixed operation rate and cost or revenue time series depending
        on whether a time series analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        self.processedOperationRateMax = (
            self.aggregatedOperationRateMax if hasTSA else self.fullOperationRateMax
        )
        self.processedOperationRateFix = (
            self.aggregatedOperationRateFix if hasTSA else self.fullOperationRateFix
        )
        self.processedCommodityCostTimeSeries = (
            self.aggregatedCommodityCostTimeSeries
            if hasTSA
            else self.fullCommodityCostTimeSeries
        )
        self.processedCommodityRevenueTimeSeries = (
            self.aggregatedCommodityRevenueTimeSeries
            if hasTSA
            else self.fullCommodityRevenueTimeSeries
        )

    def getDataForTimeSeriesAggregation(self):
        """
        Function for getting the required data if a time series aggregation is requested.
        """
        weightDict, data = {}, []
        weightDict, data = self.prepareTSAInput(
            self.fullOperationRateFix,
            self.fullOperationRateMax,
            "_operationRate_",
            self.tsaWeight,
            weightDict,
            data,
        )
        weightDict, data = self.prepareTSAInput(
            self.fullCommodityCostTimeSeries,
            None,
            "_commodityCostTimeSeries_",
            self.tsaWeight,
            weightDict,
            data,
        )
        weightDict, data = self.prepareTSAInput(
            self.fullCommodityRevenueTimeSeries,
            None,
            "_commodityRevenueTimeSeries_",
            self.tsaWeight,
            weightDict,
            data,
        )
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):
        """
        Function for determining the aggregated maximum rate and the aggregated fixed operation rate.

        :param data: Pandas DataFrame with the clustered time series data of the source component
        :type data: Pandas DataFrame
        """
        self.aggregatedOperationRateFix = self.getTSAOutput(
            self.fullOperationRateFix, "_operationRate_", data
        )
        self.aggregatedOperationRateMax = self.getTSAOutput(
            self.fullOperationRateMax, "_operationRate_", data
        )
        self.aggregatedCommodityCostTimeSeries = self.getTSAOutput(
            self.fullCommodityCostTimeSeries, "_commodityCostTimeSeries_", data
        )
        self.aggregatedCommodityRevenueTimeSeries = self.getTSAOutput(
            self.fullCommodityRevenueTimeSeries, "_commodityRevenueTimeSeries_", data
        )


class Sink(Source):
    """
    A Sink component can transfer a commodity over the energy system boundary out of the system.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
        hasCapacityVariable,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=1,
        hasIsBuiltBinaryVariable=False,
        bigM=None,
        operationRateMax=None,
        operationRateFix=None,
        tsaWeight=1,
        commodityLimitID=None,
        yearlyLimit=None,
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
        commodityCost=0,
        commodityRevenue=0,
        commodityCostTimeSeries=None,
        commodityRevenueTimeSeries=None,
        opexPerCapacity=0,
        opexIfBuilt=0,
        QPcostScale=0,
        interestRate=0.08,
        economicLifetime=10,
        technicalLifetime=None,
        balanceLimitID=None,
    ):
        """
        Constructor for creating an Sink class instance.

        The Sink class inherits from the Source class. They coincide with the input parameters
        (see Source class for the parameter description) and differ in the sign
        parameter, which is equal to -1 for Sink objects and +1 for Source objects.
        """
        Source.__init__(
            self,
            esM,
            name,
            commodity=commodity,
            hasCapacityVariable=hasCapacityVariable,
            capacityVariableDomain=capacityVariableDomain,
            capacityPerPlantUnit=capacityPerPlantUnit,
            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable,
            bigM=bigM,
            operationRateMax=operationRateMax,
            operationRateFix=operationRateFix,
            tsaWeight=tsaWeight,
            commodityLimitID=commodityLimitID,
            yearlyLimit=yearlyLimit,
            locationalEligibility=locationalEligibility,
            capacityMin=capacityMin,
            capacityMax=capacityMax,
            partLoadMin=partLoadMin,
            sharedPotentialID=sharedPotentialID,
            linkedQuantityID=linkedQuantityID,
            capacityFix=capacityFix,
            isBuiltFix=isBuiltFix,
            investPerCapacity=investPerCapacity,
            investIfBuilt=investIfBuilt,
            opexPerOperation=opexPerOperation,
            commodityCost=commodityCost,
            commodityRevenue=commodityRevenue,
            commodityCostTimeSeries=commodityCostTimeSeries,
            commodityRevenueTimeSeries=commodityRevenueTimeSeries,
            opexPerCapacity=opexPerCapacity,
            opexIfBuilt=opexIfBuilt,
            QPcostScale=QPcostScale,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
            technicalLifetime=technicalLifetime,
            balanceLimitID=balanceLimitID,
        )

        self.sign = -1


class SourceSinkModel(ComponentModel):
    """
    A SourceSinkModel class instance will be instantly created if a Source class instance or a Sink class instance is
    initialized. It is used for the declaration of the sets, variables and constraints which are valid for the
    Source/Sink class instance. These declarations are necessary for the modeling and optimization of the
    energy system model. The SourceSinkModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        """
        Constructor for creating a SourceSinkModel class instance.
        """
        self.abbrvName = "srcSnk"
        self.dimension = "1dim"
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareYearlyCommodityLimitationDict(self, pyM):
        """
        Declare source/sink components with linked commodity limits and check if the linked components have the same
        yearly upper limit.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        yearlyCommodityLimitationDict = {}
        for compName, comp in self.componentsDict.items():
            if comp.commodityLimitID is not None:
                ID, limit = comp.commodityLimitID, comp.yearlyLimit
                if (
                    ID in yearlyCommodityLimitationDict
                    and limit != yearlyCommodityLimitationDict[ID][0]
                ):
                    raise ValueError(
                        "yearlyLimitationIDs with different upper limits detected."
                    )
                yearlyCommodityLimitationDict.setdefault(ID, (limit, []))[1].append(
                    compName
                )
        setattr(
            pyM,
            "yearlyCommodityLimitationDict_" + self.abbrvName,
            yearlyCommodityLimitationDict,
        )

    def declareSets(self, esM, pyM):
        """
        Declare sets and dictionaries: design variable sets, operation variable set, operation mode sets and
        linked commodity limitation dictionary.

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

        # Declare operation variable set
        self.declareOpVarSet(esM, pyM)
        self.declareOperationBinarySet(pyM)

        # Declare sets for case differentiation of operating modes
        self.declareOperationModeSets(
            pyM, "opConstrSet", "processedOperationRateMax", "processedOperationRateFix"
        )

        # Declare commodity limitation dictionary
        self.declareYearlyCommodityLimitationDict(pyM)

        # Declare minimum yearly full load hour set
        self.declareYearlyFullLoadHoursMinSet(pyM)

        # Declare maximum yearly full load hour set
        self.declareYearlyFullLoadHoursMaxSet(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary):
        """
        Declare design and operation variables.

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
        # Operation of component [commodityUnit*hour]
        self.declareOperationVars(pyM, "op")
        # Operation of component as binary [1/0]
        self.declareOperationBinaryVars(pyM, "op_bin")

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def yearlyLimitationConstraint(self, pyM, esM):
        """
        Limit annual commodity imports/exports over the energySystemModel's boundaries for one or multiple
        Source/Sink components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        warnings.warn(
            "The yearly limit is depreceated and moved to the balanceLimit",
            DeprecationWarning,
        )
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)
        limitDict = getattr(pyM, "yearlyCommodityLimitationDict_" + abbrvName)

        def yearlyLimitationConstraint(pyM, key):
            sumEx = -sum(
                opVar[loc, compName, p, t]
                * compDict[compName].sign
                * esM.periodOccurrences[p]
                / esM.numberOfYears
                for loc, compName, p, t in opVar
                if compName in limitDict[key][1]
            )
            sign = (
                limitDict[key][0] / abs(limitDict[key][0])
                if limitDict[key][0] != 0
                else 1
            )
            return sign * sumEx <= sign * limitDict[key][0]

        setattr(
            pyM,
            "ConstrYearlyLimitation_" + abbrvName,
            pyomo.Constraint(limitDict.keys(), rule=yearlyLimitationConstraint),
        )

    def declareComponentConstraints(self, esM, pyM):
        """
        Declare time independent and dependent constraints.

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
        # Set yearly full load hours minimum limit
        self.yearlyFullLoadHoursMin(pyM, esM)
        # Set yearly full load hours maximum limit
        self.yearlyFullLoadHoursMax(pyM, esM)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [commodityUnit*h] limited by the installed capacity [commodityUnit] multiplied by the hours per
        # time step [h]
        self.operationMode1(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] equal to the installed capacity [commodityUnit] multiplied by operation time
        # series [-] and the hours per time step [h])
        self.operationMode2(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] limited by the installed capacity [commodityUnit] multiplied by operation time
        # series [-] and the hours per time step [h])
        self.operationMode3(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] equal to the operation time series [commodityUnit*h]
        self.operationMode4(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] limited by the operation time series [commodityUnit*h]
        self.operationMode5(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [physicalUnit*h] is limited by minimum part Load
        self.additionalMinPartLoad(
            pyM, esM, "ConstrOperation", "opConstrSet", "op", "op_bin", "cap"
        )

        self.yearlyLimitationConstraint(pyM, esM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        """Get contributions to shared location potential."""
        return super().getSharedPotentialContribution(pyM, key, loc)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        """
        Check if operation variables exist in the modeling class at a location which are connected to a commodity.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: Name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """
        return any(
            [
                comp.commodity == commod and comp.locationalEligibility[loc] == 1
                for comp in self.componentsDict.values()
            ]
        )

    def getBalanceLimitContribution(
        self, esM, pyM, ID, timeSeriesAggregation, loc=None
    ):
        """
        Get contribution to balanceLimitConstraint (Further read in EnergySystemModel).
        Sum of the operation time series of a SourceSink component is used as the balanceLimit contribution:

        - If component is a Source it contributes with a positive sign to the limit. Example: Electricity Purchase
        - A Sink contributes with a negative sign. Example: Sale of electricity

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
        opVar, opVarDict = getattr(pyM, "op_" + abbrvName), getattr(
            pyM, "operationVarDict_" + abbrvName
        )
        limitDict = getattr(pyM, "balanceLimitDict")

        if timeSeriesAggregation:
            periods = esM.typicalPeriods
            timeSteps = esM.timeStepsPerPeriod
        else:
            periods = esM.periods
            timeSteps = esM.totalTimeSteps
        # Check if locational input is not set in esM, if so additionally loop over all locations
        if loc is None:
            balance = sum(
                opVar[loc, compName, p, t]
                * compDict[compName].sign
                * esM.periodOccurrences[p]
                for compName in compDict.keys()
                if compName in limitDict[ID]
                for p in periods
                for t in timeSteps
                for loc in esM.locations
            )
        # Otherwise get the contribution for specific region
        else:
            balance = sum(
                opVar[loc, compName, p, t]
                * compDict[compName].sign
                * esM.periodOccurrences[p]
                for compName in compDict.keys()
                if compName in limitDict[(ID, loc)]
                for p in periods
                for t in timeSteps
            )
        return balance

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """
        Get contribution to a commodity balance.

        .. math::

            \\text{C}^{comp,comm}_{loc,p,t} = - op_{loc,p,t}^{comp,op}  \\text{Sink}

        .. math::
            \\text{C}^{comp,comm}_{loc,p,t} = op_{loc,p,t}^{comp,op} \\text{Source}

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDict = getattr(pyM, "op_" + abbrvName), getattr(
            pyM, "operationVarDict_" + abbrvName
        )
        return sum(
            opVar[loc, compName, p, t] * compDict[compName].sign
            for compName in opVarDict[loc]
            if compDict[compName].commodity == commod
        )

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.
            :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
            :type esM: esM - EnergySystemModel class instance

            :param pym: pyomo ConcreteModel which stores the mathematical formulation of the model.
            :type pym: pyomo ConcreteModel
        """

        opexOp = self.getEconomicsTD(
            pyM, esM, ["opexPerOperation"], "op", "operationVarDict"
        )
        commodCost = self.getEconomicsTD(
            pyM, esM, ["commodityCost"], "op", "operationVarDict"
        )
        commodRevenue = self.getEconomicsTD(
            pyM, esM, ["commodityRevenue"], "op", "operationVarDict"
        )
        commodCostTimeSeries = self.getEconomicsTimeSeries(
            pyM, esM, "processedCommodityCostTimeSeries", "op", "operationVarDict"
        )
        commodRevenueTimeSeries = self.getEconomicsTimeSeries(
            pyM, esM, "processedCommodityRevenueTimeSeries", "op", "operationVarDict"
        )

        return (
            super().getObjectiveFunctionContribution(esM, pyM)
            + opexOp
            + commodCost
            + commodCostTimeSeries
            - (commodRevenue + commodRevenueTimeSeries)
        )

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pym: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pym: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(
            esM, pyM, esM.locations, "commodityUnit"
        )

        # Set optimal operation variables and append optimization summary
        optVal = utils.formatOptimizationOutput(
            opVar.get_values(), "operationVariables", "1dim", esM.periodsOrder, esM=esM
        )
        self.operationVariablesOptimum = optVal

        props = ["operation", "opexOp", "commodCosts", "commodRevenues"]
        # Unit dict: Specify units for props
        units = {
            props[0]: ["[-*h]", "[-*h/a]"],
            props[1]: ["[" + esM.costUnit + "/a]"],
            props[2]: ["[" + esM.costUnit + "/a]"],
            props[3]: ["[" + esM.costUnit + "/a]"],
        }
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
            index=mIndex, columns=sorted(esM.locations)
        ).sort_index()

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(
                lambda op: op * compDict[op.name].opexPerOperation[op.index], axis=1
            )
            cCost = opSum.apply(
                lambda op: op * compDict[op.name].commodityCost[op.index], axis=1
            )
            cRevenue = opSum.apply(
                lambda op: op * compDict[op.name].commodityRevenue[op.index], axis=1
            )
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
                ox.values / esM.numberOfYears
            )

            # get empty datframe for resulting time dependent (TD) cost sum
            cRevenueTD = pd.DataFrame(0.0, index=opSum.index, columns=opSum.columns)
            cCostTD = pd.DataFrame(0.0, index=opSum.index, columns=opSum.columns)

            for compName in opSum.index:
                if not compDict[compName].processedCommodityCostTimeSeries is None:

                    # in case of time series aggregation rearange clustered cost time series
                    calcCostTD = utils.buildFullTimeSeries(
                        compDict[compName]
                        .processedCommodityCostTimeSeries.unstack(level=1)
                        .stack(level=0),
                        esM.periodsOrder,
                        esM=esM,
                        divide=False,
                    )
                    # multiply with operation values to get the total cost
                    cCostTD.loc[compName, :] = (
                        optVal.xs(compName, level=0).T.mul(calcCostTD.T).sum(axis=0)
                    )

                if not compDict[compName].processedCommodityRevenueTimeSeries is None:
                    # in case of time series aggregation rearange clustered revenue time series
                    calcRevenueTD = utils.buildFullTimeSeries(
                        compDict[compName]
                        .processedCommodityRevenueTimeSeries.unstack(level=1)
                        .stack(level=0),
                        esM.periodsOrder,
                        esM=esM,
                        divide=False,
                    )
                    # multiply with operation values to get the total revenue
                    cRevenueTD.loc[compName, :] = (
                        optVal.xs(compName, level=0).T.mul(calcRevenueTD.T).sum(axis=0)
                    )

            optSummary.loc[
                [(ix, "commodCosts", "[" + esM.costUnit + "/a]") for ix in ox.index],
                ox.columns,
            ] = (cCostTD.values + cCost.values) / esM.numberOfYears
            optSummary.loc[
                [(ix, "commodRevenues", "[" + esM.costUnit + "/a]") for ix in ox.index],
                ox.columns,
            ] = (cRevenueTD.values + cRevenue.values) / esM.numberOfYears

        # get discounted investment cost as total annual cost (TAC)
        optSummary = optSummary.append(optSummaryBasic).sort_index()

        # add operation specific contributions to the total annual cost (TAC) and substract revenues
        optSummary.loc[optSummary.index.get_level_values(1) == "TAC"] = (
            optSummary.loc[
                (optSummary.index.get_level_values(1) == "TAC")
                | (optSummary.index.get_level_values(1) == "opexOp")
                | (optSummary.index.get_level_values(1) == "commodCosts")
            ]
            .groupby(level=0)
            .sum()
            .values
            - optSummary.loc[(optSummary.index.get_level_values(1) == "commodRevenues")]
            .groupby(level=0)
            .sum()
            .values
        )

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
