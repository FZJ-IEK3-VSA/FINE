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
        pathwayBalanceLimitID=None,
        stockCommissioning=None,
        floorTechnicalLifetime=True,
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
              hasCapacityVariable is False.

        :type hasCapacityVariable: boolean

        **Default arguments:**

        :param operationRateMax: if specified, indicates a maximum operation rate for each location and each time, if required also for each investment period, if
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit for each time step.
            |br| * the default value is None
        :type operationRateMax:

            * None
            * Pandas DataFrame with positive (>= 0) entries. The row indices have
              to match the in the energy system model specified time steps. The column indices have to equal the
              in the energy system model specified locations. The data in ineligible locations are set to zero.
            * a dictionary with investment periods as keys and one of the two options above as values


        :param operationRateFix: if specified, indicates a fixed operation rate for each location and each time, if required also for each investment period,
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit for each time step.
            |br| * the default value is None
        :type operationRateFix:

            * None
            * Pandas DataFrame with positive (>=0) per investment period. The row indices have
              to match the in the energy system model specified time steps. The column indices have to equal the
              in the energy system model specified locations. The data in ineligible locations are set to zero.
            * a dictionary with investment periods as keys and one of the two options above as values


        :param commodityCostTimeSeries: if specified, indicates commodity cost rates for each location and each
            time step, if required also for each investment period, by a positive float. The values are given as specific values relative to the commodityUnit
            for each time step.
            |br| * the default value is None
        :type commodityCostTimeSeries:

            * None
            * Pandas DataFrame with positive (>= 0) entries. The row indices have
              to match the in the energy system model specified time steps. The column indices have to equal the
              in the energy system model specified locations. The data in ineligible locations are set to zero.
            * a dictionary with investment periods as keys and one of the two options above as values


        :param commodityRevenueTimeSeries:  if specified, indicates commodity revenue rate for each location and
            each time step, if required also for each investment period, by a positive float. The values are given as specific values relative to the
            commodityUnit for each time step.
            |br| * the default value is None
        :type commodityRevenueTimeSeries:

            * None
            * Pandas DataFrame with positive (>= 0) entries. The row indices
              have to match the in the energy system model specified time steps. The column indices have to equal
              the in the energy system model specified locations. The data in ineligible locations are set to zero.
            * a dictionary with investment periods as keys and one of the two options above as values


        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param commodityLimitID: can be specified to limit an annual commodity import/export over the
            energySystemModel's boundaries for one or multiple Source/Sink components. If the same ID
            is used in multiple components, the sum of all imports and exports is considered. If a
            commodityLimitID is specified, the yearlyLimit parameters has to be set as well.
            |br| * the default value is None
        :type commodityLimitID: string

        :param yearlyLimit: if specified, indicates a yearly import/export commodity limit per investment period for all components with
            the same commodityLimitID. If positive, the commodity flow leaving the energySystemModel is
            limited. If negative, the commodity flow entering the energySystemModel is limited. If a
            yearlyLimit is specified, the commodityLimitID parameters has to be set as well. The yearlyLimit can also be specified for
            every investment period year individually.
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
        :type yearlyLimit:

            * float
            * a dictionary with investment periods as keys and float as values


        :param opexPerOperation: describes the cost for one unit of the operation. The cost which is directly
            proportional to the operation of the component is obtained by multiplying the opexPerOperation parameter
            with the annual sum of the operational time series of the components.
            The opexPerOperation can either be given as a float or a Pandas Series with location specific values or a dictionary per investment period with one of the previous options.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerOperation:

            * positive (>=0) float
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.


        :param commodityCost: describes the cost value of one operation´s unit of the component.
            The cost which is directly proportional to the operation of the component
            is obtained by multiplying the commodityCost parameter with the annual sum of the
            time series of the components. The commodityCost can either be given as a
            float or a Pandas Series with location specific values or a dictionary per investment period with one of the two previous options.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            Example:

            * In a national energy system, natural gas could be purchased from another country with a
              certain cost.

            |br| * the default value is 0
        :type commodityCost:

            * positive (>=0) float
            * Pandas Series with positive (>=0).The indices of the series have to equal the in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.


        :param commodityRevenue: describes the revenue of one operation´s unit of the component.
            The revenue which is directly proportional to the operation of the component
            is obtained by multiplying the commodityRevenue parameter with the annual sum of the
            time series of the components. The commodityRevenue can either be given as a
            float or a Pandas Series with location specific values or a dictionary per investment period with one of the two previous options.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            Example:

            * Modeling a PV electricity feed-in tariff for a household

            |br| * the default value is 0
        :type commodityRevenue:

            * positive (>=0) float
            * Pandas Series with positive (>=0). The indices of the series have to equal the in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.


        :param balanceLimitID: ID for the respective balance limit (out of the balance limits introduced in the esM).
            Should be specified if the respective component of the SourceSinkModel is supposed to be included in
            the balance analysis. If the commodity is transported out of the region, it is counted as a negative, if
            it is imported into the region it is considered positive.
            |br| * the default value is None
        :type balanceLimitID: string

        :param pathwayBalanceLimitID: similar to balanceLimitID just as restriction over the entire pathway.
            |br| * the default value is None
        :type pathwayBalanceLimitID: string
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
            technicalLifetime=technicalLifetime,
            floorTechnicalLifetime=floorTechnicalLifetime,
            yearlyFullLoadHoursMin=yearlyFullLoadHoursMin,
            yearlyFullLoadHoursMax=yearlyFullLoadHoursMax,
            stockCommissioning=stockCommissioning,
        )

        # Set general source/sink data: ID and yearly limit
        utils.isEnergySystemModelInstance(esM), utils.checkCommodities(esM, {commodity})
        self.commodity, self.commodityUnit = (
            commodity,
            esM.commodityUnitsDict[commodity],
        )
        # TODO check value and type correctness
        self.commodityLimitID = commodityLimitID
        self.balanceLimitID = balanceLimitID
        self.pathwayBalanceLimitID = pathwayBalanceLimitID
        self.sign = 1
        self.modelingClass = SourceSinkModel

        # yearlyLimit
        self.yearlyLimit = yearlyLimit
        self.processedYearlyLimit = utils.checkAndSetYearlyLimit(esM, yearlyLimit)

        # opexPerOperation
        self.opexPerOperation = opexPerOperation
        self.processedOpexPerOperation = utils.checkAndSetInvestmentPeriodCostParameter(
            esM,
            name,
            opexPerOperation,
            "1dim",
            locationalEligibility,
            esM.investmentPeriods,
        )

        # commodityCost
        self.commodityCost = commodityCost
        self.processedCommodityCost = utils.checkAndSetInvestmentPeriodCostParameter(
            esM,
            name,
            commodityCost,
            "1dim",
            locationalEligibility,
            esM.investmentPeriods,
        )

        # commodtyRevenue
        self.commodityRevenue = commodityRevenue
        self.processedCommodityRevenue = utils.checkAndSetInvestmentPeriodCostParameter(
            esM,
            name,
            commodityRevenue,
            "1dim",
            locationalEligibility,
            esM.investmentPeriods,
        )

        # commodityCostTimeSeries
        self.commodityCostTimeSeries = commodityCostTimeSeries
        self.fullCommodityCostTimeSeries = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, commodityCostTimeSeries, locationalEligibility
        )
        self.aggregatedCommodityCostTimeSeries = dict.fromkeys(esM.investmentPeriods)
        self.processedCommodityCostTimeSeries = dict.fromkeys(esM.investmentPeriods)

        # commodityRevenueTimeSeries
        self.commodityRevenueTimeSeries = commodityRevenueTimeSeries
        self.fullCommodityRevenueTimeSeries = {}
        self.fullCommodityRevenueTimeSeries = (
            utils.checkAndSetInvestmentPeriodTimeSeries(
                esM, name, commodityRevenueTimeSeries, locationalEligibility
            )
        )
        self.aggregatedCommodityRevenueTimeSeries = dict.fromkeys(esM.investmentPeriods)
        self.processedCommodityRevenueTimeSeries = dict.fromkeys(esM.investmentPeriods)

        # operationRateMax
        self.operationRateMax = operationRateMax
        self.fullOperationRateMax = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, operationRateMax, locationalEligibility
        )
        self.aggregatedOperationRateMax = {}
        self.processedOperationRateMax = {}

        # operationRateFix
        self.operationRateFix = operationRateFix
        self.fullOperationRateFix = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, operationRateFix, locationalEligibility
        )
        self.aggregatedOperationRateFix = {}
        self.processedOperationRateFix = {}

        # check for operationRateMax and operationRateFix
        for ip in esM.investmentPeriods:
            if (
                self.fullOperationRateFix[ip] is not None
                and self.fullOperationRateMax[ip] is not None
            ):
                self.fullOperationRateMax[ip] = None
                if esM.verbose < 2:
                    warnings.warn(
                        "If operationRateFix is specified, the operationRateMax parameter is not required.\n"
                        + "The operationRateMax time series of investment period "
                        + f"'{esM.investmentPeriodNames[ip]}' was set to None."
                    )

        # partLoadMin
        self.processedPartLoadMin = utils.checkAndSetPartLoadMin(
            esM,
            name,
            partLoadMin,
            self.fullOperationRateMax,
            self.fullOperationRateFix,
            self.bigM,
            self.hasCapacityVariable,
        )

        utils.isPositiveNumber(tsaWeight)
        self.tsaWeight = tsaWeight

        # set parameter to None if all years have None values
        self.fullOperationRateFix = utils.setParamToNoneIfNoneForAllYears(
            self.fullOperationRateFix
        )
        self.fullOperationRateMax = utils.setParamToNoneIfNoneForAllYears(
            self.fullOperationRateMax
        )
        self.fullCommodityCostTimeSeries = utils.setParamToNoneIfNoneForAllYears(
            self.fullCommodityCostTimeSeries
        )
        self.fullCommodityRevenueTimeSeries = utils.setParamToNoneIfNoneForAllYears(
            self.fullCommodityRevenueTimeSeries
        )
        self.processedYearlyLimit = utils.setParamToNoneIfNoneForAllYears(
            self.processedYearlyLimit
        )

        if self.fullOperationRateFix is not None:
            operationTimeSeries = self.fullOperationRateFix
        elif self.fullOperationRateMax is not None:
            operationTimeSeries = self.fullOperationRateMax
        else:
            operationTimeSeries = None

        self.processedLocationalEligibility = utils.setLocationalEligibility(
            esM,
            self.locationalEligibility,
            self.processedCapacityMax,
            self.processedCapacityFix,
            self.isBuiltFix,
            self.hasCapacityVariable,
            operationTimeSeries,
        )

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

    def getDataForTimeSeriesAggregation(self, ip):
        """Function for getting the required data if a time series aggregation is requested.

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """

        weightDict, data = {}, []
        weightDict, data = self.prepareTSAInput(
            self.fullOperationRateFix,
            self.fullOperationRateMax,
            "_operationRate_",
            self.tsaWeight,
            weightDict,
            data,
            ip,
        )
        weightDict, data = self.prepareTSAInput(
            self.fullCommodityCostTimeSeries,
            None,
            "_commodityCostTimeSeries_",
            self.tsaWeight,
            weightDict,
            data,
            ip,
        )
        weightDict, data = self.prepareTSAInput(
            self.fullCommodityRevenueTimeSeries,
            None,
            "_commodityRevenueTimeSeries_",
            self.tsaWeight,
            weightDict,
            data,
            ip,
        )
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data, ip):
        """
        Function for determining the aggregated maximum rate and the aggregated fixed operation rate.

        :param data: Pandas DataFrame with the clustered time series data of the source component
        :type data: Pandas DataFrame

        :param ip: investment period of transformation path analysis.
        :type ip: int

        """

        self.aggregatedOperationRateFix[ip] = self.getTSAOutput(
            self.fullOperationRateFix, "_operationRate_", data, ip
        )
        self.aggregatedOperationRateMax[ip] = self.getTSAOutput(
            self.fullOperationRateMax, "_operationRate_", data, ip
        )
        self.aggregatedCommodityCostTimeSeries[ip] = self.getTSAOutput(
            self.fullCommodityCostTimeSeries, "_commodityCostTimeSeries_", data, ip
        )
        self.aggregatedCommodityRevenueTimeSeries[ip] = self.getTSAOutput(
            self.fullCommodityRevenueTimeSeries,
            "_commodityRevenueTimeSeries_",
            data,
            ip,
        )

    def checkProcessedDataSets(self):
        """
        Check processed time series data after applying time series aggregation. If all entries of dictionary are None
        the parameter itself is set to None.
        """
        for parameter in [
            "processedOperationRateFix",
            "processedOperationRateMax",
            "processedCommodityCostTimeSeries",
            "processedCommodityRevenueTimeSeries",
        ]:
            setattr(
                self,
                parameter,
                utils.setParamToNoneIfNoneForAllYears(getattr(self, parameter)),
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
        pathwayBalanceLimitID=None,
        stockCommissioning=None,
        floorTechnicalLifetime=True,
    ):
        """
        Constructor for creating a Sink class instance.

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
            pathwayBalanceLimitID=pathwayBalanceLimitID,
            stockCommissioning=stockCommissioning,
            floorTechnicalLifetime=floorTechnicalLifetime,
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
        super().__init__()
        self.abbrvName = "srcSnk"
        self.dimension = "1dim"
        self._operationVariablesOptimum = {}

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareYearlyCommodityLimitationDict(self, pyM, esM):
        """
        Declare source/sink components with linked commodity limits and check if the linked components have the same
        yearly upper limit.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        yearlyCommodityLimitationDict = {}
        for ip in esM.investmentPeriods:
            for compName, comp in self.componentsDict.items():
                if comp.commodityLimitID is not None:
                    ID, limit = comp.commodityLimitID, comp.processedYearlyLimit[ip]
                    if (
                        ID,
                        ip,
                    ) in yearlyCommodityLimitationDict.keys() and limit != yearlyCommodityLimitationDict[
                        (ID, ip)
                    ][
                        0
                    ]:
                        raise ValueError(
                            "yearlyLimitationIDs with different upper limits detected."
                        )
                    yearlyCommodityLimitationDict.setdefault((ID, ip), (limit, []))[
                        1
                    ].append(compName)
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
        self.declareDesignVarSet(pyM, esM)
        self.declareCommissioningVarSet(pyM, esM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        # Declare design pathway sets
        self.declarePathwaySets(pyM, esM)
        self.declareLocationComponentSet(pyM)

        # Declare operation variable set
        self.declareOpVarSet(esM, pyM)

        # Declare sets for case differentiation of operating modes
        self.declareOperationModeSets(
            pyM, "opConstrSet", "processedOperationRateMax", "processedOperationRateFix"
        )

        # Declare commodity limitation dictionary
        self.declareYearlyCommodityLimitationDict(pyM, esM)

        # Declare minimum yearly full load hour set
        self.declareYearlyFullLoadHoursMinSet(pyM)

        # Declare maximum yearly full load hour set
        self.declareYearlyFullLoadHoursMaxSet(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary, relevanceThreshold):
        """
        Declare design and operation variables.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param relaxIsBuiltBinary: states if the optimization problem should be solved as a relaxed LP to get the lower
            bound of the problem.
            |br| * the default value is False
        :type declaresOptimizationProblem: boolean

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None
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
        self.declareOperationVars(pyM, esM, "op", relevanceThreshold=relevanceThreshold)
        # Operation of component as binary [1/0]
        self.declareOperationBinaryVars(pyM, "op_bin")
        # Capacity development variables [physicalUnit]
        self.declareCommissioningVars(pyM, esM)
        self.declareDecommissioningVars(pyM, esM)

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
            "The yearly limit is deprecated and moved to the balanceLimit",
            DeprecationWarning,
        )
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)
        limitDict = getattr(pyM, "yearlyCommodityLimitationDict_" + abbrvName)

        def yearlyLimitationConstraint(pyM, key, ip):
            sumEx = -sum(
                opVar[loc, compName, ip, p, t]
                * compDict[compName].sign
                * esM.periodOccurrences[ip][p]
                / esM.numberOfYears
                for loc, compName, _ip, p, t in opVar
                if (_ip == ip and compName in limitDict[(key, ip)][1])
            )
            sign = (
                limitDict[(key, ip)][0] / abs(limitDict[(key, ip)][0])
                if limitDict[(key, ip)][0] != 0
                else 1
            )
            return sign * sumEx <= sign * limitDict[(key, ip)][0]

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
        self.capacityFix(pyM, esM)
        # Set, if applicable, the binary design variables of a component
        self.designBinFix(pyM)
        # Set yearly full load hours minimum limit
        self.yearlyFullLoadHoursMin(
            pyM, esM, "yearlyFullLoadHoursMinSet", "ConstrYearlyFullLoadHoursMin", "op"
        )
        # Set yearly full load hours maximum limit
        self.yearlyFullLoadHoursMax(
            pyM, esM, "yearlyFullLoadHoursMaxSet", "ConstrYearlyFullLoadHoursMax", "op"
        )

        ################################################################################################################
        #                                    Declare pathway constraints                                               #
        ################################################################################################################
        # Set capacity development constraints over investment periods
        self.designDevelopmentConstraint(pyM, esM)
        self.decommissioningConstraint(pyM, esM)
        self.stockCapacityConstraint(pyM, esM)
        self.stockCommissioningConstraint(pyM, esM)

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
        # Operation [physicalUnit*h] is limited by minimum part Load
        self.additionalMinPartLoad(
            pyM, esM, "ConstrOperation", "opConstrSet", "op", "op_bin", "cap"
        )

        self.yearlyLimitationConstraint(pyM, esM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

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
                comp.commodity == commod
                and comp.processedLocationalEligibility[loc] == 1
                for comp in self.componentsDict.values()
            ]
        )

    def getBalanceLimitContribution(
        self, esM, pyM, ID, ip, timeSeriesAggregation, loc, componentNames
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

        :param ip: investment period of transformation path analysis.
        :type ip: int

        :param ID: ID of the regarded balanceLimitConstraint
        :param ID: string

        :param timeSeriesAggregation: states if the optimization of the energy system model should be done with

            (a) the full time series (False) or
            (b) clustered time series data (True).

        :type timeSeriesAggregation: boolean

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param componentNames: Names of components which contribute to the balance limit
        :type componentNames: list
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)

        if timeSeriesAggregation:
            periods = esM.typicalPeriods
            if esM.segmentation:
                timeSteps = esM.segmentsPerPeriod
            else:
                timeSteps = esM.timeStepsPerPeriod
        else:
            periods = esM.periods
            timeSteps = esM.totalTimeSteps
        # Check if locational input is not set as Total in esM, if so additionally loop over all locations
        if loc == "Total":
            balance = sum(
                opVar[_loc, compName, ip, p, t]
                * compDict[compName].sign
                * esM.periodOccurrences[ip][p]
                for compName in compDict.keys()
                if compName in componentNames
                for p in periods
                for t in timeSteps
                for _loc in esM.locations
            )
        # Otherwise get the contribution for specific region
        else:
            balance = sum(
                opVar[loc, compName, ip, p, t]
                * compDict[compName].sign
                * esM.periodOccurrences[ip][p]
                for compName in compDict.keys()
                if compName in componentNames
                for p in periods
                for t in timeSteps
            )
        return balance

    def getCommodityBalanceContribution(self, pyM, commod, loc, ip, p, t):
        """Get contribution to a commodity balance.
                .. math::

            \\text{C}^{comp,comm}_{loc,ip,p,t} = - op_{loc,ip,p,t}^{comp,op}  \\text{Sink}

        .. math::
            \\text{C}^{comp,comm}_{loc,ip,p,t} = op_{loc,ip,p,t}^{comp,op} \\text{Source}
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDict = (
            getattr(pyM, "op_" + abbrvName),
            getattr(pyM, "operationVarDict_" + abbrvName),
        )
        return sum(
            opVar[loc, compName, ip, p, t] * compDict[compName].sign
            for compName in opVarDict[ip][loc]
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

        opexOp = self.getEconomicsOperation(
            pyM, esM, "TD", ["processedOpexPerOperation"], "op", "operationVarDict"
        )
        commodCost = self.getEconomicsOperation(
            pyM, esM, "TD", ["processedCommodityCost"], "op", "operationVarDict"
        )
        commodRevenue = self.getEconomicsOperation(
            pyM, esM, "TD", ["processedCommodityRevenue"], "op", "operationVarDict"
        )
        commodCostTimeSeries = self.getEconomicsOperation(
            pyM,
            esM,
            "TimeSeries",
            ["processedCommodityCostTimeSeries"],
            "op",
            "operationVarDict",
        )
        commodRevenueTimeSeries = self.getEconomicsOperation(
            pyM,
            esM,
            "TimeSeries",
            ["processedCommodityRevenueTimeSeries"],
            "op",
            "operationVarDict",
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

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """
        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(
            esM, pyM, esM.locations, "commodityUnit"
        )

        # get class related results
        resultsTAC_opexOp = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerOperation"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="TAC",
        )
        resultsTAC_commodCost = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedCommodityCost"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="TAC",
        )
        resultsTAC_commodRevenue = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedCommodityRevenue"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="TAC",
        )
        resultsTAC_commodCostTimeSeries = self.getEconomicsOperation(
            pyM,
            esM,
            "TimeSeries",
            ["processedCommodityCostTimeSeries"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="TAC",
        )
        resultsTAC_commodRevenueTimeSeries = self.getEconomicsOperation(
            pyM,
            esM,
            "TimeSeries",
            ["processedCommodityRevenueTimeSeries"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="TAC",
        )
        resultsNPV_opexOp = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerOperation"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="NPV",
        )
        resultsNPV_commodCost = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedCommodityCost"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="NPV",
        )
        resultsNPV_commodRevenue = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedCommodityRevenue"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="NPV",
        )
        resultsNPV_commodCostTimeSeries = self.getEconomicsOperation(
            pyM,
            esM,
            "TimeSeries",
            ["processedCommodityCostTimeSeries"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="NPV",
        )
        resultsNPV_commodRevenueTimeSeries = self.getEconomicsOperation(
            pyM,
            esM,
            "TimeSeries",
            ["processedCommodityRevenueTimeSeries"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="NPV",
        )

        for ip in esM.investmentPeriods:
            compDict, abbrvName = self.componentsDict, self.abbrvName
            opVar = getattr(pyM, "op_" + abbrvName)

            # Set optimal operation variables and append optimization summary
            optVal = utils.formatOptimizationOutput(
                opVar.get_values(),
                "operationVariables",
                "1dim",
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )

            self._operationVariablesOptimum[esM.investmentPeriodNames[ip]] = optVal

            props = [
                "operation",
                "opexOp",
                "commodCosts",
                "commodRevenues",
                "NPV_opexOp",
                "NPV_commodCosts",
                "NPV_commodRevenues",
            ]
            # Unit dict: Specify units for props
            units = {
                props[0]: ["[-*h]", "[-*h/a]"],
                props[1]: ["[" + esM.costUnit + "/a]"],
                props[2]: ["[" + esM.costUnit + "/a]"],
                props[3]: ["[" + esM.costUnit + "/a]"],
                props[4]: ["[" + esM.costUnit + "/a]"],
                props[5]: ["[" + esM.costUnit + "/a]"],
                props[6]: ["[" + esM.costUnit + "/a]"],
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
                    lambda x: (
                        x[0],
                        x[1],
                        x[2].replace("-", compDict[x[0]].commodityUnit),
                    )
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
                # operation
                opSum = optVal.sum(axis=1).unstack(-1)
                optSummary.loc[
                    [
                        (ix, "operation", "[" + compDict[ix].commodityUnit + "*h]")
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = opSum.values
                optSummary.loc[
                    [
                        (ix, "operation", "[" + compDict[ix].commodityUnit + "*h/a]")
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = (
                    opSum.values / esM.numberOfYears
                )

                # costs
                tac_ox = resultsTAC_opexOp[ip]
                tac_cCost = resultsTAC_commodCost[ip]
                tac_cRevenue = resultsTAC_commodRevenue[ip]
                tac_cCostTimeSeries = resultsTAC_commodCostTimeSeries[ip]
                tac_cRevenueTimeSeries = resultsTAC_commodRevenueTimeSeries[ip]

                npv_ox = resultsNPV_opexOp[ip]
                npv_cCost = resultsNPV_commodCost[ip]
                npv_cRevenue = resultsNPV_commodRevenue[ip]
                npv_cCostTimeSeries = resultsNPV_commodCostTimeSeries[ip]
                npv_cRevenueTimeSeries = resultsNPV_commodRevenueTimeSeries[ip]

                optSummary.loc[
                    [(ix, "opexOp", "[" + esM.costUnit + "/a]") for ix in tac_ox.index],
                    tac_ox.columns,
                ] = tac_ox.values
                optSummary.loc[
                    [
                        (ix, "NPV_opexOp", "[" + esM.costUnit + "/a]")
                        for ix in npv_ox.index
                    ],
                    npv_ox.columns,
                ] = npv_ox.values

                # costs: commodity costs
                tac_commodCosts = tac_cCostTimeSeries + tac_cCost
                optSummary.loc[
                    [
                        (ix, "commodCosts", "[" + esM.costUnit + "/a]")
                        for ix in tac_commodCosts.index
                    ],
                    tac_commodCosts.columns,
                ] = tac_commodCosts.values

                npv_commodCosts = npv_cCostTimeSeries + npv_cCost
                optSummary.loc[
                    [
                        (ix, "NPV_commodCosts", "[" + esM.costUnit + "/a]")
                        for ix in npv_commodCosts.index
                    ],
                    npv_commodCosts.columns,
                ] = npv_commodCosts.values

                # costs: commodity revenues
                tac_commodRevenue = tac_cRevenueTimeSeries + tac_cRevenue
                optSummary.loc[
                    [
                        (ix, "commodRevenues", "[" + esM.costUnit + "/a]")
                        for ix in tac_commodRevenue.index
                    ],
                    tac_commodRevenue.columns,
                ] = tac_commodRevenue.values

                npv_commodRevenue = npv_cRevenueTimeSeries + npv_cRevenue
                optSummary.loc[
                    [
                        (ix, "NPV_commodRevenues", "[" + esM.costUnit + "/a]")
                        for ix in npv_commodRevenue.index
                    ],
                    npv_commodRevenue.columns,
                ] = npv_commodRevenue.values

            # get discounted investment cost as total annual cost (TAC)
            optSummaryBasic_frame = optSummaryBasic[esM.investmentPeriodNames[ip]]
            if isinstance(optSummaryBasic_frame, pd.Series):
                optSummaryBasic_frame = optSummaryBasic_frame.to_frame().T

            optSummary = pd.concat(
                [
                    optSummary,
                    optSummaryBasic_frame,
                ],
                axis=0,
            ).sort_index()

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
                - optSummary.loc[
                    (optSummary.index.get_level_values(1) == "commodRevenues")
                ]
                .groupby(level=0)
                .sum()
                .values
            )
            # add operation specific contributions to the net present value (NPV) and substract revenues
            optSummary.loc[
                optSummary.index.get_level_values(1) == "NPVcontribution"
            ] = (
                optSummary.loc[
                    (optSummary.index.get_level_values(1) == "NPVcontribution")
                    | (optSummary.index.get_level_values(1) == "NPV_opexOp")
                    | (optSummary.index.get_level_values(1) == "NPV_commodCosts")
                ]
                .groupby(level=0)
                .sum()
                .values
                - optSummary.loc[
                    (optSummary.index.get_level_values(1) == "NPV_commodRevenues")
                ]
                .groupby(level=0)
                .sum()
                .values
            )

            # Delete details of NPV contributions
            optSummary = optSummary.drop("NPV_opexOp", level=1)
            optSummary = optSummary.drop("NPV_commodCosts", level=1)
            optSummary = optSummary.drop("NPV_commodRevenues", level=1)

            self._optSummary[esM.investmentPeriodNames[ip]] = optSummary

    def getOptimalValues(self, name="all", ip=0):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariables',
            * 'isBuiltVariables',
            * '_operationVariablesOptimum',
            * 'all' or another input: all variables are returned.

        |br| * the default value is 'all'
        :type name: string

        :param ip: investment period
            |br| * the default value is 0
        :type ip: int

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        return super().getOptimalValues(name, ip=ip)
