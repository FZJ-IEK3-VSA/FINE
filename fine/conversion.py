from fine.component import Component, ComponentModel
from fine import utils
import warnings
import pandas as pd
import pyomo.environ as pyomo


class Conversion(Component):
    """
    A Conversion component converts commodities into each other.
    """

    def __init__(
        self,
        esM,
        name,
        physicalUnit,
        commodityConversionFactors,
        hasCapacityVariable=True,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=1,
        linkedConversionCapacityID=None,
        hasIsBuiltBinaryVariable=False,
        bigM=None,
        operationRateMin=None,
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
        commissioningMin=None,
        commissioningMax=None,
        commissioningFix=None,
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
        yearlyFullLoadHoursMin=None,
        yearlyFullLoadHoursMax=None,
        stockCommissioning=None,
        floorTechnicalLifetime=True,
        commissioningDependentCcf=False,
        emissionFactors=None,
        flowShares=None,
    ):
        # TODO: allow that the time series data or min/max/fixCapacity/eligibility is only specified for
        # TODO: eligible locations
        """
        Constructor for creating an instance of the Conversion class. Capacities are given in the physical unit
        of the plants.
        The Conversion component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param physicalUnit: reference physical unit of the plants to which maximum capacity limitations,
            cost parameters and the operation time series refer to.
        :type physicalUnit: string

        :param commodityConversionFactors: conversion factors with which commodities are converted into each
            other with one unit of operation (dictionary). Each commodity which is converted in this component
            is indicated by a string in this dictionary. The conversion factor related to this commodity is
            given as a float (constant), pandas.Series or pandas.DataFrame (time-variable). A negative value
            indicates that the commodity is consumed. A positive value indicates that the commodity is produced.
            Check unit consistency when specifying this parameter!

            Examples:

            * An electrolyzer converts, simply put, electricity into hydrogen with an electrical efficiency
                of 70%. The physicalUnit is given as GW_electric, the unit for the 'electricity' commodity is
                given in GW_electric and the 'hydrogen' commodity is given in GW_hydrogen_lowerHeatingValue
                -> the commodityConversionFactors are defined as {'electricity':-1,'hydrogen':0.7}.
            * A fuel cell converts, simply put, hydrogen into electricity with an efficiency of 60%.
                The physicalUnit is given as GW_electric, the unit for the 'electricity' commodity is given in
                GW_electric and the 'hydrogen' commodity is given in GW_hydrogen_lowerHeatingValue
                -> the commodityConversionFactors are defined as {'electricity':1,'hydrogen':-1/0.6}.

            If a transformation pathway analysis is performed the conversion factors can also be varied
            over the transformation pathway. Therefore, two different options are available:

            1. Variation with operation year (for example to incorporate weather changes for a heat pump).
               Example:
               {2020: {'electricity':-1,'heat':pd.Series(data=[2.5, 2.8, 2.5, ...])},
               2025: {'electricity':-1,'heat':pd.Series(data=[2.7, 2.4, 2.9, ...])},
               ...}
            2. Variation with commissioning and operation year (for example to incorporate efficiency
               changes dependent on the installation year). Please note that this implementation massively
               increases the complexity of the optimization problem.
               Example:
               {(2020, 2020): {'electricity':-1,'heat':pd.Series(data=[2.5, 2.8, 2.5, ...])},
               (2020, 2025): {'electricity':-1,'heat':pd.Series(data=[2.7, 2.4, 2.9, ...])},
               (2025, 2025): {'electricity':-1,'heat':pd.Series(data=[3.7, 3.4, 3.9, ...])},
               ...}

            If a conversion component can decide between multiple in- or outputs which one to use
            (e.g. a chp plant) a flexible conversion component can be specified. This enables the
            component to substitute in- or output commodities within a commodity group. To allow
            this behavior an additional level needs to be specified:

            * A CHP plant can decide between the production of heat or electricity (or a mix of both).
                When electricity is produced the conversion factor is 0.2 and for heat 0.5:
                {'gas': -1, 'out': {electricity: 0.2, heat: 0.5}}

        :type commodityConversionFactors:

            * dictionary, assigns commodities (string) to a conversion factors
              (float, pandas.Series or pandas.DataFrame)
            * dictionary with investment periods as key and one of the first option  as value
            * dictionary with tuple of (commissioning year, investment period) as key and one of the first option above as value

        **Default arguments:**

        :param linkedConversionCapacityID: if specifies, indicates that all conversion components with the
            same ID have to have the same capacity.
            |br| * the default value is None
        :type linkedConversionCapacityID: string

        :param operationRateMin: if specified, indicates a minimum operation rate for each location and each time
            step, if required also for each investment period, by a positive float. If hasCapacityVariable is set
            to True, the values are given relative to the installed capacities (i.e. a value of 1 indicates a
            utilization of 100% of the capacity). If hasCapacityVariable is set to False, the values are given as
            absolute values in form of the physicalUnit of the plant for each time step.
            |br| * the default value is None
        :type operationRateMin:
            * None
            * pandas DataFrame with positive (>=0). The row indices have
              to match the in the energy system model specified time steps. The column indices have to match the
              in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param operationRateMax: if specified, indicates a maximum operation rate for each location and each time
            step, if required also for each investment period, by a positive float. If hasCapacityVariable is set
            to True, the values are given relative to the installed capacities (i.e. a value of 1 indicates a
            utilization of 100% of the capacity). If hasCapacityVariable is set to False, the values are given as
            absolute values in form of the physicalUnit of the plant for each time step.
            |br| * the default value is None
        :type operationRateMax:
            * None
            * pandas DataFrame with positive (>=0). The row indices have
              to match the in the energy system model specified time steps. The column indices have to match the
              in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param operationRateFix: if specified, indicates a fixed operation rate for each location and each time
            step, if required also for each investment period, by a positive float. If hasCapacityVariable is set
            to True, the values are given relative to the installed capacities (i.e. a value of 1 indicates a
            utilization of 100% of the capacity). If hasCapacityVariable is set to False, the values are given as
            absolute values in form of the physicalUnit of the plant for each time step.
            |br| * the default value is None
        :type operationRateFix:
            * None
            * Pandas DataFrame with positive (>=0). The row indices have
              to match the in the energy system model specified time steps. The column indices have to match the
              in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param opexPerOperation: describes the cost for one unit of the operation. The cost which is
            directly proportional to the operation of the component is obtained by multiplying
            the opexPerOperation parameter with the annual sum of the operational time series of the components.
            The opexPerOperation can either be given as a float or a Pandas Series with location specific values.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerOperation:
            * Pandas Series with positive (>=0) entries. The indices of the series have to equal the in the energy
              system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param commissioningDependentCcf: specifies if commodity conversion factors are dependent on commissioning
            or operation year. If set to False, the factors are only dependent on the year of operation and
            no new operation variables are introduced. If set to True, the factors are dependent on commissioning
            year and new operation variables are introduced for every commissioning year.
            |br| * the default value is False
        :type commissioningDependentCcf:bool

        :param emissionFactors: can be used to specify emissions for flexible conversion components.
            This parameter can only be specified if the component is a flexible conversion component (see explanations
            on commodity conversion factors above). When specified, the emission factors indicate what emissions
            are produced when a particular commodity is used by the component.
            Note: For non-flexible conversion components emissions must be specified as commodity conversion factors.

            Example: The CO2 emissions of a power plant are dependent on the type of fuel is used (e.g. coal has
            higher emissions than gas): {'co2': {'coal': 3, 'gas': 1}}
        :type emissionFactors: dict with emission commodities as key and a dict as value. The inner dict holds the
            emission factors which are dependent on the utilized commodity.

        :param flowShares: can be used to constrain the operation of flexible conversion components (see explanations
            on commodity conversion factors above). When used, the flow shares must be specified as 'min', 'max', or
            'fix' values that limit the commodity specific operation rate of a flexible conversion component relative
            to the overall rate of that component (e.g. if the flow share max for H2 is set to 0.75 and the overall
            operation rate is 4 MW, then the H2 operation rate must be smaller or equal to 3 MW). Flow shares can be
            set up for all, some, or none of the modeled investment periods, and can either apply to all regions
            (if defined as int) or depend on individual regions (if defined as pandas series). Flow shares must be
            between 0 and 1 if specified.

            Example: In the first investment period in Location1 only a small share of 10 % of the modeled gas heaters
            are able to burn hydrogen instead of natural gas. In the second investment period more hydrogen ready
            heaters are available and 50% of the heaters can burn hydrogen instead of natural gas (10 % of
            those heaters can only burn hydrogen). In Location2 only 5 % can burn hydrogen in first period and 40 % can
            burn hydrogen in second period:
            flowShares = {
                0: {
                    'max': {'hydrogen': pd.Series([0.1, 0.05], index=['loc1', 'loc2'])}
                },
                1: {
                    'max': {'hydrogen': pd.Series([0.1, 0.05], index=['loc1', 'loc2'])},
                    'min': {'hydrogen': pd.Series([0.1], index=['loc1'])}
                }
            }
        :type flowShares: dict
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
            commissioningMin=commissioningMin,
            commissioningMax=commissioningMax,
            commissioningFix=commissioningFix,
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

        # check for operationRateMax and operationRateFix
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            if esM.verbose < 2:
                warnings.warn(
                    "If operationRateFix is specified, the operationRateMax parameter is not required.\n"
                    + "The operationRateMax time series was set to None."
                )
        if operationRateMin is not None and operationRateFix is not None:
            operationRateMin = None
            if esM.verbose < 2:
                warnings.warn(
                    "If operationRateFix is specified, the operationRateMin parameter is not required.\n"
                    + "The operationRateMin time series was set to None."
                )
        # if both operationRateMin and operationRateMax are given, check if operationRateMin <= operationRateMax
        if operationRateMin is not None and operationRateMax is not None:
            if not (operationRateMax >= operationRateMin).all():
                # raise error
                raise ValueError(
                    "The operationRateMin time series has to be smaller or equal to the operationRateMax time series."
                )

        # operationRateMin
        self.operationRateMin = operationRateMin
        self.fullOperationRateMin = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, operationRateMin, locationalEligibility
        )
        self.aggregatedOperationRateMin = {}
        self.processedOperationRateMin = {}

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

        # partLoadMin
        self.processedPartLoadMin = utils.checkAndSetPartLoadMin(
            esM,
            name,
            self.partLoadMin,
            self.fullOperationRateMax,
            self.fullOperationRateFix,
            self.bigM,
            self.hasCapacityVariable,
            fullOperationMin=self.fullOperationRateMin,
        )

        # commodity conversions factors
        self.commissioningDependentCcf = commissioningDependentCcf
        self.commodityConversionFactors = commodityConversionFactors
        (
            self.isIpDepending,
            self.isCommisDepending,
            self.flexibleConversion
        ) = utils.checkConversionFactorProperties(self, esM, commissioningDependentCcf)
        (
            self.fullCommodityConversionFactors,
            self.processedCommodityConversionFactors,
            self.preprocessedCommodityConversionFactors,
        ) = utils.checkAndSetCommodityConversionFactor(self, esM)
        self.aggregatedCommodityConversionFactors = dict.fromkeys(
            self.fullCommodityConversionFactors.keys()
        )

        self.emissionFactors = emissionFactors
        self.processedEmissionFactors = utils.checkEmissionFactors(self, esM)

        self.flowShares = flowShares
        self.processedFlowShares = utils.checkAndSetFlowShares(self, esM)

        utils.isPositiveNumber(tsaWeight)
        self.tsaWeight = tsaWeight
        utils.checkCommodityUnits(esM, physicalUnit)
        if linkedConversionCapacityID is not None:
            utils.isString(linkedConversionCapacityID)
        self.physicalUnit = physicalUnit
        self.modelingClass = ConversionModel
        self.linkedConversionCapacityID = linkedConversionCapacityID

        if any(x is not None for x in self.fullOperationRateFix.values()):
            operationTimeSeries = self.fullOperationRateFix
        elif any(x is not None for x in self.fullOperationRateMax.values()):
            operationTimeSeries = self.fullOperationRateMax
        elif any(x is not None for x in self.fullOperationRateMin.values()):
            operationTimeSeries = self.fullOperationRateMin
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
        Function for setting the maximum operation rate and fixed operation rate depending on whether a time series
        analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        # processedOperationMax
        self.processedOperationRateMax = (
            self.aggregatedOperationRateMax if hasTSA else self.fullOperationRateMax
        )
        # processedOperationFix
        self.processedOperationRateFix = (
            self.aggregatedOperationRateFix if hasTSA else self.fullOperationRateFix
        )
        # processedOperationMin
        self.processedOperationRateMin = (
            self.aggregatedOperationRateMin if hasTSA else self.fullOperationRateMin
        )
        # processedCommodityConversions
        # timeInfo can either be ip or (commis,ip)
        for timeInfo in self.fullCommodityConversionFactors.keys():
            if self.fullCommodityConversionFactors[timeInfo] != {}:
                for commod in self.fullCommodityConversionFactors[timeInfo]:
                    self.processedCommodityConversionFactors[timeInfo][commod] = (
                        self.aggregatedCommodityConversionFactors[timeInfo][commod]
                        if hasTSA
                        else self.fullCommodityConversionFactors[timeInfo][commod]
                    )

    def getDataForTimeSeriesAggregation(self, ip):
        """Function for getting the required data if a time series aggregation is requested.

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """
        weightDict, data = {}, []
        if self.fullOperationRateFix:
            weightDict, data = self.prepareTSAInput(
                self.fullOperationRateFix,
                "_operationRateFix_",
                self.tsaWeight,
                weightDict,
                data,
                ip,
            )
        if self.fullOperationRateMin:
            weightDict, data = self.prepareTSAInput(
                self.fullOperationRateMin,
                "_operationRateMin_",
                self.tsaWeight,
                weightDict,
                data,
                ip,
            )
        if self.fullOperationRateMax:
            weightDict, data = self.prepareTSAInput(
                self.fullOperationRateMax,
                "_operationRateMax_",
                self.tsaWeight,
                weightDict,
                data,
                ip,
            )

        if not self.isCommisDepending:
            for commod in self.fullCommodityConversionFactors[ip]:
                weightDict, data = self.prepareTSAInput(
                    self.fullCommodityConversionFactors[ip][commod],
                    "_commodityConversionFactorTimeSeries" + str(commod) + "_",
                    self.tsaWeight,
                    weightDict,
                    data,
                    ip,
                )
        # for components with conversion time-series depending on commissioning years,
        # the time-series of all commissioning years relevant for the investment period must be considered
        else:
            relevantCommissioningYears = [
                x for (x, y) in self.fullCommodityConversionFactors.keys() if y == ip
            ]
            for commisYear in relevantCommissioningYears:
                # divide the weight by the number of relevant commissioning years to
                # prevent a too high total weight of the commodity conversion time-series
                for commod in self.fullCommodityConversionFactors[(commisYear, ip)]:
                    weightDict, data = self.prepareTSAInput(
                        self.fullCommodityConversionFactors[(commisYear, ip)][commod],
                        "_commodityConversionFactorTimeSeries"
                        + str(commod)
                        + str(commisYear).replace("-", "minus")
                        + "_",
                        self.tsaWeight / len(relevantCommissioningYears),
                        weightDict,
                        data,
                        ip,
                    )
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data, ip):
        """
        Function for determining the aggregated maximum rate and the aggregated fixed operation rate.

        :param data: Pandas DataFrame with the clustered time series data of the conversion component
        :type data: Pandas DataFrame

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """
        self.aggregatedOperationRateFix[ip] = self.getTSAOutput(
            self.fullOperationRateFix, "_operationRateFix_", data, ip
        )
        self.aggregatedOperationRateMin[ip] = self.getTSAOutput(
            self.fullOperationRateMin, "_operationRateMin_", data, ip
        )
        self.aggregatedOperationRateMax[ip] = self.getTSAOutput(
            self.fullOperationRateMax, "_operationRateMax_", data, ip
        )

        # get the aggregated commodity conversion factors
        # The procedure changes depending on whether the commodity conversion factors depend on the year of
        # commissioning or only on the year of operation (investment period).
        if not self.isCommisDepending:
            if self.fullCommodityConversionFactors[ip] != {}:
                self.aggregatedCommodityConversionFactors[ip] = {}
                for commod in self.fullCommodityConversionFactors[ip]:
                    self.aggregatedCommodityConversionFactors[ip][commod] = (
                        self.getTSAOutput(
                            self.fullCommodityConversionFactors[ip][commod],
                            "_commodityConversionFactorTimeSeries" + str(commod) + "_",
                            data,
                            ip,
                        )
                    )
        else:
            # if depending on the commissioning year, iterate over the relevant commissioning years for the
            # operation of the investment period ip and get the time-series
            relevantCommissioningYears = [
                x for (x, y) in self.fullCommodityConversionFactors.keys() if y == ip
            ]
            for commisYear in relevantCommissioningYears:
                if self.fullCommodityConversionFactors[(commisYear, ip)] != {}:
                    self.aggregatedCommodityConversionFactors[(commisYear, ip)] = {}
                    for commod in self.fullCommodityConversionFactors[(commisYear, ip)]:
                        self.aggregatedCommodityConversionFactors[(commisYear, ip)][
                            commod
                        ] = self.getTSAOutput(
                            self.fullCommodityConversionFactors[(commisYear, ip)][
                                commod
                            ],
                            "_commodityConversionFactorTimeSeries"
                            + str(commod)
                            + str(commisYear).replace("-", "minus")
                            + "_",
                            data,
                            ip,
                        )


class ConversionModel(ComponentModel):
    """
    A ConversionModel class instance will be instantly created if a Conversion class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Conversion class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The ConversionModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        """ " Constructor for creating a ConversionModel class instance"""
        super().__init__()
        self.abbrvName = "conv"
        self.dimension = "1dim"
        self._operationVariablesOptimum = {}

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareLinkedCapacityDict(self, pyM):
        """
        Declare conversion components with linked capacities and check if the linked components have the same
        locational eligibility.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        linkedComponentsDict, linkedComponentsList, compDict = (
            {},
            [],
            self.componentsDict,
        )
        # Collect all conversion components with the same linkedConversionComponentID
        for comp in compDict.values():
            if comp.linkedConversionCapacityID is not None:
                linkedComponentsDict.setdefault(
                    comp.linkedConversionCapacityID, []
                ).append(comp)
        # Pair the components with the same linkedConversionComponentID with each other and check that
        # they have the same locational eligibility
        for key, values in linkedComponentsDict.items():
            if len(values) > 1:
                linkedComponentsList.extend(
                    [
                        (loc, values[i].name, values[i + 1].name)
                        for i in range(len(values) - 1)
                        for loc, v in values[i].processedLocationalEligibility.items()
                        if v == 1
                    ]
                )
        for comps in linkedComponentsList:
            index1 = compDict[comps[1]].processedLocationalEligibility.index
            index2 = compDict[comps[2]].processedLocationalEligibility.index
            if not index1.equals(index2):
                raise ValueError(
                    "Conversion components ",
                    comps[1],
                    "and",
                    comps[2],
                    "are linked but do not have the same locationalEligibility.",
                )
        setattr(pyM, "linkedComponentsList_" + self.abbrvName, linkedComponentsList)

    def declareOpCommisVarSet(self, esM, pyM):
        """
        Declare the operation set for components that have commodity conversion factors that depend on the
        year in the pyomo object for a modeling class.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        commisDependingComp = [
            compName for (compName, comp) in compDict.items() if comp.isCommisDepending
        ]

        # Set for operation variables
        def declareOpCommisVarSet(pyM):
            return (
                (loc, compName, commis, ip)
                for compName in commisDependingComp
                for loc in compDict[compName].processedLocationalEligibility.index
                for (commis, ip) in compDict[
                    compName
                ].processedCommodityConversionFactors.keys()
                if compDict[compName].processedLocationalEligibility[loc] == 1
            )

        setattr(
            pyM,
            "operationCommisVarSet_" + abbrvName,
            pyomo.Set(dimen=4, initialize=declareOpCommisVarSet),
        )

    def declareOpFlexVarSets(self, esM, pyM):
        """
        Declare commodity specific operation variable set for flexible conversion components in the pyomo object
        for a modeling class.
        """
        compDict = self.componentsDict
        flexConv = [
            compName for (compName, comp) in compDict.items() if comp.flexibleConversion
        ]

        # Set which holds commodity groups for all flexible components, investment periods and locations
        def declareOpFlexGroupSet(pyM):
            return (
                (loc, compName, ip, group)
                for compName in flexConv
                for loc in compDict[compName].processedLocationalEligibility.index
                for ip in esM.investmentPeriods
                for group, group_ccf in compDict[compName].processedCommodityConversionFactors[ip].items()
                if isinstance(group_ccf, dict)
                if compDict[compName].processedLocationalEligibility[loc] == 1
            )

        setattr(
            pyM,
            "operationFlexGroupSet_" + self.abbrvName,
            pyomo.Set(dimen=4, initialize=declareOpFlexGroupSet),
        )

        # Set for flexible operation variables including commodity group
        def declareOpFlexVarSet(pyM):
            return (
                (loc, compName, ip, group, commod)
                for (loc, compName, ip, group) in getattr(pyM, "operationFlexGroupSet_" + self.abbrvName)
                for commod in compDict[compName].processedCommodityConversionFactors[ip][group].keys()
            )

        setattr(
            pyM,
            "operationFlexVarSet_" + self.abbrvName,
            pyomo.Set(dimen=5, initialize=declareOpFlexVarSet),
        )

    def declareFlexFlowShareConstrSet(self, pyM):
        """
        Declare set for flow share constraints based on the processed flow shares parameter.
        """

        def declareOpFlexFlowShareConstrSet(pyM):
            return (
                (loc, compName, ip, group, attr, commod)
                for loc, compName, ip, group in getattr(pyM, "operationFlexGroupSet_" + self.abbrvName)
                if self.componentsDict[compName].processedFlowShares
                if ip in self.componentsDict[compName].processedFlowShares.keys()
                for attr, fs in self.componentsDict[compName].processedFlowShares[ip].items()
                for commod in fs.keys()
                if loc in fs[commod].index
            )

        setattr(
            pyM,
            "operationFlexFlowShareConstrSet_" + self.abbrvName,
            pyomo.Set(dimen=6, initialize=declareOpFlexFlowShareConstrSet),
        )

    def declareOpCommisConstrSet1(self, pyM, constrSetName, rateMax, rateFix, rateMin):
        """
        Declare set of locations and components for which hasCapacityVariable is set to True and neither the
        maximum nor the fixed operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationCommisVarSet_" + abbrvName)

        def declareOpCommisConstrSet1(pyM):
            return (
                (loc, compName, commis, ip)
                for loc, compName, commis, ip in varSet
                if compDict[compName].hasCapacityVariable
                and getattr(compDict[compName], rateMax) is None
                and getattr(compDict[compName], rateFix) is None
                and getattr(compDict[compName], rateMin) is None
                and compDict[compName].isCommisDepending
            )

        setattr(
            pyM,
            constrSetName + "1_" + abbrvName,
            pyomo.Set(dimen=4, initialize=declareOpCommisConstrSet1),
        )

    def declareOpCommisConstrSet2(self, pyM, constrSetName, rateFix):
        """
        Declare set of locations and components for which hasCapacityVariable is set to True and a fixed
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationCommisVarSet_" + abbrvName)

        def declareOpCommisConstrSet2(pyM):
            return (
                (loc, compName, commis, ip)
                for loc, compName, commis, ip in varSet
                if compDict[compName].hasCapacityVariable
                and getattr(compDict[compName], rateFix)[ip] is not None
                and compDict[compName].isCommisDepending
            )

        setattr(
            pyM,
            constrSetName + "2_" + abbrvName,
            pyomo.Set(dimen=4, initialize=declareOpCommisConstrSet2),
        )

    def declareOpCommisConstrSet3(self, pyM, constrSetName, rateMax):
        """
        Declare set of locations and components for which  hasCapacityVariable is set to True and a maximum
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationCommisVarSet_" + abbrvName)

        def declareOpCommisConstrSet3(pyM):
            return (
                (loc, compName, commis, ip)
                for loc, compName, commis, ip in varSet
                if compDict[compName].hasCapacityVariable
                and getattr(compDict[compName], rateMax)[ip] is not None
                and compDict[compName].isCommisDepending
            )

        setattr(
            pyM,
            constrSetName + "3_" + abbrvName,
            pyomo.Set(dimen=4, initialize=declareOpCommisConstrSet3),
        )

    def declareOpCommisConstrSet4(self, pyM, constrSetName, rateMin):
        """
        Declare set of locations and components for which  hasCapacityVariable is set to True and a minimum
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationCommisVarSet_" + abbrvName)

        def declareOpCommisConstrSet4(pyM):
            return (
                (loc, compName, commis, ip)
                for loc, compName, commis, ip in varSet
                if compDict[compName].hasCapacityVariable
                and getattr(compDict[compName], rateMin)[ip] is not None
                and compDict[compName].isCommisDepending
            )

        setattr(
            pyM,
            constrSetName + "4_" + abbrvName,
            pyomo.Set(dimen=4, initialize=declareOpCommisConstrSet4),
        )

    def declareOpCommisConstrSetMinPartLoad(self, pyM, constrSetName):
        """
        Declare set of locations and components for which partLoadMin is not None.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationCommisVarSet_" + abbrvName)

        def declareOpCommisConstrSetMinPartLoad(pyM):
            return (
                (loc, compName, commis, ip)
                for loc, compName, commis, ip in varSet
                if getattr(compDict[compName], "processedPartLoadMin") is not None
                and compDict[compName].isCommisDepending
            )

        setattr(
            pyM,
            constrSetName + "partLoadMin_" + abbrvName,
            pyomo.Set(dimen=4, initialize=declareOpCommisConstrSetMinPartLoad),
        )

    def declareYearlyFullLoadHoursCommisMinSet(self, pyM):
        """
        Declare set of locations and components for which minimum yearly full load hours are given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationCommisVarSet_" + abbrvName)

        def declareYearlyFullLoadHoursCommisMinSet():
            return (
                (loc, compName, commis, ip)
                for loc, compName, commis, ip in varSet
                if compDict[compName].processedYearlyFullLoadHoursMin[ip] is not None
                and compDict[compName].isCommisDepending
            )

        setattr(
            pyM,
            "yearlyFullLoadHoursCommisMinSet_" + abbrvName,
            pyomo.Set(dimen=4, initialize=declareYearlyFullLoadHoursCommisMinSet()),
        )

    def declareYearlyFullLoadHoursCommisMaxSet(self, pyM):
        """
        Declare set of locations and components for which maximum yearly full load hours are given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationCommisVarSet_" + abbrvName)

        def declareYearlyFullLoadHoursCommisMaxSet():
            return (
                (loc, compName, commis, ip)
                for loc, compName, commis, ip in varSet
                if compDict[compName].processedYearlyFullLoadHoursMax[ip] is not None
                and compDict[compName].isCommisDepending
            )

        setattr(
            pyM,
            "yearlyFullLoadHoursCommisMaxSet_" + abbrvName,
            pyomo.Set(dimen=4, initialize=declareYearlyFullLoadHoursCommisMaxSet()),
        )

    def declareOperationModeSets(self, pyM, constrSetName, rateMax, rateFix, rateMin):
        super().declareOperationModeSets(pyM, constrSetName, rateMax, rateFix, rateMin)
        self.declareOpCommisConstrSet1(
            pyM, "opCommisConstrSet", rateMax, rateFix, rateMin
        )
        self.declareOpCommisConstrSet2(pyM, "opCommisConstrSet", rateFix)
        self.declareOpCommisConstrSet3(pyM, "opCommisConstrSet", rateMax)
        self.declareOpCommisConstrSet4(pyM, "opCommisConstrSet", rateMin)
        self.declareOpCommisConstrSetMinPartLoad(pyM, "opCommisConstrSet")

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
        self.declareOpFlexVarSets(esM, pyM)
        self.declareOpCommisVarSet(esM, pyM)
        self.declareFlexFlowShareConstrSet(pyM)

        # Declare operation mode sets
        self.declareOperationModeSets(
            pyM,
            "opConstrSet",
            "processedOperationRateMax",
            "processedOperationRateFix",
            "processedOperationRateMin",
        )

        # Declare linked components dictionary
        self.declareLinkedCapacityDict(pyM)

        # Declare minimum yearly full load hour set
        self.declareYearlyFullLoadHoursMinSet(pyM)
        self.declareYearlyFullLoadHoursCommisMinSet(pyM)

        # Declare maximum yearly full load hour set
        self.declareYearlyFullLoadHoursMaxSet(pyM)
        self.declareYearlyFullLoadHoursCommisMaxSet(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary, relevanceThreshold):
        """
        Declare design and operation variables

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

        # Capacity variables [physicalUnit]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not
        self.declareBinaryDesignDecisionVars(pyM, relaxIsBuiltBinary)
        # Operation of component [physicalUnit*hour]
        self.declareOperationVars(pyM, esM, "op", relevanceThreshold=relevanceThreshold)
        self.declareOperationVars(
            pyM,
            esM,
            "op_commis",
            relevanceThreshold=relevanceThreshold,
            isOperationCommisYearDepending=True,
        )
        # Flexible conversion operation variables that are also indexed over the commodity
        self.declareOperationVars(
            pyM,
            esM,
            "op_flex",
            relevanceThreshold=relevanceThreshold,
            flexibleConversion=True,
        )
        # Operation of component as binary [1/0]
        self.declareOperationBinaryVars(pyM, "op_bin")
        # Capacity development variables [physicalUnit]
        self.declareCommissioningVars(pyM, esM)
        self.declareDecommissioningVars(pyM, esM)

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def linkedCapacity(self, pyM):
        """
        Ensure that all Conversion components with the same linkedConversionCapacityID have the same capacity

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName = self.abbrvName
        capVar, linkedList = (
            getattr(pyM, "cap_" + abbrvName),
            getattr(pyM, "linkedComponentsList_" + self.abbrvName),
        )

        def linkedCapacity(pyM, loc, compName1, compName2, ip):
            return capVar[loc, compName1, ip] == capVar[loc, compName2, ip]

        setattr(
            pyM,
            "ConstrLinkedCapacity_" + abbrvName,
            pyomo.Constraint(linkedList, pyM.investSet, rule=linkedCapacity),
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
        # Set, if applicable, the binary design variables of a component
        self.designBinFix(pyM)
        # Link, if applicable, the capacity of components with the same linkedConversionCapacityID
        self.linkedCapacity(pyM)
        # Set yearly full load hours minimum limit
        self.yearlyFullLoadHoursMin(
            pyM, esM, "yearlyFullLoadHoursMinSet", "ConstrYearlyFullLoadHoursMin", "op"
        )
        self.yearlyFullLoadHoursMin(
            pyM,
            esM,
            "yearlyFullLoadHoursCommisMinSet",
            "ConstrYearlyFullLoadHoursMinCommis",
            "op_commis",
            isOperationCommisYearDepending=True,
        )
        # Set yearly full load hours maximum limit
        self.yearlyFullLoadHoursMax(
            pyM, esM, "yearlyFullLoadHoursMaxSet", "ConstrYearlyFullLoadHoursMax", "op"
        )
        self.yearlyFullLoadHoursMax(
            pyM,
            esM,
            "yearlyFullLoadHoursCommisMaxSet",
            "ConstrYearlyFullLoadHoursMaxCommis",
            "op_commis",
            isOperationCommisYearDepending=True,
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

        # Operation [physicalUnit*h] is limited by the installed capacity [physicalUnit] multiplied by the hours per
        # time step [h]
        self.operationMode1(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        self.operationMode1(
            pyM,
            esM,
            "ConstrOperationCommis",
            "opCommisConstrSet",
            "op_commis",
            isOperationCommisYearDepending=True,
        )
        # Operation [physicalUnit*h] is equal to the installed capacity [physicalUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode2(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        self.operationMode2(
            pyM,
            esM,
            "ConstrOperationCommis",
            "opCommisConstrSet",
            "op_commis",
            isOperationCommisYearDepending=True,
        )
        # Operation [physicalUnit*h] is limited by the installed capacity [physicalUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode3(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        self.operationMode3(
            pyM,
            esM,
            "ConstrOperationCommis",
            "opCommisConstrSet",
            "op_commis",
            isOperationCommisYearDepending=True,
        )
        # Operation [physicalUnit*h] is limited (min) by the installed capacity [physicalUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode4(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        self.operationMode4(
            pyM,
            esM,
            "ConstrOperationCommis",
            "opCommisConstrSet",
            "op_commis",
            isOperationCommisYearDepending=True,
        )

        # # Operation [physicalUnit*h] is limited by minimum part Load
        self.additionalMinPartLoad(
            pyM, esM, "ConstrOperation", "opConstrSet", "op", "op_bin", "cap"
        )
        self.additionalMinPartLoad(
            pyM,
            esM,
            "ConstrOperationCommis",
            "opCommisConstrSet",
            "op",
            "op_bin",
            "cap",
            isOperationCommisYearDepending=True,
        )
        # Operation for components with commissioning year dependent commodity conversion factors
        self.getTotalOperationCommissioningDependentOperation(pyM)

        # Declare flexible conversion constraints
        self.flexConversionConstraint(pyM, esM)
        self.flexConversionFlowShareConstraint(pyM)

    def getTotalOperationCommissioningDependentOperation(self, pyM):
        """
        Ensure that the sum of all commissioning dependent operating variables equals the total operating variable
        of that conversion component for each time step.
        """

        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)
        opCommisVar = getattr(pyM, "op_commis_" + abbrvName)
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def combinedOperation(pyM, loc, compName, ip, p, t):
            if not compDict[compName].isCommisDepending:
                return pyomo.Constraint.Skip
            else:
                commisYearsWithOperationInIp = [
                    _commis
                    for (_commis, _ip) in compDict[
                        compName
                    ].processedCommodityConversionFactors
                    if _ip == ip
                ]
                sumOpCommisVar = sum(
                    opCommisVar[loc, compName, commis, ip, p, t]
                    for commis in commisYearsWithOperationInIp
                )
                return opVar[loc, compName, ip, p, t] == sumOpCommisVar

        setattr(
            pyM,
            "commisCombinedOperation_conv2" + abbrvName,
            pyomo.Constraint(opVarSet, pyM.intraYearTimeSet, rule=combinedOperation),
        )

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
        # note: year definition can either
        # a) int: if the commodity conversion factor is constant for commissioning years
        # b) tuple with (commisYear,ip) if the commodity conversion is varying with the commissioning year

        operationFlexVarSet = getattr(esM.pyM, "operationFlexVarSet_" + self.abbrvName)
        flexible_commods = {
            index[4]
            for index in operationFlexVarSet
            if index[0] == loc
        }
        return any(
            [
                (
                    commod in comp.processedCommodityConversionFactors[yearDefinition]
                    and (
                        comp.processedCommodityConversionFactors[yearDefinition][commod]
                        is not None
                    )
                )
                and comp.processedLocationalEligibility[loc] == 1
                for comp in self.componentsDict.values()
                for yearDefinition in comp.processedCommodityConversionFactors.keys()
            ] +
            [
                commod in comp.processedEmissionFactors.keys()
                for comp in self.componentsDict.values()
                if comp.processedEmissionFactors is not None
            ]
        ) or commod in flexible_commods

    def flexConversionConstraint(self, pyM, esM):
        """
        Declare constraint that ensures that the sum of all flexible operation variables of one component are equal
        to the overall operation of this component.
        """

        opVar = getattr(pyM, "op_" + self.abbrvName)
        opVarFlex = getattr(pyM, "op_flex_" + self.abbrvName)
        compDict = self.componentsDict

        def input_output_constr(pyM, loc, compName, ip, group, p, t):
            ccf = compDict[compName].processedCommodityConversionFactors[ip][group]

            opVarFlexSum = sum(
                opVarFlex[loc, compName, ip, group, commod, p, t]
                for commod, value in ccf.items()
            )
            return opVarFlexSum == opVar[loc, compName, ip, p, t]

        setattr(
            pyM,
            "flexConversion_input_output_constraint_" + self.abbrvName,
            pyomo.Constraint(
                getattr(pyM, "operationFlexGroupSet_" + self.abbrvName),
                pyM.intraYearTimeSet,
                rule=input_output_constr
            ),
        )

    def flexConversionFlowShareConstraint(self, pyM):
        """
        Declare constraint that applies flow shares for each flexible component.
        """
        opVar = getattr(pyM, "op_" + self.abbrvName)
        opVarFlex = getattr(pyM, "op_flex_" + self.abbrvName)
        compDict = self.componentsDict

        def flow_share_constr(pyM, loc, compName, ip, group, attr, commod, p, t):
            flowShares = compDict[compName].processedFlowShares

            if attr == 'min':
                return (
                    opVarFlex[loc, compName, ip, group, commod, p, t] >=
                    opVar[loc, compName, ip, p, t] * flowShares[ip][attr][commod].loc[loc]
                )
            elif attr == 'max':
                return (
                    opVarFlex[loc, compName, ip, group, commod, p, t] <=
                    opVar[loc, compName, ip, p, t] * flowShares[ip][attr][commod].loc[loc]
                )
            elif attr == 'fix':
                return (
                    opVarFlex[loc, compName, ip, group, commod, p, t] ==
                    opVar[loc, compName, ip, p, t] * flowShares[ip][attr][commod].loc[loc]
                )

        setattr(
            pyM,
            "flexConversion_flow_share_constraint_" + self.abbrvName,
            pyomo.Constraint(
                getattr(pyM, "operationFlexFlowShareConstrSet_" + self.abbrvName),
                pyM.intraYearTimeSet,
                rule=flow_share_constr
            ),
        )

    def getCommodityBalanceContribution(self, pyM, commod, loc, ip, p, t):
        """Get contribution to a commodity balance.

        .. math::

            \\text{C}^{comp,comm}_{loc,ip,p,t} =  \\text{conversionFactor}^{comp}_{comm} \cdot op_{loc,ip,p,t}^{comp,op}

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)
        opVarFlex = getattr(pyM, "op_flex_" + abbrvName)
        opCommisVar = getattr(pyM, "op_commis_" + abbrvName)
        opVarDict = getattr(pyM, "operationVarDict_" + abbrvName)

        def getFactor(commodCommodityConversionFactors, loc, p, t):
            if isinstance(commodCommodityConversionFactors, (int, float)):
                return commodCommodityConversionFactors
            else:
                return commodCommodityConversionFactors[loc][p, t]

        # 1.a get balance for components, which do not have commodity conversions varying with the commissioning year
        # prepare data
        sumCommisYearIndependent = sum(
            opVar[loc, compName, ip, p, t]
            * getFactor(
                compDict[compName].processedCommodityConversionFactors[ip][commod],
                loc,
                p,
                t,
            )
            for compName in opVarDict[ip][loc]
            if not compDict[compName].isCommisDepending
            and commod in compDict[compName].processedCommodityConversionFactors[ip]
        )
        # 1.b get balance contribution for flexible conversion components
        flexible_comp_commod_groups = {
            (index[1], index[3])
            for index in getattr(pyM, "operationFlexGroupSet_" + self.abbrvName)
            if index[0] == loc
        }
        sumCommisYearIndependentFlex = sum(
            opVarFlex[loc, compName, ip, group, commod, p, t]
            * getFactor(
                compDict[compName].processedCommodityConversionFactors[ip][group][commod],
                loc,
                p,
                t,
            )
            for compName, group in flexible_comp_commod_groups
            if not compDict[compName].isCommisDepending
            if compDict[compName].flexibleConversion
            and commod in compDict[compName].processedCommodityConversionFactors[ip][group]
        )
        # 2. commodity conversions factors is depending on the commissioning year (e.g. efficiencies) if
        # a) component has isCommisDepending
        # b) component processes the commodity
        sumCommisYearDependent = 0
        if any(compDict[comp].isCommisDepending for comp in opVarDict[ip][loc]):
            # TODO implement dataframe similar to cost consideration
            for compName in opVarDict[ip][loc]:
                if compDict[compName].isCommisDepending:
                    commodConv = compDict[compName].processedCommodityConversionFactors
                    relevantCommissioningYears = [
                        x for (x, y) in commodConv.keys() if y == ip
                    ]
                    for _commis in relevantCommissioningYears:
                        if (
                            commod
                            in compDict[compName].processedCommodityConversionFactors[
                                (_commis, ip)
                            ]
                        ):
                            sumCommisYearDependent += opCommisVar[
                                loc, compName, _commis, ip, p, t
                            ] * getFactor(
                                compDict[compName].processedCommodityConversionFactors[
                                    (_commis, ip)
                                ][commod],
                                loc,
                                p,
                                t,
                            )
        # 3. get balance contribution for emissions from flexible conversion components
        emission_comps = {
            compName: list(compDict[compName].processedEmissionFactors[commod].keys())
            for compName in compDict.keys()
            if compDict[compName].processedEmissionFactors is not None
            if commod in compDict[compName].processedEmissionFactors.keys()
            if compDict[compName].processedEmissionFactors[commod] is not None
        }
        if len(emission_comps) > 0:
            sumFlexEmission = sum(
                opVarFlex[loc, compName, ip, group, commodity, p, t]
                * compDict[compName].emissionFactors[commod][commodity]
                * abs(
                    getFactor(
                        compDict[compName].processedCommodityConversionFactors[ip][group][commodity],
                        loc,
                        p,
                        t
                    )
                )
                for compName, group in flexible_comp_commod_groups
                if compName in emission_comps.keys()
                for commodity in emission_comps[compName]
                if commodity in compDict[compName].processedCommodityConversionFactors[ip][group].keys()
            )
        else:
            sumFlexEmission = 0

        return sumCommisYearIndependent + sumCommisYearDependent + sumCommisYearIndependentFlex + sumFlexEmission

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        opexOp = self.getEconomicsOperation(
            pyM, esM, "TD", ["processedOpexPerOperation"], "op", "operationVarDict"
        )

        return super().getObjectiveFunctionContribution(esM, pyM) + opexOp

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

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

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(
            esM, pyM, esM.locations, "physicalUnit"
        )

        # Get class related results
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

        for ip in esM.investmentPeriods:
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

            props = ["operation", "opexOp", "NPV_opexOp"]
            # Unit dict: Specify units for props
            units = {
                props[0]: ["[-*h]", "[-*h/a]"],
                props[1]: ["[" + esM.costUnit + "/a]"],
                props[2]: ["[" + esM.costUnit + "/a]"],
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
                        (
                            x[0],
                            x[1],
                            x[2].replace("-", compDict[x[0]].physicalUnit),
                        )
                        if x[1] == "operation"
                        else x
                    ),
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
                idx = pd.IndexSlice
                optVal = optVal.loc[
                    idx[:, :], :
                ]  # perfect foresight: added ip and deleted again
                opSum = optVal.sum(axis=1).unstack(-1)

                # operation
                optSummary.loc[
                    [
                        (ix, "operation", "[" + compDict[ix].physicalUnit + "*h/a]")
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = (
                    opSum.values / esM.numberOfYears
                )
                optSummary.loc[
                    [
                        (ix, "operation", "[" + compDict[ix].physicalUnit + "*h]")
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = opSum.values

                # operation cost - TAC
                tac_ox = resultsTAC_opexOp[ip]
                optSummary.loc[
                    [(ix, "opexOp", "[" + esM.costUnit + "/a]") for ix in tac_ox.index],
                    tac_ox.columns,
                ] = tac_ox.values
                # operation cost - NPV contribution
                npv_ox = resultsNPV_opexOp[ip]
                optSummary.loc[
                    [
                        (ix, "NPV_opexOp", "[" + esM.costUnit + "/a]")
                        for ix in npv_ox.index
                    ],
                    npv_ox.columns,
                ] = npv_ox.values

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
            # Update the NPV contribution
            optSummary.loc[
                optSummary.index.get_level_values(1) == "NPVcontribution"
            ] = (
                optSummary.loc[
                    (optSummary.index.get_level_values(1) == "NPVcontribution")
                    | (optSummary.index.get_level_values(1) == "NPV_opexOp")
                ]
                .groupby(level=0)
                .sum()
                .values
            )
            # # Delete details of NPV contributions
            optSummary = optSummary.drop("NPV_opexOp", level=1)

            # save the optimization summary
            self._optSummary[esM.investmentPeriodNames[ip]] = optSummary

    def getOptimalValues(self, name="all", ip=0):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariables',
            * 'isBuiltVariables',
            * 'operationVariablesOptimum',
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
