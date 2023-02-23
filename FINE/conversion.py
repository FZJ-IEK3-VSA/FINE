from FINE.component import Component, ComponentModel
from FINE import utils
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
        floorTechnicalLifetime=True,
        yearlyFullLoadHoursMin=None,
        yearlyFullLoadHoursMax=None,
        stockCommissioning=None,
    ):
        # TODO: allow that the time series data or min/max/fixCapacity/eligibility is only specified for
        # TODO: eligible locations
        """
        Constructor for creating an Conversion class instance. Capacities are given in the physical unit
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

            The conversion factors are currently constant over the transformation pathway!

            Examples:

            * An electrolyzer converts, simply put, electricity into hydrogen with an electrical efficiency
                of 70%. The physicalUnit is given as GW_electric, the unit for the 'electricity' commodity is
                given in GW_electric and the 'hydrogen' commodity is given in GW_hydrogen_lowerHeatingValue
                -> the commodityConversionFactors are defined as {'electricity':-1,'hydrogen':0.7}.
            * A fuel cell converts, simply put, hydrogen into electricity with an efficiency of 60%.
                The physicalUnit is given as GW_electric, the unit for the 'electricity' commodity is given in
                GW_electric and the 'hydrogen' commodity is given in GW_hydrogen_lowerHeatingValue
                -> the commodityConversionFactors are defined as {'electricity':1,'hydrogen':-1/0.6}.

        :type commodityConversionFactors: dictionary, assigns commodities (string) to a conversion factors
            (float, pandas.Series or pandas.DataFrame) or dictionary with assigned conversion factors per
            investment period

        **Default arguments:**

        :param linkedConversionCapacityID: if specifies, indicates that all conversion components with the
            same ID have to have the same capacity.
            |br| * the default value is None
        :type linkedConversionCapacityID: string

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
        )

        # commodity conversions factors
        # TODO The dependency of commodity conversion factors from
        # commissioning year is currently not supported and will be implemented
        # in the next merge request.
        if any(
            isinstance(commodityConversionFactors[x], dict)
            for x in commodityConversionFactors.keys()
        ):
            raise NotImplementedError(
                "A variation of commodity conversion factors over investment periods is currently not supported."
            )
        self.commodityConversionFactors = commodityConversionFactors
        (
            self.fullCommodityConversionFactors,
            self.processedCommodityConversionFactors,
        ) = utils.checkAndSetCommodityConversionFactor(self, esM)
        self.aggregatedCommodityConversionFactors = dict.fromkeys(
            esM.investmentPeriods, {}
        )

        utils.isPositiveNumber(tsaWeight)
        self.tsaWeight = tsaWeight
        utils.checkCommodityUnits(esM, physicalUnit)
        if linkedConversionCapacityID is not None:
            utils.isString(linkedConversionCapacityID)
        self.physicalUnit = physicalUnit
        self.modelingClass = ConversionModel
        self.linkedConversionCapacityID = linkedConversionCapacityID

        # set parameter to None if all years have None values
        self.fullOperationRateFix = utils.setParamToNoneIfNoneForAllYears(
            self.fullOperationRateFix
        )
        self.fullOperationRateMax = utils.setParamToNoneIfNoneForAllYears(
            self.fullOperationRateMax
        )

        if self.fullOperationRateFix is not None:
            operationTimeSeries = self.fullOperationRateFix
        elif self.fullOperationRateMax is not None:
            operationTimeSeries = self.fullOperationRateMax
        else:
            operationTimeSeries = None

        self.locationalEligibility = utils.setLocationalEligibility(
            esM,
            self.locationalEligibility,
            self.capacityMax,
            self.capacityFix,
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
        # processedCommodityConversions
        for ip in self.fullCommodityConversionFactors.keys():
            if self.fullCommodityConversionFactors[ip] != {}:
                for commod in self.fullCommodityConversionFactors[ip]:
                    self.processedCommodityConversionFactors[ip][commod] = (
                        self.aggregatedCommodityConversionFactors[ip][commod]
                        if hasTSA
                        else self.fullCommodityConversionFactors[ip][commod]
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
        for commod in self.fullCommodityConversionFactors[ip]:
            weightDict, data = self.prepareTSAInput(
                self.fullCommodityConversionFactors[ip][commod],
                None,
                "_commodityConversionFactorTimeSeries" + str(commod) + "_",
                self.tsaWeight,
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
            self.fullOperationRateFix, "_operationRate_", data, ip
        )
        self.aggregatedOperationRateMax[ip] = self.getTSAOutput(
            self.fullOperationRateMax, "_operationRate_", data, ip
        )

        if self.fullCommodityConversionFactors[ip] != {}:
            for commod in self.fullCommodityConversionFactors[ip]:
                self.aggregatedCommodityConversionFactors[ip][
                    commod
                ] = self.getTSAOutput(
                    self.fullCommodityConversionFactors[ip][commod],
                    "_commodityConversionFactorTimeSeries" + str(commod) + "_",
                    data,
                    ip,
                )

    def checkProcessedDataSets(self):
        """
        Check processed time series data after applying time series
        aggregation. If all entries of dictionary are None
        the parameter itself is set to None.
        """
        for parameter in ["processedOperationRateFix", "processedOperationRateMax"]:
            setattr(
                self,
                parameter,
                utils.setParamToNoneIfNoneForAllYears(getattr(self, parameter)),
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
                        for loc, v in values[i].locationalEligibility.items()
                        if v == 1
                    ]
                )
        for comps in linkedComponentsList:
            index1 = compDict[comps[1]].locationalEligibility.index
            index2 = compDict[comps[2]].locationalEligibility.index
            if not index1.equals(index2):
                raise ValueError(
                    "Conversion components ",
                    comps[1],
                    "and",
                    comps[2],
                    "are linked but do not have the same locationalEligibility.",
                )
        setattr(pyM, "linkedComponentsList_" + self.abbrvName, linkedComponentsList)

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

        # Declare operation mode sets
        self.declareOperationModeSets(
            pyM, "opConstrSet", "processedOperationRateMax", "processedOperationRateFix"
        )

        # Declare linked components dictionary
        self.declareLinkedCapacityDict(pyM)

        # Declare minimum yearly full load hour set
        self.declareYearlyFullLoadHoursMinSet(pyM)

        # Declare maximum yearly full load hour set
        self.declareYearlyFullLoadHoursMaxSet(pyM)

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
        compDict, abbrvName = self.componentsDict, self.abbrvName
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
        # Set, if applicable, the installed capacities of a component
        self.capacityFix(pyM, esM)
        # Set, if applicable, the binary design variables of a component
        self.designBinFix(pyM)
        # Link, if applicable, the capacity of components with the same linkedConversionCapacityID
        self.linkedCapacity(pyM)
        # Set yearly full load hours minimum limit
        self.yearlyFullLoadHoursMin(pyM, esM)
        # Set yearly full load hours maximum limit
        self.yearlyFullLoadHoursMax(pyM, esM)

        ################################################################################################################
        #                                    Declare pathway constraints                                               #
        ################################################################################################################
        # Set capacity development constraints over investment periods
        self.designDevelopmentConstraint(pyM, esM)
        self.decommissioningConstraint(pyM, esM)
        self.stockCapacityConstraint(pyM, esM)
        self.stockCommissioningConstaint(pyM, esM)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [physicalUnit*h] is limited by the installed capacity [physicalUnit] multiplied by the hours per
        # time step [h]
        self.operationMode1(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [physicalUnit*h] is equal to the installed capacity [physicalUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode2(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [physicalUnit*h] is limited by the installed capacity [physicalUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode3(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [physicalUnit*h] is limited by minimum part Load
        self.additionalMinPartLoad(
            pyM, esM, "ConstrOperation", "opConstrSet", "op", "op_bin", "cap"
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
        return any(
            [
                (
                    commod in comp.processedCommodityConversionFactors[ip]
                    and (
                        comp.processedCommodityConversionFactors[ip][commod] is not None
                    )
                )
                and comp.locationalEligibility[loc] == 1
                for comp in self.componentsDict.values()
                for ip in esM.investmentPeriods
            ]
        )

    def getCommodityBalanceContribution(self, pyM, commod, loc, ip, p, t):
        """Get contribution to a commodity balance.

        .. math::

            \\text{C}^{comp,comm}_{loc,ip,p,t} =  \\text{conversionFactor}^{comp}_{comm} \cdot op_{loc,ip,p,t}^{comp,op}

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDict = (
            getattr(pyM, "op_" + abbrvName),
            getattr(pyM, "operationVarDict_" + abbrvName),
        )

        def getFactor(commodCommodityConversionFactors, loc, p, t):
            if isinstance(commodCommodityConversionFactors, (int, float)):
                return commodCommodityConversionFactors
            else:
                return commodCommodityConversionFactors[loc][p, t]

        return sum(
            opVar[loc, compName, ip, p, t]
            * getFactor(
                compDict[compName].processedCommodityConversionFactors[ip][commod],
                loc,
                p,
                t,
            )
            for compName in opVarDict[ip][loc]
            if commod in compDict[compName].processedCommodityConversionFactors[ip]
        )

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
                        x[0],
                        x[1],
                        x[2].replace("-", compDict[x[0]].physicalUnit),
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

            optSummary = optSummary.append(
                optSummaryBasic[esM.investmentPeriodNames[ip]]
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
