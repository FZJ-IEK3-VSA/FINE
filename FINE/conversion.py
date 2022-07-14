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
        yearlyFullLoadHoursMin=None,
        yearlyFullLoadHoursMax=None,
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
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the physicalUnit of the plant for each time step.
            |br| * the default value is None
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries or dict with entries of
            None or Pandas DataFrame with positive (>=0) per investement period. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param operationRateFix: if specified, indicates a fixed operation rate for each location and each time
            step by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the physicalUnit of the plant for each time step.
            |br| * the default value is None
        :type operationRateFix: None or Pandas DataFrame with positive (>= 0) entries or dict with entries of
            None or Pandas DataFrame with positive (>=0) per investement period. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

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
        :type opexPerOperation: positive (>=0) float or Pandas Series with positive (>=0) values or dict with
            entries per investemenr periods of positive (>=0) float or Pandas Series with positive (>=0).
            The indices of the series have to equal the in the energy system model specified locations.
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
            yearlyFullLoadHoursMin=yearlyFullLoadHoursMin,
            yearlyFullLoadHoursMax=yearlyFullLoadHoursMax,
        )

        # check if parameter has None values, if it is a dict
        for param in [operationRateMax, operationRateFix, partLoadMin]:
            utils.checkParamInput(param)

        # new code for commodity conversions
        self.commodityConversionFactors = commodityConversionFactors
        self.fullCommodityConversionFactors = {}
        self.aggregatedCommodityConversionFactors = {}
        self.processedCommodityConversionFactors = {}

        for ip in esM.investmentPeriods:
            # 1. dict aus Jahren -> verschiedene commoditiyconversion für jahr
            if list(commodityConversionFactors.keys())[0] in esM.investmentPeriods:
                _commodityConversionFactors = commodityConversionFactors[ip]
            else:
                _commodityConversionFactors = commodityConversionFactors

            utils.checkCommodities(esM, set(_commodityConversionFactors.keys()))
            utils.checkCommodityUnits(esM, physicalUnit)
            if linkedConversionCapacityID is not None:
                utils.isString(linkedConversionCapacityID)
            self.fullCommodityConversionFactors[ip] = {}
            self.aggregatedCommodityConversionFactors[ip] = {}
            self.processedCommodityConversionFactors[ip] = {}

            for commod in _commodityConversionFactors.keys():

                if not isinstance(_commodityConversionFactors[commod], (int, float)):
                    self.fullCommodityConversionFactors[ip][
                        commod
                    ] = utils.checkAndSetTimeSeriesConversionFactors(
                        esM,
                        _commodityConversionFactors[commod],
                        self.locationalEligibility,
                    )
                    self.aggregatedCommodityConversionFactors[ip][commod] = None
                elif isinstance(self.commodityConversionFactors[commod], (int, float)):
                    self.processedCommodityConversionFactors[ip][
                        commod
                    ] = _commodityConversionFactors[commod]

        self.physicalUnit = physicalUnit
        self.modelingClass = ConversionModel
        self.linkedConversionCapacityID = linkedConversionCapacityID

        self.opexPerOperation = opexPerOperation
        self.processedOpexPerOperation = {}

        # iterate over all ips
        for ip in esM.investmentPeriods:

            # opexPerOperation
            if (
                isinstance(opexPerOperation, int)
                or isinstance(opexPerOperation, float)
                or isinstance(opexPerOperation, pd.Series)
            ):
                self.processedOpexPerOperation[ip] = utils.checkAndSetCostParameter(
                    esM, name, opexPerOperation, "1dim", locationalEligibility
                )
            elif isinstance(opexPerOperation, dict):
                self.processedOpexPerOperation[ip] = utils.checkAndSetCostParameter(
                    esM, name, opexPerOperation[ip], "1dim", locationalEligibility
                )
            else:
                raise TypeError(
                    "opexPerOperation should be a pandas series or a dictionary."
                )

        # Set location-specific operation parameters: operationRateMax or operationRateFix, tsaweight
        self.operationRateMax = operationRateMax
        self.operationRateFix = operationRateFix

        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            if esM.verbose < 2:
                warnings.warn(
                    "If operationRateFix is specified, the operationRateMax parameter is not required.\n"
                    + "The operationRateMax time series was set to None."
                )

        ## New code for perfect foresight!
        # create emtpy dicts
        self.fullOperationRateMax = {}
        self.aggregatedOperationRateMax = {}
        self.processedOperationRateMax = {}

        self.fullOperationRateFix = {}
        self.aggregatedOperationRateFix = {}
        self.processedOperationRateFix = {}

        # iterate over all ips
        for ip in esM.investmentPeriods:

            # Operation Rate Max
            if (
                isinstance(operationRateMax, pd.DataFrame)
                or isinstance(operationRateMax, pd.Series)
                or operationRateMax is None
            ):
                self.fullOperationRateMax[ip] = utils.checkAndSetTimeSeries(
                    esM, name, operationRateMax, locationalEligibility
                )
            elif isinstance(operationRateMax, dict):
                self.fullOperationRateMax[ip] = utils.checkAndSetTimeSeries(
                    esM, name, operationRateMax[ip], locationalEligibility
                )
            else:
                raise TypeError(
                    "OperationRateMax should be a pandas dataframe or a dictionary."
                )

            self.aggregatedOperationRateMax[ip], self.processedOperationRateMax[ip] = (
                None,
                None,
            )

            # Operation Rate Fix
            if (
                isinstance(operationRateFix, pd.DataFrame)
                or isinstance(operationRateFix, pd.Series)
                or operationRateFix is None
            ):  # operationRate is dataframe or series
                self.fullOperationRateFix[ip] = utils.checkAndSetTimeSeries(
                    esM, name, operationRateFix, locationalEligibility
                )
            elif isinstance(operationRateFix, dict):  # operationRate is dict
                self.fullOperationRateFix[ip] = utils.checkAndSetTimeSeries(
                    esM, name, operationRateFix[ip], locationalEligibility
                )
            # elif operationRateFix is None:
            #     pass
            else:
                raise TypeError(
                    "OperationRateFix should be a pandas dataframe or a dictionary."
                )

            self.aggregatedOperationRateFix[ip], self.processedOperationRateFix[ip] = (
                None,
                None,
            )

        # new code for perfect foresight
        self.partLoadMin = partLoadMin
        self.processedPartLoadMin = {}

        # iterate over all ips
        for ip in esM.investmentPeriods:
            if isinstance(partLoadMin, float) or partLoadMin is None:
                self.processedPartLoadMin[ip] = partLoadMin
            elif isinstance(partLoadMin, dict):
                self.processedPartLoadMin[ip] = partLoadMin[ip]

        if not any(value for value in self.processedPartLoadMin.values()):
            self.processedPartLoadMin = None

        if self.processedPartLoadMin is not None:
            for ip in esM.investmentPeriods:
                if self.processedPartLoadMin[ip] is not None:
                    if self.fullOperationRateMax[ip] is not None:
                        if (
                            (
                                (self.fullOperationRateMax[ip] > 0)
                                & (
                                    self.fullOperationRateMax[ip]
                                    < self.processedPartLoadMin[ip]
                                )
                            )
                            .any()
                            .any()
                        ):
                            raise ValueError(
                                '"operationRateMax" needs to be higher than "partLoadMin" or 0 for component '
                                + name
                            )
                    if self.fullOperationRateFix[ip] is not None:
                        if (
                            (
                                (self.fullOperationRateFix[ip] > 0)
                                & (
                                    self.fullOperationRateFix[ip]
                                    < self.processedPartLoadMin[ip]
                                )
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

        if all(
            type(value) != pd.core.frame.DataFrame
            for value in self.fullOperationRateFix.values()
        ):
            self.fullOperationRateFix = None

        if all(
            type(value) != pd.core.frame.DataFrame
            for value in self.fullOperationRateMax.values()
        ):
            self.fullOperationRateMax = None

        operationTimeSeries = {}
        if self.fullOperationRateFix is not None:
            for ip in esM.investmentPeriods:
                operationTimeSeries[ip] = self.fullOperationRateFix[ip]
        elif self.fullOperationRateMax is not None:
            for ip in esM.investmentPeriods:
                operationTimeSeries[ip] = self.fullOperationRateMax[ip]
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

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a conversion component to the given energy system model

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
            print("\n\n self.fullCommodityConversionFactors[commod]")
            print(self.fullCommodityConversionFactors[ip][commod])
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
            if getattr(self, parameter) is not None:
                if all(
                    type(value) != pd.core.frame.DataFrame
                    for value in getattr(self, parameter).values()
                ):
                    setattr(self, parameter, None)

    def initializeProcessedDataSets(self, investmentperiods):
        """
        Initialize dicts (keys are investment periods, values are None)
        for processed data sets.

        :param investmentperiods: investmentperiods of transformation path analysis.
        :type investmentperiods: list
        """
        self.processedOperationRateMax = dict.fromkeys(investmentperiods)
        self.processedOperationRateFix = dict.fromkeys(investmentperiods)


class ConversionModel(ComponentModel):
    """
    A ConversionModel class instance will be instantly created if a Conversion class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Conversion class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The ConversionModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        """ " Constructor for creating a ConversionModel class instance"""
        self.abbrvName = "conv"
        self.dimension = "1dim"
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.optSummary = {}
        self.operationVariablesOptimum = {}

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
        self.declareDesignVarSet(pyM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        # Declare operation variable sets
        self.declareOpVarSet(esM, pyM)
        self.declareOperationBinarySet(pyM)

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

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary):
        """
        Declare design and operation variables

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
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
        self.declareOperationVars(pyM, "op")
        # Operation of component as binary [1/0]
        self.declareOperationBinaryVars(pyM, "op_bin")

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

        def linkedCapacity(pyM, loc, compName1, compName2):
            return capVar[loc, compName1] == capVar[loc, compName2]

        setattr(
            pyM,
            "ConstrLinkedCapacity_" + abbrvName,
            pyomo.Constraint(linkedList, rule=linkedCapacity),
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
        # Link, if applicable, the capacity of components with the same linkedConversionCapacityID
        self.linkedCapacity(pyM)
        # Set yearly full load hours minimum limit
        self.yearlyFullLoadHoursMin(pyM, esM)
        # Set yearly full load hours maximum limit
        self.yearlyFullLoadHoursMax(pyM, esM)

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
        # Operation [physicalUnit*h] is equal to the operation time series [physicalUnit*h]
        self.operationMode4(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [physicalUnit*h] is limited by the operation time series [physicalUnit*h]
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

            \\text{C}^{comp,comm}_{loc,p,t} =  \\text{conversionFactor}^{comp}_{comm} \cdot op_{loc,p,t}^{comp,op}

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
            for compName in opVarDict[loc]
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

        opexOp = self.getEconomicsTD(
            pyM, esM, ["processedOpexPerOperation"], "op", "operationVarDict"
        )

        return super().getObjectiveFunctionContribution(esM, pyM) + opexOp

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM, ip):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(
            esM, pyM, esM.locations, "physicalUnit"
        )

        # Set optimal operation variables and append optimization summary
        optVal = utils.formatOptimizationOutput(
            opVar.get_values(),
            "operationVariables",
            "1dim",
            ip,
            esM.periodsOrder[ip],
            esM=esM,
        )
        # Quick fix if several runs with one investment period
        if type(self.operationVariablesOptimum) is not dict:
            self.operationVariablesOptimum = {}
        self.operationVariablesOptimum[ip] = optVal

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
                lambda x: (x[0], x[1], x[2].replace("-", compDict[x[0]].physicalUnit))
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
            ox = opSum.apply(
                lambda op: op
                * compDict[op.name].processedOpexPerOperation[ip][op.index],
                axis=1,
            )
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
            optSummary.loc[
                [(ix, "opexOp", "[" + esM.costUnit + "/a]") for ix in ox.index],
                ox.columns,
            ] = (
                ox.values / esM.numberOfYears
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
        # Quick fix if several runs with one investment period
        if type(self.optSummary) is not dict:
            self.optSummary = {}
        self.optSummary[ip] = optSummary

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
