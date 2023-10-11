from FINE.component import Component, ComponentModel
from FINE import utils
import pyomo.environ as pyomo
import warnings
import pandas as pd
import numpy as np


class Storage(Component):
    """
    A Storage component can store a commodity and thus transfers it between time steps.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
        chargeRate=1,
        dischargeRate=1,
        chargeEfficiency=1,
        dischargeEfficiency=1,
        selfDischarge=0,
        cyclicLifetime=None,
        stateOfChargeMin=0,
        stateOfChargeMax=1,
        hasCapacityVariable=True,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=1,
        hasIsBuiltBinaryVariable=False,
        bigM=None,
        doPreciseTsaModeling=False,
        chargeOpRateMax=None,
        chargeOpRateFix=None,
        chargeTsaWeight=1,
        dischargeOpRateMax=None,
        dischargeOpRateFix=None,
        dischargeTsaWeight=1,
        isPeriodicalStorage=False,
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
        opexPerChargeOperation=0,
        opexPerDischargeOperation=0,
        opexPerCapacity=0,
        opexIfBuilt=0,
        interestRate=0.08,
        economicLifetime=10,
        technicalLifetime=None,
        floorTechnicalLifetime=True,
        socOffsetDown=-1,
        socOffsetUp=-1,
        stockCommissioning=None,
    ):
        """
        Constructor for creating an Storage class instance.
        The Storage component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param commodity: to the component related commodity.
        :type commodity: string

        **Default arguments:**

        :param chargeRate: ratio of the maximum storage inflow (in commodityUnit/hour) to the
            storage capacity (in commodityUnit).
            Example:

            * A hydrogen salt cavern which can store 133 GWh_H2_LHV can be charged 0.45 GWh_H2_LHV during
              one hour. The chargeRate thus equals 0.45/133 1/h.

            |br| * the default value is 1
        :type chargeRate: 0 <= float <=1

        :param dischargeRate: ratio of the maximum storage outflow (in commodityUnit/hour) to
            the storage capacity (in commodityUnit).
            Example:

            * A hydrogen salt cavern which can store 133 GWh_H2_LHV can be discharged 0.45 GWh_H2_LHV during
              one hour. The dischargeRate thus equals 0.45/133.

            |br| * the default value is 1
        :type dischargeRate: 0 <= float <=1

        :param chargeEfficiency: defines the efficiency with which the storage can be charged (equals
            the percentage of the injected commodity that is transformed into stored commodity).
            Enter 0.98 for 98% etc.
            |br| * the default value is 1
        :type chargeEfficiency: 0 <= float <=1

        :param dischargeEfficiency: defines the efficiency with which the storage can be discharged
            (equals the percentage of the withdrawn commodity that is transformed into stored commodity).
            Enter 0.98 for 98% etc.
            |br| * the default value is 1
        :type dischargeEfficiency: 0 <= float <=1

        :param selfDischarge: percentage of self-discharge from the storage during one hour
            |br| * the default value is 0
        :type selfDischarge: 0 <= float <=1

        :param cyclicLifetime: if specified, the total number of full cycle equivalents that are supported
            by the technology.
            |br| * the default value is None
        :type cyclicLifetime: None or positive float

        :param stateOfChargeMin: threshold (percentage) that the state of charge can not drop under
            |br| * the default value is 0
        :type stateOfChargeMin: 0 <= float <=1

        :param stateOfChargeMax: threshold (percentage) that the state of charge can not exceed
            |br| * the default value is 1
        :type stateOfChargeMax: 0 <= float <=1

        :param doPreciseTsaModeling: determines whether the state of charge is limited precisely (True) or
            with a simplified method (False). The error is small if the selfDischarge is small.
            |br| * the default value is False
        :type doPreciseTsaModeling: boolean

        :param chargeOpRateMax: if specified, indicates a maximum charging rate for each location and each time
            step, if required also for each investment period, by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the charged commodity (before multiplying the charging efficiency)
            during one time step.
            |br| * the default value is None
        :type chargeOpRateMax:
            * None
            * Pandas DataFrame with positive (>= 0) entries. The row indices have
              to match the in the energy system model  specified time steps. The column indices have to match the
              in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param chargeOpRateFix: if specified, indicates a fixed charging rate for each location and each time
            step, if required also for each investment period, by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the charged commodity (before multiplying the charging efficiency)
            during one time step.
            |br| * the default value is None
        :type chargeOpRateFix:
            * None
            * Pandas DataFrame with positive (>= 0) entries. The row indices have
              to match the in the energy system model  specified time steps. The column indices have to match the
              in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param chargeTsaWeight: weight with which the chargeOpRate (max/fix) time series of the
            component should be considered when applying time series aggregation.
            |br| * the default value is 1
        :type chargeTsaWeight: positive (>= 0) float

        :param dischargeOpRateMax: if specified, indicates a maximum discharging rate for each location and each
            time step, if required also for each investment period, by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the discharged commodity (after multiplying the discharging
            efficiency) during one time step.
            |br| * the default value is None
        :type dischargeOpRateMax:
            * None
            * Pandas DataFrame with positive (>= 0) entries. The row indices have
              to match the in the energy system model  specified time steps. The column indices have to match the
              in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param dischargeOpRateFix: if specified, indicates a fixed discharging rate for each location and each
            time step, if required also for each investment period, by a positive float. If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit, referring to the charged commodity (after multiplying the discharging
            efficiency) during one time step.
            |br| * the default value is None
        :type dischargeOpRateFix:
            * None
            * Pandas DataFrame with positive (>= 0) entries. The row indices have
              to match the in the energy system model  specified time steps. The column indices have to match the
              in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param dischargeTsaWeight: weight with which the dischargeOpRate (max/fix) time series of the
            component should be considered when applying time series aggregation.
            |br| * the default value is 1
        :type dischargeTsaWeight: positive (>= 0) float

        :param isPeriodicalStorage: indicates if the state of charge of the storage has to be at the same value
            after the end of each period. This is especially relevant when using daily periods where short term
            storage can be restrained to daily cycles. Benefits the run time of the model.
            |br| * the default value is False
        :type isPeriodicalStorage: boolean

        :param opexPerChargeOperation: describes the cost for one unit of the charge operation.
            The cost which is directly proportional to the charge operation of the
            component is obtained by multiplying the opexPerChargeOperation parameter with the annual sum of the
            operational time series of the components. The opexPerChargeOperation can either be given as a float
            or a Pandas Series with location specific values or a dictionary per investment period with one of the two previous options.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerChargeOperation: positive (>=0) float or Pandas Series with positive (>=0) values or dict of
            positive (>=0) float or Pandas Series with positive (>=0) values per investment period.
            The indices of the series have to equal the in the energy system model specified locations.

        :param opexPerDischargeOperation: describes the cost for one unit of the discharge operation.
            The cost which is directly proportional to the discharge operation
            of the component is obtained by multiplying the opexPerDischargeOperation parameter with the annual sum
            of the operational time series of the components. The opexPerDischargeOperation can either be given as
            a float or a Pandas Series with location specific values or a dictionary per investment period with one of the two previous options.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro).
            |br| * the default value is 0
        :type opexPerDischargeOperation:
            * positive (>=0) float
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the energy system model specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param socOffsetDown: determines whether the state of charge at the end of a period p has
            to be equal to the one at the beginning of a period p+1 (socOffsetDown=-1) or if
            it can be smaller at the beginning of p+1 (socOffsetDown>=0). In the latter case,
            the product of the parameter socOffsetDown and the actual soc offset is used as a penalty
            factor in the objective function.
            |br| * the default value is -1
        :type socOffsetDown: float

        :param socOffsetUp: determines whether the state of charge at the end of a period p has
            to be equal to the one at the beginning of a period p+1 (socOffsetUp=-1) or if
            it can be larger at the beginning of p+1 (socOffsetUp>=0). In the latter case,
            the product of the parameter socOffsetUp and the actual soc offset is used as a penalty
            factor in the objective function.
            |br| * the default value is -1
        :type socOffsetUp: float
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
            interestRate=interestRate,
            economicLifetime=economicLifetime,
            technicalLifetime=technicalLifetime,
            stockCommissioning=stockCommissioning,
            floorTechnicalLifetime=floorTechnicalLifetime,
        )

        # Set general storage component data: chargeRate, dischargeRate, chargeEfficiency, dischargeEfficiency,
        # selfDischarge, cyclicLifetime, stateOfChargeMin, stateOfChargeMax, isPeriodicalStorage, doPreciseTsaModeling,
        # relaxedPeriodConnection
        utils.checkCommodities(esM, {commodity})
        self.commodity, self.commodityUnit = (
            commodity,
            esM.commodityUnitsDict[commodity],
        )
        # TODO unit and type checks
        self.chargeRate, self.dischargeRate = chargeRate, dischargeRate
        self.chargeEfficiency = chargeEfficiency
        self.dischargeEfficiency = dischargeEfficiency
        self.selfDischarge = selfDischarge
        self.cyclicLifetime = cyclicLifetime
        self.stateOfChargeMin = stateOfChargeMin
        self.stateOfChargeMax = stateOfChargeMax
        self.isPeriodicalStorage = isPeriodicalStorage
        self.doPreciseTsaModeling = doPreciseTsaModeling
        self.socOffsetUp = socOffsetUp
        self.socOffsetDown = socOffsetDown
        self.modelingClass = StorageModel

        # opexPerChargeOperation
        self.opexPerChargeOperation = opexPerChargeOperation
        self.processedOpexPerChargeOperation = (
            utils.checkAndSetInvestmentPeriodCostParameter(
                esM,
                name,
                opexPerChargeOperation,
                "1dim",
                locationalEligibility,
                esM.investmentPeriods,
            )
        )

        # opexPerDischargeOperation
        self.opexPerDischargeOperation = opexPerDischargeOperation
        self.processedOpexPerDischargeOperation = (
            utils.checkAndSetInvestmentPeriodCostParameter(
                esM,
                name,
                opexPerDischargeOperation,
                "1dim",
                locationalEligibility,
                esM.investmentPeriods,
            )
        )

        # chargeOpRateFix and chargeOpRateMax
        self.chargeOpRateMax = chargeOpRateMax
        self.chargeOpRateFix = chargeOpRateFix

        # chargeOpRateMax
        self.fullChargeOpRateMax = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, chargeOpRateMax, locationalEligibility
        )
        self.aggregatedChargeOpRateMax = dict.fromkeys(esM.investmentPeriods)

        # chargeOpRateFix
        self.fullChargeOpRateFix = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, chargeOpRateFix, locationalEligibility
        )
        self.aggregatedChargeOpRateFix = dict.fromkeys(esM.investmentPeriods)

        # dischargeOpRateMax
        self.dischargeOpRateMax = dischargeOpRateMax
        self.fullDischargeOpRateMax = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, dischargeOpRateMax, locationalEligibility
        )
        self.aggregatedDischargeOpRateMax = {}

        # dischargeOpRateFix
        self.dischargeOpRateFix = dischargeOpRateFix
        self.fullDischargeOpRateFix = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, dischargeOpRateFix, locationalEligibility
        )
        self.aggregatedDischargeOpRateFix = dict.fromkeys(esM.investmentPeriods)

        # check for chargeOpRateMax and chargeOpRateFix
        for ip in esM.investmentPeriods:
            if (
                self.fullChargeOpRateMax[ip] is not None
                and self.fullChargeOpRateFix[ip] is not None
            ):
                self.fullChargeOpRateMax[ip] = None
                if esM.verbose < 2:
                    warnings.warn(
                        "If chargeOpRateFix is specified, the chargeOpRateMax parameter is not required.\n"
                        + "The chargeOpRateMax time series was set to None."
                    )

        # partLoadMin
        self.processedPartLoadMin = utils.checkAndSetPartLoadMin(
            esM,
            name,
            partLoadMin,
            self.fullChargeOpRateMax,
            self.fullChargeOpRateFix,
            self.bigM,
            self.hasCapacityVariable,
        )

        utils.isPositiveNumber(dischargeTsaWeight)
        self.dischargeTsaWeight = dischargeTsaWeight
        utils.isPositiveNumber(chargeTsaWeight)
        self.chargeTsaWeight = chargeTsaWeight

        timeSeriesData = {}
        for ip in esM.investmentPeriods:
            tsNb = sum(
                [
                    0 if data is None else 1
                    for data in [
                        self.fullChargeOpRateMax[ip],
                        self.fullChargeOpRateFix[ip],
                        self.fullDischargeOpRateMax[ip],
                        self.fullDischargeOpRateFix[ip],
                    ]
                ]
            )
            if tsNb > 0:
                timeSeriesData[ip] = sum(
                    [
                        data
                        for data in [
                            self.fullChargeOpRateMax[ip],
                            self.fullChargeOpRateFix[ip],
                            self.fullDischargeOpRateMax[ip],
                            self.fullDischargeOpRateFix[ip],
                        ]
                        if data is not None
                    ]
                )

        self.processedLocationalEligibility = utils.setLocationalEligibility(
            esM,
            self.locationalEligibility,
            self.processedCapacityMax,
            self.processedCapacityFix,
            self.isBuiltFix,
            self.hasCapacityVariable,
            timeSeriesData,
        )

        # set parameter to None if all years have None values
        self.fullChargeOpRateFix = utils.setParamToNoneIfNoneForAllYears(
            self.fullChargeOpRateFix
        )
        self.fullChargeOpRateMax = utils.setParamToNoneIfNoneForAllYears(
            self.fullChargeOpRateMax
        )
        self.fullDischargeOpRateFix = utils.setParamToNoneIfNoneForAllYears(
            self.fullDischargeOpRateFix
        )
        self.fullDischargeOpRateMax = utils.setParamToNoneIfNoneForAllYears(
            self.fullDischargeOpRateMax
        )

    def setTimeSeriesData(self, hasTSA):
        """
        Function for setting the maximum operation rate and fixed operation rate for charging and discharging
        depending on whether a time series analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        self.processedChargeOpRateMax = (
            self.aggregatedChargeOpRateMax if hasTSA else self.fullChargeOpRateMax
        )
        self.processedChargeOpRateFix = (
            self.aggregatedChargeOpRateFix if hasTSA else self.fullChargeOpRateFix
        )
        self.processedDischargeOpRateMax = (
            self.aggregatedDischargeOpRateMax if hasTSA else self.fullDischargeOpRateMax
        )
        self.processedDischargeOpRateFix = (
            self.aggregatedDischargeOpRateFix if hasTSA else self.fullDischargeOpRateFix
        )

    def getDataForTimeSeriesAggregation(self, ip):
        """Function for getting the required data if a time series aggregation is requested.

        :param ip: investment period of transformation path analysis.
        :type ip: int

        """
        weightDict, data = {}, []
        I = [
            (
                self.fullChargeOpRateFix,
                self.fullChargeOpRateMax,
                "chargeRate_",
                self.chargeTsaWeight,
            ),
            (
                self.fullDischargeOpRateFix,
                self.fullDischargeOpRateMax,
                "dischargeRate_",
                self.dischargeTsaWeight,
            ),
        ]

        for rateFix, rateMax, rateName, rateWeight in I:
            weightDict, data = self.prepareTSAInput(
                rateFix, rateMax, rateName, rateWeight, weightDict, data, ip
            )
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data, ip):
        """
        Function for determining the aggregated maximum rate and the aggregated fixed operation rate for charging
        and discharging.

        :param data: Pandas DataFrame with the clustered time series data of the source component
        :type data: Pandas DataFrame

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """

        self.aggregatedChargeOpRateFix[ip] = self.getTSAOutput(
            self.fullChargeOpRateFix, "chargeRate_", data, ip
        )
        self.aggregatedChargeOpRateMax[ip] = self.getTSAOutput(
            self.fullChargeOpRateMax, "chargeRate_", data, ip
        )

        self.aggregatedDischargeOpRateFix[ip] = self.getTSAOutput(
            self.fullDischargeOpRateFix, "dischargeRate_", data, ip
        )
        self.aggregatedDischargeOpRateMax[ip] = self.getTSAOutput(
            self.fullDischargeOpRateMax, "dischargeRate_", data, ip
        )

    def checkProcessedDataSets(self):
        """
        Check processed time series data after applying time series aggregation. If all entries of dictionary are None
        the parameter itself is set to None.
        """
        for parameter in [
            "processedChargeOpRateFix",
            "processedChargeOpRateMax",
            "processedDischargeOpRateFix",
            "processedDischargeOpRateMax",
        ]:
            setattr(
                self,
                parameter,
                utils.setParamToNoneIfNoneForAllYears(getattr(self, parameter)),
            )


class StorageModel(ComponentModel):
    """
    A StorageModel class instance will be instantly created if a Storage class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Storage class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The StorageModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        """ " Constructor for creating a StorageModel class instance"""
        super().__init__()
        self.abbrvName = "stor"
        self.dimension = "1dim"
        self._chargeOperationVariablesOptimum = {}
        self._dischargeOperationVariablesOptimum = {}
        self._stateOfChargeOperationVariablesOptimum = {}

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """
        Declare sets: design variable sets, operation variable set, operation mode sets.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        compDict = self.componentsDict

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

        if pyM.hasTSA:
            varSet = getattr(pyM, "operationVarSet_" + self.abbrvName)

            def initVarSimpleTSASet(pyM):
                return (
                    (loc, compName, ip)
                    for loc, compName, ip in varSet
                    if not compDict[compName].doPreciseTsaModeling
                )

            setattr(
                pyM,
                "varSetSimple_" + self.abbrvName,
                pyomo.Set(dimen=3, initialize=initVarSimpleTSASet),
            )

            def initVarPreciseTSASet(pyM):
                return (
                    (loc, compName, ip)
                    for loc, compName, ip in varSet
                    if compDict[compName].doPreciseTsaModeling
                )

            setattr(
                pyM,
                "varSetPrecise_" + self.abbrvName,
                pyomo.Set(dimen=3, initialize=initVarPreciseTSASet),
            )

        def initOffsetUpSet(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in getattr(
                    pyM, "operationVarSet_" + self.abbrvName
                )
                if compDict[compName].socOffsetUp >= 0
            )

        setattr(
            pyM,
            "varSetOffsetUp_" + self.abbrvName,
            pyomo.Set(dimen=3, initialize=initOffsetUpSet),
        )

        def initOffsetDownSet(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in getattr(
                    pyM, "operationVarSet_" + self.abbrvName
                )
                if compDict[compName].socOffsetDown >= 0
            )

        setattr(
            pyM,
            "varSetOffsetDown_" + self.abbrvName,
            pyomo.Set(dimen=3, initialize=initOffsetDownSet),
        )

        # Declare sets for case differentiation of operating modes
        # * Charge operation
        self.declareOperationModeSets(
            pyM,
            "chargeOpConstrSet",
            "processedChargeOpRateMax",
            "processedChargeOpRateFix",
        )
        # * Discharge operation
        self.declareOperationModeSets(
            pyM,
            "dischargeOpConstrSet",
            "processedDischargeOpRateMax",
            "processedDischargeOpRateFix",
        )

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

        # Capacity variables [commodityUnit*hour]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not
        self.declareBinaryDesignDecisionVars(pyM, relaxIsBuiltBinary)
        # Energy amount injected into a storage (before injection efficiency losses) between two time steps
        self.declareOperationVars(
            pyM,
            esM,
            "chargeOp",
            "processedChargeOpRateFix",
            "processedChargeOpRateMax",
            relevanceThreshold=relevanceThreshold,
        )
        # Energy amount delivered from a storage (after delivery efficiency losses) between two time steps
        self.declareOperationVars(
            pyM,
            esM,
            "dischargeOp",
            "processedDischargeOpRateFix",
            "processedDischargeOpRateMax",
            relevanceThreshold=relevanceThreshold,
        )
        # Operation of component as binary [1/0]
        self.declareOperationBinaryVars(pyM, "chargeOp_bin")
        self.declareOperationBinaryVars(pyM, "dischargeOp_bin")
        # Capacity development variables [physicalUnit]
        self.declareCommissioningVars(pyM, esM)
        self.declareDecommissioningVars(pyM, esM)

        # Inventory of storage components [commodityUnit*hour]
        if not pyM.hasTSA:
            # Energy amount stored at the beginning of a time step during the (one) period (the i-th state of charge
            # refers to the state of charge at the beginning of the i-th time step, the last index is the state of
            # charge after the last time step)
            setattr(
                pyM,
                "stateOfCharge_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "operationVarSet_" + self.abbrvName),
                    pyM.interTimeStepsSet,
                    domain=pyomo.NonNegativeReals,
                ),
            )
            # Variables to allow a relaxation of the inter period storage connection
            setattr(
                pyM,
                "stateOfChargeOffsetUp_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "varSetOffsetUp_" + self.abbrvName),
                    esM.periods,
                    domain=pyomo.NonNegativeReals,
                ),
            )
            setattr(
                pyM,
                "stateOfChargeOffsetDown_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "varSetOffsetDown_" + self.abbrvName),
                    esM.periods,
                    domain=pyomo.NonNegativeReals,
                ),
            )
        else:

            def SOCBounds(pyM, loc, compName, ip, p, t):
                # Replaces the intraSOCstart constraint:
                # Declare the constraint that the (virtual) state of charge at the beginning of a typical period is zero.
                if t == 0:
                    return (0, 0)
                else:
                    return (None, None)

            # (Virtual) energy amount stored during a period (the i-th state of charge refers to the state of charge at
            # the beginning of the i-th time step, the last index is the state of charge after the last time step)
            setattr(
                pyM,
                "stateOfCharge_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "operationVarSet_" + self.abbrvName),
                    pyM.interTimeStepsSet,
                    domain=pyomo.Reals,
                    bounds=SOCBounds,
                ),
            )
            # (Virtual) minimum amount of energy stored within a period
            setattr(
                pyM,
                "stateOfChargeMin_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "operationVarSet_" + self.abbrvName),
                    esM.typicalPeriods,
                    domain=pyomo.Reals,
                ),
            )
            # (Virtual) maximum amount of energy stored within a period
            setattr(
                pyM,
                "stateOfChargeMax_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "operationVarSet_" + self.abbrvName),
                    esM.typicalPeriods,
                    domain=pyomo.Reals,
                ),
            )
            # (Real) energy amount stored at the beginning of a period between periods(the i-th state of charge refers
            # to the state of charge at the beginning of the i-th period, the last index is the state of charge after
            # the last period)
            setattr(
                pyM,
                "stateOfChargeInterPeriods_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "operationVarSet_" + self.abbrvName),
                    pyM.investPeriodInterPeriodSet,
                    domain=pyomo.NonNegativeReals,
                ),
            )
            # Variables to allow a relaxation of the inter period storage connection
            setattr(
                pyM,
                "stateOfChargeOffsetUp_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "varSetOffsetUp_" + self.abbrvName),
                    esM.interPeriodTimeSteps,
                    domain=pyomo.NonNegativeReals,
                ),
            )
            setattr(
                pyM,
                "stateOfChargeOffsetDown_" + self.abbrvName,
                pyomo.Var(
                    getattr(pyM, "varSetOffsetDown_" + self.abbrvName),
                    esM.interPeriodTimeSteps,
                    domain=pyomo.NonNegativeReals,
                ),
            )

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def connectSOCs(self, pyM, esM):
        """
        Declare the constraint for connecting the state of charge with the charge and discharge operation:
        the change in the state of charge between two points in time has to match the values of charging and
        discharging (considering the efficiencies of these processes) within the time step in between minus
        the self-discharge of the storage.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            SoC^{comp}_{loc,ip,p,t+1} - \\left( SoC^{comp}_{loc,ip,p,t} \\left( 1 - \\eta^{self-discharge} \\right)^{\\frac{\\tau^{hours}}{h}} + op^{comp,charge}_{loc,ip,p,t} \\eta^{charge} - op^{comp,discharge}_{loc,ip,p,t} / \\eta^{discharge} \\right) = 0
            \\end{eqnarray*}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOC = getattr(pyM, "stateOfCharge_" + abbrvName)
        chargeOp, dischargeOp = (
            getattr(pyM, "chargeOp_" + abbrvName),
            getattr(pyM, "dischargeOp_" + abbrvName),
        )
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def connectSOCs(pyM, loc, compName, ip, p, t):
            if not pyM.hasSegmentation:
                return (
                    SOC[loc, compName, ip, p, t + 1]
                    - SOC[loc, compName, ip, p, t]
                    * (1 - compDict[compName].selfDischarge) ** esM.hoursPerTimeStep
                    == chargeOp[loc, compName, ip, p, t]
                    * compDict[compName].chargeEfficiency
                    - dischargeOp[loc, compName, ip, p, t]
                    / compDict[compName].dischargeEfficiency
                )
            else:
                return (
                    SOC[loc, compName, ip, p, t + 1]
                    - SOC[loc, compName, ip, p, t]
                    * (1 - compDict[compName].selfDischarge)
                    ** esM.hoursPerSegment[ip].to_dict()[p, t]
                    == chargeOp[loc, compName, ip, p, t]
                    * compDict[compName].chargeEfficiency
                    - dischargeOp[loc, compName, ip, p, t]
                    / compDict[compName].dischargeEfficiency
                )

        setattr(
            pyM,
            "ConstrConnectSOC_" + abbrvName,
            pyomo.Constraint(opVarSet, pyM.intraYearTimeSet, rule=connectSOCs),
        )

    def cyclicState(self, pyM, esM):
        """
        Declare the constraint for connecting the states of charge: the state of charge at the beginning of a period
        has to be the same as the state of charge in the end of that period.

        with full temporal resolution

        .. math::
            SoC^{comp}_{loc,ip,0,0} = SoC^{comp}_{loc,ip,0,t^{total}}

        with time series aggregation:

        .. math::
            SoC^{inter}_{loc,ip,0} = SoC^{inter}_{loc,ip,p^{total}}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        abbrvName = self.abbrvName
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)
        SOC = getattr(pyM, "stateOfCharge_" + abbrvName)
        offsetUp = getattr(pyM, "stateOfChargeOffsetUp_" + abbrvName)
        offsetDown = getattr(pyM, "stateOfChargeOffsetDown_" + abbrvName)

        if not pyM.hasTSA:

            def cyclicState(pyM, loc, compName, ip, p):
                offsetUp_ = (
                    offsetUp[loc, compName, 0] if (loc, compName, 0) in offsetUp else 0
                )
                offsetDown_ = (
                    offsetDown[loc, compName, 0]
                    if (loc, compName, 0) in offsetDown
                    else 0
                )
                return SOC[loc, compName, ip, 0, 0] == SOC[
                    loc, compName, ip, 0, esM.timeStepsPerPeriod[-1] + 1
                ] + (offsetUp_ - offsetDown_)

        else:
            SOCInter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)

            # tests for testing the storage class with ip and TSAM
            def cyclicState(pyM, loc, compName, ip, p):
                # tLast = esM.interPeriodTimeSteps[-1]
                tLast = esM.numberOfInterPeriodTimeSteps
                offsetUp_ = (
                    offsetUp[loc, compName, tLast]
                    if (loc, compName, tLast) in offsetUp
                    else 0
                )
                offsetDown_ = (
                    offsetDown[loc, compName, tLast]
                    if (loc, compName, tLast) in offsetDown
                    else 0
                )
                return SOCInter[loc, compName, ip, 0] == SOCInter[
                    loc, compName, ip, tLast
                ] + (offsetUp_ - offsetDown_)

        setattr(
            pyM,
            "ConstrCyclicState_" + abbrvName,
            pyomo.Constraint(
                opVarSet, pyM.investPeriodInterPeriodSet, rule=cyclicState
            ),
        )

    def cyclicLifetime(self, pyM, esM):
        """
        Declare the constraint for limiting the number of full cycle equivalents to stay below cyclic lifetime.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            & & op^{comp,charge}_{loc,annual} \\leq \\left( \\text{SoC}^{max} - \\text{SoC}^{min} \\right) \\cdot cap^{comp}_{loc,ip} \\cdot \\frac{t^{ \\text{comp,cyclic lifetime}}}{\\tau^{ \\text{comp,economic lifetime}}_{loc}} \\\\
            \\text{with} \\\\
            & & op^{comp,charge}_{loc,annual} = \\sum_{(ip,p,t) \\in \\mathcal{P} \\times \\mathcal{T}} op^{comp,charge}_{loc,ip,p,t} \\cdot freq(p) / \\tau^{years}
            \\end{eqnarray*}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        chargeOp, capVar = (
            getattr(pyM, "chargeOp_" + abbrvName),
            getattr(pyM, "cap_" + abbrvName),
        )
        capVarSet = getattr(pyM, "designDimensionVarSet_" + abbrvName)

        def cyclicLifetime(pyM, loc, compName, ip):
            return (
                sum(
                    chargeOp[loc, compName, ip, p, t] * esM.periodOccurrences[ip][p]
                    for ip, p, t in pyM.timeSet
                )
                / esM.numberOfYears
                <= capVar[loc, compName, ip]
                * (
                    compDict[compName].stateOfChargeMax
                    - compDict[compName].stateOfChargeMin
                )
                * compDict[compName].cyclicLifetime
                / compDict[compName].economicLifetime[loc]
                if compDict[compName].cyclicLifetime is not None
                else pyomo.Constraint.Skip
            )

        setattr(
            pyM,
            "ConstrCyclicLifetime_" + abbrvName,
            pyomo.Constraint(capVarSet, rule=cyclicLifetime),
        )

    def connectInterPeriodSOC(self, pyM, esM):
        """
        Declare the constraint that the state of charge at the end of each period has to be equivalent to the state of
        charge of the period before it (minus its self discharge) plus the change in the state of charge which
        happened during the typical period which was assigned to that period.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            SoC^{inter}_{loc,ip,p+1} - SoC^{inter}_{loc,ip,p} \\cdot \\left( 1 - \\eta^{self-discharge} \\right)^{\\frac{t^{\\text{per period}} \cdot \\tau^{hours}}{h}}
            \\ SoC^{comp}_{loc,ip,map(p),t^{\\text{per period}}} = 0
            \\end{eqnarray*}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)
        SOC = getattr(pyM, "stateOfCharge_" + abbrvName)
        SOCInter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
        offsetUp = getattr(pyM, "stateOfChargeOffsetUp_" + abbrvName)
        offsetDown = getattr(pyM, "stateOfChargeOffsetDown_" + abbrvName)

        def connectInterSOC(pyM, loc, compName, ip, pInter):
            offsetUp_ = (
                offsetUp[loc, compName, pInter]
                if (loc, compName, pInter) in offsetUp
                else 0
            )
            offsetDown_ = (
                offsetDown[loc, compName, pInter]
                if (loc, compName, pInter) in offsetDown
                else 0
            )
            if not esM.pyM.hasSegmentation:
                return SOCInter[loc, compName, ip, pInter + 1] == SOCInter[
                    loc, compName, ip, pInter
                ] * (1 - compDict[compName].selfDischarge) ** (
                    (esM.timeStepsPerPeriod[-1] + 1) * esM.hoursPerTimeStep
                ) + SOC[
                    loc,
                    compName,
                    ip,
                    esM.periodsOrder[ip][pInter],
                    esM.timeStepsPerPeriod[-1] + 1,
                ] + (
                    offsetUp_ - offsetDown_
                )
            else:
                # return SOCInter[loc, compName, pInter + 1] == \
                #     SOCInter[loc, compName, pInter] * (1 - compDict[compName].selfDischarge) ** \
                #     ((esM.timeStepsPerPeriod[-1] + 1) * esM.hoursPerTimeStep) + \
                #     SOC[loc, compName, esM.periodsOrder[pInter], esM.segmentsPerPeriod[-1] + 1] + \
                #     (offsetUp_ - offsetDown_)
                return SOCInter[loc, compName, ip, pInter + 1] == SOCInter[
                    loc, compName, ip, pInter
                ] * (1 - compDict[compName].selfDischarge) ** (
                    (esM.timeStepsPerPeriod[-1] + 1) * esM.hoursPerTimeStep
                ) + SOC[
                    loc,
                    compName,
                    ip,
                    esM.periodsOrder[ip][pInter],
                    esM.segmentsPerPeriod[-1] + 1,
                ] + (
                    offsetUp_ - offsetDown_
                )

        setattr(
            pyM,
            "ConstrInterSOC_" + abbrvName,
            pyomo.Constraint(opVarSet, esM.periods, rule=connectInterSOC),
        )

    def intraSOCstart(self, pyM, esM):
        """
        Declare the constraint that the (virtual) state of charge at the beginning of a typical period is zero.

        .. math::

            SoC^{comp}_{loc,ip,p,0} = 0

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        abbrvName = self.abbrvName
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)
        SOC = getattr(pyM, "stateOfCharge_" + abbrvName)

        def intraSOCstart(pyM, loc, compName, ip, p):
            return SOC[loc, compName, ip, p, 0] == 0

        setattr(
            pyM,
            "ConstrSOCPeriodStart_" + abbrvName,
            pyomo.Constraint(opVarSet, esM.typicalPeriods, rule=intraSOCstart),
        )

    def equalInterSOC(self, pyM, esM):
        """
        Declare the constraint that, if periodic storage is selected, the states of charge between periods
        have the same value.

        .. math::

            SoC^{comp,inter}_{ip,p} = SoC^{comp,inter}_{ip,p+1}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)
        SOCInter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)

        def equalInterSOC(pyM, loc, compName, ip, pInter):
            return (
                SOCInter[loc, compName, ip, pInter]
                == SOCInter[loc, compName, ip, pInter + 1]
                if compDict[compName].isPeriodicalStorage
                else pyomo.Constraint.Skip
            )

        setattr(
            pyM,
            "ConstrEqualInterSOC_" + abbrvName,
            pyomo.Constraint(opVarSet, esM.periods, rule=equalInterSOC),
        )

    def minSOC(self, pyM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] has to be larger than the
        installed capacity [commodityUnit*h] multiplied with the relative minimum state of charge.

        .. math::

            SoC^{comp,min} \\cdot cap^{comp}_{loc,ip} \\leq SoC^{comp}_{loc,ip,0,t}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVarSet = getattr(pyM, "designDimensionVarSet_" + abbrvName)
        SOC, capVar = (
            getattr(pyM, "stateOfCharge_" + abbrvName),
            getattr(pyM, "cap_" + abbrvName),
        )

        def SOCMin(pyM, loc, compName, ip, p, t):
            return (
                SOC[loc, compName, ip, p, t]
                >= capVar[loc, compName, ip] * compDict[compName].stateOfChargeMin
            )

        setattr(
            pyM,
            "ConstrSOCMin_" + abbrvName,
            pyomo.Constraint(capVarSet, pyM.intraYearTimeSet, rule=SOCMin),
        )

    def limitSOCwithSimpleTsa(self, pyM, esM):
        """
        Simplified version of the state of charge limitation control.
        The error compared to the precise version is small in cases of small selfDischarge.

        .. math:: 
            :nowrap:

            \\begin{eqnarray*}
            & & \\underline{SoC}^{comp,sup}_{loc,ip,p,t} \\geq \\text{SoC}^{min} \\cdot cap^{comp}_{loc,ip} \\\\  
            & & \overline{SoC}^{comp,sup}_{loc,ip,p,t} \\leq \\text{SoC}^{max} \\cdot cap^{comp}_{loc,ip} \\\\
            \\text{with } \\\\ 
            & & \\underline{SoC}^{comp,sup}_{loc,ip,p,t} = SoC^{inter}_{loc,ip,p} \\cdot (1 - \\eta^{\\text{self-discharge}})^{\\frac{t^{\\text{per period}} \\cdot \\tau^{hours}}{h}}+ SoC^{min}_{loc,ip,map(p)} \\\\
            & &\\overline{SoC}^{comp,sup}_{loc,ip,p,t} = SoC^{inter}_{loc,ip,p} + SoC^{max}_{loc,ip,map(p)}
            \\end{eqnarray*}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSimpleSet = getattr(pyM, "varSetSimple_" + abbrvName)
        SOC, capVar = (
            getattr(pyM, "stateOfCharge_" + abbrvName),
            getattr(pyM, "cap_" + abbrvName),
        )
        SOCmax, SOCmin = (
            getattr(pyM, "stateOfChargeMax_" + abbrvName),
            getattr(pyM, "stateOfChargeMin_" + abbrvName),
        )
        SOCInter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)

        # The maximum (virtual) state of charge during a typical period is larger than all occurring (virtual)
        # states of charge in that period (the last time step is considered in the subsequent period for t=0).
        def SOCintraPeriodMax(pyM, loc, compName, ip, p, t):
            return SOC[loc, compName, ip, p, t] <= SOCmax[loc, compName, ip, p]

        setattr(
            pyM,
            "ConstSOCintraPeriodMax_" + abbrvName,
            pyomo.Constraint(
                varSimpleSet, pyM.intraYearTimeSet, rule=SOCintraPeriodMax
            ),
        )

        # The minimum (virtual) state of charge during a typical period is smaller than all occurring (virtual)
        # states of charge in that period (the last time step is considered in the subsequent period for t=0).
        def SOCintraPeriodMin(pyM, loc, compName, ip, p, t):
            return SOC[loc, compName, ip, p, t] >= SOCmin[loc, compName, ip, p]

        setattr(
            pyM,
            "ConstSOCintraPeriodMin_" + abbrvName,
            pyomo.Constraint(
                varSimpleSet, pyM.intraYearTimeSet, rule=SOCintraPeriodMin
            ),
        )

        # The state of charge at the beginning of one period plus the maximum (virtual) state of charge
        # during that period has to be smaller than the installed capacities multiplied with the relative maximum
        # state of charge.
        def SOCMaxSimple(pyM, loc, compName, ip, pInter):
            if compDict[compName].hasCapacityVariable:
                return (
                    SOCInter[loc, compName, ip, pInter]
                    + SOCmax[loc, compName, ip, esM.periodsOrder[ip][pInter]]
                    <= capVar[loc, compName, ip] * compDict[compName].stateOfChargeMax
                )
            else:
                pyomo.Constraint.Skip

        setattr(
            pyM,
            "ConstrSOCMaxSimple_" + abbrvName,
            pyomo.Constraint(varSimpleSet, esM.periods, rule=SOCMaxSimple),
        )

        # The state of charge at the beginning of one period plus the minimum (virtual) state of charge
        # during that period has to be larger than the installed capacities multiplied with the relative minimum
        # state of charge.
        def SOCMinSimple(pyM, loc, compName, ip, pInter):
            if compDict[compName].hasCapacityVariable:
                return (
                    SOCInter[loc, compName, ip, pInter]
                    * (1 - compDict[compName].selfDischarge)
                    ** ((esM.timeStepsPerPeriod[-1] + 1) * esM.hoursPerTimeStep)
                    + SOCmin[loc, compName, ip, esM.periodsOrder[ip][pInter]]
                    >= capVar[loc, compName, ip] * compDict[compName].stateOfChargeMin
                )
            else:
                return (
                    SOCInter[loc, compName, ip, pInter]
                    * (1 - compDict[compName].selfDischarge)
                    ** ((esM.timeStepsPerPeriod[-1] + 1) * esM.hoursPerTimeStep)
                    + SOCmin[loc, compName, ip, esM.periodsOrder[ip][pInter]]
                    >= compDict[compName].stateOfChargeMin
                )

        setattr(
            pyM,
            "ConstrSOCMinSimple_" + abbrvName,
            pyomo.Constraint(varSimpleSet, esM.periods, rule=SOCMinSimple),
        )

    def operationModeSOC(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        [commodityUnit*h] and the relative maximum state of charge [-].

        .. math::

            SoC^{comp}_{loc,ip,0,t} \\leq \\text{SoC}^{comp,max} \\cdot cap^{comp}_{loc,ip}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = (
            getattr(pyM, "stateOfCharge_" + abbrvName),
            getattr(pyM, "cap_" + abbrvName),
        )
        constrSet = getattr(pyM, "designDimensionVarSet_" + abbrvName)

        # Operation [commodityUnit*h] limited by the installed capacity [commodityUnit*h] multiplied by the relative
        # maximum state of charge.
        def op(pyM, loc, compName, ip, p, t):
            return (
                opVar[loc, compName, ip, p, t]
                <= compDict[compName].stateOfChargeMax * capVar[loc, compName, ip]
            )

        setattr(
            pyM,
            "ConstrSOCMaxPrecise_" + abbrvName,
            pyomo.Constraint(constrSet, pyM.intraYearTimeSet, rule=op),
        )

    def operationModeSOCwithTSA(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        # [commodityUnit*h] and the relative maximum state of charge [-].

        .. math::

            SoC^{inter}_{loc,ip,p} \cdot (1 - \eta^{\\text{self-discharge}})^{\\frac{t \cdot \\tau^{hours}}{h}} + SoC^{comp}_{loc,ip,map(p),t} \leq \\text{SoC}^{max} \cdot cap^{comp}_{loc,ip}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
        SOC, capVar = (
            getattr(pyM, "stateOfCharge_" + abbrvName),
            getattr(pyM, "cap_" + abbrvName),
        )
        constrSet = getattr(pyM, "designDimensionVarSet_" + abbrvName)

        def SOCMaxPrecise(pyM, loc, compName, ip, pInter, t):
            if compDict[compName].doPreciseTsaModeling:
                if not pyM.hasSegmentation:
                    return (
                        SOCinter[loc, compName, ip, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (t * esM.hoursPerTimeStep)
                        )
                        + SOC[loc, compName, ip, esM.periodsOrder[ip][pInter], t]
                        <= capVar[loc, compName, ip]
                        * compDict[compName].stateOfChargeMax
                    )
                else:
                    return (
                        SOCinter[loc, compName, ip, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (
                                esM.segmentStartTime[ip].to_dict()[
                                    esM.periodsOrder[ip][pInter], t
                                ]
                                * esM.hoursPerTimeStep
                            )
                        )
                        + SOC[loc, compName, ip, esM.periodsOrder[ip][pInter], t]
                        <= capVar[loc, compName, ip]
                        * compDict[compName].stateOfChargeMax
                    )
            else:
                return pyomo.Constraint.Skip

        setattr(
            pyM,
            "ConstrSOCMaxPrecise_" + abbrvName,
            pyomo.Constraint(
                constrSet,
                pyM.intraYearTimeSet,
                rule=SOCMaxPrecise,
            ),
        )

    def minSOCwithTSAprecise(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] at each time step cannot be smaller
        than the installed capacity [commodityUnit*h] multiplied with the relative minimum state of charge [-].

        .. math::

            \\text{SoC}^{min} \\cdot cap^{comp}_{loc,ip} \\leq SoC^{inter}_{loc,ip,p} \\cdot (1 - \\eta^{\\text{self-discharge}})^{\\frac{t \\cdot \\tau^{hours}}{h}} + SoC^{comp}_{loc,ip,map(p),t}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
        SOC, capVar = (
            getattr(pyM, "stateOfCharge_" + abbrvName),
            getattr(pyM, "cap_" + abbrvName),
        )
        preciseSet = getattr(pyM, "varSetPrecise_" + abbrvName)

        def SOCMinPrecise(pyM, loc, compName, ip, pInter, t):
            if compDict[compName].hasCapacityVariable:
                if not pyM.hasSegmentation:
                    return (
                        SOCinter[loc, compName, ip, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (t * esM.hoursPerTimeStep)
                        )
                        + SOC[loc, compName, ip, esM.periodsOrder[ip][pInter], t]
                        >= capVar[loc, compName, ip]
                        * compDict[compName].stateOfChargeMin
                    )
                else:
                    return (
                        SOCinter[loc, compName, ip, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (
                                esM.segmentStartTime[ip].to_dict()[
                                    esM.periodsOrder[ip][pInter], t
                                ]
                                * esM.hoursPerTimeStep
                            )
                        )
                        + SOC[loc, compName, ip, esM.periodsOrder[ip][pInter], t]
                        >= capVar[loc, compName, ip]
                        * compDict[compName].stateOfChargeMin
                    )
            else:
                if not pyM.hasSegmentation:
                    return (
                        SOCinter[loc, compName, ip, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (t * esM.hoursPerTimeStep)
                        )
                        + SOC[loc, compName, ip, esM.periodsOrder[ip][pInter], t]
                        >= compDict[compName].stateOfChargeMin
                    )
                else:
                    return (
                        SOCinter[loc, compName, ip, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (
                                esM.segmentStartTime[ip].to_dict()[
                                    esM.periodsOrder[ip][pInter], t
                                ]
                                * esM.hoursPerTimeStep
                            )
                        )
                        + SOC[loc, compName, ip, esM.periodsOrder[ip][pInter], t]
                        >= compDict[compName].stateOfChargeMin
                    )

        setattr(
            pyM,
            "ConstrSOCMinPrecise_" + abbrvName,
            pyomo.Constraint(
                preciseSet,
                esM.periods,
                esM.timeStepsPerPeriod,
                rule=SOCMinPrecise,
            ),
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
        # Sets, if applicable, the installed capacities of a component
        self.capacityFix(pyM, esM)
        # Sets, if applicable, the binary design variables of a component
        self.designBinFix(pyM)

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

        # Constraint for connecting the state of charge with the charge and discharge operation
        self.connectSOCs(pyM, esM)

        #                              Constraints for enforcing charging operation modes                              #

        # Charging of storage [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] multiplied by
        # the hours per time step [h] and the charging rate factor [1/h]
        self.operationMode1(
            pyM, esM, "ConstrCharge", "chargeOpConstrSet", "chargeOp", "chargeRate"
        )
        # Charging of storage [commodityUnit*h] is equal to the installed capacity [commodityUnit*h] multiplied by
        # the hours per time step [h] and the charging operation time series [1/h]
        self.operationMode2(
            pyM,
            esM,
            "ConstrCharge",
            "chargeOpConstrSet",
            "chargeOp",
            "processedChargeOpRateFix",
        )
        # Charging of storage [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] multiplied by
        # the hours per time step [h] and the charging operation time series [1/h]
        self.operationMode3(
            pyM,
            esM,
            "ConstrCharge",
            "chargeOpConstrSet",
            "chargeOp",
            "processedChargeOpRateMax",
        )

        # Operation [physicalUnit*h] is limited by minimum part Load
        self.additionalMinPartLoad(
            pyM,
            esM,
            "ConstrCharge",
            "chargeOpConstrSet",
            "chargeOp",
            "chargeOp_bin",
            "cap",
        )

        #                             Constraints for enforcing discharging operation modes                            #

        # Discharging of storage [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] multiplied
        # by the hours per time step [h] and the discharging rate factor [1/h]
        self.operationMode1(
            pyM,
            esM,
            "ConstrDischarge",
            "dischargeOpConstrSet",
            "dischargeOp",
            "dischargeRate",
        )
        # Discharging of storage [commodityUnit*h] is equal to the installed capacity [commodityUnit*h] multiplied
        # by the hours per time step [h] and the discharging operation time series [1/h]
        self.operationMode2(
            pyM,
            esM,
            "ConstrDischarge",
            "dischargeOpConstrSet",
            "dischargeOp",
            "processedDischargeOpRateFix",
        )
        # Discharging of storage [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] multiplied
        # by the hours per time step [h] and the discharging operation time series [1/h]
        self.operationMode3(
            pyM,
            esM,
            "ConstrDischarge",
            "dischargeOpConstrSet",
            "dischargeOp",
            "processedDischargeOpRateMax",
        )
        # Operation [physicalUnit*h] is limited by minimum part Load
        self.additionalMinPartLoad(
            pyM,
            esM,
            "ConstrDischarge",
            "dischargeOpConstrSet",
            "dischargeOp",
            "dischargeOp_bin",
            "cap",
        )

        # Cyclic constraint enforcing that all storages have the same state of charge at the the beginning of the first
        # and the end of the last time step
        self.cyclicState(pyM, esM)

        # Constraint for limiting the number of full cycle equivalents to stay below cyclic lifetime
        self.cyclicLifetime(pyM, esM)

        if pyM.hasTSA:
            # The state of charge at the end of each period is equivalent to the state of charge of the period before it
            # (minus its self discharge) plus the change in the state of charge which happened during the typical
            # period which was assigned to that period
            self.connectInterPeriodSOC(pyM, esM)
            # If periodic storage is selected, the states of charge between periods have the same value
            self.equalInterSOC(pyM, esM)

        # Ensure that the state of charge is within the operating limits of the installed capacities
        if not pyM.hasTSA:
            #              Constraints for enforcing a state of charge operation mode within given limits              #

            # State of charge [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] and the relative
            # maximum state of charge
            self.operationModeSOC(pyM, esM)

            # The state of charge [commodityUnit*h] has to be larger than the installed capacity [commodityUnit*h]
            # multiplied with the relative minimum state of charge
            self.minSOC(pyM)

        else:
            #                       Simplified version of the state of charge limitation control                       #
            #           (The error compared to the precise version is small in cases of small selfDischarge)           #
            self.limitSOCwithSimpleTsa(pyM, esM)

            #                        Precise version of the state of charge limitation control                         #

            # Constraints for enforcing a state of charge operation within given limits

            # State of charge [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] and the
            # relative maximum state of charge
            self.operationModeSOCwithTSA(pyM, esM)

            # The state of charge at each time step cannot be smaller than the installed capacity multiplied with the
            # relative minimum state of charge
            self.minSOCwithTSAprecise(pyM, esM)

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

    def getCommodityBalanceContribution(self, pyM, commod, loc, ip, p, t):
        """Get contribution to a commodity balance.

        .. math::

            \\text{C}^{comp,comm}_{loc,ip,p,t} = op^{comp,discharge}_{loc,ip,p,t} - op^{comp,charge}_{loc,ip,p,t}

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        chargeOp, dischargeOp = (
            getattr(pyM, "chargeOp_" + abbrvName),
            getattr(pyM, "dischargeOp_" + abbrvName),
        )
        opVarDict = getattr(pyM, "operationVarDict_" + abbrvName)
        return sum(
            dischargeOp[loc, compName, ip, p, t] - chargeOp[loc, compName, ip, p, t]
            for compName in opVarDict[ip][loc]
            if commod == self.componentsDict[compName].commodity
        )

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        capexCap = self.getEconomicsDesign(
            pyM,
            esM,
            ["processedInvestPerCapacity"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commis",
            divisorName="CCF",
        )
        capexDec = self.getEconomicsDesign(
            pyM,
            esM,
            ["processedInvestIfBuilt"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commisBin",
            divisorName="CCF",
        )
        opexCap = self.getEconomicsDesign(
            pyM,
            esM,
            ["processedOpexPerCapacity"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commis",
        )
        opexDec = self.getEconomicsDesign(
            pyM,
            esM,
            ["processedOpexIfBuilt"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commisBin",
        )
        opexOp1 = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerChargeOperation"],
            "chargeOp",
            "operationVarDict",
        )
        opexOp2 = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerDischargeOperation"],
            "dischargeOp",
            "operationVarDict",
        )

        offsetUp = getattr(pyM, "stateOfChargeOffsetUp_" + abbrvName)
        offsetDown = getattr(pyM, "stateOfChargeOffsetDown_" + abbrvName)

        offsetUpOp = sum(
            offsetUp[loc, compName, ip, period] * compDict[compName].socOffsetUp
            for loc, compName, ip, period in offsetUp
        )
        offsetDownOp = sum(
            offsetDown[loc, compName, ip, period] * compDict[compName].socOffsetDown
            for loc, compName, ip, period in offsetDown
        )

        return (
            capexCap
            + capexDec
            + opexCap
            + opexDec
            + opexOp1
            + opexOp2
            + offsetUpOp
            + offsetDownOp
        )

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
        chargeOp, dischargeOp = (
            getattr(pyM, "chargeOp_" + abbrvName),
            getattr(pyM, "dischargeOp_" + abbrvName),
        )
        SOC = getattr(pyM, "stateOfCharge_" + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(
            esM, pyM, esM.locations, "commodityUnit", "*h"
        )

        # Get class related results
        resultsTAC_opexOpCharge = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerChargeOperation"],
            "chargeOp",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="TAC",
        )
        resultsNPV_opexOpCharge = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerChargeOperation"],
            "chargeOp",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="NPV",
        )
        resultsTAC_opexOpDischarge = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerDischargeOperation"],
            "dischargeOp",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="TAC",
        )
        resultsNPV_opexOpDischarge = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerDischargeOperation"],
            "dischargeOp",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="NPV",
        )

        for ip in esM.investmentPeriods:
            # Set optimal operation variables and append optimization summary
            props = [
                "operationCharge",
                "operationDischarge",
                "opexCharge",
                "opexDischarge",
                "NPV_opexCharge",
                "NPV_opexDischarge",
            ]
            # Unit dict: Specify units for props
            units = {
                props[0]: ["[-*h]", "[-*h/a]"],
                props[1]: ["[-*h]", "[-*h/a]"],
                props[2]: ["[" + esM.costUnit + "/a]"],
                props[3]: ["[" + esM.costUnit + "/a]"],
                props[4]: ["[" + esM.costUnit + "/a]"],
                props[5]: ["[" + esM.costUnit + "/a]"],
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
                    if x[1] in ["operationCharge", "NPV_operationCharge"]
                    else x,
                    tuples,
                )
            )
            tuples = list(
                map(
                    lambda x: (
                        x[0],
                        x[1],
                        x[2].replace("-", compDict[x[0]].commodityUnit),
                    )
                    if x[1] in ["operationDischarge", "NPV_operationDischarge"]
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

            # * charge variables and contributions
            optVal_charge = utils.formatOptimizationOutput(
                chargeOp.get_values(),
                "operationVariables",
                "1dim",
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )
            self._chargeOperationVariablesOptimum[
                esM.investmentPeriodNames[ip]
            ] = optVal_charge

            if optVal_charge is not None:
                idx = pd.IndexSlice
                optVal_charge = optVal_charge.loc[
                    idx[:, :], :
                ]  # perfect foresight: added ip and deleted again
                # optVal_charge = optVal_charge.droplevel([1])
                opSum = optVal_charge.sum(axis=1).unstack(-1)

                # operation
                optSummary.loc[
                    [
                        (
                            ix,
                            "operationCharge",
                            "[" + compDict[ix].commodityUnit + "*h/a]",
                        )
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = (
                    opSum.values / esM.numberOfYears
                )
                optSummary.loc[
                    [
                        (
                            ix,
                            "operationCharge",
                            "[" + compDict[ix].commodityUnit + "*h]",
                        )
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = opSum.values

                # cost
                tac_oxCharge = resultsTAC_opexOpCharge[ip]
                optSummary.loc[
                    [
                        (ix, "opexCharge", "[" + esM.costUnit + "/a]")
                        for ix in tac_oxCharge.index
                    ],
                    tac_oxCharge.columns,
                ] = tac_oxCharge.values
                npv_oxCharge = resultsNPV_opexOpCharge[ip]
                optSummary.loc[
                    [
                        (ix, "NPV_opexCharge", "[" + esM.costUnit + "/a]")
                        for ix in npv_oxCharge.index
                    ],
                    npv_oxCharge.columns,
                ] = npv_oxCharge.values

            # * discharge variables and contributions
            optVal_discharge = utils.formatOptimizationOutput(
                dischargeOp.get_values(),
                "operationVariables",
                "1dim",
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )
            self._dischargeOperationVariablesOptimum[
                esM.investmentPeriodNames[ip]
            ] = optVal_discharge
            # Check if there are time steps, at which a storage component is both charging and discharging
            for compName in opSum.index:
                simultaneousChargeDischarge = utils.checkSimultaneousChargeDischarge(
                    tsCharge=self._chargeOperationVariablesOptimum[
                        esM.investmentPeriodNames[ip]
                    ].loc[compName],
                    tsDischarge=self._dischargeOperationVariablesOptimum[
                        esM.investmentPeriodNames[ip]
                    ].loc[compName],
                )
                if simultaneousChargeDischarge:
                    if esM.verbose < 2:
                        warnings.warn(
                            "Charge and discharge at the same time for component {}".format(
                                compName
                            ),
                            UserWarning,
                        )

            if optVal_discharge is not None:
                # operation
                opSum = optVal_discharge.sum(axis=1).unstack(-1)
                optSummary.loc[
                    [
                        (
                            ix,
                            "operationDischarge",
                            "[" + compDict[ix].commodityUnit + "*h/a]",
                        )
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = opSum.values
                optSummary.loc[
                    [
                        (
                            ix,
                            "operationDischarge",
                            "[" + compDict[ix].commodityUnit + "*h]",
                        )
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = opSum.values

                # costs
                tac_oxDischarge = resultsTAC_opexOpDischarge[ip]
                optSummary.loc[
                    [
                        (ix, "opexDischarge", "[" + esM.costUnit + "/a]")
                        for ix in tac_oxDischarge.index
                    ],
                    tac_oxDischarge.columns,
                ] = tac_oxDischarge.values
                npv_oxDischarge = resultsNPV_opexOpDischarge[ip]
                optSummary.loc[
                    [
                        (ix, "NPV_opexDischarge", "[" + esM.costUnit + "/a]")
                        for ix in npv_oxDischarge.index
                    ],
                    npv_oxDischarge.columns,
                ] = npv_oxDischarge.values

            # * set state of charge variables
            if not pyM.hasTSA:
                optVal = utils.formatOptimizationOutput(
                    SOC.get_values(),
                    "operationVariables",
                    "1dim",
                    ip,
                    esM.periodsOrder[ip],
                    esM=esM,
                )
                # Remove the last column (by applying the cycle constraint, the first and the last columns are equal to each
                # other)
                optVal = optVal.loc[:, : len(optVal.columns) - 2]
                self._stateOfChargeOperationVariablesOptimum[
                    esM.investmentPeriodNames[ip]
                ] = optVal
                utils.setOptimalComponentVariables(
                    optVal, "_stateOfChargeVariablesOptimum", compDict
                )
            else:
                SOCinter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
                stateOfChargeIntra = SOC.get_values()
                stateOfChargeInter = SOCinter.get_values()
                if stateOfChargeIntra is not None:
                    # Convert dictionary to DataFrame, transpose, put the period column first and sort the index
                    # Results in a one dimensional DataFrame
                    stateOfChargeIntra = (
                        pd.DataFrame(stateOfChargeIntra, index=[0])
                        .T.loc[:, :, ip, :, :]
                        .swaplevel(i=0, j=-2)
                        .sort_index()
                    )
                    stateOfChargeInter = (
                        pd.DataFrame(stateOfChargeInter, index=[0])
                        .T.loc[:, :, ip, :]
                        .swaplevel(i=0, j=1)
                        .sort_index()
                    )
                    # Unstack time steps (convert to a two dimensional DataFrame with the time indices being the columns)
                    stateOfChargeIntra = stateOfChargeIntra.unstack(level=-1)
                    stateOfChargeInter = stateOfChargeInter.unstack(level=-1)
                    # Get rid of the unnecessary 0 level
                    stateOfChargeIntra.columns = stateOfChargeIntra.columns.droplevel()
                    stateOfChargeInter.columns = stateOfChargeInter.columns.droplevel()
                    # If segmentation is chosen, the segments of each period need to be unravelled to the original number of
                    # time steps first
                    if esM.segmentation:
                        dataAllPeriods = []
                        for p in esM.typicalPeriods:
                            # Repeat each segment in each period as often as time steps are represented by the corresponding
                            # segment
                            repList = esM.timeStepsPerSegment[ip].loc[p, :].tolist()
                            # repList = esM.timeStepsPerSegment.loc[p, :].tolist()
                            dataPeriod = pd.DataFrame(
                                np.repeat(
                                    stateOfChargeIntra.loc[p]
                                    .loc[:, : esM.segmentsPerPeriod[-1]]
                                    .values,
                                    repList,
                                    axis=1,
                                ),
                                index=stateOfChargeIntra.xs(
                                    p, level=0, drop_level=False
                                ).index,
                            )
                            dataAllPeriods.append(dataPeriod)
                        # Concat data to multiindex dataframe with periods, components and locations as indices and inner-
                        # period time steps as columns
                        stateOfChargeIntra = pd.concat(dataAllPeriods, axis=0)
                    # Concat data according to periods order to cover the full time horizon
                    data = []
                    for count, p in enumerate(esM.periodsOrder[ip]):
                        data.append(
                            (
                                stateOfChargeInter.loc[:, count]
                                + stateOfChargeIntra.loc[p]
                                .loc[:, : esM.timeStepsPerPeriod[-1]]
                                .T
                            ).T
                        )
                    optVal = pd.concat(data, axis=1, ignore_index=True)
                else:
                    optVal = None
                self._stateOfChargeOperationVariablesOptimum[
                    esM.investmentPeriodNames[ip]
                ] = optVal
                utils.setOptimalComponentVariables(
                    optVal, "_stateOfChargeVariablesOptimum", compDict
                )

            # Append optimization summaries
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
                    | (optSummary.index.get_level_values(1) == "opexCharge")
                    | (optSummary.index.get_level_values(1) == "opexDischarge")
                ]
                .groupby(level=0)
                .sum()
                .values
            )
            optSummary.loc[
                optSummary.index.get_level_values(1) == "NPVcontribution"
            ] = (
                optSummary.loc[
                    (optSummary.index.get_level_values(1) == "NPVcontribution")
                    | (optSummary.index.get_level_values(1) == "NPV_opexCharge")
                    | (optSummary.index.get_level_values(1) == "NPV_opexDischarge")
                ]
                .groupby(level=0)
                .sum()
                .values
            )

            # # Delete details of NPV contributions
            optSummary = optSummary.drop("NPV_opexCharge", level=1)
            optSummary = optSummary.drop("NPV_opexDischarge", level=1)

            self._optSummary[esM.investmentPeriodNames[ip]] = optSummary

    def getOptimalValues(self, name="all", ip=0):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariables',
            * 'isBuiltVariables',
            * 'chargeOperationVariablesOptimum',
            * 'dischargeOperationVariablesOptimum',
            * 'stateOfChargeOperationVariablesOptimum',
            * 'all' or another input: all variables are returned.

            For optimizations with several years also following values should be returned:
            * '_operationVariablesOptimum'
            * '_decommissioningVariablesOptimum'

        |br| * the default value is 'all'
        :type name: string

        :param ip: investment period
            |br| * the default value is 0
        :type ip: int

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        if name == "capacityVariablesOptimum":
            return {
                "values": self._capacityVariablesOptimum[ip],
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "commissioningVariablesOptimum":
            return {
                "values": self._commissioningVariablesOptimum[ip],
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "decommissioningVariablesOptimum":
            return {
                "values": self._decommissioningVariablesOptimum[ip],
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "isBuiltVariablesOptimum":
            return {
                "values": self._isBuiltVariablesOptimum[ip],
                "timeDependent": False,
                "dimension": self.dimension,
            }
        elif name == "chargeOperationVariablesOptimum":
            return {
                "values": self._chargeOperationVariablesOptimum[ip],
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "dischargeOperationVariablesOptimum":
            return {
                "values": self._dischargeOperationVariablesOptimum[ip],
                "timeDependent": True,
                "dimension": self.dimension,
            }
        elif name == "stateOfChargeOperationVariablesOptimum":
            return {
                "values": self._stateOfChargeOperationVariablesOptimum[ip],
                "timeDependent": True,
                "dimension": self.dimension,
            }
        else:
            return {
                "commissioningVariablesOptimum": {
                    "values": self._commissioningVariablesOptimum[ip],
                    "timeDependent": False,
                    "dimension": self.dimension,
                },
                "decommissioningVariablesOptimum": {
                    "values": self._decommissioningVariablesOptimum[ip],
                    "timeDependent": False,
                    "dimension": self.dimension,
                },
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
                "chargeOperationVariablesOptimum": {
                    "values": self._chargeOperationVariablesOptimum[ip],
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "dischargeOperationVariablesOptimum": {
                    "values": self._dischargeOperationVariablesOptimum[ip],
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
                "stateOfChargeOperationVariablesOptimum": {
                    "values": self._stateOfChargeOperationVariablesOptimum[ip],
                    "timeDependent": True,
                    "dimension": self.dimension,
                },
            }
