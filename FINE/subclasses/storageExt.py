from FINE.component import Component, ComponentModel
from FINE.storage import Storage, StorageModel
from FINE import utils
import pyomo.environ as pyomo
import warnings
import pandas as pd


class StorageExtBETA(Storage):
    """
    A StorageExt component shows the behavior of a Storage component but it is additionally possible to set a
    state of charge time series. The StorageExt class inherits from the Storage class.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
        stateOfChargeOpRateMax=None,
        stateOfChargeOpRateFix=None,
        opexPerChargeOpTimeSeries=None,
        stateOfChargeTsaWeight=1,
        opexChargeOpTsaWeight=1,
        **kwargs
    ):
        """
        Constructor for creating an StorageExt class instance.
        The StorageExt component specific input arguments are described below. The Storage component specific
        input arguments are described in the Storage class and the general component input arguments are described in
        the Component class.

        **Default arguments:**

        :param stateOfChargeOpRateMax: if specified, indicates a maximum state of charge for each location and
            each time step by a positive float. If hasCapacityVariable is set to True, the values are given
            relative to the installed capacities (i.e. a value of 1 indicates a utilization of
            100% of the capacity). If hasCapacityVariable is set to False, the values are given as absolute
            values in form of the commodityUnit, referring to the commodity stored in the component at the
            beginning of one time step.
            |br| * the default value is None
        :type stateOfChargeOpRateMax: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model  specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param stateOfChargeOpRateFix: if specified, indicates a fixed state of charge for each location and
            each time step by a positive float. If hasCapacityVariable is set to True, the values are given
            relative to the installed capacities (i.e. a value of 1 indicates a utilization of
            100% of the capacity). If hasCapacityVariable is set to False, the values are given as absolute
            values in form of the commodityUnit, referring to the commodity stored in the component at the
            beginning of one time step.
            |br| * the default value is None
        :type stateOfChargeOpRateFix: None or Pandas DataFrame with positive (>= 0) entries. The row indices have
            to match the in the energy system model specified time steps. The column indices have to match the
            in the energy system model specified locations.

        :param stateOfChargeTsaWeight: weight with which the stateOfChargeOpRate (max/fix) time series of the
            component should be considered when applying time series aggregation.
            |br| * the default value is 1
        :type stateOfChargeTsaWeight: positive (>= 0) float
        """
        Storage.__init__(self, esM, name, commodity, **kwargs)

        self.modelingClass = StorageExtModel

        # Set location-specific operation parameters (charging rate, discharging rate, state of charge rate)
        # and time series aggregation weighting factor
        self.stateOfChargeOpRateFix = stateOfChargeOpRateFix
        self.stateOfChargeOpRateMax = stateOfChargeOpRateMax

        # The i-th state of charge (SOC) refers to the SOC before the i-th time step
        if stateOfChargeOpRateMax is not None and stateOfChargeOpRateFix is not None:
            stateOfChargeOpRateMax = None
            if esM.verbose < 2:
                warnings.warn(
                    "If stateOfChargeOpRateFix is specified, the stateOfChargeOpRateMax parameter is not +"
                    "required.\nThe stateOfChargeOpRateMax time series was set to None."
                )
        if (
            stateOfChargeOpRateMax is not None or stateOfChargeOpRateFix is not None
        ) and not self.doPreciseTsaModeling:
            self.doPreciseTsaModeling = True
            if esM.verbose < 2:
                warnings.warn(
                    "Warning only relevant when time series aggregation is used in optimization:\n"
                    + "If stateOfChargeOpRateFix or the stateOfChargeOpRateMax parameter are specified,\n"
                    + "the modeling is set to precise."
                )
        if stateOfChargeOpRateMax is not None:
            if esM.verbose < 2:
                warnings.warn(
                    "Warning only relevant when time series aggregation is used in optimization:\n"
                    + "Setting the stateOfChargeOpRateMax parameter might lead to unwanted modeling behavior\n"
                    + "and should be handled with caution."
                )
        if stateOfChargeOpRateFix is not None and not self.isPeriodicalStorage:
            self.isPeriodicalStorage = True
            if esM.verbose < 2:
                warnings.warn(
                    "Warning only relevant when time series aggregation is used in optimization:\n"
                    + "If the stateOfChargeOpRateFix parameter is specified, the storage\n"
                    + "is set to isPeriodicalStorage)."
                )

        self.fullStateOfChargeOpRateMax = utils.checkAndSetTimeSeries(
            esM, name, stateOfChargeOpRateMax, self.locationalEligibility
        )
        self.aggregatedStateOfChargeOpRateMax, self.processedStateOfChargeOpRateMax = (
            None,
            None,
        )

        self.fullStateOfChargeOpRateFix = utils.checkAndSetTimeSeries(
            esM, name, stateOfChargeOpRateFix, self.locationalEligibility
        )
        self.aggregatedStateOfChargeOpRateFix, self.processedStateOfChargeOpRateFix = (
            None,
            None,
        )

        self.opexPerChargeOpTimeSeries = opexPerChargeOpTimeSeries
        self.fullOpexPerChargeOpTimeSeries = utils.checkAndSetTimeSeries(
            esM, name, opexPerChargeOpTimeSeries, self.locationalEligibility
        )
        (
            self.aggregatedOpexPerChargeOpTimeSeries,
            self.processedOpexPerChargeOpTimeSeries,
        ) = (None, None)

        utils.isPositiveNumber(stateOfChargeTsaWeight), utils.isPositiveNumber(
            opexChargeOpTsaWeight
        )
        self.stateOfChargeTsaWeight, self.opexChargeOpTsaWeight = (
            stateOfChargeTsaWeight,
            opexChargeOpTsaWeight,
        )

        # Set locational eligibility
        timeSeriesData = None
        tsNb = sum(
            [
                0 if data is None else 1
                for data in [
                    self.chargeOpRateMax,
                    self.chargeOpRateFix,
                    self.dischargeOpRateMax,
                    self.dischargeOpRateFix,
                    stateOfChargeOpRateMax,
                    stateOfChargeOpRateFix,
                ]
            ]
        )
        if tsNb > 0:
            timeSeriesData = sum(
                [
                    data
                    for data in [
                        self.chargeOpRateMax,
                        self.chargeOpRateFix,
                        self.dischargeOpRateMax,
                        self.dischargeOpRateFix,
                        stateOfChargeOpRateMax,
                        stateOfChargeOpRateFix,
                    ]
                    if data is not None
                ]
            )
        self.locationalEligibility = utils.setLocationalEligibility(
            esM,
            self.locationalEligibility,
            self.capacityMax,
            self.capacityFix,
            self.isBuiltFix,
            self.hasCapacityVariable,
            timeSeriesData,
        )

    def addToEnergySystemModel(self, esM):
        """
        Function for adding a StorageExt component to the given energy system model.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel class instance
        """
        super().addToEnergySystemModel(esM)

    def setTimeSeriesData(self, hasTSA):
        """
        Function for setting the maximum operation rate and fixed operation rate for the state of charge, charging and
        discharging depending on whether a time series analysis is requested or not.

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
        self.processedStateOfChargeOpRateMax = (
            self.aggregatedStateOfChargeOpRateMax
            if hasTSA
            else self.fullStateOfChargeOpRateMax
        )
        self.processedDtateOfChargeOpRateFix = (
            self.aggregatedStateOfChargeOpRateFix
            if hasTSA
            else self.fullStateOfChargeOpRateFix
        )
        self.processedOpexPerChargeOpTimeSeries = (
            self.aggregatedOpexPerChargeOpTimeSeries
            if hasTSA
            else self.fullOpexPerChargeOpTimeSeries
        )

    def getDataForTimeSeriesAggregation(self):
        """Function for getting the required data if a time series aggregation is requested."""
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
            (
                self.fullStateOfChargeOpRateFix,
                self.fullStateOfChargeOpRateMax,
                "_SOCRate_",
                self.stateOfChargeTsaWeight,
            ),
            (
                self.fullOpexPerChargeOpTimeSeries,
                None,
                "_opexPerChargeOp_",
                self.opexChargeOpTsaWeight,
            ),
        ]

        for rateFix, rateMax, rateName, rateWeight in I:
            weightDict, data = self.prepareTSAInput(
                rateFix, rateMax, rateName, rateWeight, weightDict, data
            )

        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data):
        """
        Function for determining the aggregated maximum and fixed state of charge/ discharge rate/ charge rate.

        :param data: Pandas DataFrame with the clustered time series data of the source component
        :type data: Pandas DataFrame
        """
        self.aggregatedChargeOpRateFix = self.getTSAOutput(
            self.fullChargeOpRateFix, "chargeRate_", data
        )
        self.aggregatedChargeOpRateMax = self.getTSAOutput(
            self.fullChargeOpRateMax, "chargeRate_", data
        )

        self.aggregatedDischargeOpRateFix = self.getTSAOutput(
            self.fullDischargeOpRateFix, "dischargeRate_", data
        )
        self.aggregatedDischargeOpRateMax = self.getTSAOutput(
            self.fullDischargeOpRateMax, "dischargeRate_", data
        )

        self.aggregatedStateOfChargeOpRateFix = self.getTSAOutput(
            self.fullStateOfChargeOpRateFix, "_SOCRate_", data
        )
        self.aggregatedStateOfChargeOpRateMax = self.getTSAOutput(
            self.fullStateOfChargeOpRateMax, "_SOCRate_", data
        )

        self.aggregatedOpexPerChargeOpTimeSeries = self.getTSAOutput(
            self.fullOpexPerChargeOpTimeSeries, "_opexPerChargeOp_", data
        )


class StorageExtModel(StorageModel):
    """
    A StorageExtModel class instance will be instantly created if a StorageExt class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the StorageExt class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The StorageExtModel class inherits from the StorageModel class.
    """

    def __init__(self):
        """Constructor for creating a StorageExtModel class instance"""
        self.abbrvName = "storExt"
        self.dimension = "1dim"
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        (
            self.chargeOperationVariablesOptimum,
            self.dischargeOperationVariablesOptimum,
        ) = (None, None)
        self.stateOfChargeOperationVariablesOptimum = None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """
        Declare sets and dictionaries.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        super().declareSets(esM, pyM)

        # * State of charge operation TODO check if also applied for simple SOC modeling
        self.declareOperationModeSets(
            pyM,
            "stateOfChargeOpConstrSet",
            "stateOfChargeOpRateMax",
            "stateOfChargeOpRateFix",
        )

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def operationModeSOCwithTSA1(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        [commodityUnit*h] and the relative maximum state of charge [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
        SOC, capVar = getattr(pyM, "stateOfCharge_" + abbrvName), getattr(
            pyM, "cap_" + abbrvName
        )
        constrSet1 = getattr(pyM, "stateOfChargeOpConstrSet1_" + abbrvName)

        def SOCMaxPrecise1(pyM, loc, compName, pInter, t):
            if compDict[compName].doPreciseTsaModeling:
                if not pyM.hasSegmentation:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (t * esM.hoursPerTimeStep)
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        <= capVar[loc, compName] * compDict[compName].stateOfChargeMax
                    )
                else:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (
                                esM.segmentStartTime.to_dict()[
                                    esM.periodsOrder[pInter], t
                                ]
                                * esM.hoursPerTimeStep
                            )
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        <= capVar[loc, compName] * compDict[compName].stateOfChargeMax
                    )
            else:
                return pyomo.Constraint.Skip

        setattr(
            pyM,
            "ConstrSOCMaxPrecise1_" + abbrvName,
            pyomo.Constraint(
                constrSet1, esM.periods, esM.timeStepsPerPeriod, rule=SOCMaxPrecise1
            ),
        )

    def operationModeSOCwithTSA2(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is equal to the installed capacity
        [commodityUnit*h] multiplied by state of charge time series [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
        SOC, capVar = getattr(pyM, "stateOfCharge_" + abbrvName), getattr(
            pyM, "cap_" + abbrvName
        )
        constrSet2 = getattr(pyM, "stateOfChargeOpConstrSet2_" + abbrvName)

        def SOCMaxPrecise2(pyM, loc, compName, pInter, t):
            if compDict[compName].doPreciseTsaModeling:
                if not pyM.hasSegmentation:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (t * esM.hoursPerTimeStep)
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        == capVar[loc, compName]
                        * compDict[compName].processedStateOfChargeOpRateFix[loc][
                            esM.periodsOrder[pInter], t
                        ]
                    )
                else:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (
                                esM.segmentStartTime.to_dict()[
                                    esM.periodsOrder[pInter], t
                                ]
                                * esM.hoursPerTimeStep
                            )
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        == capVar[loc, compName]
                        * compDict[compName].processedStateOfChargeOpRateFix[loc][
                            esM.periodsOrder[pInter], t
                        ]
                    )
            else:
                return pyomo.Constraint.Skip

        setattr(
            pyM,
            "ConstrSOCMaxPrecise2_" + abbrvName,
            pyomo.Constraint(
                constrSet2, esM.periods, esM.timeStepsPerPeriod, rule=SOCMaxPrecise2
            ),
        )

    def operationModeSOCwithTSA3(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        [commodityUnit*h] multiplied by state of charge time series [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
        SOC, capVar = getattr(pyM, "stateOfCharge_" + abbrvName), getattr(
            pyM, "cap_" + abbrvName
        )
        constrSet3 = getattr(pyM, "stateOfChargeOpConstrSet3_" + abbrvName)

        def SOCMaxPrecise3(pyM, loc, compName, pInter, t):
            if compDict[compName].doPreciseTsaModeling:
                if not pyM.hasSegmentation:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (t * esM.hoursPerTimeStep)
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        <= capVar[loc, compName]
                        * compDict[compName].processedStateOfChargeOpRateMax[loc][
                            esM.periodsOrder[pInter], t
                        ]
                    )
                else:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (
                                esM.segmentStartTime.to_dict()[
                                    esM.periodsOrder[pInter], t
                                ]
                                * esM.hoursPerTimeStep
                            )
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        <= capVar[loc, compName]
                        * compDict[compName].processedStateOfChargeOpRateMax[loc][
                            esM.periodsOrder[pInter], t
                        ]
                    )
            else:
                return pyomo.Constraint.Skip

        setattr(
            pyM,
            "ConstrSOCMaxPrecise3_" + abbrvName,
            pyomo.Constraint(
                constrSet3, esM.periods, esM.timeStepsPerPeriod, rule=SOCMaxPrecise3
            ),
        )

    def operationModeSOCwithTSA4(self, pyM, esM):
        """
        Declare the constraint that the operation [commodityUnit*h] is equal to the operation time series
        [commodityUnit*h].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
        SOC = getattr(pyM, "stateOfCharge_" + abbrvName)
        constrSet4 = getattr(pyM, "stateOfChargeOpConstrSet4_" + abbrvName)

        def SOCMaxPrecise4(pyM, loc, compName, pInter, t):
            if compDict[compName].doPreciseTsaModeling:
                if not pyM.hasSegmentation:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (t * esM.hoursPerTimeStep)
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        == compDict[compName].processedStateOfChargeOpRateFix[loc][
                            esM.periodsOrder[pInter], t
                        ]
                    )
                else:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (
                                esM.segmentStartTime.to_dict()[
                                    esM.periodsOrder[pInter], t
                                ]
                                * esM.hoursPerTimeStep
                            )
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        == compDict[compName].processedStateOfChargeOpRateFix[loc][
                            esM.periodsOrder[pInter], t
                        ]
                    )
            else:
                return pyomo.Constraint.Skip

        setattr(
            pyM,
            "ConstrSOCMaxPrecise4_" + abbrvName,
            pyomo.Constraint(
                constrSet4, esM.periods, esM.timeStepsPerPeriod, rule=SOCMaxPrecise4
            ),
        )

    def operationModeSOCwithTSA5(self, pyM, esM):
        """
        Declare the constraint that the operation [commodityUnit*h] is limited by the operation time series
        [commodityUnit*h].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        SOCinter = getattr(pyM, "stateOfChargeInterPeriods_" + abbrvName)
        SOC = getattr(pyM, "stateOfCharge_" + abbrvName)
        constrSet5 = getattr(pyM, "stateOfChargeOpConstrSet5_" + abbrvName)

        def SOCMaxPrecise5(pyM, loc, compName, pInter, t):
            if compDict[compName].doPreciseTsaModeling:
                if not pyM.hasSegmentation:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (t * esM.hoursPerTimeStep)
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        <= compDict[compName].processedStateOfChargeOpRateMax[loc][
                            esM.periodsOrder[pInter], t
                        ]
                    )
                else:
                    return (
                        SOCinter[loc, compName, pInter]
                        * (
                            (1 - compDict[compName].selfDischarge)
                            ** (
                                esM.segmentStartTime.to_dict()[
                                    esM.periodsOrder[pInter], t
                                ]
                                * esM.hoursPerTimeStep
                            )
                        )
                        + SOC[loc, compName, esM.periodsOrder[pInter], t]
                        <= compDict[compName].processedStateOfChargeOpRateMax[loc][
                            esM.periodsOrder[pInter], t
                        ]
                    )
            else:
                return pyomo.Constraint.Skip

        setattr(
            pyM,
            "ConstrSOCMaxPrecise5_" + abbrvName,
            pyomo.Constraint(
                constrSet5, esM.periods, esM.timeStepsPerPeriod, rule=SOCMaxPrecise5
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
        self.capacityFix(pyM)
        # Sets, if applicable, the binary design variables of a component
        self.designBinFix(pyM)

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
        # Operation [commodityUnit*h] is equal to the operation time series [commodityUnit*h]
        self.operationMode4(
            pyM,
            esM,
            "ConstrCharge",
            "chargeOpConstrSet",
            "chargeOp",
            "processedChargeOpRateFix",
        )
        # Operation [commodityUnit*h] is limited by the operation time series [commodityUnit*h]
        self.operationMode5(
            pyM,
            esM,
            "ConstrCharge",
            "chargeOpConstrSet",
            "chargeOp",
            "processedChargeOpRateMax",
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
        # Operation [commodityUnit*h] is equal to the operation time series [commodityUnit*h]
        self.operationMode4(
            pyM,
            esM,
            "ConstrDischarge",
            "dischargeOpConstrSet",
            "dischargeOp",
            "processedDischargeOpRateFix",
        )
        # Operation [commodityUnit*h] is limited by the operation time series [commodityUnit*h]
        self.operationMode5(
            pyM,
            esM,
            "ConstrDischarge",
            "dischargeOpConstrSet",
            "dischargeOp",
            "processedDischargeOpRateMax",
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
            # The (virtual) state of charge at the beginning of a typical period is zero
            self.intraSOCstart(pyM, esM)
            # If periodic storage is selected, the states of charge between periods have the same value
            self.equalInterSOC(pyM, esM)

        # Ensure that the state of charge is within the operating limits of the installed capacities
        if not pyM.hasTSA:
            #              Constraints for enforcing a state of charge operation mode within given limits              #

            # State of charge [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] and the relative
            # maximum state of charge
            self.operationMode1(
                pyM,
                esM,
                "ConstrSOC",
                "stateOfChargeOpConstrSet",
                "stateOfCharge",
                "stateOfChargeMax",
                True,
            )
            # State of charge [commodityUnit*h] is equal to the installed capacity [commodityUnit*h] and the relative
            # fixed state of charge time series [-]
            self.operationMode2(
                pyM,
                esM,
                "ConstrSOC",
                "stateOfChargeOpConstrSet",
                "stateOfCharge",
                "processedStateOfChargeOpRateFix",
                True,
            )
            # State of charge [commodityUnit*h] is limited by the installed capacity [commodityUnit*h] and the relative
            # maximum state of charge time series [-]
            self.operationMode3(
                pyM,
                esM,
                "ConstrSOC",
                "stateOfChargeOpConstrSet",
                "stateOfCharge",
                "processedStateOfChargeOpRateMax",
                True,
            )
            # State of charge [commodityUnit*h] is equal to the absolute fixed state of charge time series [commodityUnit*h]
            self.operationMode4(
                pyM,
                esM,
                "ConstrSOC",
                "stateOfChargeOpConstrSet",
                "stateOfCharge",
                "processedStateOfChargeOpRateFix",
            )
            # State of charge [commodityUnit*h] is limited by the absolute maximum state of charge time series [commodityUnit*h]
            self.operationMode5(
                pyM,
                esM,
                "ConstrSOC",
                "stateOfChargeOpConstrSet",
                "stateOfCharge",
                "processedStateOfChargeOpRateMax",
            )

            # The state of charge [commodityUnit*h] has to be larger than the installed capacity [commodityUnit*h]
            # multiplied with the relative minimum state of charge
            self.minSOC(pyM)

        else:
            #                       Simplified version of the state of charge limitation control                       #
            #           (The error compared to the precise version is small in cases of small selfDischarge)           #
            self.limitSOCwithSimpleTsa(pyM, esM)

            #                        Precise version of the state of charge limitation control                         #

            # Constraints for enforcing a state of charge operation within given limits

            # State of charge [commodityUnit*h] limited by the installed capacity [commodityUnit*h] and the relative
            # maximum state of charge
            self.operationModeSOCwithTSA1(pyM, esM)
            # State of charge [commodityUnit*h] equal to the installed capacity [commodityUnit*h] multiplied by state
            # of charge time series [-]
            self.operationModeSOCwithTSA2(pyM, esM)
            # State of charge [commodityUnit*h] limited by the installed capacity [commodityUnit*h] multiplied by state
            # of charge time series [-]
            self.operationModeSOCwithTSA3(pyM, esM)
            # Operation [commodityUnit*h] equal to the operation time series [commodityUnit*h]
            self.operationModeSOCwithTSA4(pyM, esM)
            # Operation [commodityUnit*h] limited by the operation time series [commodityUnit*h]
            self.operationModeSOCwithTSA5(pyM, esM)

            # The state of charge at each time step cannot be smaller than the installed capacity multiplied with the
            # relative minimum state of charge
            self.minSOCwithTSAprecise(pyM, esM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        """
        Get contributions to shared location potential.
        """
        return super().getSharedPotentialContribution(pyM, key, loc)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        """
        Check if the storage of a commodity is eligible in a certain location.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: Name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """
        return super().hasOpVariablesForLocationCommodity(esM, loc, commod)

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """Get contribution to a commodity balance."""
        return super().getCommodityBalanceContribution(pyM, commod, loc, p, t)

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        basicContribution = super().getObjectiveFunctionContribution(esM, pyM)
        chargeOpContribution = self.getEconomicsTimeSeries(
            pyM,
            esM,
            "processedOpexPerChargeOpTimeSeries",
            "chargeOp",
            "operationVarDict",
        )

        return basicContribution + chargeOpContribution

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
        return super().setOptimalValues(esM, pyM)

    def getOptimalValues(self, name="all"):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariables',
            * 'isBuiltVariables',
            * 'chargeOperationVariablesOptimum',
            * 'dischargeOperationVariablesOptimum',
            * 'stateOfChargeOperationVariablesOptimum',
            * 'all' or another input: all variables are returned.

        |br| * the default value is 'all'
        :type name: string

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        return super().getOptimalValues(name)
