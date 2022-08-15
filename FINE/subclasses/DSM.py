"""
Basic idea inspired by:
Zerrahn, Alexander, and Wolf-Peter Schill. "On the representation of demand-side management in
power system models." Energy 84 (2015): 840-845.
Approach adapted to be compatible with time series aggregation (idea: use virtual storages).
"""

from FINE.sourceSink import Sink, Source, SourceSinkModel
from FINE import utils
import FINE as fn
import pyomo.environ as pyomo
import pandas as pd
import warnings


class DemandSideManagementBETA(Sink):
    """
    A DemandSideManagement component. Allows to shift demands (of Sink components) forwards and backwards in time.
    Initializes a Sink component and multiple Storage components which are aggregated after optimization.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
        hasCapacityVariable,
        tFwd,
        tBwd,
        operationRateFix,
        opexShift=1e-6,
        shiftUpMax=None,
        shiftDownMax=None,
        socOffsetDown=-1,
        socOffsetUp=-1,
        **kwargs
    ):
        """
        Constructor for creating an DemandSideManagement class instance.
        Note: the DemandSideManagement class inherits from the Sink class; kwargs provide input arguments
        to the Sink component.

        **Required arguments:**

        :param esM: energy system model to which the DemandSideManagement component should be added.
            Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param name: name of the component. Has to be unique (i.e. no other components with that name can
            already exist in the EnergySystemModel instance to which the component is added).
        :type name: string

        :param hasCapacityVariable: specifies if the underlying Sink component should be modeled with a
            capacity or not. Examples:

            * An electrolyzer has a capacity given in GW_electric -> hasCapacityVariable is True.
            * In the energy system, biogas can, from a model perspective, be converted into methane (and then
              used in conventional power plants which emit CO2) by getting CO2 from the environment. Thus,
              using biogas in conventional power plants is, from a balance perspective, CO2 free. This
              conversion is purely theoretical and does not require a capacity -> hasCapacityVariable
              is False.
            * A electricity cable has a capacity given in GW_electric -> hasCapacityVariable is True.
            * If the transmission capacity of a component is unlimited -> hasCapacityVariable is False.
            * A wind turbine has a capacity given in GW_electric -> hasCapacityVariable is True.
            * Emitting CO2 into the environment is not per se limited by a capacity ->
              hasCapacityVariable is False.

        :type hasCapacityVariable: boolean

        :param tFwd: the number of timesteps for backwards demand shifting.
        :type tFwd: integer (>0)

        :param tBwd: the number of timesteps for forwards demand shifting.
        :type tBwd: integer (>= 0)

        :param operationRateFix: specifies the original time series of the shiftable demand.
            If hasCapacityVariable is set to True, the values are given relative
            to the installed capacities (i.e. a value of 1 indicates a utilization of 100% of the
            capacity). If hasCapacityVariable is set to False, the values are given as absolute values in form
            of the commodityUnit for each time step.
        :type operationRateMax: None or Pandas DataFrame with positive (>= 0) entries or dict of None or Pandas
            DataFrame with positive (>= 0) entries per investment period. The row indices have
            to match the in the energy system model specified time steps. The column indices have to equal the
            in the energy system model specified locations. The data in ineligible locations are set to zero.

        **Default arguments:**

        :param opexShift: operational cost for shifting the demand (given in costUnit/commodityUnit). Setting
            this value also penalizes unreasonable, unnecessary shifting of demand.
            |br| * the default value is 1e-6
        :type opexShift: positive float (>0)

        :param shiftUpMax: maximum amount of upwards shiftable commodity at one timestep. If None, the value
            is set equal to the maximum demand of the respective location.
            |br| * the default value is None
        :type shiftUpMax: positive float or None

        :param shiftDownMax: maximum amount of downwards shiftable commodity at one timestep. If None, the
            value is set equal to the maximum demand of the respective location.
            |br| * the default value is Nonde
        :type shiftDownMax: positive float or None

        :param socOffsetDown: determines whether the state of charge at the end of a period p has
            to be equal to the one at the beginning of a period p+1 (socOffsetDown=-1) or if
            it can be smaller at the beginning of p+1 (socOffsetDown>=0). In the latter case,
            the product of the parameter socOffsetDown and the actual soc offset is used as a penalty
            factor in the objective function. (usefull when infeasibilities are encountered when using
            DemandSideManagement and time series aggregation)
            |br| * the default value is -1
        :type socOffsetDown: float

        :param socOffsetUp: determines whether the state of charge at the end of a period p has
            to be equal to the one at the beginning of a period p+1 (socOffsetUp=-1) or if
            it can be larger at the beginning of p+1 (socOffsetUp>=0). In the latter case,
            the product of the parameter socOffsetUp and the actual soc offset is used as a penalty
            factor in the objective function. (usefull when infeasibilities are encountered when using
            DemandSideManagement and time series aggregation)
            |br| * the default value is -1
        :type socOffsetUp: float
        """
        if esM.verbose < 2:
            warnings.warn(
                "The DemandSideManagement component is currently in its BETA testing phase. "
                + "Infeasiblities can occur (in this case consider using socOffsetUp/ socOffsetDown). "
                + "Best results can be obtained when tFwd+tBwd+1 is a divisor of either the total number "
                + "of timesteps or the number of time steps per period. Use with care..."
            )

        self.tBwd = tBwd
        self.tFwd = tFwd
        self.tDelta = tFwd + tBwd + 1

        #############
        _operationRateFix = {}
        self.shiftUpMax = {}
        self.shiftDownMax = {}

        for ip in esM.investmentPeriods:

            if isinstance(operationRateFix, pd.DataFrame) or isinstance(
                operationRateFix, pd.Series
            ):  # operationRateFix is dataframe or series
                _operationRateFix[ip] = pd.concat(
                    [operationRateFix.iloc[-tBwd:], operationRateFix.iloc[:-tBwd]]
                ).reset_index(drop=True)
            elif isinstance(operationRateFix, dict):  # operationRateFix is dict
                _operationRateFix[ip] = pd.concat(
                    [
                        operationRateFix[ip].iloc[-tBwd:],
                        operationRateFix[ip].iloc[:-tBwd],
                    ]
                ).reset_index(drop=True)
            else:
                raise TypeError(
                    "operationRateFix should be a pandas dataframe or a dictionary."
                )

            if shiftUpMax is None:
                self.shiftUpMax[ip] = _operationRateFix[ip].max()
                print("shiftUpMax was set to", _operationRateFix[ip].max())
            else:
                if isinstance(shiftUpMax, dict):
                    self.shiftUpMax[ip] = shiftUpMax[ip]
                else:
                    self.shiftUpMax[ip] = shiftUpMax

            if shiftDownMax is None:
                self.shiftDownMax[ip] = _operationRateFix[ip].max()
                print("shiftDownMax was set to", _operationRateFix[ip].max())
            else:
                if isinstance(shiftDownMax, dict):
                    self.shiftDownMax[ip] = shiftDownMax[ip]
                else:
                    self.shiftDownMax[ip] = shiftDownMax

        Sink.__init__(
            self,
            esM,
            name,
            commodity,
            hasCapacityVariable,
            operationRateFix=_operationRateFix,
            **kwargs
        )

        self.modelingClass = DSMModel

        for i in range(self.tDelta):
            SOCmax = {}
            dischargeFix = {}
            chargeOpRateMax = {}
            opexPerChargeOpTimeSeries = {}

            for ip in esM.investmentPeriods:
                SOCmax[ip] = _operationRateFix[ip].copy()
                SOCmax[ip][SOCmax[ip] > 0] = 0

                SOCmax_ = (
                    pd.concat(
                        [
                            _operationRateFix[ip][
                                _operationRateFix[ip].index % self.tDelta == i
                            ]
                        ]
                        * self.tDelta
                    )
                    .sort_index()
                    .reset_index(drop=True)
                )

                if len(SOCmax_) > len(esM.totalTimeSteps):
                    SOCmax_ = pd.concat(
                        [
                            SOCmax_.iloc[tFwd + tBwd - i :],
                            SOCmax_.iloc[: tFwd + tBwd - i],
                        ]
                    ).reset_index(drop=True)
                    print(
                        "tBwd+tFwd+1 is not a divisor of the total number of time steps of the energy system. "
                        + "This shortens the shiftable timeframe of demand_"
                        + str(i)
                        + " by "
                        + str(len(SOCmax_) - len(esM.totalTimeSteps))
                        + " time steps"
                    )
                    SOCmax[ip] = SOCmax_.iloc[: len(esM.totalTimeSteps)]

                elif len(SOCmax_) < len(esM.totalTimeSteps):
                    SOCmax[ip].iloc[
                        0 : len(SOCmax_.iloc[tFwd + tBwd - i :])
                    ] = SOCmax_.iloc[tFwd + tBwd - i :].values
                    if len(SOCmax_.iloc[: tFwd + tBwd - i]) > 0:
                        SOCmax[ip].iloc[
                            -len(SOCmax_.iloc[: tFwd + tBwd - i]) :
                        ] = SOCmax_.iloc[: tFwd + tBwd - i].values

                else:
                    SOCmax_ = pd.concat(
                        [
                            SOCmax_.iloc[tFwd + tBwd - i :],
                            SOCmax_.iloc[: tFwd + tBwd - i],
                        ]
                    ).reset_index(drop=True)
                    SOCmax[ip] = SOCmax_

                chargeOpRateMax[ip] = SOCmax[ip].copy()

                if i < self.tDelta - 1:
                    SOCmax[ip][SOCmax[ip].index % self.tDelta == i + 1] = 0
                else:
                    SOCmax[ip][SOCmax[ip].index % self.tDelta == 0] = 0

                dischargeFix[ip] = _operationRateFix[ip].copy()
                dischargeFix[ip][dischargeFix[ip].index % self.tDelta != i] = 0

                opexPerChargeOpTimeSeries[ip] = pd.DataFrame(
                    [
                        [opexShift for loc in self.locationalEligibility]
                        for t in esM.totalTimeSteps
                    ],
                    columns=self.locationalEligibility.index,
                )
                opexPerChargeOpTimeSeries[ip][
                    (opexPerChargeOpTimeSeries[ip].index - i) % self.tDelta == tBwd + 1
                ] = 0

            opexPerChargeOpTimeSeries = pd.DataFrame(
                [
                    [opexShift for loc in self.locationalEligibility]
                    for t in esM.totalTimeSteps
                ],
                columns=self.locationalEligibility.index,
            )
            opexPerChargeOpTimeSeries[
                (opexPerChargeOpTimeSeries.index - i) % self.tDelta == tBwd + 1
            ] = 0

            esM.add(
                fn.StorageExtBETA(
                    esM,
                    name + "_" + str(i),
                    commodity,
                    stateOfChargeOpRateMax=SOCmax,
                    dischargeOpRateFix=dischargeFix,
                    hasCapacityVariable=False,
                    chargeOpRateMax=chargeOpRateMax,
                    opexPerChargeOpTimeSeries=opexPerChargeOpTimeSeries,
                    doPreciseTsaModeling=True,
                    socOffsetDown=socOffsetDown,
                    socOffsetUp=socOffsetUp,
                )
            )


class DSMModel(SourceSinkModel):
    """
    A StorageExtModel class instance will be instantly created if a StorageExt class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the StorageExt class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The StorageExtModel class inherits from the StorageModel class.
    """

    def __init__(self):
        """Constructor for creating a DSMModel class instance"""
        self.abbrvName = "dsm"
        self.dimension = "1dim"
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.optSummary = {}
        self.operationVariablesOptimum = {}

    def limitUpDownShifts(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        [commodityUnit*h] and the relative maximum state of charge [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        chargeOp = getattr(pyM, "chargeOp_storExt")
        constrSet = getattr(pyM, "operationVarSet_" + self.abbrvName)

        def limitUpDownShifts(pyM, loc, compName, ip, p, t):
            for i in range(compDict[compName].tDelta):
                if (
                    esM.getComponent(compName + "_" + str(i))
                    .processedOpexPerChargeOpTimeSeries[ip]
                    .loc[(p, t), loc]
                    == 0
                ):
                    ixDown = str(i)
                    break

            ixUp = [
                str(i) for i in range(compDict[compName].tDelta) if str(i) != ixDown
            ]

            return sum(
                chargeOp[loc, compName + "_" + compName_i, ip, p, t]
                for compName_i in ixUp
            ) + (
                esM.getComponent(compName + "_" + ixDown)
                .processedChargeOpRateMax[ip]
                .loc[(p, t), loc]
                - chargeOp[loc, compName + "_" + ixDown, ip, p, t]
            ) <= max(
                compDict[compName].shiftUpMax[ip], compDict[compName].shiftDownMax[ip]
            )

        setattr(
            pyM,
            "limitUpDownShifts_" + abbrvName,
            pyomo.Constraint(constrSet, pyM.timeSet, rule=limitUpDownShifts),
        )

    def shiftUpMax(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        [commodityUnit*h] and the relative maximum state of charge [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        chargeOp = getattr(pyM, "chargeOp_storExt")
        constrSet = getattr(pyM, "operationVarSet_" + self.abbrvName)

        def shiftUpMax(pyM, loc, compName, ip, p, t):
            for i in range(compDict[compName].tDelta):
                if (
                    esM.getComponent(compName + "_" + str(i))
                    .processedOpexPerChargeOpTimeSeries[ip]
                    .loc[(p, t), loc]
                    == 0
                ):
                    ixDown = str(i)
                    break
            ixUp = [
                str(i) for i in range(compDict[compName].tDelta) if str(i) != ixDown
            ]

            return (
                sum(
                    chargeOp[loc, compName + "_" + compName_i, ip, p, t]
                    for compName_i in ixUp
                )
                <= compDict[compName].shiftUpMax[ip]
            )

        setattr(
            pyM,
            "shiftUpMax_" + abbrvName,
            pyomo.Constraint(constrSet, pyM.timeSet, rule=shiftUpMax),
        )

    def shiftDownMax(self, pyM, esM):
        """
        Declare the constraint that the state of charge [commodityUnit*h] is limited by the installed capacity
        [commodityUnit*h] and the relative maximum state of charge [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        chargeOp = getattr(pyM, "chargeOp_storExt")
        constrSet = getattr(pyM, "operationVarSet_" + self.abbrvName)

        def shiftDownMax(pyM, loc, compName, ip, p, t):
            for i in range(compDict[compName].tDelta):
                if (
                    esM.getComponent(compName + "_" + str(i))
                    .processedOpexPerChargeOpTimeSeries[ip]
                    .loc[(p, t), loc]
                    == 0
                ):
                    ixDown = str(i)
                    break

            return (
                esM.getComponent(compName + "_" + ixDown)
                .processedChargeOpRateMax[ip]
                .loc[(p, t), loc]
                - chargeOp[loc, compName + "_" + ixDown, ip, p, t]
                <= compDict[compName].shiftDownMax[ip]
            )

        setattr(
            pyM,
            "shiftDownMax_" + abbrvName,
            pyomo.Constraint(constrSet, pyM.timeSet, rule=shiftDownMax),
        )

    def declareComponentConstraints(self, esM, pyM):
        """
        Declare time independent and dependent constraints.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        super().declareComponentConstraints(esM, pyM)

        self.limitUpDownShifts(pyM, esM)
        self.shiftUpMax(pyM, esM)
        self.shiftDownMax(pyM, esM)

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM, ip):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pym: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pym: pyomo ConcreteModel

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super(SourceSinkModel, self).setOptimalValues(
            esM, pyM, esM.locations, "commodityUnit"
        )

        # Set optimal operation variables and append optimization summary
        chargeOp = getattr(pyM, "chargeOp_storExt")
        optVal = utils.formatOptimizationOutput(
            chargeOp.get_values(),
            "operationVariables",
            "1dim",
            ip,
            esM.periodsOrder[ip],
            esM=esM,
        )

        def groupStor(x):
            ix = optVal.loc[x].name
            for compName, comp in self.componentsDict.items():
                if ix[0] in [
                    compName + "_" + str(i) for i in range(comp.tBwd + comp.tFwd + 1)
                ]:
                    return (compName, ix[1])

        optVal = optVal.groupby(lambda x: groupStor(x)).sum()
        optVal.index = pd.MultiIndex.from_tuples(optVal.index)

        if type(self.operationVariablesOptimum) is not dict:
            self.operationVariablesOptimum = {}
        self.operationVariablesOptimum[ip] = optVal

        props = ["operation", "opexOp", "commodCosts", "commodRevenues"]
        units = [
            "[-]",
            "[" + esM.costUnit + "/a]",
            "[" + esM.costUnit + "/a]",
            "[" + esM.costUnit + "/a]",
        ]
        tuples = [
            (compName, prop, unit)
            for compName in compDict.keys()
            for prop, unit in zip(props, units)
        ]
        tuples = list(
            map(
                lambda x: (x[0], x[1], "[" + compDict[x[0]].commodityUnit + "*h/a]")
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
                lambda op: op
                * compDict[op.name].processedOpexPerOperation[ip][op.index],
                axis=1,
            )
            cCost = opSum.apply(
                lambda op: op * compDict[op.name].processedCommodityCost[ip][op.index],
                axis=1,
            )
            cRevenue = opSum.apply(
                lambda op: op
                * compDict[op.name].processedCommodityRevenue[ip][op.index],
                axis=1,
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
                [(ix, "opexOp", "[" + esM.costUnit + "/a]") for ix in ox.index],
                ox.columns,
            ] = (
                ox.values / esM.numberOfYears
            )

            # get empty datframe for resulting time dependent (TD) cost sum
            cRevenueTD = pd.DataFrame(
                0.0, index=list(compDict.keys()), columns=opSum.columns
            )
            cCostTD = pd.DataFrame(
                0.0, index=list(compDict.keys()), columns=opSum.columns
            )

            for compName in opSum.index:
                if not compDict[compName].commodityCostTimeSeries is None:
                    # in case of time series aggregation rearange clustered cost time series
                    calcCostTD = utils.buildFullTimeSeries(
                        compDict[compName]
                        .commodityCostTimeSeries[ip]
                        .unstack(level=1)
                        .stack(level=0),
                        esM.periodsOrder[ip],
                        ip,
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
                        .commodityRevenueTimeSeries[ip]
                        .unstack(level=1)
                        .stack(level=0),
                        esM.periodsOrder[ip],
                        ip,
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

        # Quick fix if several runs with one investment period
        if type(self.optSummary) is not dict:
            self.optSummary = {}
        self.optSummary[ip] = optSummary
