import matplotlib.pyplot as plt
import pandas as pd
import FINE as fn
import numpy as np
from pyomo.opt import SolverStatus, TerminationCondition
import warnings


def dsm_test_esM():
    """
    Generate a simple energy system model with one node, two fixed generators and one load time series
    for testing demand side management functionality.
    """
    # load without dsm
    now = pd.Timestamp.now().round("h")
    number_of_time_steps = 24
    # t_index = pd.date_range(now, now + pd.DateOffset(hours=number_of_timeSteps - 1), freq='h')
    t_index = range(number_of_time_steps)
    load_without_dsm = pd.Series([80.0] * number_of_time_steps, index=t_index)

    timestep_up = 10
    timestep_down = 20
    load_without_dsm[timestep_up:timestep_down] += 40.0

    time_shift = 3
    cheap_capacity = 100.0
    expensive_capacity = 20.0

    # set up energy model
    esM = fn.EnergySystemModel(
        locations={"location"},
        commodities={"electricity"},
        numberOfTimeSteps=number_of_time_steps,
        commodityUnitsDict={"electricity": r"MW$_{el}$"},
        hoursPerTimeStep=1,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="cheap",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(cheap_capacity, index=t_index),
            opexPerOperation=25,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="expensive",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(expensive_capacity, index=t_index),
            opexPerOperation=50,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="back-up",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(1000, index=t_index),
            opexPerOperation=1000,
        )
    )

    return esM, load_without_dsm, timestep_up, timestep_down, time_shift, cheap_capacity


def plotShift(
    tmin,
    tmax,
    title,
    timeSeriesAggregation,
    esM_with,
    tFwd,
    tBwd,
    shiftMax,
    numberOfTimeStepsPerPeriod,
):
    c = esM_with.componentModelingDict[
        "StorageExtModel"
    ].chargeOperationVariablesOptimum

    fig, axT = plt.subplots(1, 1, figsize=(6, 3))

    axT.plot(
        np.arange(0.5, tmax - tmin + 0.5),
        esM_with.componentModelingDict["DSMModel"]
        .operationVariablesOptimum.loc[("flexible demand", "location"), tmin : tmax - 1]
        .values,
        color="k",
        label="Demand w/- DSM",
        zorder=10,
        linestyle="-",
    )
    axT.set_ylabel("Flexible demand [kW]", fontsize=10)

    axT.set_title(title, fontsize=11)

    if timeSeriesAggregation:
        demand = pd.concat(
            [
                esM_with.getComponent("flexible demand").aggregatedOperationRateFix.loc[
                    p
                ]
                for p in esM_with.periodsOrder
            ],
            ignore_index=True,
        )
        demand = pd.concat([demand.iloc[tFwd:], demand.iloc[:tFwd]]).reset_index(
            drop=True
        )
        axT.plot(
            np.arange(0.5, tmax - tmin + 0.5),
            demand.loc[tmin : tmax - 1, "location"].values,
            color="k",
            linestyle="--",
            label="Demand w/o DSM",
        )
    else:
        demand = esM_with.getComponent("flexible demand").fullOperationRateFix.loc[0]
        demand = pd.concat([demand.iloc[tFwd:], demand.iloc[:tFwd]]).reset_index(
            drop=True
        )
        axT.plot(
            np.arange(0.5, tmax - tmin + 0.5),
            demand.loc[tmin : tmax - 1].values,
            color="k",
            linestyle="--",
            label="Demand w/o DSM",
        )
    axT.grid(False)

    axT.legend(loc=2, fontsize=8)
    axT.set_xticks(range(0, len(c.T), 4))
    axT.set_xlabel("Hour of the day [h]", fontsize=10)

    ax = axT.twinx()

    color_1 = ["blue", "grey", "red", "yellow", "green", "black", "pink"]
    marker = ["o", "x", "*"]

    ax.grid(zorder=-10)

    bottom_1 = np.array([0 for i in range(tmax - tmin)])
    bottom_2 = np.array([0 for i in range(tmax - tmin)])

    for x in range(tmax - tmin):
        h = tmin + x
        bottom = 0
        for i in range(tBwd + tFwd + 1):
            if (tFwd + h) % (tBwd + tFwd + 1) == i:
                if timeSeriesAggregation:
                    p = int(
                        (h - (h % numberOfTimeStepsPerPeriod))
                        / numberOfTimeStepsPerPeriod
                    )
                    tp = esM_with.periodsOrder[p]
                    h_ = h % numberOfTimeStepsPerPeriod
                    chargeMax = esM_with.getComponent(
                        "flexible demand_" + str(i)
                    ).aggregatedChargeOpRateMax.loc[(tp, h_), "location"]
                else:
                    chargeMax = esM_with.getComponent(
                        "flexible demand_" + str(i)
                    ).fullChargeOpRateMax.loc[(0, h), "location"]
                if x == 0:
                    ax.bar(
                        [x + 0.5],
                        -(chargeMax - c.iloc[i].values[h]),
                        color=color_1[i],
                        zorder=10,
                        label="Virtual storage_" + str(i),
                        alpha=0.5,
                    )
                else:
                    ax.bar(
                        [x + 0.5],
                        -(chargeMax - c.iloc[i].values[h]),
                        color=color_1[i],
                        zorder=10,
                        alpha=0.5,
                    )
            else:
                if x == 0:
                    ax.bar(
                        [x + 0.5],
                        c.iloc[i].values[h],
                        bottom=[bottom],
                        color=color_1[i],
                        zorder=10,
                        label="Virtual storage_" + str(i),
                        alpha=0.5,
                    )
                else:
                    ax.bar(
                        [x + 0.5],
                        c.iloc[i].values[h],
                        bottom=[bottom],
                        color=color_1[i],
                        zorder=10,
                        alpha=0.5,
                    )
                bottom += c.iloc[i].values[h]

    ax.legend(loc=3, fontsize=9)

    ax.set_ylim([-(shiftMax * 1.1), (shiftMax * 1.1)])
    ax.set_ylabel("Up-/ downshifted demand [kW]", fontsize=10)

    plt.tight_layout()
    plt.show()


def run_esM_without_DSM():
    esM_without, load_without_dsm, _, _, _, _ = dsm_test_esM()

    esM_without.add(
        fn.Sink(
            esM=esM_without,
            name="load",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=load_without_dsm,
        )
    )

    esM_without.optimize(timeSeriesAggregation=False)  # without dsm

    fig, ax = fn.plotOperation(
        esM_without, "cheap", "location", figsize=(4, 2), fontsize=10
    )
    ax.set_title("Cheap generator")
    fig, ax = fn.plotOperation(
        esM_without, "expensive", "location", figsize=(4, 2), fontsize=10
    )
    ax.set_title("Expensive generator")
    fig, ax = fn.plotOperation(
        esM_without, "load", "location", figsize=(4, 2), fontsize=10
    )
    ax.set_title("Load")
    plt.show()


def run_esM_with_DSM(
    timeSeriesAggregation,
    tBwd,
    tFwd,
    numberOfTypicalPeriods=25,
    numberOfTimeStepsPerPeriod=1,
):
    # add DSM
    dsm_test_esM_ = dsm_test_esM()
    esM_with = dsm_test_esM_[0]
    load_without_dsm = dsm_test_esM_[1]
    shiftMax = 10

    esM_with.add(
        fn.DemandSideManagementBETA(
            esM=esM_with,
            name="flexible demand",
            commodity="electricity",
            hasCapacityVariable=False,
            tFwd=tFwd,
            tBwd=tBwd,
            operationRateFix=load_without_dsm,
            opexShift=1,
            shiftDownMax=shiftMax,
            shiftUpMax=shiftMax,
            socOffsetDown=-1,
            socOffsetUp=-1,
        )
    )

    if timeSeriesAggregation:
        esM_with.cluster(
            numberOfTimeStepsPerPeriod=numberOfTimeStepsPerPeriod,
            numberOfTypicalPeriods=numberOfTypicalPeriods,
        )
        esM_with.optimize(timeSeriesAggregation=True)
        if esM_with.solverSpecs["status"] != "ok":
            print(
                esM_with.solverSpecs["status"],
                esM_with.solverSpecs["terminationCondition"],
            )
            print(
                "\n\nOptimization failed. Try againg with relaxed state of charge formulation (socOffsetDown=socOffsetDown=200)\n\n"
            )
            esM_with.add(
                fn.DemandSideManagementBETA(
                    esM=esM_with,
                    name="flexible demand",
                    commodity="electricity",
                    hasCapacityVariable=False,
                    tFwd=tFwd,
                    tBwd=tBwd,
                    operationRateFix=load_without_dsm,
                    opexShift=1,
                    shiftDownMax=shiftMax,
                    shiftUpMax=shiftMax,
                    socOffsetDown=200,
                    socOffsetUp=200,
                )
            )
            esM_with.optimize(timeSeriesAggregation=True)
    else:
        esM_with.optimize(
            timeSeriesAggregation=False,
        )
        if esM_with.solverSpecs["status"] != "ok":
            print(
                esM_with.solverSpecs["status"],
                esM_with.solverSpecs["terminationCondition"],
            )
            print(
                "\n\nOptimization failed. Try againg with relaxed state of charge formulation (socOffsetDown=socOffsetDown=200)\n\n"
            )
            esM_with.add(
                fn.DemandSideManagementBETA(
                    esM=esM_with,
                    name="flexible demand",
                    commodity="electricity",
                    hasCapacityVariable=False,
                    tFwd=tFwd,
                    tBwd=tBwd,
                    operationRateFix=load_without_dsm,
                    opexShift=1,
                    shiftDownMax=shiftMax,
                    shiftUpMax=shiftMax,
                    socOffsetDown=200,
                    socOffsetUp=200,
                )
            )
            esM_with.optimize(
                timeSeriesAggregation=False,
            )

    generator_outputs = esM_with.componentModelingDict[
        "SourceSinkModel"
    ].operationVariablesOptimum
    esM_load_with_DSM = esM_with.componentModelingDict[
        "DSMModel"
    ].operationVariablesOptimum

    fig, ax = fn.plotOperation(
        esM_with, "cheap", "location", figsize=(4, 2), fontsize=10
    )
    ax.set_title("Cheap generator")
    fig, ax = fn.plotOperation(
        esM_with, "expensive", "location", figsize=(4, 2), fontsize=10
    )
    ax.set_title("Expensive generator")
    fig, ax = fn.plotOperation(
        esM_with, "flexible demand", "location", figsize=(4, 2), fontsize=10
    )
    ax.set_title("Flexible demand")
    plt.show()

    c = esM_with.componentModelingDict[
        "StorageExtModel"
    ].chargeOperationVariablesOptimum
    chargeMax = pd.concat(
        [
            esM_with.getComponent("flexible demand_" + str(i)).fullChargeOpRateMax.loc[
                0
            ]
            for i in range(tBwd + tFwd + 1)
        ],
        axis=1,
        keys=["flexible demand_" + str(i) for i in range(tBwd + tFwd + 1)],
    ).T

    plotShift(
        0,
        len(c.T),
        "Shift profile",
        timeSeriesAggregation,
        esM_with,
        shiftMax=shiftMax,
        numberOfTimeStepsPerPeriod=numberOfTimeStepsPerPeriod,
        tBwd=tBwd,
        tFwd=tFwd,
    )
