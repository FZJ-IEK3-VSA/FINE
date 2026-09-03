"""Optimize and visualize independent maintenance windows.

The dynamic converter is cheaper than the backup source and would normally run
throughout the horizon. Two maintenance windows of at least three hours force it
offline, and the optimizer chooses when those windows occur.
"""

import matplotlib.pyplot as plt
import pandas as pd

import fine as fn


NUMBER_OF_TIME_STEPS = 24
DEMAND = 10


def build_model():
    """Return a one-location energy system with scheduled maintenance."""
    esM = fn.EnergySystemModel(
        locations={"site"},
        commodities={"fuel", "heat"},
        commodityUnitsDict={"fuel": "kW", "heat": "kW"},
        numberOfTimeSteps=NUMBER_OF_TIME_STEPS,
        hoursPerTimeStep=1,
        verboseLogLevel=0,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Fuel supply",
            commodity="fuel",
            hasCapacityVariable=False,
            commodityCost=0.05,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Backup heat",
            commodity="heat",
            hasCapacityVariable=False,
            commodityCost=1,
        )
    )
    esM.add(
        fn.ConversionDynamic(
            esM=esM,
            name="Boiler",
            physicalUnit="kW",
            commodityConversionFactors={"fuel": -1, "heat": 1},
            hasCapacityVariable=True,
            capacityFix=DEMAND,
            maintenanceTime=3,
            maintenanceOccurrences=2,
            bigM=1000,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Heat demand",
            commodity="heat",
            hasCapacityVariable=False,
            operationRateFix=pd.Series([DEMAND] * NUMBER_OF_TIME_STEPS),
        )
    )
    return esM


def plot_results(esM):
    """Plot boiler operation and shade optimizer-selected maintenance windows."""
    model = esM.componentModelingDict["ConversionDynamicModel"]
    operation = model.operationVariablesOptimum.loc["Boiler", "site"]
    maintenance = model.maintenanceActiveVariablesOptimum.loc["Boiler", "site"]
    backup_operation = DEMAND - operation

    active = maintenance.gt(0.5)
    window_starts = active & ~active.shift(fill_value=False)
    window_number = window_starts.cumsum()
    active_steps = maintenance.loc[active].rename_axis("Time step").reset_index()
    active_steps["Window"] = window_number.loc[active].to_numpy()
    maintenance_windows = active_steps.groupby("Window").agg(
        start_time_step=("Time step", "min"),
        end_time_step=("Time step", "max"),
    )

    figure, axis = plt.subplots(figsize=(12, 5))
    previous_hatch_linewidth = plt.rcParams["hatch.linewidth"]
    plt.rcParams["hatch.linewidth"] = 0.6
    axis.fill_between(
        operation.index,
        0,
        operation,
        step="post",
        facecolor=(0.12, 0.47, 0.71, 0.08),
        edgecolor="tab:blue",
        hatch="///",
        linewidth=0,
        zorder=1,
    )
    plt.rcParams["hatch.linewidth"] = previous_hatch_linewidth
    axis.step(
        backup_operation.index,
        backup_operation,
        where="post",
        linewidth=2.5,
        linestyle="--",
        color="tab:green",
        label="Backup heat supply",
        zorder=3,
    )
    for window, values in maintenance_windows.iterrows():
        axis.axvspan(
            values["start_time_step"],
            values["end_time_step"] + 1,
            alpha=0.35,
            color="tab:red",
            label="Scheduled maintenance" if window == 1 else None,
            zorder=2,
        )
    axis.step(
        operation.index,
        operation,
        where="post",
        color="tab:blue",
        linewidth=2.8,
        label="Boiler operation",
        zorder=5,
    )
    axis.set_title("Optimized boiler operation and maintenance schedule", fontsize=17)
    axis.set_xlabel("Time step", fontsize=15)
    axis.set_ylabel("Operation [kW]", fontsize=15)
    axis.set_ylim(0, DEMAND * 1.1)
    axis.tick_params(axis="both", labelsize=13, width=1.5, length=6)
    axis.grid(axis="y", alpha=0.25, linewidth=1)
    for spine in axis.spines.values():
        spine.set_linewidth(1.5)
    legend = axis.legend(fontsize=12, frameon=True)
    legend.get_frame().set_linewidth(1.2)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    energySystemModel = build_model()
    energySystemModel.optimize()
    plot_results(energySystemModel)
