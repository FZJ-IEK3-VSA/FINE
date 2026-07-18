"""Visualization helpers for the material storage perfect foresight example."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def plotSourceSinkOperation(esM, component, ip=None, style="points"):
    """Plot source or sink operation for one or more investment periods."""
    if ip is None:
        years = esM.investmentPeriodNames
    elif isinstance(ip, list):
        years = ip
    else:
        years = [ip]

    source_model = esM.componentModelingDict["SourceSinkModel"]
    operations = {
        year: source_model.operationVariablesOptimum[year].loc[
            (component, "GermanyRegion")
        ]
        for year in years
    }

    fig, ax = plt.subplots(figsize=(12, 4))

    if style == "annualTotals":
        totals = [operation.sum() for operation in operations.values()]
        ax.bar(years, totals)
        ax.set_xlabel("investment period")
        ax.set_ylabel(f"total {component} operation")
        ax.set_xticks(years)
        ax.grid(axis="y")
        return fig, ax

    for (year, operation), color in zip(operations.items(), plt.cm.tab10.colors):
        if style == "points":
            ax.plot(
                operation,
                linestyle="",
                marker=".",
                markersize=1.5,
                alpha=0.5,
                color=color,
                label=year,
            )
        elif style == "duration":
            sorted_operation = operation.sort_values(ascending=False).reset_index(
                drop=True
            )
            ax.plot(sorted_operation, color=color, label=year)
        else:
            raise ValueError(
                "style must be 'points', 'duration', or 'annualTotals'."
            )

    ax.set_xlabel("time step" if style == "points" else "sorted time step")
    ax.set_ylabel(f"{component} operation")
    ax.legend()
    ax.grid()
    return fig, ax


def plotMaterialStorageOperation(esM, ip=None, separatePlots=False):
    """Plot material storage charging and discharging operation."""
    if ip is None:
        years = esM.investmentPeriodNames
    elif isinstance(ip, list):
        years = ip
    else:
        years = [ip]

    storage_model = esM.componentModelingDict["MaterialStorageModel"]

    def plot_year(ax, year, color):
        charge = storage_model.chargeOperationVariablesOptimum[year].loc[
            ("Steel storage", "GermanyRegion")
        ]
        discharge = storage_model.dischargeOperationVariablesOptimum[year].loc[
            ("Steel storage", "GermanyRegion")
        ]
        ax.plot(charge, color=color, label=f"{year} Charging")
        ax.plot(
            discharge,
            color=color,
            linestyle="--",
            label=f"{year} Discharging",
        )

    if separatePlots:
        plots = []
        for year, color in zip(years, plt.cm.tab10.colors):
            fig, ax = plt.subplots(figsize=(12, 4))
            plot_year(ax, year, color)
            ax.legend()
            ax.grid()
            plots.append((fig, ax))
        return plots

    fig, ax = plt.subplots(figsize=(12, 4))
    for year, color in zip(years, plt.cm.tab10.colors):
        plot_year(ax, year, color)

    ax.legend()
    ax.grid()
    return fig, ax


def plotSteelDemandCoverage(esM, plotStyle="smooth"):
    """Plot steel demand coverage using smooth curves or annual steps."""
    if plotStyle not in {"smooth", "steps"}:
        raise ValueError("plotStyle must be either 'smooth' or 'steps'.")

    years = list(esM.investmentPeriodNames)
    source_model = esM.componentModelingDict["SourceSinkModel"]
    storage_model = esM.componentModelingDict["MaterialStorageModel"]

    demand = []
    supply = []
    soc = []

    for year in years:
        demand.append(
            source_model.operationVariablesOptimum[year]
            .loc[("Steel demand", "GermanyRegion")]
            .sum()
        )
        soc.append(
            storage_model.getOptimalValues(
                "stateOfChargeStartVariablesOptimum", ip=year
            )["values"].loc["Steel storage", "GermanyRegion"]
        )
        supply.append(
            source_model.operationVariablesOptimum[year]
            .loc[("Steel supply", "GermanyRegion")]
            .sum()
        )

    boundaries = years + [years[-1] + 1]
    midpoints = [year + 0.5 for year in years]
    soc.append(
        storage_model.getOptimalValues(
            "stateOfChargeEndVariablesOptimum", ip=years[-1]
        )["values"].loc["Steel storage", "GermanyRegion"]
    )

    if plotStyle == "smooth":
        flow_years = [boundaries[0]] + midpoints + [boundaries[-1]]
        plot_years = np.linspace(boundaries[0], boundaries[-1], 300)
        plot_demand = PchipInterpolator(
            flow_years, [demand[0]] + demand + [demand[-1]]
        )(plot_years)
        plot_supply = PchipInterpolator(
            flow_years, [supply[0]] + supply + [supply[-1]]
        )(plot_years)
        plot_soc = PchipInterpolator(boundaries, soc)(plot_years)
        fill_kwargs = {}
    else:
        plot_years = boundaries
        plot_demand = demand + [demand[-1]]
        plot_supply = supply + [supply[-1]]
        plot_soc = soc
        fill_kwargs = {"step": "post"}

    figsize = (8, 4) if plotStyle == "smooth" else (9, 4)
    storage_alpha = 0.3 if plotStyle == "smooth" else 0.4
    fig, ax = plt.subplots(figsize=figsize)
    supply_color = "tab:orange"
    ax.fill_between(
        plot_years,
        0,
        np.minimum(plot_demand, plot_supply),
        color=supply_color,
        alpha=0.3,
        label="Direct steel supply",
        **fill_kwargs,
    )
    ax.fill_between(
        plot_years,
        plot_demand,
        np.maximum(plot_demand, plot_supply),
        color="tab:green",
        alpha=storage_alpha,
        label="Material storage charge",
        **fill_kwargs,
    )
    ax.fill_between(
        plot_years,
        np.minimum(plot_demand, plot_supply),
        plot_demand,
        color="tab:red",
        alpha=storage_alpha,
        label="Material storage discharge",
        **fill_kwargs,
    )

    line_kwargs = {"where": "post"} if plotStyle == "steps" else {}
    line_function = ax.step if plotStyle == "steps" else ax.plot
    line_function(
        plot_years,
        plot_demand,
        color="tab:blue",
        linewidth=4 if plotStyle == "steps" else 2.5,
        label=(
            "Demand"
            if plotStyle == "steps"
            else "Demand from capacity expansion"
        ),
        **line_kwargs,
    )
    line_function(
        plot_years,
        plot_supply,
        color=supply_color,
        linestyle="--" if plotStyle == "steps" else "-",
        linewidth=2.5,
        label="Steel supply",
        **line_kwargs,
    )
    ax.scatter(
        midpoints,
        demand,
        facecolors="white",
        edgecolors="tab:blue",
        s=45,
        zorder=4,
    )
    ax.scatter(
        midpoints, supply, color=supply_color, marker="x", s=45, zorder=5
    )
    ax.set(
        xlabel="Year" if plotStyle == "steps" else "Investment period",
        ylabel="Steel [tons/year]" if plotStyle == "steps" else "Steel [tons]",
        title=(
            "Steel demand and supply as annual steps"
            if plotStyle == "steps"
            else "Steel demand and its coverage"
        ),
    )
    if plotStyle == "steps":
        ax.set_xticks(midpoints, years)
        for boundary in boundaries:
            ax.axvline(boundary, color="0.85", linewidth=0.8, zorder=0)
    else:
        ax.set_xticks(boundaries)
    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")

    ax_soc = ax.twinx()
    ax_soc.plot(
        plot_years,
        plot_soc,
        color="black",
        linestyle="--",
        marker="o" if plotStyle == "steps" else None,
        label=(
            "Material storage SOC at year boundary"
            if plotStyle == "steps"
            else "Material storage SOC at IP boundary"
        ),
    )
    if plotStyle == "smooth":
        ax_soc.scatter(boundaries, soc, color="black", s=30)
    ax_soc.set_ylabel("Material storage SOC [tons]")
    ax_soc.set_ylim(bottom=0)

    lines, labels = ax.get_legend_handles_labels()
    soc_lines, soc_labels = ax_soc.get_legend_handles_labels()
    ax.legend(lines + soc_lines, labels + soc_labels)
    return fig, (ax, ax_soc)
