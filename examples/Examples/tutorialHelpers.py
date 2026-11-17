"""Helpers shared by the notebooks of the modular examples.

The notebooks build on the model of the basic tutorial (``01_basic_tutorial.ipynb``).
This module provides:

- ``readData``: reads one of the CSV files in ``examples/data/twoRegions``.
- ``buildBasicTutorialModel``: builds the model of the basic tutorial in one call.
- ``addHydrogenSupply``: adds the hydrogen components of the Conversions notebook.
- ``COLORS`` and ``colormap``: the colors of the components, the same in all notebooks.
- ``plotCapacityAndGeneration``: the bar charts of capacity and generation per
  investment period.

Importing this module also sets the plot style that all notebooks share.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

import fine as fn

#: Folder with the input data of the tutorial notebooks.
DATA_PATH = Path(__file__).parents[1] / "data" / "twoRegions"

#: CO2 limit of the basic tutorial in million tonnes per year.
CO2_LIMIT_PER_PERIOD = {2020: 90, 2030: 45, 2040: 20}

#: Color of every component, the same in all notebooks.
COLORS = {
    # electricity from renewables
    "Wind turbines": "#0072B2",
    "Offshore wind turbines": "#56B4E9",
    "PV": "#E69F00",
    "Open-field PV": "#9E5A00",
    "Run-of-river plants": "#009E73",
    # fossil electricity and CO2
    "Gas power plants": "#7F7F7F",
    "CHP plants": "#3F3F3F",
    "Biogas power plants": "#117733",
    "Natural gas import": "#7F7F7F",
    "CO2 to environment": "#3F3F3F",
    # storage and grid
    "Batteries": "#D55E00",
    "Pumped hydro storage": "#009E73",
    "AC cables": "#332288",
    "DC cables": "#999933",
    # hydrogen
    "Electrolyzers": "#CC79A7",
    "Hydrogen power plants": "#882255",
    "Hydrogen import": "#882255",
    "Salt caverns (hydrogen)": "#CC79A7",
    # demands and exports
    "Electricity demand": "#000000",
    "Electricity export": "#44AA99",
    "Hydrogen demand": "#000000",
}

#: Line style per region, used where both regions appear in the same plot.
REGION_LINESTYLES = {"north": "-", "south": "--"}


def _setPlotStyle():
    """Set the plot style shared by all notebooks."""
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            # fixed order of colors for plots that do not use COLORS
            "axes.prop_cycle": mpl.cycler(
                color=["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
            ),
        }
    )


_setPlotStyle()


def colormap(component):
    """Return a colormap from white to the color of a component, for heat maps."""
    return LinearSegmentedColormap.from_list(
        component, ["#FFFFFF", to_rgb(COLORS[component])]
    )


def readData(fileName):
    """Read a CSV file of the tutorial data (first column as index)."""
    return pd.read_csv(DATA_PATH / fileName, index_col=0)


def buildBasicTutorialModel(
    co2LimitPerPeriod=CO2_LIMIT_PER_PERIOD,
    commodityUnits=None,
    balanceLimit=None,
    pathwayBalanceLimit=None,
    annuityPerpetuity=False,
):
    """Build the (not yet optimized) model of the basic tutorial.

    :param co2LimitPerPeriod: CO2 limit per investment period in million tonnes per
        year. If None, the model has neither a CO2 limit nor the sink
        "CO2 to environment", so that a notebook can add its own.
    :param commodityUnits: further commodities and their units, e.g.
        ``{"hydrogen": "GW_H2"}``. All commodities have to be known when the
        model is created.
    :param balanceLimit: further balance limits (a ``pd.DataFrame`` in the format of
        the ``balanceLimit`` of the ``EnergySystemModel``), which apply in every
        investment period in addition to the CO2 limit.
    :param pathwayBalanceLimit: passed on to the ``EnergySystemModel``.
    :param annuityPerpetuity: passed on to the ``EnergySystemModel``.
    """
    windCapacityFactor = readData("windCapacityFactor.csv")
    pvCapacityFactor = readData("pvCapacityFactor.csv")
    electricityDemand2020 = readData("electricityDemand.csv")
    capacityPotential = readData("capacityPotential.csv")
    distances = readData("distances.csv")

    commodityUnitsDict = {
        "electricity": "GW_el",
        "naturalGas": "GW_CH4",
        "CO2": "Mt_CO2/h",
        **(commodityUnits or {}),
    }

    balanceLimitPerPeriod = {}
    for year in [2020, 2030, 2040]:
        rows = []
        if co2LimitPerPeriod is not None:
            rows.append(
                pd.DataFrame(
                    index=["CO2 limit"],
                    columns=["Total", "lowerBound"],
                    data=[[-co2LimitPerPeriod[year], True]],
                )
            )
        if balanceLimit is not None:
            rows.append(balanceLimit)
        if rows:
            # Regions without a limit are marked with None
            limits = pd.concat(rows).astype(object)
            balanceLimitPerPeriod[year] = limits.where(limits.notna(), None)
    if not balanceLimitPerPeriod:
        balanceLimitPerPeriod = None

    esM = fn.EnergySystemModel(
        locations={"north", "south"},
        commodities=set(commodityUnitsDict),
        commodityUnitsDict=commodityUnitsDict,
        numberOfTimeSteps=8760,
        hoursPerTimeStep=1,
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=10,
        startYear=2020,
        costUnit="1e6 Euro",
        lengthUnit="km",
        balanceLimit=balanceLimitPerPeriod,
        pathwayBalanceLimit=pathwayBalanceLimit,
        annuityPerpetuity=annuityPerpetuity,
        verboseLogLevel=1,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind turbines",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=windCapacityFactor,
            capacityMax=capacityPotential.loc["Wind (onshore)"],
            investPerCapacity={2020: 1300, 2030: 1150, 2040: 1050},
            opexPerCapacity=30,
            interestRate=0.08,
            economicLifetime=20,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=pvCapacityFactor,
            capacityMax=capacityPotential.loc["PV"],
            investPerCapacity={2020: 750, 2030: 500, 2040: 380},
            opexPerCapacity=12,
            interestRate=0.08,
            economicLifetime=25,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas import",
            commodity="naturalGas",
            hasCapacityVariable=False,
            commodityCost={2020: 0.025, 2030: 0.03, 2040: 0.035},
        )
    )
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Gas power plants",
            physicalUnit="GW_el",
            commodityConversionFactors={
                "electricity": 1,
                "naturalGas": -1 / 0.6,
                "CO2": 201e-6 / 0.6,
            },
            hasCapacityVariable=True,
            investPerCapacity=650,
            opexPerCapacity=20,
            interestRate=0.08,
            economicLifetime=30,
        )
    )
    esM.add(
        fn.Storage(
            esM=esM,
            name="Batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=0.95,
            dischargeEfficiency=0.95,
            selfDischarge=4.23e-5,
            chargeRate=1,
            dischargeRate=1,
            opexPerChargeOperation=0.001,
            investPerCapacity={2020: 300, 2030: 180, 2040: 120},
            opexPerCapacity=3,
            interestRate=0.08,
            economicLifetime=10,
        )
    )
    esM.add(
        fn.Transmission(
            esM=esM,
            name="AC cables",
            commodity="electricity",
            hasCapacityVariable=True,
            distances=distances,
            losses=0.0001,
            investPerCapacity=1.5,
            opexPerCapacity=0.015,
            interestRate=0.08,
            economicLifetime=40,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix={
                2020: electricityDemand2020,
                2030: electricityDemand2020 * 1.15,
                2040: electricityDemand2020 * 1.15**2,
            },
        )
    )
    if co2LimitPerPeriod is not None:
        esM.add(
            fn.Sink(
                esM=esM,
                name="CO2 to environment",
                commodity="CO2",
                hasCapacityVariable=False,
                balanceLimitID="CO2 limit",
            )
        )
    return esM


def addHydrogenSupply(esM):
    """Add the hydrogen components of the Conversions notebook to a model.

    These are the hydrogen demand of the industry (from 2030 on), the hydrogen
    import and the electrolyzers. The model must contain the commodity "hydrogen"
    (``buildBasicTutorialModel(commodityUnits={"hydrogen": "GW_H2"})``).
    """
    hydrogenDemand = readData("hydrogenDemand.csv")
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix={
                2020: hydrogenDemand * 0,
                2030: hydrogenDemand * 0.5,
                2040: hydrogenDemand,
            },
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Hydrogen import",
            commodity="hydrogen",
            hasCapacityVariable=False,
            commodityCost={2020: 0.15, 2030: 0.12, 2040: 0.10},
        )
    )
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit="GW_el",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity={2020: 1500, 2030: 800, 2040: 500},
            opexPerCapacity=30,
            opexPerOperation=0.002,
            interestRate=0.08,
            economicLifetime=20,
        )
    )


def _capacityAndGeneration(esM, components):
    """Collect capacity and yearly generation per period, region and component."""
    rows = {}
    for year in esM.investmentPeriodNames:
        for modelingClass in ["SourceSinkModel", "ConversionModel"]:
            if modelingClass not in esM.componentModelingDict:
                continue
            summary = esM.getOptimizationSummary(modelingClass, outputLevel=1, ip=year)
            capacities = summary.xs("capacity", level="Property").droplevel("Unit")
            operations = summary.xs("operation", level="Property").droplevel("Unit")
            for component in components:
                if component in capacities.index:
                    for region in capacities.columns:
                        rows[year, region, component] = (
                            capacities.loc[component, region],
                            operations.loc[component, region] / 1000,  # TWh
                        )
    table = pd.DataFrame(rows, index=["capacity", "generation"]).T.astype(float)
    table.index.names = ["period", "region", "component"]
    return table


def plotCapacityAndGeneration(esM, components, title="", byRegion=False):
    """Plot capacity and yearly generation of electricity generators per period.

    Generators can be sources or conversions. Each component has its color from
    ``COLORS``.

    :param components: the components to plot, in the order of the stacked bars
    :param title: text added to the titles of the plots
    :param byRegion: if True, one column of plots per region; otherwise both
        regions are added up
    :return: capacity [GW] and generation [TWh per year] as DataFrames with the
        periods as rows (and, if byRegion, the regions as the first column level)
    """
    table = _capacityAndGeneration(esM, components)
    regions = sorted(table.index.get_level_values("region").unique())
    groups = regions if byRegion else [None]

    nCols = 2 if not byRegion else len(regions)
    nRows = 1 if not byRegion else 2
    fig, axes = plt.subplots(
        nRows, nCols, figsize=(13, 4.5 * nRows), sharey="row" if byRegion else False
    )
    axes = axes.reshape(nRows, nCols)

    capacity, generation = {}, {}
    for column, region in enumerate(groups):
        if region is None:
            part = table.groupby(["period", "component"]).sum()
        else:
            part = table.xs(region, level="region")
        for row, (prop, unit) in enumerate(
            [
                ("capacity", "Capacity [GW_el]"),
                ("generation", "Generation [TWh_el per year]"),
            ]
        ):
            data = part[prop].unstack("component").reindex(columns=components).fillna(0)
            if byRegion:
                ax = axes[row, column]
                regionTitle = f" ({region})"
            else:
                ax = axes[0, row]
                regionTitle = ""
            data.plot.bar(
                ax=ax,
                stacked=True,
                color=[COLORS[c] for c in components],
                rot=0,
                width=0.7,
                edgecolor="white",
                linewidth=1,
                legend=False,
                xlabel="Investment period",
                ylabel=unit,
            )
            name = (
                "Installed capacity" if prop == "capacity" else "Electricity generation"
            )
            ax.set_title(f"{name}{regionTitle}{title}")
            (capacity if prop == "capacity" else generation)[region] = data

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(0.91, 0.88)
    )
    fig.subplots_adjust(right=0.9, hspace=0.35)

    if byRegion:
        return pd.concat(capacity, axis=1), pd.concat(generation, axis=1)
    return capacity[None], generation[None]
