"""The model of the basic tutorial, for the component notebooks.

The component notebooks (``_3_sources.ipynb`` to ``_7_transmissions.ipynb``) start
from the model that the basic tutorial (``_2_basic_tutorial.ipynb``) builds step by
step. ``buildBasicTutorialModel`` builds the same model in one call, and
``plotCapacityAndGeneration`` draws the bar charts of the tutorial's results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import fine as fn

#: Folder with the input data of the tutorial notebooks.
DATA_PATH = Path(__file__).parents[2] / "data" / "twoRegions"

#: CO2 limit of the basic tutorial in million tonnes per year.
CO2_LIMIT_PER_PERIOD = {2020: 90, 2030: 45, 2040: 20}


def readData(fileName):
    """Read a CSV file of the tutorial data (first column as index)."""
    return pd.read_csv(DATA_PATH / fileName, index_col=0)


def buildBasicTutorialModel(
    co2LimitPerPeriod=CO2_LIMIT_PER_PERIOD, commodityUnits=None
):
    """Build the (not yet optimized) model of the basic tutorial.

    :param co2LimitPerPeriod: CO2 limit per investment period in million tonnes per
        year. If None, the model has neither a CO2 limit nor the sink
        "CO2 to environment", so that a notebook can add its own.
    :param commodityUnits: further commodities and their units, e.g.
        ``{"hydrogen": "GW_H2"}``. All commodities have to be known when the
        model is created.
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

    balanceLimit = None
    if co2LimitPerPeriod is not None:
        balanceLimit = {
            year: pd.DataFrame(
                index=["CO2 limit"],
                columns=["Total", "lowerBound"],
                data=[[-limit, True]],
            )
            for year, limit in co2LimitPerPeriod.items()
        }

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
        balanceLimit=balanceLimit,
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


def plotCapacityAndGeneration(esM, colors, title=""):
    """Plot capacity and yearly generation of electricity generators per period.

    Both regions are added up. Generators can be sources or conversions.

    :param colors: component names (in the order of the stacked bars) and their colors
    :param title: text added to the titles of both plots
    """
    periods = esM.investmentPeriodNames
    capacity = pd.DataFrame(0.0, index=periods, columns=list(colors))
    generation = capacity.copy()
    for year in periods:
        for modelingClass in ["SourceSinkModel", "ConversionModel"]:
            summary = esM.getOptimizationSummary(modelingClass, outputLevel=1, ip=year)
            capacities = summary.xs("capacity", level="Property").droplevel("Unit")
            operations = summary.xs("operation", level="Property").droplevel("Unit")
            for component in colors:
                if component in capacities.index:
                    capacity.loc[year, component] = capacities.loc[component].sum()
                    generation.loc[year, component] = (
                        operations.loc[component].sum() / 1000
                    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plotSettings = dict(
        stacked=True, color=list(colors.values()), rot=0, xlabel="Investment period"
    )
    capacity.plot.bar(
        ax=axes[0],
        legend=False,
        ylabel="Capacity [GW_el]",
        title=f"Cumulative installed capacity{title}",
        **plotSettings,
    )
    generation.plot.bar(
        ax=axes[1],
        ylabel="Generation [TWh_el per year]",
        title=f"Electricity generation{title}",
        **plotSettings,
    )
    axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    return capacity, generation
