import FINE as fn
import numpy as np
import pandas as pd
import pytest

from pyomo.opt import SolverFactory


@pytest.mark.skipif(
    not SolverFactory("gurobi").available(),
    reason="QP solver required (check for license)",
)
def test_LPinvest(minimal_test_esM):
    """
    Get the minimal test system, and check if invest of Electrolyzer without quadratic approach is unchanged.
    """
    esM = minimal_test_esM

    esM.optimize(timeSeriesAggregation=False, solver="gurobi")

    # get TAC of Electrolyzer

    invest = (
        esM.getOptimizationSummary("ConversionModel")
        .loc["Electrolyzers"]
        .loc["invest"]["ElectrolyzerLocation"]
        .values.astype(int)[0]
    )

    assert invest == 8571428


@pytest.mark.skipif(
    not SolverFactory("gurobi").available(),
    reason="QP solver required (check for license",
)
def test_QPinvest():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"location1"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    # Buy electricity at the electricity market
    costs = pd.Series([0.5, 0.4, 0.2, 0.5], index=[0, 1, 2, 3])
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCostTimeSeries=costs,
        )
    )  # euro/kWh

    # Electrolyzers
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            QPcostScale=0.1,
            capacityMin=0,
            capacityMax=10,
        )
    )

    # Industry site
    demand = pd.Series([10000.0, 10000.0, 10000.0, 10000.0], index=[0, 1, 2, 3])
    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # Optimize (just executed if gurobi is installed)
    esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    invest = round(
        esM.getOptimizationSummary("ConversionModel")
        .loc["Electrolyzer"]
        .loc["invest"]["location1"]
        .values.astype(float)[0],
        3,
    )
    assert invest == 3148.179

    # flag = True
    # try:
    #     esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    # except:
    #     flag = False

    # if flag:
    #     invest = round(
    #         esM.getOptimizationSummary("ConversionModel")
    #         .loc["Electrolyzer"]
    #         .loc["invest"]["location1"]
    #         .values.astype(float)[0],
    #         3,
    #     )
    #     assert invest == 3148.179
