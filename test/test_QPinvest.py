import fine as fn
import pandas as pd
import pytest

from pyomo.opt import SolverFactory


@pytest.mark.skipif(
    not SolverFactory("gurobi").available(),
    reason="QP solver required (check for license",
)
@pytest.mark.parametrize("capacityMin", [0, 5])
def test_QPinvest(capacityMin):
    capacityMin_variation = 5

    ###########################################################################
    # 1. Set up energy system model
    ###########################################################################
    # 1.1 initialize energy system model
    numberOfTimeSteps = 4
    hoursPerTimeStep = 1
    esM = fn.EnergySystemModel(
        locations={"location1"},
        commodities={"input1", "output1"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "input1": "input1",
            "output1": "output1",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # 1.2 add basic components
    esM.add(
        fn.Source(
            esM=esM,
            name="dummy_source",
            commodity="input1",
            hasCapacityVariable=True,
            operationRateFix=pd.Series(data=[1, 1, 1, 1]),
        )
    )
    fix_demand = 10
    esM.add(
        fn.Sink(
            esM=esM,
            name="dummy_sink",
            commodity="output1",
            hasCapacityVariable=False,
            operationRateFix=pd.Series(
                data=[fix_demand, fix_demand, fix_demand, fix_demand]
            ),
        )
    )

    # 1.3 add component with qp
    esM.add(
        fn.Conversion(
            esM=esM,
            name="qp_technology",
            physicalUnit="input1",
            commodityConversionFactors={"input1": -1, "output1": 1},
            hasCapacityVariable=True,
            investPerCapacity=800,  # euro/kW
            opexPerCapacity=800 * 0.1,
            interestRate=0.0,
            economicLifetime=20,
            QPcostScale=0.25,
            capacityMin=capacityMin_variation,
            capacityMax=40,
        )
    )

    # 1.4 optimize
    esM.optimize(timeSeriesAggregation=False, solver="gurobi")

    ###########################################################################
    # 2. check results
    ###########################################################################
    # 2.1 investment costs
    invest = (
        esM.getOptimizationSummary("ConversionModel")
        .loc["qp_technology"]
        .loc["invest"]["location1"]
        .values.astype(int)[0]
    )
    assert invest == 650 * 10
    # qp installes between 600 and 700 (800-0.5*0.25*800)
    # therefore the average investment costs are 650
    # (the cheapest is 600, the most expensive 700, average 650)

    # 2.2 opex fix
    correct_opex_fix_contribution = 650 * 0.1 * 10
    opexCap = (
        esM.getOptimizationSummary("ConversionModel")
        .loc["qp_technology"]
        .loc["opexCap"]["location1"]
        .values.astype(int)[0]
    )
    assert correct_opex_fix_contribution == opexCap

    # 2.3 NPV contribution and TAC
    npv_contribution = (
        esM.getOptimizationSummary("ConversionModel")
        .loc["qp_technology"]
        .loc["NPVcontribution"]["location1"]
        .values.astype(int)[0]
    )
    tac = (
        esM.getOptimizationSummary("ConversionModel")
        .loc["qp_technology"]
        .loc["TAC"]["location1"]
        .values.astype(int)[0]
    )

    assert npv_contribution == tac

    correct_capex_contribution = 650 * 10 * 0.05
    # 650 as average capex, 10 as commissioning,
    # devide by 20 as annuity for lifetime 20 and interestRate 0

    assert tac == correct_capex_contribution + correct_opex_fix_contribution
