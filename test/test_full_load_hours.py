import pandas as pd


def test_fullloadhours_above(minimal_test_esM):
    """
    Get the minimal test system, and check if the fulllload hours of electrolyzer are above 4000.
    """
    esM = minimal_test_esM

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # get cumulative operation
    operationSum = (
        esM.componentModelingDict["ConversionModel"]
        .operationVariablesOptimum.xs("Electrolyzers")
        .sum()
        .sum()
    )

    # get capacity
    capacitySum = (
        esM.componentModelingDict["ConversionModel"]
        .capacityVariablesOptimum.xs("Electrolyzers")
        .sum()
    )

    # calculate fullloadhours
    fullloadhours = operationSum / capacitySum

    assert fullloadhours > 4000.0


def test_fullloadhours_max(minimal_test_esM):
    """
    Get the minimal test system, and check if the fulllload hour limitation works
    """

    # modify full load hour limit
    esM = minimal_test_esM

    # get components
    electrolyzer = esM.getComponent("Electrolyzers")
    market = esM.getComponent("Electricity market")

    # set fullloadhour limit
    electrolyzer.yearlyFullLoadHoursMax = pd.Series(100.0, index=esM.locations)
    electrolyzer.processedYearlyFullLoadHoursMax = {
        0: pd.Series(100.0, index=esM.locations)
    }
    market.hasCapacityVariable = True
    market.yearlyFullLoadHoursMax = pd.Series(3000.0, index=esM.locations)
    market.processedYearlyFullLoadHoursMax = {0: pd.Series(3000.0, index=esM.locations)}

    # optimize
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # get cumulative operation
    operationSum = (
        esM.componentModelingDict["ConversionModel"]
        .operationVariablesOptimum.xs("Electrolyzers")
        .sum()
        .sum()
    )
    operationSumMarket = (
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum.xs("Electricity market")
        .sum()
        .sum()
    )

    # get capacity
    capacitySum = (
        esM.componentModelingDict["ConversionModel"]
        .capacityVariablesOptimum.xs("Electrolyzers")
        .sum()
    )
    capacitySumMarket = (
        esM.componentModelingDict["SourceSinkModel"]
        .capacityVariablesOptimum.xs("Electricity market")
        .sum()
    )

    # calculate fullloadhours
    fullloadhours = (operationSum / capacitySum) / esM.numberOfYears
    fullloadhoursMarket = (operationSumMarket / capacitySumMarket) / esM.numberOfYears

    assert fullloadhours < 100.1
    assert fullloadhoursMarket < 3000.1


def test_fullloadhours_min(minimal_test_esM):
    """
    Get the minimal test system, and check if the fulllload hour limitation works
    """

    # modify full load hour limit
    esM = minimal_test_esM

    # get components
    electrolyzer = esM.getComponent("Electrolyzers")
    market = esM.getComponent("Electricity market")

    # set fullloadhour limit
    electrolyzer.yearlyFullLoadHoursMin = pd.Series(5000.0, index=esM.locations)
    electrolyzer.processedYearlyFullLoadHoursMin = {
        0: pd.Series(5000.0, index=esM.locations)
    }
    market.hasCapacityVariable = True
    market.yearlyFullLoadHoursMin = pd.Series(3000.0, index=esM.locations)
    market.processedYearlyFullLoadHoursMin = {0: pd.Series(3000.0, index=esM.locations)}

    # optimize
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # get cumulative operation
    operationSum = (
        esM.componentModelingDict["ConversionModel"]
        .operationVariablesOptimum.xs("Electrolyzers")
        .sum()
        .sum()
    )
    operationSumMarket = (
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum.xs("Electricity market")
        .sum()
        .sum()
    )

    # get capacity
    capacitySum = (
        esM.componentModelingDict["ConversionModel"]
        .capacityVariablesOptimum.xs("Electrolyzers")
        .sum()
    )
    capacitySumMarket = (
        esM.componentModelingDict["SourceSinkModel"]
        .capacityVariablesOptimum.xs("Electricity market")
        .sum()
    )

    # calculate fullloadhours
    fullloadhours = (operationSum / capacitySum) / esM.numberOfYears
    fullloadhoursMarket = (operationSumMarket / capacitySumMarket) / esM.numberOfYears

    assert fullloadhours > 4999.99
    assert fullloadhoursMarket > 3000.1


def test_init_full_load_hours(minimal_test_esM):
    import FINE as fn
    import pandas as pd

    # load minimal test system
    esM = minimal_test_esM

    # add a component with yearly minimal load hours
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_minFLH",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            yearlyFullLoadHoursMin=5000,
        )
    )

    full_load_hours_min = esM.getComponent(
        "Electrolyzers_minFLH"
    ).processedYearlyFullLoadHoursMin
    full_load_hours_max = esM.getComponent(
        "Electrolyzers_minFLH"
    ).processedYearlyFullLoadHoursMax

    assert isinstance(full_load_hours_min[0], pd.Series)
    assert full_load_hours_max is None
