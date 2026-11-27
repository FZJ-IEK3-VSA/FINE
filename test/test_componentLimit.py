import fine as fn
import pandas as pd
import numpy as np
import pytest


def test_componentLimitConstraint(multi_node_test_esM_init):
    locations = multi_node_test_esM_init.locations

    # 1) Define componentLimit

    # We are constructing 3 componentLimits
    # 1. LIM1: Capacity limit of 20 for Wind (offshore) in locations cluster_0, cluster_1, cluster_2
    # 2. LIM2: Operation limit of 10000 for Wind (onshore) in locations cluster_5, cluster_6, cluster_7
    # 3. LIM3: Fixed capacity limit of 10 for Wind (offshore) and Wind (onshore) combined in locations cluster_0, cluster_1

    # componentLimit is a single DataFrame; "ip"/"ipEnd" are internal investment
    # period indices (None = open-ended) and "commodity" is only used for conversions.
    _componentLimit = pd.DataFrame(
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        index=["LIM1", "LIM2", "LIM3"],
        data=[
            [20, "upper", "capacity", None, 0, None],
            [10000, "upper", "operation", None, 0, None],
            [10, "fixed", "capacity", None, 0, None],
        ],
    )

    _componentLimitEligibility = pd.DataFrame(
        columns=["LIM1", "LIM2", "LIM3"],
        index=sorted(list(locations)),
        data=[
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
    )

    multi_node_test_esM_init.updateComponent(
        componentName="Wind (offshore)",
        updateAttrs={"componentLimitID": ["LIM1", "LIM3"]},
    )
    multi_node_test_esM_init.updateComponent(
        componentName="Wind (onshore)",
        updateAttrs={"componentLimitID": ["LIM2", "LIM3"]},
    )

    # Add componentLimit to esM manually. Not recommended for normal use.
    multi_node_test_esM_init.processedComponentLimit = _componentLimit
    multi_node_test_esM_init.processedComponentLimitEligibility = (
        _componentLimitEligibility
    )

    # 2) Optimize esM
    multi_node_test_esM_init.aggregateTemporally(
        numberOfTypicalPeriods=3,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    multi_node_test_esM_init.optimize(timeSeriesAggregation=True, solver="glpk")

    _data = multi_node_test_esM_init.componentModelingDict[
        "SourceSinkModel"
    ].capacityVariablesOptimum
    elig = (
        _componentLimitEligibility[_componentLimitEligibility == 1]["LIM1"]
        .dropna()
        .index
    )
    value = _data.loc["Wind (offshore)", elig].sum()

    assert np.greater_equal(20, np.round(value, 0))

    _data = multi_node_test_esM_init.componentModelingDict[
        "SourceSinkModel"
    ].operationVariablesOptimum
    elig = (
        _componentLimitEligibility[_componentLimitEligibility == 1]["LIM2"]
        .dropna()
        .index
    )
    value = _data.loc[("Wind (onshore)", elig), :].sum().sum()
    assert np.greater_equal(10000, np.round(value, 0))

    _data = multi_node_test_esM_init.componentModelingDict[
        "SourceSinkModel"
    ].capacityVariablesOptimum
    elig = (
        _componentLimitEligibility[_componentLimitEligibility == 1]["LIM3"]
        .dropna()
        .index
    )
    techs = ["Wind (offshore)", "Wind (onshore)"]
    value = _data.loc[techs, elig].sum().sum()

    assert np.isclose(10, value)


def test_componentLimit_with_location_specific_conversion_factor():
    """A componentLimit weighted by a commodity uses the factor of each location
    when the factor is given per location (pd.Series), e.g. for a myopic stock
    component whose locations hold builds of different efficiency. Before the
    fix, building the constraint failed with 'invalid index to scalar variable'."""
    esM = fn.EnergySystemModel(
        locations={"Loc1", "Loc2"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=1,
        commodityUnitsDict={"electricity": r"kW$_{el}$", "hydrogen": r"kW$_{H_{2},LHV}$"},
        hoursPerTimeStep=1,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )
    esM.add(fn.Source(esM=esM, name="Electricity market", commodity="electricity",
                      hasCapacityVariable=False, commodityCost=1.0))
    # cheaper hydrogen: without the limit, the electrolyser would not run
    esM.add(fn.Source(esM=esM, name="H2 import", commodity="hydrogen",
                      hasCapacityVariable=False, commodityCost=0.1))
    esM.add(fn.Sink(esM=esM, name="H2 demand", commodity="hydrogen", hasCapacityVariable=False,
                    operationRateFix=pd.DataFrame({"Loc1": [10.0], "Loc2": [20.0]},
                                                  index=esM.totalTimeSteps)))
    esM.add(fn.Conversion(esM=esM, name="Electrolyser", physicalUnit=r"kW$_{el}$",
                          commodityConversionFactors={
                              "electricity": -1,
                              "hydrogen": pd.Series({"Loc1": 0.5, "Loc2": 1.0}),
                          },
                          hasCapacityVariable=True, investPerCapacity=0, opexPerCapacity=0,
                          interestRate=0.08, economicLifetime=10,
                          componentLimitID="H2_MIN_LOC1"))
    # at least 4 units of hydrogen from the electrolyser in Loc1
    esM.processedComponentLimit = pd.DataFrame(
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        index=["H2_MIN_LOC1"],
        data=[[4.0, "lower", "operation", "hydrogen", 0, None]],
    )
    esM.processedComponentLimitEligibility = pd.DataFrame(
        {"H2_MIN_LOC1": [1, 0]}, index=["Loc1", "Loc2"])

    import pyomo.environ as pyomo

    solver = next((s for s in ("glpk", "highs", "gurobi")
                   if pyomo.SolverFactory(s).available(exception_flag=False)), None)
    if solver is None:
        pytest.skip("no LP solver available")
    esM.optimize(timeSeriesAggregation=False, solver=solver)

    op = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs("Electrolyser")
    # 4 hydrogen at a factor of 0.5 needs 8 electricity (a factor of 1 would give 4)
    np.testing.assert_almost_equal(op.loc["Loc1", 0], 8.0, decimal=6)
    np.testing.assert_almost_equal(op.loc["Loc2", 0], 0.0, decimal=6)
