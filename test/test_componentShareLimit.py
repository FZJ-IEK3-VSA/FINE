"""Tests for the share bounds ("shareMax"/"shareMin") of the componentLimit feature.

The share bounds add an anti-concentration constraint of the form

    e[group] <= value * sum_r e[r]      (shareMax)
    e[group] >= value * sum_r e[r]      (shareMin)

where ``e`` is either the annual operation or the capacity of the limited
component(s), the sum runs over all eligible regions (the "universe", including
the group itself) and ``group`` is either a single region (default) or a bucket of
regions defined via ``componentLimitGrouping``.
"""

import fine as fn
import pandas as pd
import pytest


def _share_componentLimit(ID, value, limitType, bound="shareMax"):
    """Build a one-row componentLimit DataFrame for a single share-limited ID."""
    return pd.DataFrame(
        {
            "value": [value],
            "bound": [bound],
            "type": [limitType],
            "commodity": [None],
            "ip": [2020],
            "ipEnd": [None],
        },
        index=[ID],
    )


def _build_el_esM(
    locations, demandRegion, nhours=24, demand=100.0, **componentLimitKwargs
):
    """Tiny fully-connected electricity system used by the share-limit tests.

    One ``wind`` source is available in every region, demand sits in a single
    region (so production is forced) and a cheap, lossless transmission grid lets
    production flow between regions. The ``wind`` source itself is added by the
    individual tests (operation- vs capacity-limited).
    """
    locations = list(locations)
    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities={"electricity"},
        numberOfTimeSteps=nhours,
        commodityUnitsDict={"electricity": r"MW$_{el}$"},
        hoursPerTimeStep=1,
        startYear=2020,
        costUnit="1e6 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        **componentLimitKwargs,
    )

    # constant demand in a single region (must be met -> forces production)
    demandDf = pd.DataFrame(0.0, index=range(nhours), columns=sorted(locations))
    demandDf[demandRegion] = demand
    esM.add(
        fn.Sink(
            esM=esM,
            name="demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demandDf,
        )
    )
    return esM, locations


def _add_cheap_grid(esM):
    esM.add(
        fn.Transmission(
            esM=esM,
            name="cables",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=0.0001,
            interestRate=0.08,
            economicLifetime=40,
        )
    )


def _wind_operation_per_region(esM, locations):
    opt = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum
    return {loc: float(opt.loc["wind", loc].sum()) for loc in locations}


def _wind_capacity_per_region(esM, locations):
    cap = esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum
    return {loc: float(cap.loc["wind", loc]) for loc in locations}


def test_componentShareLimit_operation_per_region():
    """ShareMax on operation, one constraint per region (no grouping)."""
    locations = ["r0", "r1", "r2"]
    # r0 cheapest -> without the share bound everything would be produced in r0
    costPerRegion = pd.Series({"r0": 0.001, "r1": 0.002, "r2": 0.003})
    alpha = 0.4

    componentLimit = _share_componentLimit("wind_share", alpha, "operation")
    componentLimitEligibility = pd.DataFrame(
        1, index=sorted(locations), columns=["wind_share"]
    )

    esM, locations = _build_el_esM(
        locations,
        demandRegion="r0",
        componentLimit=componentLimit,
        componentLimitEligibility=componentLimitEligibility,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="wind",
            commodity="electricity",
            hasCapacityVariable=False,
            opexPerOperation=costPerRegion,
            componentLimitID=["wind_share"],
        )
    )
    _add_cheap_grid(esM)

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    prod = _wind_operation_per_region(esM, locations)
    total = sum(prod.values())
    assert total > 0
    tol = 1e-4
    # the anti-concentration bound holds for every region
    for loc in locations:
        assert prod[loc] <= alpha * total + tol * total
    # the bound actually binds: the cheapest region is pushed down to its share
    assert prod["r0"] == pytest.approx(alpha * total, rel=1e-3)


def test_componentShareLimit_operation_grouped():
    """ShareMax on operation with a region->group mapping (per-country style)."""
    locations = ["r0", "r1", "r2", "r3"]
    costPerRegion = pd.Series({"r0": 0.001, "r1": 0.002, "r2": 0.003, "r3": 0.004})
    alpha = 0.6

    componentLimit = _share_componentLimit("wind_share", alpha, "operation")
    componentLimitEligibility = pd.DataFrame(
        1, index=sorted(locations), columns=["wind_share"]
    )
    # two groups: A = {r0, r1}, B = {r2, r3}
    groups = {"r0": "A", "r1": "A", "r2": "B", "r3": "B"}
    componentLimitGrouping = pd.DataFrame({"wind_share": pd.Series(groups)}).reindex(
        sorted(locations)
    )

    esM, locations = _build_el_esM(
        locations,
        demandRegion="r0",
        componentLimit=componentLimit,
        componentLimitEligibility=componentLimitEligibility,
        componentLimitGrouping=componentLimitGrouping,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="wind",
            commodity="electricity",
            hasCapacityVariable=False,
            opexPerOperation=costPerRegion,
            componentLimitID=["wind_share"],
        )
    )
    _add_cheap_grid(esM)

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    prod = _wind_operation_per_region(esM, locations)
    total = sum(prod.values())
    assert total > 0
    tol = 1e-4
    groupProd = {"A": prod["r0"] + prod["r1"], "B": prod["r2"] + prod["r3"]}
    for g in ("A", "B"):
        assert groupProd[g] <= alpha * total + tol * total
    # the cheaper group A is pushed down to its share
    assert groupProd["A"] == pytest.approx(alpha * total, rel=1e-3)


def test_componentShareLimit_capacity():
    """ShareMax on capacity, one constraint per region."""
    locations = ["r0", "r1", "r2"]
    alpha = 0.4
    # r0 cheapest capacity -> unconstrained, capacity concentrates in r0
    investPerCapacity = pd.Series({"r0": 0.10, "r1": 0.11, "r2": 0.12})

    componentLimit = _share_componentLimit("wind_cap_share", alpha, "capacity")
    componentLimitEligibility = pd.DataFrame(
        1, index=sorted(locations), columns=["wind_cap_share"]
    )

    esM, locations = _build_el_esM(
        locations,
        demandRegion="r0",
        componentLimit=componentLimit,
        componentLimitEligibility=componentLimitEligibility,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="wind",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=pd.DataFrame(
                1.0, index=range(24), columns=sorted(locations)
            ),
            investPerCapacity=investPerCapacity,
            interestRate=0.08,
            economicLifetime=20,
            componentLimitID=["wind_cap_share"],
        )
    )
    _add_cheap_grid(esM)

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    cap = _wind_capacity_per_region(esM, locations)
    totalCap = sum(cap.values())
    assert totalCap > 0
    tol = 1e-4
    for loc in locations:
        assert cap[loc] <= alpha * totalCap + tol * totalCap


def test_componentShareLimit_rejects_transmission():
    """Share bounds combined with the 2-dim (transmission) eligibility are rejected."""
    locations = ["r0", "r1"]
    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities={"electricity"},
        numberOfTimeSteps=24,
        commodityUnitsDict={"electricity": r"MW$_{el}$"},
        hoursPerTimeStep=1,
        startYear=2020,
        costUnit="1e6 Euro",
        lengthUnit="km",
    )
    componentLimit = _share_componentLimit("grid_share", 0.4, "capacity")
    componentLimitEligibility = pd.DataFrame(
        1, index=sorted(locations), columns=["grid_share"]
    )
    componentLimitEligibility2dim = pd.DataFrame(
        1, index=["r0_r1", "r1_r0"], columns=["grid_share"]
    )
    with pytest.raises(ValueError, match="not supported for"):
        fn.utils.checkAndSetComponentLimit(
            esM,
            componentLimit,
            componentLimitEligibility,
            componentLimitEligibility2dim,
            None,
            esM.locations,
        )


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_componentShareLimit_rejects_value_outside_zero_to_one(value):
    """A share is a fraction, so a value outside [0, 1] is rejected."""
    locations = ["r0", "r1"]
    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities={"electricity"},
        numberOfTimeSteps=24,
        commodityUnitsDict={"electricity": r"MW$_{el}$"},
        hoursPerTimeStep=1,
        startYear=2020,
        costUnit="1e6 Euro",
        lengthUnit="km",
    )
    with pytest.raises(ValueError, match="fraction"):
        fn.utils.checkAndSetComponentLimit(
            esM,
            _share_componentLimit("wind_share", value, "capacity"),
            pd.DataFrame(1, index=sorted(locations), columns=["wind_share"]),
            None,
            None,
            esM.locations,
        )
