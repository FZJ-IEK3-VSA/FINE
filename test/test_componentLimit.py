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


def _build_esM_with_component_limit(componentLimit, componentLimitEligibility=None):
    """Build a two-region esM through the public API, so the input checks run."""
    locations = {"R1", "R2"}
    if componentLimitEligibility is None:
        componentLimitEligibility = pd.DataFrame(
            index=sorted(locations),
            columns=list(componentLimit.index),
            data=1,
        )
    return fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity"},
        numberOfTimeSteps=4,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        # one hour per time step, so a capacity of 1 delivers 1 unit per step and
        # the capacity limit below is the binding constraint
        hoursPerTimeStep=1,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=componentLimit,
        componentLimitEligibility=componentLimitEligibility,
    )


def test_componentLimit_through_public_api():
    """A capacity limit set through the constructor has to bind the optimum.

    This goes through checkAndSetComponentLimit, unlike the test above, which
    assigns the processed attributes directly.
    """
    componentLimit = pd.DataFrame(
        index=["capLimit"],
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        data=[[3.0, "upper", "capacity", None, 0, None]],
    )
    esM = _build_esM_with_component_limit(componentLimit)

    # the input DataFrame must not be changed under the caller
    assert esM.componentLimit["ip"].tolist() == [0]
    assert componentLimit["ipEnd"].tolist() == [None]

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=0.0,
            opexPerCapacity=0.0,
            interestRate=0.0,
            componentLimitID=["capLimit"],
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                [[10.0, 10.0]] * 4, columns=sorted(esM.locations)
            ),
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Backup",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=1.0,
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    capacities = esM.componentModelingDict[
        "SourceSinkModel"
    ].capacityVariablesOptimum.loc["PV"]
    # PV is free and Backup costs money, so the optimum builds PV up to the
    # limit and no further. An unbound model would build 20.
    assert np.isclose(capacities.sum(), 3.0)


def test_componentLimit_rejects_unknown_id():
    """A component may only name a componentLimitID that componentLimit declares."""
    componentLimit = pd.DataFrame(
        index=["capLimit"],
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        data=[[3.0, "upper", "capacity", None, 0, None]],
    )
    esM = _build_esM_with_component_limit(componentLimit)
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            componentLimitID=["typoLimit"],
        )
    )

    with pytest.raises(ValueError, match="typoLimit"):
        esM.declareOptimizationProblem(timeSeriesAggregation=False)


def test_componentLimit_rejects_invalid_bound_and_type():
    """Only the documented bound and type values are accepted."""
    componentLimit = pd.DataFrame(
        index=["capLimit"],
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        data=[[3.0, "maximum", "capacity", None, 0, None]],
    )
    with pytest.raises(ValueError, match="bound"):
        _build_esM_with_component_limit(componentLimit)

    componentLimit["bound"] = "upper"
    componentLimit["type"] = "power"
    with pytest.raises(ValueError, match="type"):
        _build_esM_with_component_limit(componentLimit)


def test_componentLimit_rejects_capacity_over_period_range():
    """An installed capacity is a stock, so it cannot span investment periods."""
    componentLimit = pd.DataFrame(
        index=["capLimit"],
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        data=[[3.0, "upper", "capacity", None, 0, 0]],
    )
    with pytest.raises(NotImplementedError, match="capacity"):
        _build_esM_with_component_limit(componentLimit)


@pytest.mark.parametrize(
    "given, expected",
    [(None, None), ("LIM1", ["LIM1"]), (["LIM1", "LIM2"], ["LIM1", "LIM2"])],
)
def test_checkAndSetComponentLimitID_normalises_to_list(given, expected):
    """A single ID is accepted and wrapped, a list is kept as it is."""
    assert fn.utils.checkAndSetComponentLimitID(given) == expected


def test_checkAndSetComponentLimitID_rejects_other_types():
    """Anything that is not a string, a list of strings or None is an error."""
    with pytest.raises(ValueError):
        fn.utils.checkAndSetComponentLimitID(1)
    with pytest.raises(ValueError):
        fn.utils.checkAndSetComponentLimitID(["LIM1", 2])
