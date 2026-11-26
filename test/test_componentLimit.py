import fine as fn
import pandas as pd
import numpy as np
import pytest

from fine.utils import ImplementedSolvers


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


def _twoRegionEsM(componentLimit, eligibility, eligibility2dim=None, **esMKwargs):
    """Two regions, one time step per hour, electricity demand of 10 per step."""
    esM = fn.EnergySystemModel(
        locations={"R1", "R2"},
        commodities={"electricity", "gas", "biogas"},
        numberOfTimeSteps=4,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "gas": r"kW$_{gas}$",
            "biogas": r"kW$_{gas}$",
        },
        hoursPerTimeStep=1,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=componentLimit,
        componentLimitEligibility=eligibility,
        componentLimitEligibility2dim=eligibility2dim,
        **esMKwargs,
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
    # expensive fallback, so that every cheaper path is used up to its limit
    esM.add(
        fn.Source(
            esM=esM,
            name="Backup",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=100.0,
        )
    )
    return esM


def _gasLimit(value, ip=0):
    """Gas use of at most `value`: gas has a negative conversion factor."""
    return pd.DataFrame(
        index=["gasUse"],
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        data=[[-value, "lower", "operation", "gas", ip, None]],
    )


_ELIGIBLE_EVERYWHERE = pd.DataFrame(1, index=["R1", "R2"], columns=["gasUse"])


def _plantFactors(variant):
    if variant == "scalar":
        return {"electricity": 1, "gas": -2}, {}
    if variant == "series_per_location":
        return {"electricity": 1, "gas": pd.Series({"R1": -2.0, "R2": -2.5})}, {}
    if variant == "commissioning_dependent":
        return {(2020, 2020): {"electricity": 1, "gas": -2}}, {
            "commissioningDependentCcf": True
        }
    if variant == "flexible":
        return {"electricity": 1, "fuel": {"gas": -2, "biogas": -2}}, {}
    raise ValueError(variant)


@pytest.mark.parametrize(
    "variant",
    ["scalar", "series_per_location", "commissioning_dependent", "flexible"],
)
def test_componentLimit_conversion_operation_in_commodity(variant):
    """A Conversion's operation counts in the limit's commodity.

    The plant burns cheap gas to cover the demand, so its gas use is bounded
    only by the componentLimit. All gas comes from the source "Gas", so the
    source's annual output is the plant's gas use.
    """
    esMKwargs = {"startYear": 2020} if variant == "commissioning_dependent" else {}
    esM = _twoRegionEsM(
        _gasLimit(20.0, ip=2020 if esMKwargs else 0),
        _ELIGIBLE_EVERYWHERE,
        **esMKwargs,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Gas",
            commodity="gas",
            hasCapacityVariable=False,
            commodityCost=1.0,
        )
    )
    # biogas only feeds the flexible plant, and is cheaper than the backup
    esM.add(
        fn.Source(
            esM=esM,
            name="Biogas",
            commodity="biogas",
            hasCapacityVariable=False,
            commodityCost=10.0,
        )
    )
    factors, plantKwargs = _plantFactors(variant)
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Plant",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors=factors,
            hasCapacityVariable=True,
            investPerCapacity=0.0,
            interestRate=0.0,
            componentLimitID="gasUse",
            **plantKwargs,
        )
    )

    esM.optimize(
        timeSeriesAggregation=False, solver=ImplementedSolvers.STANDARD_SOLVER.value
    )

    operation = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum
    gasUse = operation.loc["Gas"].to_numpy().sum()
    # an unbound model would burn far more than 20 units of gas
    assert np.isclose(gasUse, 20.0)


def test_componentLimit_conversion_ignores_unconverted_commodity():
    """A Conversion that does not convert the limit's commodity contributes 0."""
    esM = _twoRegionEsM(_gasLimit(20.0), _ELIGIBLE_EVERYWHERE)
    esM.add(
        fn.Source(
            esM=esM,
            name="Biogas",
            commodity="biogas",
            hasCapacityVariable=False,
            commodityCost=1.0,
        )
    )
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Plant",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": 1, "biogas": -2},
            hasCapacityVariable=True,
            investPerCapacity=0.0,
            interestRate=0.0,
            componentLimitID="gasUse",
        )
    )

    esM.optimize(
        timeSeriesAggregation=False, solver=ImplementedSolvers.STANDARD_SOLVER.value
    )

    # nothing limits the plant, so it covers the whole demand of 80
    operation = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum
    assert np.isclose(operation.loc["Backup"].to_numpy().sum(), 0.0)


def test_componentLimit_transmission_operation_on_connections():
    """An operation limit on connections bounds the flow on exactly those."""
    componentLimit = pd.DataFrame(
        index=["lineFlow"],
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        data=[[15.0, "upper", "operation", None, 0, None]],
    )
    eligibility2dim = pd.DataFrame(
        1, index=pd.MultiIndex.from_tuples([("R1", "R2")]), columns=["lineFlow"]
    )
    esM = _twoRegionEsM(
        componentLimit,
        pd.DataFrame(index=["R1", "R2"]),
        eligibility2dim=eligibility2dim,
    )
    # cheap electricity is available in R1 only
    esM.add(
        fn.Source(
            esM=esM,
            name="Cheap",
            commodity="electricity",
            hasCapacityVariable=False,
            locationalEligibility=pd.Series({"R1": 1, "R2": 0}),
            commodityCost=1.0,
        )
    )
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Line",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=0.0,
            interestRate=0.0,
            componentLimitID="lineFlow",
        )
    )

    esM.optimize(
        timeSeriesAggregation=False, solver=ImplementedSolvers.STANDARD_SOLVER.value
    )

    # R2 needs 40 over the four steps; the line may carry 15 of it
    operation = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum
    assert np.isclose(operation.loc[("Backup", "R2")].sum(), 25.0)


def _esMWithLimitInput(
    limitRows=None, eligibility="default", eligibility2dim=None, grouping=None, **kw
):
    """Create a three-period, two-region esM with the given componentLimit input."""
    columns = ["value", "bound", "type", "commodity", "ip", "ipEnd"]
    componentLimit = (
        None if limitRows is None else pd.DataFrame(**limitRows, columns=columns)
    )
    if isinstance(eligibility, str):
        eligibility = pd.DataFrame(1, index=["R1", "R2"], columns=["L"])
    return fn.EnergySystemModel(
        locations={"R1", "R2"},
        commodities={"electricity"},
        numberOfTimeSteps=4,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=1,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        startYear=2020,
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        componentLimit=componentLimit,
        componentLimitEligibility=eligibility,
        componentLimitEligibility2dim=eligibility2dim,
        componentLimitGrouping=grouping,
        **kw,
    )


def _row(value=1.0, bound="upper", typ="operation", ip=2020, ipEnd=None, ID="L"):
    return {"index": [ID], "data": [[value, bound, typ, None, ip, ipEnd]]}


_CONNECTIONS = pd.MultiIndex.from_tuples([("R1", "R2"), ("R2", "R1")])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"limitRows": _row(ip=2030, ipEnd=2020)}, "before 'ip'"),
        ({"limitRows": _row(ip=None)}, "has to name an investment period"),
        ({"limitRows": _row(value=np.nan)}, "finite number"),
        ({"limitRows": _row(value="many")}, "finite number"),
        (
            {"limitRows": _row(ipEnd=2030), "stochasticModel": True},
            "stochastic model",
        ),
        ({"limitRows": _row(), "eligibility": None}, "needs componentLimitEligibility"),
        (
            {
                "limitRows": None,
                "eligibility": None,
                "eligibility2dim": pd.DataFrame(1, index=_CONNECTIONS, columns=["L"]),
            },
            "given without componentLimit",
        ),
        (
            {
                "limitRows": _row(),
                "eligibility2dim": pd.DataFrame(1, index=_CONNECTIONS, columns=["L"]),
            },
            "exactly one",
        ),
        (
            {"limitRows": _row(ID="other")},
            "exactly one",
        ),
        (
            {
                "limitRows": _row(),
                "eligibility": pd.DataFrame(0, index=["R1", "R2"], columns=["L"]),
            },
            "No location or connection",
        ),
        (
            {
                "limitRows": _row(),
                "eligibility": None,
                "eligibility2dim": pd.DataFrame(
                    1, index=["R1_R2", "R2_R1"], columns=["L"]
                ),
            },
            "pairs of two different model locations",
        ),
        (
            {
                "limitRows": _row(),
                "eligibility": None,
                "eligibility2dim": pd.DataFrame(
                    1,
                    index=pd.MultiIndex.from_tuples([("R1", "R3")]),
                    columns=["L"],
                ),
            },
            "pairs of two different model locations",
        ),
    ],
)
def test_componentLimit_input_checks(kwargs, message):
    """Inputs the specification rejects raise an error when the esM is created."""
    with pytest.raises(ValueError, match=message):
        _esMWithLimitInput(**kwargs)


def test_componentLimit_accepts_connections_only():
    """A componentLimit may be summed over connections alone."""
    esM = _esMWithLimitInput(
        limitRows=_row(),
        eligibility=None,
        eligibility2dim=pd.DataFrame(1, index=_CONNECTIONS, columns=["L"]),
    )
    assert esM.processedComponentLimit is not None
    assert esM.processedComponentLimitEligibility is None


def test_componentLimit_accepts_rows_with_and_without_ipEnd():
    """Rows with and without a range may be mixed in one componentLimit.

    A mixed ipEnd column becomes a float column with NaN, which has to be
    read as "not set".
    """
    esM = _esMWithLimitInput(
        limitRows={
            "index": ["L", "L"],
            "data": [
                [1.0, "upper", "operation", None, 2020, None],
                [1.0, "upper", "operation", None, 2020, 2030],
            ],
        }
    )
    ipEnd = esM.processedComponentLimit["ipEnd"]
    assert pd.isna(ipEnd.iloc[0])
    assert ipEnd.iloc[1] == 2
