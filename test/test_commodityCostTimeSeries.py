# https://unix.stackexchange.com/questions/74571/vim-shortcut-to-open-a-file-under-cursor-in-an-already-opened-window!/usr/bin/env python

# # Workflow for a multi-regional energy system
#
# In this application of the FINE framework, a multi-regional energy system is modeled and optimized.
#
# All classes which are available to the user are utilized and examples of the selection of different parameters within these classes are given.
#
# The workflow is structures as follows:
# 1. Required packages are imported and the input data path is set
# 2. An energy system model instance is created
# 3. Commodity sources are added to the energy system model
# 4. Commodity conversion components are added to the energy system model
# 5. Commodity storages are added to the energy system model
# 6. Commodity transmission components are added to the energy system model
# 7. Commodity sinks are added to the energy system model
# 8. The energy system model is optimized
# 9. Selected optimization results are presented
#

# 1. Import required packages and set input data path

import fine as fn
import pandas as pd
import numpy as np
import pytest

from fine.utils import ImplementedSolvers


def test_miniSystem():
    locations = {"loc1", "loc2"}
    numberOfTimeSteps = 365
    hoursPerTimeStep = 24
    commodities = {"electricity"}
    commodityUnitDict = {"electricity": r"GW$_{el}$"}

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    costTS = pd.DataFrame(
        [
            [(j % 5) * (i + 1) for i in range(len(locations))]
            for j in range(numberOfTimeSteps)
        ],
        columns=["loc1", "loc2"],
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity purchase",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=costTS,
        )
    )

    demandTS = pd.DataFrame(
        [[i + 1 for i in range(len(locations))] for j in range(numberOfTimeSteps)],
        columns=["loc1", "loc2"],
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            operationRateFix=demandTS,
            hasCapacityVariable=False,
            commodityRevenue=costTS,
        )
    )

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    summary = esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)
    np.testing.assert_almost_equal(
        summary.loc[("Electricity demand", "TAC", "[Euro/a]"), "loc1"], -730
    )
    np.testing.assert_almost_equal(
        summary.loc[("Electricity demand", "TAC", "[Euro/a]"), "loc2"], -2920
    )
    np.testing.assert_almost_equal(
        summary.loc[("Electricity purchase", "TAC", "[Euro/a]"), "loc1"], 730
    )
    np.testing.assert_almost_equal(
        summary.loc[("Electricity purchase", "TAC", "[Euro/a]"), "loc2"], 2920
    )

    esM.aggregateTemporally(
        numberOfTypicalPeriods=5,
        numberOfTimeStepsPerPeriod=1,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    esM.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    summary = esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)
    np.testing.assert_almost_equal(
        summary.loc[("Electricity demand", "TAC", "[Euro/a]"), "loc1"], -730
    )
    np.testing.assert_almost_equal(
        summary.loc[("Electricity demand", "TAC", "[Euro/a]"), "loc2"], -2920
    )
    np.testing.assert_almost_equal(
        summary.loc[("Electricity purchase", "TAC", "[Euro/a]"), "loc1"], 730
    )
    np.testing.assert_almost_equal(
        summary.loc[("Electricity purchase", "TAC", "[Euro/a]"), "loc2"], 2920
    )


@pytest.mark.parametrize("component", [fn.Source, fn.Sink])
@pytest.mark.parametrize(
    "deprecatedName, name",
    [
        ("commodityCostTimeSeries", "commodityCost"),
        ("commodityRevenueTimeSeries", "commodityRevenue"),
    ],
)
def test_deprecated_commodity_time_series_keywords(component, deprecatedName, name):
    esM = fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity"},
        numberOfTimeSteps=2,
        commodityUnitsDict={"electricity": "GW"},
        hoursPerTimeStep=4380,
        verboseLogLevel=2,
    )
    costTS = pd.DataFrame({"loc1": [1.0, 2.0], "loc2": [3.0, 4.0]})

    with pytest.warns(DeprecationWarning, match=deprecatedName):
        comp = component(esM, "comp", "electricity", False, **{deprecatedName: costTS})
    pd.testing.assert_frame_equal(getattr(comp, name), costTS)

    with pytest.raises(TypeError, match=deprecatedName):
        component(
            esM, "comp", "electricity", False, **{deprecatedName: costTS, name: 1}
        )


@pytest.mark.parametrize("component", [fn.Source, fn.Sink])
def test_unknown_keyword_is_rejected(component):
    esM = fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW"},
        verboseLogLevel=2,
    )
    with pytest.raises(TypeError, match="comodityCost"):
        component(esM, "comp", "electricity", False, comodityCost=1)


@pytest.mark.parametrize("asDict", [False, True])
def test_commodity_cost_as_time_indexed_series(asDict):
    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity"},
        numberOfTimeSteps=2,
        commodityUnitsDict={"electricity": "GW"},
        hoursPerTimeStep=4380,
        verboseLogLevel=2,
    )
    costTS = pd.Series([1.0, 2.0])

    source = fn.Source(
        esM,
        "source",
        "electricity",
        False,
        commodityCost={0: costTS} if asDict else costTS,
    )

    assert list(source.fullCommodityCost[0]["loc1"]) == [1.0, 2.0]


def test_commodity_cost_series_with_unknown_index_is_rejected():
    esM = fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW"},
        verboseLogLevel=2,
    )
    with pytest.raises(ValueError):
        fn.Source(
            esM, "source", "electricity", False, commodityCost=pd.Series({"x": 1.0})
        )


def test_default_commodity_cost_builds_no_time_series():
    esM = fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW"},
        verboseLogLevel=2,
    )
    source = fn.Source(esM, "source", "electricity", False)

    assert source.fullCommodityCost[0] is None
    assert source.fullCommodityRevenue[0] is None
