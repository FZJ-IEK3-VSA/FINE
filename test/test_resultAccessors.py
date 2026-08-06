"""Tests for the result-dict accessors on the modeling classes.

``ComponentModel.getResultSummaryDict`` and ``ComponentModel.getResultOptimalValues`` read
the raw results dict (``_rawResults`` / ``_rawResults1dim``) that ``setOptimalValues``
fills, and return per-component, ``to_xarray``-ready entries. They are the accessors the
xarray/netCDF export is built on; the export itself is refactored separately, so these
tests exercise the accessors directly and deliberately do not import ``xarrayIO``.
"""

import numpy as np
import pandas as pd
import pytest

import fine.subclasses.conversionPartLoad as partload_module
import fine.subclasses.lopf as lopf_module


def _optimize(esM):
    esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    return esM, esM.investmentPeriodNames[0]


def test_requireRawResults_raises_before_the_model_is_optimized(minimal_test_esM):
    """The raw results exist only after optimize(); asking earlier must explain that.

    An EnergySystemModel read back from netCDF is in exactly this state (the reader
    restores the summary and the optimum attributes but not the raw results), so the
    error has to name the remedy instead of surfacing as an AttributeError.
    """
    model = minimal_test_esM.componentModelingDict["ConversionModel"]
    assert model._rawResults == {}

    with pytest.raises(
        RuntimeError, match="optimize the model or load an optimized model"
    ):
        model.getResultOptimalValues(0)
    with pytest.raises(
        RuntimeError, match="optimize the model or load an optimized model"
    ):
        model.getResultSummaryDict(minimal_test_esM, 0)


def test_requireRawResults_raises_for_an_unknown_investment_period(minimal_test_esM):
    """A wrong investment period must be reported as such, listing the available ones."""
    esM, ip = _optimize(minimal_test_esM)
    model = esM.componentModelingDict["ConversionModel"]

    with pytest.raises(KeyError, match="notAnInvestmentPeriod"):
        model.getResultOptimalValues("notAnInvestmentPeriod")


def test_getResultOptimalValues_returns_the_solved_optimum_frames(minimal_test_esM):
    """The accessor must hand back the very values stored in the optimum attributes.

    ``extractRawResults`` puts the same frame object into ``_rawResults`` and into the
    ``_*VariablesOptimum`` attribute, so the two must agree value for value.
    """
    esM, ip = _optimize(minimal_test_esM)
    model = esM.componentModelingDict["ConversionModel"]

    values = model.getResultOptimalValues(ip)["Electrolyzers"]

    capacity, unit = values["capacityVariablesOptimum"]
    assert unit is None, "optimum values carry no unit"
    expected = model._capacityVariablesOptimum[ip].loc["Electrolyzers"]
    np.testing.assert_allclose(
        capacity.sort_index().values, expected.sort_index().values
    )

    operation, _ = values["operationVariablesOptimum"]
    assert operation.index.names == ["time", "location"]
    expected_operation = model._operationVariablesOptimum[ip].loc["Electrolyzers"]
    assert operation.sum() == pytest.approx(expected_operation.values.sum())


def test_getResultOptimalValues_shapes_a_two_dimensional_component(minimal_test_esM):
    """2-dim components are split into (locationIn, locationOut) instead of connections."""
    esM, ip = _optimize(minimal_test_esM)
    model = esM.componentModelingDict["TransmissionModel"]

    values = model.getResultOptimalValues(ip)["Pipelines"]

    capacity, _ = values["capacityVariablesOptimum"]
    assert capacity.index.names == ["locationIn", "locationOut"]
    operation, _ = values["operationVariablesOptimum"]
    assert operation.index.names == ["time", "locationIn", "locationOut"]
    # every connection resolves to two distinct existing locations
    for locationIn, locationOut in capacity.index:
        assert locationIn in esM.locations and locationOut in esM.locations
        assert locationIn != locationOut


def test_exportOptimumVarMap_covers_the_subclass_variables(minimal_test_esM):
    """Each modeling class must export its own variables, not only the design ones."""
    esM, ip = _optimize(minimal_test_esM)

    base = {
        "capacityVariablesOptimum",
        "isBuiltVariablesOptimum",
        "operationVariablesOptimum",
        "commissioningVariablesOptimum",
        "decommissioningVariablesOptimum",
    }
    conversion = {
        name
        for _, name, _, _ in esM.componentModelingDict[
            "ConversionModel"
        ]._exportOptimumVarMap()
    }
    assert conversion == base

    storage = {
        name
        for _, name, _, _ in esM.componentModelingDict[
            "StorageModel"
        ]._exportOptimumVarMap()
    }
    # a storage has no plain operation but charge, discharge and state of charge
    assert {
        "chargeOperationVariablesOptimum",
        "dischargeOperationVariablesOptimum",
        "stateOfChargeOperationVariablesOptimum",
    } <= storage
    assert "operationVariablesOptimum" not in storage


def test_getResultSummaryDict_agrees_with_the_optimization_summary(minimal_test_esM):
    """The summary dict and the optimization summary must be views of the same values.

    This is the invariant the refactor rests on: the optimization summary is rendered
    from the raw results dict, so every value (and its unit) that the accessor reports
    has to match the corresponding row of ``getOptimizationSummary``.
    """
    esM, ip = _optimize(minimal_test_esM)

    for name in (
        "SourceSinkModel",
        "ConversionModel",
        "StorageModel",
        "TransmissionModel",
    ):
        model = esM.componentModelingDict[name]
        summary = esM.getOptimizationSummary(name, ip=ip)
        resultDict = model.getResultSummaryDict(esM, ip)

        checked = 0
        for component, properties in resultDict.items():
            for prop, (series, unit) in properties.items():
                if (component, prop, unit) not in summary.index:
                    # rows that are entirely absent are not written to the summary
                    assert series.isna().all(), (
                        f"{name}/{component}/{prop} has values but no summary row"
                    )
                    continue
                expected = summary.loc[(component, prop, unit)]
                if model.dimension == "2dim":
                    expected = expected.stack()
                for location in series.index:
                    got = series.loc[location]
                    want = expected.loc[location]
                    if pd.isna(got) or pd.isna(want):
                        assert pd.isna(got) and pd.isna(want), (
                            f"{name}/{component}/{prop}@{location} NaN mismatch"
                        )
                    else:
                        assert float(got) == pytest.approx(float(want)), (
                            f"{name}/{component}/{prop}@{location}"
                        )
                    checked += 1

        # Check the opposite direction too: a nonempty summary row must not disappear from
        # the accessor merely because the accessor and summary share helper mappings.
        for index, row in summary.iterrows():
            if row.isna().all():
                continue
            component, prop, unit = index[:3]
            assert prop in resultDict[component], (
                f"{name}/{component}/{prop} exists in the summary but not the accessor"
            )
            assert resultDict[component][prop][1] == unit
        assert checked > 0, f"nothing compared for {name}"


def test_getResultSummaryDict_fills_absent_1dim_rows_with_nan(minimal_test_esM):
    """1-dim components report the full property set, NaN where a property does not apply.

    'Electricity market' is a source without a capacity variable, so it has no capacity,
    but the property is still reported (as NaN over every location) - the export relies
    on the fixed property set being present for every 1-dim component.
    """
    esM, ip = _optimize(minimal_test_esM)
    model = esM.componentModelingDict["SourceSinkModel"]

    resultDict = model.getResultSummaryDict(esM, ip)

    capacity, _ = resultDict["Electricity market"]["capacity"]
    assert sorted(capacity.index) == sorted(esM.locations)
    assert capacity.isna().all()

    # a component that does have a capacity reports real numbers
    conversion = esM.componentModelingDict["ConversionModel"]
    electrolyzer, _ = conversion.getResultSummaryDict(esM, ip)["Electrolyzers"][
        "capacity"
    ]
    assert electrolyzer.notna().any()


def test_getResultSummaryDict_drops_absent_2dim_rows(minimal_test_esM):
    """2-dim components omit properties that are all NaN, rather than reporting them.

    The asymmetry with the 1-dim case is deliberate and is what the export expects:
    a transmission has no isBuilt / capexIfBuilt / opexIfBuilt values here, so those
    properties must not appear at all.
    """
    esM, ip = _optimize(minimal_test_esM)
    model = esM.componentModelingDict["TransmissionModel"]

    properties = model.getResultSummaryDict(esM, ip)["Pipelines"]

    assert "capacity" in properties
    for absent in ("isBuilt", "capexIfBuilt", "opexIfBuilt"):
        assert absent not in properties, (
            f"{absent} should be dropped for a 2-dim component"
        )


def test_registerExtraSummaryRows_are_reported_by_the_summary_dict(minimal_test_esM):
    """Rows contributed after setOptimalValues (expansion modules) must reach the accessor.

    The piecewise linear cost function publishes its etl/eos rows this way, so they have
    to appear for the components they apply to - and only for those.
    """
    esM, ip = _optimize(minimal_test_esM)
    model = esM.componentModelingDict["ConversionModel"]

    frame = pd.DataFrame(
        [[1.5, 2.5]], index=["Electrolyzers"], columns=sorted(esM.locations)
    )
    model.registerExtraSummaryRows(ip, [("knowledgeStock_ETL", frame, "[kW]")])

    properties = model.getResultSummaryDict(esM, ip)["Electrolyzers"]
    assert "knowledgeStock_ETL" in properties
    series, unit = properties["knowledgeStock_ETL"]
    assert unit == "[kW]"
    np.testing.assert_allclose(
        series.sort_index().values, frame.loc["Electrolyzers"].sort_index().values
    )


def test_registerExtraSummaryRows_is_refused_for_2dim_models(minimal_test_esM):
    """A 2-dim model is indexed by connection, so location-indexed rows cannot be mapped."""
    esM, ip = _optimize(minimal_test_esM)
    model = esM.componentModelingDict["TransmissionModel"]

    frame = pd.DataFrame(
        [[1.0, 2.0]], index=["Pipelines"], columns=sorted(esM.locations)
    )
    with pytest.warns(UserWarning, match="cannot be added to the result export"):
        model.registerExtraSummaryRows(ip, [("knowledgeStock_ETL", frame, "[kW]")])

    assert "knowledgeStock_ETL" not in model.getResultSummaryDict(esM, ip)["Pipelines"]


def test_public_optimum_names_are_set_for_every_modeling_class(minimal_test_esM):
    """optimize() must publish the internal ``_*`` attributes under their public names.

    The renaming is driven from EnergySystemModel.optimize, so it applies to every
    modeling class - including one that overrides setOptimalValues without knowing about
    it. A single-year model is unwrapped to the one dataframe it holds.
    """
    esM, ip = _optimize(minimal_test_esM)

    for name in esM.componentModelingDict:
        model = esM.componentModelingDict[name]
        assert isinstance(model.optSummary, pd.DataFrame), name
        # the public name exists for every internal one the class actually holds; the
        # value is None where the variable does not apply (a source without a capacity
        # variable has no capacity optimum)
        for internal in ("_capacityVariablesOptimum", "_commissioningVariablesOptimum"):
            public = internal[1:]
            assert hasattr(model, public), f"{name}.{public}"
            assert getattr(model, public) is None or isinstance(
                getattr(model, public), pd.DataFrame
            ), f"{name}.{public}"

    # the storage state of charge is published too - its entry used to carry a typo
    # ("VSariables"), so this public attribute was silently never created
    storage = esM.componentModelingDict["StorageModel"]
    assert hasattr(storage, "stateOfChargeOperationVariablesOptimum")
    assert isinstance(storage.stateOfChargeOperationVariablesOptimum, pd.DataFrame)


def test_public_optimum_names_survive_a_custom_setOptimalValues(minimal_test_esM):
    """A class overriding setOptimalValues must still get its public attributes.

    The renaming used to be done at the end of each modeling class' setOptimalValues, so
    an override that did not repeat the call silently lost the public names.
    """
    esM = minimal_test_esM
    model = esM.componentModelingDict["ConversionModel"]
    calls = []

    original = type(model).setOptimalValues

    def overriding_setOptimalValues(self, esM_, pyM):
        calls.append(1)
        return original(self, esM_, pyM)

    type(model).setOptimalValues = overriding_setOptimalValues
    try:
        esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    finally:
        type(model).setOptimalValues = original

    assert calls, "the override was not exercised"
    assert isinstance(model.capacityVariablesOptimum, pd.DataFrame)
    assert isinstance(model.optSummary, pd.DataFrame)


def test_extra_summary_rows_are_cleared_by_a_new_optimization(minimal_test_esM):
    """Rows from a previous run must not survive into the next one."""
    esM, ip = _optimize(minimal_test_esM)
    model = esM.componentModelingDict["ConversionModel"]

    frame = pd.DataFrame(
        [[1.5, 2.5]], index=["Electrolyzers"], columns=sorted(esM.locations)
    )
    model.registerExtraSummaryRows(ip, [("knowledgeStock_ETL", frame, "[kW]")])
    assert "knowledgeStock_ETL" in model.getResultSummaryDict(esM, ip)["Electrolyzers"]

    esM.optimize(timeSeriesAggregation=False, solver="gurobi")

    assert (
        "knowledgeStock_ETL" not in model.getResultSummaryDict(esM, ip)["Electrolyzers"]
    )


def test_getOptimalValues_covers_the_subclass_variables_of_every_class():
    """Every modeling class must report its own optimum variables, and all of them.

    A subclass that reimplements the variable table instead of extending the base one
    tends to fall behind it - the commissioning/decommissioning entries were missing from
    the LOPF and part-load tables for exactly that reason.
    """
    design = {
        "capacityVariablesOptimum",
        "isBuiltVariablesOptimum",
        "commissioningVariablesOptimum",
        "decommissioningVariablesOptimum",
    }

    lopf = lopf_module.LOPFModel()
    partload = partload_module.ConversionPartLoadModel()

    # patch in empty optimum dicts so getOptimalValues can be called without a solve
    for model, extra in (
        (lopf, ["_phaseAngleVariablesOptimum"]),
        (
            partload,
            [
                "_discretizationPointVariablesOptimum",
                "_discretizationSegmentConVariablesOptimum",
                "_discretizationSegmentBinVariablesOptimum",
            ],
        ),
    ):
        for attr in [
            "_capacityVariablesOptimum",
            "_isBuiltVariablesOptimum",
            "_operationVariablesOptimum",
            "_commissioningVariablesOptimum",
            "_decommissioningVariablesOptimum",
            *extra,
        ]:
            setattr(model, attr, {0: pd.DataFrame()})

    assert design <= set(lopf.getOptimalValues("all", ip=0))
    assert "phaseAngleVariablesOptimum" in lopf.getOptimalValues("all", ip=0)
    assert design <= set(partload.getOptimalValues("all", ip=0))
    assert {
        "discretizationPointVariablesOptimum",
        "discretizationSegmentConVariablesOptimum",
        "discretizationSegmentBinVariablesOptimum",
    } <= set(partload.getOptimalValues("all", ip=0))

    # an unrecognised name returns every variable, as documented
    assert "phaseAngleVariablesOptimum" in lopf.getOptimalValues("nonsense", ip=0)
    assert "discretizationPointVariablesOptimum" in partload.getOptimalValues(
        "nonsense", ip=0
    )
    # a single named variable still returns just that one entry
    assert "values" in lopf.getOptimalValues("capacityVariablesOptimum", ip=0)
    assert "values" in partload.getOptimalValues("capacityVariablesOptimum", ip=0)
    assert "values" in lopf.getOptimalValues("phaseAngleVariablesOptimum", ip=0)


def test_named_base_optimum_does_not_require_subclass_results():
    """A base result remains independently accessible when subclass results are absent.

    This occurs, for example, on models restored by the current netCDF reader, which
    reconstructs the standard design/operation optima but not LOPF phase angles or part-load
    discretization variables.
    """
    for model in (lopf_module.LOPFModel(), partload_module.ConversionPartLoadModel()):
        model._capacityVariablesOptimum = {0: pd.DataFrame()}

        result = model.getOptimalValues("capacityVariablesOptimum", ip=0)

        assert "values" in result
        assert isinstance(result["values"], pd.DataFrame)
