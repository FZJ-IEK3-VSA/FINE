

import numpy as np
import pandas as pd
import pytest

from fine.energySystemModel import EnergySystemModel
from tsam import ExtremeConfig


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _sampleTimeSeries():
    """Ten days of hourly data for two components, deterministic via seed."""
    generator = np.random.default_rng(42)
    index = pd.date_range("2050-01-01", periods=240, freq="h")
    return pd.DataFrame(
        {"load": generator.random(240), "wind": generator.random(240)},
        index=index,
    )


def _bareEnergySystemModel():
    """An EnergySystemModel instance without __init__.

    _buildTsamAggregation only calls self._translate_legacy_kwargs and never
    reads any attribute set in __init__, so an uninitialized instance is a
    valid, dependency-free way to test the translation.
    """
    return EnergySystemModel.__new__(EnergySystemModel)


# _translate_legacy_kwargs does not use `self`; call it unbound with a dummy.
_translate = EnergySystemModel._translate_legacy_kwargs




def test_no_kwargs_returns_no_extremes_and_empty_passthrough():
    extremeConfig, passthrough = _translate(None, {})
    assert extremeConfig is None
    assert passthrough == {}


def test_unknown_kwargs_are_passed_through_untouched():
    extremeConfig, passthrough = _translate(None, {"n_jobs": -1})
    assert extremeConfig is None
    assert passthrough == {"n_jobs": -1}


def test_directly_supplied_extremeconfig_takes_precedence():
    supplied = ExtremeConfig(max_value=["load"], method="append")
    extremeConfig, passthrough = _translate(
        None, {"extremes": supplied, "n_jobs": 2}
    )
    # the user's own ExtremeConfig is used verbatim, nothing is translated
    assert extremeConfig is supplied
    assert passthrough == {"n_jobs": 2}


def test_legacy_extreme_args_with_method_build_extremeconfig():
    with pytest.warns(DeprecationWarning):
        extremeConfig, passthrough = _translate(
            None,
            {
                "addPeakMax": ["load"],
                "addPeakMin": ["wind"],
                "extremePeriodMethod": "new_cluster_center",
            },
        )
    assert isinstance(extremeConfig, ExtremeConfig)
    # value renames from the tsam migration guide
    assert extremeConfig.max_value == ["load"]
    assert extremeConfig.min_value == ["wind"]
    assert extremeConfig.method == "new_cluster"  # renamed from new_cluster_center
    # translated keys are consumed, not left in the passthrough
    assert passthrough == {}


def test_peak_columns_without_method_are_ignored_v2_parity():
    # tsam v2 parity: peaks are only added when extremePeriodMethod is set.
    with pytest.warns(UserWarning):
        extremeConfig, passthrough = _translate(None, {"addPeakMax": ["load"]})
    assert extremeConfig is None


def test_translation_does_not_mutate_the_input_dict():
    original = {"addPeakMax": ["load"], "extremePeriodMethod": "append"}
    _translate(None, original)
    # the method copies kwargs first; the caller's dict must be untouched
    assert original == {"addPeakMax": ["load"], "extremePeriodMethod": "append"}




def test_build_without_segmentation_returns_period_timestep_shape():
    esM = _bareEnergySystemModel()
    aggregation = esM._buildTsamAggregation(
        timeSeriesData=_sampleTimeSeries(),
        weightDict={"load": 1.0, "wind": 1.0},
        numberOfTypicalPeriods=4,
        hoursPerPeriod=24,
        segmentation=False,
        numberOfSegmentsPerPeriod=6,
        clusterMethod="hierarchical",
        representationMethod="durationRepresentation",
        sortValues=False,
        rescaleClusterPeriods=False,
        solver="highs",
        kwargs={},
    )
    representatives = aggregation.cluster_representatives
    # 4 typical periods x 24 time steps, 2 components
    assert representatives.shape == (96, 2)
    assert "Segment Duration" not in list(representatives.index.names)


def test_build_with_segmentation_exposes_segment_duration_level():
    esM = _bareEnergySystemModel()
    aggregation = esM._buildTsamAggregation(
        timeSeriesData=_sampleTimeSeries(),
        weightDict={"load": 1.0, "wind": 1.0},
        numberOfTypicalPeriods=4,
        hoursPerPeriod=24,
        segmentation=True,
        numberOfSegmentsPerPeriod=6,
        clusterMethod="hierarchical",
        representationMethod="durationRepresentation",
        sortValues=False,
        rescaleClusterPeriods=False,
        solver="highs",
        kwargs={},
    )
    representatives = aggregation.cluster_representatives
    # 4 typical periods x 6 segments
    assert representatives.shape[0] == 24
    # the segment-duration information FINE relies on must be present
    assert "Segment Duration" in list(representatives.index.names)


def test_build_translates_fine_cluster_method_alias():
    # A FINE-style name ("k_means") must be translated to the tsam name
    # ("kmeans") and run without raising.
    esM = _bareEnergySystemModel()
    aggregation = esM._buildTsamAggregation(
        timeSeriesData=_sampleTimeSeries(),
        weightDict={"load": 1.0, "wind": 1.0},
        numberOfTypicalPeriods=4,
        hoursPerPeriod=24,
        segmentation=False,
        numberOfSegmentsPerPeriod=6,
        clusterMethod="k_means",
        representationMethod="meanRepresentation",
        sortValues=False,
        rescaleClusterPeriods=False,
        solver="highs",
        kwargs={},
    )
    assert aggregation.cluster_assignments is not None


def test_build_passes_extreme_kwargs_into_aggregation():
    # Legacy extreme-period kwargs should be honoured end-to-end: with
    # append + a peak column the reconstruction must still cover the input.
    esM = _bareEnergySystemModel()
    with pytest.warns(DeprecationWarning):
        aggregation = esM._buildTsamAggregation(
            timeSeriesData=_sampleTimeSeries(),
            weightDict={"load": 1.0, "wind": 1.0},
            numberOfTypicalPeriods=4,
            hoursPerPeriod=24,
            segmentation=False,
            numberOfSegmentsPerPeriod=6,
            clusterMethod="hierarchical",
            representationMethod="durationRepresentation",
            sortValues=False,
            rescaleClusterPeriods=False,
            solver="highs",
            kwargs={"addPeakMax": ["load"], "extremePeriodMethod": "append"},
        )
    # append adds an extra typical period beyond the requested 4
    assert len(aggregation.cluster_representatives.index.get_level_values(0).unique()) >= 4