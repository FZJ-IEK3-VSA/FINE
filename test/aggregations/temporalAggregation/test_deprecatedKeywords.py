"""Tests for the deprecated ETHOS.TSAM 3.x keywords.

Delete this module together with
:mod:`fine.aggregations.temporalAggregation.deprecatedKeywords` when the
deprecation period ends.
"""

import inspect
import warnings
from copy import deepcopy

import pytest
from tsam import ClusterConfig, Distribution, MinMaxMean, SegmentConfig

import fine as fn
from fine.aggregations.temporalAggregation import deprecatedKeywords as deprecated


def convert(**deprecatedKwargs):
    """Convert deprecated keywords assuming hourly time steps."""
    return deprecated.convertKeywords(deprecatedKwargs, hoursPerTimeStep=1)


# ==================================== Keyword conversion ====================================#


def test_deprecated_keywords_are_converted():
    arguments = convert(
        noTypicalPeriods=4,
        hoursPerPeriod=24,
        rescaleClusterPeriods=True,
        rescaleExcludeColumns=["wind"],
        roundOutput=4,
        numericalTolerance=1e-10,
        clusterMethod="k_means",
        representationMethod="durationRepresentation",
        sortValues=False,
        sameMean=True,
        evalSumPeriods=True,
        solver="gurobi",
    )

    assert arguments["n_clusters"] == 4
    assert arguments["period_duration"] == 24
    assert arguments["preserve_column_means"] is True
    assert arguments["rescale_exclude_columns"] == ["wind"]
    assert arguments["round_decimals"] == 4
    assert arguments["numerical_tolerance"] == 1e-10

    cluster = arguments["cluster"]
    assert cluster.method == "kmeans"
    assert cluster.representation == "distribution"
    assert cluster.use_duration_curves is False
    assert cluster.scale_by_column_means is True
    assert cluster.include_period_sums is True
    assert cluster.solver == "gurobi"


def test_numberOfTimeStepsPerPeriod_becomes_a_period_length():
    arguments = deprecated.convertKeywords(
        {"numberOfTimeStepsPerPeriod": 24}, hoursPerTimeStep=2190
    )
    assert arguments["period_duration"] == 24 * 2190


def test_sortValues_becomes_use_duration_curves():
    assert convert(sortValues=True)["cluster"].use_duration_curves is True


@pytest.mark.parametrize(
    ("deprecatedName", "newName"), sorted(deprecated.CLUSTER_METHOD_MAP.items())
)
def test_cluster_methods_are_converted(deprecatedName, newName):
    assert convert(clusterMethod=deprecatedName)["cluster"].method == newName


@pytest.mark.parametrize(
    ("deprecatedName", "newName"), sorted(deprecated.REPRESENTATION_METHOD_MAP.items())
)
def test_representations_are_converted(deprecatedName, newName):
    assert convert(representationMethod=deprecatedName)["cluster"].representation == (
        newName
    )


def test_new_names_and_typed_representations_pass_through():
    arguments = convert(
        clusterMethod="contiguous",
        representationMethod=MinMaxMean(max_columns=["solar"]),
    )
    assert arguments["cluster"].method == "contiguous"
    assert arguments["cluster"].representation == MinMaxMean(max_columns=["solar"])


def test_distributionPeriodWise_becomes_a_distribution_object():
    arguments = convert(
        representationMethod="distributionAndMinMaxRepresentation",
        distributionPeriodWise=False,
    )
    assert arguments["cluster"].representation == Distribution(
        scope="global", preserve_minmax=True
    )


def test_concurrency_keywords_become_a_distribution_object():
    arguments = convert(
        representationMethod="durationRepresentation",
        representationReferenceAttribute="demand",
        representationConcurrencyMethod="reference",
    )
    assert arguments["cluster"].representation == Distribution(
        reference_attribute="demand", concurrency="reference"
    )


def test_representationDict_becomes_a_minMaxMean_object():
    arguments = convert(
        representationMethod="minmaxmeanRepresentation",
        representationDict={"solar": "max", "demand": "min", "wind": "mean"},
    )
    assert arguments["cluster"].representation == MinMaxMean(
        max_columns=["solar"], min_columns=["demand"]
    )


def test_segments_inherit_the_cluster_representation():
    """ETHOS.TSAM 3.x let the segments inherit representationMethod, 4.x does not."""
    arguments = convert(
        representationMethod="durationRepresentation", segmentation=True, noSegments=6
    )
    assert arguments["segments"] == SegmentConfig(
        n_segments=6, representation="distribution"
    )


def test_concurrency_is_not_inherited_by_the_segments():
    """A segment carries a single value per column, so tsam rejects it there."""
    arguments = convert(
        representationMethod="durationRepresentation",
        representationReferenceAttribute="demand",
        segmentation=True,
        noSegments=3,
    )
    assert arguments["segments"].representation == Distribution(scope="local")


def test_explicit_segment_representation_wins_over_inheritance():
    arguments = convert(
        representationMethod="durationRepresentation",
        segmentation=True,
        noSegments=6,
        segmentRepresentationMethod="meanRepresentation",
    )
    assert arguments["segments"].representation == "mean"


def test_segmentation_off_switches_the_segments_off():
    assert convert(segmentation=False, noSegments=6)["segments"] is None


def test_segmentation_on_without_a_count_keeps_the_default():
    """The number of segments is the default of aggregateTemporally, not of the shim."""
    assert convert(segmentation=True) == {}


def test_segment_representation_without_a_count_is_rejected():
    with pytest.raises(ValueError, match="numberOfSegmentsPerPeriod"):
        convert(segmentRepresentationMethod="meanRepresentation")


@pytest.mark.parametrize(
    ("deprecatedName", "newName"),
    [
        ("append", "append"),
        ("replace_cluster_center", "replace"),
        ("new_cluster_center", "new_cluster"),
    ],
)
def test_extreme_period_methods_are_converted(deprecatedName, newName):
    extremes = convert(
        extremePeriodMethod=deprecatedName,
        addPeakMax=["demand"],
        addPeakMin=["solar"],
        addMeanMax=["wind"],
        addMeanMin=["solar"],
    )["extremes"]
    assert extremes.method == newName
    assert extremes.max_value == ["demand"]
    assert extremes.min_value == ["solar"]
    assert extremes.max_period == ["wind"]
    assert extremes.min_period == ["solar"]


def test_extremePeriodMethod_none_disables_extreme_periods():
    assert convert(extremePeriodMethod="None", addPeakMax=["demand"]) == {}


def test_nothing_given_converts_to_nothing():
    assert convert() == {}


@pytest.mark.parametrize(
    "invalidKwargs",
    [
        {"representationMethod": "notARepresentation"},
        {"extremePeriodMethod": "notAMethod", "addPeakMax": ["demand"]},
        {
            "representationMethod": "minmaxmeanRepresentation",
            "representationDict": {"demand": "notAMode"},
        },
    ],
)
def test_invalid_values_are_rejected(invalidKwargs):
    with pytest.raises(ValueError):
        convert(**invalidKwargs)


def test_representationDict_without_its_representation_is_ignored():
    """As in ETHOS.TSAM 3.x, where only minmaxmeanRepresentation read the dict."""
    arguments = convert(representationDict={"demand": "max"})
    assert arguments["cluster"].representation == "distribution"


def test_clustering_transfer_keywords_are_rejected():
    with pytest.raises(TypeError, match="ClusteringResult.apply"):
        convert(predefClusterOrder=[0, 1])


# ==================================== The decorator ====================================#


def test_deprecated_keywords_warn(minimal_test_esM):
    with pytest.warns(DeprecationWarning, match="numberOfTypicalPeriods -> n_clusters"):
        minimal_test_esM.aggregateTemporally(
            numberOfTypicalPeriods=2, numberOfTimeStepsPerPeriod=2, segmentation=False
        )


def test_the_two_interfaces_cannot_be_mixed(minimal_test_esM):
    with pytest.raises(TypeError, match="cannot be combined"):
        minimal_test_esM.aggregateTemporally(n_clusters=2, numberOfTimeStepsPerPeriod=2)


def test_storeTSAinstance_is_not_part_of_either_interface(minimal_test_esM):
    """It is FINE's own argument, so it must not count as mixing."""
    with pytest.warns(DeprecationWarning):
        minimal_test_esM.aggregateTemporally(
            numberOfTypicalPeriods=2,
            numberOfTimeStepsPerPeriod=2,
            segmentation=False,
            storeTSAinstance=True,
        )
    assert minimal_test_esM.tsaInstance is not None


@pytest.mark.parametrize(
    "modelDerived", [{"weightDict": {"a": 1.0}}, {"resolution": 1}]
)
def test_model_derived_keywords_are_rejected(minimal_test_esM, modelDerived):
    with pytest.raises(TypeError, match="derived from the energy system model"):
        minimal_test_esM.aggregateTemporally(numberOfTypicalPeriods=2, **modelDerived)


def test_the_deprecated_interface_reaches_the_aggregation(minimal_test_esM):
    with pytest.warns(DeprecationWarning):
        minimal_test_esM.aggregateTemporally(
            numberOfTypicalPeriods=1,
            numberOfTimeStepsPerPeriod=4,
            segmentation=True,
            numberOfSegmentsPerPeriod=3,
            clusterMethod="hierarchical",
            representationMethod="meanRepresentation",
            sortValues=False,
            rescaleClusterPeriods=False,
            storeTSAinstance=True,
        )
    tsaInstance = minimal_test_esM.tsaInstance

    assert tsaInstance.clustering.cluster_config == ClusterConfig(
        method="hierarchical", representation="mean", use_duration_curves=False
    )
    assert tsaInstance.clustering.segment_config == SegmentConfig(
        n_segments=3, representation="mean"
    )
    assert tsaInstance.clustering.preserve_column_means is False


def test_both_interfaces_describe_the_same_aggregation(minimal_test_esM):
    esM_deprecated = deepcopy(minimal_test_esM)
    with pytest.warns(DeprecationWarning):
        esM_deprecated.aggregateTemporally(
            numberOfTypicalPeriods=1,
            numberOfTimeStepsPerPeriod=4,
            segmentation=True,
            numberOfSegmentsPerPeriod=3,
            clusterMethod="hierarchical",
            representationMethod="meanRepresentation",
            sortValues=False,
            rescaleClusterPeriods=False,
            storeTSAinstance=True,
        )

    esM_new = deepcopy(minimal_test_esM)
    esM_new.aggregateTemporally(
        n_clusters=1,
        period_duration=4 * minimal_test_esM.hoursPerTimeStep,
        cluster=ClusterConfig(method="hierarchical", representation="mean"),
        segments=SegmentConfig(n_segments=3, representation="mean"),
        preserve_column_means=False,
        storeTSAinstance=True,
    )

    assert esM_new.segmentsPerPeriod == esM_deprecated.segmentsPerPeriod
    assert esM_new.tsaInstance.cluster_representatives.equals(
        esM_deprecated.tsaInstance.cluster_representatives
    )
    for ip in esM_deprecated.investmentPeriods:
        assert esM_new.hoursPerSegment[ip].equals(esM_deprecated.hoursPerSegment[ip])


def test_the_deprecated_defaults_still_apply(minimal_test_esM):
    """A deprecated call that omits everything else must keep behaving as before."""
    esM_deprecated = deepcopy(minimal_test_esM)
    with pytest.warns(DeprecationWarning):
        esM_deprecated.aggregateTemporally(
            numberOfTypicalPeriods=2,
            numberOfTimeStepsPerPeriod=2,
            storeTSAinstance=True,
        )

    esM_new = deepcopy(minimal_test_esM)
    esM_new.aggregateTemporally(
        n_clusters=2,
        period_duration=2 * minimal_test_esM.hoursPerTimeStep,
        storeTSAinstance=True,
    )

    assert esM_new.segmentation == esM_deprecated.segmentation
    assert esM_new.segmentsPerPeriod == esM_deprecated.segmentsPerPeriod
    assert (
        esM_new.tsaInstance.clustering.cluster_config
        == esM_deprecated.tsaInstance.clustering.cluster_config
    )
    assert (
        esM_new.tsaInstance.clustering.segment_config
        == esM_deprecated.tsaInstance.clustering.segment_config
    )
    assert esM_new.tsaInstance.cluster_representatives.equals(
        esM_deprecated.tsaInstance.cluster_representatives
    )


def test_the_new_interface_bypasses_the_shim(minimal_test_esM):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        minimal_test_esM.aggregateTemporally(
            n_clusters=2,
            period_duration=2 * minimal_test_esM.hoursPerTimeStep,
            segments=None,
        )
    assert minimal_test_esM.segmentation is False


# ==================================== The clusterMethod decorator ====================================#


@deprecated.translateDeprecatedClusterMethod
def clusterMethodOf(esM, clusterMethod="hierarchical"):
    """Stand in for the expansion module functions, which need a solver to run."""
    return clusterMethod


@pytest.mark.parametrize(
    ("deprecatedName", "newName"), sorted(deprecated.RENAMED_CLUSTER_METHODS.items())
)
def test_deprecated_cluster_methods_are_translated(deprecatedName, newName):
    with pytest.warns(DeprecationWarning, match=f"Use {newName!r} instead"):
        assert clusterMethodOf(None, clusterMethod=deprecatedName) == newName


def test_a_positional_cluster_method_is_translated():
    with pytest.warns(DeprecationWarning):
        assert clusterMethodOf(None, "k_means") == "kmeans"


@pytest.mark.parametrize(
    "clusterMethod", ["averaging", "hierarchical", "kmeans", "contiguous"]
)
def test_current_cluster_methods_do_not_warn(clusterMethod):
    """Including the two names both interfaces share, which were never renamed."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert clusterMethodOf(None, clusterMethod=clusterMethod) == clusterMethod


def test_the_default_cluster_method_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert clusterMethodOf(None) == "hierarchical"


def test_an_unknown_cluster_method_is_left_to_tsam():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert clusterMethodOf(None, clusterMethod="notAMethod") == "notAMethod"


@pytest.mark.parametrize(
    "function", [fn.optimizeTSAmultiStage, fn.optimizeSimpleMyopic]
)
def test_the_expansion_modules_translate_their_clusterMethod(function):
    """They pass clusterMethod straight to ClusterConfig, so it has to be converted."""
    assert getattr(function, "__wrapped__", None) is not None
    assert "clusterMethod" in inspect.signature(function).parameters
