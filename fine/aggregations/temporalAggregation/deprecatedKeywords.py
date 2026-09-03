"""Conversion of the deprecated ETHOS.TSAM 3.x keywords to the 4.x interface.

ETHOS.TSAM 4.0 replaced the class based ``TimeSeriesAggregation`` interface with
the function :func:`tsam.aggregate` and the configuration objects
:class:`tsam.ClusterConfig`, :class:`tsam.SegmentConfig` and
:class:`tsam.ExtremeConfig`, renaming every parameter from camelCase to
snake_case on the way. FINE follows the new interface, and this module lets the
old keywords keep working for a deprecation period:

* :func:`convertKeywords` is the conversion itself, a pure function from the old
  keywords to the arguments of
  :meth:`~fine.energySystemModel.EnergySystemModel.aggregateTemporally`.
* :func:`translateDeprecatedKeywords` wraps that method so the conversion happens
  before it is entered. The two interfaces describe the same aggregation in
  incompatible terms, so combining them is refused; using the deprecated one
  warns.
* :func:`translateDeprecatedClusterMethod` does the same for the ``clusterMethod``
  argument of :func:`~fine.expansionModules.optimizeTSAmultiStage.optimizeTSAmultiStage`
  and :func:`~fine.expansionModules.transformationPath.optimizeSimpleMyopic`. Those
  keep their FINE parameter names, but the values they accept were renamed along
  with the rest of the ETHOS.TSAM interface.

.. note::
    This module exists only for the deprecation period. To end it, delete

    1. this file and ``test/aggregations/temporalAggregation/
       test_deprecatedKeywords.py``,
    2. the ``@translateDeprecatedKeywords`` decorator on
       :meth:`~fine.energySystemModel.EnergySystemModel.aggregateTemporally` and
       the ``@translateDeprecatedClusterMethod`` decorators on
       :func:`~fine.expansionModules.optimizeTSAmultiStage.optimizeTSAmultiStage`
       and :func:`~fine.expansionModules.transformationPath.optimizeSimpleMyopic`,
       together with the two imports marked as a deprecation shim,
    3. the ``.. deprecated:: 2.8.0`` note in the docstring of each of those three
       functions,
    4. the 'Deprecated ETHOS.TSAM 3.x keywords' section of
       ``docs/user_guide/python_package/aggregations.md``.

    Nothing else refers to it: the three functions are written against the
    ETHOS.TSAM 4.x interface only.
"""

import functools
import inspect
import warnings


from tsam import ClusterConfig, Distribution, ExtremeConfig, MinMaxMean, SegmentConfig

#: Deprecated ``clusterMethod`` values mapped to their ETHOS.TSAM 4.x names.
CLUSTER_METHOD_MAP = {
    "averaging": "averaging",
    "k_means": "kmeans",
    "k_medoids": "kmedoids",
    "k_maxoids": "kmaxoids",
    "hierarchical": "hierarchical",
    "adjacent_periods": "contiguous",
}

#: Deprecated ``representationMethod`` values mapped to their 4.x names.
#: ``durationRepresentation`` and ``distributionRepresentation`` always
#: described the same representation and both map to ``distribution``.
REPRESENTATION_METHOD_MAP = {
    "meanRepresentation": "mean",
    "medoidRepresentation": "medoid",
    "maxoidRepresentation": "maxoid",
    "distributionRepresentation": "distribution",
    "durationRepresentation": "distribution",
    "distributionAndMinMaxRepresentation": "distribution_minmax",
    "minmaxmeanRepresentation": "minmax_mean",
}

#: Deprecated ``extremePeriodMethod`` values mapped to their 4.x names.
#: ``'None'`` disables extreme period handling, i.e. no ``ExtremeConfig``.
EXTREME_PERIOD_METHOD_MAP = {
    "None": None,
    "append": "append",
    "replace_cluster_center": "replace",
    "new_cluster_center": "new_cluster",
}

_VALID_REPRESENTATIONS = frozenset(REPRESENTATION_METHOD_MAP.values())

# Deprecated keywords mapping one to one onto an argument of aggregateTemporally.
_DIRECT_RENAMES = {
    "numberOfTypicalPeriods": "n_clusters",
    "noTypicalPeriods": "n_clusters",
    "rescaleClusterPeriods": "preserve_column_means",
    "rescaleExcludeColumns": "rescale_exclude_columns",
    "roundOutput": "round_decimals",
    "numericalTolerance": "numerical_tolerance",
}

# Deprecated keywords describing the length of a period.
_PERIOD_KEYWORDS = ("numberOfTimeStepsPerPeriod", "hoursPerPeriod")

# Deprecated keywords mapping onto a ClusterConfig field.
_CLUSTER_RENAMES = {
    "clusterMethod": "method",
    "sameMean": "scale_by_column_means",
    "sortValues": "use_duration_curves",
    "evalSumPeriods": "include_period_sums",
    "solver": "solver",
}

# Deprecated keywords mapping onto an ExtremeConfig field.
_EXTREME_RENAMES = {
    "addPeakMax": "max_value",
    "addPeakMin": "min_value",
    "addMeanMax": "max_period",
    "addMeanMin": "min_period",
}

# Deprecated keywords shaping the representation without a direct counterpart:
# they are folded into a typed representation object instead.
_REPRESENTATION_KEYWORDS = (
    "representationMethod",
    "distributionPeriodWise",
    "representationDict",
    "representationReferenceAttribute",
    "representationConcurrencyMethod",
)

# Deprecated keywords describing the segmentation.
_SEGMENT_KEYWORDS = (
    "segmentation",
    "numberOfSegmentsPerPeriod",
    "noSegments",
    "segmentRepresentationMethod",
)

#: Deprecated keywords for transferring a clustering. ETHOS.TSAM 4.x handles this
#: through ``ClusteringResult.apply()``, so they cannot be converted.
UNSUPPORTED_KEYWORDS = (
    "predefClusterOrder",
    "predefClusterCenterIndices",
    "predefSegmentOrder",
    "predefSegmentDurations",
    "predefSegmentCenters",
)

#: Deprecated keywords describing data FINE derives from the model itself, which
#: never were and still are not settable from the outside.
MODEL_DERIVED_KEYWORDS = ("weightDict", "resolution")

#: The clustering defaults the deprecated interface carried in its signature.
#: They are filled in as soon as any clustering keyword is given, so that a
#: partial set of deprecated keywords keeps describing the aggregation it used
#: to. Where no clustering keyword is given at all, the default of
#: ``aggregateTemporally`` applies instead, which is the same clustering.
DEPRECATED_CLUSTER_DEFAULTS = {
    "clusterMethod": "hierarchical",
    "representationMethod": "durationRepresentation",
    "sortValues": False,
}

#: Every deprecated keyword, mapped to the argument that replaces it. Used to
#: tell the two interfaces apart and to point at the argument to migrate to.
DEPRECATED_KEYWORDS = {
    **_DIRECT_RENAMES,
    **dict.fromkeys(_PERIOD_KEYWORDS, "period_duration"),
    **dict.fromkeys(_CLUSTER_RENAMES, "cluster"),
    **dict.fromkeys(_REPRESENTATION_KEYWORDS, "cluster"),
    **dict.fromkeys(_SEGMENT_KEYWORDS, "segments"),
    **dict.fromkeys((*_EXTREME_RENAMES, "extremePeriodMethod"), "extremes"),
    **dict.fromkeys(UNSUPPORTED_KEYWORDS, "ClusteringResult.apply()"),
    "weightDict": "weights",
    "resolution": "temporal_resolution",
}


def _minMaxMeanFromDict(representationDict):
    """Build a :class:`tsam.MinMaxMean` from a deprecated ``representationDict``.

    :param representationDict: Mapping of column name to 'max', 'min' or
        'mean'. Columns mapped to 'mean' (and columns not listed at all) are
        represented by their mean.
    :type representationDict: dict

    :returns: The equivalent per column representation object.
    :rtype: tsam.MinMaxMean
    """
    maxColumns, minColumns = [], []
    for column, mode in representationDict.items():
        if mode == "max":
            maxColumns.append(column)
        elif mode == "min":
            minColumns.append(column)
        elif mode != "mean":
            raise ValueError(
                f"Invalid representationDict entry {column!r}: {mode!r}. "
                "Valid values are 'max', 'min' and 'mean'."
            )
    return MinMaxMean(max_columns=maxColumns, min_columns=minColumns)


def convertRepresentation(
    representationMethod=None,
    distributionPeriodWise=None,
    representationDict=None,
    representationReferenceAttribute=None,
    representationConcurrencyMethod=None,
):
    """Convert a deprecated representation specification into its 4.x equivalent.

    **Default arguments:**

    :param representationMethod: A deprecated name (e.g.
        'durationRepresentation'), a 4.x short name (e.g. 'distribution'), a
        typed representation object, or None to keep the default of the
        clustering method.
        |br| * the default value is None
    :type representationMethod: string, tsam.Distribution, tsam.MinMaxMean or None

    :param distributionPeriodWise: Deprecated switch between a per cluster
        (True) and an overall (False) distribution. Only meaningful for the
        distribution representations.
        |br| * the default value is None
    :type distributionPeriodWise: boolean or None

    :param representationDict: Deprecated per column mapping to 'max', 'min' or
        'mean'. Only meaningful for the min/max/mean representation.
        |br| * the default value is None
    :type representationDict: dict or None

    :param representationReferenceAttribute: Deprecated name of the column whose
        temporal ordering is applied to all columns. Only meaningful for the
        distribution representations.
        |br| * the default value is None
    :type representationReferenceAttribute: string or None

    :param representationConcurrencyMethod: Deprecated strategy used to derive
        the synthetic time axis ('independent', 'reference', 'medoid',
        'consensus' or 'assignment'). Only meaningful for the distribution
        representations.
        |br| * the default value is None
    :type representationConcurrencyMethod: string or None

    :returns: A representation accepted by :class:`tsam.ClusterConfig` and
        :class:`tsam.SegmentConfig`, or None to keep the ETHOS.TSAM default.
    :rtype: string, tsam.Distribution, tsam.MinMaxMean or None
    """
    if isinstance(representationMethod, (Distribution, MinMaxMean)):
        return representationMethod

    if representationMethod is None:
        # No representation requested: keep the method specific ETHOS.TSAM
        # default, unless a per column mapping pins the representation.
        return _minMaxMeanFromDict(representationDict) if representationDict else None

    name = REPRESENTATION_METHOD_MAP.get(representationMethod, representationMethod)
    if name not in _VALID_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation method {representationMethod!r}. Valid deprecated "
            f"names are {sorted(REPRESENTATION_METHOD_MAP)}, valid ETHOS.TSAM 4.x "
            f"names are {sorted(_VALID_REPRESENTATIONS)}."
        )

    if name == "minmax_mean" and representationDict:
        return _minMaxMeanFromDict(representationDict)

    if name in ("distribution", "distribution_minmax"):
        distributionKwargs = {}
        if distributionPeriodWise is not None:
            distributionKwargs["scope"] = (
                "local" if distributionPeriodWise else "global"
            )
        if representationReferenceAttribute:
            distributionKwargs["reference_attribute"] = representationReferenceAttribute
        if representationConcurrencyMethod:
            distributionKwargs["concurrency"] = representationConcurrencyMethod
        # Only build the object where a deprecated keyword actually asks for one;
        # the plain string keeps the ETHOS.TSAM defaults and reads better.
        if distributionKwargs:
            return Distribution(
                preserve_minmax=(name == "distribution_minmax"), **distributionKwargs
            )

    return name


def convertCluster(deprecatedKwargs):
    """Convert the deprecated clustering keywords into a :class:`tsam.ClusterConfig`.

    Consumes ``clusterMethod``, ``sortValues``, ``sameMean``, ``evalSumPeriods``,
    ``solver`` and the representation keywords from ``deprecatedKwargs``. As soon
    as one of them is given, the others are filled in from
    :data:`DEPRECATED_CLUSTER_DEFAULTS` rather than from the ETHOS.TSAM defaults,
    because that is what the deprecated signature did. A keyword given as None
    stays None and thereby keeps the ETHOS.TSAM default, as it did before.

    :param deprecatedKwargs: The deprecated keywords, consumed in place.
    :type deprecatedKwargs: dict

    :returns: The cluster configuration, or None if no clustering keyword was
        given at all.
    :rtype: tsam.ClusterConfig or None
    """
    clusterKeywords = set(_CLUSTER_RENAMES) | set(_REPRESENTATION_KEYWORDS)
    if not clusterKeywords & set(deprecatedKwargs):
        return None

    given = dict(DEPRECATED_CLUSTER_DEFAULTS)
    given.update(
        {
            name: deprecatedKwargs.pop(name)
            for name in list(deprecatedKwargs)
            if name in clusterKeywords
        }
    )

    configKwargs = {}
    for deprecatedName, newName in _CLUSTER_RENAMES.items():
        value = given.get(deprecatedName)
        if value is None:
            continue
        if deprecatedName == "clusterMethod":
            value = CLUSTER_METHOD_MAP.get(value, value)
        configKwargs[newName] = value

    representation = convertRepresentation(
        **{name: given.get(name) for name in _REPRESENTATION_KEYWORDS}
    )
    if representation is not None:
        configKwargs["representation"] = representation

    return ClusterConfig(**configKwargs)


def convertSegments(deprecatedKwargs, clusterRepresentation=None):
    """Convert the deprecated segmentation keywords into a ``segments`` argument.

    Consumes ``segmentation``, ``numberOfSegmentsPerPeriod``, ``noSegments`` and
    ``segmentRepresentationMethod`` from ``deprecatedKwargs``.

    .. note::
        ETHOS.TSAM 3.x let the segments inherit ``representationMethod`` when no
        ``segmentRepresentationMethod`` was given, whereas
        :class:`tsam.SegmentConfig` defaults to 'mean'. The inheritance is
        reproduced here so that the deprecated keywords keep producing the
        results they used to.

    :param deprecatedKwargs: The deprecated keywords, consumed in place.
    :type deprecatedKwargs: dict

    **Default arguments:**

    :param clusterRepresentation: The cluster representation the segments
        inherit when no segment representation was given.
        |br| * the default value is None
    :type clusterRepresentation: string, tsam.Distribution, tsam.MinMaxMean or None

    :returns: ``{'segments': ...}``, holding a :class:`tsam.SegmentConfig` or
        None to switch segmentation off. An empty dict where the keywords say
        nothing the default does not already describe.
    :rtype: dict
    """
    segmentation = deprecatedKwargs.pop("segmentation", None)
    numberOfSegments = deprecatedKwargs.pop("numberOfSegmentsPerPeriod", None)
    fallbackNumberOfSegments = deprecatedKwargs.pop("noSegments", None)
    numberOfSegments = (
        fallbackNumberOfSegments if numberOfSegments is None else numberOfSegments
    )
    representation = deprecatedKwargs.pop("segmentRepresentationMethod", None)

    if segmentation is False:
        return {"segments": None}
    if numberOfSegments is None:
        if representation is not None:
            raise ValueError(
                "segmentRepresentationMethod describes the segments but their number "
                "is missing. Add numberOfSegmentsPerPeriod."
            )
        # Only segmentation=True, which the default already describes.
        return {}

    # A concurrency ordering cannot be inherited: a segment collapses to a single
    # value per column, so ETHOS.TSAM rejects it there, and ETHOS.TSAM 3.x did
    # not pass it on to the segmentation either.
    inherited = (
        Distribution(scope=clusterRepresentation.scope)
        if isinstance(clusterRepresentation, Distribution)
        else clusterRepresentation
    )
    representation = (
        convertRepresentation(representation) if representation else inherited
    )
    if representation is None:
        return {"segments": SegmentConfig(n_segments=int(numberOfSegments))}
    return {
        "segments": SegmentConfig(
            n_segments=int(numberOfSegments), representation=representation
        )
    }


def convertExtremes(deprecatedKwargs):
    """Convert the deprecated extreme period keywords into a :class:`tsam.ExtremeConfig`.

    Consumes ``extremePeriodMethod``, ``addPeakMax``, ``addPeakMin``,
    ``addMeanMax`` and ``addMeanMin`` from ``deprecatedKwargs``.

    :param deprecatedKwargs: The deprecated keywords, consumed in place.
    :type deprecatedKwargs: dict

    :returns: The extreme period configuration, or None if no extreme periods
        were requested.
    :rtype: tsam.ExtremeConfig or None
    """
    method = deprecatedKwargs.pop("extremePeriodMethod", None)
    configKwargs = {}
    for deprecatedName, newName in _EXTREME_RENAMES.items():
        value = deprecatedKwargs.pop(deprecatedName, None)
        if value:
            configKwargs[newName] = list(value)

    if method is not None:
        if method not in EXTREME_PERIOD_METHOD_MAP:
            raise ValueError(
                f"Unknown extremePeriodMethod {method!r}. Valid deprecated names are "
                f"{sorted(EXTREME_PERIOD_METHOD_MAP)}."
            )
        convertedMethod = EXTREME_PERIOD_METHOD_MAP[method]
        if convertedMethod is None:
            # 'None' explicitly switches extreme period handling off.
            return None
        configKwargs["method"] = convertedMethod

    return ExtremeConfig(**configKwargs) if configKwargs else None


def convertKeywords(deprecatedKwargs, hoursPerTimeStep):
    """Convert deprecated keywords into arguments of ``aggregateTemporally``.

    :param deprecatedKwargs: The deprecated keywords the caller passed.
    :type deprecatedKwargs: dict

    :param hoursPerTimeStep: Length of one time step of the energy system model,
        needed to express ``numberOfTimeStepsPerPeriod`` as a period length.
    :type hoursPerTimeStep: strictly positive float

    :returns: The equivalent ETHOS.TSAM 4.x arguments.
    :rtype: dict

    :raises TypeError: If a keyword has no ETHOS.TSAM 4.x counterpart.
    """
    remaining = dict(deprecatedKwargs)

    unsupported = sorted(set(remaining) & set(UNSUPPORTED_KEYWORDS))
    if unsupported:
        raise TypeError(
            f"The keyword(s) {unsupported} are not supported by ETHOS.TSAM 4.x. "
            "Transfer a clustering to new data with ClusteringResult.apply(data) "
            "instead, reachable as esM.tsaInstance.clustering.apply(data)."
        )

    arguments = {}
    for deprecatedName, newName in _DIRECT_RENAMES.items():
        if deprecatedName in remaining:
            arguments[newName] = remaining.pop(deprecatedName)

    if "hoursPerPeriod" in remaining:
        arguments["period_duration"] = remaining.pop("hoursPerPeriod")
    elif "numberOfTimeStepsPerPeriod" in remaining:
        arguments["period_duration"] = (
            remaining.pop("numberOfTimeStepsPerPeriod") * hoursPerTimeStep
        )

    cluster = convertCluster(remaining)
    if cluster is not None:
        arguments["cluster"] = cluster

    arguments.update(
        convertSegments(
            remaining,
            clusterRepresentation=cluster.representation
            if cluster is not None
            else None,
        )
    )

    extremes = convertExtremes(remaining)
    if extremes is not None:
        arguments["extremes"] = extremes

    if remaining:
        raise TypeError(f"Unconverted deprecated keyword(s): {sorted(remaining)}.")
    return arguments


def translateDeprecatedKeywords(method):
    """Let ``aggregateTemporally`` accept the deprecated ETHOS.TSAM 3.x keywords.

    Converts them to the arguments of the wrapped method before it is entered,
    so that the method itself only ever sees the ETHOS.TSAM 4.x interface. Since
    the two interfaces describe the same aggregation in incompatible terms,
    combining them is refused rather than resolved.

    :param method: The ``aggregateTemporally`` method to wrap.

    :returns: The wrapped method.

    :raises TypeError: If both interfaces are used at once.
    :raises DeprecationWarning: Not raised but warned, whenever a deprecated
        keyword is used.
    """
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        passedThrough = bound.arguments.get("kwargs", {})
        deprecatedKwargs = {
            name: value
            for name, value in passedThrough.items()
            if name in DEPRECATED_KEYWORDS
        }
        if not deprecatedKwargs:
            return method(self, *args, **kwargs)

        modelDerived = sorted(set(deprecatedKwargs) & set(MODEL_DERIVED_KEYWORDS))
        if modelDerived:
            raise TypeError(
                f"The keyword(s) {modelDerived} are derived from the energy system "
                "model itself and cannot be set for the time series aggregation."
            )

        given = sorted(
            (set(bound.arguments) | set(passedThrough))
            - {"self", "kwargs", "storeTSAinstance"}
            - set(deprecatedKwargs)
        )
        replacements = ", ".join(
            f"{name} -> {DEPRECATED_KEYWORDS[name]}"
            for name in sorted(deprecatedKwargs)
        )
        if given:
            raise TypeError(
                f"The deprecated keyword(s) {sorted(deprecatedKwargs)} cannot be "
                f"combined with the ETHOS.TSAM 4.x argument(s) {given}. Describe the "
                f"aggregation through one interface only ({replacements})."
            )
        warnings.warn(
            f"The keyword(s) {sorted(deprecatedKwargs)} belong to the ETHOS.TSAM 3.x "
            "interface, which is deprecated and will be removed in a future FINE "
            f"release. Use the ETHOS.TSAM 4.x interface instead ({replacements}); see "
            "fine.aggregations.temporalAggregation.deprecatedKeywords for the full "
            "mapping.",
            DeprecationWarning,
            stacklevel=2,
        )

        for name in deprecatedKwargs:
            passedThrough.pop(name)
        bound.arguments["kwargs"] = {
            **passedThrough,
            **convertKeywords(deprecatedKwargs, self.hoursPerTimeStep),
        }
        return method(*bound.args, **bound.kwargs)

    return wrapper


#: Deprecated ``clusterMethod`` values whose ETHOS.TSAM 4.x name differs. The
#: values that were not renamed are left out, so that they do not warn.
RENAMED_CLUSTER_METHODS = {
    deprecatedName: newName
    for deprecatedName, newName in CLUSTER_METHOD_MAP.items()
    if deprecatedName != newName
}


def translateDeprecatedClusterMethod(function):
    """Let a ``clusterMethod`` argument accept the deprecated ETHOS.TSAM 3.x values.

    ``optimizeTSAmultiStage`` and ``optimizeSimpleMyopic`` pass their
    ``clusterMethod`` on to :class:`tsam.ClusterConfig`, which knows the 4.x
    names only. This converts a renamed value before the wrapped function is
    entered, so that models calling them with the old value keep running. Values
    that were not renamed, and unknown ones, are passed through untouched and
    left for ETHOS.TSAM to reject.

    :param function: The function taking a ``clusterMethod`` argument to wrap.

    :returns: The wrapped function.

    :raises DeprecationWarning: Not raised but warned, whenever a deprecated
        value is used.
    """
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        clusterMethod = bound.arguments.get("clusterMethod")
        if clusterMethod not in RENAMED_CLUSTER_METHODS:
            return function(*args, **kwargs)

        newName = RENAMED_CLUSTER_METHODS[clusterMethod]
        warnings.warn(
            f"The clusterMethod {clusterMethod!r} belongs to the ETHOS.TSAM 3.x "
            "interface, which is deprecated and will be removed in a future FINE "
            f"release. Use {newName!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        bound.arguments["clusterMethod"] = newName
        return function(*bound.args, **bound.kwargs)

    return wrapper
