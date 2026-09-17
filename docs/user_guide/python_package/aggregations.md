# Aggregations

FINE aggregation modules provide additional functionalities for reducing the complexity of spatially
highly-resolved models.

## Spatial Aggregation

::: fine.aggregations.spatialAggregation.manager
    options:
        show_root_heading: true
        show_source: false

## Technology Aggregation

::: fine.aggregations.technologyAggregation.techAggregation
    options:
        show_root_heading: true
        show_source: false

## Temporal Aggregation

`EnergySystemModel.aggregateTemporally` clusters the time series data with
[ETHOS.TSAM](https://github.com/FZJ-IEK3-VSA/tsam). Its parameters are the ETHOS.TSAM 4.x ones —
`n_clusters`, `period_duration`, `cluster`, `segments`, `extremes` and `preserve_column_means`.

The aggregation calls `tsam.aggregate` directly and, with `storeTSAinstance=True`, stores the
`tsam.AggregationResult` it returns as `esM.tsaInstance`. That is the ETHOS.TSAM 4.x object
itself, so its own attributes apply: `cluster_representatives`, `cluster_assignments`,
`period_index`, `n_clusters`, `n_segments`, `accuracy` and `clustering`. The removed
`TimeSeriesAggregation` attribute names (`clusterPeriodDict`, `clusterOrder`,
`noTypicalPeriods`, …) are gone with it.

### Deprecated ETHOS.TSAM 3.x keywords

The ETHOS.TSAM 3.x keywords (`numberOfTypicalPeriods`, `clusterMethod`, `representationMethod`,
`addPeakMax`, …) can still be passed to `aggregateTemporally` as keyword arguments. The module below
converts them and warns. The two interfaces cannot be combined in one call — mixing them raises a
`TypeError` naming the replacement for every deprecated keyword. This module is removed once the
deprecation period ends.

::: fine.aggregations.temporalAggregation.deprecatedKeywords
    options:
        show_root_heading: true
        show_source: false
