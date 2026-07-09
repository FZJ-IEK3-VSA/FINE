# Expansion Modules

FINE expansion modules provide additional functionalities to the user. These functionalities either run energy
system models multiple times or can be applied in the pre- and post-processing of an energy system model run.

## Optimize TSA Multi Stage

::: fine.expansionModules.optimizeTSAmultiStage
    options:
        show_root_heading: true
        show_source: false

## Rolling Horizon

::: fine.expansionModules.rollingHorizon
    options:
        show_root_heading: true
        show_source: false

## Transformation Path

!!! warning "Deprecated"
    `optimizeSimpleMyopic` is deprecated and no longer maintained. Use
    [Rolling Horizon](#rolling-horizon) with
    `numberOfInvestmentPeriodsForRollingHorizon=1` instead, which covers the
    same myopic foresight use case.

::: fine.expansionModules.transformationPath
    options:
        show_root_heading: true
        show_source: false

## Piecewise Linear Cost Function

::: fine.expansionModules.piecewiseLinearCostFunction
    options:
        show_root_heading: true
        show_source: false
