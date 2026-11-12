# EnergySystemModel

A description of the `EnergySystemModel` class is given below. This is the central class of ETHOS.FINE.
It is used to create an energy system model, add components to it, build the optimization problem, solve it
and visualize/export results.

## Infeasibility pre-checks

Before the optimization problem is built, `optimize` and `declareOptimizationProblem` check the model for
setups which cannot be solved, for example a demand without any supply. Without these checks, such errors only
show up as an "infeasible" solver status after the whole problem has been built and solved, which is hard to
trace back in a large model.

If a pre-check detects a problem, a `ValueError` is raised. The message lists every problem with the name of
the check, the investment period (for models with several investment periods), the time step where applicable,
and the commodity and locations concerned:

```text
The infeasibility pre-checks detected that the model cannot be solved:
  - checkJointInputDemandAggregated: Total time horizon: the joint demand of 11.43 for the commodity 'electricity' in the locations ['Region1', 'Region2', 'Region3'] exceeds the supply of 10 and no conversion component produces this commodity.
  - checkJointInputDemandPerTimeStep: Time step 3: the joint demand of 7.143 for the commodity 'electricity' in the locations ['Region1', 'Region2', 'Region3'] exceeds the supply of 4.85 and no conversion component produces this commodity.

The pre-checks can be deactivated by setting 'runInfeasibilityPrechecks' to False.
```

In this example, electrolyzers have to cover a fixed hydrogen demand. Over the time horizon they need 11.43
units of electricity, but the wind turbines can supply at most 10, and in the last time step the batteries
cannot close the gap either.

!!! note "Passing the pre-checks is no feasibility guarantee"
    Every check evaluates a necessary condition for feasibility and assumes unknown or unbounded values to be
    as favorable as possible, for example an unlimited capacity if no `capacityMax` is given. A reported problem
    therefore proves that the model is infeasible. A model which passes all checks can still be infeasible.

### What is checked

| Check | Detects |
|---|---|
| `checkCommodityReachability` | A demand at a location which no source, conversion or transmission path can supply at all, e.g. because a component or a transmission link is missing or a supply time series is zero. The check is purely structural; quantities are ignored. |
| `checkBalanceLimits` | A `balanceLimit` which the components with this `balanceLimitID` cannot meet, e.g. fixed emissions which exceed an emission limit. Operation only counts as enforced if it is fixed by `operationRateFix` and, for components with a capacity variable, by a guaranteed capacity. Transmission contributes at most its capacity. |
| `checkJointInputDemandAggregated` | Over the whole time horizon, the demand of all sinks, traced back through the conversions to their inputs, exceeds what the sources can supply. This catches shortages which only occur because several conversions compete for the same input, e.g. electrolyzers and heat pumps which both consume electricity. |
| `checkTimeStepBalance` | In a time step, the demand cannot be delivered over the network because the supply (sources, storage discharge and conversion output) or the transmission capacity is too small. This includes bottlenecks on paths across intermediate locations. |
| `checkJointInputDemandPerTimeStep` | Like `checkJointInputDemandAggregated`, but for every time step, with the storage discharge added to the supply. |

For the demand, only what has to be met in every feasible solution counts. For a sink with a capacity
variable, `operationRateFix` is scaled by its guaranteed capacity, which is given by `capacityFix` or
`capacityMin` and, in the first investment period, at least by the existing stock. A component with an
`isBuilt` binary variable only counts as built where `isBuiltFix` is 1. Pooled commodities are balanced
over each pool as a whole, like over transmission links with unlimited capacity.

### Known limitations

The pre-checks do not consider:

- the state of charge of storage components over time: in each time step, a storage can discharge up to its
  capacity, and over the whole time horizon storage is ignored, because it only shifts energy in time,
- the coupling between investment periods: each investment period is checked on its own, so capacities installed
  in earlier periods and the decommissioning of the stock are not taken into account,
- `pathwayBalanceLimit`, minimum operation rates and full load hours,
- the competition of conversions for the same input in `checkTimeStepBalance`, and transmission capacities in the
  two joint input demand checks,
- the input demand of conversions with flexible inputs (commodity groups) in the quantity-based checks; only
  `checkCommodityReachability` requires one of their inputs to be available.

With time series aggregation (`timeSeriesAggregation=True`), the pre-checks are skipped, because they evaluate the
full time series and would not describe the aggregated model.

### Disabling the pre-checks

The pre-checks run by default. They can be disabled with the `runInfeasibilityPrechecks` parameter of `optimize`
and `declareOptimizationProblem`, for example for a large model which is known to be feasible:

```python
esM.optimize(timeSeriesAggregation=False, runInfeasibilityPrechecks=False)
```

::: fine.energySystemModel
    options:
        show_root_heading: true
        show_source: false
