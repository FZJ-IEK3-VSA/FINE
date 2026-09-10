# Progress Protocol: Rolling Horizon / Reoptimization Parameter Filtering

## Context

Rolling Horizon and event-triggered reoptimization both rebuild a smaller `EnergySystemModel` from an existing full transformation pathway. In both cases, already optimized or already existing capacities are passed to the next optimization as `stockCommissioning`.

This creates two different kinds of years:

- **active optimization years**: years that are part of the newly built model and can be optimized.
- **stock years**: earlier commissioning years of already existing assets that remain relevant for lifetime and material accounting.

Example for a disruption in 2035:

```text
full model years:      2025, 2030, 2035, 2040, 2045, 2050
committed stock years: 2025, 2030
tail years:            2035, 2040, 2045, 2050
```

## Key Distinction

Old stock years are needed for asset history, but they are not active operation years of the new model.

Therefore, parameter filtering must distinguish between:

- stock- or commissioning-relevant parameters
- operational parameters

## Stock-Relevant Parameters

Some parameters must keep old stock years because old assets may decommission during the tail optimization.

Example: `materialIntensity`

If a wind plant was commissioned in 2030 and decommissions in 2045, the tail model still needs to know how much material was embedded in the 2030 stock.

For such parameters, the relevant years are:

```text
active optimization years + stock years
```

Example:

```text
materialIntensity: 2025, 2030, 2035, 2040, 2045, 2050
```

or, after lifetime pruning:

```text
materialIntensity: relevant stock years + 2035, 2040, 2045, 2050
```

## Operational Parameters

Operational parameters describe behavior in years that the rebuilt model actually optimizes.

Examples:

- `operationRateMax`
- `operationRateMin`
- `operationRateFix`
- operation-related time series
- material import disruption time series

These parameters must only keep active optimization years.

For a tail starting in 2035:

```text
operationRateMax: 2035, 2040, 2045, 2050
```

They must not keep old stock years:

```text
wrong: operationRateMax: 2025, 2030, 2035, 2040, 2045, 2050
```

The old years 2025 and 2030 are no longer optimized in the tail model. Passing them as operational parameter keys can make FINE reject the component because the parameter contains years that are not part of the rebuilt model.

## Why This Matters

The earlier Rolling Horizon filtering logic used a broad rule for many dictionary parameters:

```python
relevantYears = rollingHorizonYears + stockYears
```

This is useful for some stock-dependent parameters, but it is too broad for operational parameters.

It can fail for IP-dependent operational inputs such as a material import source with:

```python
operationRateMax = {
    2025: ...,
    2030: ...,
    2035: ...,
    ...
}
```

If an interval or tail model starts in 2035, the years 2025 and 2030 must not remain in `operationRateMax`.

## Current Reoptimization Rule

The reoptimization helper should use the following filtering concept:

```text
materialIntensity:          active years + stock years
materialCollection:         active years only
commodityConversionFactors: operation year must be active;
                            commissioning year may be active or stock
ordinary operational dicts: active years only
```

This keeps old assets materially and technically meaningful without leaking old operation years into the new model.

## Implication for Rolling Horizon

This issue is not unique to reoptimization. Classic Rolling Horizon can run into the same problem if operational parameters are IP-dependent and stock years are added too broadly.

The reoptimization block currently handles this more strictly. A future cleanup should align Rolling Horizon filtering with the same distinction between stock-relevant and operational parameters.

## Open Follow-Up

Review `_filterComponentParametersForInterval` in `fine/expansionModules/rollingHorizon.py` and consider replacing the broad dictionary rule with explicit parameter groups:

```text
stock-aware parameters:     rollingHorizonYears + stockYears
operational parameters:     rollingHorizonYears only
materialIntensity:          rollingHorizonYears + stockYears
materialCollection:         rollingHorizonYears only
```

This should be tested with an IP-dependent `operationRateMax` material import disruption case.
