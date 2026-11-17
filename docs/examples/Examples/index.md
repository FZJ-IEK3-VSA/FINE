# Modular Examples

The modular examples are a tutorial series for users who have never used ETHOS.FINE before.
Start with the basic tutorial, which builds, optimizes and analyzes a complete small energy system
model in one notebook. All other notebooks build on this model: each of them changes or extends it
to introduce a few parameters or features. They can be read in order, but each of them also works
on its own.

The shared code of the notebooks (the model of the basic tutorial, the colors and a plot function)
is in `examples/Examples/tutorialHelpers.py`, the input data in `examples/data/twoRegions`.

## Getting started

| Example | Description |
|---|---|
| [Basic Tutorial](01_basic_tutorial.ipynb) | The electricity system of Germany in two regions (north and south) over three investment periods (2020, 2030, 2040) with one example of every component type: sources, sinks, a conversion, a storage and a transmission. A tightening CO<sub>2</sub> limit drives the system from gas power to wind, PV and batteries. **Recommended as starting point.** |

## Component types

| Example | Description |
|---|---|
| [Sources](11_sources.ipynb) | Offshore wind turbines that are only allowed in the north (`locationalEligibility`), an expansion target (`capacityMin` per investment period) and existing run-of-river plants (`capacityFix`, `operationRateFix`). |
| [Sinks](12_sinks.ipynb) | A CO<sub>2</sub> price instead of a CO<sub>2</sub> limit (`commodityCost` on a sink) and electricity exports through cables with a fixed capacity (`capacityFix`, `commodityRevenue`). |
| [Conversions](13_conversions.ipynb) | Existing gas power plants (`stockCommissioning`, `esM.updateComponent`) and electrolyzers that produce hydrogen for the industry (`opexPerOperation`), next to hydrogen imports. |
| [Storages](14_storages.ipynb) | Existing pumped hydro storage (different `chargeRate` and `dischargeRate`) and seasonal hydrogen storage in salt caverns (`stateOfChargeMin`) in a climate-neutral system in 2040, and how storage works with typical days. |
| [Transmissions](15_transmissions.ipynb) | The existing AC grid (`stockCommissioning` per connection) and planned HVDC cables with lower losses (`commissioningFix`), drawn on a map with `fn.plotTransmission`. |

## Limits for all component types

| Example | Description |
|---|---|
| [Capacity Limits](21_capacity_limits.ipynb) | Capacity and commissioning and their limits (`capacityMin/Max/Fix`, `commissioningMin/Max/Fix`), a maximum building rate for wind turbines (`commissioningMax`) and open-field PV that shares its land with wind turbines (`sharedPotentialID`). |
| [Operation Limits](22_operation_limits.ipynb) | Limits per time step and per year (`operationRateMin/Max/Fix`, `yearlyFullLoadHoursMin/Max`): must-run CHP plants in winter (`operationRateMin`) and biogas power plants with limited fuel (`yearlyFullLoadHoursMax`). |

## Optimization and results

| Example | Description |
|---|---|
| [Time Series Aggregation](31_time_series_aggregation.ipynb) | How typical days are made (`aggregateTemporally`), how many are needed, fewer time steps per day (`segments`) and extreme days (`extremes`). |
| [Solver Settings and Saving](32_solver_settings_and_saving.ipynb) | Choosing a solver and its settings (`solver`, `timeLimit`, `optimizationSpecs`, `threads`), less output (`verboseLogLevel`), `esM.solverSpecs`, and saving and loading a model as a NetCDF file. |
| [Results and Plots](33_results_and_plots.ipynb) | The optimization summaries and their properties, hourly results (`getOptimalValues`), electricity prices from shadow prices (`fn.getShadowPrices`) and the plotting functions of FINE. |

## Advanced features

| Example | Description |
|---|---|
| [Balance Limits](41_balance_limits.ipynb) | A limit on the net imports of one region (regional `balanceLimit`, `balanceLimitID` on a transmission) and a CO<sub>2</sub> budget for the whole pathway (`pathwayBalanceLimit`). |
| [Investment Periods](42_investment_periods.ipynb) | How FINE computes the total cost of a pathway (annuities and discounting, `NPVcontribution`), lifetimes and whole periods (`technicalLifetime`, `floorTechnicalLifetime`) and the end of the pathway (`annuityPerpetuity`). |
| [Binary Decisions](43_binary_decisions.ipynb) | A hydrogen pipeline with a fixed cost (`hasIsBuiltBinaryVariable`, `investIfBuilt`, `bigM`) and gas power plants in whole units (`capacityVariableDomain="discrete"`, `capacityPerPlantUnit`). |
| [Power Plant Flexibility](44_power_plant_flexibility.ipynb) | Ramping limits (`rampUpMax`, `rampDownMax`), a minimum load (`partLoadMin`) and minimum operating times (`fn.ConversionDynamic` with `upTimeMin`, `downTimeMin`). |
| [Flexible Conversions](45_flexible_conversions.ipynb) | Conversion factors that change over time or between regions, and gas power plants that can burn natural gas or hydrogen (`emissionFactors`, `flowShares`). |
