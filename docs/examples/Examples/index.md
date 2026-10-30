# Modular Examples

The modular examples are a tutorial series for users who have never used ETHOS.FINE before.
Start with the basic tutorial, which builds, optimizes and analyzes a complete small energy system
model in one notebook. The component notebooks that follow build on this model. Each of them adds
new components of one type and introduces further parameters.

## Overview
| Example | Description |
|---|---|
| [Basic Tutorial](_00_introduction/_2_basic_tutorial.ipynb) | The electricity system of Germany in two regions (north and south) over three investment periods (2020, 2030, 2040) with one example of every component type: sources, sinks, a conversion, a storage and a transmission. A tightening CO<sub>2</sub> limit drives the system from gas power to wind, PV and batteries. **Recommended as starting point.** |
| [Sources](_00_introduction/_3_sources.ipynb) | Offshore wind turbines that are only allowed in the north (`locationalEligibility`), an expansion target (`capacityMin` per investment period) and existing run-of-river plants (`capacityFix`, `operationRateFix`). |
| [Sinks](_00_introduction/_4_sinks.ipynb) | A CO<sub>2</sub> price instead of a CO<sub>2</sub> limit (`commodityCost` on a sink) and electricity exports through cables with a fixed capacity (`capacityFix`, `commodityRevenue`). |
| [Conversions](_00_introduction/_5_conversions.ipynb) | Existing gas power plants (`stockCommissioning`, `esM.updateComponent`) and electrolyzers that produce hydrogen for the industry (`opexPerOperation`), next to hydrogen imports. |
| [Storages](_00_introduction/_6_storages.ipynb) | Existing pumped hydro storage (different `chargeRate` and `dischargeRate`) and seasonal hydrogen storage in salt caverns (`stateOfChargeMin`) in a climate-neutral system in 2040, and how storage works with typical days. |
| [Transmissions](_00_introduction/_7_transmissions.ipynb) | The existing AC grid (`stockCommissioning` per connection) and planned HVDC cables with lower losses (`commissioningFix`), drawn on a map with `fn.plotTransmission`. |
