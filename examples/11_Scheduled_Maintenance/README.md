# Scheduled maintenance

This example demonstrates independent optimizer-scheduled maintenance for a
`ConversionDynamic` component. The boiler must undergo two distinct maintenance
windows, each lasting at least three hours. A more expensive backup source meets
the heat demand while the boiler is offline.

Run the example from the repository root:

```bash
python examples/11_Scheduled_Maintenance/scheduled_maintenance.py
```

The resulting plot shows boiler operation and shades the maintenance periods.
The active schedule is also available through
`maintenanceActiveVariablesOptimum` and is included by the standard xarray and
netCDF result exports.
