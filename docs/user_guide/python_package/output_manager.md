# Input Output Manager

FINE provides additional functionalities for input/output management. These include plotting functions,
saving results in Excel files, storing input and output data in netCDF files, and the exploitation of the
optimized energy system.

## Storage formats

A model can be written in three formats. All three read back into an equal `EnergySystemModel`.

| Format | Write | Read | Use it when |
|---|---|---|---|
| One netCDF file | `writeEnergySystemModelToNetCDF` | `readNetCDFtoEnergySystemModel` | The model is small enough to move as one file. |
| A netCDF folder | `writeDatasetsToNetCDFfolder` | `readNetCDFfolderToDatasets` | Writing or reading is slow. One file per dataset means the files are independent, so they can be handled by several processes at once or read on demand. |
| A Zarr store | `writeDatasetsToZarr` | `readZarrToEnergySystemModel` | You want to read one variable across all components without reading everything. The components of a class are stacked into one array, and the arrays are chunked and compressed. |

The netCDF formats record the shape of a component parameter in a prefix on the variable name (`0d_`,
`1d_`, `2d_`, `ts_`). The Zarr format cannot, because it puts every component of a class into one dataset
and a name has to mean the same thing for all of them. It stores a `dimension_mask` and a `was_none_mask`
per component instead. The was-none mask is what makes the round trip exact: xarray has no `None`, so a
parameter that was `None` is written as `NaN`, and without the mask it would read back as `NaN`, which
means something else.

`writeEnergySystemModelToDatasetsBoth` builds the netCDF view and the Zarr view from a single export.
That matters for a temporally aggregated model, where the export rebuilds the full time series and is by
far the expensive step.

## Standard I/O

::: fine.IOManagement.standardIO
    options:
        show_root_heading: true
        show_source: false

## xarray I/O

::: fine.IOManagement.xarrayIO
    options:
        show_root_heading: true
        show_source: false

## Exploit Output

::: fine.IOManagement.dictIO
    options:
        show_root_heading: true
        show_source: false
