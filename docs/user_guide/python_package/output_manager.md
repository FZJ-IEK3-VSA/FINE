# Input Output Manager

FINE provides additional functionalities for input/output management. These include plotting functions,
saving results in Excel files, storing input and output data in netCDF files, and the exploitation of the
optimized energy system.

## Storage formats

`convertEnergySystemModelToDatasets` is the one export. It turns a model into the canonical nested
dictionary of xarray datasets, and every format is written from that. A model can be written in three
formats. All three read back into an equal `EnergySystemModel`.

| Format | Write | Read | Use it when |
|---|---|---|---|
| One netCDF file | `writeEnergySystemModelToNetCDF` | `readNetCDFToEnergySystemModel` | The model is small enough to move as one file. |
| A netCDF folder | `writeEnergySystemModelToNetCDFfolder` | `readNetCDFfolderToEnergySystemModel` | Writing or reading is slow. One file per dataset means the files are independent, so they can be handled by several processes at once or read on demand. |
| A Zarr store | `writeEnergySystemModelToZarr` | `readZarrToEnergySystemModel` | You want to read one variable across all components without reading everything. The components of a class are stacked into one array, and the arrays are chunked and compressed. |

`writeDatasetsToNetCDFfolder`, `writeDatasetsToZarr`, `readNetCDFfolderToDatasets` and
`readZarrToDatasets` are the same formats one level down, for a caller that holds the datasets already.
`readZarrToDatasets` returns the store as it was written, that is stacked and lazy, which is the point of
the format.

### How the Zarr store records a shape

The netCDF formats record the shape of a component parameter in a prefix on the variable name (`0d_`,
`1d_`, `2d_`, `ts_`). The Zarr format cannot, because it puts every component of a class into one dataset,
concatenated along `component`, and a name has to mean the same thing for all of them. It carries two mask
variables over `(component, parameter)` instead:

| Variable | Type | Value |
|---|---|---|
| `variable_present` | bool | `False` means this component did not hold this parameter |
| `variable_dims` | string | the index names, comma joined and in order; `""` means scalar |

`variable_dims` holds the index names, that is the dimensions plus the scalar coordinates. The netCDF
builder calls `squeeze()`, so a component that uses a single location holds `space` as a scalar
coordinate, and the dimensions alone would report a scalar and rebuild the wrong name.
`utilsIO.stackComponents` and `utilsIO.unstackComponents` are the inverse pair that writes and reads them,
and `structure.json` records the layout version as `fine_zarr_format`.

The presence mask is what keeps the round trip exact. xarray has no `None`, so a parameter that was `None`
is written as `NaN`; an all-`NaN` variable is skipped on read, which leaves the parameter at its default,
`None`.

Known limitation: the masks restore a missing variable, they do not restore a missing coordinate.
Concatenating along `component` widens every variable to the union of the components' coordinates, so a
component that uses two of five locations comes back padded to five, and a genuine all-`NaN` row cannot be
told from padding.

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
