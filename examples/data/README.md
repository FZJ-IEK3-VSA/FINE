# Shared example data

Several examples use the same input data. The data is stored here one time.

| Folder | Used by |
| --- | --- |
| `multiRegion` | 03 Multi-regional Energy System Workflow, 07 Spatial and technology aggregation |
| `oneNode` | 01 1node Energy System Workflow, 09 PerfectForesight, 10 Partload |

The loader functions are in [`../exampleData.py`](../exampleData.py). Each
example has a small `getData.py`. This `getData.py` gives the function
`getData()` and the constant `INPUT_DATA_PATH` for the example.

Use `INPUT_DATA_PATH` in a notebook to find a data file:

```python
from getData import getData, INPUT_DATA_PATH

data = getData()
shapeFilePath = INPUT_DATA_PATH / "SpatialData" / "ShapeFiles" / "clusteredRegions.shp"
```

Examples with their own input data (02 EnergyLand, 04 District Optimization)
keep that data in their own folder.
