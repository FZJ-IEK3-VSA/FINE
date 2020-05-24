*********************
Data Input and Output
*********************

SPAGAT is build upon xarray, as energy system data comprises spatiotemporal data for many different energy system components and is thus multi-dimensional.

Furthermore, interfaces to pandas are provided and geopandas and shapely are used for geometric calculations and plotting.

Finally, dask leverages HPC - if available - to accelerate the clustering algorithms using parallelization.