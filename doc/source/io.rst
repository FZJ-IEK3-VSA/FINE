*********************
Data Input and Output
*********************

SPAGAT is build upon [xarray (N-D labeled arrays and datasets in Python)](http://xarray.pydata.org/en/stable/), as energy system data comprises spatiotemporal data for many different energy system components and is thus multi-dimensional.

Furthermore, interfaces to [pandas](https://pandas.pydata.org/) are provided and [GeoPandas](https://geopandas.org/) and [Shapely](https://pypi.org/project/Shapely/) are used for geometric calculations and plotting.

Finally, [dask](https://dask.org/) can be used to leverages HPC - if available - to accelerate the clustering algorithms using parallelization.