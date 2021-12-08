"""Representation of RE technologies in every region.
"""
import logging
import numpy as np
import xarray as xr
from sklearn.cluster import AgglomerativeClustering
import FINE.spagat.utils as spu
import FINE.spagat.RE_representation_utils as RE_rep_utils

logger_RERep = logging.getLogger("spatial_RE_representation")


@spu.timer
def represent_RE_technology(
    gridded_RE_ds,
    CRS_attr,
    shp_file,
    n_timeSeries_perRegion=1,
    capacity_var_name="capacity",
    capfac_var_name="capacity factor",
    longitude="x",
    latitude="y",
    time="time",
    index_col="region_ids",
    geometry_col="geometry",
    linkage="average",
):
    """
    Reduces the number of a particular RE technology (e.g. onshore wind turbine)
    to a desired number, within each region.

    .. note::
        The explanation below uses wind
        turbines as an example. It could, in reality, be any variable RE technology like
        PV, offshore wind turbine, etc.

    The number of simulated wind turbines could be huge. This function reduces them to a
    few turbine types, in each of the defined region. Each wind turbine is characterised by
    its capacity and capacity factor time series.

    The basic idea here is to group the turbines, within each region, such that the turbines
    with most similar capacity factor time series appear in the same group. Next, the turbines in
    each group are aggregated to obtain one turbine type, per group, thereby reducing the total
    number of turbines.

    Please go through the parameters list below for more information.

    :param gridded_RE_ds: Either the path to the dataset or the read-in xr.Dataset\n
        * Dimensions in this data: `latitude`, `longitude`, and `time`
        * Variables: `capacity_var_name` and `capfac_var_name`
    :type gridded_RE_ds: str/xr.Dataset

    :param CRS_attr: The attribute in `gridded_RE_ds` that holds its
        Coordinate Reference System (CRS) information
    :type CRS_attr: str

    :param shp_file: Either the path to the shapefile or the read-in shapefile
        that should be overlapped with `gridded_RE_ds`, in order to
        obtain regions' information
    :type shp_file: str/GeoDataFrame

    **Default arguments:**

    :param n_timeSeries_perRegion: The number of time series to which the original set should be aggregated,
        within each region.\n
        * If set to 1, performs simple aggregation\n
            - Within every region, calculates the weighted mean of RE
              time series (capacities being weights), and sums the capacities.
        * If set to a value greater than 1, time series clustering is employed\n
            - Clustering method: Sklearn's agglomerative hierarchical clustering
            - Distance measure: Euclidean distance
            - Aggregation within each resulting cluster is the same as simple
              aggregation
        |br| * the default value is 1
    :type n_timeSeries_perRegion: strictly positive int

    :param capacity_var_name: The name of the data variable in `gridded_RE_ds` that corresponds
        to capacity
        |br| * the default value is 'capacity'
    :type capacity_var_name: str

    :param capfac_var_name: The name of the data variable in `gridded_RE_ds` that corresponds
        to capacity factor time series
        |br| * the default value is 'capacity factor'
    :type capfac_var_name: str

    :param longitude: The dimension name in `gridded_RE_ds` that corresponds to longitude
        |br| * the default value is 'x'
    :type longitude: str

    :param latitude: The dimension name in `gridded_RE_ds` that corresponds to latitude
        |br| * the default value is 'y'
    :type latitude: str

    :param time: The dimension name in `gridded_RE_ds` that corresponds to time
        |br| * the default value is 'time'
    :type time: str

    :param index_col: The column in `shp_file` that needs to be taken as location-index in `gridded_RE_ds`
        |br| * the default value is 'region_ids'
    :type index_col: str

    :param geometry_col: The column in `shp_file` that holds geometries
        |br| * the default value is 'geometry'
    :type geometry_col: str

    :param linkage:\n
        * Relevant only if `n_timeSeries_perRegion` is greater than 1.
        * The linkage criterion to be used with agglomerative hierarchical clustering.
          Can be 'complete', 'single', etc. Refer to Sklearn's documentation for more info.
        |br| * the default value is 'average'
    :type linkage: str

    :returns: represented_RE_ds\n
        * Dimensions in this data: `time`, 'region_ids'
        * The dimension 'region_ids' has its coordinates corresponding to `index_col`\n
        If `n_timeSeries_perRegion` is greater than 1, additional dimension 'TS_ids' is present.
        Within each region, different time series are indicated by this 'TS_ids'
    :rtype: xr.Dataset
    """

    def _preprocess_regional_xr_ds(region):
        """
        Private function to preprocess regional data.
        """
        # Get regional data
        regional_ds = rasterized_RE_ds.sel(region_ids=region)

        regional_capfac_da = regional_ds[capfac_var_name].where(
            regional_ds.rasters == 1
        )
        regional_capacity_da = regional_ds[capacity_var_name].where(
            regional_ds.rasters == 1
        )

        # Restructure data
        regional_capfac_da = regional_capfac_da.stack(x_y=[longitude, latitude])
        regional_capfac_da = regional_capfac_da.transpose(transpose_coords=True)

        regional_capacity_da = regional_capacity_da.stack(x_y=[longitude, latitude])
        regional_capacity_da = regional_capacity_da.transpose(transpose_coords=True)

        # Remove all time series with 0 values
        regional_capfac_da = regional_capfac_da.where(regional_capacity_da > 0)
        regional_capacity_da = regional_capacity_da.where(regional_capacity_da > 0)

        # Drop NAs
        regional_capfac_da = regional_capfac_da.dropna(dim="x_y")
        regional_capacity_da = regional_capacity_da.dropna(dim="x_y")

        # Print out number of time series in the region
        n_ts = len(regional_capfac_da["x_y"].values)
        logger_RERep.info(f"Number of time series in {region}: {n_ts}")

        # Get power curves from capacity factor time series and capacities
        regional_power_da = regional_capacity_da * regional_capfac_da

        return regional_capfac_da, regional_capacity_da, regional_power_da

    # STEP 1. Rasterize the gridded dataset
    rasterized_RE_ds = RE_rep_utils.rasterize_xr_ds(
        gridded_RE_ds, CRS_attr, shp_file, index_col, geometry_col, longitude, latitude
    )

    region_ids = rasterized_RE_ds["region_ids"].values
    n_regions = len(region_ids)

    time_steps = rasterized_RE_ds[time].values
    n_timeSteps = len(time_steps)

    if n_timeSeries_perRegion == 1:

        # STEP 2. Create resultant xarray dataset
        ## time series
        data = np.zeros((n_timeSteps, n_regions))

        represented_timeSeries = xr.DataArray(
            data, [(time, time_steps), ("region_ids", region_ids)]
        )

        # capacities
        data = np.zeros(n_regions)
        represented_capacities = xr.DataArray(data, [("region_ids", region_ids)])

        # STEP 3. Aggregation in every region...
        for region in region_ids:
            # Preprocess regional data
            (
                regional_capfac_da,
                regional_capacity_da,
                regional_power_da,
            ) = _preprocess_regional_xr_ds(region)

            # Aggregation
            ## capacity
            capacity_total = regional_capacity_da.sum(dim="x_y").values
            represented_capacities.loc[region] = capacity_total

            ## capacity factor
            power_total = regional_power_da.sum(dim="x_y").values
            capfac_total = power_total / capacity_total

            represented_timeSeries.loc[:, region] = capfac_total

        # STEP 4. Create resulting dataset
        regional_represented_RE_ds = xr.Dataset(
            {
                capacity_var_name: represented_capacities,
                capfac_var_name: represented_timeSeries,
            }
        )

        return regional_represented_RE_ds

    else:

        # STEP 2. Create resultant xarray dataset
        TS_ids = [f"TS_{i}" for i in range(n_timeSeries_perRegion)]

        ## time series
        data = np.zeros((n_timeSteps, n_regions, n_timeSeries_perRegion))

        represented_timeSeries = xr.DataArray(
            data, [(time, time_steps), ("region_ids", region_ids), ("TS_ids", TS_ids)]
        )

        data = np.zeros((n_regions, n_timeSeries_perRegion))

        # capacities
        represented_capacities = xr.DataArray(
            data, [("region_ids", region_ids), ("TS_ids", TS_ids)]
        )

        # STEP 3. Clustering in every region...
        for region in region_ids:
            # Preprocess regional data
            (
                regional_capfac_da,
                regional_capacity_da,
                regional_power_da,
            ) = _preprocess_regional_xr_ds(region)

            # Clustering
            agg_cluster = AgglomerativeClustering(
                n_clusters=n_timeSeries_perRegion, affinity="euclidean", linkage=linkage
            )
            agglomerative_model = agg_cluster.fit(regional_capfac_da)

            # Aggregation
            for i in range(np.unique(agglomerative_model.labels_).shape[0]):
                ## Aggregate capacities
                cluster_capacity = regional_capacity_da[
                    agglomerative_model.labels_ == i
                ]
                cluster_capacity_total = cluster_capacity.sum(dim="x_y").values

                represented_capacities.loc[region, TS_ids[i]] = cluster_capacity_total

                # aggregate capacity factor
                cluster_power = regional_power_da[agglomerative_model.labels_ == i]
                cluster_power_total = cluster_power.sum(dim="x_y").values
                cluster_capfac_total = cluster_power_total / cluster_capacity_total

                represented_timeSeries.loc[:, region, TS_ids[i]] = cluster_capfac_total

        # STEP 4. Create resulting dataset
        regional_represented_RE_ds = xr.Dataset(
            {
                capacity_var_name: represented_capacities,
                capfac_var_name: represented_timeSeries,
            }
        )

        return regional_represented_RE_ds
