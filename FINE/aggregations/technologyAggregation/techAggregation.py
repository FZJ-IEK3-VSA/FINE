"""
Aggregation of RE technologies in every region.
"""
import logging
import numpy as np
import xarray as xr
from sklearn.cluster import AgglomerativeClustering
from FINE.IOManagement.standardIO import timer
from FINE.aggregations.technologyAggregation import techAggregationUtils

logger_tech_agg = logging.getLogger("technology_representation")


@timer
def aggregate_RE_technology(
    gridded_RE_ds=None,
    CRS_attr=None,
    shp_file=None,
    non_gridded_RE_ds=None,
    n_timeSeries_perRegion=1,
    capacity_var_name="capacity",
    capfac_var_name="capacity factor",
    region_var_name="region",
    longitude_dim_name="x",
    latitude_dim_name="y",
    time_dim_name="time",
    location_dim_name="locations",
    shp_index_col="region_ids",
    shp_geometry_col="geometry",
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

    **Default arguments:**

    :param gridded_RE_ds: Either the path to the dataset or the read-in xr.Dataset

        * Dimensions in this data - `latitude_dim_name`, `longitude_dim_name`, and `time_dim_name`
        * Variables: `capacity_var_name` and `capfac_var_name`

    :type gridded_RE_ds: str/xr.Dataset

    :param CRS_attr: The attribute in `gridded_RE_ds` that holds its
        Coordinate Reference System (CRS) information
    :type CRS_attr: str

    :param shp_file: Either the path to the shapefile or the read-in shapefile
        that should be overlapped with `gridded_RE_ds`, in order to
        obtain regions' information
    :type shp_file: str/GeoDataFrame

    :param non_gridded_RE_ds: Either the path to the dataset or the read-in xr.Dataset

        * Dimensions in this data - `location_dim_name` and `time_dim_name`
        * Variables - `capacity_var_name`, `capfac_var_name`, and `region_var_name`

        One can either pass `gridded_RE_ds` or `non_gridded_RE_ds` to work with. If both are passed,
        `gridded_RE_ds` is considered

    :type non_gridded_RE_ds: str/xr.Dataset

    :param n_timeSeries_perRegion: The number of time series to which the original set should be aggregated,
        within each region.

        * If set to 1, performs simple aggregation

            - Within every region, calculates the weighted mean of RE
              time series (capacities being weights), and sums the capacities.

        * If set to a value greater than 1, time series clustering is employed

            - Clustering method: Sklearn's agglomerative hierarchical clustering
            - Distance measure: Euclidean distance
            - Aggregation within each resulting cluster is the same as simple
              aggregation

        |br| * the default value is 1
    :type n_timeSeries_perRegion: strictly positive int

    :param capacity_var_name: The name of the data variable in the provided dataset that corresponds to capacity
        |br| * the default value is 'capacity'
    :type capacity_var_name: str

    :param capfac_var_name: The name of the data variable in the provided dataset that corresponds
        to capacity factor time series
        |br| * the default value is 'capacity factor'
    :type capfac_var_name: str

    :param region_var_name: The name of the data variable in `non_gridded_RE_ds` that contains region IDs
        |br| * the default value is 'region'
    :type region_var_name: str

    :param longitude_dim_name: The dimension name in `gridded_RE_ds` that corresponds to longitude
        |br| * the default value is 'x'
    :type longitude_dim_name: str

    :param latitude_dim_name: The dimension name in `gridded_RE_ds` that corresponds to latitude
        |br| * the default value is 'y'
    :type latitude_dim_name: str

    :param time_dim_name: The dimension name in in the provided dataset that corresponds to time
        |br| * the default value is 'time'
    :type time_dim_name: str

    :param location_dim_name: The dimension name in `non_gridded_RE_ds` that corresponds to locations
        |br| * the default value is 'locations'
    :type location_dim_name: str

    :param shp_index_col: The column in `shp_file` that needs to be taken as location-index in `gridded_RE_ds`
        |br| * the default value is 'region_ids'
    :type shp_index_col: str

    :param shp_geometry_col: The column in `shp_file` that holds geometries
        |br| * the default value is 'geometry'
    :type shp_geometry_col: str

    :param linkage:

        * Relevant only if `n_timeSeries_perRegion` is greater than 1.
        * The linkage criterion to be used with agglomerative hierarchical clustering.
          Can be 'complete', 'single', etc. Refer to Sklearn's documentation for more info.

        |br| * the default value is 'average'
    :type linkage: str

    :returns: regional_aggregated_RE_ds

        * Dimensions in this data: `time_dim_name`, 'region_ids'
        * The dimension 'region_ids' has its coordinates corresponding to `shp_index_col` if
            `gridded_RE_ds` is passed. Otherwise, it corresponds to `region_var_name` values

        If `n_timeSeries_perRegion` is greater than 1, additional dimension - 'TS_ids' is present
        * Within each region, different time series are indicated by this 'TS_ids'

        * In addition, the dataset also contains attributes which indicate which time series were
            clustered. Calling represented_RE_ds.attrs would render a dictionary that contains
            <region_ids>.<TS_ids> as keys and a list of original locations as values. In case of
            `gridded_RE_ds`, these locations are a tuple - (x/longitude, y/latitude)

    :rtype: xr.Dataset
    """

    def _preprocess_regional_xr_ds_gridded(region):
        """
        Private function to preprocess regional gridded data
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
        regional_capfac_da = regional_capfac_da.stack(
            x_y=[longitude_dim_name, latitude_dim_name]
        )
        regional_capfac_da = regional_capfac_da.transpose("x_y", time_dim_name)

        regional_capacity_da = regional_capacity_da.stack(
            x_y=[longitude_dim_name, latitude_dim_name]
        )

        # Remove all time series with 0 values
        regional_capfac_da = regional_capfac_da.where(regional_capacity_da > 0)
        regional_capacity_da = regional_capacity_da.where(regional_capacity_da > 0)

        # Drop NAs
        regional_capfac_da = regional_capfac_da.dropna(dim="x_y")
        regional_capacity_da = regional_capacity_da.dropna(dim="x_y")

        # Print out number of time series in the region
        n_ts = len(regional_capfac_da["x_y"].values)
        logger_tech_agg.info(f"Number of time series in {region}: {n_ts}")

        # Get power curves from capacity factor time series and capacities
        regional_power_da = regional_capacity_da * regional_capfac_da

        return regional_capfac_da, regional_capacity_da, regional_power_da

    def _preprocess_regional_xr_ds_non_gridded(region):
        """
        Private function to preprocess regional non gridded data
        """
        # Get regional data
        regional_ds = non_gridded_RE_ds.where(
            non_gridded_RE_ds[region_var_name] == region
        )

        # Rename locations dim to keep results same as _preprocess_regional_xr_ds_gridded()
        regional_ds = regional_ds.rename({location_dim_name: "x_y"})

        regional_capfac_da = regional_ds[capfac_var_name]
        ## make sure the dimensions are in the right order
        regional_capfac_da = regional_capfac_da.transpose("x_y", time_dim_name)

        regional_capacity_da = regional_ds[capacity_var_name]

        # Remove all time series with 0 values
        regional_capfac_da = regional_capfac_da.where(regional_capacity_da > 0)
        regional_capacity_da = regional_capacity_da.where(regional_capacity_da > 0)

        # Drop NAs
        regional_capfac_da = regional_capfac_da.dropna(dim="x_y")
        regional_capacity_da = regional_capacity_da.dropna(dim="x_y")

        # Print out number of time series in the region
        n_ts = len(regional_capfac_da["x_y"].values)
        logger_tech_agg.info(f"Number of time series in {region}: {n_ts}")

        # Get power curves from capacity factor time series and capacities
        regional_power_da = regional_capacity_da * regional_capfac_da

        return regional_capfac_da, regional_capacity_da, regional_power_da

    # STEP 0. Do all checks
    ## either provide gridded_RE_ds or non_gridded_RE_ds
    if gridded_RE_ds is None and non_gridded_RE_ds is None:
        raise ValueError("Either gridded_RE_ds or non_gridded_RE_ds must be passed")

    if gridded_RE_ds is not None and non_gridded_RE_ds is not None:
        logger_tech_agg.warn(
            "Both gridded_RE_ds and non_gridded_RE_ds are passed. \
            gridded_RE_ds will be used"
        )

        non_gridded_RE_ds = None

    ## gridded_RE_ds requires CRS_attr and shp_file to be set
    if gridded_RE_ds is not None:
        if CRS_attr is None:
            raise ValueError(
                "You have passed gridded_RE_ds. It requires setting CRS_attr"
            )
        elif shp_file is None:
            raise ValueError(
                "You have passed gridded_RE_ds. It requires a shapefile (shp_file) too"
            )

    # STEP 1. Rasterize the gridded dataset
    if gridded_RE_ds is not None:
        rasterized_RE_ds = techAggregationUtils.rasterize_xr_ds(
            gridded_RE_ds,
            CRS_attr,
            shp_file,
            shp_index_col,
            shp_geometry_col,
            longitude_dim_name,
            latitude_dim_name,
        )

        region_ids = rasterized_RE_ds["region_ids"].values
        n_regions = len(region_ids)

        time_steps = rasterized_RE_ds[time_dim_name].values
        n_timeSteps = len(time_steps)

    else:
        # Read in the file
        if isinstance(non_gridded_RE_ds, str):
            try:
                non_gridded_RE_ds = xr.open_dataset(non_gridded_RE_ds)
            except:
                raise FileNotFoundError("The gridded_RE_ds path specified is not valid")

        elif not isinstance(non_gridded_RE_ds, xr.Dataset):
            raise TypeError(
                "gridded_RE_ds must either be a path to a netcdf file or xarray dataset"
            )

        region_ids = np.unique(non_gridded_RE_ds[region_var_name].values)
        n_regions = len(region_ids)

        time_steps = non_gridded_RE_ds[time_dim_name].values
        n_timeSteps = len(time_steps)

    if n_timeSeries_perRegion == 1:
        # STEP 2. Create resultant xarray dataset
        ## time series
        data = np.zeros((n_timeSteps, n_regions))
        represented_timeSeries = xr.DataArray(
            data, [(time_dim_name, time_steps), ("region_ids", region_ids)]
        )

        # capacities
        data = np.zeros(n_regions)
        represented_capacities = xr.DataArray(data, [("region_ids", region_ids)])

        # STEP 3. Aggregation in every region...
        for region in region_ids:
            # Preprocess regional data
            if gridded_RE_ds != None:
                (
                    regional_capfac_da,
                    regional_capacity_da,
                    regional_power_da,
                ) = _preprocess_regional_xr_ds_gridded(region)
            else:
                (
                    regional_capfac_da,
                    regional_capacity_da,
                    regional_power_da,
                ) = _preprocess_regional_xr_ds_non_gridded(region)

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

        aggregated_timeSeries = xr.DataArray(
            data,
            [
                (time_dim_name, time_steps),
                ("region_ids", region_ids),
                ("TS_ids", TS_ids),
            ],
        )

        ## capacities
        data = np.zeros((n_regions, n_timeSeries_perRegion))
        aggregated_capacities = xr.DataArray(
            data, [("region_ids", region_ids), ("TS_ids", TS_ids)]
        )

        # ## ts groups
        cluster_labels = {}

        # STEP 3. Clustering in every region...
        for region in region_ids:
            # Preprocess regional data
            if gridded_RE_ds != None:
                (
                    regional_capfac_da,
                    regional_capacity_da,
                    regional_power_da,
                ) = _preprocess_regional_xr_ds_gridded(region)
            else:
                (
                    regional_capfac_da,
                    regional_capacity_da,
                    regional_power_da,
                ) = _preprocess_regional_xr_ds_non_gridded(region)

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

                aggregated_capacities.loc[region, TS_ids[i]] = cluster_capacity_total

                # aggregate capacity factor
                cluster_power = regional_power_da[agglomerative_model.labels_ == i]
                cluster_power_total = cluster_power.sum(dim="x_y").values
                cluster_capfac_total = cluster_power_total / cluster_capacity_total

                aggregated_timeSeries.loc[:, region, TS_ids[i]] = cluster_capfac_total

                ## lables
                cluster_labels[f"{region}.{TS_ids[i]}"] = list(
                    regional_capacity_da["x_y"].values[agglomerative_model.labels_ == i]
                )

        # STEP 4. Create resulting dataset
        regional_aggregated_RE_ds = xr.Dataset(
            {
                capacity_var_name: aggregated_capacities,
                capfac_var_name: aggregated_timeSeries,
            }
        )

        regional_aggregated_RE_ds.attrs = cluster_labels

        return regional_aggregated_RE_ds
