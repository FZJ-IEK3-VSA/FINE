import pytest

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon, MultiPolygon
from collections import namedtuple

import FINE.aggregations.spatialAggregation.managerUtils as manUtils

# ============================================Fixtures for Grouping==================================================#


@pytest.fixture()
def xr_for_connectivity():

    space_list = [
        "01_reg",
        "02_reg",
        "03_reg",
        "04_reg",
        "05_reg",
        "06_reg",
        "07_reg",
        "08_reg",
    ]
    time_list = ["T0", "T1"]

    ## ts variable data
    operationRateMax = np.array([[1] * 8 for i in range(2)])

    operationRateMax_da = xr.DataArray(
        operationRateMax,
        coords=[time_list, space_list],
        dims=["time", "space"],
    )

    ## 1d variable data
    capacityMax_1d = np.array([14] * 8)

    capacityMax_1d_da = xr.DataArray(
        capacityMax_1d, coords=[space_list], dims=["space"]
    )

    source_comp_ds = xr.Dataset(
        {
            "ts_operationRateMax": operationRateMax_da,
            "1d_capacityMax": capacityMax_1d_da,
        }
    )

    ## 2d variable data
    capacityMax_2d = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 3, 5, 0],
        ]
    )

    capacityMax_2d_da = xr.DataArray(
        capacityMax_2d,
        coords=[space_list, space_list],
        dims=["space", "space_2"],
    )

    locationalEligibility_2d = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    locationalEligibility_2d_da = xr.DataArray(
        locationalEligibility_2d,
        coords=[space_list, space_list],
        dims=["space", "space_2"],
    )

    trans_comp_ds = xr.Dataset(
        {
            "2d_capacityMax": capacityMax_2d_da,
            "2d_locationalEligibility": locationalEligibility_2d_da,
        }
    )

    input_xr_dict = {
        "Source": {"source_comp": source_comp_ds},
        "Transmission": {"trans_comp": trans_comp_ds},
    }

    # Geometries
    test_geometries = [
        Polygon([(0, 3), (1, 3), (1, 4), (0, 4)]),
        Polygon([(1, 3), (2, 3), (2, 4), (4, 1)]),
        Polygon([(2, 3), (3, 3), (3, 4), (2, 4)]),
        Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        Polygon([(1, 2), (2, 2), (2, 3), (1, 3)]),
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(2.5, 0), (3.5, 0), (3.5, 1), (2.5, 1)]),
    ]

    gdf = gpd.GeoDataFrame({"index": space_list, "geometry": test_geometries})

    geom_xr = manUtils.create_geom_xarray(gdf)

    test_ds_dict = {"Input": input_xr_dict, "Geometry": geom_xr}

    return test_ds_dict


@pytest.fixture()
def data_for_distance_measure():
    ## ts dict
    matrix_ts = np.array([[1, 2, 3], [1, 2, 3]])

    test_ts_dict = {}
    test_ts_dict["ts_operationRateMax"] = {
        "wind turbine": matrix_ts,
        "PV": matrix_ts,
    }
    test_ts_dict["ts_operationRateFix"] = {
        "electricity demand": matrix_ts,
        "hydrogen demand": matrix_ts,
    }

    array_1d_2d = np.array([1, 2, 3])

    ## 1d dict
    test_1d_dict = {}
    test_1d_dict["1d_capacityMax"] = {
        "wind turbine": array_1d_2d,
        "PV": array_1d_2d,
    }
    test_1d_dict["1d_capacityFix"] = {
        "electricity demand": array_1d_2d,
        "hydrogen demand": array_1d_2d,
    }

    ## 2d dict
    test_2d_dict = {}
    test_2d_dict["2d_distance"] = {
        "AC cables": array_1d_2d,
        "DC cables": array_1d_2d,
    }
    test_2d_dict["2d_losses"] = {
        "AC cables": array_1d_2d,
        "DC cables": array_1d_2d,
    }

    return namedtuple("test_ts_1d_2s_dicts", "test_ts_dict test_1d_dict test_2d_dict")(
        test_ts_dict, test_1d_dict, test_2d_dict
    )


@pytest.fixture()
def xr_for_parameter_based_grouping():
    time_list = ["T0", "T1"]
    space_list = ["01_reg", "02_reg", "03_reg"]

    ## Source: wind turbine
    operationRateMax = np.array([[0.2, 0.1, 0.1] for i in range(2)])
    operationRateMax = xr.DataArray(
        operationRateMax,
        coords=[time_list, space_list],
        dims=["time", "space"],
    )

    capacityMax = np.array([1, 1, 0.2])
    capacityMax = xr.DataArray(capacityMax, coords=[space_list], dims=["space"])

    wind_offshore_ds = xr.Dataset(
        {"ts_operationRateMax": operationRateMax, "1d_capacityMax": capacityMax}
    )

    ## Source, PV
    pv_ds = xr.Dataset(
        {"ts_operationRateMax": operationRateMax, "1d_capacityMax": capacityMax}
    )

    ## Transmission: AC cables
    transmissionDistance = np.array([[0, 0.2, 0.7], [0.2, 0, 0.2], [0.7, 0.2, 0]])

    transmissionDistance = xr.DataArray(
        transmissionDistance,
        coords=[space_list, space_list],
        dims=["space", "space_2"],
    )

    trans_ds = xr.Dataset({"2d_transmissionDistance": transmissionDistance})

    input_xr_dict = {
        "Source": {"Wind offshore": wind_offshore_ds, "PV": pv_ds},
        "Transmission": {"AC cables": trans_ds},
    }

    # Geometries
    test_geometries = [
        Polygon([(0, 3), (1, 3), (1, 4), (0, 4)]),
        Polygon([(1, 3), (2, 3), (2, 4), (4, 1)]),
        Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
    ]

    gdf = gpd.GeoDataFrame({"index": space_list, "geometry": test_geometries})

    geom_xr = manUtils.create_geom_xarray(gdf)

    test_ds_dict = {"Input": input_xr_dict, "Geometry": geom_xr}

    return test_ds_dict


# ============================================Fixtures for Basic Representation==================================================#


@pytest.fixture()
def xr_and_dict_for_basic_representation():
    """
    xarray to test basic representation functions-
    1. test_aggregate_based_on_sub_to_sup_region_id_dict()
    2. test_aggregate_time_series()
    3. test_aggregate_values()
    4. test_aggregate_connections()
    5. test_create_grid_shapefile()
    5. test_aggregate_geometries()
    """
    # DICT
    sub_to_sup_region_id_dict = {
        "01_reg_02_reg": ["01_reg", "02_reg"],
        "03_reg_04_reg": ["03_reg", "04_reg"],
    }

    # input data
    time_list = ["T0", "T1"]
    space_list = ["01_reg", "02_reg", "03_reg", "04_reg"]

    ## Source comp
    operationRateMax = np.array([[3, 3, 3, 3] for i in range(2)])

    operationRateMax_da = xr.DataArray(
        operationRateMax,
        coords=[time_list, space_list],
        dims=["time", "space"],
    )

    capacityMax_1d = np.array([15, 15, 0, 0])

    capacityMax_1d_da = xr.DataArray(
        capacityMax_1d, coords=[space_list], dims=["space"]
    )

    source_comp = xr.Dataset(
        {
            "ts_operationRateMax": operationRateMax_da,
            "1d_capacityMax": capacityMax_1d_da,
        }
    )

    ## Sink comp
    operationRateFix = np.array([[5, 5, 5, 5] for i in range(2)])

    operationRateFix_da = xr.DataArray(
        operationRateFix,
        coords=[time_list, space_list],
        dims=["time", "space"],
    )

    capacityFix_1d = np.array([5, 5, 5, 5])

    capacityFix_1d_da = xr.DataArray(
        capacityFix_1d, coords=[space_list], dims=["space"]
    )

    sink_comp = xr.Dataset(
        {
            "ts_operationRateFix": operationRateFix_da,
            "1d_capacityFix": capacityFix_1d_da,
        }
    )

    ## transmission comp
    capacityMax_2d = np.array([[0, 5, 5, 5], [5, 0, 5, 5], [5, 5, 0, 5], [5, 5, 5, 0]])

    capacityMax_2d_da = xr.DataArray(
        capacityMax_2d,
        coords=[space_list, space_list],
        dims=["space", "space_2"],
    )

    locationalEligibility_2d = np.array(
        [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    )

    locationalEligibility_2d_da = xr.DataArray(
        locationalEligibility_2d,
        coords=[space_list, space_list],
        dims=["space", "space_2"],
    )

    trans_comp = xr.Dataset(
        {
            "2d_capacityMax": capacityMax_2d_da,
            "2d_locationalEligibility": locationalEligibility_2d_da,
        }
    )

    input_xr_dict = {
        "Source": {"source_comp": source_comp},
        "Sink": {"sink_comp": sink_comp},
        "Transmission": {"trans_comp": trans_comp},
    }

    # geometry data
    test_geometries = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    ]

    gdf = gpd.GeoDataFrame({"index": space_list, "geometry": test_geometries})
    geom_xr = manUtils.create_geom_xarray(gdf)

    # parameter data
    parameters_ds = xr.Dataset()
    parameters_ds.attrs = {"locations": space_list}

    test_ds_dict = {
        "Input": input_xr_dict,
        "Geometry": geom_xr,
        "Parameters": parameters_ds,
    }

    return namedtuple("dict_and_xr", "sub_to_sup_region_id_dict test_xr")(
        sub_to_sup_region_id_dict, test_ds_dict
    )


# ============================================Fixtures for RE Representation==================================================#


@pytest.fixture
def gridded_RE_data(scope="session"):
    time_steps = 10
    x_coordinates = 5
    y_coordinates = 3

    time = np.arange(time_steps)
    x_locations = [1, 2, 3, 4, 5]
    y_locations = [1, 2, 3]

    # capacity factor time series
    capfac_xr_da = xr.DataArray(
        coords=[x_locations, y_locations, time], dims=["x", "y", "time"]
    )

    capfac_xr_da.loc[[1, 2, 5], :, :] = [np.full((3, 10), 1) for x in range(3)]
    capfac_xr_da.loc[3:4, :, :] = [np.full((3, 10), 2) for x in range(2)]

    # capacities
    test_data = np.ones((x_coordinates, y_coordinates))
    capacity_xr_da = xr.DataArray(
        test_data, coords=[x_locations, y_locations], dims=["x", "y"]
    )

    test_xr_ds = xr.Dataset({"capacity": capacity_xr_da, "capfac": capfac_xr_da})

    test_xr_ds.attrs["SRS"] = "epsg:3035"

    return test_xr_ds


@pytest.fixture
def non_gridded_RE_data(scope="session"):
    time_steps = 10
    n_locations = 8

    time = np.arange(time_steps)
    locations = [1, 2, 3, 4, 5, 6, 7, 8]

    # capacity factor time series
    capfac_xr_da = xr.DataArray(coords=[locations, time], dims=["locations", "time"])

    capfac_xr_da.loc[[1, 2, 5, 6], :] = np.full((4, 10), 1)
    capfac_xr_da.loc[[3, 4, 7, 8], :] = np.full((4, 10), 2)

    # capacities
    test_data = np.ones(n_locations)
    capacity_xr_da = xr.DataArray(test_data, coords=[locations], dims=["locations"])

    # regions
    test_data = [
        "region1",
        "region1",
        "region1",
        "region1",
        "region2",
        "region2",
        "region2",
        "region2",
    ]
    regions_xr_da = xr.DataArray(test_data, coords=[locations], dims=["locations"])

    test_xr_ds = xr.Dataset(
        {"capacity": capacity_xr_da, "capfac": capfac_xr_da, "region": regions_xr_da}
    )

    return test_xr_ds


@pytest.fixture
def sample_shapefile(scope="session"):
    polygon1 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    polygon2 = Polygon([(4, 0), (7, 0), (7, 4), (4, 4)])

    test_geometries = [MultiPolygon([polygon1]), MultiPolygon([polygon2])]

    df = pd.DataFrame({"region_ids": ["reg_01", "reg_02"]})

    gdf = gpd.GeoDataFrame(df, geometry=test_geometries, crs={"init": "epsg:3035"})

    return gdf
