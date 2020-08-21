import os
import pathlib

import pytest
import numpy as np
import xarray as xr

import FINE.spagat.representation as spr
import FINE.spagat.grouping as spg
import FINE.spagat.dataset as spd

import pathlib

# paths
cfd = os.path.dirname(os.path.realpath(__file__))
data_path = pathlib.Path(cfd, "data/InputData/SpatialData/")


def test_add_region_centroids_and_distances(sds):

    spr.add_region_centroids(sds)
    spr.add_centroid_distances(sds)


def test_aggregate_based_on_sub_to_sup_region_id_dict():

    # read the xr_dataset and manipulate it to obtain a test xarray dataset

    test_sds = spd.SpagatDataset()
    test_sds.read_dataset(
        sds_folder_path=pathlib.Path("tests/data/input"),
        sds_regions_filename="sds_regions.shp",
        sds_xr_dataset_filename="test_xr_dataset.nc4",
    )

    test_xr_dataset = xr.open_dataset("tests/data/input/sds_xr_dataset.nc4")
    test_xr_dataset = test_xr_dataset.where(test_xr_dataset == 1, other=1)

    # TODO: correct test data after naming decision for either 'space' or 'region_ids'
    test_xr_dataset = test_xr_dataset.rename(
        {"region_ids": "space", "region_ids_2": "space_2",}
    )
    test_sds.xr_dataset = test_sds.xr_dataset.rename(
        {"region_ids": "space", "region_ids_2": "space_2",}
    )

    # get the dictonary output by string_based_clustering function
    test_xr_dataset_dict = spg.string_based_clustering(test_xr_dataset["space"].values)

    # run the function

    aggregation_function_dict = {
        "AC_cable_incidence": ("bool", None),
        "wind cf": ("mean", "wind capacity"),
    }  # per default the aggregation function is unweighted sum, specify otherwise

    test_output_sds = spr.aggregate_based_on_sub_to_sup_region_id_dict(
        test_sds, test_xr_dataset_dict, aggregation_function_dict
    )
    test_output_xr_dataset = test_output_sds.xr_dataset

    # assert functions
    # testing aggregate_connections, with mode='bool'
    assert test_output_sds.xr_dataset["AC_cable_incidence"].loc["de", "es"].item() == 1
    # testing aggregate_connections, with mode='sum'
    assert test_output_sds.xr_dataset["AC_cable_capacity"].loc["de", "es"].item() == 25
    # testing aggregate_time_series, with mode='sum'
    assert test_output_xr_dataset["e_load"].loc["de", 8].item() == 5
    # testing aggregate_time_series, with mode='weighted mean'
    assert test_output_sds.xr_dataset["wind cf"].loc["de", 6].item() == 1
    # testing aggregate_values, with mode='sum'
    assert test_output_sds.xr_dataset["wind capacity"].loc["de"].item() == 5.0


@pytest.mark.skip(reason="test not yet implemented")
def test_aggregate_geometries():
    pass


# TODO: rename and correctly set testdata
testdata = [
    ("mean", np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0],])),
    ("sum", np.array([[0, 1, 2], [1, 0, 0], [2, 0, 0],])),
    ("bool", np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0],])),
]


@pytest.mark.parametrize("mode, expected", testdata)
def test_aggregate_connections(mode, expected):

    # TODO: improve test data (component dimension needs to be added)
    ds = xr.open_dataset("tests/data/input/sds_xr_dataset.nc4")
    ds_reduced = ds.isel(region_ids=range(5), region_ids_2=range(5))

    xr_data_array_in = ds_reduced["AC_cable_capacity"]
    xr_data_array_in = xr_data_array_in.rename(
        {"region_ids": "space", "region_ids_2": "space_2",}
    )
    data = np.array(
        [
            [0, 0, 1, 0, 0],  # es-pt
            [0, 0, 0, 1, 1],  # es-nl, es-de
            [1, 0, 0, 0, 0],  # pt-es
            [0, 1, 0, 0, 0],  # nl-es
            [0, 1, 0, 0, 0],  # de-es
        ]
    )
    # sum: es-others: 2, es-pt: 1, pt-others: 0
    # mean: 1, 1, 0
    # bool: 1, 1, 0

    sub_to_sup_region_id_dict = {
        "es": ["06_es", "11_es"],
        "pt": ["13_pt"],
        "others": ["30_nl", "31_de"],
    }

    sub_to_sup_region_id_dict = {
        "es": ["06_es", "11_es"],
        "pt": ["13_pt"],
        "others": ["30_nl", "31_de"],
    }

    xr_data_array_in.data = data
    # TODO: rename test data properly OR implement a check, whether coords are called space or not and change if not

    ds_reduced_aggregated = spr.aggregate_connections(
        xr_data_array_in,
        sub_to_sup_region_id_dict,
        mode=mode,
        set_diagonal_to_zero=True,
    )

    assert np.array_equal(ds_reduced_aggregated.data, expected)


testdata = [("mean", 5), ("weighted mean", 5), ("sum", 25)]


@pytest.mark.parametrize("mode, expected", testdata)
def test_aggregate_time_series(mode, expected):
    ds = xr.open_dataset("tests/data/input/sds_xr_dataset.nc4")

    # get the dictionary output by string_based_clustering function
    dict_ds = spg.string_based_clustering(ds["region_ids"].values)

    # A test_xr_DataArray is created with dummy values, coordinates and dimensions being region_ids and time
    region_ids = ds["region_ids"].values
    time = ds["time"].values
    temp_num = 60 * 8760
    xr_DataArray_values = np.ones(temp_num) * 5  # create array of 5s
    xr_DataArray_values = np.reshape(xr_DataArray_values, (60, 8760))

    test_xr_DataArray = xr.DataArray(
        xr_DataArray_values, coords=[region_ids, time], dims=["space", "time"]
    )
    time_series_aggregated = spr.aggregate_time_series(
        test_xr_DataArray, dict_ds, mode=mode, xr_weight_array=test_xr_DataArray
    )
    # assert function
    assert time_series_aggregated.sel(time=4, space="de").values == expected


# spagat.output
@pytest.mark.skip(reason="not yet implemented")
def test_create_grid_shapefile():
    pass
