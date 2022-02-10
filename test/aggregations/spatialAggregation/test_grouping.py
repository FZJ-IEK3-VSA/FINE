import os
import pytest

import numpy as np
import xarray as xr
import pandas as pd

from sklearn.datasets import make_blobs
from shapely.geometry import Point

from FINE.aggregations.spatialAggregation import grouping


@pytest.mark.parametrize(
    "string_list, expected_keys, expected_value, separator, position",
    [
        (
            ["01_es", "02_es", "01_de", "02_de", "01_nl", "01_os"],
            ["es", "de", "nl", "os"],
            ["01_es", "02_es"],
            "_",
            None,
        ),
        (
            ["abc123", "abc456", "def123", "def456"],
            ["abc", "def"],
            ["abc123", "abc456"],
            None,
            3,
        ),
        (
            ["123abc456", "456abc345", "456def123", "897def456"],
            ["abc", "def"],
            ["123abc456", "456abc345"],
            None,
            (3, 6),
        ),
    ],
)
def test_perform_string_based_grouping(
    string_list, expected_keys, expected_value, separator, position
):
    clustered_regions_dict = grouping.perform_string_based_grouping(
        string_list, separator=separator, position=position
    )

    assert sorted(clustered_regions_dict.keys()) == sorted(expected_keys)
    assert list(clustered_regions_dict.values())[0] == expected_value


def test_perform_distance_based_grouping():
    # TEST DATA
    space_list = ["01_reg", "02_reg", "03_reg", "04_reg", "05_reg"]

    sample_data, sample_labels = make_blobs(
        n_samples=5, centers=3, n_features=2, random_state=0
    )

    test_centroids = [np.nan for i in range(5)]
    for i, data_point in enumerate(sample_data):
        test_centroids[i] = Point(data_point)

    centroid_da = xr.DataArray(
        pd.Series(test_centroids).values, coords=[space_list], dims=["space"]
    )

    test_geom_xr = xr.Dataset({"centroids": centroid_da})

    # FUNCTION CALL
    output_dict = grouping.perform_distance_based_grouping(test_geom_xr)

    # ASSERTION
    assert output_dict == {
        "01_reg": ["01_reg"],  ## Based on sample_labels ([2, 0, 0, 1, 1])
        "02_reg_03_reg": ["02_reg", "03_reg"],
        "04_reg_05_reg": ["04_reg", "05_reg"],
    }


@pytest.mark.parametrize("aggregation_method", ["kmedoids_contiguity", "hierarchical"])
@pytest.mark.parametrize(
    "weights, expected_region_groups",
    [
        # no weights
        (None, ["02_reg", "03_reg"]),
        # particular components, particular variables
        (
            {
                "components": {"Wind offshore": 5, "PV": 10},
                "variables": ["capacityMax"],
            },
            ["01_reg", "02_reg"],
        ),
        # particular component, all variables
        ({"components": {"AC cables": 10}, "variables": "all"}, ["01_reg", "03_reg"]),
    ],
)
def test_perform_parameter_based_grouping(
    aggregation_method, weights, expected_region_groups, xr_for_parameter_based_grouping
):

    regions_list = xr_for_parameter_based_grouping.get("Geometry")["space"].values

    # FUNCTION CALL
    output_dict = grouping.perform_parameter_based_grouping(
        xr_for_parameter_based_grouping,
        n_groups=2,
        aggregation_method=aggregation_method,
        weights=weights,
        solver="glpk",
    )

    # ASSERTION
    for key, value in output_dict.items():
        if (
            len(value) == 2
        ):  # NOTE: required to assert separately, because they are permuted
            assert (key == "_".join(expected_region_groups)) & (
                value == expected_region_groups
            )
        else:
            remaining_region = np.setdiff1d(regions_list, expected_region_groups)
            assert (key == remaining_region.item()) & (value == remaining_region)
