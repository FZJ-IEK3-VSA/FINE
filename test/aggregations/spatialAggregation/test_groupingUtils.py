import pytest

import numpy as np
import xarray as xr

import FINE.aggregations.spatialAggregation.groupingUtils as gprUtils


@pytest.mark.parametrize(
    "test_array", [np.array([[10, 9, 8], [7, 4, 6], [2, 1, 0]]), np.array([10, 5, 0])]
)
def test_get_normalized_array(test_array):

    expected_array = 0.1 * test_array

    output_array = gprUtils.get_normalized_array(test_array)

    assert np.isclose(output_array, expected_array).all()


def test_get_normalized_array_flat():

    test_array = np.array([5, 5, 5])
    expected_array = np.array([1, 1, 1])

    output_array = gprUtils.get_normalized_array(test_array)

    assert np.isclose(output_array, expected_array).all()


def test_preprocess_dataset():
    # TEST DATA
    time_list = ["T0", "T1"]
    space_list = ["01_reg", "02_reg", "03_reg"]

    ### time series
    var_ts_data = np.array([[0, 1, 1], [1, 1, 10]])

    var_ts_da = xr.DataArray(
        var_ts_data,
        coords=[time_list, space_list],
        dims=["time", "space"],
    )

    ### 2d
    var_2d_data = np.array([[0, 1, 10], [1, 0, 1], [10, 1, 0]])

    var_2d_da = xr.DataArray(
        var_2d_data,
        coords=[space_list, space_list],
        dims=["space", "space_2"],
    )

    ### 1d
    var_1d_data = np.array([0, 1, 10])

    var_1d_da = xr.DataArray(var_1d_data, coords=[space_list], dims=["space"])

    ### 0d
    var_0d_da = xr.DataArray(10, coords=[], dims=[])

    classA_compA_ds = xr.Dataset(
        {
            "ts_var": var_ts_da,
            "1d_var": var_1d_da,
            "0d_var": var_0d_da,
        }
    )

    classA_compB_ds = xr.Dataset({"ts_var": var_ts_da})

    classB_compB_ds = xr.Dataset(
        {
            "2d_var": var_2d_da,
            "0d_var": var_0d_da,
        }
    )

    test_xr_dict = {
        "ClassA": {"CompA": classA_compA_ds, "CompB": classA_compB_ds},
        "ClassB": {"CompB": classB_compB_ds},
    }

    # EXPECTED DATA
    ## time series dict
    expected_ts_dict = {}

    # ts dict
    expected_ts_dict = {}
    var_ts_data_norm = 0.1 * var_ts_data
    expected_ts_dict["ts_var"] = {"CompA": var_ts_data_norm, "CompB": var_ts_data_norm}

    # 2d dict
    expected_2d_dict = {}
    expected_2d_dict["2d_var"] = {"CompB": np.array([0.9, 0.0, 0.9])}

    ## 1d dict
    expected_1d_dict = {}
    expected_1d_data_norm = 0.1 * var_1d_data
    expected_1d_dict["1d_var"] = {"CompA": expected_1d_data_norm}

    # FUNCTION CALL
    output_ts_dict, output_1d_dict, output_2d_dict = gprUtils.preprocess_dataset(
        test_xr_dict
    )

    # ASSERTION
    ## ts
    assert output_ts_dict.keys() == expected_ts_dict.keys()
    for var in output_ts_dict.keys():
        expected_var_dict = expected_ts_dict[var]
        output_var_dict = output_ts_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys()
        for comp in output_var_dict.keys():
            assert np.isclose(
                output_var_dict.get(comp), expected_var_dict.get(comp)
            ).all()

    ## 1d
    assert output_1d_dict.keys() == expected_1d_dict.keys()
    for var in output_1d_dict.keys():
        expected_var_dict = expected_1d_dict[var]
        output_var_dict = output_1d_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys()
        for comp in output_var_dict.keys():
            assert np.isclose(
                output_var_dict.get(comp), expected_var_dict.get(comp)
            ).all()

    ## 2d
    assert output_2d_dict.keys() == expected_2d_dict.keys()
    for var in output_2d_dict.keys():
        expected_var_dict = expected_2d_dict[var]
        output_var_dict = output_2d_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys()
        for comp in output_var_dict.keys():
            assert np.isclose(
                output_var_dict.get(comp), expected_var_dict.get(comp)
            ).all()


@pytest.mark.parametrize(
    "weights, expected_dist_matrix",
    [
        # no weights are given
        (None, np.array([[0, 16, 64], [16, 0, 48], [64, 48, 0]])),
        # particular component(s), particular variable(s)
        (
            {
                "components": {"wind turbine": 2},
                "variables": ["operationRateMax", "capacityMax"],
            },
            np.array([[0, 19, 76], [19, 0, 51], [76, 51, 0]]),
        ),
        # particular component(s), all variables
        (
            {"components": {"electricity demand": 3}, "variables": "all"},
            np.array([[0, 22, 88], [22, 0, 54], [88, 54, 0]]),
        ),
        # all components, particular variable(s)
        (
            {"components": {"all": 2.5}, "variables": ["distance", "losses"]},
            np.array([[0, 22, 88], [22, 0, 102], [88, 102, 0]]),
        ),
        # skipping 'variables' key
        (
            {"components": {"AC cables": 2}},
            np.array([[0, 18, 72], [18, 0, 66], [72, 66, 0]]),
        ),
    ],
)
def test_get_custom_distance_matrix(
    weights, expected_dist_matrix, data_for_distance_measure
):

    test_ts_dict, test_1d_dict, test_2d_dict = data_for_distance_measure

    # FUNCTION CALL
    n_regions = 3
    output_dist_matrix = gprUtils.get_custom_distance_matrix(
        test_ts_dict, test_1d_dict, test_2d_dict, n_regions, weights
    )

    # ASSERTION
    assert np.isclose(expected_dist_matrix, output_dist_matrix).all()


@pytest.mark.parametrize(
    "weights",
    [
        # 'components' key not present
        {"variables": ["capacityMax", "capacityFix"]},
        # dictionary not adhering to the template
        {"variable_set": ["capacityMax", "capacityFix"], "component": "all"},
    ],
)
def test_get_custom_distance_matrix_with_unusual_weights(
    weights, data_for_distance_measure
):

    test_ts_dict, test_1d_dict, test_2d_dict = data_for_distance_measure

    # FUNCTION CALL
    n_regions = 3
    with pytest.raises(ValueError):
        output_dist_matrix = gprUtils.get_custom_distance_matrix(
            test_ts_dict, test_1d_dict, test_2d_dict, n_regions, weights
        )


def test_get_connectivity_matrix(xr_for_connectivity):
    # EXPECTED
    expected_matrix = np.array(
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1],
        ]
    )

    # FUNCTION CALL
    output_matrix = gprUtils.get_connectivity_matrix(xr_for_connectivity)

    # ASSERTION
    assert np.array_equal(output_matrix, expected_matrix)
