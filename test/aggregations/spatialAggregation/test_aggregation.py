import pytest
import numpy as np
import xarray as xr

from FINE.aggregations.spatialAggregation import aggregation


def test_aggregate_geometries(xr_and_dict_for_basic_representation):

    (
        sub_to_sup_region_id_dict,
        xr_for_basic_representation,
    ) = xr_and_dict_for_basic_representation

    test_xarray = xr_for_basic_representation.get("Geometry")["geometries"]

    # FUNCTION CALL
    output_xarray = aggregation.aggregate_geometries(
        test_xarray, sub_to_sup_region_id_dict
    )

    # ASSERTION
    assert list(output_xarray.space.values) == list(sub_to_sup_region_id_dict.keys())

    for (
        joined_polygon
    ) in (
        output_xarray.values
    ):  # NOTE: Only length and geom_type of the resutling polygons
        # are tested. Even in shapely source code only geom_type is checked!
        # Need to unpermute the resulting coordinates in order to assert
        # against expected coordinates.

        assert len(joined_polygon.exterior.coords) == 7
        assert joined_polygon.geom_type in ("MultiPolygon")


@pytest.mark.parametrize("mode, expected", [("mean", 3), ("sum", 6)])
def test_aggregate_time_series_mean_and_sum(
    xr_and_dict_for_basic_representation, mode, expected
):

    (
        sub_to_sup_region_id_dict,
        xr_for_basic_representation,
    ) = xr_and_dict_for_basic_representation

    test_ds = xr_for_basic_representation.get("Input").get("Source").get("source_comp")
    test_xarray = test_ds["ts_operationRateMax"]

    # FUNCTION CALL
    time_series_aggregated = aggregation.aggregate_time_series_spatially(
        test_xarray, sub_to_sup_region_id_dict, mode=mode
    )

    # ASSERTION
    assert time_series_aggregated.loc["T0", "01_reg_02_reg"].values == expected


@pytest.mark.parametrize(
    "data, weight, expected_grp1, expected_grp2",
    [
        # all non zero values
        (
            np.array([[3, 3, 3, 3] for i in range(2)]),
            np.array([3, 3, 3, 3]),
            3,
            3,
        ),
        # all zero values in one region group
        (
            np.array([[3, 3, 0, 0] for i in range(2)]),
            np.array([3, 3, 0, 0]),
            3,
            0,
        ),
        # all zero values for in one region
        (
            np.array([[3, 3, 3, 0] for i in range(2)]),
            np.array([3, 3, 3, 0]),
            3,
            3,
        ),
    ],
)
def test_aggregate_time_series_weighted_mean(
    data, weight, expected_grp1, expected_grp2
):

    space_list = ["01_reg", "02_reg", "03_reg", "04_reg"]
    time_list = ["T0", "T1"]

    data_xr = xr.DataArray(
        data,
        coords=[time_list, space_list],
        dims=["time", "space"],
    )

    weight_xr = xr.DataArray(weight, coords=[space_list], dims=["space"])

    sub_to_sup_region_id_dict = {
        "01_reg_02_reg": ["01_reg", "02_reg"],
        "03_reg_04_reg": ["03_reg", "04_reg"],
    }

    # FUNCTION CALL
    time_series_aggregated = aggregation.aggregate_time_series_spatially(
        data_xr,
        sub_to_sup_region_id_dict,
        mode="weighted mean",
        xr_weight_array=weight_xr,
    )

    # ASSERTION
    assert time_series_aggregated.loc["T0", "01_reg_02_reg"].values == expected_grp1
    assert time_series_aggregated.loc["T0", "03_reg_04_reg"].values == expected_grp2


@pytest.mark.parametrize(
    "mode, expected_grp1, expected_grp2",
    [("mean", 15, 0), ("sum", 30, 0), ("bool", 1, 0)],
)
def test_aggregate_values_spatially(
    xr_and_dict_for_basic_representation, mode, expected_grp1, expected_grp2
):

    (
        sub_to_sup_region_id_dict,
        xr_for_basic_representation,
    ) = xr_and_dict_for_basic_representation

    test_ds = xr_for_basic_representation.get("Input").get("Source").get("source_comp")
    test_xarray = test_ds["1d_capacityMax"]

    # FUNCTION CALL
    values_aggregated = aggregation.aggregate_values_spatially(
        test_xarray, sub_to_sup_region_id_dict, mode=mode
    )

    # ASSERTION
    assert values_aggregated.loc["01_reg_02_reg"].values == expected_grp1
    assert values_aggregated.loc["03_reg_04_reg"].values == expected_grp2


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("mean", np.array([[0, 5], [5, 0]])),
        ("sum", np.array([[0, 20], [20, 0]])),
        ("bool", np.array([[0, 1], [1, 0]])),
    ],
)
def test_aggregate_connections(xr_and_dict_for_basic_representation, mode, expected):

    (
        sub_to_sup_region_id_dict,
        xr_for_basic_representation,
    ) = xr_and_dict_for_basic_representation

    test_ds = (
        xr_for_basic_representation.get("Input").get("Transmission").get("trans_comp")
    )
    test_xarray = test_ds["2d_capacityMax"]

    # FUNCTION CALL
    connections_aggregated = aggregation.aggregate_connections(
        test_xarray, sub_to_sup_region_id_dict, mode=mode
    )

    # ASSERTION
    assert np.array_equal(connections_aggregated, expected)


test_data = [
    (None, 3, 5, 15, 5, np.array([[0, 5], [5, 0]]), np.array([[0, 1], [0, 0]])),
    (
        {
            "operationRateMax": ("weighted mean", "capacityMax"),
            "operationRateFix": ("mean", None),
            "capacityMax": ("sum", None),
            "capacityFix": ("sum", None),
            "locationalEligibility": ("bool", None),
        },
        3,
        5,
        30,
        10,
        np.array([[0, 20], [20, 0]]),
        np.array([[0, 1], [0, 0]]),
    ),
]


@pytest.mark.parametrize(
    "aggregation_function_dict, expected_ts_operationRateMax, \
                          expected_ts_operationRateFix, expected_1d_capacityMax, \
                          expected_1d_capacityFix, expected_2d_capacityMax, \
                          expected_2d_locationalEligibility",
    test_data,
)
def test_aggregate_based_on_sub_to_sup_region_id_dict(
    xr_and_dict_for_basic_representation,
    aggregation_function_dict,
    expected_ts_operationRateMax,
    expected_ts_operationRateFix,
    expected_1d_capacityMax,
    expected_1d_capacityFix,
    expected_2d_capacityMax,
    expected_2d_locationalEligibility,
):

    sub_to_sup_region_id_dict, test_xr = xr_and_dict_for_basic_representation

    output_ds_dict = aggregation.aggregate_based_on_sub_to_sup_region_id_dict(
        test_xr,
        sub_to_sup_region_id_dict,
        aggregation_function_dict=aggregation_function_dict,
    )

    # ASSERTION
    ## Time series variables
    output_xarray = (
        output_ds_dict.get("Input")
        .get("Source")
        .get("source_comp")["ts_operationRateMax"]
    )
    assert (
        output_xarray.loc["T0", "01_reg_02_reg"].values == expected_ts_operationRateMax
    )

    output_xarray = (
        output_ds_dict.get("Input").get("Sink").get("sink_comp")["ts_operationRateFix"]
    )
    assert (
        output_xarray.loc["T0", "01_reg_02_reg"].values == expected_ts_operationRateFix
    )

    ## 1d variable
    output_xarray = (
        output_ds_dict.get("Input").get("Source").get("source_comp")["1d_capacityMax"]
    )
    assert output_xarray.loc["01_reg_02_reg"].values == expected_1d_capacityMax

    output_xarray = (
        output_ds_dict.get("Input").get("Sink").get("sink_comp")["1d_capacityFix"]
    )
    assert output_xarray.loc["01_reg_02_reg"].values == expected_1d_capacityFix

    ## 2d variable
    output_xarray = (
        output_ds_dict.get("Input")
        .get("Transmission")
        .get("trans_comp")["2d_capacityMax"]
    )
    assert np.array_equal(output_xarray.values, expected_2d_capacityMax)

    output_xarray = (
        output_ds_dict.get("Input")
        .get("Transmission")
        .get("trans_comp")["2d_locationalEligibility"]
    )
    assert np.array_equal(output_xarray.values, expected_2d_locationalEligibility)
