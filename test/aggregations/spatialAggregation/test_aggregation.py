import pytest
import numpy as np

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

    ## check total number of coords of new polygons
    ploy1 = output_xarray.values[0]
    ploy2 = output_xarray.values[1]

    assert len(ploy1.exterior.coords) == 7
    assert len(ploy2.exterior.coords) == 10


test_data = [
    (None, 3, 5, 15, 5, np.array([[0, 5], [5, 0]]), np.array([[0, 1], [1, 0]])),
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
        15,
        5,
        np.array([[0, 5], [5, 0]]),
        np.array([[0, 1], [1, 0]]),
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
        output_xarray.loc["T0", "03_reg_04_reg_05_reg"].values
        == expected_ts_operationRateMax
    )

    output_xarray = (
        output_ds_dict.get("Input").get("Sink").get("sink_comp")["ts_operationRateFix"]
    )
    assert (
        output_xarray.loc["T0", "03_reg_04_reg_05_reg"].values
        == expected_ts_operationRateFix
    )

    ## 1d variable
    output_xarray = (
        output_ds_dict.get("Input").get("Source").get("source_comp")["1d_capacityMax"]
    )
    assert output_xarray.loc["03_reg_04_reg_05_reg"].values == expected_1d_capacityMax

    output_xarray = (
        output_ds_dict.get("Input").get("Sink").get("sink_comp")["1d_capacityFix"]
    )
    assert output_xarray.loc["03_reg_04_reg_05_reg"].values == expected_1d_capacityFix

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
