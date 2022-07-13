import pytest
import numpy as np

from FINE.aggregations.technologyAggregation.techAggregation import (
    aggregate_RE_technology,
)


@pytest.mark.parametrize("n_timeSeries_perRegion", [1, 2])
def test_aggregate_RE_technology_gridded(
    gridded_RE_data, sample_shapefile, n_timeSeries_perRegion
):

    # Function call
    represented_RE_ds = aggregate_RE_technology(
        gridded_RE_data,
        "SRS",
        sample_shapefile,
        n_timeSeries_perRegion=n_timeSeries_perRegion,
        capacity_var_name="capacity",
        capfac_var_name="capfac",
    )

    # Assertion
    if n_timeSeries_perRegion == 1:
        ## first region
        assert represented_RE_ds["capacity"].loc["reg_01"] == 9

        capfac = np.round(represented_RE_ds["capfac"].loc[:, "reg_01"], 2)
        assert np.all(capfac == 1.33)

        ## second region
        assert represented_RE_ds["capacity"].loc["reg_02"] == 6
        assert np.all(np.isclose(represented_RE_ds["capfac"].loc[:, "reg_02"], 1.5))

    else:

        # Expected
        expected_capfac = np.array([[1, 2] for i in range(10)])
        expected_capfac_shuffled = expected_capfac[:, [1, 0]]

        expected_capacities_reg01 = np.array([6, 3])
        expected_capacities_reg01_shuffled = expected_capacities_reg01[[1, 0]]

        expected_capacities_reg02 = np.array([3, 3])

        ## first region
        try:
            assert np.array_equal(
                represented_RE_ds["capfac"].loc[:, "reg_01", :], expected_capfac
            )
            assert np.array_equal(
                represented_RE_ds["capacity"].loc["reg_01", :],
                expected_capacities_reg01,
            )

        except:
            assert np.array_equal(
                represented_RE_ds["capfac"].loc[:, "reg_01", :],
                expected_capfac_shuffled,
            )
            assert np.array_equal(
                represented_RE_ds["capacity"].loc["reg_01", :],
                expected_capacities_reg01_shuffled,
            )

        ## second region
        try:
            assert np.array_equal(
                represented_RE_ds["capfac"].loc[:, "reg_02", :], expected_capfac
            )
        except:
            assert np.array_equal(
                represented_RE_ds["capfac"].loc[:, "reg_02", :],
                expected_capfac_shuffled,
            )

        assert np.array_equal(
            represented_RE_ds["capacity"].loc["reg_02", :], expected_capacities_reg02
        )

        ## original locations mapping
        try:
            assert represented_RE_ds.attrs["reg_02.TS_0"] == [(5, 1), (5, 2), (5, 3)]

        except:
            assert represented_RE_ds.attrs["reg_02.TS_0"] == [(4, 1), (4, 2), (4, 3)]


@pytest.mark.parametrize("n_timeSeries_perRegion", [1, 2])
def test_aggregate_RE_technology_non_gridded(
    non_gridded_RE_data, n_timeSeries_perRegion
):

    # Function call
    represented_RE_ds = aggregate_RE_technology(
        non_gridded_RE_ds=non_gridded_RE_data,
        n_timeSeries_perRegion=n_timeSeries_perRegion,
        capacity_var_name="capacity",
        capfac_var_name="capfac",
        region_var_name="region",
        location_dim_name="locations",
        time_dim_name="time",
    )

    # Assertion
    if n_timeSeries_perRegion == 1:
        expected_capfac = np.full((10, 2), 1.5)
        expected_capacities = np.array([4, 4])

        ## capfacs
        assert np.array_equal(represented_RE_ds["capfac"].values, expected_capfac)

        ## capacities
        assert np.array_equal(represented_RE_ds["capacity"].values, expected_capacities)

    else:
        # Expected
        expected_capfac = np.array([[1, 2] for i in range(10)])
        expected_capfac_shuffled = expected_capfac[:, [1, 0]]

        expected_capacities = np.full((2, 2), 2)

        ## capfacs
        for region in ["region1", "region2"]:

            try:
                assert np.array_equal(
                    represented_RE_ds["capfac"].loc[:, region, :], expected_capfac
                )

            except:
                assert np.array_equal(
                    represented_RE_ds["capfac"].loc[:, region, :],
                    expected_capfac_shuffled,
                )

        ## capacities
        assert np.array_equal(represented_RE_ds["capacity"].values, expected_capacities)

        ## original locations mapping
        try:
            assert represented_RE_ds.attrs["region1.TS_0"] == [1, 2]

        except:
            assert represented_RE_ds.attrs["region1.TS_0"] == [3, 4]
