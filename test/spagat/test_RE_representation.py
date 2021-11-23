import pytest
import numpy as np

from FINE.spagat.RE_representation import represent_RE_technology


def test_represent_RE_technology_1ts(gridded_RE_data, sample_shapefile):

    # Function call
    represented_RE_ds = represent_RE_technology(
        gridded_RE_data,
        "SRS",
        sample_shapefile,
        n_timeSeries_perRegion=1,
        capacity_var_name="capacity",
        capfac_var_name="capfac",
    )

    print(represented_RE_ds)
    # Assertion
    ## first region
    assert represented_RE_ds["capacity"].loc["reg_01"] == 9

    capfac = np.round(represented_RE_ds["capfac"].loc[:, "reg_01"], 2)
    assert np.all(capfac == 1.33)

    ## second region
    assert represented_RE_ds["capacity"].loc["reg_02"] == 6
    assert np.all(np.isclose(represented_RE_ds["capfac"].loc[:, "reg_02"], 1.5))


def test_represent_RE_technology_2ts(gridded_RE_data, sample_shapefile):

    # Expected
    expected_capfac = np.array([[1, 2] for i in range(10)])
    expected_capfac_shuffled = expected_capfac[:, [1, 0]]

    expected_capacities_reg01 = np.array([6, 3])
    expected_capacities_reg01_shuffled = expected_capacities_reg01[[1, 0]]

    expected_capacities_reg02 = np.array([3, 3])

    # Function call
    represented_RE_ds = represent_RE_technology(
        gridded_RE_data,
        "SRS",
        sample_shapefile,
        n_timeSeries_perRegion=2,
        capacity_var_name="capacity",
        capfac_var_name="capfac",
    )

    # Assertion
    # reg_01
    try:
        assert np.array_equal(
            represented_RE_ds["capfac"].loc[:, "reg_01", :], expected_capfac
        )
        assert np.array_equal(
            represented_RE_ds["capacity"].loc["reg_01", :], expected_capacities_reg01
        )

    except:
        assert np.array_equal(
            represented_RE_ds["capfac"].loc[:, "reg_01", :], expected_capfac_shuffled
        )
        assert np.array_equal(
            represented_RE_ds["capacity"].loc["reg_01", :],
            expected_capacities_reg01_shuffled,
        )

    # reg_02
    try:
        assert np.array_equal(
            represented_RE_ds["capfac"].loc[:, "reg_02", :], expected_capfac
        )
    except:
        assert np.array_equal(
            represented_RE_ds["capfac"].loc[:, "reg_02", :], expected_capfac_shuffled
        )

    assert np.array_equal(
        represented_RE_ds["capacity"].loc["reg_02", :], expected_capacities_reg02
    )
