import pytest
import numpy as np

from FINE.aggregations.technologyAggregation import techAggregationUtils


def test_rasterize_xr_ds(gridded_RE_data, sample_shapefile):
    # Expected
    expected_raster_reg01 = np.array([[1, 1, 1, np.nan, np.nan] for i in range(3)])
    expected_raster_reg02 = np.array([[np.nan, np.nan, np.nan, 1, 1] for i in range(3)])

    # Function call
    rasterized_RE_ds = techAggregationUtils.rasterize_xr_ds(
        gridded_RE_data,
        "SRS",
        sample_shapefile,
        index_col="region_ids",
        geometry_col="geometry",
        longitude="x",
        latitude="y",
    )

    # Assertion
    np.testing.assert_equal(
        rasterized_RE_ds["rasters"].loc["reg_01", :, :].values, expected_raster_reg01
    )
    np.testing.assert_equal(
        rasterized_RE_ds["rasters"].loc["reg_02", :, :].values, expected_raster_reg02
    )
