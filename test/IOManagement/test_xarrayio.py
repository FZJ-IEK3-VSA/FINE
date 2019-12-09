import FINE 
import numpy as np
import os
import pytest
import xarray as xr
import json

from FINE.IOManagement import xarray_io


def test_dimensional_data_to_xarray_multinode(multi_node_test_esM_init):

    nc_path = os.path.join(os.path.dirname(__file__), '../data/ds_multinode.nc4')

    ds_extracted = xarray_io.dimensional_data_to_xarray(multi_node_test_esM_init)

    # ds_extracted.to_netcdf(nc_path)

    ds_expected = xr.open_dataset(nc_path)

    xr.testing.assert_allclose(ds_extracted.sortby('location'), ds_expected.sortby('location'))


def test_dimensional_data_to_xarray_minimal(minimal_test_esM):

    nc_path = os.path.join(os.path.dirname(__file__), '../data/ds_minimal.nc4')

    ds_extracted = xarray_io.dimensional_data_to_xarray(minimal_test_esM)

    # ds_extracted.to_netcdf(nc_path)

    ds_expected = xr.open_dataset(nc_path)

    xr.testing.assert_allclose(ds_extracted.sortby('location'), ds_expected.sortby('location'))