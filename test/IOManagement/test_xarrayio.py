import FINE 
import numpy as np
import os
import pytest
import xarray as xr
import json

from FINE.IOManagement import xarray_io as xrio


def test_dimensional_data_to_xarray_multinode(multi_node_test_esM_init):

    nc_path = os.path.join(os.path.dirname(__file__), '../data/ds_multinode.nc4')

    ds_extracted = xrio.dimensional_data_to_xarray(multi_node_test_esM_init)

    # ds_extracted.to_netcdf(nc_path)

    ds_expected = xr.open_dataset(nc_path)

    xr.testing.assert_allclose(ds_extracted.sortby('location'), ds_expected.sortby('location'))


def test_dimensional_data_to_xarray_minimal(minimal_test_esM):

    nc_path = os.path.join(os.path.dirname(__file__), '../data/ds_minimal.nc4')

    ds_extracted = xrio.dimensional_data_to_xarray(minimal_test_esM)

    # ds_extracted.to_netcdf(nc_path)

    ds_expected = xr.open_dataset(nc_path)

    xr.testing.assert_allclose(ds_extracted.sortby('location'), ds_expected.sortby('location'))


def test_spatial_aggregation(multi_node_test_esM_init):

    shapefile_folder = os.path.join(os.path.dirname(__file__), '../../examples/Multi-regional Energy System Workflow/', 
                                    'InputData/SpatialData/ShapeFiles/')

    locFilePath = os.path.join(shapefile_folder, 'clusteredRegions.shp')

    esM_aggregated = xrio.spatial_aggregation(esM=multi_node_test_esM_init, n_regions=3, 
                                                locFilePath=locFilePath,
                                                aggregatedShapefileFolderPath=shapefile_folder)
    # TODO: add aggregatedShapefileFolderPath as temporary path to save shapefiles temporarily
    # TODO: test whether shapefiles are similar
    # TODO: test whether optimization initializes
    # TODO: test whether results are the same 
    # TODO: reimplement test, with updated aggregation_function_dict 
    # (ATTENTION: due to simplified SPAGAT aggregation functions, results are not yet correct)

    esM_aggregated.cluster(numberOfTypicalPeriods=7)
    
    esM_aggregated.optimize(timeSeriesAggregation=True, optimizationSpecs='OptimalityTol=1e-3 method=2 cuts=0')

    aggregated_regions_FilePath = os.path.join(shapefile_folder, 'aggregated_regions.shp')
    aggregated_grid_FilePath = os.path.join(shapefile_folder, 'aggregated_grid.shp')

    # assert gdf_results == gdf_expected