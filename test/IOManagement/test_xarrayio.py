import FINE as fn
import numpy as np
import os
import pytest
import xarray as xr
import json
import geopandas as gpd

def test_dimensional_data_to_xarray_dataset_multinode(multi_node_test_esM_init):

    nc_path = os.path.join(os.path.dirname(__file__), '../data/ds_multinode.nc4')

    esm_dict, component_dict = fn.dictIO.exportToDict(multi_node_test_esM_init)

    ds_extracted = fn.xarray_io.dimensional_data_to_xarray_dataset(esm_dict, component_dict)

    expected_number_of_locations = len(multi_node_test_esM_init.locations)
    actual_number_of_locations = len(ds_extracted.space)

    assert actual_number_of_locations == expected_number_of_locations


def test_dimensional_data_to_xarray_dataset_minimal(minimal_test_esM):

    nc_path = os.path.join(os.path.dirname(__file__), '../data/ds_minimal.nc4')

    esm_dict, component_dict = fn.dictIO.exportToDict(minimal_test_esM)

    ds_extracted = fn.xarray_io.dimensional_data_to_xarray_dataset(esm_dict, component_dict)

    expected_number_of_locations = len(minimal_test_esM.locations)
    actual_number_of_locations = len(ds_extracted.space)

    assert actual_number_of_locations == expected_number_of_locations


def test_spatial_aggregation_multinode(multi_node_test_esM_init):
    '''Test whether spatial aggregation of the Multi-Node Energy System Model (from examples) and subsequent optimization works'''

    shapefileFolder = os.path.join(os.path.dirname(__file__), '../../examples/Multi-regional Energy System Workflow/', 
                                    'InputData/SpatialData/ShapeFiles/')

    inputShapefile = 'clusteredRegions.shp'

    esM_aggregated = multi_node_test_esM_init.spatial_aggregation(numberOfRegions=3, clusterMethod="centroid-based", shapefileFolder=shapefileFolder, inputShapefile=inputShapefile)

    esM_aggregated.cluster(numberOfTypicalPeriods=2)
    
    esM_aggregated.optimize(timeSeriesAggregation=True, solver='glpk')

    # TODO: test against results

@pytest.mark.skip('not on CAESAR')
def test_spatial_aggregation_ehighway(european_model):
    '''Test whether spatial aggregation of the EuropeanModel and subsequent optimization works'''

    input_shape_path = '/home/r-beer/code/spagatti/bin/output/analysis_merra_2019-09-28_16_22_30_nocrossover_30d/initial_96/supregions.shp'
    gdf_regions = gpd.read_file(input_shape_path)


    esM_aggregated, xarray_dataset = fn.xarray_io.spatial_aggregation(esM=european_model, n_regions=3, 
                                                                      gdf_regions=gdf_regions)

    esM_aggregated.cluster(numberOfTypicalPeriods=2)
    
    # esM_aggregated.optimize(timeSeriesAggregation=True, optimizationSpecs='OptimalityTol=1e-3 method=2 cuts=0')
    esM_aggregated.optimize(timeSeriesAggregation=True, solver='glpk')

