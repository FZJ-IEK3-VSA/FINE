import os
import pytest

import numpy as np
import xarray as xr
import geopandas as gpd

import FINE as fn



def test_generate_iteration_dicts(minimal_test_esM):
    esm_dict, component_dict = fn.dictIO.exportToDict(minimal_test_esM)
    output_df_dict, output_series_dict = fn.xarray_io.generate_iteration_dicts(esm_dict, component_dict)

    # output_df_dict is empty  #TODO: is this a bug is fn.EnergySystemModel()? 

    assert output_series_dict.get('locationalEligibility') == [('Source', 'Electricity market'), 
                                                                ('Sink', 'Industry site'), 
                                                                ('Conversion', 'Electrolyzers'),
                                                                ('Storage', 'Pressure tank'),
                                                                ('Transmission', 'Pipelines')]

    assert output_series_dict.get('commodityRevenue') == [('Source', 'Electricity market'), 
                                                            ('Sink', 'Industry site')]

    assert output_series_dict.get('opexPerChargeOperation') == [('Storage', 'Pressure tank')]
     

def test_dimensional_data_to_xarray_dataset_multinode(multi_node_test_esM_init):

    esm_dict, component_dict = fn.dictIO.exportToDict(multi_node_test_esM_init)

    ds_extracted = fn.xarray_io.dimensional_data_to_xarray_dataset(esm_dict, component_dict)

    expected_number_of_locations = len(multi_node_test_esM_init.locations)
    actual_number_of_locations = len(ds_extracted.space)

    assert actual_number_of_locations == expected_number_of_locations


def test_dimensional_data_to_xarray_dataset_minimal(minimal_test_esM):

    esm_dict, component_dict = fn.dictIO.exportToDict(minimal_test_esM)

    ds_extracted = fn.xarray_io.dimensional_data_to_xarray_dataset(esm_dict, component_dict)

    expected_number_of_locations = len(minimal_test_esM.locations)
    actual_number_of_locations = len(ds_extracted.space)

    assert actual_number_of_locations == expected_number_of_locations

@pytest.mark.skip('Yet to be implemented')
def test_update_dicts_based_on_xarray_dataset():
    pass 

@pytest.mark.skip('Needs to be adapted to changes in the corresponding function')
def test_spatial_aggregation_multinode(multi_node_test_esM_init, solver):   #TODO: after fixing the spatial_aggregation function, rewrite this test WITH ASSERT STATEMENT 
    '''Test whether spatial aggregation of the Multi-Node Energy System Model (from examples) and subsequent optimization works'''

    shapefileFolder = os.path.join(os.path.dirname(__file__), '../../examples/Multi-regional_Energy_System_Workflow/', 
                                    'InputData/SpatialData/ShapeFiles/')

    inputShapefile = 'clusteredRegions.shp'

    esM_aggregated = multi_node_test_esM_init.spatial_aggregation(numberOfRegions=3, clusterMethod="centroid-based", shapefileFolder=shapefileFolder, inputShapefile=inputShapefile)   

    esM_aggregated.cluster(numberOfTypicalPeriods=2)
    
    esM_aggregated.optimize(timeSeriesAggregation=True, solver=solver)

    # TODO: test against results

