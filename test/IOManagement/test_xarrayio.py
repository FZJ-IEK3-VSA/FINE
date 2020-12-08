import os
import pytest

import numpy as np
import xarray as xr
import geopandas as gpd

import FINE as fn

def test_generate_iteration_dicts(minimal_test_esM):
    esm_dict, component_dict = fn.dictIO.exportToDict(minimal_test_esM)
    output_df_dict, output_series_dict = fn.xarray_io.generate_iteration_dicts(esm_dict, component_dict)

    assert output_series_dict.get('locationalEligibility') == [('Source', 'Electricity market'), 
                                                                ('Sink', 'Industry site'), 
                                                                ('Conversion', 'Electrolyzers'),
                                                                ('Storage', 'Pressure tank'),
                                                                ('Transmission', 'Pipelines')]

    assert output_series_dict.get('commodityRevenue') == [('Source', 'Electricity market'), 
                                                            ('Sink', 'Industry site')]

    assert output_series_dict.get('opexPerChargeOperation') == [('Storage', 'Pressure tank')]

    assert output_df_dict == {'operationRateMax': [('Source', 'Electricity market')],
                            'commodityCostTimeSeries': [('Source', 'Electricity market')],
                            'commodityRevenueTimeSeries': [('Source', 'Electricity market')],
                            'operationRateFix': [('Sink', 'Industry site')]} 

def test_dimensional_data_to_xarray_dataset(minimal_test_esM):
    esm_dict, component_dict = fn.dictIO.exportToDict(minimal_test_esM)

    expected_locations = list(minimal_test_esM.locations).sort()
    expected_Electricitymarket_operationRateMax = \
        minimal_test_esM.getComponentAttribute('Electricity market', 'operationRateMax').values
    expected_Electrolyzers_investPerCapacity = \
        minimal_test_esM.getComponentAttribute('Electrolyzers', 'investPerCapacity').values
    
    #FUNCTION CALL
    output_xarray = fn.xarray_io.dimensional_data_to_xarray_dataset(esm_dict, component_dict)
    ## a time series variable
    output_locations = list(output_xarray.space.values).sort()
    output_Electricitymarket_operationRateMax = \
                    output_xarray.operationRateMax.loc['Source, Electricity market', 0, :, :].values
    ## a 1d variable
    output_Electrolyzers_investPerCapacity = \
        output_xarray['1d_investPerCapacity'].loc['Conversion, Electrolyzers', :].values

    ## a 2d variable
    output_Pipelines_investPerCapacity = \
        output_xarray['2d_investPerCapacity'].loc['Transmission, Pipelines', :, :].values

    #ASSERTION
    assert output_locations == expected_locations
    assert np.array_equal(output_Electricitymarket_operationRateMax, expected_Electricitymarket_operationRateMax)
    assert np.array_equal(output_Electrolyzers_investPerCapacity, expected_Electrolyzers_investPerCapacity)
    assert (output_Pipelines_investPerCapacity[0][0] == 0.0) and \
                      (output_Pipelines_investPerCapacity[0][1] == 0.0885)

    

@pytest.mark.skip('To be implemented only if the function is useful')
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

