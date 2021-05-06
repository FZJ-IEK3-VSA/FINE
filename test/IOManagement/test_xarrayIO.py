import os
import pytest

import numpy as np
import xarray as xr
import geopandas as gpd

import FINE as fn

from FINE.IOManagement import dictIO
import FINE.IOManagement.xarrayIO as xrIO 


def test_generate_iteration_dicts(minimal_test_esM):
    esm_dict, component_dict = dictIO.exportToDict(minimal_test_esM)
    output_df_dict, output_series_dict, output_constants_dict = xrIO.generate_iteration_dicts(component_dict)

    # check output_df_dict
    assert output_df_dict == {'operationRateMax': [('Source', 'Electricity market')],
                            'commodityCostTimeSeries': [('Source', 'Electricity market')],
                            'commodityRevenueTimeSeries': [('Source', 'Electricity market')],
                            'operationRateFix': [('Sink', 'Industry site')]} 

    # check output_series_dict
    assert output_series_dict.get('commodityRevenue') == [('Source', 'Electricity market'), 
                                                            ('Sink', 'Industry site')]

    assert output_series_dict.get('opexPerChargeOperation') == [('Storage', 'Pressure tank')]

    # check output_constants_dict
    assert len(output_constants_dict.get('name')) == 5 # 5 components present in the esM instance 
    assert output_constants_dict.get('physicalUnit') == [('Conversion', 'Electrolyzers')] 
    assert output_constants_dict.get('commodityConversionFactors.electricity') == [('Conversion', 'Electrolyzers')] 

    # In some cases, when a varaible is not present, it is None and is captured by output_constants_dict
    assert set(output_df_dict.get('operationRateFix')).isdisjoint(output_constants_dict.get('operationRateFix')) 

    
def test_convert_esM_instance_to_xarray_dataset(minimal_test_esM):
   
    expected_locations = list(minimal_test_esM.locations).sort()

    expected_Electricitymarket_operationRateMax = \
        minimal_test_esM.getComponentAttribute('Electricity market', 'operationRateMax').values
        
    expected_Electrolyzers_investPerCapacity = \
        minimal_test_esM.getComponentAttribute('Electrolyzers', 'investPerCapacity').values

    expected_Electrolyzers_commodityConversionFactors_electricity = \
        minimal_test_esM.getComponentAttribute('Electrolyzers', 'commodityConversionFactors').get('electricity')
    
    expected_Electricitymarket_hasCapacityVariable = \
        minimal_test_esM.getComponentAttribute('Electricity market', 'hasCapacityVariable')

    #FUNCTION CALL
    output_xarray = xrIO.convert_esM_instance_to_xarray_dataset(minimal_test_esM)

    ## a time series variable
    output_locations = list(output_xarray.space.values).sort()
    output_Electricitymarket_operationRateMax = \
                    output_xarray['ts_operationRateMax'].loc['Source, Electricity market', :, :].values
    ## a 1d variable
    output_Electrolyzers_investPerCapacity = \
        output_xarray['1d_investPerCapacity'].loc['Conversion, Electrolyzers', :].values

    ## a 2d variable
    output_Pipelines_investPerCapacity = \
        output_xarray['2d_investPerCapacity'].loc['Transmission, Pipelines', :, :].values

    ## a constant
    output_Electrolyzers_commodityConversionFactors_electricity = \
        output_xarray['0d_commodityConversionFactors.electricity'].loc['Conversion, Electrolyzers'].values

    ## a constant, bool 
    output_Electricitymarket_hasCapacityVariable = \
        output_xarray['0d_hasCapacityVariable'].loc['Source, Electricity market'].values

    #ASSERTION
    assert output_locations == expected_locations
    assert np.array_equal(output_Electricitymarket_operationRateMax, expected_Electricitymarket_operationRateMax)
    assert np.array_equal(output_Electrolyzers_investPerCapacity, expected_Electrolyzers_investPerCapacity)
    assert (output_Pipelines_investPerCapacity[0][0] == 0.0) and \
                      (output_Pipelines_investPerCapacity[0][1] == 0.0885)
    assert output_Electrolyzers_commodityConversionFactors_electricity == expected_Electrolyzers_commodityConversionFactors_electricity
    
    assert output_xarray.attrs.get('commodities') == minimal_test_esM.commodities

    assert output_Electricitymarket_hasCapacityVariable == expected_Electricitymarket_hasCapacityVariable

def test_convert_xarray_dataset_to_esM_instance(multi_node_test_esM_init):
    
    #FUNCTION CALL
    output_xarray = xrIO.convert_esM_instance_to_xarray_dataset(multi_node_test_esM_init)
    esM = xrIO.convert_xarray_dataset_to_esM_instance(output_xarray)

    # EXPECTED 
    expected_locations = list(output_xarray.space.values).sort()
    expected_opexPerCapacity_Windoffshore = \
        sorted(output_xarray['1d_opexPerCapacity'].loc['Source, Wind (offshore)', :].values)
    expected_Electrolyzers_commodityConversionFactors_electricity= \
        sorted(output_xarray['0d_commodityConversionFactors.electricity'].loc['Conversion, Electrolyzers'].values)
        
    # OUTPUT
    output_locations = esM.locations
    output_opexPerCapacity_Windoffshore = \
        minimal_test_esM.getComponentAttribute('Wind (offshore)', 'opexPerCapacity').values
    output_Electrolyzers_commodityConversionFactors_electricity = \
        minimal_test_esM.getComponentAttribute('Electrolyzers', 'commodityConversionFactors').get('electricity')

    
    #ASSERTION 
    assert output_locations == expected_locations
    assert output_opexPerCapacity_Windoffshore == expected_opexPerCapacity_Windoffshore
    assert output_Electrolyzers_commodityConversionFactors_electricity == expected_Electrolyzers_commodityConversionFactors_electricity
