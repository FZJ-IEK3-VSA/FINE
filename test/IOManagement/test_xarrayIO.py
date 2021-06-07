import os
import pytest

import numpy as np
import pandas as pd 
import xarray as xr 

from FINE.IOManagement import dictIO
import FINE.IOManagement.xarrayIO as xrIO 


def test_generateIterationDicts(minimal_test_esM):
    esm_dict, component_dict = dictIO.exportToDict(minimal_test_esM)
    output_df_dict, output_series_dict, output_constants_dict = xrIO.generateIterationDicts(component_dict)

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

 
def test_convertEsmInstanceToXarrayDataset(multi_node_test_esM_init):
    """
    Tests if conversion of esm instance to xarray dataset is correct 
    """
    #FUNCTION CALL
    output_xarray = xrIO.convertEsmInstanceToXarrayDataset(multi_node_test_esM_init)
    
    #ASSERTION 
    ## locations 
    expected_locations = list(multi_node_test_esM_init.locations).sort()
    output_locations = list(output_xarray.space.values).sort()

    assert output_locations == expected_locations

    ## commodities 
    assert output_xarray.attrs.get('commodities') == multi_node_test_esM_init.commodities

    ## time series
    expected_ts = multi_node_test_esM_init.getComponentAttribute('Wind (offshore)', 'operationRateMax').values
    output_ts = output_xarray['ts_operationRateMax'].loc['Source, Wind (offshore)', :, :].values

    assert np.array_equal(output_ts, expected_ts)

    ## 2 dimensional   
    expected_2d = multi_node_test_esM_init.getComponentAttribute('AC cables', 'capacityFix')
    output_2d = output_xarray['2d_capacityFix'].loc['LinearOptimalPowerFlow, AC cables', :, :].values
    
    assert (output_2d[0][0] == 0.0) and (output_2d[3][6] == expected_2d['cluster_3_cluster_6'])

    ## 1 dimensional   
    expected_1d = multi_node_test_esM_init.getComponentAttribute('CCGT plants (methane)', 'investPerCapacity').values
    output_1d = output_xarray['1d_investPerCapacity'].loc['Conversion, CCGT plants (methane)', :].values

    assert np.array_equal(output_1d, expected_1d)

    ## constant 
    expected_0d = multi_node_test_esM_init.getComponentAttribute('Electroylzers', 'commodityConversionFactors').get('electricity')
    output_0d = output_xarray['0d_commodityConversionFactors.electricity'].loc['Conversion, Electroylzers'].values
    
    assert output_0d == expected_0d

    ## constant bool 
    expected_0d_bool = multi_node_test_esM_init.getComponentAttribute('Li-ion batteries', 'hasCapacityVariable')
    output_0d_bool = output_xarray['0d_hasCapacityVariable'].loc['Storage, Li-ion batteries'].values
    
    assert output_0d_bool == expected_0d_bool

   
    
def test_convertXarrayDatasetToEsmInstance(multi_node_test_esM_init):
    """
    Tests if conversion of xarray dataset back to esm instance is correct 
    """
    #FUNCTION CALL 
    test_xarray = xrIO.convertEsmInstanceToXarrayDataset(multi_node_test_esM_init)
    output_esM = xrIO.convertXarrayDatasetToEsmInstance(test_xarray)

    #ASSERTION 
    ## locations 
    init_esm_locations = list(multi_node_test_esM_init.locations).sort()
    test_xarray_locations = list(test_xarray.space.values).sort()
    output_esm_locations = list(output_esM.locations).sort()

    assert init_esm_locations == test_xarray_locations == output_esm_locations

    ## commodities 
    init_esm_commodities = multi_node_test_esM_init.commodities
    test_xarray_commodities = test_xarray.attrs.get('commodities')
    output_esm_commodities = output_esM.commodities

    assert init_esm_commodities == test_xarray_commodities == output_esm_commodities

    ## a time series variable
    init_esm_ts = multi_node_test_esM_init.getComponentAttribute('Hydrogen demand', 'operationRateFix').values
    test_xarray_ts = test_xarray['ts_operationRateFix'].loc['Sink, Hydrogen demand', :, :].values
    output_esm_ts = output_esM.getComponentAttribute('Hydrogen demand', 'operationRateFix').values
    
    assert np.isclose(init_esm_ts, test_xarray_ts, output_esm_ts).all()

    ## a 2d variable
    init_esm_2d = multi_node_test_esM_init.getComponentAttribute('Pipelines (biogas)', 'locationalEligibility')
    init_esm_2d = init_esm_2d['cluster_1_cluster_5']

    test_xarray_2d = test_xarray['2d_locationalEligibility'].loc['Transmission, Pipelines (biogas)', :].values
    test_xarray_2d = test_xarray_2d[1][5]

    output_esm_2d = output_esM.getComponentAttribute('Pipelines (biogas)', 'locationalEligibility')
    output_esm_2d = output_esm_2d['cluster_1_cluster_5']

    assert init_esm_2d == test_xarray_2d == output_esm_2d

    ## a 1d variable
    init_esm_1d = multi_node_test_esM_init.getComponentAttribute('Salt caverns (hydrogen)', 'capacityMax').values
    test_xarray_1d = test_xarray['1d_capacityMax'].loc['Storage, Salt caverns (hydrogen)', :].values
    output_esm_1d = output_esM.getComponentAttribute('Salt caverns (hydrogen)', 'capacityMax').values

    assert np.isclose(init_esm_1d, test_xarray_1d, output_esm_1d).all()

    ## a constant
    init_esm_0d = multi_node_test_esM_init.getComponentAttribute('Pumped hydro storage', 'selfDischarge')
    test_xarray_0d = test_xarray['0d_selfDischarge'].loc['Storage, Pumped hydro storage'].values
    output_esm_0d = output_esM.getComponentAttribute('Pumped hydro storage', 'selfDischarge')

    assert init_esm_0d == test_xarray_0d == output_esm_0d

    ## a constant, bool 
    init_esm_0d_bool = multi_node_test_esM_init.getComponentAttribute('New CCGT plants (hydrogen)', 'hasCapacityVariable')
    test_xarray_0d_bool = test_xarray['0d_hasCapacityVariable'].loc['Conversion, New CCGT plants (hydrogen)'].values
    output_esm_0d_bool = output_esM.getComponentAttribute('New CCGT plants (hydrogen)', 'hasCapacityVariable')

    assert init_esm_0d_bool == test_xarray_0d_bool == output_esm_0d_bool

    #additionally check if otimizaiton actually runs through 
    output_esM.cluster(numberOfTypicalPeriods=3)
    output_esM.optimize(timeSeriesAggregation=True, solver = 'glpk')



@pytest.mark.parametrize("balanceLimit", [None, 

                                        pd.Series([100, 200], index=['electricity', 'hydrogen']),

                                        pd.DataFrame(np.array([[100] * 8, [50] * 8]), 
                                        columns= ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7'],
                                        index=['electricity', 'hydrogen'])
                    ])
def test_savingAndReadingNetcdfFiles(manuallyReadIn, balanceLimit, multi_node_test_esM_init):
    """
    Tests if esm instance can be saved as a netcdf file and read back in 
    to set up the instance again. 
    """

    multi_node_test_esM_init.balanceLimit = balanceLimit 

    PATH_TO_SAVE = os.path.join(os.path.dirname(__file__))
    file = os.path.join(PATH_TO_SAVE, 'esM_instance.nc4')

    # convert esm instance to xarray dataset and save a netcdf file 
    xrIO.convertEsmInstanceToXarrayDataset(multi_node_test_esM_init,
                                                           save = True, 
                                                           file_name = file)  


    #set up an esm instance directly from a netcdf file 
    output_esM = xrIO.convertXarrayDatasetToEsmInstance(file)

    o_bl = output_esM.balanceLimit 
    e_bl = multi_node_test_esM_init.balanceLimit

    assert output_esM.getComponentAttribute('Biogas purchase', 'commodity') == \
        multi_node_test_esM_init.getComponentAttribute('Biogas purchase', 'commodity')
    
    assert output_esM.getComponentAttribute('New CCGT plants (biogas)', 'commodityConversionFactors') == \
        multi_node_test_esM_init.getComponentAttribute('New CCGT plants (biogas)', 'commodityConversionFactors')
    
    assert np.array_equal(output_esM.getComponentAttribute('PV', 'capacityMax'), \
        multi_node_test_esM_init.getComponentAttribute('PV', 'capacityMax'))

    assert np.array_equal(output_esM.getComponentAttribute('Li-ion batteries', 'investPerCapacity'), \
        multi_node_test_esM_init.getComponentAttribute('Li-ion batteries', 'investPerCapacity'))

    # if there are no problems setting it up, delete the create file 
    os.remove(file)



