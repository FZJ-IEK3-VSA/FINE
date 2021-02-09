import pytest 
import numpy as np

import FINE.spagat.RE_representation_utils as RE_rep_utils
from FINE.spagat.RE_representation import represent_RE_technology, get_one_REtech_per_region
 

def test_represent_RE_technology(sample_shapefile, gridded_RE_data):
    #Expected 
    expected_capfac = np.array([[1, 2] for i in range(10)])
    expected_capfac_shuffled = expected_capfac[:, [1, 0]]

    expected_capacities_reg01 = np.array([6, 3])
    expected_capacities_reg01_shuffled = expected_capacities_reg01[[1, 0]]  
                                        
    expected_capacities_reg02 = np.array([3, 3])

    #Function call 
    regional_RE_xr_ds = RE_rep_utils.add_shapes_from_shp(gridded_RE_data, 
                                                sample_shapefile, 
                                                index_col='region_ids', 
                                                geometry_col='geometry',
                                                longitude='x', 
                                                latitude='y')

    represented_RE_ds = represent_RE_technology(regional_RE_xr_ds, 2)

    #Assertion 
    #reg_01 
    try:
        assert np.array_equal(represented_RE_ds['capfac'].loc[:, 'reg_01',:], expected_capfac)
        assert np.array_equal(represented_RE_ds['capacity'].loc['reg_01',:], expected_capacities_reg01)
        
    except:
        assert np.array_equal(represented_RE_ds['capfac'].loc[:, 'reg_01',:], expected_capfac_shuffled)
        assert np.array_equal(represented_RE_ds['capacity'].loc['reg_01',:], expected_capacities_reg01_shuffled)

    #reg_02
    try:
        assert np.array_equal(represented_RE_ds['capfac'].loc[:, 'reg_02',:], expected_capfac)        
    except:
        assert np.array_equal(represented_RE_ds['capfac'].loc[:, 'reg_02',:], expected_capfac_shuffled)
        
    assert np.array_equal(represented_RE_ds['capacity'].loc['reg_02',:], expected_capacities_reg02) 


def test_get_one_REtech_per_region(sample_shapefile, gridded_RE_data):

    #Function call 
    regional_RE_xr_ds = RE_rep_utils.add_shapes_from_shp(gridded_RE_data, 
                                                sample_shapefile, 
                                                index_col='region_ids', 
                                                geometry_col='geometry',
                                                longitude='x', 
                                                latitude='y')

    aggregated_RE_ds = get_one_REtech_per_region(regional_RE_xr_ds)
    
    #Assertion
    ## first region 
    assert aggregated_RE_ds['capacity'].loc['reg_01'] == 9

    capfac = np.round(aggregated_RE_ds['capfac'].loc[:, 'reg_01'], 2)
    assert np.all(capfac == 1.33)

    ## second region
    assert aggregated_RE_ds['capacity'].loc['reg_02'] == 6
    assert np.all(np.isclose(aggregated_RE_ds['capfac'].loc[:, 'reg_02'], 1.5))