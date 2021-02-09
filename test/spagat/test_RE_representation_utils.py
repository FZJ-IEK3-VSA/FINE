import pytest 
import numpy as np 

import FINE.spagat.RE_representation_utils as RE_rep_utils


def test_add_shapes_from_shp(gridded_RE_data, sample_shapefile):
    #Expected
    expected_raster_reg01 = np.array([[ 1,  1,  1, np.nan, np.nan] for i in range(3)])
    expected_raster_reg02 = np.array([[np.nan, np.nan, np.nan,  1,  1] for i in range(3)])

    #Function call
    regional_RE_xr_ds = RE_rep_utils.add_shapes_from_shp(gridded_RE_data, #sds.xr_ds_wind
                                                sample_shapefile, 
                                                index_col='region_ids', 
                                                geometry_col='geometry',
                                                longitude='x', 
                                                latitude='y')
    
    #Assertion 
    np.testing.assert_equal(regional_RE_xr_ds['rasters'].loc['reg_01',:,:].values, expected_raster_reg01)
    np.testing.assert_equal(regional_RE_xr_ds['rasters'].loc['reg_02',:,:].values, expected_raster_reg02)

    

    

    

