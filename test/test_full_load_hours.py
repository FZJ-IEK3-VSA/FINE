import numpy as np


# fixture for pyomo model
# 

@pytest.mark.parametrize('time_series, full_load_hour_limit, expected', 
                         [
                            # ((np.random(), 5000), True), 
                            (np.random(8760), 8760, True)
                            (np.random(8760), 1, False)
                            # ((np.random()), True), 
                            # time series with different resolutions and # time steps
                            # capacity factor time series VS. operation time series

                         ])
def test_full_load_hour_below_limit():

    assert full_load_hour_below_limit(time_series, capacity) == expected 
    



