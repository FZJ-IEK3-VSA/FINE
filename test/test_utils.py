import FINE.utils as utils
import pandas as pd
import itertools
import numpy as np

def test_transform1dSeriesto2dDataFrame():

    locations = ['loc1', 'loc2', 'loc3']

    index = ['loc1_loc2', 'loc2_loc1']
    values = np.ones(2)
    series = pd.Series(values, index=index)

    df_result = utils.transform1dSeriesto2dDataFrame(series, locations)

    df_expected = pd.DataFrame(np.zeros((len(locations), len(locations))), index=locations, columns=locations)
    df_expected.loc['loc1', 'loc2'] = 1
    df_expected.loc['loc2', 'loc1'] = 1

    pd.testing.assert_frame_equal(df_result, df_expected)