import pytest

import spagat.representation as spr
import spagat.grouping as spg

@pytest.mark.skip()
def test_string_based_clustering():
    pass

def test_distance_based_clustering(sds):
    '''Test whether the distance-based clustering works'''

    spg.distance_based_clustering(sds, mode='hierarchical', verbose=False, ax_illustration=None, save_fig=None)
