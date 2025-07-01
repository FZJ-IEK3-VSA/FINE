import pytest
import pandas as pd
import numpy as np
import os
from .test_data import test_data_folder
from modelBuilder.inputDataHandler.demands.extract_default_demands import _get_abs_electricity_demand_per_gid1

def test__get_abs_electricty_demand_per_gid1():

    path_abs_demands = os.path.join(
        test_data_folder, 
        "input_data", 
        "demand_electricity",
        "<YEAR>",
        "absolute_electricity_demands_<YEAR>_GWh_gid1.csv"
    )

    abs_demands = _get_abs_electricity_demand_per_gid1(
        year_demand=2020,
        path_abs_demands=path_abs_demands,
    )

    assert isinstance(abs_demands, pd.DataFrame)
    assert (sorted(list(abs_demands.columns)) == ['GID_0', 'total_el_demand'])
    assert np.isclose(abs_demands.total_el_demand.sum(), 21512782.69265002) #real values
    assert np.isclose(abs_demands.total_el_demand.std(), 24959.454323469763) #real values
