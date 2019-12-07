from FINE.IOManagement.plot import piechart_plot_function
import pytest
import geopandas as gpd
import pathlib
import geokit as gk
import os
import numpy as np
import pandas as pd


def test_piecharts_geokit(multi_node_geokit_shapes):
    """Tests whether abstract piechart function works"""

    shapes = multi_node_geokit_shapes

    columns = ['a', 'b', 'c']

    pieDataframe = pd.DataFrame(np.random.random((len(shapes.index), len(columns))),
                                          columns=columns, 
                                          index=shapes.index)

    piechart_plot_function(shapes, pieDataframe)

@pytest.mark.skip("Function not yet implemented")
def test_piecharts_gpd(multi_node_geopandas_shapes):
    """Tests whether abstract piechart function works"""

    shapes = multi_node_geopandas_shapes

    columns = ['a', 'b', 'c']

    pieDataframe = pd.DataFrame(np.random.random((len(shapes.index), len(columns))),
                                          columns=columns, 
                                          index=shapes.index)

    piechart_plot_function(shapes, pieDataframe, transmission_dataframe=None, pieColors=None, piechart_locations=None,ax=None, plot_settings=None)

@pytest.mark.skip("not yet implemented")
def test_operation_transmission_plot():
    """Tests whether abstract transmission operation function works"""

@pytest.mark.skip("not yet implemented")
def test_operational_commodity_balance_plots():
    """Tests whether abstract ... function works"""

