from FINE.IOManagement.plot import piechart_plot_function
import pytest
import geopandas as gpd
import pathlib

try:
    import geokit as gk
except ImportError:
    print("Geokit not available.")


import os
import numpy as np
import pandas as pd


def test_piecharts_geokit():
    """Tests whether abstract piechart function works"""

    shape_file_path = os.path.join(os.path.join(os.path.dirname(__file__), 
        "../examples/Multi-regional Energy System Workflow/InputData/SpatialData/ShapeFiles/clusteredRegions.shp"))

    geokit_shapes = gk.vector.extractFeatures(shape_file_path)
    geokit_shapes.set_index('index', inplace=True)

    
    shapes = geokit_shapes

    columns = ['a', 'b', 'c']

    pieDataframe = pd.DataFrame(np.random.random((len(shapes.index), len(columns))),
                                          columns=columns, 
                                          index=shapes.index)

    piechart_plot_function(shapes, pieDataframe)

@pytest.mark.skip("Function not yet implemented")
def test_piecharts_gpd(multi_node_geopandas_shapes):
    """Tests whether abstract piechart function works"""

    shape_file_path = os.path.join(os.path.join(os.path.dirname(__file__), 
        "../examples/Multi-regional Energy System Workflow/InputData/SpatialData/ShapeFiles/clusteredRegions.shp"))

    geokit_shapes = gpd.read_file(shape_file_path)
    geokit_shapes.set_index('index', inplace=True)

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

