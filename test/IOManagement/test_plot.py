import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from fine.IOManagement import standardIO as stdio
import geopandas as gpd
from shapely.geometry import Polygon


class MockComponentModel:
    def __init__(self, values):
        self.values = values

    def getOptimalValues(self, variableName, ip=0):
        return {
            "values": self.values
        }

class MockComponent:
    commodity = "electricity"


class MockEnergySystemModel:
    def __init__(self, values):
        self.componentNames = {
            "test_component": "SourceSinkModel"
        }
        self.componentModelingDict = {
            "SourceSinkModel": MockComponentModel(values)
        }
        self.commodityUnitsDict = {
            "electricity": "MW"
        }

        self.hoursPerTimeStep = 1

    def getComponent(self, compName):
        return MockComponent()

@pytest.fixture
def operation_esM():
    values = pd.DataFrame(
        [[1, 2, 3, 4]],
        index=pd.MultiIndex.from_tuples(
            [("test_component", "test_location")]
        )
    )

    return MockEnergySystemModel(values)


def test_plot_operation_returns_figure_and_axis(operation_esM):
    fig, ax = stdio.plotOperation(
        esM=operation_esM,
        compName="test_component",
        loc="test_location",
    )

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    assert ax.get_xlabel() == "time step"
    assert ax.get_ylabel() == "operation time series"

    plt.close(fig)

def test_plot_operation_colormap_returns_figure_and_axis(operation_esM):
    fig, ax = stdio.plotOperationColorMap(
        esM=operation_esM,
        compName="test_component",
        loc="test_location",
        nbPeriods=2,
        nbTimeStepsPerPeriod=2,
    )

    assert fig is not None
    assert ax is not None

    assert ax.get_xlabel() == "period"
    assert ax.get_ylabel() == "timestep per period"

    assert len(ax.collections) > 0

    # main axis + colorbar axis
    assert len(fig.axes) == 2

    plt.close(fig)

@pytest.fixture
def simple_shapefile(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "region": ["A", "B"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            ],
        },
        crs="EPSG:3035",
    )

    shp_path = tmp_path / "regions.shp"
    gdf.to_file(shp_path)

    return shp_path

def test_plot_locations_returns_figure_and_axis(simple_shapefile):
    fig, ax = stdio.plotLocations(
        locationsShapeFileName=simple_shapefile,
        indexColumn="region",
        plotLocNames=True,
    )

    assert fig is not None
    assert ax is not None

    assert ax.get_aspect() == 1.0
    assert not ax.axison

    assert len(ax.collections) > 0

    plt.close(fig)

@pytest.fixture
def transmission_esM():
    values = pd.DataFrame(
        [[10, 5], [0, 20]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    values = pd.concat(
        {"test_transmission": values},
        names=["Component"],
    )

    class MockTransmissionModel:
        def getOptimalValues(self, variableName, ip=0):
            return {"values": values}

    class MockTransmissionESM:
        componentNames = {"test_transmission": "TransmissionModel"}
        componentModelingDict = {"TransmissionModel": MockTransmissionModel()}

        def getComponentAttribute(self, compName, attributeName):
            return "MW"

    return MockTransmissionESM()

def test_plot_transmission_returns_figure_and_axis(transmission_esM, simple_shapefile):
    fig, ax = stdio.plotTransmission(
        esM=transmission_esM,
        compName="test_transmission",
        transmissionShapeFileName=simple_shapefile,
        loc0="region",
        loc1="region",
    )

    assert fig is not None
    assert ax is not None
    assert not ax.axison
    assert ax.get_legend() is not None

    plt.close(fig)

def test_plot_operation_colormap_returns_figure_and_axis(operation_esM):
    fig, ax = stdio.plotOperationColorMap(
        esM=operation_esM,
        compName="test_component",
        loc="test_location",
        nbPeriods=2,
        nbTimeStepsPerPeriod=2,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "period"
    assert ax.get_ylabel() == "timestep per period"
    assert len(ax.collections) > 0
    assert len(fig.axes) == 2

    plt.close(fig)

@pytest.fixture
def locational_esM():
    values = pd.Series(
        [100, 200],
        index=["A", "B"],
        name="capacity",
    )

    values = pd.concat(
        {"test_component": values},
        names=["Component"],
    )

    class MockLocationalModel:
        def getOptimalValues(self, variableName, ip=0):
            return {"values": values}

    class MockComponent:
        commodity = "electricity"

    class MockLocationalESM:
        componentNames = {"test_component": "SourceSinkModel"}

        componentModelingDict = {
            "SourceSinkModel": MockLocationalModel()
        }

        commodityUnitsDict = {
            "electricity": "MW"
        }

        def getComponent(self, compName):
            return MockComponent()

    return MockLocationalESM()

def test_plot_locational_colormap_returns_figure_and_axis(
    locational_esM,
    simple_shapefile,
):
    fig, ax = stdio.plotLocationalColorMap(
        esM=locational_esM,
        compName="test_component",
        locationsShapeFileName=simple_shapefile,
        indexColumn="region",
        perArea=False,
    )

    assert fig is not None
    assert ax is not None

    assert not ax.axison

    assert len(ax.collections) > 0

    # map axis + colorbar axis
    assert len(fig.axes) == 2

    plt.close(fig)

def test_plot_pie_chart_returns_figure_and_axis(simple_shapefile):
    index = pd.MultiIndex.from_tuples(
        [
            ("capacity", "MW", "Component 1"),
            ("capacity", "MW", "Component 2"),
        ],
        names=["Property", "Unit", "Component"],
    )

    results_df = pd.DataFrame(
        {
            "A": [10, 30],
            "B": [20, 40],
        },
        index=index,
    )

    stdio.plotPieChart(
        locFilePath=simple_shapefile,
        results_df=results_df,
        Property_to_plot="capacity",
        indexColumn_in_shp="region",
    )

    fig = plt.gcf()
    ax = plt.gca()

    assert fig is not None
    assert ax is not None
    assert ax.get_legend() is not None
    assert len(ax.patches) >= 4

    plt.close(fig)