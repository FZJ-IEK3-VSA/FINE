import matplotlib as mpl

mpl.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import fine as fn
from fine.utils import ImplementedSolvers
from shapely.geometry import Polygon


def test_plot_operation_returns_figure_and_axis(minimal_test_esM):
    esM = minimal_test_esM

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    fig, ax = fn.plotOperation(
        esM,
        "Electrolyzers",
        "ElectrolyzerLocation",
    )

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    assert ax.get_xlabel() == "time step"

    plt.close(fig)


def test_plot_operation_colormap_returns_figure_and_axis(minimal_test_esM):
    esM = minimal_test_esM

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    fig, ax = fn.plotOperationColorMap(
        esM,
        "Electrolyzers",
        "ElectrolyzerLocation",
        figsize=(4, 3),
        nbTimeStepsPerPeriod=1,
        nbPeriods=4,
        yticks=[0, 1],
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "period"
    assert ax.get_ylabel() == "timestep per period"

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
    fig, ax = fn.plotLocations(
        locationsShapeFileName=simple_shapefile,
        indexColumn="region",
        plotLocNames=True,
    )

    assert fig is not None
    assert ax is not None
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
        """Mock transmission model for plotting tests."""

        def getOptimalValues(self, variableName, ip=0):
            """Return mocked optimal values."""
            return {"values": values}

    class MockTransmissionESM:
        """Mock energy system model for transmission plotting tests."""

        componentNames = {"test_transmission": "TransmissionModel"}
        componentModelingDict = {"TransmissionModel": MockTransmissionModel()}

        def getComponentAttribute(self, compName, attributeName):
            """Return mocked component attribute."""
            return "MW"

    return MockTransmissionESM()


def test_plot_transmission_returns_figure_and_axis(transmission_esM, simple_shapefile):
    fig, ax = fn.plotTransmission(
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
        """Mock locational model for plotting tests."""

        def getOptimalValues(self, variableName, ip=0):
            """Return mocked optimal values."""
            return {"values": values}

    class MockLocationalComponent:
        """Mock locational component for plotting tests."""

        commodity = "electricity"

    class MockLocationalESM:
        """Mock energy system model for locational plotting tests."""

        componentNames = {"test_component": "SourceSinkModel"}
        componentModelingDict = {"SourceSinkModel": MockLocationalModel()}
        commodityUnitsDict = {"electricity": "MW"}

        def getComponent(self, compName):
            """Return mocked component."""
            return MockLocationalComponent()

    return MockLocationalESM()


def test_plot_locational_colormap_returns_figure_and_axis(
    locational_esM,
    simple_shapefile,
):
    fig, ax = fn.plotLocationalColorMap(
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

    fn.plotPieChart(
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
