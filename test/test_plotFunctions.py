import tempfile
from pathlib import Path
import types

import fine as fn
import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, Polygon

from fine.utils import ImplementedSolvers


def test_plotTransmission(minimal_test_esM):
    esM = minimal_test_esM

    comp_name = "Pipelines"
    loc0 = "ElectrolyzerLocation"
    loc1 = "IndustryLocation"

    component_model = esM.componentModelingDict[esM.componentNames[comp_name]]

    transmission_values = pd.DataFrame(
        [[10.0, 5.0]],
        index=[comp_name],
        columns=pd.MultiIndex.from_tuples(
            [
                (loc0, loc1),
                (loc1, loc0),
            ]
        ),
    )

    def get_optimal_values(self, variableName, ip=0):
        return {"values": transmission_values}

    component_model.getOptimalValues = types.MethodType(
        get_optimal_values,
        component_model,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = Path(tmpdir) / "transmission.shp"

        gdf = gpd.GeoDataFrame(
            {
                "loc0": [loc0, loc1],
                "loc1": [loc1, loc0],
                "geometry": [
                    LineString([(0, 1), (0, 0)]),
                    LineString([(0, 0), (0, 1)]),
                ],
            },
            crs="EPSG:3035",
        )
        gdf.to_file(shp_path)

        fig, ax = fn.plotTransmission(
            esM=esM,
            compName=comp_name,
            transmissionShapeFileName=str(shp_path),
            loc0="loc0",
            loc1="loc1",
            linewidth=12,
            save=False,
        )

        plotted_lines = [
            child for child in ax.get_children() if isinstance(child, LineCollection)
        ]

        detected_widths = []
        for line in plotted_lines:
            detected_widths.extend(line.get_linewidths())

        assert fig is not None
        assert ax is not None
        assert len(plotted_lines) > 0
        assert any(width > 0 for width in detected_widths)
        assert ax.get_legend() is not None

        plt.close(fig)


def test_plotOperation(minimal_test_esM):
    esM = minimal_test_esM

    comp_name = "Electricity market"
    loc = "ElectrolyzerLocation"

    component_model = esM.componentModelingDict[esM.componentNames[comp_name]]

    operation_values = pd.DataFrame(
        data=[[1, 2, 3, 4]],
        index=pd.MultiIndex.from_tuples([(comp_name, loc)]),
    )

    def get_optimal_values(self, variableName, ip=0):
        return {"values": operation_values}

    component_model.getOptimalValues = types.MethodType(
        get_optimal_values,
        component_model,
    )

    fig, ax = fn.plotOperation(
        esM=esM,
        compName=comp_name,
        loc=loc,
        tMin=1,
        tMax=4,
        xlabel="test x label",
        ylabel="test y label",
        save=False,
    )

    plotted_line = ax.lines[0]

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    assert list(plotted_line.get_ydata()) == [2, 3, 4]
    assert ax.get_xlabel() == "test x label"
    assert ax.get_ylabel() == "test y label"

    plt.close(fig)

    default_fig, default_ax = fn.plotOperation(
        esM=esM,
        compName=comp_name,
        loc=loc,
        save=False,
    )

    assert default_ax.get_xlabel() == "time step"

    plt.close(default_fig)


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


@pytest.mark.parametrize(
    "plot_loc_names, index_column, expected_labels",
    [
        (False, "name", []),
        (True, "name", ["ElectrolyzerLocation", "IndustryLocation"]),
        (True, "", ["0", "1"]),
    ],
)
def test_plotLocations(plot_loc_names, index_column, expected_labels):
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = Path(tmpdir) / "locations.shp"

        gdf = gpd.GeoDataFrame(
            {
                "name": ["ElectrolyzerLocation", "IndustryLocation"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            },
            crs="EPSG:4326",
        )
        gdf.to_file(shp_path)

        fig, ax = fn.plotLocations(
            locationsShapeFileName=str(shp_path),
            indexColumn=index_column,
            plotLocNames=plot_loc_names,
            crs="EPSG:3035",
            faceColor="none",
            edgeColor="black",
            linewidth=0.5,
            figsize=(6, 6),
            fontsize=12,
            save=False,
        )

        actual_labels = [text.get_text() for text in ax.texts]

        assert fig is not None
        assert ax is not None
        assert actual_labels == expected_labels
        assert ax.axison is False
        assert len(ax.collections) > 0

        plt.close(fig)


@pytest.mark.parametrize(
    "per_area, area_factor",
    [
        (False, 1e3),
        (True, 1e3),
    ],
)
def test_plotLocationalColorMap(minimal_test_esM, per_area, area_factor):
    esM = minimal_test_esM

    comp_name = "Electrolyzers"

    component_model = esM.componentModelingDict[esM.componentNames[comp_name]]

    capacity_values = pd.Series(
        data=[10.0, 20.0],
        index=pd.MultiIndex.from_tuples(
            [
                (comp_name, "ElectrolyzerLocation"),
                (comp_name, "IndustryLocation"),
            ]
        ),
    )

    def get_optimal_values(self, variableName, ip=0):
        return {"values": capacity_values}

    component_model.getOptimalValues = types.MethodType(
        get_optimal_values,
        component_model,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = Path(tmpdir) / "locations.shp"

        gdf = gpd.GeoDataFrame(
            {
                "region": ["ElectrolyzerLocation", "IndustryLocation"],
                "geometry": [
                    Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),
                    Polygon([(2000, 0), (3000, 0), (3000, 1000), (2000, 1000)]),
                ],
            },
            crs="EPSG:3035",
        )
        gdf.to_file(shp_path)

        fig, ax = fn.plotLocationalColorMap(
            esM=esM,
            compName=comp_name,
            locationsShapeFileName=str(shp_path),
            indexColumn="region",
            perArea=per_area,
            areaFactor=area_factor,
            crs="EPSG:3035",
            variableName="capacityVariablesOptimum",
            doSum=False,
            zlabel="Test label",
            save=False,
        )

        assert fig is not None
        assert ax is not None
        assert ax.axison is False
        assert len(fig.axes) == 2
        assert len(ax.collections) > 0

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
