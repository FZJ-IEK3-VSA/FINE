import tempfile
from pathlib import Path
import types

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, Polygon

import fine as fn
from fine.IOManagement.standardIO import (
    plotTransmission,
    plotOperation,
    plotLocations,
    plotLocationalColorMap,
)


def build_plot_test_system():
    """Create a small real EnergySystemModel for plot tests."""
    esM = fn.EnergySystemModel(
        locations={"North", "South"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW"},
        numberOfTimeSteps=5,
        hoursPerTimeStep=1,
        costUnit="EUR",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="electricity_source",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=10.0,
            investPerCapacity=1.0,
            opexPerOperation=0.0,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="electricity_sink",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=1.0,
        )
    )

    esM.add(
        fn.Transmission(
            esM=esM,
            name="grid",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=10.0,
            investPerCapacity=1.0,
            opexPerOperation=0.0,
            distances=pd.Series(
                {
                    "North_South": 1.0,
                    "South_North": 1.0,
                }
            ),
            losses=pd.Series(
                {
                    "North_South": 0.0,
                    "South_North": 0.0,
                }
            ),
        )
    )

    return esM


def test_plotTransmission():
    esM = build_plot_test_system()

    component_model = esM.componentModelingDict[esM.componentNames["grid"]]

    transmission_values = pd.DataFrame(
        [[10.0, 5.0]],
        index=["grid"],
        columns=pd.MultiIndex.from_tuples(
            [
                ("North", "South"),
                ("South", "North"),
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
                "loc0": ["North", "South"],
                "loc1": ["South", "North"],
                "geometry": [
                    LineString([(0, 1), (0, 0)]),
                    LineString([(0, 0), (0, 1)]),
                ],
            },
            crs="EPSG:3035",
        )
        gdf.to_file(shp_path)

        fig, ax = plotTransmission(
            esM=esM,
            compName="grid",
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

        plt.close(fig)


def test_plotOperation():
    esM = build_plot_test_system()

    component_model = esM.componentModelingDict[
        esM.componentNames["electricity_source"]
    ]

    operation_values = pd.DataFrame(
        data=[[1, 2, 3, 4, 5]],
        index=pd.MultiIndex.from_tuples([("electricity_source", "North")]),
    )

    def get_optimal_values(self, variableName, ip=0):
        return {"values": operation_values}

    component_model.getOptimalValues = types.MethodType(
        get_optimal_values,
        component_model,
    )

    fig, ax = plotOperation(
        esM=esM,
        compName="electricity_source",
        loc="North",
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


@pytest.mark.parametrize(
    "plot_loc_names, index_column, expected_labels",
    [
        (False, "name", []),
        (True, "name", ["North", "South"]),
        (True, "", ["0", "1"]),
    ],
)
def test_plotLocations(plot_loc_names, index_column, expected_labels):
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = Path(tmpdir) / "locations.shp"

        gdf = gpd.GeoDataFrame(
            {
                "name": ["North", "South"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ],
            },
            crs="EPSG:4326",
        )
        gdf.to_file(shp_path)

        fig, ax = plotLocations(
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

        plt.close(fig)


@pytest.mark.parametrize(
    "per_area, area_factor",
    [
        (False, 1e3),
        (True, 1e3),
    ],
)
def test_plotLocationalColorMap(per_area, area_factor):
    esM = build_plot_test_system()

    component_model = esM.componentModelingDict[
        esM.componentNames["electricity_source"]
    ]

    capacity_values = pd.Series(
        data=[10.0, 20.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("electricity_source", "North"),
                ("electricity_source", "South"),
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
                "region": ["North", "South"],
                "geometry": [
                    Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),
                    Polygon([(2000, 0), (3000, 0), (3000, 1000), (2000, 1000)]),
                ],
            },
            crs="EPSG:3035",
        )
        gdf.to_file(shp_path)

        fig, ax = plotLocationalColorMap(
            esM=esM,
            compName="electricity_source",
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

        plt.close(fig)
