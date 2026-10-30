"""Create the two-region input data of the modular examples.

The modular examples (``examples/Examples``) use a small model of Germany with two
regions, ``north`` and ``south``. Its data is derived from the multi-region data
in ``examples/data/multiRegion``, in which Germany is divided into eight regions
(``cluster_0`` to ``cluster_7``). This script merges these eight regions into two
and writes the result into this folder:

====================================  ===================================================
File                                  Content
====================================  ===================================================
``windCapacityFactor.csv``            Hourly capacity factors of onshore wind turbines [-]
``offshoreWindCapacityFactor.csv``    Hourly capacity factors of offshore wind turbines [-]
``pvCapacityFactor.csv``              Hourly capacity factors of PV [-]
``runOfRiverProfile.csv``             Hourly operation of run-of-river plants, relative
                                      to their capacity [-]
``electricityDemand.csv``             Hourly electricity demand [GW_el]
``hydrogenDemand.csv``                Hourly hydrogen demand [GW_H2]
``capacityPotential.csv``             Maximum capacity of onshore and offshore wind
                                      turbines and PV [GW_el] and of hydrogen storage in
                                      salt caverns [GWh_H2]
``existingCapacity.csv``              Existing run-of-river plants and gas power plants
                                      [GW_el] and pumped hydro storage [GWh_el]
``existingCapacityACcables.csv``      Existing AC cables between the regions [GW_el]
``existingCapacityDCcables.csv``      DC cables between the regions [GW_el]
``distances.csv``                     Distance between the centers of the regions [km]
``regions.geojson``                   Shapes of the regions (for map plots)
``lines.geojson``                     Straight lines between the region centers (for map
                                      plots)
====================================  ===================================================

The time series have one row per hour of a year (8760 rows, index 0 to 8759) and
one column per region. In a region, a capacity factor (or the run-of-river profile)
is the mean of the values of its original regions, weighted with their capacity.
Demands and capacities are added up. Cables count if they connect an original
region in the north with one in the south.

Run the script from any folder to create the files again::

    python createTwoRegionData.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

OUTPUT_PATH = Path(__file__).parent.absolute()
sys.path.insert(0, str(OUTPUT_PATH.parents[1]))

from exampleData import MULTI_REGION_DATA_PATH, getMultiRegionData  # noqa: E402

#: The original regions that form each of the two new regions.
REGIONS = {
    "north": ["cluster_1", "cluster_2", "cluster_3", "cluster_5", "cluster_6"],
    "south": ["cluster_0", "cluster_4", "cluster_7"],
}

#: Coordinate reference system for areas and distances (in meters).
CRS = "EPSG:3035"


def weightedMean(timeSeries, weights):
    """Merge the columns of a time series per new region, weighted with ``weights``.

    A new region without any weight (e.g. no offshore wind in the south) gets 0.
    """
    return pd.DataFrame(
        {
            region: (timeSeries[clusters] * weights[clusters]).sum(axis=1)
            / weights[clusters].sum()
            if weights[clusters].sum() > 0
            else 0.0
            for region, clusters in REGIONS.items()
        }
    )


def total(data):
    """Add up the columns of a time series (or the entries of a Series) per new region."""
    if isinstance(data, pd.Series):
        return pd.Series(
            {region: data[clusters].sum() for region, clusters in REGIONS.items()}
        )
    return pd.DataFrame(
        {region: data[clusters].sum(axis=1) for region, clusters in REGIONS.items()}
    )


def betweenRegions(matrix):
    """Add up a matrix of connections between original regions per pair of new regions."""
    names = list(REGIONS)
    return pd.DataFrame(
        [
            [
                matrix.loc[REGIONS[a], REGIONS[b]].values.sum() if a != b else 0.0
                for b in names
            ]
            for a in names
        ],
        index=names,
        columns=names,
    )


def writeTimeSeries(data, fileName, decimals):
    """Write a time series with ``decimals`` decimal places into this folder."""
    data = data.round(decimals)
    data.index.name = "hour"
    data.to_csv(OUTPUT_PATH / fileName)


def writeTable(rows, fileName, indexName):
    """Write a table with one row per entry of ``rows`` and one column per region."""
    table = pd.DataFrame(rows).T.round(1)
    table.index.name = indexName
    table.to_csv(OUTPUT_PATH / fileName)


def main():
    """Create all files of the two-region data."""
    data = getMultiRegionData()

    onshoreCapacityMax = data["Wind (onshore), capacityMax"]
    offshoreCapacityMax = data["Wind (offshore), capacityMax"]
    pvCapacityMax = data["PV, capacityMax"]
    runOfRiverCapacity = data["Existing run-of-river plants, capacityFix"]

    # Time series
    writeTimeSeries(
        weightedMean(data["Wind (onshore), operationRateMax"], onshoreCapacityMax),
        "windCapacityFactor.csv",
        decimals=4,
    )
    writeTimeSeries(
        weightedMean(data["Wind (offshore), operationRateMax"], offshoreCapacityMax),
        "offshoreWindCapacityFactor.csv",
        decimals=4,
    )
    writeTimeSeries(
        weightedMean(data["PV, operationRateMax"], pvCapacityMax),
        "pvCapacityFactor.csv",
        decimals=4,
    )
    writeTimeSeries(
        weightedMean(
            data["Existing run-of-river plants, operationRateFix"], runOfRiverCapacity
        ),
        "runOfRiverProfile.csv",
        decimals=4,
    )
    writeTimeSeries(
        total(data["Electricity demand, operationRateFix"]),
        "electricityDemand.csv",
        decimals=3,
    )
    writeTimeSeries(
        total(data["Hydrogen demand, operationRateFix"]),
        "hydrogenDemand.csv",
        decimals=3,
    )

    # Capacities per region
    writeTable(
        {
            "Wind (onshore)": total(onshoreCapacityMax),
            "Wind (offshore)": total(offshoreCapacityMax),
            "PV": total(pvCapacityMax),
            "Salt caverns (hydrogen)": total(
                data["Salt caverns (hydrogen), capacityMax"]
            ),
        },
        "capacityPotential.csv",
        indexName="technology",
    )
    writeTable(
        {
            "Run-of-river plants": total(runOfRiverCapacity),
            "Gas power plants": total(
                data["Existing CCGT plants (methane), capacityMax"]
            ),
            "Pumped hydro storage": total(data["Pumped hydro storage, capacityFix"]),
        },
        "existingCapacity.csv",
        indexName="technology",
    )

    # Cables between the regions
    betweenRegions(data["AC cables, capacityFix"]).to_csv(
        OUTPUT_PATH / "existingCapacityACcables.csv"
    )
    betweenRegions(data["DC cables, capacityFix"]).to_csv(
        OUTPUT_PATH / "existingCapacityDCcables.csv"
    )

    # Shapes of the new regions
    clusters = gpd.read_file(
        MULTI_REGION_DATA_PATH / "SpatialData" / "ShapeFiles" / "clusteredRegions.shp"
    ).to_crs(CRS)
    clusterToRegion = {c: region for region, cs in REGIONS.items() for c in cs}
    clusters["region"] = clusters["index"].map(clusterToRegion)
    regions = clusters[["region", "geometry"]].dissolve(by="region").reset_index()
    regions["geometry"] = regions.geometry.simplify(1000)  # 1 km, keeps the file small
    regions.to_file(OUTPUT_PATH / "regions.geojson", driver="GeoJSON")

    # Distances and lines between the centers of the regions
    centers = regions.set_index("region").geometry.centroid
    names = list(REGIONS)
    distances = pd.DataFrame(
        [[round(centers[a].distance(centers[b]) / 1000) for b in names] for a in names],
        index=names,
        columns=names,
    )
    distances.to_csv(OUTPUT_PATH / "distances.csv")

    lines = gpd.GeoDataFrame(
        [
            {
                "index": f"{a}_{b}",
                "loc0": a,
                "loc1": b,
                "geometry": LineString([centers[a], centers[b]]),
            }
            for a in names
            for b in names
            if a != b
        ],
        crs=CRS,
    )
    lines.to_file(OUTPUT_PATH / "lines.geojson", driver="GeoJSON")


if __name__ == "__main__":
    main()
