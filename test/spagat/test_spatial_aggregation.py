import os
import pytest
import numpy as np
from FINE import xarrayIO as xrIO


@pytest.mark.parametrize("use_saved_file", [True, False])
def test_esm_to_xr_and_back_during_spatial_aggregation(
    use_saved_file, multi_node_test_esM_init
):
    """Resulting number of regions would be the same as the original number. No aggregation
    actually takes place. Tests:
        - if the esm instance, created after spatial aggregation
        is run, has all the info originally present.
        - If the saved netcdf file can be reconstructed into an esm instance
            and has all the info originally present.
        - If temporal aggregation and optimization run successfully
    """

    SHAPEFILE_PATH = os.path.join(
        os.path.dirname(__file__),
        "../../examples/Multi-regional_Energy_System_Workflow/",
        "InputData/SpatialData/ShapeFiles/clusteredRegions.shp",
    )

    PATH_TO_SAVE = os.path.join(os.path.dirname(__file__))
    netcdf_file_name = "my_xr.nc"
    shp_file_name = "my_shp"

    # FUNCTION CALL
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(
        shapefile=SHAPEFILE_PATH,
        n_groups=8,
        aggregatedResultsPath=PATH_TO_SAVE,
        aggregated_xr_filename=netcdf_file_name,
        aggregated_shp_name=shp_file_name,
        solver="glpk",
    )

    if use_saved_file:
        saved_file = os.path.join(PATH_TO_SAVE, netcdf_file_name)
        xr_dss = xrIO.readNetCDFToDatasets(filePath=saved_file)
        aggregated_esM = xrIO.convertDatasetsToEnergySystemModel(xr_dss)

    # ASSERTION
    assert sorted(aggregated_esM.locations) == sorted(
        multi_node_test_esM_init.locations
    )

    expected_ts = multi_node_test_esM_init.getComponentAttribute(
        "Hydrogen demand", "operationRateFix"
    ).values
    output_ts = aggregated_esM.getComponentAttribute(
        "Hydrogen demand", "operationRateFix"
    ).values
    assert np.array_equal(expected_ts, output_ts)

    expected_2d = multi_node_test_esM_init.getComponentAttribute(
        "DC cables", "locationalEligibility"
    ).values
    output_2d = aggregated_esM.getComponentAttribute(
        "DC cables", "locationalEligibility"
    ).values
    assert np.array_equal(output_2d, expected_2d)

    expected_1d = multi_node_test_esM_init.getComponentAttribute(
        "Pumped hydro storage", "capacityFix"
    ).values
    output_1d = aggregated_esM.getComponentAttribute(
        "Pumped hydro storage", "capacityFix"
    ).values
    assert np.array_equal(output_1d, expected_1d)

    expected_0d = multi_node_test_esM_init.getComponentAttribute(
        "Electroylzers", "investPerCapacity"
    ).values
    output_0d = aggregated_esM.getComponentAttribute(
        "Electroylzers", "investPerCapacity"
    ).values
    assert np.array_equal(output_0d, expected_0d)

    expected_0d_bool = multi_node_test_esM_init.getComponentAttribute(
        "CO2 from enviroment", "hasCapacityVariable"
    )
    output_0d_bool = aggregated_esM.getComponentAttribute(
        "CO2 from enviroment", "hasCapacityVariable"
    )
    assert output_0d_bool == expected_0d_bool

    # additionally, check if clustering and optimization run through
    aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=4)
    aggregated_esM.optimize(timeSeriesAggregation=True)

    # if there are no problems, delete the saved files
    os.remove(os.path.join(PATH_TO_SAVE, netcdf_file_name))

    file_extensions_list = [".cpg", ".dbf", ".prj", ".shp", ".shx"]

    for ext in file_extensions_list:
        os.remove(os.path.join(PATH_TO_SAVE, f"{shp_file_name}{ext}"))


def test_error_in_reading_shp(multi_node_test_esM_init):
    """Checks if relevant errors are raised when invalid shapefile
    is passed to aggregateSpatially().
    """

    ## Case 1: invalid path
    with pytest.raises(FileNotFoundError):
        SHAPEFILE_PATH = os.path.join(
            os.path.dirname(__file__),
            "../../examples/Multi-regional_Energy_System_Workflow/",
            "InputData/SpatialData/ShapeFiles",
        )

        aggregated_esM = multi_node_test_esM_init.aggregateSpatially(
            shapefile=SHAPEFILE_PATH, n_groups=7, solver="glpk"
        )

    ## Case 2: invalid shapefile type
    with pytest.raises(TypeError):
        aggregated_esM = multi_node_test_esM_init.aggregateSpatially(
            shapefile=multi_node_test_esM_init, n_groups=7, solver="glpk"
        )

    ## Case 3: invalid nRegionsForRepresentation for the shapefile
    with pytest.raises(ValueError):
        SHAPEFILE_PATH = os.path.join(
            os.path.dirname(__file__),
            "../../examples/Multi-regional_Energy_System_Workflow/",
            "InputData/SpatialData/ShapeFiles/clusteredRegions.shp",
        )

        aggregated_esM = multi_node_test_esM_init.aggregateSpatially(
            shapefile=SHAPEFILE_PATH, n_groups=10, solver="glpk"
        )


def test_spatial_aggregation_string_based(multi_node_test_esM_init):

    SHAPEFILE_PATH = os.path.join(
        os.path.dirname(__file__),
        "../../examples/Multi-regional_Energy_System_Workflow/",
        "InputData/SpatialData/ShapeFiles/clusteredRegions.shp",
    )

    # FUNCTION CALL
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(
        shapefile=SHAPEFILE_PATH,
        grouping_mode="string_based",
        aggregatedResultsPath=None,
        separator="_",
    )

    # ASSERTION
    assert len(aggregated_esM.locations) == 8
    # Additional check - if the optimization runs through
    aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver="glpk")


@pytest.mark.parametrize("n_regions", [2, 3])
def test_spatial_aggregation_distance_based(multi_node_test_esM_init, n_regions):

    SHAPEFILE_PATH = os.path.join(
        os.path.dirname(__file__),
        "../../examples/Multi-regional_Energy_System_Workflow/",
        "InputData/SpatialData/ShapeFiles/clusteredRegions.shp",
    )

    # FUNCTION CALL
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(
        shapefile=SHAPEFILE_PATH,
        grouping_mode="distance_based",
        n_groups=n_regions,
        aggregatedResultsPath=None,
    )

    # ASSERTION
    assert len(aggregated_esM.locations) == n_regions
    # Additional check - if the optimization runs through
    aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver="glpk")


@pytest.mark.parametrize(
    "aggregation_function_dict",
    [
        None,
        {
            "operationRateMax": ("weighted mean", "capacityMax"),
            "operationRateFix": ("sum", None),
            "capacityMax": ("sum", None),
            "capacityFix": ("sum", None),
            "locationalEligibility": ("bool", None),
        },
    ],
)
@pytest.mark.parametrize("n_regions", [3, 6])
def test_spatial_aggregation_parameter_based(
    multi_node_test_esM_init, aggregation_function_dict, n_regions
):

    SHAPEFILE_PATH = os.path.join(
        os.path.dirname(__file__),
        "../../examples/Multi-regional_Energy_System_Workflow/",
        "InputData/SpatialData/ShapeFiles/clusteredRegions.shp",
    )

    # FUNCTION CALL
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(
        shapefile=SHAPEFILE_PATH,
        grouping_mode="parameter_based",
        n_groups=n_regions,
        aggregatedResultsPath=None,
        aggregation_function_dict=aggregation_function_dict,
        var_weights={"1d_vars": 10},
        solver="glpk",
    )

    # ASSERTION
    assert len(aggregated_esM.locations) == n_regions
    # Additional check - if the optimization runs through
    aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver="glpk")
