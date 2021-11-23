from copy import deepcopy
import FINE.IOManagement.xarrayIO as xrIO


def test_esm_input_to_dataset_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)

    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esM)
    esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_datasets.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )
    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_datasets.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )

    assert (
        esM.getComponentAttribute("Pipelines", "investPerCapacity")[
            "ElectrolyzerLocation_IndustryLocation"
        ]
        == esm_from_datasets.getComponentAttribute("Pipelines", "investPerCapacity")[
            "ElectrolyzerLocation_IndustryLocation"
        ]
    )


def test_esm_output_to_dataset_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)
    esM.optimize()
    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esM)
    esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    assert (
        (
            esM.componentModelingDict["ConversionModel"]
            .getOptimalValues("operationVariablesOptimum")["values"]
            .loc["Electrolyzers"]
            .values.T
        )
        == (
            esm_datasets["Results"]["ConversionModel"]["Electrolyzers"][
                "operationVariablesOptimum"
            ].values
        )
    ).all()

    assert list(
        esM.getOptimizationSummary("SourceSinkModel", outputLevel=0)
        .loc["Electricity market"]
        .loc["TAC"]
        .values[0]
    ) == list(
        esm_datasets["Results"]["SourceSinkModel"]["Electricity market"].TAC.values
    )


def test_input_esm_to_netcdf_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)
    _ = xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath="test_esM.nc")
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM.nc")

    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_netcdf.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )
    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_netcdf.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )


def test_output_esm_to_netcdf_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)
    esM.optimize()
    _ = xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath="test_esM.nc")
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM.nc")
    print(
        esm_from_netcdf.componentModelingDict["ConversionModel"]
        .getOptimalValues("operationVariablesOptimum")["values"]
        .loc["Electrolyzers"]
        .values
    )
    assert (
        (
            esM.componentModelingDict["ConversionModel"]
            .getOptimalValues("operationVariablesOptimum")["values"]
            .loc["Electrolyzers"]
            .values
        )
        == (
            esm_from_netcdf.componentModelingDict["ConversionModel"]
            .getOptimalValues("operationVariablesOptimum")["values"]
            .loc["Electrolyzers"]
            .values
        )
    ).all()

    assert list(
        esM.getOptimizationSummary("SourceSinkModel", outputLevel=0)
        .loc["Electricity market"]
        .loc["TAC"]
        .values[0]
    ) == list(
        esm_from_netcdf.getOptimizationSummary("SourceSinkModel", outputLevel=0)
        .loc["Electricity market"]
        .loc["TAC"]
        .values[0]
    )
