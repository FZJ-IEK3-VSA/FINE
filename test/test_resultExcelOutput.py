import os
import pandas as pd
import FINE as fn
from FINE.IOManagement.standardIO import writeOptimizationOutputToExcel


def test_compareResults(minimal_test_esM):
    cwd = os.getcwd()
    dataPath = os.path.join(cwd, "test", "data")

    # create new result excel files
    pathWithoutSegmentation_output = os.path.join(dataPath, "output_result")
    pathWithSegmentation_output = os.path.join(dataPath, "output_result_segmentation")
    saveExcelResults(
        minimal_test_esM, pathWithoutSegmentation_output, pathWithSegmentation_output
    )

    # compare to correct result excel files
    pathWithoutSegmentation_output = pathWithoutSegmentation_output + ".xlsx"
    pathWithSegmentation_output = pathWithSegmentation_output + ".xlsx"
    pathWithoutSegmentation_expected = os.path.join(
        dataPath, "expected_result_minisystem.xlsx"
    )
    pathWithSegmentation_expected = os.path.join(
        dataPath, "expected_result_minisystem_segmentation.xlsx"
    )

    compareTwoExcelFiles(
        pathWithoutSegmentation_expected, pathWithoutSegmentation_output
    )
    compareTwoExcelFiles(pathWithSegmentation_expected, pathWithSegmentation_output)


def compareTwoExcelFiles(path1, path2):
    xl = pd.ExcelFile(path1)
    # check all sheets
    for sheet in xl.sheet_names:
        # load as dataframe and round for numeric reasons
        expected = pd.read_excel(path1, sheet_name=sheet).round(4)
        output = pd.read_excel(path2, sheet_name=sheet).round(4)
        # check for differences
        if list(expected.index) != list(output.index):
            raise ValueError(f"Diferent index for sheet {sheet}")
        if list(expected.columns) != list(output.columns):
            raise ValueError(f"Diferent columns for sheet {sheet}")
        if len(expected.compare(output)) > 0:
            raise ValueError(f"There are wrong exported results in sheet {sheet}")


def saveExcelResults(
    minimal_test_esM, savePathWithoutSegmentation, savePathWithSegmentation
):
    # run and save model without segmentation
    minimal_test_esM.optimize()
    writeOptimizationOutputToExcel(
        minimal_test_esM,
        outputFileName=savePathWithoutSegmentation,
        optSumOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
        },
        optValOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
        },
    )
    # # run and save model with segmentation
    minimal_test_esM.aggregateTemporally(
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=2,
        storeTSAinstance=False,
        segmentation=True,
        numberOfSegmentsPerPeriod=2,
        clusterMethod="hierarchical",
        sortValues=False,
        rescaleClusterPeriods=False,
    )
    minimal_test_esM.optimize(timeSeriesAggregation=True, solver="glpk")
    writeOptimizationOutputToExcel(
        minimal_test_esM,
        outputFileName=savePathWithSegmentation,
        optSumOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
        },
        optValOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
        },
    )
