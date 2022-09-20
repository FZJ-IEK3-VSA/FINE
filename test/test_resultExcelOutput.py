import os
import pandas as pd
import numpy as np
import FINE as fn
from FINE.IOManagement.standardIO import writeOptimizationOutputToExcel


def test_compareResults_miniSystem(minimal_test_esM):
    cwd = os.getcwd()
    dataPath = os.path.join(cwd, "test", "data")

    # create new result excel files
    pathWithoutSegmentation_output = os.path.join(dataPath, "output_result_minisystem")
    pathWithSegmentation_output = os.path.join(dataPath, "output_result_minisystem_segmentation")
    saveExcelResultsWithSegmentation(
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


def test_compareResults_multiNodeSystem(multi_node_test_esM_init):
    cwd = os.getcwd()
    dataPath = os.path.join(cwd, "test", "data")

    # create new result excel files
    pathMultiNode_output = os.path.join(dataPath, "output_result_multinode")
    saveExcelResults(
       multi_node_test_esM_init, pathMultiNode_output,
    )

    # compare to correct result excel files
    pathMultiNodeExcel_output = pathMultiNode_output + ".xlsx"
    pathMultiNodeExcel_expected = os.path.join(
        dataPath, "expected_result_multinode.xlsx"
    )

    compareTwoExcelFiles(
        pathMultiNodeExcel_expected, pathMultiNodeExcel_output
    )



def compareTwoExcelFiles(path1, path2):
    xl = pd.ExcelFile(path1)
    
    # check all sheets
    for sheet in xl.sheet_names:
        # read in the correct index
        if "OptSummary_1dim" in sheet:
            idx_col=[0,1,2]
        elif "OptSummary_2dim" in sheet:
            idx_col=[0,1,2,3]
        elif "TIoptVar_1dim" in sheet:
            idx_col=[0,1]
        elif "TIoptVar_2dim" in sheet:
            idx_col=[0,1,2]
        elif "TDoptVar_1dim" in sheet:
            idx_col=[0,1,2]
        elif "TDoptVar_2dim" in sheet:
            idx_col=[0,1,2,3]
        elif "Misc" in sheet:
            idx_col=[]
        else:
            raise ValueError(f"Unknown index cols for sheet {sheet}")
        
        # load as dataframe and round for numeric reasons
        expected = pd.read_excel(path1, sheet_name=sheet,index_col=idx_col).round(4)
        output = pd.read_excel(path2, sheet_name=sheet,index_col=idx_col).round(4)

        
        # check if data has same columns
        if list(expected.columns) != list(output.columns):
            raise ValueError(f"Different columns for sheet {sheet}")
        # check if output excel results contains all rows of exected excel results 
        # (new excel results can contain more data)
        filtered_output=output.loc[expected.index] 
        if len(expected.compare(filtered_output)) > 0:
            raise ValueError(
                f"There are wrong exported results in sheet {sheet}: "+
                f"\n {expected.compare(filtered_output)}")


def saveExcelResults(
    multi_node_test_esM_init, savePathWithoutSegmentation
):
    # run and save model without segmentation
    multi_node_test_esM_init.aggregateTemporally(numberOfTypicalPeriods=3)
    multi_node_test_esM_init.optimize(timeSeriesAggregation=True, solver="glpk")
    writeOptimizationOutputToExcel(
        multi_node_test_esM_init,
        outputFileName=savePathWithoutSegmentation,
        optSumOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
            "LOPFModel":0
        },
        optValOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
            "LOPFModel":0
        },
    )

def saveExcelResultsWithSegmentation(
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
