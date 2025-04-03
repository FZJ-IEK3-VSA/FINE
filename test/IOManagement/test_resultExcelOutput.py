import os
import inspect
from pathlib import Path

import pandas as pd
from fine import subclasses
import fine as fn

from fine.IOManagement.standardIO import writeOptimizationOutputToExcel


def test_compareResults_longClassNames():
    '''
    Tests long, non-conventional class names which can lead to an error when writing the excel file (at most 31 characters allowed)

    Tests all possible subclasses (and subclasses of subclasses) of the component class
    '''

    #recursively get all subclasses which inherit from "component":

    def recursive_check_inherits_from_component(obj):
        #check whether it is empty
        if len(obj.__bases__) == 0: #obj is empty and not component
            return False
        if "component" in str(obj.__bases__[0]):
            return True
        return recursive_check_inherits_from_component(obj.__bases__[0])


    subclass_objects = []
    for name, obj in inspect.getmembers(subclasses):
        try:
            inheritsFromComponent = recursive_check_inherits_from_component(obj)
        except Exception:
            inheritsFromComponent = False
        if inheritsFromComponent:
            subclass_objects.append(obj)

    #create ESM:
    esM = fn.EnergySystemModel(locations={"Test", "Test1"},
        commodities={"TestCom", "TargetCom"},
        commodityUnitsDict={"TestCom" : "TestUnit",
                            "TargetCom" : "TargetUnit"})
    #add all subclasses which inherit from component to the esm:
    '''
    Adding all possible future subclasses does not work, because it is not clear which future parameters are required.
    Therefore, only the currently known subclasses "ConversionDynamic", "ConversionPartLoad" and "LinearOptimalPowerFlow" are added and an Error is raised if there are subclasses which do not match the ones known.
    '''
    for possibleClass in subclass_objects:
        if "ConversionDynamic" in str(possibleClass):
            esM.add(possibleClass(esM=esM, name=str(possibleClass),investPerCapacity=1, hasCapacityVariable=True,partLoadMin=0.2,bigM=1000, physicalUnit="TestUnit", commodityConversionFactors={"TestCom":-1, "TargetCom": 0.6}))
        elif "ConversionPartLoad" in str(possibleClass):
            continue #conversionPartLoad has an "internal" problem not related to this test. Need to be fixed before this test works properly.
            Operation_level = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            Efficiency = [0.1, 0.15, 0.5, 0.7, 0.7, 0.65, 0.63, 0.62, 0.61, 0.60]
            d = {"x": Operation_level, "y": Efficiency}
            partLoadData = pd.DataFrame(d)
            esM.add(possibleClass(esM=esM, name=str(possibleClass),investPerCapacity=1, hasCapacityVariable=True, physicalUnit="TestUnit", commodityConversionFactors={"TestCom":-1, "TargetCom": 0.5},partLoadMin=0.2,bigM=1000, commodityConversionFactorsPartLoad={'TestCom':-1,'TargetCom':partLoadData}))
        elif "LinearOptimalPowerFlow" in str(possibleClass):
            reactances  = pd.DataFrame(index=["Test", "Test1"], columns=["Test", "Test1"], data = [[1,1],[1,1]])
            esM.add(possibleClass(esM=esM, name=str(possibleClass), commodity="TestCom", reactances =reactances ))
        else:
            raise NotImplementedError(f"Test for class: {possibleClass} not implemented. If a new subclass is added, also add a possible abbreviation in case the name is too long for saving to excel.")

    esM.optimize()

    #save to excel:

    module_directory = Path(__file__).parent.absolute()
    dataPath = os.path.join(module_directory, "..", "data")
    # create new result excel files
    savePath = os.path.join(dataPath, "excelOutputLongClassNames")

    writeOptimizationOutputToExcel(
        esM,
        outputFileName=savePath,
        optSumOutputLevel=2,
        optValOutputLevel=2
    )

def test_compareResults_miniSystem(minimal_test_esM):
    module_directory = Path(__file__).parent.absolute()
    dataPath = os.path.join(module_directory, "..", "data")

    # create new result excel files
    pathWithoutSegmentation_output = os.path.join(dataPath, "output_result_minisystem")
    pathWithSegmentation_output = os.path.join(
        dataPath, "output_result_minisystem_segmentation"
    )
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
    module_directory = Path(__file__).parent.absolute()
    dataPath = os.path.join(module_directory, "..", "data")

    # create new result excel files
    pathMultiNode_output = os.path.join(dataPath, "output_result_multinode")
    saveExcelResults(
        multi_node_test_esM_init,
        pathMultiNode_output,
    )

    # compare to correct result excel files
    # In the change from Pandas 1.X to 2.X there have been changes in how excel
    # files are treated.  We could not identify the underlying changes yet.
    # Therfore we include different references which only differ in the total
    # operation for location 1 by a very small percentage: PV Operation Sum:
    # 1.X: 69472.8, 2.X: 69471.2 Wind (onshore) Operation Sum: 1.X: 282041.2,
    # 2.X: 282042.9
    # -- KK
    pathMultiNodeExcel_output = pathMultiNode_output + ".xlsx"
    pathMultiNodeExcel_expected = os.path.join(
        dataPath, "expected_result_multinode.xlsx"
    )
    pathMultiNodeExcel_expected_pandas1 = os.path.join(
        dataPath, "expected_result_multinode_pandas1.xlsx"
    )  # An adaptation of the expected output was necessary due to the changes in MR 368 / Issue 367 which affected the storage (if there is self-discharge and no precise TSA)

    try:
        compareTwoExcelFiles(pathMultiNodeExcel_expected, pathMultiNodeExcel_output)
    except ValueError:
        compareTwoExcelFiles(
            pathMultiNodeExcel_expected_pandas1, pathMultiNodeExcel_output
        )


def compareTwoExcelFiles(path1, path2):
    xl = pd.ExcelFile(path1)

    # check all sheets
    for sheet in xl.sheet_names:
        # read in the correct index
        if "OptSummary_1dim" in sheet:
            idx_col = [0, 1, 2]
        elif "OptSummary_2dim" in sheet:
            idx_col = [0, 1, 2, 3]
        elif "TIoptVar_1dim" in sheet:
            idx_col = [0, 1]
        elif "TIoptVar_2dim" in sheet:
            idx_col = [0, 1, 2]
        elif "TDoptVar_1dim" in sheet:
            idx_col = [0, 1, 2]
        elif "TDoptVar_2dim" in sheet:
            idx_col = [0, 1, 2, 3]
        elif "Misc" in sheet:
            idx_col = []
        else:
            raise ValueError(f"Unknown index cols for sheet {sheet}")

        # load as dataframe and round for numeric reasons
        expected = pd.read_excel(path1, sheet_name=sheet, index_col=idx_col).round(4)
        output = pd.read_excel(path2, sheet_name=sheet, index_col=idx_col).round(4)

        # check if data has same columns
        if list(expected.columns) != list(output.columns):
            raise ValueError(f"Different columns for sheet {sheet}")
        # 1. check if output excel results contains all rows of exected excel results
        # (new excel results can contain more data) and do not compare the state of charge variables optimum as these can easily differ
        idx = expected.index
        if sheet == "Storage_TDoptVar_1dim":
            idx = [
                x
                for x in expected.index
                if x[0] != "stateOfChargeOperationVariablesOptimum"
            ]
        filtered_output = output.loc[idx]
        expected = expected.loc[idx]
        if len(expected.compare(filtered_output)) > 0:
            # 2.check if sum has difference above one decimal
            # (operation can be quite different)
            # index with different values between expected and output
            idx = expected.compare(filtered_output).index
            # sum of data with different
            _expected_sum = expected.loc[idx].sum(axis=1).round(1)
            _output_sum = filtered_output.loc[idx].sum(axis=1).round(1)
            # check if sum has difference above one decimal
            # (operation can be quite different)
            if not _expected_sum.compare(_output_sum).empty:
                # 3. ignore state of charge
                raise ValueError(
                    f"There are wrong exported results in sheet {sheet} for index "
                    + f"\n {_expected_sum.compare(_output_sum).index}"
                )


def saveExcelResults(multi_node_test_esM_init, savePathWithoutSegmentation):
    # run and save model without segmentation
    multi_node_test_esM_init.aggregateTemporally(
        numberOfTypicalPeriods=3,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )
    multi_node_test_esM_init.optimize(timeSeriesAggregation=True, solver="glpk")
    writeOptimizationOutputToExcel(
        multi_node_test_esM_init,
        outputFileName=savePathWithoutSegmentation,
        optSumOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
            "LOPFModel": 0,
        },
        optValOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
            "LOPFModel": 0,
        },
    )


def saveExcelResultsWithSegmentation(
    minimal_test_esM, savePathWithoutSegmentation, savePathWithSegmentation
):
    # run and save model without segmentation
    minimal_test_esM.optimize(solver="glpk")
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
