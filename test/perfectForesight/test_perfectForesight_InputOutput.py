import numpy as np
import pandas as pd
import os
import shutil

import FINE as fn
from FINE.IOManagement.standardIO import writeOptimizationOutputToExcel


def test_perfectForesight_excel(perfectForesight_test_esM):
    # optimize perfect foresight model
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    
    # create empty directory to save results
    cwd = os.getcwd()
    dataPath = os.path.join(cwd, "test", "data")
    resultPath=os.path.join(dataPath,"perfect_foresight_results")
    os.makedirs(resultPath, exist_ok=True)
    
    # write excel output to results folder
    files=os.path.join(resultPath,"pf_results")
    writeOptimizationOutputToExcel(
        perfectForesight_test_esM,
        outputFileName=files, 
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
    
    # check results and excel output
    for ip in perfectForesight_test_esM.investmentPeriodList:
        filePath=files+f"_{ip}.xlsx"
        
        # check if all files are in folder
        if not os.path.isfile(filePath):
            raise ValueError(f"Result excel missing for {ip}.")
        
        # check if results (which are different between the ips) are correctly saved 
        expected_PV_operation=perfectForesight_test_esM.getOptimizationSummary("SourceSinkModel",ip=ip).loc["PV","operation","[kW$_{el}$*h/a]"]["ForesightLand"]
        expected_PV_opexCap=perfectForesight_test_esM.getOptimizationSummary("SourceSinkModel",ip=ip).loc["PV","opexCap","[1 Euro/a]"]["ForesightLand"]
        expected_PV_npv=perfectForesight_test_esM.getOptimizationSummary("SourceSinkModel",ip=ip).loc["PV","NPVcontribution","[1 Euro]"]["PerfectLand"]
        savedExcel=pd.read_excel(filePath,sheet_name="SourceSinkOptSummary_1dim", index_col=[0,1,2])
        output_PV_operation=savedExcel.loc["PV","operation","[kW$_{el}$*h/a]"]["ForesightLand"]
        output_PV_opexCap=savedExcel.loc["PV","opexCap","[1 Euro/a]"]["ForesightLand"]
        output_PV_npv=savedExcel.loc["PV","NPVcontribution","[1 Euro]"]["PerfectLand"]
        np.testing.assert_almost_equal(
            expected_PV_operation,
            output_PV_operation)
        np.testing.assert_almost_equal(
            expected_PV_opexCap,
            output_PV_opexCap)
        np.testing.assert_almost_equal(
            expected_PV_npv, 
            output_PV_npv)
    
    # delete folder with result files
    shutil.rmtree(resultPath)
