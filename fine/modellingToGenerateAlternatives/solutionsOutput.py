import pandas as pd
import os
import time
import fine as fn

def setSolutions(
    esM,
    iteration,
    optSumOutputLevel
):
    for ip in esM.investmentPeriodNames:
        esM.output[iteration][ip] = {}
        optValOutputLevel = 1

        for name in esM.componentModelingDict.keys():

            abbreviatedName = name[:-5] #last 5 letters are "Model" and cut off

            oL = optSumOutputLevel
            oL_ = oL[name] if isinstance(oL, dict) else oL

            optSum = esM.getOptimizationSummary(name, ip=ip, outputLevel=oL_)
            esM.output[iteration][ip][abbreviatedName + "OptSummary_" + esM.componentModelingDict[name].dimension] = optSum

            data = esM.componentModelingDict[name].getOptimalValues(ip=ip)
            oL = optValOutputLevel
            oL_ = oL[name] if isinstance(oL, dict) else oL
            dataTD1dim, indexTD1dim, dataTD2dim, indexTD2dim = [], [], [], []
            dataTI, indexTI = [], []
            for key, d in data.items():
                if d["values"] is None:
                    continue
                if d["timeDependent"]:
                    if d["dimension"] == "1dim":
                        dataTD1dim.append(d["values"]), indexTD1dim.append(key)
                    elif d["dimension"] == "2dim":
                        dataTD2dim.append(d["values"]), indexTD2dim.append(key)
                else:
                    dataTI.append(d["values"]), indexTI.append(key)
            if dataTD1dim:
                names = ["Variable", "Component", "Location"]
                dfTD1dim = pd.concat(dataTD1dim, keys=indexTD1dim, names=names)
                if oL_ == 1:
                    dfTD1dim = dfTD1dim.loc[
                        ((dfTD1dim != 0) & (~dfTD1dim.isnull())).any(axis=1)
                    ]
                esM.output[iteration][ip][abbreviatedName + "_TDoptVar_1dim"] = dfTD1dim
            if dataTD2dim:
                names = ["Variable", "Component", "LocationIn", "LocationOut"]
                dfTD2dim = pd.concat(dataTD2dim, keys=indexTD2dim, names=names)
                if oL_ == 1:
                    dfTD2dim = dfTD2dim.loc[
                        ((dfTD2dim != 0) & (~dfTD2dim.isnull())).any(axis=1)
                    ]
                esM.output[iteration][ip][abbreviatedName + "_TDoptVar_2dim"] = dfTD2dim
            if dataTI:
                if esM.componentModelingDict[name].dimension == "1dim":
                    names = ["Variable type", "Component"]
                elif esM.componentModelingDict[name].dimension == "2dim":
                    names = ["Variable type", "Component", "Location"]
                dfTI = pd.concat(dataTI, keys=indexTI, names=names)
                if oL_ == 1:
                    dfTI = dfTI.loc[((dfTI != 0) & (~dfTI.isnull())).any(axis=1)]
                esM.output[iteration][ip][abbreviatedName + "_TIoptVar_" + esM.componentModelingDict[name].dimension] = dfTI

def writeSolutions(
    esM,
    operationRateinOutput,
    set_solutions
):
    fn.utils.output("\nWriting optimization output to Excel files\n", esM.verbose, 0)

    cwd = os.getcwd()
    directory = os.path.join(cwd, "OutputData")

    if not os.path.exists(directory):
        os.mkdir(directory)

    fn.utils.output(f"Output saved in {directory}\n", esM.verbose, 0)

    # if operationRateinOutput is False, we do not require operation rate variables in the output files.
    if operationRateinOutput:
        optimalParameters = ["OptSummary_", "_TDoptVar_", "_TIoptVar_"]
    else:
        optimalParameters = ["OptSummary_", "_TIoptVar_"]

    for ip in esM.investmentPeriods:

        outdir = os.path.join(directory, f"IP{ip}")
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        for name in esM.componentModelingDict.keys():
            _t = time.time()
            abbreviatedName = name[:-5]

            fn.utils.output(f"\tWriting {abbreviatedName} output....", esM.verbose, 0)
            file_name = abbreviatedName + "Model.xlsx"
            outputFile = os.path.join(outdir, file_name)

            with pd.ExcelWriter(outputFile) as writer:
                for item in optimalParameters:
                    for key in set_solutions.keys():
                        for parameter,val in esM.output[key][ip].items():
                            if abbreviatedName + item in parameter:
                                val.to_excel(writer, sheet_name=item + str(key))
            fn.utils.output("\t\t (%.4f)" % (time.time() - _t) + " sec\n", esM.verbose, 0)
