import pandas as pd
import os
import time
import fine as fn
def writeSolutions(
    esM,
    operationRateinOutput,
    set_solutions,
    summary_solutions,
    getOptimizationSummary
):
    fn.utils.output("\nWriting optimization output to Excel files\n", esM.verbose, 0)

    cwd = os.getcwd()
    directory = os.path.join(cwd, "OutputData")

    if not os.path.exists(directory):
        os.mkdir(directory)

    fn.utils.output(f"Output saved in {directory}\n", esM.verbose, 0)

    # if optimalValueParameters is True, we do not require operation rate variables in the output files.
    if not operationRateinOutput:
        esM.optimalValueParameters = ["cap_"]

    for ip in esM.investmentPeriods:

        outdir = os.path.join(directory, f"IP{ip}")
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        for key, mdl in esM.componentModelingDict.items():
            _t = time.time()
            fn.utils.output(f"\tWriting {key} output....", esM.verbose, 0)
            outputData = {}
            file_name = f"{key}.xlsx"
            outputFile = os.path.join(outdir, file_name)
            with pd.ExcelWriter(outputFile) as writer:
                for parameter in esM.optimalValueParameters:
                    for k in range(len(set_solutions)):
                        if parameter == "op_":
                            if key not in ["TransmissionModel","StorageModel"]:
                                outputData[f'{parameter}_{k}'] = fn.utils.formatOptimizationOutput(
                                    set_solutions[k][key][parameter],
                                    "operationVariables",
                                    "1dim",
                                    ip,
                                    esM.periodsOrder[ip],
                                    esM=esM,
                                )
                                outputData[f'{parameter}_{k}'].to_excel(writer, sheet_name=f'{parameter}_{k}')

                            elif key == "StorageModel":
                                for action in esM.storageParameters:
                                    outputData[f'{action}_{k}'] = fn.utils.formatOptimizationOutput(
                                    set_solutions[k][key][action],
                                    "operationVariables",
                                    "1dim",
                                    ip,
                                    esM.periodsOrder[ip],
                                    esM=esM,
                                )
                                    outputData[f'{action}_{k}'].to_excel(writer, sheet_name=f'{action}_{k}')

                            else:
                                outputData[f'{parameter}_{k}'] = fn.utils.formatOptimizationOutput(
                                    set_solutions[k][key][parameter],
                                    "operationVariables",
                                    "2dim",
                                    ip,
                                    esM.periodsOrder[ip],
                                    compDict=mdl.componentsDict,
                                    esM=esM,
                                )
                                outputData[f'{parameter}_{k}'].to_excel(writer, sheet_name=f'{parameter}_{k}')

                        else:
                            outputData[f'{parameter}_{k}'] = fn.utils.formatOptimizationOutput(
                                set_solutions[k][key][parameter],
                                "designVariables",
                                mdl.dimension,
                                ip,
                                compDict=mdl.componentsDict,
                            )
                            outputData[f'{parameter}_{k}'].to_excel(writer, sheet_name=f'{parameter}_{k}')
            fn.utils.output("\t\t (%.4f)" % (time.time() - _t) + " sec\n", esM.verbose, 0)

    if getOptimizationSummary:
        print("Writing Optimization Summary....")
        _t = time.time()
        for ip in esM.investmentPeriods:
            outdir = os.path.join(directory, f"IP{ip}")
            for key, mdl in esM.componentModelingDict.items():
                file_name = f"{key}.xlsx"
                outputFile = os.path.join(outdir, file_name)
                with pd.ExcelWriter(outputFile,engine="openpyxl", mode="a") as writer:
                    for k in range(len(set_solutions)):
                        summary_solutions[ip][k][key].to_excel(writer, sheet_name=f"summary_{k}")
        fn.utils.output("\t\t (%.4f)" % (time.time() - _t) + " sec\n", esM.verbose, 0)

    # Following code is for consolidating output to single sheets. All the capacity and operation rate variables of all the
    # iterations are written to a single sheet. Time series data are summed up over the time steps.
    print("\nConsolidating output to single sheets \n")

    def get_data(df, iteration, parameter, inputFile, multi_index, key):

        input_data_op = pd.read_excel( inputFile, sheet_name=f"{parameter}__{iteration}")

        if key != "TransmissionModel":
            input_data_op['Unnamed: 0'] = input_data_op['Unnamed: 0'].ffill()
            input_data_op.set_index(['Unnamed: 0','Unnamed: 1'],inplace=True)

        else:
            input_data_op['Unnamed: 0'] = input_data_op['Unnamed: 0'].ffill()
            input_data_op['Unnamed: 1'] = input_data_op['Unnamed: 1'].ffill()
            input_data_op.set_index(['Unnamed: 0','Unnamed: 1','Unnamed: 2'],inplace=True)

        for item in multi_index:
            df.loc[iteration,item] = input_data_op.loc[item].sum()

        return df

    for ip in esM.investmentPeriods:
        outdir = os.path.join(directory, f"IP{ip}")

        for key, mdl in esM.componentModelingDict.items():

            _t = time.time()
            file_name = f"{key}.xlsx"
            new_file_name = f"{key}_consolidated.xlsx"
            inputFile = os.path.join(outdir, file_name)
            outputFile = os.path.join(outdir, new_file_name)
            column_list = list(esM.locations)
            column_list.sort()
            row_index = [iteration for iteration in range(len(set_solutions))]

            with pd.ExcelWriter(outputFile) as writer:
            # mode = "a" if operationRateinOutput else "w"
            # with pd.ExcelWriter(outputFile,engine="openpyxl", mode=mode) as writer:
                if key != "TransmissionModel":
                    data = pd.read_excel(inputFile, sheet_name="cap__0",index_col=0)
                    index_list = data.index
                    multi_index = pd.MultiIndex.from_product([index_list,column_list])
                    print(f"for {key}....")
                    df = pd.DataFrame(index=row_index, columns=multi_index)

                    for item in index_list:
                        for location in column_list:
                            items = []
                            for iteration in row_index:
                                input_data_cap = pd.read_excel( inputFile, sheet_name=f"cap__{iteration}",index_col=0)
                                items.append(input_data_cap.loc[item][location])
                            df.loc[:, (item,location)] = items
                else:
                    data = pd.read_excel(inputFile, sheet_name="cap__0")
                    data['Unnamed: 0'] = data['Unnamed: 0'].ffill()
                    multi_index = [(data.loc[i,"Unnamed: 0"],data.loc[i,"Unnamed: 1"],
                                    column) for i in data.index for column in column_list]
                    print(f"for {key}....")
                    df = pd.DataFrame(index=row_index, data=0.0, 
                                                columns=pd.MultiIndex.from_tuples(multi_index))
                    for iteration in row_index:
                        input_data_cap = pd.read_excel(inputFile, sheet_name=f"cap__{iteration}")
                        input_data_cap['Unnamed: 0'] = input_data_cap['Unnamed: 0'].ffill()
                        input_data_cap.set_index(['Unnamed: 0','Unnamed: 1'],inplace=True)
                        for item in multi_index:
                            df.loc[iteration,item] = input_data_cap.loc[(item[0],item[1]),item[2]]
                    # df = df.dropna(axis=1, how='all')
                df.to_excel(writer, sheet_name="cap")

            if operationRateinOutput:

                # with pd.ExcelWriter(outputFile) as writer:
                with pd.ExcelWriter(outputFile,engine="openpyxl", mode="a") as writer:

                    if key == "StorageModel":
                        data = pd.read_excel(inputFile, sheet_name="chargeOp__0")
                        data['Unnamed: 0'] = data['Unnamed: 0'].ffill()
                        multi_index = [(data.loc[i,"Unnamed: 0"],data.loc[i,"Unnamed: 1"]) for i in data.index]

                        params = ["chargeOp", "dischargeOp"]
                        for param in params:
                            print(f"\t {key}_{param}....")
                            df = pd.DataFrame(index=row_index, data=0.0,
                                                columns=pd.MultiIndex.from_tuples(multi_index))
                            for iteration in row_index:
                                df = get_data(df,iteration,param, inputFile, multi_index, key)
                            df.to_excel(writer, sheet_name=param)

                    elif key == "TransmissionModel":
                        data = pd.read_excel(inputFile, sheet_name="op__0")
                        data['Unnamed: 0'] = data['Unnamed: 0'].ffill()
                        data['Unnamed: 1'] = data['Unnamed: 1'].ffill()
                        multi_index = [(data.loc[i,"Unnamed: 0"],
                                        data.loc[i,"Unnamed: 1"],
                                        data.loc[i,"Unnamed: 2"])
                                        for i in data.index]

                        # print(f"for {key}....")
                        df = pd.DataFrame(index=row_index, data=0.0, columns=pd.MultiIndex.from_tuples(multi_index))
                        for iteration in row_index:
                            df = get_data(df,iteration,"op", inputFile, multi_index, key)
                        df.to_excel(writer, sheet_name="op")

                    else:
                        data = pd.read_excel(inputFile, sheet_name="op__0")
                        data['Unnamed: 0'] = data['Unnamed: 0'].ffill()
                        multi_index = [(data.loc[i,"Unnamed: 0"],data.loc[i,"Unnamed: 1"]) for i in data.index]

                        # print(f"for {key}....")
                        df = pd.DataFrame(index=row_index, data=0.0, columns=pd.MultiIndex.from_tuples(multi_index))
                        for iteration in row_index:
                            df = get_data(df,iteration,"op", inputFile, multi_index, key)
                        df.to_excel(writer, sheet_name="op")

            fn.utils.output("\t\t (%.4f)" % (time.time() - _t) + " sec\n", esM.verbose, 0)
