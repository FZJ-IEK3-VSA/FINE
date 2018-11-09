import FINE as fn
import pandas as pd
import ast
import inspect
import time


def writeToExcel(esM, outputFileName='scenarioOutput', optSumOutputLevel=2, optValOutputLevel=2):
    print('\nWriting output to Excel... ', end='')
    _t = time.time()
    writer = pd.ExcelWriter(outputFileName + '.xlsx')

    for name in esM.componentModelingDict.keys():
        oL = optSumOutputLevel
        oL_ = oL[name] if type(oL) == dict else oL
        esM.getOptimizationSummary(name, outputLevel=oL_).to_excel(writer, name[:-5] + 'OptSummary')

        data = esM.componentModelingDict[name].getOptimalValues()
        oL = optValOutputLevel
        oL_ = oL[name] if type(oL) == dict else oL
        dataTD, indexTD = [], []
        dataTI, indexTI = [], []
        for key, d in data.items():
            if d['timeDependent']:
                dataTD.append(d['values']), indexTD.append(key)
            else:
                dataTI.append(d['values']), indexTI.append(key)
        if dataTD:
            dfTD = pd.concat(dataTD, keys=indexTD)
            if oL_ == 1:
                dfTD = dfTD.loc[((dfTD != 0) & (~dfTD.isnull())).any(axis=1)]
            dfTD.to_excel(writer, name[:-5] + 'TDoptVar')
        if dataTI:
            dfTI = pd.concat(dataTI, keys=indexTI)
            if oL_ == 1:
                dfTD = dfTD.loc[((dfTD != 0) & (~dfTD.isnull())).any(axis=1)]
            dfTI.to_excel(writer, name[:-5] + 'TIoptVar')

    periodsOrder = pd.DataFrame([esM.periodsOrder], index=['periodsOrder'], columns=esM.periods)
    periodsOrder.to_excel(writer, 'Misc')

    print("(%.4f" % (time.time() - _t), "sec)")


def energySystemModelRunFromExcel(fileName='scenarioInput.xlsx'):
    file = pd.ExcelFile(fileName)

    esMData = pd.read_excel(file, sheetname='EnergySystemModel', index_col=0, squeeze=True)
    esMData = esMData.apply(lambda v: ast.literal_eval(v) if type(v) == str and v[0] == '{' else v)

    kw = inspect.getargspec(fn.EnergySystemModel.__init__).args
    esM = fn.EnergySystemModel(**esMData[esMData.index.isin(kw)])

    for comp in esMData['componentClasses']:
        data = pd.read_excel(file, sheetname=comp)
        dataKeys = set(data['name'].values)
        if comp + 'LocSpecs' in file.sheet_names:
            dataLoc = pd.read_excel(file, sheetname=comp + 'LocSpecs', index_col=[0, 1, 2]).sort_index()
            dataLocKeys = set(dataLoc.index.get_level_values(0).unique())
            if not dataLocKeys <= dataKeys:
                raise ValueError('Invalid key(s) detected in ' + comp + '\n', dataLocKeys - dataKeys)
            if dataLoc.isnull().any().any():
                print(comp)
                break
        if comp + 'TimeSeries' in file.sheet_names:
            dataTS = pd.read_excel(file, sheetname=comp + 'TimeSeries', index_col=[0, 1, 2]).sort_index()
            dataTSKeys = set(dataTS.index.get_level_values(0).unique())
            if not dataTSKeys <= dataKeys:
                raise ValueError('Invalid key(s) detected in ' + comp + '\n', dataTSKeys - dataKeys)
            if dataTS.isnull().any().any():
                print(comp)
                break

        for key, row in data.iterrows():
            temp = row.dropna()
            temp = temp.drop(temp[temp == 'None'].index)
            temp = temp.apply(lambda v: ast.literal_eval(v) if type(v) == str and v[0] == '{' else v)

            if comp + 'LocSpecs' in file.sheet_names:
                dataLoc_ = dataLoc[dataLoc.index.get_level_values(0) == temp['name']]
                for ix in dataLoc_.index.get_level_values(1).unique():
                    temp.set_value(ix, dataLoc.loc[(temp['name'], ix)].squeeze())

            if comp + 'TimeSeries' in file.sheet_names:
                dataTS_ = dataTS[dataTS.index.get_level_values(0) == temp['name']]
                for ix in dataTS_.index.get_level_values(1).unique():
                    temp.set_value(ix, dataTS_.loc[(temp['name'], ix)].T)

            kwargs = temp
            esM.add(getattr(fn, comp)(esM, **kwargs))

    if esMData['cluster'] != {}:
        esM.cluster(**esMData['cluster'])
    esM.optimize(**esMData['optimize'])

    writeToExcel(esM, **esMData['output'])
