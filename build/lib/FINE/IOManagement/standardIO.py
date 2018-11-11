import FINE as fn
import FINE.utils as utils
import pandas as pd
import ast
import inspect
import time


def writeOptimizationOutputToExcel(esM, outputFileName='scenarioOutput', optSumOutputLevel=2, optValOutputLevel=2):
    """
    Writes optimization output to an Excel file

    :param esM: EnergySystemModel instance in which the optimized model is hold
    :type esM: EnergySystemModel instance

    :param outputFileName: name of the Excel output file (without .xlsx ending)
        |br| * the default value is 'scenarioOutput'
    :type outputFileName: string

    :param optSumOutputLevel: output level of the optimization summary (see EnergySystemModel)
        |br| * the default value is 2
    :type optSumOutputLevel: int (0,1,2)

    :param optValOutputLevel: output level of the optimal values. 0: all values are kept. 1: Lines containing only
        zeroes are dropped.
        |br| * the default value is 1
    :type optValOutputLevel: int (0,1)
    """
    utils.output('\nWriting output to Excel... ', esM.verbose, 0)
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

    utils.output('\t\t(%.4f' % (time.time() - _t) + ' sec)\n', esM.verbose, 0)


def readEnergySystemModelFromExcel(fileName='scenarioInput.xlsx'):
    """
    Reads energy system model from excel file

    :param fileName: excel file name or path (including .xlsx ending)
        |br| * the default value is 'scenarioInput.xlsx'
    :type fileName: string

    :return: esM, esMData - an EnergySystemModel class instance and general esMData as a Series
    """
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
                raise ValueError('NaN values in ' + comp + 'LocSpecs data detected.')
        if comp + 'TimeSeries' in file.sheet_names:
            dataTS = pd.read_excel(file, sheetname=comp + 'TimeSeries', index_col=[0, 1, 2]).sort_index()
            dataTSKeys = set(dataTS.index.get_level_values(0).unique())
            if not dataTSKeys <= dataKeys:
                raise ValueError('Invalid key(s) detected in ' + comp + '\n', dataTSKeys - dataKeys)
            if dataTS.isnull().any().any():
                raise ValueError('NaN values in ' + comp + 'TimeSeries data detected.')

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

    return esM, esMData


def energySystemModelRunFromExcel(fileName='scenarioInput.xlsx'):
    """
    Runs an energy system model from excel file

    :param fileName: excel file name or path (including .xlsx ending)
        |br| * the default value is 'scenarioInput.xlsx'
    :type fileName: string

    :return: esM - an EnergySystemModel class instance and general esMData as a Series
    """
    esM, esMData = readEnergySystemModelFromExcel(fileName)

    if esMData['cluster'] != {}:
        esM.cluster(**esMData['cluster'])
    esM.optimize(**esMData['optimize'])

    writeOptimizationOutputToExcel(esM, **esMData['output'])
    return esM


def getDualValues(pyM):
    """
    Gets dual values of an optimized pyomo instance

    :param pyM: optimized pyomo instance

    :return: Pandas Series with dual values
    """
    return pd.Series(list(pyM.dual.values()), index=pd.Index(list(pyM.dual.keys())))


def getShadowPrices(pyM, constraint, dualValues=None):
    """
    Gets dual values of constraint ("shadow prices")

    :param pyM: pyomo model instance with optimized optimization problem

    :param constraint: constraint from which the dual values should be obtained (e.g. pyM.commodityBalanceConstraint)

    :param dualValues: dual values of the optimized model instance (if not specified is call using the function#
        getDualValues)
        |br| * the default value is None
    :type dualValues: None or Series

    :return: Pandas Series with the dual values of the specified constraint
    """
    if not dualValues:
        dualValues = getDualValues(pyM)
    return pd.Series(list(constraint.values()), index=pd.Index(list(constraint.keys()))).map(dualValues)
