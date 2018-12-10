import FINE as fn
import FINE.utils as utils
import matplotlib.pyplot as plt
import pandas as pd
import ast
import inspect
import time
import warnings

try:
    import geopandas as gpd
except ImportError:
    warnings.warn('The GeoPandas python package could not be imported.')


def writeOptimizationOutputToExcel(esM, outputFileName='scenarioOutput', optSumOutputLevel=2, optValOutputLevel=2):
    """
    Write optimization output to an Excel file.

    :param esM: EnergySystemModel instance in which the optimized model is hold
    :type esM: EnergySystemModel instance

    :param outputFileName: name of the Excel output file (without .xlsx ending)
        |br| * the default value is 'scenarioOutput'
    :type outputFileName: string

    :param optSumOutputLevel: output level of the optimization summary (see EnergySystemModel)
        |br| * the default value is 2
    :type optSumOutputLevel: int (0,1,2)

    :param optValOutputLevel: output level of the optimal values.
        - 0: all values are kept.
        - 1: Lines containing only zeroes are dropped.
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
    writer.save()

    utils.output('\t\t(%.4f' % (time.time() - _t) + ' sec)\n', esM.verbose, 0)


def readEnergySystemModelFromExcel(fileName='scenarioInput.xlsx'):
    """
    Read energy system model from excel file.

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
    Run an energy system model from excel file.

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

def readOptimizationOutputFromExcel(esM, fileName='scenarioOutput.xlsx'):
    """
    Read optimization output from an excel file.

    :param esM: EnergySystemModel instance which includes the setting of the optimized model
    :type esM: EnergySystemModel instance

    :param fileName: excel file name oder path (including .xlsx ending) to an execl file written by
        writeOptimizationOutputToExcel()
        |br| * the default value is 'scenarioOutput.xlsx'
    :type fileName: string

    :return: esM - an EnergySystemModel class instance
    """

    # Read excel file with optimization output
    file = pd.ExcelFile(fileName)
    # Check if optimization output matches the given energy system model (sufficient condition)
    utils.checkModelClassEquality(esM, file)
    utils.checkComponentsEquality(esM, file)
    # set attributes of esM
    for mdl in esM.componentModelingDict.keys():
        id_c = [0, 1, 2] if '1' in esM.componentModelingDict[mdl].dimension else [0, 1, 2, 3]
        setattr(esM.componentModelingDict[mdl], 'optSummary',
                pd.read_excel(file, sheetname=mdl[0:-5] + 'OptSummary', index_col=id_c))
        sheets = []
        sheets += (sheet for sheet in file.sheet_names if mdl[0:-5] in sheet and 'optVar' in sheet)
        if len(sheets) > 0:
            for sheet in sheets:
                optVal = pd.read_excel(file,sheetname=sheet, index_col=id_c[0:-1] if 'TIoptVar' in sheet else id_c)
                for var in optVal.index.levels[0]: setattr(esM.componentModelingDict[mdl], var, optVal.loc[var])
    return esM


def getDualValues(pyM):
    """
    Get dual values of an optimized pyomo instance.

    :param pyM: optimized pyomo instance
    :type pyM: pyomo Concrete Model

    :return: Pandas Series with dual values
    """
    return pd.Series(list(pyM.dual.values()), index=pd.Index(list(pyM.dual.keys())))


def getShadowPrices(pyM, constraint, dualValues=None):
    """
    Get dual values of constraint ("shadow prices").

    :param pyM: pyomo model instance with optimized optimization problem
    :type pyM: pyomo Concrete Model

    :param constraint: constraint from which the dual values should be obtained (e.g. pyM.commodityBalanceConstraint)
    :type constraint: pyomo.core.base.constraint.SimpleConstraint

    :param dualValues: dual values of the optimized model instance. If it is not specified, it is set by using the
        function getDualValues().
        |br| * the default value is None
    :type dualValues: None or Series

    :return: Pandas Series with the dual values of the specified constraint
    """
    if not dualValues:
        dualValues = getDualValues(pyM)
    return pd.Series(list(constraint.values()), index=pd.Index(list(constraint.keys()))).map(dualValues)


def plotOperation(esM, compName, loc, locTrans=None, tMin=0, tMax=-1, variableName='operationVariablesOptimum',
                  xlabel='time step', ylabel='operation time series', figsize=(12, 4),
                  color="k", fontsize=12, save=False, fileName='operation.png', dpi=200, **kwargs):
    """
    Plot operation time series of a component at a location.

    **Required arguments:**

    :param esM: considered energy system model
    :type esM: EnergySystemModel class instance

    :param compName: component name
    :type compName: string

    :param loc: location
    :type loc: string

    **Default arguments:**

    :param locTrans: second location, required when Transmission components are plotted
    :type locTrans: string

    :param tMin: first time step to be plotted (starting from 0)
        |br| * the default value is 0
    :type tMin: integer

    :param tMax: last time step to be plotted
        |br| * the default value is -1 (i.e. the last available index)
    :type tMax: integer

    :param variableName: name of the operation time series. Checkout the component model class to see which options
        are available.
        |br| * the default value is 'operationVariablesOptimum'
    :type variableName: string

    :param xlabel: x-label of the plot
        |br| * the default value is 'time step'
    :type xlabel: string

    :param ylabel: y-label of the plot
        |br| * the default value is 'operation time series'
    :type ylabel: string

    :param figsize: figure size in inches
        |br| * the default value is (12,4)
    :type figsize: tuple of positive floats

    :param color: color of the operation line
        |br| * the default value is 'k'
    :type color: string

    :param fontsize: font size of the axis
        |br| * the default value is 12
    :type fontsize: positive float

    :param save: indicates if figure should be saved
        |br| * the default value is False
    :type save: boolean

    :param fileName: output file name
        |br| * the default value is 'operation.png'
    :type fileName: string

    :param dpi: resolution in dots per inch
        |br| * the default value is 200
    :type dpi: scalar > 0
    """
    data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(variableName)
    if data is None:
        return
    if locTrans is None:
        timeSeries = data['values'].loc[(compName, loc)].values
    else:
        timeSeries = data['values'].loc[(compName, loc, locTrans)].values

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.grid(True)
    ax.plot(timeSeries[tMin:tMax], color=color)

    ax.tick_params(labelsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plotOperationColorMap(esM, compName, loc, locTrans=None, nbPeriods=365, nbTimeStepsPerPeriod=24,
                          variableName='operationVariablesOptimum', cmap='viridis', vmin=0, vmax=-1,
                          xlabel='day', ylabel='hour', zlabel='operation', figsize=(12, 4),
                          fontsize=12, save=False, fileName='', dpi=200, **kwargs):
    """
    Plot operation time series of a component at a location.

    **Required arguments:**

    :param esM: considered energy system model
    :type esM: EnergySystemModel class instance

    :param compName: component name
    :type compName: string

    :param loc: location
    :type loc: string

    **Default arguments:**

    :param locTrans: second location, required when Transmission components are plotted
    :type locTrans: string

    :param nbPeriods: number of periods to be plotted
        |br| * the default value is 365
    :type nbPeriods: integer

    :param nbTimeStepsPerPeriod: time steps per period to be plotted (nbPeriods*nbTimeStepsPerPeriod=length of time
        series)
        |br| * the default value is 24
    :type nbTimeStepsPerPeriod: integer

    :param variableName: name of the operation time series. Checkout the component model class to see which options
        are available.
        |br| * the default value is 'operationVariablesOptimum'
    :type variableName: string

    :param cmap: heat map (color map) (see matplotlib options)
        |br| * the default value is 'viridis'
    :type cmap: string

    :param vmin: minimum value in heat map
        |br| * the default value is 0
    :type vmin: integer

    :param vmax: maximum value in heat map. If -1, vmax is set to the maximum value of the operation time series.
        |br| * the default value is -1
    :type vmax: integer

    :param xlabel: x-label of the plot
        |br| * the default value is 'day'
    :type xlabel: string

    :param ylabel: y-label of the plot
        |br| * the default value is 'hour'
    :type ylabel: string

    :param zlabel: z-label of the plot
        |br| * the default value is 'operation'
    :type zlabel: string

    :param figsize: figure size in inches
        |br| * the default value is (12,4)
    :type figsize: tuple of positive floats

    :param fontsize: font size of the axis
        |br| * the default value is 12
    :type fontsize: positive float

    :param save: indicates if figure should be saved
        |br| * the default value is False
    :type save: boolean

    :param fileName: output file name
        |br| * the default value is 'operation.png'
    :type fileName: string

    :param dpi: resolution in dots per inch
        |br| * the default value is 200
    :type dpi: scalar > 0
    """
    data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(variableName)
    if locTrans is None:
        timeSeries = data['values'].loc[(compName, loc)].values
    else:
        timeSeries = data['values'].loc[(compName, loc, locTrans)].values
    timeSeries = timeSeries.reshape(nbPeriods, nbTimeStepsPerPeriod).T
    vmax = timeSeries.max() if vmax == -1 else vmax

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.pcolormesh(range(nbPeriods), range(nbTimeStepsPerPeriod), timeSeries, cmap=cmap, vmin=vmin,
                  vmax=vmax, **kwargs)
    ax.axis([0, nbPeriods-1, 0, nbTimeStepsPerPeriod-1])
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm1._A = []
    cb1 = fig.colorbar(sm1, ax=ax, pad=0.05, aspect=7, fraction=0.07)
    cb1.ax.tick_params(labelsize=fontsize)
    cb1.ax.set_xlabel(zlabel, size=fontsize)
    cb1.ax.xaxis.set_label_position('top')

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plotLocations(locationsShapeFileName, indexColumn, plotLocNames=False, crs='epsg:3035', faceColor="none",
                  edgeColor="black", fig=None, ax=None, linewidth=0.5, figsize=(6, 6), fontsize=12,
                  save=False, fileName='', dpi=200, **kwargs):

    """
    Plot locations from a shape file.

    **Required arguments:**

    :param locationsShapeFileName: file name or path to a shape file
    :type locationsShapeFileName: string

    :param indexColumn: name of the column in which the location indices are stored
    :type indexColumn: string

    **Default arguments:**

    :param plotLocNames: indicates if the names of the locations should be plotted
        |br| * the default value is False
    :type plotLocNames: boolean

    :param crs: coordinate reference system
        |br| * the default value is 'epsg:3035'
    :type crs: string

    :param faceColor: face color of the plot
        |br| * the default value is 'none'
    :type faceColor: string

    :param edgeColor: edge color of the plot
        |br| * the default value is 'black'
    :type edgeColor: string

    :param fig: None or figure to which the plot should be added
        |br| * the default value is None
    :type fig: matplotlib Figure

    :param ax: None or ax to which the plot should be added
        |br| * the default value is None
    :type ax: matplotlib Axis

    :param linewidth: linewidth of the plot
        |br| * the default value is 0.5
    :type linewidth: positive float

    :param figsize: figure size in inches
        |br| * the default value is (6,6)
    :type figsize: tuple of positive floats

    :param fontsize: font size of the axis
        |br| * the default value is 12
    :type fontsize: positive float

    :param save: indicates if figure should be saved
        |br| * the default value is False
    :type save: boolean

    :param fileName: output file name
        |br| * the default value is 'operation.png'
    :type fileName: string

    :param dpi: resolution in dots per inch
        |br| * the default value is 200
    :type dpi: scalar > 0
    """
    gdf = gpd.read_file(locationsShapeFileName).to_crs({'init': crs})

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.set_aspect("equal")
    ax.axis("off")
    gdf.plot(ax=ax, facecolor=faceColor, edgecolor=edgeColor, linewidth=linewidth)
    if plotLocNames:
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.9)
        for ix, row in gdf.iterrows():
            locName = ix if indexColumn == '' else row[indexColumn]
            ax.annotate(s=locName, xy=(row.geometry.centroid.x, row.geometry.centroid.y), horizontalalignment='center',
                        fontsize=fontsize, bbox=bbox_props)

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plotTransmission(esM, compName, transmissionShapeFileName, loc0, loc1, crs='epsg:3035',
                     variableName='capacityVariablesOptimum', color='k', loc=7, alpha=0.5, ax=None, fig=None, linewidth=10,
                     figsize=(6, 6), fontsize=12, save=False, fileName='', dpi=200, **kwargs):
    """
    Plot build transmission lines from a shape file.

    **Required arguments:**

    :param esM: considered energy system model
    :type esM: EnergySystemModel class instance

    :param compName: component name
    :type compName: string

    :param transmissionShapeFileName: file name or path to a shape file
    :type transmissionShapeFileName: string

    :param loc0: name of the column in which the location indices are stored (e.g. start/end of line)
    :type loc0: string

    :param loc1: name of the column in which the location indices are stored (e.g. end/start of line)
    :type loc1: string

    **Default arguments:**

    :param crs: coordinate reference system
        |br| * the default value is 'epsg:3035'
    :type crs: string

    :param variableName: name of the operation time series. Checkout the component model class to see which options
        are available.
        |br| * the default value is 'capacityVariables'
    :type variableName: string

    :param color: color of the transmission line
        |br| * the default value is 'k'
    :type color: string

    :param loc: location of the legend in the plot
        |br| * the default value is 7
    :type loc: 0 <= integer <= 10

    :param alpha: transparency of the legend
        |br| * the default value is 0.5
    :type alpha: 0 <= scalar <= 1

    :param fig: None or figure to which the plot should be added
        |br| * the default value is None
    :type fig: matplotlib Figure

    :param ax: None or ax to which the plot should be added
        |br| * the default value is None
    :type ax: matplotlib Axis

    :param linewidth: line width of the plot
        |br| * the default value is 0.5
    :type linewidth: positive float

    :param figsize: figure size in inches
        |br| * the default value is (6,6)
    :type figsize: tuple of positive floats

    :param fontsize: font size of the axis
        |br| * the default value is 12
    :type fontsize: positive float

    :param save: indicates if figure should be saved
        |br| * the default value is False
    :type save: boolean

    :param fileName: output file name
        |br| * the default value is 'operation.png'
    :type fileName: string

    :param dpi: resolution in dots per inch
        |br| * the default value is 200
    :type dpi: scalar > 0
    """
    data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(variableName)
    unit = esM.getComponentAttribute(compName, 'commodityUnit')
    if data is None:
        return fig, ax
    cap = data['values'].loc[compName].copy()
    capMax = cap.max().max()
    if capMax == 0:
        return fig, ax
    cap = cap/capMax
    gdf = gpd.read_file(transmissionShapeFileName).to_crs({'init': crs})

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.set_aspect("equal")
    ax.axis("off")
    for key, row in gdf.iterrows():
        capacity = cap.loc[row[loc0], row[loc1]]
        gdf[gdf.index == key].plot(ax=ax, color=color, linewidth=linewidth*capacity)

    lineMax = plt.Line2D(range(1), range(1), linewidth=linewidth, color=color, marker='_',
                         label="{:>4.4}".format(str(capMax), unit) + ' ' + unit)
    lineMax23 = plt.Line2D(range(1), range(1), linewidth=linewidth*2/3, color=color, marker='_',
                             label="{:>4.4}".format(str(capMax*2/3)) + ' ' + unit)
    lineMax13 = plt.Line2D(range(1), range(1), linewidth=linewidth*1/3, color=color, marker='_',
                             label="{:>4.4}".format(str(capMax*1/3)) + ' ' + unit)

    leg = ax.legend(handles=[lineMax, lineMax23, lineMax13], prop={'size': fontsize}, loc=loc)
    leg.get_frame().set_edgecolor('white')
    leg.get_frame().set_alpha(alpha)

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches='tight')

    return fig, ax


def plotLocationalColorMap(esM, compName, locationsShapeFileName, indexColumn, perArea=True, areaFactor=1e3,
                           crs='epsg:3035', variableName='capacityVariablesOptimum', doSum=False, cmap='viridis', vmin=0,
                           vmax=-1, zlabel='Installed capacity\nper kilometer\n', figsize=(6, 6), fontsize=12, save=False,
                           fileName='', dpi=200, **kwargs):
    """
    Plot the data of a component for each location.

    **Required arguments:**

    :param esM: considered energy system model
    :type esM: EnergySystemModel class instance

    :param compName: component name
    :type compName: string

    :param locationsShapeFileName: file name or path to a shape file
    :type locationsShapeFileName: string

    :param indexColumn: name of the column in which the location indices are stored
    :type indexColumn: string

    **Default arguments:**

    :param perArea: indicates if the capacity should be given per area
        |br| * the default value is False
    :type perArea: boolean

    :param areaFactor: meter * areaFactor = km --> areaFactor = 1e3 (--> capacity/km)
        |br| * the default value is 1e3
    :type areaFactor: scalar > 0

    :param crs: coordinate reference system
        |br| * the default value is 'epsg:3035'
    :type crs: string

    :param variableName: name of the operation time series. Checkout the component model class to see which options
        are available.
        |br| * the default value is 'operationVariablesOptimum'
    :type variableName: string

    :param doSum: indicates if the variable has to be summarized for the location (e.g. for operation
        variables)
        |br| * the default value is False
    :type doSum: boolean

    :param cmap: heat map (color map) (see matplotlib options)
        |br| * the default value is 'viridis'
    :type cmap: string

    :param vmin: minimum value in heat map
        |br| * the default value is 0
    :type vmin: integer

    :param vmax: maximum value in heat map. If -1, vmax is set to the maximum value of the operation time series.
        |br| * the default value is -1
    :type vmax: integer

    :param zlabel: z-label of the plot
        |br| * the default value is 'operation'
    :type zlabel: string

    :param figsize: figure size in inches
        |br| * the default value is (12,4)
    :type figsize: tuple of positive floats

    :param fontsize: font size of the axis
        |br| * the default value is 12
    :type fontsize: positive float

    :param save: indicates if figure should be saved
        |br| * the default value is False
    :type save: boolean

    :param fileName: output file name
        |br| * the default value is 'operation.png'
    :type fileName: string

    :param dpi: resolution in dots per inch
        |br| * the default value is 200
    :type dpi: scalar > 0
    """
    data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(variableName)
    data = data['values'].loc[(compName)]
    if doSum:
        data = data.sum(axis=1)
    gdf = gpd.read_file(locationsShapeFileName).to_crs({'init': crs})
    if perArea:
        gdf.loc[gdf[indexColumn] == data.index, "data"] = \
            data.fillna(0).values/(gdf.loc[gdf[indexColumn] == data.index].geometry.area/areaFactor**2)
    else:
        gdf.loc[gdf[indexColumn] == data.index, "data"] = data.fillna(0).values
    vmax = gdf["data"].max() if vmax == -1 else vmax

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
    ax.set_aspect("equal")
    ax.axis("off")

    gdf.plot(column="data", ax=ax, cmap=cmap, edgecolor='black', alpha=1, linewidth=0.2, vmin=vmin, vmax=vmax)

    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm1._A = []
    cb1 = fig.colorbar(sm1, ax=ax, pad=0.05, aspect=7, fraction=0.07)
    cb1.ax.tick_params(labelsize=fontsize)
    cb1.ax.set_xlabel(zlabel, size=fontsize)
    cb1.ax.xaxis.set_label_position('top')

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches='tight')

    return fig, ax
