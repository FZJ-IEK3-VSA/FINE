import FINE as fn
import FINE.utils as utils
import pandas as pd
import ast
import inspect
import time
import warnings
from functools import wraps
import matplotlib.patches as mpatches

try:
    import geopandas as gpd
except ImportError:
    warnings.warn(
        "The package geopandas is not installed. Spatial aggregation cannot be used without it."
    )

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("Matplotlib.pyplot could not be imported.")


def timer(func):
    """
    Wrapper around a function to track the time taken by the function.

    :param func: Function

    .. note:: Usage as a decorator before a function -> @timer
    """

    @wraps(func)  # Required to get documentation for functions using this decorator
    def f(*args, **kwargs):
        before = time.perf_counter()
        rv = func(*args, **kwargs)
        after = time.perf_counter()
        print(
            "elapsed time for {.__name__}: {:.2f} minutes".format(
                func, (after - before) / 60
            )
        )
        return rv

    return f


def writeOptimizationOutputToExcel(
    esM,
    outputFileName="scenarioOutput",
    optSumOutputLevel=2,
    optValOutputLevel=1,
):
    """
    Write optimization output to an Excel file.

    :param esM: EnergySystemModel instance in which the optimized model is hold
    :type esM: EnergySystemModel instance

    :param outputFileName: name of the Excel output file (without .xlsx ending)
        |br| * the default value is 'scenarioOutput'
    :type outputFileName: string

    :param optSumOutputLevel: output level of the optimization summary (see EnergySystemModel). Either an integer
        (0,1,2) which holds for all model classes or a dictionary with model class names as keys and an integer
        (0,1,2) for each key (e.g. {'StorageModel':1,'SourceSinkModel':1,...}
        |br| * the default value is 2
    :type optSumOutputLevel: int (0,1,2) or dict

    :param optValOutputLevel: output level of the optimal values. Either an integer (0,1) which holds for all
        model classes or a dictionary with model class names as keys and an integer (0,1) for each key
        (e.g. {'StorageModel':1,'SourceSinkModel':1,...}

        * 0: all values are kept.
        * 1: Lines containing only zeroes are dropped.

        |br| * the default value is 1
    :type optValOutputLevel: int (0,1) or dict
    """
    for ip in esM.investmentPeriodNames:
        if len(esM.investmentPeriodNames) > 1:
            _outputFileName = outputFileName + f"_{ip}"
        else:
            _outputFileName = outputFileName
        utils.output("\nWriting output to Excel... ", esM.verbose, 0)
        _t = time.time()
        writer = pd.ExcelWriter(_outputFileName + ".xlsx")

        for name in esM.componentModelingDict.keys():
            utils.output("\tProcessing " + name + " ...", esM.verbose, 0)
            oL = optSumOutputLevel
            oL_ = oL[name] if type(oL) == dict else oL

            optSum = esM.getOptimizationSummary(name, ip=ip, outputLevel=oL_)
            if not optSum.empty:
                optSum.to_excel(
                    writer,
                    name[:-5]
                    + "OptSummary_"
                    + esM.componentModelingDict[name].dimension,
                )

            data = esM.componentModelingDict[name].getOptimalValues(ip=ip)
            oL = optValOutputLevel
            oL_ = oL[name] if type(oL) == dict else oL
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
                if not dfTD1dim.empty:
                    dfTD1dim.to_excel(writer, name[:-5] + "_TDoptVar_1dim")
            if dataTD2dim:
                names = ["Variable", "Component", "LocationIn", "LocationOut"]
                dfTD2dim = pd.concat(dataTD2dim, keys=indexTD2dim, names=names)
                if oL_ == 1:
                    dfTD2dim = dfTD2dim.loc[
                        ((dfTD2dim != 0) & (~dfTD2dim.isnull())).any(axis=1)
                    ]
                if not dfTD2dim.empty:
                    dfTD2dim.to_excel(writer, name[:-5] + "_TDoptVar_2dim")
            if dataTI:
                if esM.componentModelingDict[name].dimension == "1dim":
                    names = ["Variable type", "Component"]
                elif esM.componentModelingDict[name].dimension == "2dim":
                    names = ["Variable type", "Component", "Location"]
                dfTI = pd.concat(dataTI, keys=indexTI, names=names)
                if oL_ == 1:
                    dfTI = dfTI.loc[((dfTI != 0) & (~dfTI.isnull())).any(axis=1)]
                if not dfTI.empty:
                    dfTI.to_excel(
                        writer,
                        name[:-5]
                        + "_TIoptVar_"
                        + esM.componentModelingDict[name].dimension,
                    )
        # get internal name of investment period
        _ip = esM.investmentPeriodNames.index(ip)
        # save periods Order to excel output
        periodsOrder = pd.DataFrame(
            [esM.periodsOrder[_ip]], index=["periodsOrder"], columns=esM.periods
        )
        periodsOrder.to_excel(writer, "Misc")
        if esM.segmentation:
            ls = []
            for i in esM.periodsOrder[_ip].tolist():
                ls.append(esM.timeStepsPerSegment[_ip][i])
            segmentDuration = pd.concat(ls, axis=1).rename(
                columns={"Segment Duration": "timeStepsPerSegment"}
            )
            segmentDuration.index.name = "segmentNumber"
            segmentDuration.to_excel(writer, "Misc", startrow=3)
        utils.output("\tSaving file...", esM.verbose, 0)
        writer.close()
        utils.output("Done. (%.4f" % (time.time() - _t) + " sec)", esM.verbose, 0)


def readEnergySystemModelFromExcel(fileName="scenarioInput.xlsx", engine="openpyxl"):
    """
    Read energy system model from excel file.

    **Default arguments:**

    :param fileName: excel file name or path (including .xlsx ending)
        |br| * the default value is 'scenarioInput.xlsx'
    :type fileName: string

    :param engine: Used engine for reading the excel file. Please consider that the corresponding
        python package has to be installed. openpyxl and xlrd are already part of the requirements of FINE.
        For further information see the documentation of pandas.read_excel().

        * 'openpyxl' supports newer Excel file formats
        * 'xlrd' supports old-style Excel files (.xls)
        * 'odf' supports OpenDocument file formats (.odf, .ods, .odt)

        |br| * the default value is 'openpyxl'.
    :type engine: string

    :return: esM, esMData - an EnergySystemModel class instance and general esMData as a Series
    """
    file = pd.ExcelFile(fileName, engine=engine)
    esMData = (
        pd.read_excel(
            file,
            sheet_name="EnergySystemModel",
            index_col=0,
        )
        .squeeze("columns")
        .dropna(axis="index", how="all")
    )
    esMData = esMData.apply(
        lambda v: ast.literal_eval(v) if type(v) == str and v[0] == "{" else v
    )

    kw = inspect.getargspec(fn.EnergySystemModel.__init__).args
    esM = fn.EnergySystemModel(**esMData[esMData.index.isin(kw)])

    for comp in esMData["componentClasses"]:
        data = pd.read_excel(file, sheet_name=comp).dropna(axis="index", how="all")
        dataKeys = set(data["name"].values)
        if comp + "LocSpecs" in file.sheet_names:
            dataLoc = (
                pd.read_excel(file, sheet_name=comp + "LocSpecs", index_col=[0, 1, 2])
                .dropna(axis="columns", how="all")
                .sort_index()
            )
            dataLocKeys = set(dataLoc.index.get_level_values(0).unique())
            if not dataLocKeys <= dataKeys:
                raise ValueError(
                    "Invalid key(s) detected in " + comp + "\n", dataLocKeys - dataKeys
                )
            if dataLoc.isnull().any().any():
                raise ValueError("NaN values in " + comp + "LocSpecs data detected.")
        if comp + "TimeSeries" in file.sheet_names:
            dataTS = (
                pd.read_excel(file, sheet_name=comp + "TimeSeries", index_col=[0, 1, 2])
                .dropna(axis="columns", how="all")
                .sort_index()
            )
            dataTSKeys = set(dataTS.index.get_level_values(0).unique())
            if not dataTSKeys <= dataKeys:
                raise ValueError(
                    "Invalid key(s) detected in " + comp + "\n", dataTSKeys - dataKeys
                )
            if dataTS.isnull().any().any():
                raise ValueError("NaN values in " + comp + "TimeSeries data detected.")

        for key, row in data.iterrows():
            temp = row.dropna()
            temp = temp.drop(temp[temp == "None"].index)
            temp = temp.apply(
                lambda v: ast.literal_eval(v) if type(v) == str and v[0] == "{" else v
            )

            if comp + "LocSpecs" in file.sheet_names:
                dataLoc_ = dataLoc[dataLoc.index.get_level_values(0) == temp["name"]]
                for ix in dataLoc_.index.get_level_values(1).unique():
                    temp[ix] = dataLoc.loc[(temp["name"], ix)].squeeze()

            if comp + "TimeSeries" in file.sheet_names:
                dataTS_ = dataTS[dataTS.index.get_level_values(0) == temp["name"]]
                for ix in dataTS_.index.get_level_values(1).unique():
                    temp[ix] = dataTS_.loc[(temp["name"], ix)].T

            kwargs = temp
            esM.add(getattr(fn, comp)(esM, **kwargs))

    return esM, esMData


def energySystemModelRunFromExcel(fileName="scenarioInput.xlsx", engine="openpyxl"):
    """
    Run an energy system model from excel file.

    **Default arguments:**

    :param fileName: excel file name or path (including .xlsx ending)
        |br| * the default value is 'scenarioInput.xlsx'
    :type fileName: string

    :param engine: Used engine for reading the excel file. Please consider that the corresponding
        python package has to be installed. openpyxl and xlrd are already part of the requirements of FINE.
        For further information see the documentation of pandas.read_excel().

        * 'openpyxl' supports newer Excel file formats
        * 'xlrd' supports old-style Excel files (.xls)
        * 'odf' supports OpenDocument file formats (.odf, .ods, .odt)

        |br| * the default value is 'openpyxl'.
    :type engine: string

    :return: esM - an EnergySystemModel class instance and general esMData as a Series
    """
    esM, esMData = readEnergySystemModelFromExcel(fileName, engine=engine)

    if esMData["cluster"] != {}:
        esM.aggregateTemporally(**esMData["cluster"])
    esM.optimize(**esMData["optimize"])

    writeOptimizationOutputToExcel(esM, **esMData["output"])
    return esM


def readOptimizationOutputFromExcel(
    esM, fileName="scenarioOutput.xlsx", engine="openpyxl"
):
    """
    Read optimization output from an excel file.

    :param esM: EnergySystemModel instance which includes the setting of the optimized model
    :type esM: EnergySystemModel instance

    **Default arguments:**

    :param fileName: excel file name oder path (including .xlsx ending) to an execl file written by
        writeOptimizationOutputToExcel()
        |br| * the default value is 'scenarioOutput.xlsx'
    :type fileName: string

    :param engine: Used engine for reading the excel file. Please consider that the corresponding
        python package has to be installed. openpyxl and xlrd are already part of the requirements of FINE.
        For further information see the documentation of pandas.read_excel().

        * 'openpyxl' supports newer Excel file formats
        * 'xlrd' supports old-style Excel files (.xls)
        * 'odf' supports OpenDocument file formats (.odf, .ods, .odt)

        |br| * the default value is 'openpyxl'.
    :type engine: string

    :return: esM - an EnergySystemModel class instance
    """

    # Read excel file with optimization output
    file = pd.ExcelFile(fileName, engine=engine)
    # Check if optimization output matches the given energy system model (sufficient condition)
    utils.checkModelClassEquality(esM, file)
    utils.checkComponentsEquality(esM, file)
    # set attributes of esM
    for mdl in esM.componentModelingDict.keys():
        dim = esM.componentModelingDict[mdl].dimension
        idColumns1dim = [0, 1, 2]
        idColumns2dim = [0, 1, 2, 3]
        idColumns = idColumns1dim if "1" in dim else idColumns2dim
        setattr(
            esM.componentModelingDict[mdl],
            "optSummary",
            pd.read_excel(
                fileName,
                sheet_name=mdl[0:-5] + "OptSummary_" + dim,
                index_col=idColumns,
                engine=engine,
            ),
        )
        sheets = []
        sheets += (
            sheet
            for sheet in file.sheet_names
            if mdl[0:-5] in sheet and "optVar" in sheet
        )
        if len(sheets) > 0:
            for sheet in sheets:
                if "TDoptVar_1dim" in sheet:
                    index_col = idColumns1dim
                elif "TIoptVar_1dim" in sheet:
                    index_col = idColumns1dim[:-1]
                elif "TDoptVar_2dim" in sheet:
                    index_col = idColumns2dim
                elif "TIoptVar_2dim" in sheet:
                    index_col = idColumns2dim[:-1]
                else:
                    continue
                optVal = pd.read_excel(
                    fileName, sheet_name=sheet, index_col=index_col, engine=engine
                )
                for var in optVal.index.levels[0]:
                    setattr(esM.componentModelingDict[mdl], var, optVal.loc[var])
    return esM


def getDualValues(pyM):
    """
    Get dual values of an optimized pyomo instance.

    :param pyM: optimized pyomo instance
    :type pyM: pyomo Concrete Model

    :return: Pandas Series with dual values
    """
    return pd.Series(list(pyM.dual.values()), index=pd.Index(list(pyM.dual.keys())))


def getShadowPrices(
    esM,
    constraint,
    ip=0,
    dualValues=None,
    hasTimeSeries=False,
    periodOccurrences=None,
    periodsOrder=None,
):
    """
    Get dual values of constraint ("shadow prices").

    :param esM: considered energy system model
    :type esM: EnergySystemModel class instance

    :param constraint: constraint from which the dual values should be obtained (e.g. pyM.commodityBalanceConstraint)
    :type constraint: pyomo.core.base.constraint.SimpleConstraint

    :param ip: investment period of transformation path analysis.
    :type ip: int

    :param dualValues: dual values of the optimized model instance. If it is not specified, it is set by using the
        function getDualValues().
        |br| * the default value is None
    :type dualValues: None or Series

    :param hasTimeSeries: If the constaint is time dependent, this parameter concatenates the dual values
        to a full time series (particularly usefull if time series aggregation was considered).
        |br| * the default value is False
    :type hasTimeSeries: bool

    :param periodOccurrences: Only required if hasTimeSeries is set to True.
        |br| * the default value is None
    :type periodOccurrences: list or None

    :param periodsOrder: Only required if hasTimeSeries is set to True.
        |br| * the default value is None
    :type periodsOrder: list or None

    :return: Pandas Series with the dual values of the specified constraint
    """
    if dualValues is None:
        dualValues = getDualValues(esM.pyM)

    SP = pd.Series(
        list(constraint.values()), index=pd.Index(list(constraint.keys()))
    ).map(dualValues)
    # Select rows where ip is equal to investigated ip
    SP = SP.iloc[SP.index.get_level_values(2) == ip]
    # Delete ip from multiindex
    SP = SP.droplevel(2, axis=0)

    if hasTimeSeries:
        SP = pd.DataFrame(SP).swaplevel(i=0, j=-2).sort_index()
        SP = SP.unstack(level=-1)
        SP.columns = SP.columns.droplevel()
        SP = SP.apply(lambda x: x / (periodOccurrences[ip][x.name[0]]), axis=1)
        SP = fn.utils.buildFullTimeSeries(
            SP, periodsOrder[ip], ip, esM=esM, divide=False
        )
        SP = SP.stack()

    return SP


def plotOperation(
    esM,
    compName,
    loc,
    ip=0,
    locTrans=None,
    tMin=0,
    tMax=-1,
    variableName="operationVariablesOptimum",
    xlabel="time step",
    ylabel="operation time series",
    figsize=(12, 4),
    color="k",
    fontsize=12,
    save=False,
    fileName="operation.png",
    dpi=200,
    **kwargs,
):
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

    :param ip: investment period
        |br| * the default value is 0
    :type ip: int

    :param locTrans: second location, required when Transmission components are plotted
        |br| * the default value is None
    :type locTrans: string

    :param tMin: first time step to be plotted (starting from 0)
        |br| * the default value is 0
    :type tMin: integer

    :param tMax: last time step to be plotted
        |br| * the default value is -1 (i.e. the last available index)
    :type tMax: integer

    :param variableName: name of the operation time series. Checkout the component model class to see which options
        are available.
        |br| * the default value is '_operationVariablesOptimum'
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
    data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(
        variableName, ip=ip
    )
    if data is None:
        return
    if locTrans is None:
        timeSeries = data["values"].loc[(compName, loc)].values
    else:
        timeSeries = data["values"].loc[(compName, loc, locTrans)].values

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.grid(True)
    ax.plot(timeSeries[tMin:tMax], color=color)

    ax.tick_params(labelsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plotOperationColorMap(
    esM,
    compName,
    loc,
    ip=0,
    locTrans=None,
    nbPeriods=365,
    nbTimeStepsPerPeriod=24,
    variableName="operationVariablesOptimum",
    cmap="viridis",
    vmin=0,
    vmax=-1,
    xlabel="period",
    ylabel="timestep per period",
    zlabel="",
    figsize=(12, 4),
    fontsize=12,
    save=False,
    fileName="",
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    monthlabels=False,
    dpi=200,
    pad=0.12,
    aspect=15,
    fraction=0.2,
    orientation="horizontal",
    **kwargs,
):
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

    :param ip: investment period of transformation path analysis.
    :type ip: int

    :param locTrans: second location, required when Transmission components are plotted
        |br| * the default value is None
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
        |br| * the default value is '_operationVariablesOptimum'
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

    :param xticks: user specified ticks of the x axis
        |br| * the default value is None
    :type xticks: list

    :param yticks: user specified ticks of the ý axis
        |br| * the default value is None
    :type yticks: list

    :param xticklabels: user specified tick labels of the x axis
        |br| * the default value is None
    :type xticklabels: list

    :param yticklabels: user specified tick labels of the ý axis
        |br| * the default value is None
    :type yticklabels: list

    :param monthlabels: specifies if month labels are to be plotted (only works correctly if
        365 days are specified as the number of periods)
        |br| * the default value is False
    :type monthlabels: boolean

    :param dpi: resolution in dots per inch
        |br| * the default value is 200
    :type dpi: scalar > 0

    :param pad: pad parameter of colorbar
        |br| * the default value is 0.12
    :type pad: float

    :param aspect: aspect parameter of colorbar
        |br| * the default value is 15
    :type aspect: float

    :param fraction: fraction parameter of colorbar
        |br| * the default value is 0.2
    :type fraction: float

    :param orientation: orientation parameter of colorbar
        |br| * the default value is 'horizontal'
    :type orientation: float

    """
    isStorage = False

    if isinstance(esM.getComponent(compName), fn.Conversion):
        unit = esM.getComponent(compName).physicalUnit
    else:
        unit = esM.commodityUnitsDict[esM.getComponent(compName).commodity]

    if isinstance(esM.getComponent(compName), fn.Storage):
        isStorage = True
        unit = unit + "*h"

    data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(
        variableName, ip=ip
    )

    if locTrans is None:
        timeSeries = data["values"].loc[(compName, loc)].values
    else:
        timeSeries = data["values"].loc[(compName, loc, locTrans)].values
    timeSeries = timeSeries / esM.hoursPerTimeStep if not isStorage else timeSeries

    try:
        timeSeries = timeSeries.reshape(nbPeriods, nbTimeStepsPerPeriod).T
    except ValueError as e:
        raise ValueError(
            "Could not reshape array. Your timeSeries has {} values and it is therefore not possible".format(
                len(timeSeries)
            )
            + " to reshape it to ({}, {}). Please correctly specify nbPeriods".format(
                nbPeriods, nbTimeStepsPerPeriod
            )
            + " and nbTimeStepsPerPeriod The error was: {}.".format(e)
        )
    vmax = timeSeries.max() if vmax == -1 else vmax

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.pcolormesh(
        range(nbPeriods + 1),
        range(nbTimeStepsPerPeriod + 1),
        timeSeries,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )
    ax.axis([0, nbPeriods, 0, nbTimeStepsPerPeriod])
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.xaxis.set_label_position("top"), ax.xaxis.set_ticks_position("top")

    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm1._A = []
    cb1 = fig.colorbar(
        sm1, ax=ax, pad=pad, aspect=aspect, fraction=fraction, orientation=orientation
    )
    cb1.ax.tick_params(labelsize=fontsize)
    if zlabel != "":
        cb1.ax.set_xlabel(zlabel, size=fontsize)
    elif isStorage:
        cb1.ax.set_xlabel("Storage inventory" + " [" + unit + "]", size=fontsize)
    else:
        cb1.ax.set_xlabel("Operation" + " [" + unit + "]", size=fontsize)
    cb1.ax.xaxis.set_label_position("top")

    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels, fontsize=fontsize)
    if yticklabels:
        ax.set_yticklabels(yticklabels, fontsize=fontsize)

    if monthlabels:
        import datetime

        xticks, xlabels = [], []
        for i in range(1, 13, 2):
            xlabels.append(datetime.date(2050, i + 1, 1).strftime("%b"))
            xticks.append(datetime.datetime(2050, i + 1, 1).timetuple().tm_yday)
            ax.set_xticks(xticks), ax.set_xticklabels(xlabels, fontsize=fontsize)

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plotLocations(
    locationsShapeFileName,
    indexColumn,
    plotLocNames=False,
    crs="epsg:3035",
    faceColor="none",
    edgeColor="black",
    fig=None,
    ax=None,
    linewidth=0.5,
    figsize=(6, 6),
    fontsize=12,
    save=False,
    fileName="",
    dpi=200,
    **kwargs,
):
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

    gdf = gpd.read_file(locationsShapeFileName).to_crs({"init": crs})

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.set_aspect("equal")
    ax.axis("off")
    gdf.plot(ax=ax, facecolor=faceColor, edgecolor=edgeColor, linewidth=linewidth)
    if plotLocNames:
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.9)
        for ix, row in gdf.iterrows():
            locName = ix if indexColumn == "" else row[indexColumn]
            ax.annotate(
                text=locName,
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                horizontalalignment="center",
                fontsize=fontsize,
                bbox=bbox_props,
            )

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plotTransmission(
    esM,
    compName,
    transmissionShapeFileName,
    loc0,
    loc1,
    ip=0,
    crs="epsg:3035",
    variableName="capacityVariablesOptimum",
    color="k",
    loc=7,
    alpha=0.5,
    ax=None,
    fig=None,
    linewidth=10,
    figsize=(6, 6),
    fontsize=12,
    save=False,
    fileName="",
    dpi=200,
    **kwargs,
):
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

    :param ip: investment periods
        |br| * the default value is 0
    :type ip: int

    :param crs: coordinate reference system
        |br| * the default value is 'epsg:3035'
    :type crs: string

    :param variableName: parameter for plotting installed capacity ('_capacityVariablesOptimum') or operation
        ('_operationVariablesOptimum').
        |br| * the default value is '_capacityVariablesOptimum'
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
    data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(
        variableName, ip=ip
    )
    unit = esM.getComponentAttribute(compName, "commodityUnit")
    if data is None:
        return fig, ax
    cap = data["values"].loc[compName].copy()
    capMax = cap.max().max()
    if capMax == 0:
        return fig, ax
    cap = cap / capMax
    gdf = gpd.read_file(transmissionShapeFileName).to_crs({"init": crs})

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.set_aspect("equal")
    ax.axis("off")
    for key, row in gdf.iterrows():
        capacity = cap.loc[row[loc0], row[loc1]]
        gdf[gdf.index == key].plot(ax=ax, color=color, linewidth=linewidth * capacity)

    lineMax = plt.Line2D(
        range(1),
        range(1),
        linewidth=linewidth,
        color=color,
        marker="_",
        label="{:>4.4}".format(str(capMax), unit) + " " + unit,
    )
    lineMax23 = plt.Line2D(
        range(1),
        range(1),
        linewidth=linewidth * 2 / 3,
        color=color,
        marker="_",
        label="{:>4.4}".format(str(capMax * 2 / 3)) + " " + unit,
    )
    lineMax13 = plt.Line2D(
        range(1),
        range(1),
        linewidth=linewidth * 1 / 3,
        color=color,
        marker="_",
        label="{:>4.4}".format(str(capMax * 1 / 3)) + " " + unit,
    )

    leg = ax.legend(
        handles=[lineMax, lineMax23, lineMax13], prop={"size": fontsize}, loc=loc
    )
    leg.get_frame().set_edgecolor("white")
    leg.get_frame().set_alpha(alpha)

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plotLocationalColorMap(
    esM,
    compName,
    locationsShapeFileName,
    indexColumn,
    ip=0,
    perArea=True,
    areaFactor=1e3,
    crs="epsg:3035",
    variableName="capacityVariablesOptimum",
    doSum=False,
    cmap="viridis",
    vmin=0,
    vmax=-1,
    zlabel=None,
    figsize=(6, 6),
    fontsize=12,
    save=False,
    fileName="capacity.png",
    dpi=200,
    **kwargs,
):
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

    :param ip: investment period
        |br| * the default value is 0
    :type ip: int

    :param perArea: indicates if the capacity should be given per area
        |br| * the default value is False
    :type perArea: boolean

    :param areaFactor: meter * areaFactor = km --> areaFactor = 1e3 (--> capacity/km)
        |br| * the default value is 1e3
    :type areaFactor: scalar > 0

    :param crs: coordinate reference system
        |br| * the default value is 'epsg:3035'
    :type crs: string

    :param variableName: parameter for plotting installed capacity ('_capacityVariablesOptimum') or operation
        ('_operationVariablesOptimum'). In case of plotting the operation, set the parameter doSum to True.
        |br| * the default value is '_capacityVariablesOptimum'
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
        |br| * the default value is 'capacity.png'
    :type fileName: string

    :param dpi: resolution in dots per inch
        |br| * the default value is 200
    :type dpi: scalar > 0
    """
    data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(
        variableName, ip=ip
    )
    data = data["values"].loc[(compName)]

    if doSum:
        data = data.sum(axis=1)
    gdf = gpd.read_file(locationsShapeFileName).to_crs({"init": crs})

    # Make sure the data and gdf indices match
    ## 1. Sort the indices to obtain same order
    data.sort_index(inplace=True)
    gdf.sort_values(indexColumn, inplace=True)

    ## 2. Take first 20 characters of the string for matching. (In gdfs usually long strings are cut in the end)
    gdf[indexColumn] = gdf[indexColumn].apply(lambda x: x[:20])
    data.index = data.index.str[:20]

    ## 3. Merge data on the indices of the gdf, additional (pseudo) regions in data are ignored
    data = pd.DataFrame(data)
    data = data.rename({data.columns.values[0]: "data"}, axis=1)
    gdf = pd.merge(gdf, data, left_on=indexColumn, right_index=True, how="left")
    gdf = gdf.fillna(0)

    ## 4. Print the names of the excluded (pseudo) regions
    regions_data = list(data.index)
    regions_gdf = list(gdf.loc[:, indexColumn])

    excluded_regions = [item for item in regions_data if item not in regions_gdf]

    if len(excluded_regions) > 0:
        print(
            f"Missing regions: {compName} - {variableName} \n",
            f"The following regions are not plotted as they are not contained in the provided shapefile: \n",
            f"{excluded_regions} \n",
        )

    if perArea:
        gdf.loc[:, "data"] = gdf.loc[:, "data"] / (gdf.geometry.area / areaFactor**2)
        if zlabel is None:
            if isinstance(esM.getComponent(compName), fn.Conversion):
                unit = esM.getComponent(compName).physicalUnit
            else:
                unit = esM.commodityUnitsDict[esM.getComponent(compName).commodity]

            if areaFactor == 1e3:
                area_unit = "km$^2$"
            elif areaFactor == 1:
                area_unit = "m$^2$"
            else:
                raise NotImplementedError(
                    "Area Factor not supported. Supported Area Factors {0},{1}".format(
                        1, 1e3
                    )
                )

            unit = " [" + unit + "/" + area_unit + "]"
            zlabel = "Installed capacity \n" + unit + "\n"

    else:
        if zlabel is None:
            if isinstance(esM.getComponent(compName), fn.Conversion):
                unit = esM.getComponent(compName).physicalUnit
            else:
                unit = esM.commodityUnitsDict[esM.getComponent(compName).commodity]
            zlabel = f"Installed capacity \n [ {unit} ] \n"

    vmax = gdf["data"].max() if vmax == -1 else vmax

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)
    ax.set_aspect("equal")
    ax.axis("off")

    gdf.plot(
        column="data",
        ax=ax,
        cmap=cmap,
        edgecolor="black",
        alpha=1,
        linewidth=0.2,
        vmin=vmin,
        vmax=vmax,
    )

    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm1._A = []
    cb1 = fig.colorbar(sm1, ax=ax, pad=0.05, aspect=7, fraction=0.07)
    cb1.ax.tick_params(labelsize=fontsize)
    cb1.ax.set_xlabel(zlabel, size=fontsize)
    cb1.ax.xaxis.set_label_position("top")

    fig.tight_layout()

    if save:
        plt.savefig(fileName, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plotPieChart(
    locFilePath,
    results_df,
    Property_to_plot="capacity",
    indexColumn_in_shp="index",
    color_list=[
        "skyBlue",
        "green",
        "yellowGreen",
        "#FFB732",
        "yellow",
        "darkOrange",
        "#996300",
        "steelBlue",
        "darkBlue",
    ],
    scaling_factor=500,
    legend_fontsize=14,
):
    # Import shapefile, add centroid information
    shapefile = gpd.read_file(locFilePath)
    shapefile["centroid"] = shapefile.geometry.centroid

    # Subset, change NAs to 0s, Transpose, set indexColumn name in the property data
    property_subset = results_df.iloc[
        results_df.index.get_level_values("Property") == Property_to_plot
    ]

    property_subset = property_subset.droplevel(["Property", "Unit"]).fillna(0)
    property_subset = property_subset.transpose()
    property_subset.index.name = indexColumn_in_shp

    # Total property values in each region
    regional_property_sum = property_subset.sum(axis=1)

    fig, ax = plotLocations(
        locFilePath, plotLocNames=False, indexColumn=indexColumn_in_shp
    )
    ax.set_aspect("equal")

    Total_degree = 360

    for region in shapefile[indexColumn_in_shp]:  # LOOP OVER REGIONS
        xValue = float(
            shapefile.loc[shapefile[indexColumn_in_shp] == region].centroid.x.values
        )
        yValue = float(
            shapefile.loc[shapefile[indexColumn_in_shp] == region].centroid.y.values
        )

        total_property_value = regional_property_sum[region]

        theta1 = 0
        for i, component in enumerate(
            property_subset.columns
        ):  # LOOP OVER TECHNOLOGIES
            component_property_value = property_subset.loc[region, component]

            share = (component_property_value / total_property_value) * Total_degree

            theta2 = theta1 + share

            wedge = mpatches.Wedge(
                (xValue, yValue),
                total_property_value
                * (10 / property_subset.values.mean())
                * scaling_factor,  # radius
                theta1,  # theta1
                theta2,  # theta2
                fc=color_list[i],  # color
                lw=0.6,
                zorder=2,
                edgecolor="black",
            )
            theta1 = theta2

            ax.add_artist(wedge)

    # Legend
    handles = []
    for i, component in enumerate(property_subset.columns):
        component_patch = mpatches.Patch(color=color_list[i], label=component)

        handles.append(component_patch)

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=legend_fontsize,
    )
