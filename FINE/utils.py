"""
Last edited: May 13 2018

@author: Lara Welder
"""
import warnings
import pandas as pd
import FINE as fn

def isString(string):
    """ Check if the input argument is a string. """
    if not type(string) == str:
        raise TypeError("The input argument has to be a string")


def equalStrings(ref, test):
    """ Check if two strings are equal to each other. """
    if ref != test:
        print('Reference string: ' + str(ref))
        print('String: ' + str(test))
        raise ValueError('Strings do not match')


def isStrictlyPositiveInt(value):
    """ Check if the input argument is a strictly positive integer. """
    if not type(value) == int:
        raise TypeError("The input argument has to be an integer")
    if not value > 0:
        raise ValueError("The input argument has to be strictly positive")


def isStrictlyPositiveNumber(value):
    """ Check if the input argument is a strictly positive number. """
    if not (isinstance(value, float) or isinstance(value, int)):
        raise TypeError("The input argument has to be an number")
    if not value > 0:
        raise ValueError("The input argument has to be strictly positive")


def isPositiveNumber(value):
    """ Check if the input argument is a positive number. """
    if not (isinstance(value, float) or isinstance(value, int)):
        raise TypeError("The input argument has to be an number")
    if not value >= 0:
        raise ValueError("The input argument has to be positive")


def isSetOfStrings(setOfStrings):
    """ Check if the input argument is a set of strings. """
    if not type(setOfStrings) == set:
        raise TypeError("The input argument has to be a set")
    if not any([type(r) == str for r in setOfStrings]):
        raise TypeError("The list entries in the input argument must be strings")


def isEnergySystemModelInstance(esM):
    if not isinstance(esM, fn.EnergySystemModel):
        raise TypeError('The input is not an EnergySystemModel instance.')


def checkEnergySystemModelInput(locations, commodities, commodityUnitsDict, numberOfTimeSteps, hoursPerTimeStep,
                                costUnit, lengthUnit):
    """ Check input arguments of an EnergySystemModel instance for value/type correctness. """

    # Locations and commodities have to be sets
    isSetOfStrings(locations), isSetOfStrings(commodities)

    # The commodityUnitDict has to be a dictionary which keys equal the specified commodities and which values are
    # strings
    if not type(commodityUnitsDict) == dict:
        raise TypeError("The commodityUnitsDict input argument has to be a dictionary.")
    if commodities != set(commodityUnitsDict.keys()):
        raise ValueError("The keys of the commodityUnitDict must equal the specified commodities.")
    isSetOfStrings(set(commodityUnitsDict.values()))

    # The numberOfTimeSteps and the hoursPerTimeStep have to be strictly positive numbers
    isStrictlyPositiveInt(numberOfTimeSteps), isStrictlyPositiveNumber(hoursPerTimeStep)

    # The costUnit and lengthUnit input parameter have to be strings
    isString(costUnit), isString(lengthUnit)


def checkTimeUnit(timeUnit):
    """
    Check if the timeUnit input argument is equal to 'h'.
    """
    if not timeUnit == 'h':
        raise ValueError("The timeUnit input argument has to be \'h\'")


def checkTimeSeriesIndex(esM, data):
    """
    Necessary if the data rows represent the time-dependent data:
    Check if the row-indices of the data match the time indices of the energy system model.
    """
    if list(data.index) != esM.totalTimeSteps:
        raise ValueError('Time indices do not match the one of the specified energy system model.\n' +
                         'Data indices: ' + str(set(data.index)) + '\n' +
                         'Energy system model time steps: ' + str(esM._timeSteps))


def checkRegionalColumnTitles(esM, data):
    """
    Necessary if the data columns represent the location-dependent data:
    Check if the columns indices match the location indices of the energy system model.
    """
    if set(data.columns) != esM.locations:
        raise ValueError('Location indices do not match the one of the specified energy system model.\n' +
                         'Data columns: ' + str(set(data.columns)) + '\n' +
                         'Energy system model regions: ' + str(esM.locations))


def checkRegionalIndex(esM, data):
    """
    Necessary if the data rows represent the location-dependent data:
    Check if the row-indices match the location indices of the energy system model.
    """
    if set(data.index) != esM.locations:
        raise ValueError('Location indices do not match the one of the specified energy system model.\n' +
                         'Data indices: ' + str(set(data.index)) + '\n' +
                         'Energy system model regions: ' + str(esM.locations))


def checkConnectionIndex(data, locationalEligibility):
    """
    Necessary for transmission components:
    Check if the indices of the connection data match the eligible connections.
    """
    if not set(data.index) == set(locationalEligibility.index):
        raise ValueError('Indices do not match the eligible connections of the component.\n' +
                         'Data indices: ' + str(set(data.index)) + '\n' +
                         'Eligible connections: ' + str(set(locationalEligibility.index)))


def checkCommodities(esM, commodities):
    """ Check if the commodity is considered in the energy system model. """
    if not commodities.issubset(esM.commodities):
        raise ValueError('Commodity does not match the ones of the specified energy system model.\n' +
                         'Commodity: ' + str(set(commodities)) + '\n' +
                         'Energy system model commodities: ' + str(esM.commodities))


def checkCommodityUnits(esM, commodityUnit):
    """ Check if the commodity unit matches the in the energy system model defined commodity units."""
    if not commodityUnit in esM.commodityUnitsDict.values():
        raise ValueError('Commodity unit does not match the ones of the specified energy system model.\n' +
                         'Commodity unit: ' + str(commodityUnit) + '\n' +
                         'Energy system model commodityUnits: ' + str(esM.commodityUnitsDict.values()))


def checkAndSetDistances(distances, locationalEligibility, esM):
    """
    Check if the given values for the distances are valid (i.e. positive). If the distances parameter is None,
    the distances for the eligible connections are set to 1.
    """
    if distances is None:
        output('The distances of a component are set to a normalized value of 1.', esM.verbose, 0)
        distances = pd.Series([1 for loc in locationalEligibility.index], index=locationalEligibility.index)
    else:
        if not isinstance(distances, pd.Series):
            raise TypeError('Input data has to be a pandas DataFrame or Series')
        if (distances < 0).any():
            raise ValueError('Distance values smaller than 0 were detected.')
        checkConnectionIndex(distances, locationalEligibility)
    return distances


def checkAndSetTransmissionLosses(losses, distances, locationalEligibility):
    """
    Check if the type of the losses are valid (i.e. a number, pandas DataFrame or a pandas Series),
    and if the given values for the losses of the transmission component are valid (i.e. between 0 and 1).
    """
    if not (isinstance(losses, int) or isinstance(losses, float) or isinstance(losses, pd.DataFrame)
            or isinstance(losses, pd.Series)):
        raise TypeError('The input data has to be a number, a pandas DataFrame or a pandas Series.')

    if isinstance(losses, int) or isinstance(losses, float):
        if losses < 0 or losses > 1:
            raise ValueError('Losses have to be values between 0 <= losses <= 1.')
        return pd.Series([float(losses) for loc in locationalEligibility.index], index=locationalEligibility.index)
    checkConnectionIndex(losses, locationalEligibility)

    losses = losses.astype(float)
    if losses.isnull().any():
        raise ValueError('The losses parameter contains values which are not a number.')
    if (losses < 0).any() or (losses > 1).any():
            raise ValueError('Losses have to be values between 0 <= losses <= 1.')
    if (1-losses*distances < 0).any():
        raise ValueError('The losses per distance multiplied with the distances result in negative values.')

    return losses


def getCapitalChargeFactor(interestRate, economicLifetime):
    """ Compute and return capital charge factor (inverse of annuity factor). """
    CCF = 1 / interestRate - 1 / (pow(1 + interestRate, economicLifetime) * interestRate)
    CCF = CCF.fillna(economicLifetime)
    return CCF


def castToSeries(data, esM):
    if data is None:
        return None
    elif isinstance(data, pd.Series):
        return data
    isPositiveNumber(data)
    return pd.Series([data], index=list(esM.locations))


def checkLocationSpecficDesignInputParams(comp, esM):
    if len(esM.locations) == 1:
        comp.capacityMin = castToSeries(comp.capacityMin, esM)
        comp.capacityFix = castToSeries(comp.capacityFix, esM)
        comp.capacityMax = castToSeries(comp.capacityMax, esM)
        comp.locationalEligibility = castToSeries(comp.locationalEligibility, esM)
        comp.isBuiltFix = castToSeries(comp.isBuiltFix, esM)

    capacityMin, capacityFix, capacityMax = comp.capacityMin, comp.capacityFix, comp.capacityMax
    locationalEligibility, isBuiltFix = comp.locationalEligibility, comp.isBuiltFix
    hasCapacityVariable, hasIsBuiltBinaryVariable = comp.hasCapacityVariable, comp.hasIsBuiltBinaryVariable
    sharedPotentialID = comp.sharedPotentialID

    for data in [capacityMin, capacityFix, capacityMax, locationalEligibility, isBuiltFix]:
        if data is not None:
            if comp.dimension == '1dim':
                if not isinstance(data, pd.Series):
                    raise TypeError('Input data has to be a pandas Series')
                checkRegionalIndex(esM, data)
            elif comp.dimension == '2dim':
                if not isinstance(data, pd.Series):
                    raise TypeError('Input data has to be a pandas DataFrame')
                checkConnectionIndex(data, comp.locationalEligibility)
            else:
                raise ValueError("The dimension parameter has to be either \'1dim\' or \'2dim\' ")

    if capacityMin is not None and (capacityMin < 0).any():
        raise ValueError('capacityMin values smaller than 0 were detected.')

    if capacityFix is not None and (capacityFix < 0).any():
        raise ValueError('capacityFix values smaller than 0 were detected.')

    if capacityMax is not None and (capacityMax < 0).any():
        raise ValueError('capacityMax values smaller than 0 were detected.')

    if (capacityMin is not None or capacityMax is not None or capacityFix is not None) and not hasCapacityVariable:
        raise ValueError('Capacity bounds are given but hasDesignDimensionVar was set to False.')

    if isBuiltFix is not None and not hasIsBuiltBinaryVariable:
        raise ValueError('Fixed design decisions are given but hasIsBuiltBinaryVariable was set to False.')

    if sharedPotentialID is not None:
        isString(sharedPotentialID)

    if sharedPotentialID is not None and capacityMax is None:
        raise ValueError('A capacityMax parameter is required if a sharedPotentialID is considered.')

    if capacityMin is not None and capacityMax is not None:
        if (capacityMin > capacityMax).any():
            raise ValueError('capacityMin values > capacityMax values detected.')

    if capacityFix is not None and capacityMax is not None:
        if (capacityFix > capacityMax).any():
            raise ValueError('capacityFix values > capacityMax values detected.')

    if capacityFix is not None and capacityMin is not None:
        if (capacityFix < capacityMin).any():
            raise ValueError('capacityFix values < capacityMax values detected.')

    if locationalEligibility is not None:
        # Check if values are either one or zero
        if ((locationalEligibility != 0) & (locationalEligibility != 1)).any():
            raise ValueError('The locationEligibility entries have to be either 0 or 1.')
        # Check if given capacities indicate the same eligibility
        if capacityFix is not None:
            data = capacityFix.copy()
            data[data > 0] = 1
            if (data != locationalEligibility).any():
                raise ValueError('The locationEligibility and capacityFix parameters indicate different eligibilities.')
        if capacityMax is not None:
            data = capacityMax.copy()
            data[data > 0] = 1
            if (data != locationalEligibility).any():
                raise ValueError('The locationEligibility and capacityMax parameters indicate different eligibilities.')
        if capacityMin is not None:
            data = capacityMin.copy()
            data[data > 0] = 1
            if (data > locationalEligibility).any():
                raise ValueError('The locationEligibility and capacityMin parameters indicate different eligibilities.')
        if isBuiltFix is not None:
            if (isBuiltFix != locationalEligibility).any():
                raise ValueError('The locationEligibility and isBuiltFix parameters indicate different' +
                                 'eligibilities.')

    if isBuiltFix is not None:
        # Check if values are either one or zero
        if ((isBuiltFix != 0) & (isBuiltFix != 1)).any():
            raise ValueError('The isBuiltFix entries have to be either 0 or 1.')
        # Check if given capacities indicate the same design decisions
        if capacityFix is not None:
            data = capacityFix.copy()
            data[data > 0] = 1
            if (data > isBuiltFix).any():
                raise ValueError('The isBuiltFix and capacityFix parameters indicate different design decisions.')
        if capacityMax is not None:
            data = capacityMax.copy()
            data[data > 0] = 1
            if (data > isBuiltFix).any():
                if esM.verbose < 2:
                    warnings.warn('The isBuiltFix and capacityMax parameters indicate different design options.')
        if capacityMin is not None:
            data = capacityMin.copy()
            data[data > 0] = 1
            if (data > isBuiltFix).any():
                raise ValueError('The isBuiltFix and capacityMin parameters indicate different design decisions.')


def setLocationalEligibility(esM, locationalEligibility, capacityMax, capacityFix, isBuiltFix,
                             hasCapacityVariable, operationTimeSeries, dimension='1dim'):
    if locationalEligibility is not None:
        return locationalEligibility
    else:
        # If the location eligibility is None set it based on other information available
        if not hasCapacityVariable and operationTimeSeries is not None:
            if dimension == '1dim':
                data = operationTimeSeries.copy().sum()
                data[data > 0] = 1
                return data
            elif dimension == '2dim':
                data = operationTimeSeries.copy().sum()
                data.loc[:] = 1
                locationalEligibility = data
                return locationalEligibility
            else:
                raise ValueError("The dimension parameter has to be either \'1dim\' or \'2dim\' ")
        elif capacityFix is None and capacityMax is None and isBuiltFix is None:
            # If no information is given all values are set to 1
            if dimension == '1dim':
                return pd.Series([1 for loc in esM.locations], index=esM.locations)
            else:
                keys = {loc1 + '_' + loc2 for loc1 in esM.locations for loc2 in esM.locations if loc1 != loc2}
                return pd.Series([1 for key in keys], index=keys)
        elif isBuiltFix is not None:
            # If the isBuiltFix is not empty, the eligibility is set based on the fixed capacity
            data = isBuiltFix.copy()
            data[data > 0] = 1
            return data
        else:
            # If the fixCapacity is not empty, the eligibility is set based on the fixed capacity
            data = capacityFix.copy() if capacityFix is not None else capacityMax.copy()
            data[data > 0] = 1
            return data


def checkAndSetTimeSeries(esM, operationTimeSeries, locationalEligibility, dimension='1dim'):
    if operationTimeSeries is not None:
        if not isinstance(operationTimeSeries, pd.DataFrame):
            if len(esM.locations) == 1:
                if isinstance(operationTimeSeries, pd.Series):
                    operationTimeSeries = pd.DataFrame(operationTimeSeries.values, index=operationTimeSeries.index,
                                                       columns=list(esM.locations))
                else:
                    raise TypeError('The operation time series data type has to be a pandas DataFrame or Series')
            else:
                raise TypeError('The operation time series data type has to be a pandas DataFrame')
        checkTimeSeriesIndex(esM, operationTimeSeries)

        if dimension == '1dim':
            checkRegionalColumnTitles(esM, operationTimeSeries)

            if locationalEligibility is not None and operationTimeSeries is not None:
                # Check if given capacities indicate the same eligibility
                data = operationTimeSeries.copy().sum()
                data[data > 0] = 1
                if (data > locationalEligibility).any().any():
                    raise ValueError('The locationEligibility and operationTimeSeries parameters indicate different' +
                                     ' eligibilities.')
        elif dimension == '2dim':
            keys = {loc1 + '_' + loc2 for loc1 in esM.locations for loc2 in esM.locations}
            columns = set(operationTimeSeries.columns)
            if not columns <= keys:
                raise ValueError('False column index detected in time series. ' +
                                 'The indicies have to be in the format \'loc1_loc2\' ' +
                                 'with loc1 and loc2 being locations in the energy system model.')

            for loc1 in esM.locations:
                for loc2 in esM.locations:
                    if loc1 + '_' + loc2 in columns and not loc2 + '_' + loc1 in columns:
                        raise ValueError('Missing data in time series DataFrame of a location connecting \n' +
                                         'component. If the flow is specified from loc1 to loc2, \n' +
                                         'then it must also be specified from loc2 to loc1.\n')

            if locationalEligibility is not None and operationTimeSeries is not None:
                # Check if given capacities indicate the same eligibility
                keys = set(locationalEligibility.index)
                if not columns == keys:
                    raise ValueError('The locationEligibility and operationTimeSeries parameters indicate different' +
                                     ' eligibilities.')

        if (operationTimeSeries < 0).any().any():
            raise ValueError('operationTimeSeries values smaller than 0 were detected.')

        operationTimeSeries = operationTimeSeries.copy()
        operationTimeSeries["Period"], operationTimeSeries["TimeStep"] = 0, operationTimeSeries.index
        return operationTimeSeries.set_index(['Period', 'TimeStep'])
    else:
        return None


def checkDesignVariableModelingParameters(capacityVariableDomain, hasCapacityVariable, capacityPerPlantUnit,
                                          hasIsBuiltBinaryVariable, bigM):
    if capacityVariableDomain != 'continuous' and capacityVariableDomain != 'discrete':
        raise ValueError('The capacity variable domain has to be either \'continuous\' or \'discrete\'.')

    if not isinstance(hasIsBuiltBinaryVariable, bool):
        raise TypeError('The hasCapacityVariable variable domain has to be a boolean.')

    isStrictlyPositiveNumber(capacityPerPlantUnit)

    if not hasCapacityVariable and hasIsBuiltBinaryVariable:
        raise ValueError('To consider additional fixed cost contributions when installing\n' +
                         'capacities, capacity variables are required.')

    if bigM is None and hasIsBuiltBinaryVariable:
        raise ValueError('A bigM value needs to be specified when considering fixed cost contributions.')

    if bigM is not None:
        isinstance(bigM, bool)


def checkAndSetCostParameter(esM, name, data, dimension, locationEligibility):
    if dimension == '1dim':
        if not (isinstance(data, int) or isinstance(data, float) or isinstance(data, pd.Series)):
            raise TypeError('Type error in ' + name + ' detected.\n' +
                            'Economic parameters have to be a number or a pandas Series.')
    elif dimension == '2dim':
        if not (isinstance(data, int) or isinstance(data, float) or isinstance(data, pd.Series)):
            raise TypeError('Type error in ' + name + ' detected.\n' +
                            'Economic parameters have to be a number or a pandas Series.')
    else:
        raise ValueError("The dimension parameter has to be either \'1dim\' or \'2dim\' ")

    if dimension == '1dim':
        if isinstance(data, int) or isinstance(data, float):
            if data < 0:
                raise ValueError('Value error in ' + name + ' detected.\n Economic parameters have to be positive.')
            return pd.Series([float(data) for loc in esM.locations], index=esM.locations)
        checkRegionalIndex(esM, data)
    else:
        if isinstance(data, int) or isinstance(data, float):
            if data < 0:
                raise ValueError('Value error in ' + name + ' detected.\n Economic parameters have to be positive.')
            return pd.Series([float(data) for loc in locationEligibility.index], index=locationEligibility.index)
        checkConnectionIndex(data, locationEligibility)

    _data = data.astype(float)
    if _data.isnull().any():
        raise ValueError('Value error in ' + name + ' detected.\n' +
                         'An economic parameter contains values which are not numbers.')
    if (_data < 0).any():
        raise ValueError('Value error in ' + name + ' detected.\n' +
                         'All entries in economic parameter series have to be positive.')
    return _data


def checkClusteringInput(numberOfTypicalPeriods, numberOfTimeStepsPerPeriod, totalNumberOfTimeSteps):
    isStrictlyPositiveInt(numberOfTypicalPeriods), isStrictlyPositiveInt(numberOfTimeStepsPerPeriod)
    if not totalNumberOfTimeSteps % numberOfTimeStepsPerPeriod == 0:
        raise ValueError('The numberOfTimeStepsPerPeriod has to be an integer divisor of the total number of time\n' +
                         ' steps considered in the energy system model.')
    if totalNumberOfTimeSteps < numberOfTypicalPeriods * numberOfTimeStepsPerPeriod:
        raise ValueError('The product of the numberOfTypicalPeriods and the numberOfTimeStepsPerPeriod has to be \n' +
                         'smaller than the total number of time steps considered in the energy system model.')


def checkDeclareOptimizationProblemInput(timeSeriesAggregation, isTimeSeriesDataClustered):
    if not isinstance(timeSeriesAggregation, bool):
        raise TypeError('The timeSeriesAggregation parameter has to be a boolean.')

    if timeSeriesAggregation and not isTimeSeriesDataClustered:
        raise ValueError('The time series flag indicates possible inconsistencies in the aggregated time series '
                         ' data.\n--> Call the cluster function first, then the optimize function.')


def checkOptimizeInput(timeSeriesAggregation, isTimeSeriesDataClustered, logFileName, threads, solver,
                       timeLimit, optimizationSpecs, warmstart):
    checkDeclareOptimizationProblemInput(timeSeriesAggregation, isTimeSeriesDataClustered)

    if not isinstance(logFileName, str):
        raise TypeError('The logFileName parameter has to be a string.')

    if not isinstance(threads, int) or threads < 0:
        raise TypeError('The threads parameter has to be a nonnegative integer.')

    if not isinstance(solver, str):
        raise TypeError('The solver parameter has to be a string.')

    if timeLimit is not None:
        isStrictlyPositiveNumber(timeLimit)

    if not isinstance(optimizationSpecs, str):
        raise TypeError('The optimizationSpecs parameter has to be a string.')

    if not isinstance(warmstart, bool):
        raise ValueError('The warmstart parameter has to be a boolean.')


def setFormattedTimeSeries(timeSeries):
    if timeSeries is None:
        return timeSeries
    else:
        data = timeSeries.copy()
        data["Period"], data["TimeStep"] = 0, data.index
        return data.set_index(['Period', 'TimeStep'])


def buildFullTimeSeries(df, periodsOrder):
    data = []
    for p in periodsOrder:
        data.append(df.loc[p])
    return pd.concat(data, axis=1, ignore_index=True)


def formatOptimizationOutput(data, varType, dimension, periodsOrder=None, compDict=None):
    # If data is an empty dictionary (because no variables of that type were declared) return None
    if not data:
        return None
    # If the dictionary is not empty, format it into a DataFrame
    if varType == 'designVariables' and dimension == '1dim':
        # Convert dictionary to DataFrame, transpose, put the components name first and sort the index
        # Results in a one dimensional DataFrame
        df = pd.DataFrame(data, index=[0]).T.swaplevel(i=0, j=1, axis=0).sort_index()
        # Unstack the regions (convert to a two dimensional DataFrame with the region indices being the columns)
        # and fill NaN values (i.e. when a component variable was not initiated for that region)
        df = df.unstack(level=-1)
        # Get rid of the unnecessary 0 level
        df.columns = df.columns.droplevel()
        return df
    elif varType == 'designVariables' and dimension == '2dim':
        # Convert dictionary to DataFrame, transpose, put the components name first while keeping the order of the
        # regions and sort the index
        # Results in a one dimensional DataFrame
        df = pd.DataFrame(data, index=[0]).T
        indexNew = []
        for tup in df.index.tolist():
            loc1, loc2 = compDict[tup[1]]._mapC[tup[0]]
            indexNew.append((loc1, loc2, tup[1]))
        df.index = pd.MultiIndex.from_tuples(indexNew)
        df = df.swaplevel(i=0, j=2, axis=0).swaplevel(i=1, j=2, axis=0).sort_index()
        # Unstack the regions (convert to a two dimensional DataFrame with the region indices being the columns)
        # and fill NaN values (i.e. when a component variable was not initiated for that region)
        df = df.unstack(level=-1)
        # Get rid of the unnecessary 0 level
        df.columns = df.columns.droplevel()
        return df
    elif varType == 'operationVariables' and dimension == '1dim':
        # Convert dictionary to DataFrame, transpose, put the period column first and sort the index
        # Results in a one dimensional DataFrame
        df = pd.DataFrame(data, index=[0]).T.swaplevel(i=0, j=-2).sort_index()
        # Unstack the time steps (convert to a two dimensional DataFrame with the time indices being the columns)
        df = df.unstack(level=-1)
        # Get rid of the unnecessary 0 level
        df.columns = df.columns.droplevel()
        # Re-engineer full time series by using Pandas' concat method (only one loop if time series aggregation was not
        # used)
        return buildFullTimeSeries(df, periodsOrder)
    elif varType == 'operationVariables' and dimension == '2dim':
        # Convert dictionary to DataFrame, transpose, put the period column first while keeping the order of the
        # regions and sort the index
        # Results in a one dimensional DataFrame
        df = pd.DataFrame(data, index=[0]).T
        indexNew = []
        for tup in df.index.tolist():
            loc1, loc2 = compDict[tup[1]]._mapC[tup[0]]
            indexNew.append((loc1, loc2, tup[1], tup[2], tup[3]))
        df.index = pd.MultiIndex.from_tuples(indexNew)
        df = df.swaplevel(i=1, j=2, axis=0).swaplevel(i=0, j=3, axis=0).swaplevel(i=2, j=3, axis=0).sort_index()
        # Unstack the time steps (convert to a two dimensional DataFrame with the time indices being the columns)
        df = df.unstack(level=-1)
        # Get rid of the unnecessary 0 level
        df.columns = df.columns.droplevel()
        # Re-engineer full time series by using Pandas' concat method (only one loop if time series aggregation was not
        # used)
        return buildFullTimeSeries(df, periodsOrder)
    else:
        raise ValueError('The varType parameter has to be either \'designVariables\' or \'operationVariables\'\n' +
                         'and the dimension parameter has to be either \'1dim\' or \'2dim\'.')


def setOptimalComponentVariables(optVal, varType, compDict):
    if optVal is not None:
        for compName, comp in compDict.items():
            if compName in optVal.index:
                setattr(comp, varType, optVal.loc[compName])
            else:
                setattr(comp, varType, None)


def preprocess2dimData(data, mapC=None):
    if data is not None and isinstance(data, pd.DataFrame):
        if mapC is None:
            index, data_ = [], []
            for loc1 in data.index:
                for loc2 in data.columns:
                    if data[loc1][loc2] > 0:
                        index.append(loc1 + '_' + loc2), data_.append(data[loc1][loc2])
            return pd.Series(data_, index=index)
        else:
            return pd.Series(mapC).apply(lambda loc: data[loc[0]][loc[1]])
    else:
        return data


def map2dimData(data, mapC):
    if data is not None and isinstance(data, pd.DataFrame):
        return pd.Series(mapC).apply(lambda loc: data[loc[0]][loc[1]])
    else:
        return data


def output(output, verbose, val):
    if verbose == val:
        print(output)

def checkModelClassEquality(esM, file):
    mdlListFromModel = list(esM.componentModelingDict.keys())
    mdlListFromExcel = []
    for sheet in file.sheet_names:
        mdlListFromExcel += [cl for cl in mdlListFromModel if (cl[0:-5] in sheet and cl not in mdlListFromExcel)]
    if set(mdlListFromModel) != set(mdlListFromExcel):
        raise ValueError('Loaded Output does not match the given energy system model.')

def checkComponentsEquality(esM, file):
    compListFromExcel = []
    compListFromModel = list(esM.componentNames.keys())
    for mdl in esM.componentModelingDict.keys():
        readSheet = pd.read_excel(file, sheetname=mdl[0:-5] + 'OptSummary', index_col=[0, 1, 2, 3])
        compListFromExcel += list(readSheet.index.levels[0])
    if not set(compListFromExcel) <= set(compListFromModel):
            raise ValueError('Loaded Output does not match the given energy system model.')
