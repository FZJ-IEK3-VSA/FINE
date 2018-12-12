"""
Last edited: May 13 2018

@author: Lara Welder
"""
import warnings
import pandas as pd


def isString(string):
    """ Checks if the input argument is a string """
    if not type(string) == str:
        raise TypeError("The input argument has to be a string")


def equalStrings(ref, test):
    """ Checks if two strings are equal to each other """
    if ref != test:
        print('Reference string: ' + str(ref))
        print('String: ' + str(test))
        raise ValueError('Strings do not match')


def isStrictlyPositiveInt(value):
    """ Checks if the input argument is a strictly positive integer """
    if not type(value) == int:
        raise TypeError("The input argument has to be an integer")
    if not value > 0:
        raise ValueError("The input argument has to be strictly positive")


def isStrictlyPositiveNumber(value):
    """ Checks if the input argument is a strictly positive number """
    if not (type(value) == float or type(value) == int):
        raise TypeError("The input argument has to be an number")
    if not value > 0:
        raise ValueError("The input argument has to be strictly positive")


def isSetOfStrings(setOfStrings):
    """
    Checks if the input argument is a set of strings
    """
    if not type(setOfStrings) == set:
        raise TypeError("The input argument has to be a list")
    if not any([type(r) == str for r in setOfStrings]):
        raise TypeError("The list entries in the input argument" +
                        " must be strings")


def checkTimeUnit(timeUnit):
    """
    Function used when an EnergySystemModel instance is initialized
    Checks if the timeUnit input argument is equal to 'h'
    """
    if not timeUnit == 'h':
        raise ValueError("The timeUnit input argument has to be \'h\'")


def checkTimeSeriesIndex(esM, data):
    if list(data.index) != esM._totalTimeSteps:
        raise ValueError('Time indices do not match the one of the specified energy system model.\n' +
                         'Data indicies: ' + str(set(data.index)) + '\n' +
                         'Energy system model time steps: ' + str(esM._timeSteps))


def checkRegionalColumnTitles(esM, data):
    if set(data.columns) != esM._locations:
        raise ValueError('Location indices do not match the one of the specified energy system model.\n' +
                         'Data columns: ' + str(set(data.columns)) + '\n' +
                         'Energy system model regions: ' + str(esM._locations))


def checkRegionalIndex(esM, data):
    if set(data.index) != esM._locations:
        raise ValueError('Location indices do not match the one of the specified energy system model.\n' +
                         'Data indicies: ' + str(set(data.index)) + '\n' +
                         'Energy system model regions: ' + str(esM._locations))


def checkCommodities(esM, commodity):
    if not commodity.issubset(esM._commodities):
        raise ValueError('Location indices do not match the one of the specified energy system model.\n' +
                         'Commodity: ' + str(set(commodity)) + '\n' +
                         'Energy system model regions: ' + str(esM._commodities))


def checkAndSetDistances(esM, distances):
    if distances is None:
        print('The distances of a component are set to a normalized values of 1.')
        return pd.DataFrame([[1 for loc in esM._locations] for loc in esM._locations],
                            index=esM._locations, columns=esM._locations)
    else:
        if not isinstance(distances, pd.DataFrame):
            raise TypeError('Input data has to be a pandas DataFrame')
        if (distances < 0).any().any():
            raise ValueError('distances values smaller than 0 were detected.')
        checkRegionalColumnTitles(esM, distances), checkRegionalIndex(esM, distances)
        return distances


def checkLocationSpecficDesignInputParams(esM, hasDesignDimensionVariables, hasDesignDecisionVariables,
                                          capacityMin, capacityMax, capacityFix,
                                          locationalEligibility, designDecisionFix, sharedPotentialID, dimension):
    for data in [capacityMin, capacityFix, capacityMax, locationalEligibility, designDecisionFix]:
        if data is not None:
            if dimension == '1dim':
                if not isinstance(data, pd.Series):
                    raise TypeError('Input data has to be a pandas Series')
                checkRegionalIndex(esM, data)
            elif dimension == '2dim':
                if not isinstance(data, pd.DataFrame):
                    raise TypeError('Input data has to be a pandas DataFrame')
                checkRegionalColumnTitles(esM, data), checkRegionalIndex(esM, data)
            else:
                raise ValueError("The dimension parameter has to be either \'1dim\' or \'2dim\' ")

    if capacityMin is not None and (capacityMin < 0).any().any():
        raise ValueError('capacityMin values smaller than 0 were detected.')

    if capacityFix is not None and (capacityFix < 0).any().any():
        raise ValueError('capacityFix values smaller than 0 were detected.')

    if capacityMax is not None and (capacityMax < 0).any().any():
        raise ValueError('capacityMax values smaller than 0 were detected.')

    if (capacityMin is not None or capacityMax is not None or capacityFix is not None) and not hasDesignDimensionVariables:
        raise ValueError('Capacity bounds are given but hasDesignDimensionVar was set to False.')

    if designDecisionFix is not None and not hasDesignDecisionVariables:
        raise ValueError('Fixed design decisions are given but hasDesignDecisionVariables was set to False.')

    if sharedPotentialID is not None and capacityMax is None:
        raise ValueError('A capacityMax parameter is required if a sharedPotentialID is considered.')

    if capacityMin is not None and capacityMax is not None:
        if (capacityMin > capacityMax).any().any():
            raise ValueError('capacityMin values > capacityMax values detected.')

    if capacityFix is not None and capacityMax is not None:
        if (capacityFix > capacityMax).any().any():
            raise ValueError('capacityFix values > capacityMax values detected.')

    if capacityFix is not None and capacityMin is not None:
        if (capacityFix < capacityMin).any().any():
            raise ValueError('capacityFix values < capacityMax values detected.')

    if locationalEligibility is not None:
        # Check if values are either one or zero
        if ((locationalEligibility != 0) & (locationalEligibility != 1)).any().any():
            raise ValueError('The locationEligibility entries have to be either 0 or 1.')
        # Check if given capacities indicate the same eligibility
        if capacityFix is not None:
            data = capacityFix.copy()
            data[data > 0] = 1
            if (data != locationalEligibility).any().any():
                raise ValueError('The locationEligibility and capacityFix parameters indicate different eligibilities.')
        if capacityMax is not None:
            data = capacityMax.copy()
            data[data > 0] = 1
            if (data != locationalEligibility).any().any():
                raise ValueError('The locationEligibility and capacityFix parameters indicate different eligibilities.')
        if capacityMin is not None:
            data = capacityMin.copy()
            data[data > 0] = 1
            if (data > locationalEligibility).any().any():
                raise ValueError('The locationEligibility and capacityFix parameters indicate different eligibilities.')
        if designDecisionFix is not None:
            if (designDecisionFix != locationalEligibility).any().any():
                raise ValueError('The locationEligibility and designDecisionFix parameters indicate different' +
                                 'eligibilities.')

    if designDecisionFix is not None:
        # Check if values are either one or zero
        if ((designDecisionFix != 0) & (designDecisionFix != 1)).any().any():
            raise ValueError('The designDecisionFix entries have to be either 0 or 1.')
        # Check if given capacities indicate the same design decisions
        if capacityFix is not None:
            data = capacityFix.copy()
            data[data > 0] = 1
            if (data > designDecisionFix).any().any():
                raise ValueError('The designDecisionFix and capacityFix parameters indicate different design decisions.')
        if capacityMax is not None:
            data = capacityMax.copy()
            data[data > 0] = 1
            if (data > designDecisionFix).any().any():
                warnings.warn('The designDecisionFix and capacityMax parameters indicate different design options.')
        if capacityMin is not None:
            data = capacityMin.copy()
            data[data > 0] = 1
            if (data > designDecisionFix).any().any():
                raise ValueError('The designDecisionFix and capacityMin parameters indicate different design decisions.')


def setLocationalEligibility(esM, locationalEligibility, capacityMax, capacityFix, designDecisionFix,
                             hasDesignDimensionVariables, operationTimeSeries, dimension='1dim'):
    if locationalEligibility is not None:
        return locationalEligibility
    else:
        # If the location eligibility is None set it based on other information available
        if not hasDesignDimensionVariables and operationTimeSeries is not None:
            if dimension == '1dim':
                data = operationTimeSeries.copy().sum()
                data[data > 0] = 1
                print('The locationalEligibility of a component was set based on the '
                      'given operation time series of the component.')
                return data
            elif dimension == '2dim':
                data = operationTimeSeries.copy().sum()
                data.loc[:] = 1
                data = data.unstack(level=-1).fillna(0)
                _locationalEligibility = pd.DataFrame([[0 for loc in esM._locations] for loc in esM._locations],
                                                      index=esM._locations, columns=esM._locations)
                _locationalEligibility.loc[data.index, data.columns] = data
                print('The locationalEligibility of a component was set based on the '
                      'given operation time series of the component.')
                return _locationalEligibility
            else:
                raise ValueError("The dimension parameter has to be either \'1dim\' or \'2dim\' ")
        elif capacityFix is None and capacityMax is None and designDecisionFix is None:
            # If no information is given all values are set to 1
            if dimension == '1dim':
                print('The locationalEligibility of a component is set to 1 (eligible) for all locations.')
                return pd.Series([1 for loc in esM._locations], index=esM._locations)
            else:
                print('The locationalEligibility of a component is set to 1 (eligible) for all locations.')
                return pd.DataFrame([[1 if loc != loc_ else 0 for loc in esM._locations] for loc_ in esM._locations],
                                    index=esM._locations, columns=esM._locations)
        elif designDecisionFix is not None:
            # If the designDecisionFix is not empty, the eligibility is set based on the fixed capacity
            data = designDecisionFix.copy()
            data[data > 0] = 1
            print('The locationalEligibility of a component was set based on the '
                  'given fixed design decisions of the component.')
            return data
        else:
            # If the fixCapacity is not empty, the eligibility is set based on the fixed capacity
            data = capacityFix.copy() if capacityFix is not None else capacityMax.copy()
            data[data > 0] = 1
            print('The locationalEligibility of a component was set based on the '
                  'given fixed/maximum capacity of the component.')
            return data


def checkOperationTimeSeriesInputParameters(esM, operationTimeSeries, locationalEligibility, dimension='1dim'):
    if operationTimeSeries is not None:
        if not isinstance(operationTimeSeries, pd.DataFrame):
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
            columns = operationTimeSeries.columns
            if columns.nlevels != 2:
                raise TypeError('The operation time series DataFrame of a location connecting component must have ' +
                                'two headers.\nThe first header must describe from which location the flow is coming' +
                                ' from and the second one the location to which the flow is going to.')
            if not set(operationTimeSeries.columns.get_level_values(level=0).unique()).issubset(esM._locations) or \
               not set(operationTimeSeries.columns.get_level_values(level=1).unique()).issubset(esM._locations):
                raise ValueError('Locations in column indices of a location connecting time series\n' +
                                 'detected which are not specified in the energy system model.')

            set1 = set(zip(list(columns.get_level_values(level=0)), list(columns.get_level_values(level=1))))
            set2 = set(zip(list(columns.get_level_values(level=1)), list(columns.get_level_values(level=0))))
            if not set1 == set2:
                raise ValueError('Missing data in time series DataFrame of a location connecting component.\n' +
                                 'If the flow is specified from loc1 to loc2, then it must also be specified\n' +
                                 'from loc2 to loc1.')

            if locationalEligibility is not None and operationTimeSeries is not None:
                # Check if given capacities indicate the same eligibility
                data = operationTimeSeries.copy().sum()
                data.loc[:] = 1
                data = data.unstack(level=-1).fillna(0)
                if (data > locationalEligibility.loc[data.index, data.columns]).any().any(0):
                    raise ValueError('The locationEligibility and operationTimeSeries parameters indicate different' +
                                     ' eligibilities.')

    if operationTimeSeries is not None and (operationTimeSeries < 0).any().any():
        raise ValueError('operationTimeSeries values smaller than 0 were detected.')


def checkDesignVariableModelingParameters(designDimensionVariableDomain, hasDesignDimensionVariables,
                                          hasDesignDecisionVariables, bigM):
    if designDimensionVariableDomain != 'continuous' and designDimensionVariableDomain != 'discrete':
        raise ValueError('The design dimension variable domain has to be either \'continuous\' or \'discrete\'.')

    if not isinstance(hasDesignDimensionVariables, bool):
        raise ValueError('The hasDesignDimensionVariables variable domain has to be a boolean.')

    if not isinstance(hasDesignDecisionVariables, bool):
        raise ValueError('The hasDesignDimensionVariables variable domain has to be a boolean.')

    if not hasDesignDimensionVariables and hasDesignDecisionVariables:
        raise ValueError('To consider additional fixed cost contributions when installing'
                         'capacities, design dimension variables are required.')

    if bigM is None and hasDesignDecisionVariables:
        raise ValueError('A bigM value needs to be specified when considering fixed cost contributions.')


def checkAndSetCostParameter(esM, name, data, dimension='1dim'):
    if dimension == '1dim':
        if not (isinstance(data, int) or isinstance(data, float) or isinstance(data, pd.Series)):
            raise TypeError('Type error in ' + name + ' detected.\n' +
                            'Economic parameters have to be a number or a pandas Series.')
    elif dimension == '2dim':
        if not (isinstance(data, int) or isinstance(data, float) or isinstance(data, pd.DataFrame)):
            raise TypeError('Type error in ' + name + ' detected.\n' +
                            'Economic parameters have to be a number or a pandas DataFrame.')
    else:
        raise ValueError("The dimension parameter has to be either \'1dim\' or \'2dim\' ")

    if dimension == '1dim':
        if isinstance(data, int) or isinstance(data, float):
            if data < 0:
                raise ValueError('Value error in ' + name + ' detected.\n Economic parameters have to be positive.')
            return pd.Series([float(data) for loc in esM._locations], index=esM._locations)
    else:
        if isinstance(data, int) or isinstance(data, float):
            if data < 0:
                raise ValueError('Value error in ' + name + ' detected.\n Economic parameters have to be positive.')
            return pd.DataFrame([[float(data) for loc in esM._locations] for loc in esM._locations],
                                index=esM._locations, columns=esM._locations)
    checkRegionalColumnTitles(esM, data), checkRegionalIndex(esM, data)
    _data = data.astype(float)
    if _data.isnull().any().any():
        raise ValueError('Value error in ' + name + ' detected.\n' +
                         'An economic parameter contains values which are not a number.')
    if (_data < 0).any().any():
        raise ValueError('Value error in ' + name + ' detected.\n' +
                         'All entries in economic parameter series have to be positive.')
    return _data


def setFormattedTimeSeries(timeSeries):
    if timeSeries is None:
        return timeSeries
    else:
        data = timeSeries.copy()
        data["Period"], data["TimeStep"] = 0, data.index
        return data.set_index(['Period', 'TimeStep'])
