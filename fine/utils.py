import logging
import math
import warnings

import numpy as np
import pandas as pd
import gurobipy as gp

import fine as fn
from fine.enums import Dimension, VarType


def checkAndSetBalanceLimitID(balanceLimitID):
    """Missing."""
    if balanceLimitID is None or isinstance(balanceLimitID, str):
        return balanceLimitID
    raise ValueError("The input argument needs to be a string or None.")


def checkCapacityOrCommissioningTransmission(df):
    """MISSING."""
    if isinstance(df, (pd.DataFrame, pd.Series, dict, float, int)):
        return df
    if df is None:
        return df
    raise ValueError(
        "The input argument needs to be a dataframe, series, dict, float or int."
    )


def isInRange(value, lowerBound, upperBound):
    """Check if the input value is in the given range."""
    if not (isinstance(value, float) or isinstance(value, int)):
        raise TypeError("The input argument has to be a number")
    if value <= upperBound and value >= lowerBound:
        return value
    raise ValueError(
        f"The input argument has to be in the range [{lowerBound},{upperBound}]"
    )


def isString(string):
    """Check if the input argument is a string."""
    if not isinstance(string, str):
        raise TypeError("The input argument has to be a string")


def isStrictlyPositiveInt(value):
    """Check if the input argument is a strictly positive integer."""
    if not isinstance(value, int):
        raise TypeError("The input argument has to be an integer")
    if not value > 0:
        raise ValueError("The input argument has to be strictly positive")


def isStrictlyPositiveNumber(value):
    """Check if the input argument is a strictly positive number."""
    if not (isinstance(value, float) or isinstance(value, int)):
        raise TypeError("The input argument has to be an number")
    if not value > 0:
        raise ValueError("The input argument has to be strictly positive")


def isPositiveNumber(value):
    """Check if the input argument is a positive number."""
    if not (isinstance(value, float) or isinstance(value, int)):
        raise TypeError("The input argument has to be an number")
    if not value >= 0:
        raise ValueError("The input argument has to be positive")


def isSetOfStrings(setOfStrings):
    """Check if the input argument is a set of strings."""
    if not isinstance(setOfStrings, set):
        raise TypeError("The input argument has to be a set")
    if not any([isinstance(currentString, str) for currentString in setOfStrings]):
        raise TypeError("The list entries in the input argument must be strings")


def isEnergySystemModelInstance(esM):
    """Check if input is an EnergySystemModel instance."""
    if not isinstance(esM, fn.EnergySystemModel):
        raise TypeError("The input is not an EnergySystemModel instance.")


def checkEnergySystemModelInput(
    locations,
    commodities,
    commodityUnitsDict,
    numberOfTimeSteps,
    hoursPerTimeStep,
    numberOfInvestmentPeriods,
    investmentPeriodInterval,
    startyear,
    stochasticModel,
    costUnit,
    lengthUnit,
):
    """Check input arguments of an EnergySystemModel instance for value/type correctness."""
    # Locations and commodities have to be sets
    isSetOfStrings(locations), isSetOfStrings(commodities)

    # The commodityUnitDict has to be a dictionary which keys equal the specified commodities and which values are
    # strings
    if not isinstance(commodityUnitsDict, dict):
        raise TypeError("The commodityUnitsDict input argument has to be a dictionary.")
    if commodities != set(commodityUnitsDict.keys()):
        raise ValueError(
            "The keys of the commodityUnitDict must equal the specified commodities."
        )
    isSetOfStrings(set(commodityUnitsDict.values()))

    isStrictlyPositiveInt(numberOfTimeSteps), isStrictlyPositiveNumber(hoursPerTimeStep)

    # check transformation path variables and mode
    if not isinstance(startyear, int):
        raise TypeError("Startyear must be an integer")

    isStrictlyPositiveInt(numberOfInvestmentPeriods)
    isStrictlyPositiveNumber(investmentPeriodInterval)

    if numberOfInvestmentPeriods == 1 and investmentPeriodInterval > 1:
        warnings.warn(
            "Energy system model has only one investment period. However the investment period "
            + f"interval is set to {investmentPeriodInterval}. This may results in a higher objective value. "
        )

    if stochasticModel and numberOfInvestmentPeriods == 1:
        raise ValueError(
            "A stochastic optimization needs more than one numberOfInvestmentPeriod"
        )

    # The costUnit and lengthUnit input parameter have to be strings
    isString(costUnit), isString(lengthUnit)


def checkTimeSeriesIndex(esM, data):
    """Necessary if the data rows represent the time-dependent data:
    Check if the row-indices of the data match the time indices of the energy system model.
    """
    if list(data.index) != esM.totalTimeSteps:
        raise ValueError(
            "Time indices do not match the one of the specified energy system model."
        )


def checkRegionalColumnTitles(esM, data, locationalEligibility):
    """Necessary if the data columns represent the location-dependent data:
    Check if the columns indices match the location indices of the energy system model.
    """
    # If its a single node esM set up via netCDF file, time series data is
    # pd.series with multiindex columns. First column index is the variables's
    # name. This needs to be dropped before checking Column Titles.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel()

    if set(data.columns) != esM.locations:
        # if the data locations do not match the esm locations:
        # force user to pass locationalEligibility if it is None.
        # if locationalEligibility is already passed, simply add 0s to data in missing locations
        # in a later stage this data is crosschecked with locationalEligibility to see if they match
        if locationalEligibility is None:
            raise ValueError(
                "Location indices do not match the one of the specified energy system model.\n"
                + "Data columns: "
                + str(set(data.columns))
                + "\n"
                + "Energy system model regions: "
                + str(esM.locations)
                + "If this was intentional, please provide locationalEligibility to cross-check."
            )
        data = addEmptyRegions(esM, data)

    # Sort data according to _locationsOrdered, if not already sorted
    if not np.array_equal(data.columns, esM._locationsOrdered):
        data.sort_index(inplace=True, axis=1)

    return data


def checkRegionalIndex(esM, data, locationalEligibility):
    """Necessary if the data rows represent the location-dependent data:
    Check if the row-indices match the location indices of the energy system model.
    """
    if set(data.index) != esM.locations:
        # if the data locations do not match the esm locations:
        # force user to pass locationalEligibility if it is None.
        # if locationalEligibility is already passed, simply add 0s to data in missing locations
        # in a later stage this data is crosschecked with locationalEligibility to see if they match
        if locationalEligibility is None:
            raise ValueError(
                "Location indices do not match the one of the specified energy system model.\n"
                + "Data indices: "
                + str(set(data.index))
                + "\n"
                + "Energy system model regions: "
                + str(esM.locations)
                + "If this was intentional, please provide locationalEligibility to cross-check."
            )
        data = addEmptyRegions(esM, data)

    # Sort data according to _locationsOrdered, if not already sorted
    if not np.array_equal(data.index, esM._locationsOrdered):
        data.sort_index(inplace=True)

    return data


def checkConnectionIndex(data, locationalEligibility):
    """Necessary for transmission components:
    Check if the indices of the connection data match the eligible connections.
    """
    if not set(data.index).issubset(locationalEligibility.index):
        raise ValueError(
            "Indices do not match the eligible connections of the component.\n"
            + "Data indices: "
            + str(set(data.index))
            + "\n"
            + "Eligible connections: "
            + str(set(locationalEligibility.index))
        )
    # Sort data according to _locationsOrdered, if not already sorted
    if not np.array_equal(data.index, locationalEligibility.index):
        data = data.reindex(locationalEligibility.index).fillna(0)

    return data


def checkCommodities(esM, commodities):
    """Check if the commodity is considered in the energy system model."""
    if not commodities.issubset(esM.commodities):
        raise ValueError(
            "Commodity does not match the ones of the specified energy system model.\n"
            + "Commodity: "
            + str(set(commodities))
            + "\n"
            + "Energy system model commodities: "
            + str(esM.commodities)
        )


def checkCommodityUnits(esM, commodityUnit):
    """Check if the commodity unit matches the in the energy system model defined commodity units."""
    if commodityUnit not in esM.commodityUnitsDict.values():
        raise ValueError(
            "Commodity unit does not match the ones of the specified energy system model.\n"
            + "Commodity unit: "
            + str(commodityUnit)
            + "\n"
            + "Energy system model commodityUnits: "
            + str(esM.commodityUnitsDict.values())
        )


def checkCallableConversionFactor(conversionFactor):
    """Check if the callable conversion factor includes only conversion factors greater than 0 in the relevant part load range."""
    nPointsForTesting = 1001
    xTest = np.linspace(0, 1, nPointsForTesting)
    yTest = [conversionFactor(xTest_i) for xTest_i in xTest]

    if any(yTest_i <= 0 for yTest_i in yTest):
        raise ValueError(
            "The callable part load conversion factor is smaller or equal to 0 at least once within [0,1]."
        )


def checkDataFrameConversionFactor(conversionFactor):
    """Check if the callable conversion factor covers part loads from 0 to 1 and
    if it includes only conversion factors greater than 0 in the relevant part load range.
    """
    if conversionFactor.shape[1] > 2:
        raise ValueError("The pandas dataframe has more than two columns.")

    xTest = np.array(conversionFactor.iloc[:, 0])
    yTest = np.array(conversionFactor.iloc[:, 1])

    if np.isnan(xTest).any() or np.isnan(yTest).any():
        raise ValueError(
            "At least one value in the raw conversion factor data is non-numeric."
        )

    if any(yTest_i <= 0 for yTest_i in yTest):
        raise ValueError(
            "The callable part load conversion factor is smaller or equal to 0 at least once within [0,1]."
        )


def checkAndSetDistances(distances, locationalEligibility, esM):
    """Check if the given values for the distances are valid (i.e. positive). If the distances parameter is None,
    the distances for the eligible connections are set to 1.
    """
    if distances is None:
        output(
            "The distances of a component are set to a normalized value of 1.",
            esM.verboseLogLevel,
            0,
        )
        distances = pd.Series(
            [1 for loc in locationalEligibility.index],
            index=locationalEligibility.index,
        )
    else:
        if not isinstance(distances, pd.Series):
            raise TypeError("Input data has to be a pandas DataFrame or Series")
        if (distances < 0).any():
            raise ValueError("Distance values smaller than 0 were detected.")
        distances = checkConnectionIndex(distances, locationalEligibility)
    return distances


def checkAndSetTransmissionLosses(losses, distances, locationalEligibility):
    """Check if the type of the losses are valid (i.e. a number, pandas DataFrame or a pandas Series),
    and if the given values for the losses of the transmission component are valid (i.e. between 0 and 1).
    """
    if not (
        isinstance(losses, int)
        or isinstance(losses, float)
        or isinstance(losses, pd.DataFrame)
        or isinstance(losses, pd.Series)
    ):
        raise TypeError(
            "The input data has to be a number, a pandas DataFrame or a pandas Series."
        )

    if isinstance(losses, int) or isinstance(losses, float):
        if losses < 0 or losses > 1:
            raise ValueError("Losses have to be values between 0 <= losses <= 1.")
        return pd.Series(
            [float(losses) for loc in locationalEligibility.index],
            index=locationalEligibility.index,
        )
    losses = checkConnectionIndex(losses, locationalEligibility)

    losses = losses.astype(float)
    if losses.isnull().any():
        raise ValueError("The losses parameter contains values which are not a number.")
    if (losses < 0).any() or (losses > 1).any():
        raise ValueError("Losses have to be values between 0 <= losses <= 1.")
    if (1 - losses * distances < 0).any():
        raise ValueError(
            "The losses per distance multiplied with the distances result in negative values."
        )

    return losses


def getCapitalChargeFactor(interestRate, economicLifetime, investmentPeriods):
    """Compute and return capital charge factor (inverse of annuity factor)."""
    CCF = {}
    for ip in investmentPeriods:
        CCF[ip] = 1 / interestRate - 1 / (
            pow(1 + interestRate, economicLifetime) * interestRate
        )
        CCF[ip] = CCF[ip].fillna(economicLifetime)
    return CCF


def castToSeries(data, esM):
    """MISSING."""
    if data is None:
        return None
    if isinstance(data, pd.Series):
        return data
    isPositiveNumber(data)
    return pd.Series(data, index=list(esM.locations))


def getQPbound(investmentPeriods, QPcostScale, capacityMax, capacityMin):
    """Compute and return lower and upper capacity bounds."""
    QPbound = {}
    for ip in investmentPeriods:
        index = QPcostScale[ip].index
        QPbound[ip] = pd.Series([np.inf] * len(index), index)
        if ip >= 0:  # QP only relevant for future years
            if capacityMin[ip] is not None and capacityMax[ip] is not None:
                minS = pd.Series(capacityMin[ip].isna(), index)
                maxS = pd.Series(capacityMax[ip].isna(), index)
                for x in index:
                    if not minS.loc[x] and not maxS.loc[x]:
                        QPbound[ip].loc[x] = capacityMax[ip].loc[x]
    return QPbound


def getQPcostDev(investmentPeriods, QPcostScale):
    """MISSING."""
    QPcostDev = {}
    for ip in investmentPeriods:
        QPcostDev[ip] = 1 - QPcostScale[ip]
    return QPcostDev


def checkLocationSpecficDesignInputParams(comp, esM):
    """MISSING."""
    if len(esM.locations) == 1:
        comp.locationalEligibility = castToSeries(comp.locationalEligibility, esM)
        comp.isBuiltFix = castToSeries(comp.isBuiltFix, esM)

    capacityMin = comp.processedCapacityMin
    capacityFix = comp.processedCapacityFix
    capacityMax = comp.processedCapacityMax
    QPcostScale = comp.processedQPcostScale
    locationalEligibility = comp.locationalEligibility
    isBuiltFix = comp.isBuiltFix
    hasCapacityVariable = comp.hasCapacityVariable
    hasIsBuiltBinaryVariable = comp.hasIsBuiltBinaryVariable
    sharedPotentialID = comp.sharedPotentialID
    hasCapacityVariable = comp.hasCapacityVariable

    def checkAndSet(data, comp, esM):
        if data is not None:
            if comp.dimension == Dimension.ONE:
                if not isinstance(data, pd.Series):
                    raise TypeError("Input data has to be a pandas Series")
                data = checkRegionalIndex(esM, data, comp.locationalEligibility)
            elif comp.dimension == Dimension.TWO:
                if not isinstance(data, pd.Series):
                    raise TypeError("Input data has to be a pandas DataFrame")
                data = checkConnectionIndex(data, comp.locationalEligibility)
            else:
                raise ValueError(
                    "The dimension parameter has to be either '1dim' or '2dim' "
                )
            return data
        return None

    locationalEligibility = checkAndSet(locationalEligibility, comp, esM)
    isBuiltFix = checkAndSet(isBuiltFix, comp, esM)

    for ip in comp.processedStockYears + esM.investmentPeriods:
        checkAndSet(QPcostScale[ip], comp, esM)

    if isBuiltFix is not None and not hasIsBuiltBinaryVariable:
        raise ValueError(
            "Fixed design decisions are given but hasIsBuiltBinaryVariable was set to False."
        )

    if sharedPotentialID is not None:
        isString(sharedPotentialID)

        if capacityMax is None:
            raise ValueError(
                "A capacityMax parameter is required if a sharedPotentialID is considered."
            )

    for ip in esM.investmentPeriods:
        capacityMin[ip] = checkAndSet(capacityMin[ip], comp, esM)
        capacityMax[ip] = checkAndSet(capacityMax[ip], comp, esM)
        capacityFix[ip] = checkAndSet(capacityFix[ip], comp, esM)

        if (
            capacityMin[ip] is not None
            or capacityMax[ip] is not None
            or capacityFix[ip] is not None
        ) and not hasCapacityVariable:
            raise ValueError(
                "Capacity bounds are given but hasCapacityVariable was set to False."
            )

        if locationalEligibility is not None:
            # Check if given capacities indicate the same eligibility
            if capacityFix[ip] is not None:
                data = capacityFix[ip].copy()
                if not set(data.index.values).issubset(
                    set(locationalEligibility.index.values)
                ):
                    raise ValueError(
                        "CapacityFix values are provided for non-eligible locations."
                    )
            # Check if given capacities indicate the same eligibility
            if capacityFix[ip] is not None:
                data = capacityFix[ip].copy()
                if not set(data.index.values).issubset(
                    set(locationalEligibility.index.values)
                ):
                    raise ValueError(
                        "CapacityFix values are provided for non-eligible locations."
                    )
            if capacityMax[ip] is not None:
                data = capacityMax[ip].copy()
                data[data > 0] = 1
                if (data != locationalEligibility).any():
                    raise ValueError(
                        "The locationalEligibility and capacityMax parameters indicate different eligibilities."
                    )
            if capacityMin[ip] is not None:
                data = capacityMin[ip].copy()
                data[data > 0] = 1
                if (data > locationalEligibility).any():
                    raise ValueError(
                        "The locationalEligibility and capacityMin parameters indicate different eligibilities."
                    )

        if isBuiltFix is not None:
            # Check if values are either one or zero
            if ((isBuiltFix != 0) & (isBuiltFix != 1)).any():
                raise ValueError("The isBuiltFix entries have to be either 0 or 1.")
            # Check if given capacities indicate the same design decisions
            if capacityFix[ip] is not None:
                data = capacityFix[ip].copy()
                data[data > 0] = 1
                if (data > isBuiltFix).any():
                    raise ValueError(
                        "The isBuiltFix and capacityFix parameters indicate different design decisions."
                    )
            if capacityMax[ip] is not None:
                data = capacityMax[ip].copy()
                data[data > 0] = 1
                if (data > isBuiltFix).any():
                    if esM.verboseLogLevel < 2:
                        warnings.warn(
                            "The isBuiltFix and capacityMax parameters indicate different design options."
                        )
            if capacityMin[ip] is not None:
                data = capacityMin[ip].copy()
                data[data > 0] = 1
                if (data > isBuiltFix).any():
                    raise ValueError(
                        "The isBuiltFix and capacityMin parameters indicate different design decisions."
                    )

        if capacityMax[ip] is None or capacityMin[ip] is None:
            if (QPcostScale[ip] > 0).any():
                raise ValueError(
                    "QPcostScale is given but lower or upper capacity bounds are not specified."
                )

    for ip in esM.investmentPeriods:
        if capacityFix[ip] is None or capacityMax[ip] is None:
            continue

        for loc in capacityFix[ip].index:
            fixedShareSum = capacityFix[ip].loc[loc] / capacityMax[ip].loc[loc]

            for otherCompName in esM.sharedPotentialDict.get(
                (sharedPotentialID, loc, ip), []
            ):
                if otherCompName == comp.name:
                    continue

                otherComp = esM.getComponent(otherCompName)
                otherCapacityFix = otherComp.processedCapacityFix[ip]
                otherCapacityMax = otherComp.processedCapacityMax[ip]

                if otherCapacityFix is None or otherCapacityMax is None:
                    continue

                if loc not in otherCapacityFix.index:
                    continue

                fixedShareSum += otherCapacityFix.loc[loc] / otherCapacityMax.loc[loc]

            if fixedShareSum > 1:
                raise ValueError(
                    "The sum of fixed capacities of components with "
                    f"sharedPotentialID '{sharedPotentialID}' exceeds the "
                    f"available shared potential in location '{loc}'."
                )

    if locationalEligibility is not None:
        # Check if values are either one or zero
        if ((locationalEligibility != 0) & (locationalEligibility != 1)).any():
            raise ValueError(
                "The locationalEligibility entries have to be either 0 or 1."
            )
        if isBuiltFix is not None:
            if (isBuiltFix != locationalEligibility).any():
                raise ValueError(
                    "The locationalEligibility and isBuiltFix parameters indicate different"
                    + "eligibilities."
                )
    for ip in esM.investmentPeriods:
        if capacityMax is None or capacityMin is None:
            if (QPcostScale[ip] > 0).any():
                raise ValueError(
                    "QPcostScale is given but lower or upper capacity bounds are not specified."
                )

    # check the costscale
    for ip in esM.investmentPeriods + comp.processedStockYears:
        comp.processedQPcostScale[ip] = castToSeries(comp.processedQPcostScale[ip], esM)
        if (QPcostScale[ip] < 0).any() or (QPcostScale[ip] > 1).any():
            raise ValueError('QPcostScale must ba a number between "0" and "1".')


def processBoundParams(esM, param):
    """MISSING."""
    processedParam = {}
    for ip in esM.investmentPeriods:
        if param is None:
            processedParam[ip] = None
        if isinstance(param, dict):
            if param[esM.investmentPeriodNames[ip]] is None:
                processedParam[ip] = None
            else:
                processedParam[ip] = castToSeries(
                    param[esM.investmentPeriodNames[ip]], esM
                )
        elif isinstance(param, pd.DataFrame) or isinstance(param, pd.Series):
            processedParam[ip] = castToSeries(param, esM)
        elif isinstance(param, int) or isinstance(param, float):
            processedParam[ip] = castToSeries(param, esM)
    return processedParam


def checkAndSetBounds(esM, name, paramName, MinVal, MaxVal, FixVal):
    """MISSING."""
    checkInvestmentPeriodParameters(name, MinVal, esM.investmentPeriodNames)
    checkInvestmentPeriodParameters(name, MaxVal, esM.investmentPeriodNames)
    checkInvestmentPeriodParameters(name, FixVal, esM.investmentPeriodNames)

    # set up parameter as dict with investment periods as keys and
    # dataframe with locations as values
    processedMinVal = processBoundParams(esM, MinVal)
    processedMaxVal = processBoundParams(esM, MaxVal)
    processedFixVal = processBoundParams(esM, FixVal)

    for ip in esM.investmentPeriods:
        if processedMinVal[ip] is not None and (processedMinVal[ip] < 0).any():
            raise ValueError(
                f"{paramName}Min values for {name} smaller than 0 were detected."
            )

        if processedFixVal[ip] is not None and (processedFixVal[ip] < 0).any():
            raise ValueError(
                f"{paramName}Fix values for {name} smaller than 0 were detected."
            )

        if processedMaxVal[ip] is not None and (processedMaxVal[ip] < 0).any():
            raise ValueError(
                f"{paramName}Max values for {name} smaller than 0 were detected."
            )

        if processedMinVal[ip] is not None and processedMaxVal[ip] is not None:
            # Test that capacityMin and capacityMax has the same index for comparing.
            # If capacityMin is missing for some locations, it´s set to 0.
            if set(processedMinVal[ip].index).issubset(processedMaxVal[ip].index):
                processedMinVal[ip] = (
                    processedMinVal[ip].reindex(processedMaxVal[ip].index).fillna(0)
                )
            if (processedMinVal[ip] > processedMaxVal[ip]).any():
                raise ValueError(
                    f"{paramName}Min values > {paramName}Max values detected."
                )

        if processedFixVal[ip] is not None and processedMaxVal[ip] is not None:
            if (processedFixVal[ip] > processedMaxVal[ip]).any():
                raise ValueError(
                    f"{paramName}Fix values > {paramName}Max values detected."
                )

        if processedFixVal[ip] is not None and processedMinVal[ip] is not None:
            if (processedFixVal[ip] < processedMinVal[ip]).any():
                raise ValueError(
                    f"{paramName}Fix values < {paramName}Min values detected."
                )

    return processedMinVal, processedMaxVal, processedFixVal


def checkInvestmentPeriodParameters(name, param, years):
    """MISSING."""
    if isinstance(param, dict):
        if len(param.keys()) != len(years):
            raise ValueError(
                f"A parameter for '{name}' is initialized as dict for the years {sorted(list(param.keys()))}, but the expected years are {sorted(years)}"
            )
        if sorted(param.keys()) != sorted(years):
            raise ValueError(
                f"'{name}' has different ip-names ('{param.keys()}')"
                + f" than the investment periods of the esM ('{years}')",
            )


def checkAndSetInvestmentPeriodParameters(name, param, esM):
    """MISSING."""
    checkInvestmentPeriodParameters(name, param, esM.investmentPeriodNames)
    processedParam = {}
    for ip in esM.investmentPeriods:
        if param is None:
            processedParam[ip] = None
        if isinstance(param, dict):
            processedParam[ip] = param[esM.investmentPeriodNames[ip]]
        else:
            processedParam[ip] = param
    return processedParam


def checkCapacityDevelopmentWithStock(
    investmentPeriods,
    capacityMax,
    capacityFix,
    stockCommissioning,
    technicalLifetime,
    floorTechnicalLifetime,
):
    """MISSING."""
    if stockCommissioning is None:
        pass
    else:
        # if there is a stock, consider it for the capacity development
        # create a dataframe with columns for location, index for the years and
        # stock capacities as values
        locations = stockCommissioning[-1].index
        years = [x for x in stockCommissioning.keys()] + investmentPeriods
        stockCapacity = pd.DataFrame(0.0, index=years, columns=locations).sort_index()
        for ip, stockCommis in stockCommissioning.items():
            for loc in stockCommis.index:
                if floorTechnicalLifetime:
                    _techLifetime = math.floor(technicalLifetime[loc])
                else:
                    _techLifetime = math.ceil(technicalLifetime[loc])
                yearRange = list(range(ip, ip + _techLifetime))
                yearRange = [x for x in yearRange if x <= max(investmentPeriods)]

                # for floating numbers a normal sum can lead to floating point issues,
                # e.g. 4.19+2.19=6.380000000000001
                # therefore a rounding is applied, as otherwise the following
                # errors can be wrongly raised
                if stockCommis[loc] - round(stockCommis[loc], 10) != 0:
                    warnings.warn(
                        f"A stock commissioning of {stockCommis[loc]} was "
                        + f"passed for location {loc} in year {ip}. "
                        + "It will be rounded to 10 digits to "
                        + "check if the installed stock capacity does "
                        + "not exceed capacityMax and capacityFix"
                    )
                stockCapacity.loc[yearRange, loc] += stockCommis[loc]
                stockCapacity.loc[yearRange, loc] = round(
                    stockCapacity.loc[yearRange, loc], 10
                )
        # check that the capacity max is not lower as the resulting
        # stock capacity
        for loc in stockCapacity.columns:
            for year in investmentPeriods:
                if capacityMax[year] is not None:
                    if stockCapacity.loc[year, loc] > capacityMax[year][loc]:
                        raise ValueError(
                            "Mismatch between stock capacity (by its "
                            + "commissioning and the technical lifetime) and "
                            + "capacityMax"
                        )
                if capacityFix[year] is not None:
                    if stockCapacity.loc[year, loc] > capacityFix[year][loc]:
                        raise ValueError(
                            "Mismatch between stock capacity (by its "
                            + "commissioning and the technical lifetime) and "
                            + "capacityFix"
                        )

    if capacityFix is not None:
        if all(x is None for x in capacityFix.values()):
            return
        # get future capacity by capacityFix
        futureCapacityDevelopment = pd.DataFrame(index=investmentPeriods)
        for ip in investmentPeriods:
            for loc in capacityFix[ip].index:
                futureCapacityDevelopment.loc[ip, loc] = capacityFix[ip][loc]

        # create the total capacity development, if stock with past years of stock
        if stockCommissioning is None:
            capacityDevelopment = futureCapacityDevelopment
        else:
            pastYears = [x for x in stockCapacity.index if x < 0]
            capacityDevelopment = pd.concat(
                [stockCapacity.loc[pastYears], futureCapacityDevelopment]
            )

        if floorTechnicalLifetime:
            maxTechnicalLifetime = math.floor(technicalLifetime.max())
        else:
            maxTechnicalLifetime = math.ceil(technicalLifetime.max())
        capacityDevelopment = capacityDevelopment.reindex(
            range(-maxTechnicalLifetime - 1, max(investmentPeriods) + 1)
        ).fillna(0)

        # check that decreasing capacity matches the commissioning
        issueLocations = []
        for loc in capacityDevelopment.columns:
            capacityDevelopmentDiff = capacityDevelopment[loc].diff().fillna(0)
            for ip in investmentPeriods:
                # get technical lifetime
                if floorTechnicalLifetime:
                    roundedTechnicalLifetime = math.floor(technicalLifetime[loc])
                else:
                    roundedTechnicalLifetime = math.ceil(technicalLifetime[loc])

                # technical lifetime smaller one lead to new commissioning in
                # each investment period, independent of previous investment
                # periods and therefore no contradicting between decreasing
                # capacityFix and commissioning
                if roundedTechnicalLifetime <= 1:
                    continue

                capacityFixDiffOfIp = capacityDevelopmentDiff[ip]
                capacityFixDiffOneTechnicalLifetimeAgo = capacityDevelopmentDiff[
                    ip - roundedTechnicalLifetime
                ]

                if (
                    capacityFixDiffOfIp < 0
                    and capacityFixDiffOneTechnicalLifetimeAgo >= 0
                ):
                    # capacity reduction cannot exceed commissioning one
                    # technical lifetime ago
                    # 1) filter for commissioning
                    if capacityDevelopmentDiff[ip - roundedTechnicalLifetime] >= 0:
                        # 2) check that capacity reduction is not higher than commissioning
                        if (-capacityDevelopmentDiff[ip]) > capacityDevelopmentDiff[
                            ip - maxTechnicalLifetime
                        ]:
                            issueLocations.append(loc)
        issueLocations = list(set(issueLocations))

        if len(issueLocations) > 0:
            raise ValueError(
                f"Decreasing capacity fix set for regions {issueLocations} do"
                + " not match with the decommissioning with its "
                + "technical lifetime."
            )


def checkAndSetAnnuityPerpetuity(annuityPerpetuity, numberOfInvestmentPeriods):
    """MISSING."""
    if not isinstance(annuityPerpetuity, bool):
        raise ValueError("annuityPerpetuity must be a bool.")
    if annuityPerpetuity and numberOfInvestmentPeriods == 1:
        raise ValueError(
            "Annuity Perpetuity can only be set for a transformation "
            + "pathway more than one investment period."
        )
    return annuityPerpetuity


def checkAndSetInterestRate(esM, name, interestRate, dimension, elig):
    """Set up interest rate per investment period."""
    # set up interest rate per investment period
    processedInterestRate = checkAndSetCostParameter(
        esM, name, interestRate, dimension, elig
    )
    # if annuity perpetuity is used, the interest rate cannot be 0
    if esM.annuityPerpetuity:
        for ip in esM.investmentPeriods:
            if (processedInterestRate[ip] == 0).any():
                raise ValueError(
                    "An interest rate of 0 cannot be set if also using annuityPerpetuity"
                )
    return processedInterestRate


def checkRampRates(
    esM,
    name,
    rampUpMax,
    rampDownMax,
):
    """Check if ramprates are positive floats or ints."""
    if rampUpMax is not None:
        # Check if values are postive floats or ints
        if not isinstance(rampUpMax, (float, int)) and rampUpMax <= 0:
            raise TypeError(
                "rampUpMax for " + name + " needs to be a positive float or int."
            )

    if rampDownMax is not None:
        # Check if values are postive floats or ints
        if not isinstance(rampDownMax, (float, int)) and rampDownMax <= 0:
            raise TypeError(
                "rampUpMax for " + name + " needs to be a positive float or int."
            )


def checkConversionDynamicSpecficDesignInputParams(compFancy, esM):
    """MISSING."""
    downTimeMin = compFancy.downTimeMin
    upTimeMin = compFancy.upTimeMin
    numberOfTimeSteps = esM.numberOfTimeSteps
    name = compFancy.name
    bigM = compFancy.bigM
    useTemporalCyclicConstraints = compFancy.useTemporalCyclicConstraints

    if downTimeMin is not None:
        # Check if values are integers and in the intervall ]0,numberOfTimeSteps].
        if not isinstance(downTimeMin, int):
            raise TypeError(
                "downTimeMin for "
                + name
                + " needs to be an integer in the intervall ]0,numberOfTimeSteps]."
            )
        if downTimeMin <= 0:
            raise ValueError(
                "downTimeMin for "
                + name
                + " needs to be an integer in the intervall ]0,numberOfTimeSteps]."
            )
        if downTimeMin > numberOfTimeSteps:
            raise ValueError(
                "downTimeMin for "
                + name
                + " needs to be an integer in the intervall ]0,numberOfTimeSteps]."
            )

    if upTimeMin is not None:
        # Check if values are integers and in the intervall ]0,numberOfTimeSteps].
        if not isinstance(upTimeMin, int):
            raise TypeError(
                "upTimeMin for "
                + name
                + " needs to be an integer in the intervall ]0,numberOfTimeSteps]."
            )
        if upTimeMin <= 0:
            raise ValueError(
                "upTimeMin for "
                + name
                + " needs to be an integer in the intervall ]0,numberOfTimeSteps]."
            )
        if upTimeMin > numberOfTimeSteps:
            raise ValueError(
                "upTimeMin for "
                + name
                + " needs to be an integer in the intervall ]0,numberOfTimeSteps]."
            )

    if any(x is not None for x in [downTimeMin, upTimeMin]):
        if bigM is None:
            raise ValueError(
                "bigM for "
                + name
                + " needs to be specified when considering dynamic constraints."
            )

    # check cyclic constraints
    if not isinstance(useTemporalCyclicConstraints, bool):
        raise ValueError("useTemporalCyclicConstraints must be a boolean.")


def setLocationalEligibility(
    esM,
    locationalEligibility,
    capacityMax,
    capacityFix,
    isBuiltFix,
    hasCapacityVariable,
    operationTimeSeries,
    dimension=Dimension.ONE,
):
    """MISSING."""
    # ruff: noqa: PLR0911 # needed to avoid ruff saying "too many return statements"
    if locationalEligibility is not None:
        if isinstance(locationalEligibility, pd.Series):
            esm_locations = set(esM.locations)
            le_index = set(locationalEligibility.index)
            if dimension == Dimension.ONE:
                if esm_locations != le_index:
                    raise ValueError(
                        "if locationalEligibility (1dim) is specified, it needs to match the esM locations"
                    )
            elif dimension == Dimension.TWO:
                le_index_2dim = set(
                    f"{a}_{b}"
                    for a in sorted(esm_locations)
                    for b in sorted(esm_locations)
                    if a != b
                )
                if (
                    le_index > le_index_2dim
                ):  # location eligibility can only be defined between existing esm locations
                    raise ValueError(
                        "if locationalEligibility is specified, it can only be defined between existing esm locations"
                    )
            else:
                raise ValueError("dimensions needs to be either '1dim' or '2dim'")
            if not locationalEligibility.isin([0, 1, True, False]).all():
                raise ValueError(
                    "all values in locationalEligibility must be either True or False or 1 or 0"
                )
        else:
            raise ValueError(
                "locationalEligibility needs to be a Series after it has been preprocessed"
            )
        return locationalEligibility

    # If the location eligibility is None set it based on other information available
    def defineLocDependencyCapacityBounds(name, capacityBound):
        if capacityBound is None:
            return False
        anyLocDependent = any(
            x is not None and not isinstance(x, (int, float))
            for x in capacityBound.values()
        )
        if anyLocDependent:
            return True
        return False

    isCapacityMaxLocDepending = defineLocDependencyCapacityBounds(
        "capacityMax", capacityMax
    )
    isCapacityFixLocDepending = defineLocDependencyCapacityBounds(
        "capacityFix", capacityFix
    )

    if isinstance(operationTimeSeries, dict) and len(operationTimeSeries) == 0:
        operationTimeSeries = None

    if (
        not hasCapacityVariable
        and operationTimeSeries is not None
        and any(ots is not None for ots in operationTimeSeries.values())
    ):
        if dimension == Dimension.ONE:
            data = 0
            # sum values over ips
            for ip in esM.investmentPeriods:
                if operationTimeSeries[ip] is not None:
                    data += operationTimeSeries[ip].copy().sum()
            data[data > 0] = 1
            return data
        # Problems here ? Adapt this?
        if dimension == Dimension.TWO:
            # New for perfect foresight
            data = 0
            # sum values over ips
            for ip in esM.investmentPeriods:
                data += operationTimeSeries[ip].copy().sum()

            data.loc[:] = 1
            return data
        raise ValueError("The dimension parameter has to be either '1dim' or '2dim' ")
    if (
        (not isCapacityMaxLocDepending)
        and (not isCapacityFixLocDepending)
        and (isBuiltFix is None or isinstance(isBuiltFix, int))
    ):
        # If no information is given, or all information is given as float or integer, all values are set to 1
        if dimension == Dimension.ONE:
            return pd.Series([1 for loc in esM.locations], index=esM.locations)
        keys = {
            loc1 + "_" + loc2
            for loc1 in esM.locations
            for loc2 in esM.locations
            if loc1 != loc2
        }
        data = pd.Series([1 for key in keys], index=keys)
        data.sort_index(inplace=True)
        return data
    if isBuiltFix is not None and isinstance(isBuiltFix, pd.Series):
        # If the isBuiltFix is not empty, the eligibility is set based on the fixed capacity
        data = isBuiltFix.copy()
        data[data > 0] = 1
        data.sort_index(inplace=True)
        return data
    # If the fixCapacity is not empty, the eligibility is set based on the fixed capacity
    # either use capacityFix or capacityMax
    if isinstance(capacityFix, dict):
        if all(x is None for x in capacityFix.values()):
            data = capacityMax
        else:
            data = capacityFix
    elif capacityFix is None:
        data = capacityMax
    else:
        raise NotImplementedError()

    # First setup series with only 0
    if dimension == Dimension.ONE:
        regions = esM.locations
    else:
        firstYear = sorted(data.keys())[0]
        regions = data[firstYear].index
    _data = pd.Series(index=sorted(regions), data=0)

    # set location eligibility to 1 if capacity bound exists
    for ip in esM.investmentPeriods:
        if data[ip] is not None:
            loc_idx = data[ip][data[ip] > 0].index
            _data[loc_idx] = 1

    return _data


def checkAndSetInvestmentPeriodTimeSeries(
    esM, name, data, locationalEligibility, dimension=Dimension.ONE
):
    """MISSING."""
    checkInvestmentPeriodParameters(name, data, esM.investmentPeriodNames)
    parameter = {}
    for _ip in esM.investmentPeriodNames:
        # map name of investment period (e.g. 2020) to index (e.g. 0)
        ip = esM.investmentPeriodNames.index(_ip)
        if (
            isinstance(data, pd.DataFrame)
            or data is None
            or isinstance(data, pd.Series)
        ):
            parameter[ip] = checkAndSetTimeSeries(
                esM, name, data, locationalEligibility, dimension
            )
        elif isinstance(data, dict):
            parameter[ip] = checkAndSetTimeSeries(
                esM, name, data[_ip], locationalEligibility, dimension
            )
        elif isinstance(data, int) or isinstance(data, float):
            _data = pd.DataFrame(
                {loc: [data] * esM.numberOfTimeSteps for loc in esM.locations}
            )
            parameter[ip] = checkAndSetTimeSeries(esM, name, _data, None, dimension)
        else:
            raise TypeError(f"Parameter of {name} does not match required type.")
    return parameter


def checkAndSetInvestmentPeriodCostTimeSeries(
    esM, name, data, locationalEligibility, dimension=Dimension.ONE
):
    """MISSING."""
    if (
        isinstance(data, dict)
        and any(x is None for x in data.values())
        and not all(x is None for x in data.values())
    ):
        raise TypeError(
            f"Parameter of {name} can not be None for individual investment periods if specified for as dict."
        )
    return checkAndSetInvestmentPeriodTimeSeries(
        esM, name, data, locationalEligibility, dimension
    )


def checkAndSetTimeSeries(
    esM, name, operationTimeSeries, locationalEligibility, dimension=Dimension.ONE
):
    """MISSING."""
    if operationTimeSeries is not None:
        if not isinstance(operationTimeSeries, pd.DataFrame):
            if len(esM.locations) == 1:
                if isinstance(operationTimeSeries, pd.Series):
                    operationTimeSeries = pd.DataFrame(
                        operationTimeSeries.values,
                        index=operationTimeSeries.index,
                        columns=list(esM.locations),
                    )
                else:
                    raise TypeError(
                        "Type error in "
                        + name
                        + " detected.\n"
                        + "operationTimeSeries parameters have to be a pandas DataFrame."
                    )
            else:
                raise TypeError(
                    "Type error in "
                    + name
                    + " detected.\n"
                    + "operationTimeSeries parameters have to be a pandas DataFrame."
                )
        checkTimeSeriesIndex(esM, operationTimeSeries)

        if dimension == Dimension.ONE:
            operationTimeSeries = checkRegionalColumnTitles(
                esM, operationTimeSeries, locationalEligibility
            )

            if locationalEligibility is not None:
                # Check if given capacities indicate the same eligibility
                data = operationTimeSeries.copy().sum()
                data[data > 0] = 1

                if (data > locationalEligibility).any().any():
                    raise ValueError(
                        "The locationalEligibility and "
                        + name
                        + " parameters indicate different"
                        + " eligibilities."
                    )

        elif dimension == Dimension.TWO:
            keys = {
                loc1 + "_" + loc2 for loc1 in esM.locations for loc2 in esM.locations
            }
            columns = set(operationTimeSeries.columns)
            if not columns <= keys:
                raise ValueError(
                    "False column index detected in"
                    + name
                    + " time series. "
                    + "The indicies have to be in the format 'loc1_loc2' "
                    + "with loc1 and loc2 being locations in the energy system model."
                )

            for loc1 in esM.locations:
                for loc2 in esM.locations:
                    if (
                        loc1 + "_" + loc2 in columns
                        and loc2 + "_" + loc1 not in columns
                    ):
                        raise ValueError(
                            "Missing data in "
                            + name
                            + " time series DataFrame of a location connecting \n"
                            + "component. If the flow is specified from loc1 to loc2, \n"
                            + "then it must also be specified from loc2 to loc1.\n"
                        )

            if locationalEligibility is not None:
                # Check if given capacities indicate the same eligibility
                keys = set(locationalEligibility.index)
                if not columns == keys:
                    raise ValueError(
                        "The locationalEligibility and "
                        + name
                        + " parameters indicate different"
                        + " eligibilities."
                    )

        _operationTimeSeries = operationTimeSeries.astype(float)
        if _operationTimeSeries.isnull().any().any():
            raise ValueError(
                "Value error in "
                + name
                + " detected.\n"
                + "An operationTimeSeries parameter contains values which are not numbers."
            )
        if (_operationTimeSeries < 0).any().any():
            raise ValueError(
                "Value error in "
                + name
                + " detected.\n"
                + "All entries in operationTimeSeries parameter series have to be positive."
            )

        _operationTimeSeries = _operationTimeSeries.copy()
        _operationTimeSeries["Period"], _operationTimeSeries["TimeStep"] = (
            0,
            _operationTimeSeries.index,
        )
        return _operationTimeSeries.set_index(["Period", "TimeStep"])
    return None


def checkOperationRateForCapacityVariable(
    name, hasCapacityVariable, *operationRateDicts
):
    """Warn when hasCapacityVariable=True but operationRate values exceed 1.0."""
    if not hasCapacityVariable:
        return
    for opDict in operationRateDicts:
        if opDict is None:
            continue
        for ts in opDict.values():
            if ts is not None and (ts > 1.0).any().any():
                warnings.warn(
                    f"'{name}': hasCapacityVariable is True, so operationRate values are"
                    " expected to be relative capacity factors in [0, 1]. Values > 1.0 were"
                    " detected. If this is unintentional, check that absolute values are not"
                    " being passed as capacity factors."
                )
                return


def checkDesignVariableModelingParameters(
    esM,
    capacityVariableDomain,
    hasCapacityVariable,
    capacityPerPlantUnit,
    hasIsBuiltBinaryVariable,
    bigM,
):
    """MISSING."""
    if capacityVariableDomain not in ("continuous", "discrete"):
        raise ValueError(
            "The capacity variable domain has to be either 'continuous' or 'discrete'."
        )

    if not isinstance(hasIsBuiltBinaryVariable, bool):
        raise TypeError("The hasCapacityVariable variable domain has to be a boolean.")

    for ip in esM.investmentPeriods:
        isStrictlyPositiveNumber(capacityPerPlantUnit[ip])

    if not hasCapacityVariable and hasIsBuiltBinaryVariable:
        raise ValueError(
            "To consider additional fixed cost contributions when installing\n"
            + "capacities, capacity variables are required."
        )

    if bigM is None and hasIsBuiltBinaryVariable:
        raise ValueError(
            "A bigM value needs to be specified when considering fixed cost contributions."
        )

    if bigM is not None and hasIsBuiltBinaryVariable:
        isPositiveNumber(bigM)
    elif bigM is not None and not hasIsBuiltBinaryVariable:
        if esM.verboseLogLevel < 2:
            warnings.warn(
                "The declared bigM variable is not used in the problem formulation for hasIsBuiltBinaryVariable, since hasIsBuiltBinaryVariable is set to false. \n"
                "Check if bigM is needed for other binary variables (like partLoadMin). Else it is ignored."
            )


def checkTechnicalLifetime(esM, technicalLifetime, economicLifetime):
    """Set technical lifetime to economical lifetime if not explicitly given."""
    if technicalLifetime is None:
        technicalLifetime = economicLifetime
    return technicalLifetime


def checkEconomicAndTechnicalLifetime(economicLifetime, technicalLifetime):
    """Ensure that economic lifetime is smaller than technical lifetime."""
    if (economicLifetime.sort_index() > technicalLifetime.sort_index()).any():
        raise ValueError("Economic Lifetime must be smaller than technical Lifetime.")


def checkFlooringParameter(floorTechnicalLifetime, technicalLifetime, interval):
    """MISSING."""
    if not isinstance(floorTechnicalLifetime, bool):
        raise ValueError("floorTechnicalLifetime must be a bool")
    if floorTechnicalLifetime and any(
        (technicalLifetime.loc[technicalLifetime != 0] / interval) < 1
    ):
        raise ValueError(
            "Flooring of a lifetime smaller than the interval not possible"
        )
    return floorTechnicalLifetime


def checkAndSetCostParameter(esM, name, data, dimension, locationalEligibility):
    """MISSING."""
    if isinstance(data, pd.Series) and data.isnull().any():
        raise ValueError(
            f"Initialization error in {name} detected.\n"
            "Economic parameters contain NaN values which are not allowed."
        )
    if isinstance(data, (int, float)) and pd.isnull(data):
        raise ValueError(
            f"Initialization error in {name} detected.\n"
            "Economic parameters contain NaN values which are not allowed."
        )
    if dimension == Dimension.ONE:
        if not (
            isinstance(data, int)
            or isinstance(data, float)
            or isinstance(data, pd.Series)
        ):
            raise TypeError(
                "Type error in "
                + name
                + " detected.\n"
                + "Economic parameters have to be a number or a pandas Series."
            )
    elif dimension == Dimension.TWO:
        if not (
            isinstance(data, int)
            or isinstance(data, float)
            or isinstance(data, pd.Series)
        ):
            raise TypeError(
                "Type error in "
                + name
                + " detected.\n"
                + "Economic parameters have to be a number or a pandas Series."
            )
    else:
        raise ValueError("The dimension parameter has to be either '1dim' or '2dim' ")

    if dimension == Dimension.ONE:
        if isinstance(data, int) or isinstance(data, float):
            if data < 0:
                raise ValueError(
                    "Value error in "
                    + name
                    + " detected.\n Economic parameters have to be positive."
                )
            return pd.Series(
                [float(data) for loc in esM.locations], index=esM.locations
            )
        data = checkRegionalIndex(esM, data, locationalEligibility)
    else:
        if isinstance(data, int) or isinstance(data, float):
            if data < 0:
                raise ValueError(
                    "Value error in "
                    + name
                    + " detected.\n Economic parameters have to be positive."
                )
            return pd.Series(
                [float(data) for loc in locationalEligibility.index],
                index=locationalEligibility.index,
            )
        data = checkConnectionIndex(data, locationalEligibility)

    _data = data.astype(float)
    if _data.isnull().any():
        raise ValueError(
            "Value error in "
            + name
            + " detected.\n"
            + "An economic parameter contains values which are not numbers."
        )
    if (_data < 0).any():
        raise ValueError(
            "Value error in "
            + name
            + " detected.\n"
            + "All entries in economic parameter series have to be positive."
        )
    return _data


def setPartLoadMin(esM, partLoadMin):
    """Set minimum part load."""
    partLoadMin_ip = {}
    for _ip in esM.investmentPeriodNames:
        # map name of investment period (e.g. 2020) to index (e.g. 0)
        ip = esM.investmentPeriodNames.index(_ip)
        if isinstance(partLoadMin, float) or partLoadMin is None:
            partLoadMin_ip[ip] = partLoadMin
        elif isinstance(partLoadMin, dict):
            partLoadMin_ip[ip] = partLoadMin[_ip]
    return partLoadMin_ip


def checkAndSetPartLoadMin(
    esM,
    name,
    partLoadMin,
    fullOperationMax,
    fullOperationFix,
    bigM,
    hasCapacityVariable,
    fullOperationMin=None,
):
    """MISSING."""

    # checking function
    def checkPartLoadMin(partLoadMin, bigM, hasCapacityVariable):
        # Check if values are floats and the intervall ]0,1].
        if not isinstance(partLoadMin, float):
            raise TypeError(
                "partLoadMin for "
                + name
                + " needs to be a float in the intervall ]0,1]."
            )
        if partLoadMin <= 0:
            raise ValueError(
                "partLoadMin for "
                + name
                + " needs to be a float in the intervall ]0,1]."
            )
        if partLoadMin > 1:
            raise ValueError(
                "partLoadMin for "
                + name
                + " needs to be a float in the intervall ]0,1]."
            )
        if bigM is None:
            raise ValueError(
                "bigM needs to be defined for component "
                + name
                + " if partLoadMin is not None."
            )
        if not hasCapacityVariable:
            raise ValueError(
                "hasCapacityVariable needs to be True for component "
                + name
                + " if partLoadMin is not None."
            )

    # check the raw partloadmin
    if partLoadMin is not None:
        checkInvestmentPeriodParameters(name, partLoadMin, esM.investmentPeriodNames)
        if isinstance(partLoadMin, dict):
            for ip in esM.investmentPeriodNames:
                if partLoadMin[ip] is not None:
                    checkPartLoadMin(partLoadMin[ip], bigM, hasCapacityVariable)
        elif isinstance(partLoadMin, int) or isinstance(partLoadMin, float):
            checkPartLoadMin(partLoadMin, bigM, hasCapacityVariable)

        else:
            raise TypeError(
                "Wrong datatype for partLoadMin. "
                + "Either a dict, int or float is accepted."
            )

    # set part load min per investment period
    partLoadMin_ip = setPartLoadMin(esM, partLoadMin)

    if not any(value for value in partLoadMin_ip.values()):
        partLoadMin_ip = None
    if partLoadMin_ip is not None:
        for _ip in esM.investmentPeriodNames:
            # map name of investment period (e.g. 2020) to index (e.g. 0)
            ip = esM.investmentPeriodNames.index(_ip)
            if fullOperationMax[ip] is not None:
                if (
                    (
                        (fullOperationMax[ip] > 0)
                        & (fullOperationMax[ip] < partLoadMin_ip[ip])
                    )
                    .any()
                    .any()
                ):
                    raise ValueError(
                        '"operationRateMax" needs to be higher than "partLoadMin" or 0 for component '
                        + name
                    )
            if fullOperationFix[ip] is not None:
                if (
                    (
                        (fullOperationFix[ip] > 0)
                        & (fullOperationFix[ip] < partLoadMin_ip[ip])
                    )
                    .any()
                    .any()
                ):
                    raise ValueError(
                        '"fullOperationRateFix" needs to be higher than "partLoadMin" or 0 for component '
                        + name
                    )
            if fullOperationMin[ip] is not None:
                raise ValueError(
                    '"operationRateMin" must not be set if "partLoadMin" is set for component '
                    + name
                )
    return partLoadMin_ip


def checkAndSetInvestmentPeriodCostParameter(
    esM, name, data, dimension, locationalEligibility, years
):
    """MISSING."""
    # stock years are only considered for parameter for which the
    # years contain investment periods and stock years
    _years = [int(esM.startYear + ip * esM.investmentPeriodInterval) for ip in years]
    checkInvestmentPeriodParameters(name, data, _years)

    # set the costs
    parameter = {}
    for ip in years:
        # map of year name (e.g. 2020) to intenral name (e.g. 0)
        # ip=int((_ip-esM.startYear)/esM.investmentPeriodInterval)
        _ip = int(esM.startYear + ip * esM.investmentPeriodInterval)
        if (
            isinstance(data, int)
            or isinstance(data, float)
            or isinstance(data, pd.Series)
        ):
            parameter[ip] = checkAndSetCostParameter(
                esM, name, data, dimension, locationalEligibility
            )
        elif isinstance(data, dict):
            parameter[ip] = checkAndSetCostParameter(
                esM, name, data[_ip], dimension, locationalEligibility
            )
        else:
            raise TypeError(
                f"Parameter of {name} should be a pandas series or a dictionary."
            )
    return parameter


def checkAndSetLifetimeInvestmentPeriod(esM, name, lifetime):
    """Calculate lifetime in investement periods."""
    return lifetime / esM.investmentPeriodInterval


def checkAndSetTimeSeriesConversionFactors(
    esM, commodityConversionFactorsTimeSeries, locationalEligibility
):
    """MISSING."""
    if commodityConversionFactorsTimeSeries is not None:
        if not isinstance(commodityConversionFactorsTimeSeries, pd.DataFrame):
            if len(esM.locations) == 1:
                if isinstance(commodityConversionFactorsTimeSeries, pd.Series):
                    fullCommodityConversionFactorsTimeSeries = pd.DataFrame(
                        commodityConversionFactorsTimeSeries.values,
                        index=commodityConversionFactorsTimeSeries.index,
                        columns=list(esM.locations),
                    )
                else:
                    raise TypeError(
                        "The commodityConversionFactorsTimeSeries data type has to be a pandas DataFrame or Series"
                    )
            else:
                raise TypeError(
                    "The commodityConversionFactorsTimeSeries data type has to be a pandas DataFrame"
                )
        elif isinstance(commodityConversionFactorsTimeSeries, pd.DataFrame):
            fullCommodityConversionFactorsTimeSeries = (
                commodityConversionFactorsTimeSeries
            )
        else:
            raise TypeError(
                "The commodityConversionFactorsTimeSeries data type has to be a pandas DataFrame or Series"
            )

        checkTimeSeriesIndex(esM, fullCommodityConversionFactorsTimeSeries)

        checkRegionalColumnTitles(
            esM, fullCommodityConversionFactorsTimeSeries, locationalEligibility
        )

        if (
            locationalEligibility is not None
            and fullCommodityConversionFactorsTimeSeries is not None
        ):
            # Check if given conversion factors indicate the same eligibility
            data = fullCommodityConversionFactorsTimeSeries.copy().sum().abs()
            data[data > 0] = 1
            if (data.sort_index() > locationalEligibility.sort_index()).any().any():
                warnings.warn(
                    "The locationalEligibility and commodityConversionFactorsTimeSeries parameters "
                    "indicate different eligibilities."
                )

        fullCommodityConversionFactorsTimeSeries = (
            fullCommodityConversionFactorsTimeSeries.copy()
        )
        (
            fullCommodityConversionFactorsTimeSeries["Period"],
            fullCommodityConversionFactorsTimeSeries["TimeStep"],
        ) = (0, fullCommodityConversionFactorsTimeSeries.index)

        return fullCommodityConversionFactorsTimeSeries.set_index(
            ["Period", "TimeStep"]
        )
    return None


def _addColumnsBalanceLimit(balanceLimit, locations):
    # check and set lower bounds
    if "lowerBound" not in balanceLimit.columns:
        # default as in docs: lowerBound is set to False
        balanceLimit["lowerBound"] = 0
    elif any(x for x in balanceLimit["lowerBound"] if x not in [0, 1]):
        raise ValueError(
            "lowerBound in balanceLimit must be set to either True, False, 0 or 1"
        )
    # check and set locations:
    for loc in list(locations) + ["Total"]:
        if loc not in balanceLimit.columns:
            balanceLimit[loc] = None
    return balanceLimit


def checkAndSetPathwayBalanceLimit(esM, pathwayBalanceLimit, locations):
    """MISSING."""
    # pathwayBalanceLimit has to be DataFrame with locations as columns,
    # if valid for whole model
    if pathwayBalanceLimit is None:
        processedPathwayBalanceLimit = None
    else:
        if not isinstance(pathwayBalanceLimit, pd.DataFrame):
            raise ValueError("Wrong datatype for pathwayBalanceLimit")
        processedPathwayBalanceLimit = _addColumnsBalanceLimit(
            pathwayBalanceLimit, locations
        )
    return processedPathwayBalanceLimit


def checkAndSetBalanceLimit(esM, balanceLimit, locations):
    """MISSING."""
    # balanceLimit has to be DataFrame with locations as columns or Dict per
    # investment periods as keys and described dataframe as values,
    # if valid for whole model

    if balanceLimit is None:
        return None

    checkInvestmentPeriodParameters(
        "balanceLimit", balanceLimit, esM.investmentPeriodNames
    )
    processedBalanceLimit = {}

    for ip in esM.investmentPeriods:
        _ip = esM.investmentPeriodNames[ip]

        if isinstance(balanceLimit, dict):
            if balanceLimit[_ip] is None:
                _balanceLimit = None
            else:
                _balanceLimit = balanceLimit[_ip].copy()
        else:
            _balanceLimit = balanceLimit.copy()

        if _balanceLimit is not None:
            if not isinstance(_balanceLimit, pd.DataFrame):
                raise TypeError(
                    "The balanceLimit input argument has to be a pandas.DataFrame."
                )
            if not all(
                [
                    col in list(locations) + ["Total", "lowerBound"]
                    for col in _balanceLimit.columns
                ]
            ):
                raise ValueError(
                    "Location indices in the balanceLimit do not match the input locations.\n"
                    + "balanceLimit columns: "
                    + str(set(_balanceLimit.columns))
                    + "\n"
                    + "Input regions: "
                    + str(locations)
                )
            processedBalanceLimit[ip] = _balanceLimit
        else:
            processedBalanceLimit[ip] = None

        if processedBalanceLimit[ip] is not None:
            processedBalanceLimit[ip] = _addColumnsBalanceLimit(
                processedBalanceLimit[ip], locations
            )
    return processedBalanceLimit


def checkAndSetFullLoadHoursParameter(
    esM, name, data, dimension, locationalEligibility
):
    """MISSING."""
    checkInvestmentPeriodParameters(name, data, esM.investmentPeriodNames)
    parameter = {}
    for ip in esM.investmentPeriods:
        _ip = esM.investmentPeriodNames[ip]
        if data is None:
            parameter[ip] = None
        else:
            if isinstance(data, dict):
                _data = data[_ip]
            else:
                _data = data

            if isinstance(_data, int) or isinstance(_data, float):
                if _data < 0:
                    raise ValueError(
                        "Value error in "
                        + name
                        + " detected.\n Full load hours limitations have to be positive."
                    )
                if dimension == Dimension.ONE:
                    parameter[ip] = pd.Series(
                        [float(_data) for loc in esM.locations], index=esM.locations
                    )
                elif dimension == Dimension.TWO:
                    parameter[ip] = pd.Series(
                        [float(_data) for loc in locationalEligibility.index],
                        index=locationalEligibility.index,
                    )
            elif isinstance(_data, pd.Series):
                _data = checkConnectionIndex(_data, locationalEligibility)
                _data = _data.astype(float)
                if _data.isnull().any():
                    raise ValueError(
                        "Value error in "
                        + name
                        + " detected.\n"
                        + "An economic parameter contains values which are not numbers."
                    )
                if (_data < 0).any():
                    raise ValueError(
                        "Value error in "
                        + name
                        + " detected.\n"
                        + "All entries in economic parameter series have to be positive."
                    )
                parameter[ip] = _data
            elif _data is None:
                parameter[ip] = None
    return parameter


def checkClusteringInput(
    numberOfTypicalPeriods, numberOfTimeStepsPerPeriod, totalNumberOfTimeSteps
):
    """MISSING."""
    (
        isStrictlyPositiveInt(numberOfTypicalPeriods),
        isStrictlyPositiveInt(numberOfTimeStepsPerPeriod),
    )
    if not totalNumberOfTimeSteps % numberOfTimeStepsPerPeriod == 0:
        raise ValueError(
            f"The numberOfTimeStepsPerPeriod ({numberOfTimeStepsPerPeriod}) has to be an integer divisor of the total number of time"
            + f" steps considered in the energy system model ({totalNumberOfTimeSteps})."
        )
    if totalNumberOfTimeSteps < numberOfTypicalPeriods * numberOfTimeStepsPerPeriod:
        raise ValueError(
            "The product of the numberOfTypicalPeriods and the numberOfTimeStepsPerPeriod has to be \n"
            + "smaller than the total number of time steps considered in the energy system model."
        )


def checkDeclareOptimizationProblemInput(
    timeSeriesAggregation, isTimeSeriesDataClustered
):
    """MISSING."""
    if not isinstance(timeSeriesAggregation, bool):
        raise TypeError("The timeSeriesAggregation parameter has to be a boolean.")

    if timeSeriesAggregation and not isTimeSeriesDataClustered:
        raise ValueError(
            "The time series flag indicates possible inconsistencies in the aggregated time series "
            " data.\n--> Call the cluster function first, then the optimize function."
        )


def checkOptimizeInput(
    timeSeriesAggregation,
    isTimeSeriesDataClustered,
    logFileName,
    threads,
    solver,
    timeLimit,
    optimizationSpecs,
    warmstart,
):
    """Ensure validity of input parameters for the optimization."""
    checkDeclareOptimizationProblemInput(
        timeSeriesAggregation, isTimeSeriesDataClustered
    )

    if not isinstance(logFileName, str):
        raise TypeError("The logFileName parameter has to be a string.")

    if not isinstance(threads, int) or threads < 0:
        raise TypeError("The threads parameter has to be a nonnegative integer.")

    if not isinstance(solver, str):
        raise TypeError("The solver parameter has to be a string.")

    if timeLimit is not None:
        isStrictlyPositiveNumber(timeLimit)

    if not isinstance(optimizationSpecs, str):
        raise TypeError("The optimizationSpecs parameter has to be a string.")

    if not isinstance(warmstart, bool):
        raise ValueError("The warmstart parameter has to be a boolean.")


def buildFullTimeSeries(df, periodsOrder, ip, axis=1, esM=None, divide=True):
    """MISSING."""
    # If segmentation is chosen, the segments of each period need to be unravelled to the original number of
    # time steps first
    if esM is not None and esM.segmentation:
        dataAllPeriods = []
        for p in esM.typicalPeriods:
            # Repeat each segment in each period as often as time steps are represented by the corresponding
            # segment
            repList = (
                esM.timeStepsPerSegment[ip].loc[p, :].tolist()
            )  # timeStepsPerSegment now ip-dependent
            # if divide is set to True, the values are divided when being unravelled, e.g. in order to fit provided
            # energy per segment provided energy per time step
            if divide:
                dataPeriod = pd.DataFrame(
                    np.repeat(np.divide(df.loc[p].values, repList), repList, axis=1),
                    index=df.xs(p, level=0, drop_level=False).index,
                )

            # if divide is set to Frue, the values are not divided when being unravelled e.g. in case of time-
            # independent costs
            else:
                dataPeriod = pd.DataFrame(
                    np.repeat(df.loc[p].values, repList, axis=1),
                    index=df.xs(p, level=0, drop_level=False).index,
                )
            dataAllPeriods.append(dataPeriod)
        # Concat data to multiindex dataframe with periods, components and locations as indices and inner-
        # period time steps as columns
        df = pd.concat(dataAllPeriods, axis=0)
    # Concat data according to periods order to cover the full time horizon
    data = []
    for p in periodsOrder:
        data.append(df.loc[p])

    return pd.concat(data, axis=axis, ignore_index=True)


def formatOptimizationOutput(
    data, varType, dimension, ip, periodsOrder=None, compDict=None, esM=None
):
    """Functionality for formatting the optimization output. The function is used in the
    setOptimalValues()-method of the ComponentModel class.

    **Required arguments:**

    :param data: Optimized values that should be formatted given as dictionary with the keys (component, location).
    :type data: dict

    :param varType: Define which type of variables are formatted. Options:
        * 'designVariables',
        * 'operationVariables'.
    :type varType: string

    :param dimension: Define the dimension of the data. Options:
        * '1dim',
        * '2dim'.
    :type dimension: string

    :param ip: investment period of transformation path analysis.
    :type ip: int

    **Default arguments:**
    :param periodsOrder: order of the periods of the time series data
        (list, [0] when considering a full temporal resolution,
        [typicalPeriod(0), ... ,typicalPeriod(totalNumberOfTimeSteps/numberOfTimeStepsPerPeriod-1)]
        when applying time series aggregation).
        The periodsOrder must be given if the varType is operationVariables because the full time series has to
        be re-engineered (not necessarily required if no time series aggregation methods are used).
        |br| * the default value is None.
    :type periodsOrder: list

    :param compDict: Dictionary of the component instances of interest.
        compDict is required if dimension is set to 2.
        |br| * the default value is None.
    :type compDict: dict

    :param esM: EnergySystemModel instance representing the energy system in which the components are modeled.
        An esM instance must be given if the varType is operationVariables because the full time series has to
        be re-engineered (not necessarily required if no time series aggregation methods are used).
        |br| * the default value is None
    :type esM: EnergySystemModel instance

    :return: formatted version of data. If data is an empty dictionary, it returns None.
    :rtype: pandas DataFrame
    """
    # If data is an empty dictionary (because no variables of that type were declared) return None
    if not data:
        return None
    # If the dictionary is not empty, format it into a DataFrame
    if varType == VarType.DESIGN and dimension == Dimension.ONE:
        # Convert dictionary to DataFrame, transpose, put the components name first and sort the index
        # Results in a one dimensional DataFrame
        df = pd.DataFrame(data, index=[0]).T.swaplevel(i=0, j=1, axis=0).sort_index()
        df = df[df.index.get_level_values(2) == ip]
        df = df.reset_index(level=2, drop=True)
        # Unstack the regions (convert to a two dimensional DataFrame with the region indices being the columns)
        # and fill NaN values (i.e. when a component variable was not initiated for that region)

        df = df.unstack(level=-1)
        # Get rid of the unnecessary 0 level
        df.columns = df.columns.droplevel()
        return df
    if varType == VarType.DESIGN and dimension == Dimension.TWO:
        # Convert dictionary to DataFrame, transpose, put the components name first while keeping the order of the
        # regions and sort the index
        # Results in a one dimensional DataFrame
        df = pd.DataFrame(data, index=[0]).T
        df = df[df.index.get_level_values(2) == ip]
        df = df.reset_index(level=2, drop=True)
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
    if varType == VarType.OPERATION and dimension == Dimension.ONE:
        # Convert dictionary to DataFrame, transpose, put the period column first and sort the index

        # Results in a one dimensional DataFrame
        df = (
            pd.DataFrame(data, index=[0]).T.swaplevel(i=0, j=-2).sort_index()
        )  # swap location with periods --> periods is first column
        # Unstack the time steps (convert to a two dimensional DataFrame with the time indices being the columns)
        df = df.unstack(level=-1)
        # Get rid of the unnecessary 0 level
        df.columns = df.columns.droplevel()
        # Re-engineer full time series by using Pandas' concat method (only one loop if time series aggregation was not
        # used)
        # filter results for ip
        df = df[df.index.get_level_values(2) == ip]
        # drop ip from index
        df.reset_index(level=2, drop=True, inplace=True)
        return buildFullTimeSeries(df, periodsOrder, ip, esM=esM)
    if varType == VarType.OPERATION and dimension == Dimension.TWO:
        # Convert dictionary to DataFrame, transpose, put the period column first while keeping the order of the
        # regions and sort the index
        # Results in a one dimensional DataFrame
        df = pd.DataFrame(data, index=[0]).T
        indexNew = []
        for tup in df.index.tolist():
            loc1, loc2 = compDict[tup[1]]._mapC[tup[0]]
            indexNew.append((loc1, loc2, tup[1], tup[2], tup[3], tup[4]))
            # indexNew.append((loc1, loc2, tup[1], tup[2], tup[3]))
        df.index = pd.MultiIndex.from_tuples(indexNew)

        # Select rows where ip is equal to investigated ip
        df = df.iloc[df.index.get_level_values(3) == ip]
        # Delete ip from multiindex
        df = df.droplevel(3, axis=0)

        df = (
            df.swaplevel(i=1, j=2, axis=0)
            .swaplevel(i=0, j=3, axis=0)
            .swaplevel(i=2, j=3, axis=0)
            .sort_index()
        )
        # Unstack the time steps (convert to a two dimensional DataFrame with the time indices being the columns)
        df = df.unstack(level=-1)

        # Get rid of the unnecessary 0 level
        df.columns = df.columns.droplevel()

        # Re-engineer full time series by using Pandas' concat method (only one loop if time series aggregation was not
        # used)
        return buildFullTimeSeries(df, periodsOrder, ip, esM=esM)
    raise ValueError(
        "The varType parameter has to be either 'designVariables' or 'operationVariables'\n"
        + "and the dimension parameter has to be either '1dim' or '2dim'."
    )


def setOptimalComponentVariables(optVal, varType, compDict):
    """MISSING."""
    if optVal is not None:
        for compName, comp in compDict.items():
            if compName in optVal.index:
                setattr(comp, varType, optVal.loc[compName])
            else:
                setattr(comp, varType, None)


def process2dimCapacityData(esM, name, data, years):
    """MISSING."""
    data = preprocess2dimInvestmentPeriodData(esM, name, data, years)
    for year in years:
        data[year] = preprocess2dimData(data[year])
    return data


def preprocess2dimInvestmentPeriodData(
    esM,
    name,
    data,
    ComponentInvestmentPeriods,
    locationalEligibility=None,
    mapC=None,
    discard=True,
):
    """MISSING."""
    parameter = {}
    for ip in ComponentInvestmentPeriods:
        # map of year name (e.g. 2020) to internal name (e.g. 0)
        year = int(esM.startYear + ip * esM.investmentPeriodInterval)

        if (
            isinstance(data, int)
            or isinstance(data, float)
            or isinstance(data, pd.DataFrame)
            or isinstance(data, pd.Series)
            or data is None
        ):
            parameter[ip] = data
        elif isinstance(data, dict):
            parameter[ip] = preprocess2dimData(
                data[year], mapC, locationalEligibility, discard
            )
        else:
            raise TypeError(
                f"Parameter of {name} should be a pandas dataframe or a dictionary."
            )

    return parameter


def preprocess2dimData(data, mapC=None, locationalEligibility=None, discard=True):
    """Change format of 2-dimensional data (for transmission components)."""

    def preprocessDataPerIp(data):
        if data is not None and isinstance(data, pd.DataFrame):
            if mapC is None:
                index, data_ = [], []
                counter = 0
                if data.isnull().values.any():
                    data.fillna(0, inplace=True)
                    warnings.warn(
                        "Invalid input.  A matrix contains NaNs. NaN-values are adapted to Zero automatically. Please check your input!"
                    )

                for loc1 in data.columns:
                    for loc2 in data.index:
                        if loc1 != loc2:
                            if discard:
                                # Structure: data[column][row]
                                if data[loc1][loc2] > 0:
                                    (
                                        index.append(loc1 + "_" + loc2),
                                        data_.append(data[loc1][loc2]),
                                    )
                            elif data[loc1][loc2] >= 0:
                                (
                                    index.append(loc1 + "_" + loc2),
                                    data_.append(data[loc1][loc2]),
                                )
                        elif counter == 0:
                            if data[loc1][loc2] != 0:
                                warnings.warn(
                                    "Matrix diagonale contains Non-Zeros. Location is connected to itself. Matrix adapted automatically. Please check your input!"
                                )
                                counter = counter + 1

                data_ = pd.Series(data_, index=index)
                data_.sort_index(inplace=True)
                return data_
            data_ = pd.Series(mapC).apply(lambda loc: data[loc[0]][loc[1]])
            data_.sort_index(inplace=True)
            return data_
        if isinstance(data, float) and locationalEligibility is not None:
            data_ = data * locationalEligibility
            data_.sort_index(inplace=True)
            return data_
        if isinstance(data, int) and locationalEligibility is not None:
            data_ = data * locationalEligibility
            data_.sort_index(inplace=True)
            return data_
        if isinstance(data, pd.Series):
            return data.sort_index()
        return data

    if isinstance(data, dict):
        return {ip: preprocessDataPerIp(data[ip]) for ip in data.keys()}
    return preprocessDataPerIp(data)


def map2dimData(data, mapC):
    """Missing."""
    if data is not None and isinstance(data, pd.DataFrame):
        return pd.Series(mapC).apply(lambda loc: data[loc[0]][loc[1]])
    return data


def output(output, verbose, val):
    """Output a message using logging.

    :param output: The message to output
    :type output: str
    :param verbose: The current verbosity level
    :type verbose: int
    :param val: The verbosity threshold for this message (0 = INFO, >0 = DEBUG)
    :type val: int
    """
    if verbose == val:
        logger = logging.getLogger(__name__)
        if val == 0:
            logger.info(output)
        else:
            logger.debug(output)


def checkNumberOfConversionFactors(commods):
    """Missing."""
    if len(commods) > 2:
        if all([isinstance(value, (int, float)) for value in commods.values()]):
            raise ValueError(
                "Currently commodityConversionFactors are overwritten by commodityConversionFactorsPartLoad."
            )
        raise ValueError(
            "Currently only two commodities are allowed in conversion processes that use commodityConversionFactorsPartLoad."
        )
    return True


# ruff: noqa: PLW0127 # included technically unnecessary elif for readability
def checkAndSetTimeHorizon(
    startYear, endYear=None, nbOfSteps=None, nbOfRepresentedYears=None
):
    """Check if there are enough input parameters given for defining the time horizon for the myopic approach.
    Calculate the number of optimization steps and the number of represented years per each step if not given.
    """
    if (endYear is not None) & (nbOfSteps is None) & (nbOfRepresentedYears is None):
        # endYear is given; determine the nbOfRepresentedYears
        diff = endYear - startYear

        def biggestDivisor(diff):
            for i in [10, 5, 3, 2, 1]:
                if diff % i == 0:
                    return i
            return i

        nbOfRepresentedYears = biggestDivisor(diff)
        nbOfSteps = int(diff / nbOfRepresentedYears)
    elif (
        (endYear is None) & (nbOfSteps is not None) & (nbOfRepresentedYears is not None)
    ):
        # Endyear will be calculated by nbOfSteps and nbOfRepresentedYears
        nbOfSteps = nbOfSteps
    elif (endYear is None) & (nbOfSteps is not None) & (nbOfRepresentedYears is None):
        # If number of steps is given but no endyear and no the number of represented years per optimization run,
        # nbOfRepresentedYears is set to 1 year.
        nbOfRepresentedYears = 1
    elif (endYear is not None) & (nbOfSteps is not None):
        diff = endYear - startYear
        if diff % nbOfSteps != 0:
            raise ValueError(
                "Number of Steps does not fit for the given time horizon between start and end year."
            )
        if (diff % nbOfSteps == 0) & (nbOfRepresentedYears is not None):
            if diff / nbOfSteps != nbOfRepresentedYears:
                raise ValueError(
                    "Number of represented years does not fit for the given time horizon and the number of steps."
                )
    elif (
        (endYear is not None) & (nbOfSteps is None) & (nbOfRepresentedYears is not None)
    ):
        diff = endYear - startYear
        if diff % nbOfRepresentedYears != 0:
            raise ValueError(
                "Number of represented Years is not an integer divisor of the requested time horizon."
            )
        nbOfSteps = int(diff / nbOfRepresentedYears)
    else:
        nbOfSteps = 1
        nbOfRepresentedYears = 1

    return nbOfSteps, nbOfRepresentedYears


def checkStockYears(
    stockCommissioning, startYear, investmentPeriodInterval, ipTechnicalLifetime
):
    """Missing."""
    if stockCommissioning is None:
        return [], []
    if not isinstance(stockCommissioning, dict):
        raise ValueError("stockCommissioning must be None or a dict")

    # check years
    for year, yearly_stock in stockCommissioning.items():
        if not isinstance(year, int):
            raise ValueError("Years of stockCommissioning must be int")
        if year >= startYear:
            raise ValueError("Stock years must be smaller than the start year")
        if (year - startYear) % investmentPeriodInterval != 0:
            raise ValueError(
                f"stockCommissioning was initialized for {year} "
                + "but can only be initialized for "
                + "years which are a multiple of the investment period length."
            )
    stockYears = [x for x in stockCommissioning.keys()]
    processedStockYears = [
        int((x - startYear) / investmentPeriodInterval)
        for x in stockCommissioning.keys()
    ]
    processedStockYears = [
        x for x in processedStockYears if x >= -ipTechnicalLifetime.max()
    ]

    return stockYears, processedStockYears


def checkAndSetStock(component, esM, stockCommissioning):
    """Missing."""
    if stockCommissioning is None:
        return stockCommissioning

    # check type of stockCommissioning
    if not isinstance(stockCommissioning, dict):
        raise TypeError("stockCommissioning must be None or a dict")

    # get regions
    if component.dimension == Dimension.ONE:
        regions = esM.locations
    if component.dimension == Dimension.TWO:
        regions = [
            loc1 + "_" + loc2
            for loc1 in esM.locations
            for loc2 in esM.locations
            if loc1 != loc2
        ]
    # check data for stockCommissioning
    for year, yearly_stock in stockCommissioning.items():
        if not isinstance(year, int):
            raise ValueError("Years of stockCommissioning must be int")
        if (year - esM.startYear) % esM.investmentPeriodInterval != 0:
            raise ValueError(
                f"stockCommissioning was initialized for {year} "
                + "but can only be initialized for "
                + "years which are a multiple of the investment period length."
            )
        # float and int for capacity are only allowed if there is only one region
        if isinstance(yearly_stock, int) or isinstance(yearly_stock, float):
            if not len(esM.locations) == 1:
                raise ValueError(
                    "esM has more than one location, so the location of the stock has to be set."
                )
            # if there is only one region, convert into pd.series region:stock
            isPositiveNumber(yearly_stock)
            stockCommissioning[year] = pd.Series(
                data={list(esM.locations)[0]: yearly_stock}
            )
        elif isinstance(yearly_stock, pd.Series):
            # series must have all locations as index and float/int for values

            if not sorted(yearly_stock.index) == sorted(regions):
                raise ValueError(
                    f"Initialize the stock for all regions for the year '{year}'"
                    + " even if its just 0"
                )
            if any(
                not isinstance(x, float)
                and not isinstance(x, int)
                and not isinstance(x, np.int64)
                for x in yearly_stock.values
            ):
                raise ValueError(f"Stock capacities in year '{year}' must be int/float")

        else:
            raise TypeError(
                "stockCommissioning must be a dict of keys for years and "
                + "pd.Series with location as index and stock as value."
            )

    # check if capacityFix and capacityMax is kept per region
    for loc in regions:
        installed_sum = 0
        for year in stockCommissioning.keys():
            if year < esM.startYear - component.technicalLifetime[loc]:
                pass
            else:
                # for floating numbers a normal sum can lead to floating point issues,
                # e.g. 4.19+2.19=6.380000000000001
                # therefore a rounding is applied, as otherwise the following
                if (
                    stockCommissioning[year][loc]
                    - round(stockCommissioning[year][loc], 10)
                    != 0
                ):
                    warnings.warn(
                        f"A stock comissioning of {stockCommissioning[year][loc]} was "
                        + f"passed for location {loc} in year {year}. "
                        + "It will be rounded to 10 digits to "
                        + "check if the installed stock capacity does "
                        + "not exceed capacityMax and capacityFix"
                    )
                installed_sum += round(stockCommissioning[year][loc], 10)
                installed_sum = round(installed_sum, 10)
        # reduce the installed_sum by the decommissioning, which will occur in
        # the first year
        if (
            esM.startYear - component.technicalLifetime[loc]
            in stockCommissioning.keys()
        ):
            installed_sum -= stockCommissioning[
                esM.startYear - component.technicalLifetime[loc]
            ][loc]
        if component.processedCapacityMax[0] is not None:
            if installed_sum > component.processedCapacityMax[0][loc]:
                raise ValueError(
                    f"The stock of {installed_sum} for '{component.name}' in region '{loc}' "
                    + f"exceeds its capacityMax of '{component.processedCapacityMax}' in the first year"
                )
        if component.processedCapacityFix[0] is not None:
            if installed_sum > component.processedCapacityFix[0][loc]:
                raise ValueError(
                    f"The stock of '{component.name}' in region '{loc}' "
                    + f"exceeds its capacityFix of '{component.processedCapacityFix}' in the first year"
                )

    # set into correct format, add 0'values and transform ip into [-1,-2,-3,...]
    # filter for commissioned stock older than technical lifetime and set to 0
    stock_df = pd.DataFrame.from_dict(stockCommissioning).T
    for loc in regions:
        yearsWithStockOlderThanTechLifetime = [
            x
            for x in stock_df.index
            if x < esM.startYear - component.technicalLifetime[loc]
        ]
        stockOlderThanTechnicalLifetime = stock_df.loc[
            yearsWithStockOlderThanTechLifetime, loc
        ]
        if len(yearsWithStockOlderThanTechLifetime) > 0:
            warnings.warn(
                f"Stock of component {component.name} in location "
                + f"{loc} will not be considered "
                + f"for years {list(stockOlderThanTechnicalLifetime.index)} as it "
                + "exceeds the technical lifetime. A capacity of "
                + f"{stockOlderThanTechnicalLifetime.sum().sum()} will be dropped."
            )
            stock_df.loc[yearsWithStockOlderThanTechLifetime, loc] = 0

    # convert original years to ip named years (e.g. -1,-2,-3)
    stock_df.index = [
        int((x - esM.startYear) / esM.investmentPeriodInterval) for x in stock_df.index
    ]

    # fill missing year for timeframe of entire technical lifetime
    if component.floorTechnicalLifetime:
        maxTechnicalLifetime = math.floor(component.ipTechnicalLifetime.max())
    else:
        maxTechnicalLifetime = math.ceil(component.ipTechnicalLifetime.max())
    allStockYears = [x for x in range(-1, -maxTechnicalLifetime - 1, -1)]
    stock_df = stock_df.reindex(allStockYears).fillna(0)
    return stock_df.T.to_dict(orient="series")


def setStockCapacityStartYear(component, esM, dimension):
    """Missing."""
    if dimension == Dimension.ONE:
        regions = esM.locations
    elif dimension == Dimension.TWO:
        regions = [
            loc1 + "_" + loc2
            for loc1 in esM.locations
            for loc2 in esM.locations
            if loc1 != loc2
        ]
    if component.processedStockCommissioning is None:
        return pd.Series(index=regions, data=0)

    stockCapacityStartYear = pd.Series()
    for loc in regions:
        _stock_location = 0
        if component.floorTechnicalLifetime:
            ipTechLifetime = math.floor(component.ipTechnicalLifetime[loc])
        else:
            ipTechLifetime = math.ceil(component.ipTechnicalLifetime[loc])
        for year in range(-1, -ipTechLifetime - 1, -1):
            _stock_location += component.processedStockCommissioning[year].loc[loc]
        stockCapacityStartYear[loc] = _stock_location
    return stockCapacityStartYear


def checkCO2ReductionTargets(CO2ReductionTargets, nbOfSteps):
    """Check if the CO2 reduction target is either None or the length of the given list equals the number of optimization steps."""
    if CO2ReductionTargets is not None:
        if len(CO2ReductionTargets) != nbOfSteps + 1:
            raise ValueError(
                "CO2ReductionTargets has to be None, or the length of the given list must equal the number \
 of optimization steps."
            )


def checkSinkCompCO2toEnvironment(esM, CO2ReductionTargets):
    """Check if a sink component object called >CO2 to environment< exists.
    This component is required if CO2 reduction targets are given.
    """
    if CO2ReductionTargets is not None:
        if "CO2 to environment" not in esM.componentNames:
            warnings.warn(
                "CO2 emissions are not considered in the current esM. CO2ReductionTargets will be ignored."
            )
            return None
    return CO2ReductionTargets


def checkSimultaneousChargeDischarge(tsCharge, tsDischarge):
    """Check if simultaneous charge and discharge occurs for StorageComponent.
    :param tsCharge: Charge time series of component, which is checked. Can be retrieved from
        chargeOperationVariablesOptimum.loc[compName]. Columns are the time steps, index are the regions.
    :type tsCharge: pd.DataFrame
    :param tsDischarge: Discharge time series of component, which is checked. Can be retrieved from
        dischargeOperationVariablesOptimum.loc[compName]. Columns are the time steps, index are the regions.
    :type tsDischarge: pd.DataFrame.

    :return: simultaneousChargeDischarge: Boolean with information if simultaneous charge & discharge happens
    :type simultaneousChargeDischarge: bool
    """
    # Merge Charge and Discharge Series
    ts = pd.concat([tsCharge.T, tsDischarge.T], axis=1)
    # If no simultaneous charge and discharge occurs ts[region][ts[region] > 0] will only return nan values. After
    # dropping them the len() is 0 and the check returns False. This is done for all regions in the list comprehension.
    # If any() region returns True the check returns True.
    return any(
        [
            len(ts[region][ts[region] > 0].dropna()) > 0
            for region in set(ts.columns.values)
        ]
    )


def addEmptyRegions(esM, data):
    """Check empty regions.
    If data for a region is missing, fill with 0s.
    """
    esM_locations = esM.locations
    data_locations = data.index
    missing_locations = [loc for loc in esM_locations if loc not in data_locations]

    if isinstance(data, pd.Series):
        for loc in missing_locations:
            tst = pd.Series([0], index=[loc])
            data = pd.concat([data, tst], axis=0)

    elif isinstance(data, pd.DataFrame):
        for loc in missing_locations:
            if loc not in data.columns:
                data[loc] = 0

    return data


def annuityPresentValueFactor(esM, compName, loc, years):
    """Calculate annuity of present value factor."""
    # DE:Rentenbarwertfaktor
    interestRate = esM.getComponent(compName).interestRate[loc]
    if interestRate == 0:
        return years
    return (((1 + interestRate) ** (years)) - 1) / (
        interestRate * (1 + interestRate) ** (years)
    )


def discountFactor(esM, ip, compName, loc):
    """Calculate discount factors."""
    return (
        1
        / (1 + esM.getComponent(compName).interestRate[loc])
        ** (ip * esM.investmentPeriodInterval)
        * (1 + esM.getComponent(compName).interestRate[loc])
    )


def checkConversionFactorProperties(comp, esM, commisDependingCcf):
    """Check commodity conversion factors (ccf) in order to determine if the conversion component is.
    a) ipDepending (ccf changes with investment period it is operated (e.g. due to weather changes))
    b) commisDepending (ccf changes based on year a component is commissioned (e.g. due to technological improvements)
    c) flexibleConversion (component can decide which commodity to use (within a specified commodity group)).
    """
    isIpDepending = False
    isCommisDepending = False
    flexibleConversion = False
    # Check that type is a dict
    if not isinstance(comp.commodityConversionFactors, dict):
        raise ValueError("commodityConversionFactor must be a dict")

    # 0. get a copy of the commodityConversionFactors
    commodityConversionFactors = comp.commodityConversionFactors.copy()

    # 1. check if the commodity conversion varies
    # a) not at all over transformation pathway
    # b) per investment period -> weather dependency
    # c) per commissioning year and investment period
    dictInDict = any(isinstance(x, dict) for x in commodityConversionFactors.values())
    commisInvestmentPeriodTuple = [
        (x, y)
        for x in (comp.stockYears + esM.investmentPeriodNames)
        for y in esM.investmentPeriodNames
        if x <= y < x + comp.technicalLifetime.max()
    ]
    dictKeys = sorted(list(commodityConversionFactors.keys()))

    if not dictInDict and commisDependingCcf:
        raise ValueError(
            'If parameter "commisDependingCcf" is set to True '
            "commodity conversion factors must be specified per "
            f"investment period. Please check {comp.name}"
        )
    if dictInDict and dictKeys == esM.investmentPeriodNames:
        # commodity conversion is not varied between the investment periods
        # the CCF can either be depended on the commissioning year or investment period
        if commisDependingCcf:
            isCommisDepending = True
        else:
            isIpDepending = True
        if any(
            isinstance(x, dict)
            for x in commodityConversionFactors[dictKeys[0]].values()
        ):
            flexibleConversion = True

    elif dictInDict and dictKeys == sorted(commisInvestmentPeriodTuple):
        # input keys of commodity conversion are varied over investment periods and commissioning years
        if any(
            isinstance(x, dict)
            for x in commodityConversionFactors[dictKeys[0]].values()
        ):
            raise NotImplementedError(
                "The combination of flexible and "
                "commissioning dependent conversion is not supported"
            )
        isIpDepending = True
        isDataVariating = False
        # check if also data is varied over commissioning year
        for ip in esM.investmentPeriodNames:
            # get commodity conversion factors in ip for all possible commissioning years
            _commisYearsForIp = [
                (x, y) for (x, y) in commisInvestmentPeriodTuple if y == ip
            ]
            _commodConvFactorForIp = [
                commodityConversionFactors[(x, y)] for (x, y) in _commisYearsForIp
            ]
            # define first one commodityConversionFactor of the ip as base to compare
            _baseCommodConvFactor = commodityConversionFactors[_commisYearsForIp[0]]

            # compare if all commodities are same as in the baseCommodConvFactor
            for ccf in _commodConvFactorForIp:
                for commod in ccf.keys():
                    # check for same datatype
                    if type(ccf[commod]) is not type(_baseCommodConvFactor[commod]):
                        raise ValueError(
                            f"Unallowed data type variation for commodity {commod} for yearly dependency."
                        )
                    if isinstance(ccf[commod], (pd.Series, pd.DataFrame)):
                        if not ccf[commod].equals(_baseCommodConvFactor[commod]):
                            isDataVariating = True
                            break
                    elif not ccf[commod] == _baseCommodConvFactor[commod]:
                        isDataVariating = True
                        break
        # if data is varying, set commis depending true
        if isDataVariating:
            isCommisDepending = True
    elif dictInDict and all(isinstance(x, str) for x in dictKeys):
        flexibleConversion = True
    elif dictInDict:
        raise ValueError(
            f"Wrong format for commodityConversionFactors for {comp.name}. "
            f"Please check the init."
        )

    return (isIpDepending, isCommisDepending, flexibleConversion)


def checkNestedNanValues(obj):
    """Missing."""
    if isinstance(obj, dict):
        return any(checkNestedNanValues(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(checkNestedNanValues(v) for v in obj)
    if isinstance(obj, pd.Series):
        return obj.isnull().any()
    if isinstance(obj, pd.DataFrame):
        return obj.isnull().values.any()
    if isinstance(obj, float):
        return math.isnan(obj)
    return False


def checkAndSetCommodityConversionFactor(comp, esM):
    """Set up the full commodity conversion factor, if necessary depending on
    commissioning year and investment period. Location-dependent parameter
    can be provided as pandas.Series indexed by locations or pandas.DataFrame
    with locations as columns and timesteps as index.
    """
    iterationList = esM.investmentPeriodNames
    commodityConversionFactors = comp.commodityConversionFactors.copy()
    if comp.isCommisDepending:
        iterationList = [
            (x, y)
            for x in (comp.stockYears + esM.investmentPeriodNames)
            for y in esM.investmentPeriodNames
            if x <= y < x + comp.technicalLifetime.max()
        ]
    elif comp.isIpDepending and isinstance(
        list(comp.commodityConversionFactors.keys())[0], tuple
    ):
        commodityConversionFactors = {
            ip: comp.commodityConversionFactors[(x, y)]
            for ip in esM.investmentPeriodNames
            for (x, y) in comp.commodityConversionFactors.keys()
            if y == ip
        }

    # 2. Check and set up commodity conversion factors
    def checkFactorCommod(ccf):
        if comp.flexibleConversion:
            commodities = []
            commodTypes = []
            for item in ccf.items():
                if isinstance(item[1], dict):
                    if item[0] in esM.commodities:
                        raise ValueError(
                            "Commodity group names must be different from commodity names. "
                            f"Group name '{item[0]}' is not valid.\n"
                            "Hint: If you want investment-period-dependent conversion factors, use:\n"
                            "  {YEAR: {'electricity': ..., 'hydrogen': ...}, ...}\n"
                            "and not:\n"
                            "  {'hydrogen': {YEAR: ...}}"
                        )
                    commodities += list(item[1].keys())
                    commodTypes += [
                        type(x)
                        for x in item[1].values()
                        if isinstance(x, (pd.Series, pd.DataFrame))
                    ]
                    if checkNestedNanValues(item[1]):
                        raise ValueError(
                            f"Commodity conversion factors for '{item[0]}' contain NaN values."
                        )
                    vals = list(item[1].values())
                    if not any(isinstance(v, (pd.Series, pd.DataFrame)) for v in vals):
                        if not (all(v > 0 for v in vals) or all(v < 0 for v in vals)):
                            raise ValueError(
                                f"All commodity conversion factors of {comp.name}"
                                f" in commodity group '{item[0]}' must have the same sign."
                            )
                else:
                    commodities.append(item[0])
                    commodTypes.append(type(item[1]))
        else:
            commodities = list(set(ccf.keys()))
            commodTypes = [
                type(x)
                for x in ccf.values()
                if isinstance(x, (pd.Series, pd.DataFrame))
            ]

            for key, value in ccf.items():
                if isinstance(value, dict):
                    raise ValueError(
                        f"{comp.name}: Invalid commodityConversionFactors format: found a nested dict under key '{key}'. "
                        "If you want investment-period-dependent conversion factors, use:\n"
                        "  {YEAR: {'electricity': ..., 'hydrogen': ...}, ...}\n"
                        "and not:\n"
                        "  {'hydrogen': {YEAR: ...}}"
                    )

                if isinstance(value, float) and math.isnan(value):
                    raise ValueError(f"NaN found at key '{key}'")

                if isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, float) and math.isnan(v):
                            raise ValueError(
                                f"NaN found at key '{key}' in list index {i}"
                            )

        checkCommodities(esM, set(commodities))
        return commodTypes

    def isLocationSeries(series):
        return set(series.index) <= set(esM.locations)

    if comp.isIpDepending or comp.isCommisDepending:
        commodTypesList = []
        for ccf in commodityConversionFactors.values():
            commodTypes = checkFactorCommod(ccf)
            commodTypesList += commodTypes
        if (pd.Series in commodTypesList or pd.DataFrame in commodTypesList) and len(
            set(commodTypesList)
        ) > 1:
            raise ValueError(
                f"Unallowed data type variation in commodity conversion factors of {comp.name} for yearly dependency."
            )
    else:
        checkFactorCommod(commodityConversionFactors)

    # 3. Setup of fullCommodityConversionFactor, processedConversionFactor and preprocessedConversionFactor
    fullCommodityConversionFactor = {}
    processedCommodityConversionFactor = {}
    preprocessedCommodityConversionFactor = {}
    for _key in iterationList:
        # get the required name for the keys of the resulting dict
        if comp.isCommisDepending:
            (y1, y2) = _key
            commisYearInternalName = int(
                (y1 - esM.startYear) / esM.investmentPeriodInterval
            )
            yearInternalName = esM.investmentPeriodNames.index(y2)
            newKeyName = (commisYearInternalName, yearInternalName)
        else:
            newKeyName = esM.investmentPeriodNames.index(_key)

        # get the original commodity conversion
        if comp.isIpDepending or comp.isCommisDepending:
            _commodityConversionFactors = commodityConversionFactors[_key]
        else:
            _commodityConversionFactors = commodityConversionFactors

        # initialize empty
        fullCommodityConversionFactor[newKeyName] = {}
        processedCommodityConversionFactor[newKeyName] = {}
        preprocessedCommodityConversionFactor[newKeyName] = {}

        for key, value in _commodityConversionFactors.items():
            if isinstance(value, dict):
                group = key
                processedCommodityConversionFactor[newKeyName][group] = {}
                processedCommodityConversionFactor[newKeyName][group] = {}
                preprocessedCommodityConversionFactor[newKeyName][group] = {}
                for commod in value.keys():
                    if isinstance(
                        _commodityConversionFactors[group][commod],
                        (pd.Series, pd.DataFrame),
                    ):
                        raise NotImplementedError(
                            "Flexible conversion components currently do not support "
                            "time series data for commodity conversion factors."
                        )

                    if isinstance(
                        _commodityConversionFactors[group][commod], (int, float)
                    ):
                        # fix values do not need a time-series aggregation and are written
                        # directly to processedCommodityConversion
                        processedCommodityConversionFactor[newKeyName][group][
                            commod
                        ] = _commodityConversionFactors[group][commod]
                        preprocessedCommodityConversionFactor[newKeyName][group][
                            commod
                        ] = processedCommodityConversionFactor[newKeyName][group][
                            commod
                        ]
                    else:
                        raise ValueError(
                            f"Data type '{_commodityConversionFactors}' for commodity "
                            + f"{commod} in {_key} not accepted."
                        )
            else:
                commod = key
                _factor = _commodityConversionFactors[commod]
                if isinstance(_factor, pd.Series):
                    if isLocationSeries(_factor):
                        processedCommodityConversionFactor[newKeyName][commod] = (
                            checkRegionalIndex(
                                esM, _factor.copy(), comp.locationalEligibility
                            )
                        )
                        preprocessedCommodityConversionFactor[newKeyName][commod] = (
                            processedCommodityConversionFactor[newKeyName][commod]
                        )
                    else:
                        fullCommodityConversionFactor[newKeyName][commod] = (
                            checkAndSetTimeSeriesConversionFactors(
                                esM, _factor, comp.locationalEligibility
                            )
                        )
                        preprocessedCommodityConversionFactor[newKeyName][commod] = (
                            fullCommodityConversionFactor[newKeyName][commod]
                        )
                elif isinstance(_factor, pd.DataFrame):
                    fullCommodityConversionFactor[newKeyName][commod] = (
                        checkAndSetTimeSeriesConversionFactors(
                            esM, _factor, comp.locationalEligibility
                        )
                    )
                    preprocessedCommodityConversionFactor[newKeyName][commod] = (
                        fullCommodityConversionFactor[newKeyName][commod]
                    )
                elif isinstance(_factor, (int, float)):
                    # fix values do not need a time-series aggregation and are written
                    # directly to processedCommodityConversion
                    processedCommodityConversionFactor[newKeyName][commod] = (
                        _commodityConversionFactors[commod]
                    )
                    preprocessedCommodityConversionFactor[newKeyName][commod] = (
                        processedCommodityConversionFactor[newKeyName][commod]
                    )
                else:
                    raise ValueError(
                        f"Data type '{_commodityConversionFactors}' for commodity "
                        + f"{commod} in {_key} not accepted."
                    )

    if comp.isCommisDepending and comp.flexibleConversion:
        raise ValueError(
            "Flexible Conversion is currently not available for commissioning"
            " year depended commodity conversion factors"
        )
    return (
        fullCommodityConversionFactor,
        processedCommodityConversionFactor,
        preprocessedCommodityConversionFactor,
    )


def checkEmissionFactors(comp, esM):
    """Check emission factors for flexible conversion components."""

    def is_nan(val):
        return isinstance(val, float) and math.isnan(val)

    if comp.emissionFactors is None:
        return None
    if not comp.flexibleConversion:
        raise NotImplementedError(
            "Emission factors can only be defined for flexible conversion components. "
            "For non flexible conversion components emission factors must be introduced "
            "within the commodity conversion factors. "
            f"Please check parameters for {comp.name}."
        )

    if any(not isinstance(key, str) for key in comp.emissionFactors.keys()):
        raise NotImplementedError(
            "Emission factors can not be specified per investment period."
        )
    for key, value in comp.emissionFactors.items():
        if isinstance(value, float) and is_nan(value):
            raise ValueError(f"NaN found in emission factor for key '{key}'")
        if isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, float) and is_nan(v):
                    raise ValueError(
                        f"NaN found in emission factor list for key '{key}', index {i}"
                    )
        if checkNestedNanValues(value):
            raise ValueError(
                f"Emission factors for '{comp.name}', '{key}' contain NaN values."
            )

    emission_commodities = list(comp.emissionFactors.keys())
    commodities = [
        commod
        for ef_group in comp.emissionFactors.values()
        for commod in ef_group.keys()
    ]

    if not set(emission_commodities + commodities).issubset(esM.commodities):
        raise ValueError(
            f"Error in emission commodities of emission factors for {comp.name}. "
            f"One or more of the emission commodities are not defined in the model."
        )

    flex_commodities = [
        commod
        for group in list(comp.processedCommodityConversionFactors.values())[0].values()
        if isinstance(group, dict)
        for commod in group.keys()
    ]

    if not set(commodities).issubset(flex_commodities):
        raise ValueError(
            "Emission factor commodities must also be present in conversion factor commodities. "
            f"Please check {comp.name}"
        )

    if any(
        emission_factor <= 0
        for emission_commod in comp.emissionFactors.keys()
        for emission_factor in comp.emissionFactors[emission_commod].values()
    ):
        raise ValueError(f"Emission factors of {comp.name} must be positive numbers.")

    return comp.emissionFactors


def checkAndSetFlowShares(comp, esM):
    """Check flow shares for flexible conversion components."""
    if comp.flowShares is None:
        return None
    if not comp.flexibleConversion:
        raise NotImplementedError(
            "Flow shares can only be defined for flexible conversion components. "
            f"Please check parameters for {comp.name}."
        )
    if not isinstance(comp.flowShares, dict):
        raise ValueError("Flow shares must be defined as a dictionary.")
    flex_commodities = [
        commod
        for group in list(comp.processedCommodityConversionFactors.values())[0].values()
        if isinstance(group, dict)
        for commod in group.keys()
    ]

    def checkFlowShares(flowShares):
        if not set(flowShares.keys()).issubset({"min", "max", "fix"}):
            raise ValueError(
                'Flow shares must be specified as "min", "max", or "fix" '
                f"values. Please check parameters for {comp.name}."
            )
        if "fix" in flowShares.keys() and not len(flowShares.keys()) == 1:
            raise ValueError(
                "If a flow share fix is passed no other flow shares are allowed."
            )
        for param in flowShares.keys():
            for commod, flowShare in flowShares[param].items():
                if commod not in flex_commodities:
                    raise ValueError(
                        "Flow shares commodities must be defined as flexible "
                        f"commodity. Please check {commod} in {comp.name}"
                    )
                if isinstance(flowShare, pd.Series):
                    if not ((flowShare <= 1).all() and (flowShare >= 0).all()):
                        raise ValueError("Flow shares must be between 0 and 1.")
                    if not flowShare.index.isin(esM.locations).all():
                        raise ValueError(
                            "Some of the specified locations in the "
                            "Flow shares are not represented in the ESM."
                        )
                elif not (0 <= flowShare <= 1):
                    raise ValueError("Flow shares must be between 0 and 1.")
                else:
                    flowShares[param][commod] = pd.Series(
                        flowShare, index=esM.locations
                    )
        return flowShares

    if set(comp.flowShares.keys()).issubset(set(esM.investmentPeriodNames)):
        if list(comp.flowShares.keys()) != esM.investmentPeriodNames:
            warnings.warn(
                f"Flow shares for {comp.name} were not defined for all investment periods."
            )
        processedFlowShares = {}
        for ipName, flowSharesIp in comp.flowShares.items():
            ip = esM.investmentPeriodNames.index(ipName)
            processedFlowShares[ip] = checkFlowShares(flowSharesIp)
    else:
        processedFlowShares = {
            ip: checkFlowShares(comp.flowShares) for ip in esM.investmentPeriods
        }

    return processedFlowShares


def getParametersForUnevenLifetimes(compName, loc, lifetimeAttr, esM):
    """Get parameters for uneven lifetimes."""
    ipEconomicLifetime = getattr(esM.getComponent(compName), "ipEconomicLifetime")[loc]
    ipTechnicalLifetime = getattr(esM.getComponent(compName), "ipTechnicalLifetime")[
        loc
    ]

    # A) Fix operational costs for design variables.
    # Fix operation costs are applied over the entire operational time.
    # The duration of the operation time depends on the technical lifetime and
    # (in case it is not a multiple of the interval) weather it is floored
    # or ceiled to the next interval.
    if lifetimeAttr == "ipTechnicalLifetime":
        if esM.getComponent(compName).floorTechnicalLifetime:
            intervalsWithCompleteCosts = math.floor(ipTechnicalLifetime)
        else:
            intervalsWithCompleteCosts = math.ceil(ipTechnicalLifetime)
        # The following two parameters unrelevant for operation costs
        hasDesignCostsInEndingPartOfLastTechnicalLifetimeInterval = False
        hasDesignCostsInStartingPartOfLastEconomicLifetimeInterval = False

    # B) Costs for design variables.
    # The applied costs for the design variables are more complex.
    # The cost distrubutiuon depends on the economic lifetime, the technical
    # lifetime, the flooring/ceiling of the technical lifetime to the next
    # interval and the length of the interval.
    # Complex example: interval of 5 years, economic lifetime of 8 years,
    # technical lifetime of 13 years and technical lifetime is ceiled to 15 years
    # Then design costs need to be applied for
    # - first interval (0-4): all years of interval with costs
    # - second interval (5-9): costs only in years 5,6,7
    # - third interval (10-14): costs only in years 14,15 (as new capacity is required,
    #   the specific costs of the first interval are used)
    else:
        # if the technical and economic lifetime are in the same interval, both are affected by flooring
        economicAndTechnicalLifetimeInSameInterval = math.floor(
            ipEconomicLifetime
        ) == math.floor(ipTechnicalLifetime)
        if (
            economicAndTechnicalLifetimeInSameInterval
            and esM.getComponent(compName).floorTechnicalLifetime
        ):
            # example: interval 5, economic lifetime 6, technical lifetime 7
            # both lifetimes are then floored to 5
            _ipEconomicLifetime = math.floor(ipEconomicLifetime)
            _ipTechnicalLifetime = math.floor(ipTechnicalLifetime)
            # by rounding, no intervals will contain costs only for a few years
            hasDesignCostsInEndingPartOfLastTechnicalLifetimeInterval = False
            hasDesignCostsInStartingPartOfLastEconomicLifetimeInterval = False
        else:
            # example: interval 5, economic lifetime 7, technical lifetime 12
            _ipEconomicLifetime = ipEconomicLifetime
            if esM.getComponent(compName).floorTechnicalLifetime:
                # example: technical lifetime is floored to 10, year 10 and 11 not relevant and without costs
                hasDesignCostsInEndingPartOfLastTechnicalLifetimeInterval = False
                _ipTechnicalLifetime = math.floor(ipTechnicalLifetime)
            else:
                # example: technical lifetime is ceiled to 15, year 10 and 11 without costs, year 12,13,14 require additional costs
                hasDesignCostsInEndingPartOfLastTechnicalLifetimeInterval = True
                _ipTechnicalLifetime = ipTechnicalLifetime

            # economic lifetime leading to overhead years in last interval
            if _ipEconomicLifetime % 1 != 0:
                hasDesignCostsInStartingPartOfLastEconomicLifetimeInterval = True
            else:
                hasDesignCostsInStartingPartOfLastEconomicLifetimeInterval = False

        # interval with cost in all included years
        intervalsWithCompleteCosts = math.floor(_ipEconomicLifetime)

    return (
        intervalsWithCompleteCosts,
        hasDesignCostsInStartingPartOfLastEconomicLifetimeInterval,
        hasDesignCostsInEndingPartOfLastTechnicalLifetimeInterval,
    )

# ============================== START OF INFEASIBILITY CHECKS ==============================
# Infeasibility pre-checks
#
# The functions below detect model setups which are provably infeasible
# before the optimization problem is declared, so that the user gets a
# readable message instead of a solver 'infeasible' status.
#
# Every check is a NECESSARY condition and is deliberately optimistic:
# unknown or unbounded quantities are resolved in favour of feasibility.
# A reported problem therefore proves infeasibility, while a clean run is
# no feasibility guarantee. Known blind spots are documented per check.

def _getFirstInvestmentPeriodData(data):
    """Return the data of the first investment period.

    Processed component attributes are stored as dictionaries keyed by
    investment period, e.g. {0: pandas Series}. Any other input is
    returned unchanged.

    :param data: attribute value of a component
    :type data: dict, pandas Series, pandas DataFrame, number or None

    :return: data of the first investment period
    :rtype: pandas Series, pandas DataFrame, number or None
    """
    if isinstance(data, dict):
        if not data:
            return None
        return data[sorted(data.keys())[0]]
    return data


def _getComponentTimeSeries(comp, baseName):
    """Return a time series of a component independent of the attribute naming.

    The processed, full and raw attribute names are tried in that order.
    The processed time series is preferred because it is the one the
    optimization problem is built from.

    :param comp: component of interest
    :type comp: Component instance

    :param baseName: name of the time series without prefix,
        e.g. 'operationRateFix'
    :type baseName: string

    :return: time series with the locations as columns, or None if the
        component does not hold such a time series
    :rtype: pandas DataFrame or None
    """
    for attr in (
        f"processed{baseName[0].upper()}{baseName[1:]}",
        f"full{baseName[0].upper()}{baseName[1:]}",
        baseName,
    ):
        timeSeries = _getFirstInvestmentPeriodData(getattr(comp, attr, None))
        if isinstance(timeSeries, pd.DataFrame):
            return timeSeries
    return None


def _getEligibleLocations(comp, esM):
    """Return the locations at which a component can be built or operated.

    A locationalEligibility of None means that the component is eligible
    everywhere.

    :param comp: component of interest
    :type comp: Component instance

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :return: eligible locations
    :rtype: set of strings
    """
    eligibility = _getFirstInvestmentPeriodData(
        getattr(comp, "processedLocationalEligibility", None)
    )
    if eligibility is None:
        eligibility = _getFirstInvestmentPeriodData(
            getattr(comp, "locationalEligibility", None)
        )
    if eligibility is not None:
        return {loc for loc, val in eligibility.items() if val > 0}
    return set(esM.locations)


def _parseTransmissionEdgeKey(edgeKey, locations):
    """Split a transmission edge key 'loc1_loc2' into its two locations.

    The known location names are matched (longest first) instead of
    splitting at the underscore, so that location names containing
    underscores are handled correctly.

    :param edgeKey: edge key of a transmission component
    :type edgeKey: string

    :param locations: locations of the energy system model
    :type locations: set of strings

    :return: the two connected locations
    :rtype: tuple of two strings
    """
    for loc in sorted(locations, key=len, reverse=True):
        if edgeKey.startswith(loc + "_"):
            rest = edgeKey[len(loc) + 1 :]
            if rest in locations:
                return loc, rest
    raise ValueError(f"The edge key '{edgeKey}' could not be parsed.")


def _getCapacityPerLocation(comp, esM):
    """Return the upper capacity bound of a component per location.

    A fixed capacity takes precedence over a maximum capacity. Missing
    values are set to infinity, i.e. an unbounded capacity, which keeps
    the checks optimistic. A scalar capacity applies per location.

    :param comp: component of interest
    :type comp: Component instance

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :return: capacity bound indexed by location
    :rtype: pandas Series
    """
    for attr in (
        "processedCapacityFix",
        "capacityFix",
        "processedCapacityMax",
        "capacityMax",
    ):
        capacity = _getFirstInvestmentPeriodData(getattr(comp, attr, None))
        if capacity is None:
            continue
        if isinstance(capacity, pd.Series):
            return capacity.astype(float).fillna(np.inf)
        return pd.Series(
            float(capacity), index=sorted(_getEligibleLocations(comp, esM))
        )
    return pd.Series(np.inf, index=sorted(_getEligibleLocations(comp, esM)))


def _getCapacityPerEdge(comp):
    """Return the capacity bound of a transmission component per edge.

    A defaultdict is returned so that a lookup works for scalar
    capacities and for edges which are not contained in the data. Missing
    values are set to infinity.

    :param comp: transmission component of interest
    :type comp: Transmission instance

    :return: capacity bound indexed by edge key
    :rtype: collections.defaultdict
    """
    for attr in (
        "processedCapacityFix",
        "capacityFix",
        "processedCapacityMax",
        "capacityMax",
    ):
        capacity = _getFirstInvestmentPeriodData(getattr(comp, attr, None))
        if capacity is None:
            continue
        if isinstance(capacity, pd.Series):
            capacityDict = capacity.astype(float).fillna(np.inf).to_dict()
            return defaultdict(lambda: np.inf, capacityDict)
        value = float(capacity)
        return defaultdict(lambda: value)
    return defaultdict(lambda: np.inf)


def _getEdgeValue(data, edgeKey, default=0.0):
    """Resolve a transmission attribute for one specific edge.

    The attribute may be None, a scalar or a Series indexed by edge key.
    NaN values are replaced by the default.

    :param data: attribute value, e.g. losses or distances
    :type data: None, number, pandas Series or dict

    :param edgeKey: edge key of interest
    :type edgeKey: string

    :param default: value used if no entry exists
        |br| * the default value is 0.0
    :type default: float

    :return: value of the attribute at this edge
    :rtype: float
    """
    data = _getFirstInvestmentPeriodData(data)
    if data is None:
        return default
    if isinstance(data, pd.Series):
        if edgeKey in data.index:
            value = data[edgeKey]
            return default if pd.isna(value) else float(value)
        return default
    return float(data)


def _hasPositiveSupply(comp, loc):
    """Check whether a source can deliver a positive amount at a location.

    A source with an operation rate time series which is zero at a
    location cannot supply anything there, even though it may be eligible.

    :param comp: source component of interest
    :type comp: Source instance

    :param loc: location of interest
    :type loc: string

    :return: True if the source can deliver at this location
    :rtype: bool
    """
    for baseName in ("operationRateMax", "operationRateFix"):
        rate = _getComponentTimeSeries(comp, baseName)
        if rate is not None and loc in rate.columns:
            return rate[loc].sum() > 0
    return True


def _splitSourcesAndSinks(esM):
    """Split the components of the SourceSinkModel into sources and sinks.

    A Sink inherits from Source, therefore the sign attribute
    (1 for sources, -1 for sinks) is used to distinguish them.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :return: sources and sinks
    :rtype: tuple of two lists
    """
    sources, sinks = [], []
    srcSnkModel = esM.componentModelingDict.get("SourceSinkModel")
    if srcSnkModel:
        for comp in srcSnkModel.componentsDict.values():
            (sources if comp.sign == 1 else sinks).append(comp)
    return sources, sinks


def _getConversionFactors(comp):
    """Return the commodity conversion factors as a flat dictionary.

    An investment-period-dependent nesting is unwrapped. It is
    recognizable by integer keys instead of commodity names.

    :param comp: conversion component of interest
    :type comp: Conversion instance

    :return: conversion factors indexed by commodity
    :rtype: dict
    """
    factors = comp.commodityConversionFactors
    if isinstance(factors, dict) and all(isinstance(key, int) for key in factors):
        factors = _getFirstInvestmentPeriodData(factors)
    return factors


def _getTransmissionLinks(esM):
    """Return the usable transmission links with their maximum flow per time step.

    The maximum flow is the capacity multiplied by the hours per time step
    and reduced by the transmission losses. If processed losses are
    available they already include the distance, otherwise the losses per
    distance unit are multiplied by the distance.

    Both directions of a symmetric eligibility matrix describe the same
    physical link and are therefore not added up. Parallel links, i.e.
    several transmission components between the same locations, are added up.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :return: maximum flow per time step, indexed by commodity and by the
        frozenset of the two connected locations
    :rtype: dict of dicts
    """
    hoursPerTimeStep = esM.hoursPerTimeStep
    links = defaultdict(dict)
    transModel = esM.componentModelingDict.get("TransmissionModel")

    if transModel is None:
        return {}

    for comp in transModel.componentsDict.values():
        eligibility = _getFirstInvestmentPeriodData(
            getattr(comp, "processedLocationalEligibility", None)
        )
        if eligibility is None:
            eligibility = _getFirstInvestmentPeriodData(
                getattr(comp, "locationalEligibility", None)
            )

        if eligibility is None:
            # Optimistic fallback. It is stated explicitly because it can
            # hide network bottlenecks from all subsequent checks.
            warnings.warn(
                f"No locationalEligibility found for '{comp.name}'. The "
                "infeasibility pre-checks assume that all location pairs "
                "are connected."
            )
            edgeItems = [
                (f"{loc1}_{loc2}", 1)
                for loc1 in esM.locations
                for loc2 in esM.locations
                if loc1 != loc2
            ]
        else:
            edgeItems = list(eligibility.items())

        capacities = _getCapacityPerEdge(comp)

        # Processed losses, if available, already include the distance
        losses = getattr(comp, "processedLosses", None)
        lossesIncludeDistance = losses is not None
        if losses is None:
            losses = getattr(comp, "losses", None)
        distances = getattr(comp, "processedDistances", None)
        if distances is None:
            distances = getattr(comp, "distances", None)

        for edgeKey, isEligible in edgeItems:
            if isEligible <= 0:
                continue
            loc1, loc2 = _parseTransmissionEdgeKey(edgeKey, esM.locations)
            capacity = capacities[edgeKey]
            loss = _getEdgeValue(losses, edgeKey, 0.0)
            if lossesIncludeDistance:
                efficiency = max(0.0, 1.0 - loss)
            else:
                distance = _getEdgeValue(distances, edgeKey, 0.0)
                efficiency = max(0.0, 1.0 - loss * distance)
            flow = float(capacity) * hoursPerTimeStep * efficiency

            key = frozenset({loc1, loc2})
            previousFlow = links[comp.commodity].get((comp.name, key), 0.0)
            links[comp.commodity][(comp.name, key)] = max(previousFlow, flow)

    aggregatedLinks = defaultdict(lambda: defaultdict(float))
    for commodity, componentLinks in links.items():
        for (_, key), flow in componentLinks.items():
            aggregatedLinks[commodity][key] += flow
    return {
        commodity: dict(componentLinks)
        for commodity, componentLinks in aggregatedLinks.items()
    }


def _getTransmissionIslands(esM, transmissionEdges):
    """Return the connected groups of locations per commodity.

    Locations which are not connected by a transmission component of that
    commodity form a group of their own. Every commodity of the energy
    system model is contained in the result, so that a lookup never fails
    for commodities which only occur inside conversion processes.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :param transmissionEdges: edges given as (commodity, loc1, loc2)
    :type transmissionEdges: list of tuples

    :return: mapping of location to its connected group, per commodity
    :rtype: dict of dicts
    """
    edgesPerCommodity = defaultdict(set)
    for commodity, loc1, loc2 in transmissionEdges:
        edgesPerCommodity[commodity].add((loc1, loc2))

    allCommodities = (
        set(getattr(esM, "commodities", set()))
        | set(edgesPerCommodity)
        | {
            comp.commodity
            for model in esM.componentModelingDict.values()
            for comp in model.componentsDict.values()
            if hasattr(comp, "commodity")
        }
    )

    islands = {}
    for commodity in allCommodities:
        remaining = set(esM.locations)
        locationToIsland = {}
        adjacency = defaultdict(set)
        for loc1, loc2 in edgesPerCommodity.get(commodity, ()):
            adjacency[loc1].add(loc2)
            adjacency[loc2].add(loc1)
        while remaining:
            start = remaining.pop()
            group, stack = {start}, [start]
            while stack:
                for neighbour in adjacency[stack.pop()]:
                    if neighbour in remaining:
                        remaining.discard(neighbour)
                        group.add(neighbour)
                        stack.append(neighbour)
            island = frozenset(group)
            for loc in group:
                locationToIsland[loc] = island
        islands[commodity] = locationToIsland
    return islands


def _getMaxTransportableFlow(supplyPerLocation, demandPerLocation, links, tol=1e-6):
    """Return the maximum transportable flow of one commodity in one time step.

    A maximum flow problem is solved on a network with an artificial
    source connected to the local supply, an artificial sink connected to
    the local demand and the transmission links in between. By the
    max-flow min-cut theorem this covers every possible group of
    locations at once and therefore also detects bottlenecks on a path
    across intermediate locations.

    :param supplyPerLocation: local supply indexed by location
    :type supplyPerLocation: dict

    :param demandPerLocation: fixed demand indexed by location
    :type demandPerLocation: dict

    :param links: maximum flow per link, indexed by the frozenset of the
        two connected locations
    :type links: dict

    :param tol: numerical tolerance
        |br| * the default value is 1e-6
    :type tol: float

    :return: maximum transportable flow and total demand
    :rtype: tuple of two floats
    """
    totalDemand = sum(value for value in demandPerLocation.values() if value > 0)
    if totalDemand <= tol:
        return np.inf, 0.0

    # An infinite capacity is replaced by a value which clearly exceeds
    # the total demand, because the maximum flow algorithm requires
    # finite capacities.
    bigCapacity = totalDemand * 1e6

    graph = nx.DiGraph()
    for loc, supply in supplyPerLocation.items():
        if supply > 0:
            graph.add_edge("_supply_", loc, capacity=min(supply, bigCapacity))
    for loc, demand in demandPerLocation.items():
        if demand > 0:
            graph.add_edge(loc, "_demand_", capacity=demand)
    for locations, capacity in links.items():
        loc1, loc2 = sorted(locations)
        capacity = min(capacity, bigCapacity)
        graph.add_edge(loc1, loc2, capacity=capacity)
        graph.add_edge(loc2, loc1, capacity=capacity)

    if "_supply_" not in graph or "_demand_" not in graph:
        return 0.0, totalDemand

    maxFlow, _ = nx.maximum_flow(graph, "_supply_", "_demand_")
    return maxFlow, totalDemand


def _hasConsistentTimeSeriesLength(esM, timeSeriesList):
    """Check whether all given time series cover the full time horizon.

    Time-step-resolved checks can only be evaluated if the time series
    match the number of time steps of the energy system model, which is
    not the case for aggregated time series.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :param timeSeriesList: time series to be checked
    :type timeSeriesList: list of pandas DataFrames and None

    :return: True if all time series cover the full time horizon
    :rtype: bool
    """
    return all(
        len(timeSeries) == esM.numberOfTimeSteps
        for timeSeries in timeSeriesList
        if timeSeries is not None
    )


def checkCommodityReachability(esM):
    """Check whether every demanded commodity can be provided at its location.

    Starting from the commodities which the sources can deliver, the set
    of available (commodity, location) pairs is extended until it does not
    grow any further: a conversion component adds its output commodities
    at a location if all of its input commodities are available there, and
    a transmission component spreads a commodity to the connected location.

    The check is purely qualitative and detects structural errors such as
    missing components, missing transmission links or supply time series
    which are zero. Quantities are not considered at all.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :return: description of every detected problem, empty if the check passed
    :rtype: list of strings
    """
    problems = []
    sources, sinks = _splitSourcesAndSinks(esM)
    convModel = esM.componentModelingDict.get("ConversionModel")
    transModel = esM.componentModelingDict.get("TransmissionModel")

    available = set()
    for comp in sources:
        for loc in _getEligibleLocations(comp, esM):
            if _hasPositiveSupply(comp, loc):
                available.add((comp.commodity, loc))

    transmissionEdges = []
    if transModel:
        for comp in transModel.componentsDict.values():
            eligibility = _getFirstInvestmentPeriodData(
                getattr(comp, "processedLocationalEligibility", None)
            )
            if eligibility is None:
                eligibility = _getFirstInvestmentPeriodData(
                    getattr(comp, "locationalEligibility", None)
                )
            if eligibility is not None:
                locationPairs = [
                    _parseTransmissionEdgeKey(edgeKey, esM.locations)
                    for edgeKey, isEligible in eligibility.items()
                    if isEligible > 0
                ]
            else:
                locationPairs = [
                    (loc1, loc2)
                    for loc1 in esM.locations
                    for loc2 in esM.locations
                    if loc1 != loc2
                ]
            for loc1, loc2 in locationPairs:
                # Transmission components can be operated in both directions
                transmissionEdges.append((comp.commodity, loc1, loc2))
                transmissionEdges.append((comp.commodity, loc2, loc1))

    changed = True
    while changed:
        changed = False
        if convModel:
            for comp in convModel.componentsDict.values():
                factors = _getConversionFactors(comp)
                inputs = {
                    commod for commod, factor in factors.items() if factor < 0
                }
                outputs = {
                    commod for commod, factor in factors.items() if factor > 0
                }
                for loc in _getEligibleLocations(comp, esM):
                    if all((commod, loc) in available for commod in inputs):
                        newlyAvailable = {
                            (commod, loc) for commod in outputs
                        } - available
                        if newlyAvailable:
                            available |= newlyAvailable
                            changed = True
        for commodity, loc1, loc2 in transmissionEdges:
            if (commodity, loc1) in available and (commodity, loc2) not in available:
                available.add((commodity, loc2))
                changed = True

    for snk in sinks:
        demand = _getComponentTimeSeries(snk, "operationRateFix")
        if demand is None:
            continue
        for loc in demand.columns:
            if demand[loc].sum() > 0 and (snk.commodity, loc) not in available:
                problems.append(
                    f"The sink '{snk.name}' has a demand for the commodity "
                    f"'{snk.commodity}' in location '{loc}', but the commodity "
                    "can neither be produced nor imported there."
                )
    return problems


def checkJointInputDemand(esM, aggregate=True, tol=1e-6, maxIteration=50):
    """Check whether the sources cover the input demand of all sinks at once.

    The fixed demand of all sinks is propagated backwards through the
    conversion chains, so that conversion components which consume the
    same commodity are considered simultaneously. This detects a shortage
    which only occurs because several conversion components compete for
    the same input commodity, e.g. an electrolyzer and a heat pump which
    both consume electricity.

    The check is exact if every commodity is produced by exactly one type
    of conversion component. If several components produce the same
    commodity, only their combined capacity is checked and the input
    demand is not propagated, which keeps the check optimistic.

    Not considered: the state of charge of storage components over time,
    and transmission capacities within a connected group of locations.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :param aggregate: if True, the balance is set up for the whole time
        horizon and storage components are not considered, because they
        only shift energy in time. If False, the balance is set up for
        each time step and the discharge bound of the storage components
        is added to the supply.
        |br| * the default value is True
    :type aggregate: bool

    :param tol: numerical tolerance
        |br| * the default value is 1e-6
    :type tol: float

    :param maxIteration: maximum number of backward propagation steps
        |br| * the default value is 50
    :type maxIteration: int

    :return: description of every detected problem, empty if the check passed
    :rtype: list of strings
    """
    problems = []
    hoursPerTimeStep = esM.hoursPerTimeStep
    numberOfTimeSteps = esM.numberOfTimeSteps

    sources, sinks = _splitSourcesAndSinks(esM)
    convModel = esM.componentModelingDict.get("ConversionModel")
    storModel = esM.componentModelingDict.get("StorageModel")
    conversions = list(convModel.componentsDict.values()) if convModel else []
    storages = list(storModel.componentsDict.values()) if storModel else []

    transmissionLinks = _getTransmissionLinks(esM)
    islands = _getTransmissionIslands(
        esM,
        [
            (commodity, *sorted(locations))
            for commodity, links in transmissionLinks.items()
            for locations in links
        ],
    )

    sourceData = [
        (
            comp,
            _getComponentTimeSeries(comp, "operationRateFix"),
            _getComponentTimeSeries(comp, "operationRateMax"),
            _getCapacityPerLocation(comp, esM),
            _getEligibleLocations(comp, esM),
        )
        for comp in sources
    ]

    conversionData = []
    for comp in conversions:
        factors = _getConversionFactors(comp)
        conversionData.append(
            (
                comp,
                {
                    commod: abs(factor)
                    for commod, factor in factors.items()
                    if factor < 0
                },
                {
                    commod: factor
                    for commod, factor in factors.items()
                    if factor > 0
                },
                _getCapacityPerLocation(comp, esM),
                _getEligibleLocations(comp, esM),
            )
        )

    sinkData = [
        (snk, _getComponentTimeSeries(snk, "operationRateFix")) for snk in sinks
    ]

    if not aggregate:
        allTimeSeries = [rate for _, rate in sinkData] + [
            rateFix if rateFix is not None else rateMax
            for _, rateFix, rateMax, _, _ in sourceData
        ]
        if not _hasConsistentTimeSeriesLength(esM, allTimeSeries):
            output(
                "The time-step-resolved input demand check is skipped because "
                "the time series do not cover the full time horizon.",
                esM.verboseLogLevel,
                0,
            )
            return problems

    timeSteps = [None] if aggregate else list(range(numberOfTimeSteps))

    for timeStep in timeSteps:
        label = "Total time horizon" if timeStep is None else f"Time step {timeStep}"
        numberOfSteps = numberOfTimeSteps if timeStep is None else 1

        # Supply per commodity and connected group of locations
        supply = defaultdict(float)
        for comp, rateFix, rateMax, capacities, eligibleLocations in sourceData:
            rate = rateFix if rateFix is not None else rateMax
            for loc in eligibleLocations:
                key = (comp.commodity, islands[comp.commodity][loc])
                if comp.hasCapacityVariable:
                    capacity = float(capacities.get(loc, np.inf))
                    if rate is not None and loc in rate.columns:
                        relativeOperation = (
                            float(rate[loc].sum())
                            if timeStep is None
                            else float(rate[loc].iloc[timeStep])
                        )
                    else:
                        relativeOperation = float(numberOfSteps)
                    supply[key] += (
                        0.0
                        if relativeOperation == 0
                        else capacity * relativeOperation * hoursPerTimeStep
                    )
                elif rate is not None and loc in rate.columns:
                    supply[key] += (
                        float(rate[loc].sum())
                        if timeStep is None
                        else float(rate[loc].iloc[timeStep])
                    )
                else:
                    supply[key] += np.inf

        if timeStep is not None:
            # The discharge bound is an upper bound only. Whether the
            # storage could have been charged before is not checked.
            for comp in storages:
                capacities = _getCapacityPerLocation(comp, esM)
                dischargeRate = float(getattr(comp, "dischargeRate", 1) or 1)
                dischargeEfficiency = float(
                    getattr(comp, "dischargeEfficiency", 1) or 1
                )
                for loc in _getEligibleLocations(comp, esM):
                    key = (comp.commodity, islands[comp.commodity][loc])
                    supply[key] += (
                        float(capacities.get(loc, np.inf))
                        * dischargeRate
                        * dischargeEfficiency
                        * hoursPerTimeStep
                    )

        # Fixed demand per commodity and connected group of locations
        required = defaultdict(float)
        for snk, demand in sinkData:
            if demand is None:
                continue
            commodity = snk.commodity
            for loc in demand.columns:
                demandValue = (
                    float(demand[loc].sum())
                    if timeStep is None
                    else float(demand[loc].iloc[timeStep])
                )
                if demandValue > 0:
                    required[(commodity, islands[commodity][loc])] += demandValue

        # Backward propagation of the deficits through the conversion chains
        for _ in range(maxIteration):
            changed = False
            for (commodity, island), requiredValue in list(required.items()):
                deficit = requiredValue - supply[(commodity, island)]
                if deficit <= tol:
                    continue

                producers = [
                    (comp, inputs, outputs, capacities, eligibleLocations)
                    for comp, inputs, outputs, capacities, eligibleLocations in conversionData
                    if commodity in outputs
                    and any(loc in island for loc in eligibleLocations)
                ]

                if not producers:
                    problems.append(
                        f"{label}: the joint demand of {requiredValue:.4g} for the "
                        f"commodity '{commodity}' in the locations {sorted(island)} "
                        f"exceeds the supply of {supply[(commodity, island)]:.4g} "
                        "and no component produces this commodity."
                    )
                    supply[(commodity, island)] = requiredValue
                    changed = True
                    continue

                combinedOutput = sum(
                    float(capacities.get(loc, np.inf))
                    * hoursPerTimeStep
                    * numberOfSteps
                    * outputs[commodity]
                    for _, _, outputs, capacities, eligibleLocations in producers
                    for loc in eligibleLocations
                    if loc in island
                )
                if combinedOutput + tol < deficit:
                    problems.append(
                        f"{label}: the deficit of {deficit:.4g} for the commodity "
                        f"'{commodity}' in the locations {sorted(island)} exceeds "
                        f"the combined conversion capacity of {combinedOutput:.4g}."
                    )

                if len(producers) == 1:
                    # A unique producer allows to propagate its input demand
                    _, inputs, outputs, _, eligibleLocations = producers[0]
                    operation = deficit / outputs[commodity]
                    locationsInIsland = [
                        loc for loc in eligibleLocations if loc in island
                    ]
                    for inputCommodity, factor in inputs.items():
                        inputIslands = {
                            islands[inputCommodity][loc] for loc in locationsInIsland
                        }
                        if len(inputIslands) == 1:
                            required[
                                (inputCommodity, next(iter(inputIslands)))
                            ] += operation * factor
                    for outputCommodity, factor in outputs.items():
                        if outputCommodity != commodity:
                            outputIslands = {
                                islands[outputCommodity][loc]
                                for loc in locationsInIsland
                            }
                            if len(outputIslands) == 1:
                                supply[
                                    (outputCommodity, next(iter(outputIslands)))
                                ] += operation * factor

                supply[(commodity, island)] = requiredValue
                changed = True
            if not changed:
                break

    return problems


def checkJointInputDemandAggregated(esM):
    """Check the joint input demand for the whole time horizon.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :return: description of every detected problem, empty if the check passed
    :rtype: list of strings
    """
    return checkJointInputDemand(esM, aggregate=True)


def checkJointInputDemandPerTimeStep(esM):
    """Check the joint input demand for each time step.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :return: description of every detected problem, empty if the check passed
    :rtype: list of strings
    """
    return checkJointInputDemand(esM, aggregate=False)


def checkTimeStepBalance(esM, tol=1e-6, maxIteration=50):
    """Check the commodity balance for each time step and each location.

    Three necessary conditions are evaluated per time step and commodity:

    a) per location: the local supply of the sources, the discharge bound
       of the storage components, the local conversion output and the
       capacity of the adjacent transmission links have to cover the
       fixed demand of that location,
    b) per connected group of locations: the pooled supply has to cover
       the pooled demand,
    c) over the whole network: the maximum transportable flow has to
       cover the total demand. This also detects bottlenecks on a path
       across intermediate locations, which a) and b) cannot see.

    Condition c) is the strictest one, a) and b) are kept because they
    localize the shortage and therefore give a more specific message.

    Not considered: the state of charge of storage components over time,
    the competition of conversion components for the same input commodity
    and the simultaneous optimization of several commodities.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :param tol: numerical tolerance
        |br| * the default value is 1e-6
    :type tol: float

    :param maxIteration: maximum number of conversion iterations
        |br| * the default value is 50
    :type maxIteration: int

    :return: description of every detected problem, empty if the check passed
    :rtype: list of strings
    """
    problems = []
    hoursPerTimeStep = esM.hoursPerTimeStep
    numberOfTimeSteps = esM.numberOfTimeSteps

    sources, sinks = _splitSourcesAndSinks(esM)
    convModel = esM.componentModelingDict.get("ConversionModel")
    storModel = esM.componentModelingDict.get("StorageModel")
    conversions = list(convModel.componentsDict.values()) if convModel else []
    storages = list(storModel.componentsDict.values()) if storModel else []

    transmissionLinks = _getTransmissionLinks(esM)
    transmissionEdges = [
        (commodity, *sorted(locations))
        for commodity, links in transmissionLinks.items()
        for locations in links
    ]
    islands = _getTransmissionIslands(esM, transmissionEdges)

    importCapacity = defaultdict(float)
    for commodity, links in transmissionLinks.items():
        for locations, flow in links.items():
            for loc in locations:
                importCapacity[(commodity, loc)] += flow

    sourceData = [
        (
            comp,
            _getComponentTimeSeries(comp, "operationRateFix"),
            _getComponentTimeSeries(comp, "operationRateMax"),
            _getCapacityPerLocation(comp, esM),
            _getEligibleLocations(comp, esM),
        )
        for comp in sources
    ]

    storageSupply = defaultdict(float)
    for comp in storages:
        capacities = _getCapacityPerLocation(comp, esM)
        dischargeRate = float(getattr(comp, "dischargeRate", 1) or 1)
        dischargeEfficiency = float(getattr(comp, "dischargeEfficiency", 1) or 1)
        for loc in _getEligibleLocations(comp, esM):
            storageSupply[(comp.commodity, loc)] += (
                float(capacities.get(loc, np.inf))
                * dischargeRate
                * dischargeEfficiency
                * hoursPerTimeStep
            )

    conversionData = []
    for comp in conversions:
        factors = _getConversionFactors(comp)
        conversionData.append(
            (
                comp,
                {
                    commod: abs(factor)
                    for commod, factor in factors.items()
                    if factor < 0
                },
                {
                    commod: factor
                    for commod, factor in factors.items()
                    if factor > 0
                },
                _getCapacityPerLocation(comp, esM),
                _getEligibleLocations(comp, esM),
            )
        )

    sinkData = [
        (snk, _getComponentTimeSeries(snk, "operationRateFix")) for snk in sinks
    ]

    allTimeSeries = [rate for _, rate in sinkData] + [
        rateFix if rateFix is not None else rateMax
        for _, rateFix, rateMax, _, _ in sourceData
    ]
    if not _hasConsistentTimeSeriesLength(esM, allTimeSeries):
        output(
            "The time-step-resolved balance check is skipped because the time "
            "series do not cover the full time horizon.",
            esM.verboseLogLevel,
            0,
        )
        return problems

    for timeStep in range(numberOfTimeSteps):
        # Local supply of the sources and the storage components
        localSupply = defaultdict(float)
        for comp, rateFix, rateMax, capacities, eligibleLocations in sourceData:
            rate = rateFix if rateFix is not None else rateMax
            for loc in eligibleLocations:
                if comp.hasCapacityVariable:
                    capacity = float(capacities.get(loc, np.inf))
                    relativeOperation = (
                        float(rate[loc].iloc[timeStep])
                        if rate is not None and loc in rate.columns
                        else 1.0
                    )
                    localSupply[(comp.commodity, loc)] += (
                        0.0
                        if relativeOperation == 0
                        else capacity * relativeOperation * hoursPerTimeStep
                    )
                elif rate is not None and loc in rate.columns:
                    localSupply[(comp.commodity, loc)] += float(
                        rate[loc].iloc[timeStep]
                    )
                else:
                    localSupply[(comp.commodity, loc)] += np.inf
        for key, value in storageSupply.items():
            localSupply[key] += value

        # Conversion output, with the inputs taken from the connected group
        conversionOutput = defaultdict(float)
        for _ in range(maxIteration):
            pooledSupply = defaultdict(float)
            for (commodity, loc), value in localSupply.items():
                pooledSupply[(commodity, islands[commodity][loc])] += value
            for (commodity, loc), value in conversionOutput.items():
                pooledSupply[(commodity, islands[commodity][loc])] += value

            newOutput = defaultdict(float)
            for _, inputs, outputs, capacities, eligibleLocations in conversionData:
                for loc in eligibleLocations:
                    capacity = float(capacities.get(loc, np.inf))
                    operation = (
                        np.inf
                        if np.isinf(capacity)
                        else capacity * hoursPerTimeStep
                    )
                    for inputCommodity, factor in inputs.items():
                        operation = min(
                            operation,
                            pooledSupply[
                                (inputCommodity, islands[inputCommodity][loc])
                            ]
                            / factor,
                        )
                    if operation > 0:
                        for outputCommodity, factor in outputs.items():
                            newOutput[(outputCommodity, loc)] += operation * factor

            keys = set(conversionOutput) | set(newOutput)
            if all(
                math.isclose(
                    conversionOutput[key], newOutput[key], rel_tol=1e-9, abs_tol=tol
                )
                or (np.isinf(conversionOutput[key]) and np.isinf(newOutput[key]))
                for key in keys
            ):
                conversionOutput = newOutput
                break
            conversionOutput = newOutput

        totalLocalSupply = defaultdict(float, localSupply)
        for key, value in conversionOutput.items():
            totalLocalSupply[key] += value

        pooledSupply = defaultdict(float)
        for (commodity, loc), value in totalLocalSupply.items():
            pooledSupply[(commodity, islands[commodity][loc])] += value

        # a) Balance per location
        islandDemand = defaultdict(float)
        locationDemand = defaultdict(float)
        for snk, demand in sinkData:
            if demand is None:
                continue
            commodity = snk.commodity
            for loc in demand.columns:
                demandValue = float(demand[loc].iloc[timeStep])
                if demandValue <= 0:
                    continue
                islandDemand[(commodity, islands[commodity][loc])] += demandValue
                locationDemand[(commodity, loc)] += demandValue

        for (commodity, loc), demandValue in locationDemand.items():
            supply = totalLocalSupply[(commodity, loc)]
            imports = importCapacity[(commodity, loc)]
            if supply + imports + tol < demandValue:
                problems.append(
                    f"Time step {timeStep}: the demand of {demandValue:.4g} for the "
                    f"commodity '{commodity}' in the location '{loc}' exceeds the "
                    f"local supply of {supply:.4g} plus the import capacity of "
                    f"{imports:.4g}."
                )

        # b) Balance per connected group of locations
        for (commodity, island), demandValue in islandDemand.items():
            supply = pooledSupply[(commodity, island)]
            if supply + tol < demandValue:
                problems.append(
                    f"Time step {timeStep}: the demand of {demandValue:.4g} for the "
                    f"commodity '{commodity}' in the locations {sorted(island)} "
                    f"exceeds the maximum supply of {supply:.4g}."
                )

        # c) Maximum transportable flow over the whole network
        for commodity in {commodity for commodity, _ in locationDemand}:
            supplyPerLocation = {
                loc: totalLocalSupply[(commodity, loc)] for loc in esM.locations
            }
            demandPerLocation = {
                loc: value
                for (commod, loc), value in locationDemand.items()
                if commod == commodity
            }
            maxFlow, totalDemand = _getMaxTransportableFlow(
                supplyPerLocation,
                demandPerLocation,
                transmissionLinks.get(commodity, {}),
                tol,
            )
            if maxFlow + tol < totalDemand:
                problems.append(
                    f"Time step {timeStep}: the maximum transportable flow of "
                    f"{maxFlow:.4g} for the commodity '{commodity}' is smaller than "
                    f"the total demand of {totalDemand:.4g}. The supply or the "
                    "transmission capacity of the network is insufficient."
                )

    return problems


#: Pre-checks which are run by default, ordered from a coarse structural
#: check to the more detailed quantitative ones. Each of them detects a
#: type of error which the others cannot detect.
INFEASIBILITY_PRECHECKS = (
    checkCommodityReachability,
    checkJointInputDemandAggregated,
    checkTimeStepBalance,
    checkJointInputDemandPerTimeStep,
)


def runInfeasibilityPrechecks(esM, checks=INFEASIBILITY_PRECHECKS, raiseError=True):
    """Run the infeasibility pre-checks on an energy system model.

    Every check proves infeasibility if it reports a problem, therefore a
    ValueError is raised by default. A check which fails to run, e.g.
    because a component holds unexpected data, only causes a warning so
    that it never blocks a valid model.

    :param esM: EnergySystemModel instance
    :type esM: EnergySystemModel instance

    :param checks: pre-checks to be run. Every check is a function which
        takes an EnergySystemModel instance and returns a list of strings.
        |br| * the default value is INFEASIBILITY_PRECHECKS
    :type checks: tuple of functions

    :param raiseError: if True, a ValueError is raised if a problem was
        detected. If False, the problems are only returned and logged.
        |br| * the default value is True
    :type raiseError: bool

    :return: description of every detected problem, empty if all checks passed
    :rtype: list of strings
    """
    isEnergySystemModelInstance(esM)

    problems = []
    for check in checks:
        checkName = getattr(check, "__name__", str(check))
        try:
            checkProblems = list(check(esM))
        except Exception as exception:
            warnings.warn(
                f"The infeasibility pre-check '{checkName}' could not be run "
                f"and is skipped: {exception!r}"
            )
            continue

        if checkProblems:
            problems.extend(f"{checkName}: {problem}" for problem in checkProblems)
        else:
            output(
                f"The infeasibility pre-check '{checkName}' passed.",
                esM.verboseLogLevel,
                0,
            )

    if problems:
        message = (
            "The infeasibility pre-checks detected that the model cannot be "
            "solved:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + "\n\nThe pre-checks can be deactivated by setting "
            "'runInfeasibilityPrechecks' to False."
        )
        if raiseError:
            raise ValueError(message)
        warnings.warn(message)
    else:
        output(
            "All infeasibility pre-checks passed. Note that they evaluate "
            "necessary conditions only and are no feasibility guarantee.",
            esM.verboseLogLevel,
            0,
        )

    return problems

# ============================== END OF INFEASIBILITY CHECKS ==============================

class _Solver:
    """Solver identifier with mutable value."""

    def __init__(self, value):
        self.value = value


class ImplementedSolvers:
    """Implemented solvers."""

    GLPK = _Solver("glpk")
    GUROBI = _Solver("gurobi")
    HIGHS = _Solver("highs")
    STANDARD_SOLVER = _Solver("gurobi")  # Use Gurobi if available, otherwise use highs

    @staticmethod
    def _gurobi_available():
        """Check if Gurobi is installed with a valid full (non-size-limited) license.

        Creates a Gurobi model that exceeds the 2000-variable limit of the
        restricted license bundled with the gurobipy pip package, then tries
        to optimize it.  If creating the environment fails, no license is
        available at all; if optimize fails, only the restricted license is
        present.  Model and environment are properly disposed of so that
        license tokens are released.

        See https://support.gurobi.com/hc/en-us/articles/4424054948881
        """
        env = None
        model = None
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            model = gp.Model(env=env)
            model.addVars(2001)
            model.optimize()
            return True
        except gp.GurobiError:
            return False
        finally:
            if model is not None:
                model.close()
            if env is not None:
                env.close()

    @classmethod
    def set_standard_solver(cls):
        """Detect available solver and set STANDARD_SOLVER accordingly."""
        if cls._gurobi_available():
            cls.STANDARD_SOLVER.value = cls.GUROBI.value
        else:
            cls.STANDARD_SOLVER.value = cls.HIGHS.value
