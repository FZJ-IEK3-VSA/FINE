import warnings

import pandas as pd
import numpy as np
import math

import FINE as fn


def isString(string):
    """Check if the input argument is a string."""
    if not type(string) == str:
        raise TypeError("The input argument has to be a string")


def equalStrings(ref, test):
    """Check if two strings are equal to each other."""
    if ref != test:
        print("Reference string: " + str(ref))
        print("String: " + str(test))
        raise ValueError("Strings do not match")


def isStrictlyPositiveInt(value):
    """Check if the input argument is a strictly positive integer."""
    if not type(value) == int:
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
    if not type(setOfStrings) == set:
        raise TypeError("The input argument has to be a set")
    if not any([type(r) == str for r in setOfStrings]):
        raise TypeError("The list entries in the input argument must be strings")


def isEnergySystemModelInstance(esM):
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
    if not type(commodityUnitsDict) == dict:
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


def checkTimeUnit(timeUnit):
    """
    Check if the timeUnit input argument is equal to 'h'.
    """
    if not timeUnit == "h":
        raise ValueError("The timeUnit input argument has to be 'h'")


def checkTimeSeriesIndex(esM, data):
    """
    Necessary if the data rows represent the time-dependent data:
    Check if the row-indices of the data match the time indices of the energy system model.
    """
    if list(data.index) != esM.totalTimeSteps:
        raise ValueError(
            "Time indices do not match the one of the specified energy system model."
        )


def checkRegionalColumnTitles(esM, data, locationalEligibility):
    """
    Necessary if the data columns represent the location-dependent data:
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
        else:
            data = addEmptyRegions(esM, data)

    # Sort data according to _locationsOrdered, if not already sorted
    if not np.array_equal(data.columns, esM._locationsOrdered):
        data.sort_index(inplace=True, axis=1)

    return data


def checkRegionalIndex(esM, data, locationalEligibility):
    """
    Necessary if the data rows represent the location-dependent data:
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
        else:
            data = addEmptyRegions(esM, data)

    # Sort data according to _locationsOrdered, if not already sorted
    if not np.array_equal(data.index, esM._locationsOrdered):
        data.sort_index(inplace=True)

    return data


def checkConnectionIndex(data, locationalEligibility):
    """
    Necessary for transmission components:
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
    if not commodityUnit in esM.commodityUnitsDict.values():
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
    """
    Check if the callable conversion factor covers part loads from 0 to 1 and
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
    """
    Check if the given values for the distances are valid (i.e. positive). If the distances parameter is None,
    the distances for the eligible connections are set to 1.
    """
    if distances is None:
        output(
            "The distances of a component are set to a normalized value of 1.",
            esM.verbose,
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
    """
    Check if the type of the losses are valid (i.e. a number, pandas DataFrame or a pandas Series),
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
    if data is None:
        return None
    elif isinstance(data, pd.Series):
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
                        QPbound[ip].loc[x] = (
                            capacityMax[ip].loc[x] - capacityMin[ip].loc[x]
                        )
    return QPbound


def getQPcostDev(investmentPeriods, QPcostScale):
    QPcostDev = {}
    for ip in investmentPeriods:
        QPcostDev[ip] = 1 - QPcostScale[ip]
    return QPcostDev


def checkLocationSpecficDesignInputParams(comp, esM):
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
    partLoadMin = comp.partLoadMin
    name = comp.name
    bigM = comp.bigM
    hasCapacityVariable = comp.hasCapacityVariable

    def checkAndSet(data, comp, esM):
        if data is not None:
            if comp.dimension == "1dim":
                if not isinstance(data, pd.Series):
                    raise TypeError("Input data has to be a pandas Series")
                data = checkRegionalIndex(esM, data, comp.locationalEligibility)
            elif comp.dimension == "2dim":
                if not isinstance(data, pd.Series):
                    raise TypeError("Input data has to be a pandas DataFrame")
                data = checkConnectionIndex(data, comp.locationalEligibility)
            else:
                raise ValueError(
                    "The dimension parameter has to be either '1dim' or '2dim' "
                )
            return data

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

    if sharedPotentialID is not None and capacityMax is None:
        raise ValueError(
            "A capacityMax parameter is required if a sharedPotentialID is considered."
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
                    if esM.verbose < 2:
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


def checkAndSetCapacityBounds(esM, name, capacityMin, capacityMax, capacityFix):
    checkInvestmentPeriodParameters(name, capacityMin, esM.investmentPeriodNames)
    checkInvestmentPeriodParameters(name, capacityMax, esM.investmentPeriodNames)
    checkInvestmentPeriodParameters(name, capacityFix, esM.investmentPeriodNames)

    def _checkAndSet(name, param):
        if isinstance(param, dict):
            if all(x is None for x in param.values()):
                return None
            if any(x is None for x in param.values()):
                raise ValueError(
                    f"{name} cannot switch between None and values for different investment periods."
                )

        processedParam = {}
        for ip in esM.investmentPeriods:
            if param is None:
                processedParam[ip] = None
            if isinstance(param, dict):
                processedParam[ip] = castToSeries(
                    param[esM.investmentPeriodNames[ip]], esM
                )
            elif isinstance(param, pd.DataFrame) or isinstance(param, pd.Series):
                processedParam[ip] = castToSeries(param, esM)
            elif isinstance(param, int) or isinstance(param, float):
                processedParam[ip] = castToSeries(param, esM)
        return processedParam

    # set up parameter as dict with investment periods as keys and
    # dataframe with locations as values
    processedCapacityMin = _checkAndSet(name, capacityMin)
    processedCapacityMax = _checkAndSet(name, capacityMax)
    processedCapacityFix = _checkAndSet(name, capacityFix)

    for ip in esM.investmentPeriods:
        if (
            processedCapacityMin[ip] is not None
            and (processedCapacityMin[ip] < 0).any()
        ):
            raise ValueError("capacityMin values smaller than 0 were detected.")

        if (
            processedCapacityFix[ip] is not None
            and (processedCapacityFix[ip] < 0).any()
        ):
            raise ValueError("capacityFix values smaller than 0 were detected.")

        if (
            processedCapacityMax[ip] is not None
            and (processedCapacityMax[ip] < 0).any()
        ):
            raise ValueError("capacityMax values smaller than 0 were detected.")

        if (
            processedCapacityMin[ip] is not None
            and processedCapacityMax[ip] is not None
        ):
            # Test that capacityMin and capacityMax has the same index for comparing.
            # If capacityMin is missing for some locations, it´s set to 0.
            if set(processedCapacityMin[ip].index).issubset(
                processedCapacityMax[ip].index
            ):
                processedCapacityMin[ip] = (
                    processedCapacityMin[ip]
                    .reindex(processedCapacityMax[ip].index)
                    .fillna(0)
                )
            if (processedCapacityMin[ip] > processedCapacityMax[ip]).any():
                raise ValueError("capacityMin values > capacityMax values detected.")

        if (
            processedCapacityFix[ip] is not None
            and processedCapacityMax[ip] is not None
        ):
            if (processedCapacityFix[ip] > processedCapacityMax[ip]).any():
                raise ValueError("capacityFix values > capacityMax values detected.")

        if (
            processedCapacityFix[ip] is not None
            and processedCapacityMin[ip] is not None
        ):
            if (processedCapacityFix[ip] < processedCapacityMin[ip]).any():
                raise ValueError("capacityFix values < capacityMax values detected.")

    # check if there is a mix of None and specified boundaries in one of the capacityBounds
    def checkForConsistency(name, capacityBound):
        if isinstance(capacityBound, dict):
            if any(x is not None for x in capacityBound.values()):
                if not all(x is not None for x in capacityBound.values()):
                    raise ValueError(
                        "A mix between None and specified values is not allowed "
                        + f"between investment periods for {name}."
                    )

    checkForConsistency("capacityMax", capacityMax)
    checkForConsistency("capacityMin", capacityMin)
    checkForConsistency("capacityFix", capacityFix)

    return processedCapacityMin, processedCapacityMax, processedCapacityFix


def checkInvestmentPeriodParameters(name, param, years):
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

        for key, value in param.items():
            if value is None:
                raise ValueError(
                    f"Currently a dict containing None values cannot be passed for '{name}'"
                )


def checkCapacityDevelopmentWithStock(
    investmentPeriods,
    capacityMax,
    capacityFix,
    stockCommissioning,
    technicalLifetime,
    floorTechnicalLifetime,
):
    if stockCommissioning is None:
        pass
    else:
        # if there is a stock, consider it for the capacity development
        # create a dataframe with columns for location, index for the years and
        # stock capacities as values
        locations = stockCommissioning[-1].index
        years = [x for x in stockCommissioning.keys()] + investmentPeriods
        stockCapacity = (
            pd.DataFrame(index=years, columns=locations).sort_index().fillna(0)
        )
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
                if capacityMax is not None:
                    if stockCapacity.loc[year, loc] > capacityMax[year][loc]:
                        raise ValueError(
                            "Mismatch between stock capacity (by its "
                            + "commissioning and the technical lifetime) and "
                            + "capacityMax"
                        )
                if capacityFix is not None:
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
                    if capacityDevelopmentDiff[ip - technicalLifetime[loc]] >= 0:
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
    if not isinstance(annuityPerpetuity, bool):
        raise ValueError("annuityPerpetuity must be a bool.")
    if annuityPerpetuity and numberOfInvestmentPeriods == 1:
        raise ValueError(
            "Annuity Perpetuity can only be set for a transformation "
            + "pathway more than one investment period."
        )
    return annuityPerpetuity


def checkAndSetInterestRate(esM, name, interestRate, dimension, elig):
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


def checkInvestmentPeriodsCommodityConversion(commodityConversion, investmentPeriods):
    # If the commodity conversion is depending on commissioning year and investment period,
    # the input shall be a dict with keys of commissioning year and ip and then another dict
    # for commodity conversions
    if any(
        isinstance(commodityConversion[x], dict) for x in commodityConversion.keys()
    ):
        if len(commodityConversion.keys()) != len(investmentPeriods):
            raise ValueError(
                "CommodityConversion is initialized as dict but does not "
                + "contain values for each investment-period"
            )
        if sorted(commodityConversion.keys()) != sorted(investmentPeriods):
            raise ValueError(
                f"CommodityConversion has different ip-names "
                + f"('{commodityConversion.keys()}') than the investment "
                + f"periods of the esM ('{investmentPeriods}')",
            )


def checkConversionDynamicSpecficDesignInputParams(compFancy, esM):
    downTimeMin = compFancy.downTimeMin
    upTimeMin = compFancy.upTimeMin
    numberOfTimeSteps = esM.numberOfTimeSteps
    name = compFancy.name
    rampUpMax = compFancy.rampUpMax
    rampDownMax = compFancy.rampDownMax

    if downTimeMin is not None:
        # Check if values are integers and in the intervall ]0,numberOfTimeSteps].
        if type(downTimeMin) != int:
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
        if type(upTimeMin) != int:
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

    if rampUpMax is not None:
        # Check if values are floats and the intervall ]0,1].
        if type(rampUpMax) != float:
            raise TypeError(
                "rampUpMax for " + name + " needs to be a float in the intervall ]0,1]."
            )
        if rampUpMax <= 0:
            raise ValueError(
                "rampUpMax for " + name + " needs to be a float in the intervall ]0,1]."
            )
        if rampUpMax > 1:
            raise ValueError(
                "rampUpMax for " + name + " needs to be a float in the intervall ]0,1]."
            )

    if rampDownMax is not None:
        # Check if values are floats and the intervall ]0,1].
        if type(rampDownMax) != float:
            raise TypeError(
                "rampDownMax for "
                + name
                + " needs to be a float in the intervall ]0,1]."
            )
        if rampDownMax <= 0:
            raise ValueError(
                "rampDownMax for "
                + name
                + " needs to be a float in the intervall ]0,1]."
            )
        if rampDownMax > 1:
            raise ValueError(
                "rampDownMax for "
                + name
                + " needs to be a float in the intervall ]0,1]."
            )


def setLocationalEligibility(
    esM,
    locationalEligibility,
    capacityMax,
    capacityFix,
    isBuiltFix,
    hasCapacityVariable,
    operationTimeSeries,
    dimension="1dim",
):
    if locationalEligibility is not None:
        # TODO implement checks for the locationalEligiblity, especially for transmission components
        return locationalEligibility
    else:  # if locationalEligibility is not None
        # If the location eligibility is None set it based on other information available
        # if not hasCapacityVariable and all(not isinstance(value,type(None)) for value in operationTimeSeries.values()):
        def defineLocDependencyCapacityBounds(name, capacityBound):
            if capacityBound is None:
                return False
            anyLocIndependent = any(
                x is None or isinstance(x, (int, float)) for x in capacityBound.values()
            )
            anyLocDependent = any(
                x is not None and not isinstance(x, (int, float))
                for x in capacityBound.values()
            )
            if anyLocDependent and anyLocIndependent:
                raise ValueError(
                    f"Please implement {name} either as location dependent or indendent consistent over entire pathway."
                )
            if anyLocDependent:
                return True
            else:
                return False

        isCapacityMaxLocDepending = defineLocDependencyCapacityBounds(
            "capacityMax", capacityMax
        )
        isCapacityFixLocDepending = defineLocDependencyCapacityBounds(
            "capacityFix", capacityFix
        )

        if not hasCapacityVariable and operationTimeSeries is not None:
            if dimension == "1dim":
                data = 0
                # sum values over ips
                for ip in esM.investmentPeriods:
                    data += operationTimeSeries[ip].copy().sum()
                data[data > 0] = 1
                return data
            # Problems here ? Adapt this?
            elif dimension == "2dim":
                # New for perfect foresight
                data = 0
                # sum values over ips
                for ip in esM.investmentPeriods:
                    data += operationTimeSeries[ip].copy().sum()

                data.loc[:] = 1
                locationalEligibility = data
                return locationalEligibility
            else:
                raise ValueError(
                    "The dimension parameter has to be either '1dim' or '2dim' "
                )
        elif (
            (not isCapacityMaxLocDepending)
            and (not isCapacityFixLocDepending)
            and (isBuiltFix is None or isinstance(isBuiltFix, int))
        ):
            # If no information is given, or all information is given as float or integer, all values are set to 1
            if dimension == "1dim":
                return pd.Series([1 for loc in esM.locations], index=esM.locations)
            else:
                keys = {
                    loc1 + "_" + loc2
                    for loc1 in esM.locations
                    for loc2 in esM.locations
                    if loc1 != loc2
                }
                data = pd.Series([1 for key in keys], index=keys)
                data.sort_index(inplace=True)
                return data
        elif isBuiltFix is not None and isinstance(isBuiltFix, pd.Series):
            # If the isBuiltFix is not empty, the eligibility is set based on the fixed capacity
            data = isBuiltFix.copy()
            data[data > 0] = 1
            data.sort_index(inplace=True)
            return data
        else:
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
            if dimension == "1dim":
                regions = esM.locations
            else:
                firstYear = sorted(data.keys())[0]
                regions = data[firstYear].index
            _data = pd.Series(index=sorted(regions), data=0)

            # set location eligibility to 1 if capacity bound exists
            for ip in esM.investmentPeriods:
                loc_idx = data[ip][data[ip] > 0].index
                _data[loc_idx] = 1

            return _data


def checkAndSetInvestmentPeriodTimeSeries(
    esM, name, data, locationalEligibility, dimension="1dim"
):
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
        else:
            raise TypeError(
                f"Parameter of {name} should be a pandas dataframe or a dictionary."
            )
    return parameter


def checkAndSetInvestmentPeriodTimeSeries(
    esM, name, data, locationalEligibility, dimension="1dim"
):
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
        else:
            raise TypeError(
                f"Parameter of {name} should be a pandas dataframe or a dictionary."
            )
    return parameter


def checkAndSetTimeSeries(
    esM, name, operationTimeSeries, locationalEligibility, dimension="1dim"
):
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

        if dimension == "1dim":
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

        elif dimension == "2dim":
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
                        and not loc2 + "_" + loc1 in columns
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

    else:
        return None


def checkDesignVariableModelingParameters(
    esM,
    capacityVariableDomain,
    hasCapacityVariable,
    capacityPerPlantUnit,
    hasIsBuiltBinaryVariable,
    bigM,
):
    if capacityVariableDomain != "continuous" and capacityVariableDomain != "discrete":
        raise ValueError(
            "The capacity variable domain has to be either 'continuous' or 'discrete'."
        )

    if not isinstance(hasIsBuiltBinaryVariable, bool):
        raise TypeError("The hasCapacityVariable variable domain has to be a boolean.")

    isStrictlyPositiveNumber(capacityPerPlantUnit)

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
        if esM.verbose < 2:
            warnings.warn(
                "A declaration of bigM is not necessary if hasIsBuiltBinaryVariable is set to false. "
                "The value of bigM will be ignored in the optimization."
            )


def checkTechnicalLifetime(esM, technicalLifetime, economicLifetime):
    if technicalLifetime is None:
        technicalLifetime = economicLifetime
    return technicalLifetime


def checkEconomicAndTechnicalLifetime(economicLifetime, technicalLifetime):
    if (economicLifetime.sort_index() > technicalLifetime.sort_index()).any():
        raise ValueError("Economic Lifetime must be smaller than technical Lifetime.")


def checkFlooringParameter(floorTechnicalLifetime, technicalLifetime, interval):
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
    if dimension == "1dim":
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
    elif dimension == "2dim":
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

    if dimension == "1dim":
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
):
    # checking function
    def checkPartLoadMin(partLoadMin, bigM, hasCapacityVariable):
        # Check if values are floats and the intervall ]0,1].
        if type(partLoadMin) != float:
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
    return partLoadMin_ip


def checkAndSetInvestmentPeriodCostParameter(
    esM, name, data, dimension, locationalEligibility, years
):
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
    ip_LifeTime = lifetime / esM.investmentPeriodInterval
    return ip_LifeTime


def checkAndSetTimeSeriesConversionFactors(
    esM, commodityConversionFactorsTimeSeries, locationalEligibility
):
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

    else:
        return None


def checkAndSetYearlyLimit(esM, yearlyLimit):
    checkInvestmentPeriodParameters(
        "yearlyLimit", yearlyLimit, esM.investmentPeriodNames
    )
    processedYearlyLimit = {}
    for ip in esM.investmentPeriods:
        _ip = esM.investmentPeriodNames[ip]
        if yearlyLimit is None:
            processedYearlyLimit[ip] = None
        else:
            if isinstance(yearlyLimit, dict):
                _data = yearlyLimit[_ip]
            else:
                _data = yearlyLimit
            if isinstance(_data, int) or isinstance(_data, float):
                if _data < 0:
                    raise ValueError(
                        "Value error in detected.\n "
                        + "Yearly Limit limitations have to be positive."
                    )
                processedYearlyLimit[ip] = _data
            else:
                raise ValueError(
                    "Value error in detected.\n "
                    + "Yearly Limit limitations have to be positive float."
                )
    return processedYearlyLimit


def _addColumnsBalanceLimit(balanceLimit, locations):
    # check and set lower bounds
    if "lowerBound" not in balanceLimit.columns:
        # default as in docs: lowerBound is set to False
        balanceLimit["lowerBound"] = 0
    else:
        if any(x for x in balanceLimit["lowerBound"] if x not in [0, 1]):
            raise ValueError(
                "lowerBound in balanceLimit must be set to either True, False, 0 or 1"
            )
    # check and set locations:
    for loc in list(locations) + ["Total"]:
        if loc not in balanceLimit.columns:
            balanceLimit[loc] = None
    return balanceLimit


def checkAndSetPathwayBalanceLimit(esM, pathwayBalanceLimit, locations):
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
    # balanceLimit has to be DataFrame with locations as columns or Dict per
    # investment periods as keys and described dataframe as values,
    # if valid for whole model

    checkInvestmentPeriodParameters(
        "balanceLimit", balanceLimit, esM.investmentPeriodNames
    )
    processedBalanceLimit = {}

    for ip in esM.investmentPeriods:
        _ip = esM.investmentPeriodNames[ip]

        if isinstance(balanceLimit, dict):
            _balanceLimit = balanceLimit[_ip]
        else:
            _balanceLimit = balanceLimit

        if _balanceLimit is not None:
            if not type(_balanceLimit) == pd.DataFrame:
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
                if dimension == "1dim":
                    parameter[ip] = pd.Series(
                        [float(_data) for loc in esM.locations], index=esM.locations
                    )
                elif dimension == "2dim":
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
    return parameter


def checkClusteringInput(
    numberOfTypicalPeriods, numberOfTimeStepsPerPeriod, totalNumberOfTimeSteps
):
    isStrictlyPositiveInt(numberOfTypicalPeriods), isStrictlyPositiveInt(
        numberOfTimeStepsPerPeriod
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


def setFormattedTimeSeries(timeSeries):
    if timeSeries is None:
        return timeSeries
    else:
        data = timeSeries.copy()
        data["Period"], data["TimeStep"] = 0, data.index
        return data.set_index(["Period", "TimeStep"])


def buildFullTimeSeries(df, periodsOrder, ip, axis=1, esM=None, divide=True):
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
    """
    Functionality for formatting the optimization output. The function is used in the
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
    if varType == "designVariables" and dimension == "1dim":
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
    elif varType == "designVariables" and dimension == "2dim":
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
    elif varType == "operationVariables" and dimension == "1dim":
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
    elif varType == "operationVariables" and dimension == "2dim":
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
    else:
        raise ValueError(
            "The varType parameter has to be either 'designVariables' or 'operationVariables'\n"
            + "and the dimension parameter has to be either '1dim' or '2dim'."
        )


def setOptimalComponentVariables(optVal, varType, compDict):
    if optVal is not None:
        for compName, comp in compDict.items():
            if compName in optVal.index:
                setattr(comp, varType, optVal.loc[compName])
            else:
                setattr(comp, varType, None)


def process2dimCapacityData(esM, name, data, years):
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
    """
    Change format of 2-dimensional data (for transmission components).
    """
    if data is not None and isinstance(data, pd.DataFrame):
        if mapC is None:
            index, data_ = [], []
            for loc1 in data.columns:
                for loc2 in data.index:
                    if discard:
                        # Structure: data[column][row]
                        if data[loc1][loc2] > 0:
                            index.append(loc1 + "_" + loc2), data_.append(
                                data[loc1][loc2]
                            )
                    else:
                        if data[loc1][loc2] >= 0:
                            index.append(loc1 + "_" + loc2), data_.append(
                                data[loc1][loc2]
                            )
            data_ = pd.Series(data_, index=index)
            data_.sort_index(inplace=True)
            return data_
        else:
            data_ = pd.Series(mapC).apply(lambda loc: data[loc[0]][loc[1]])
            data_.sort_index(inplace=True)
            return data_
    elif isinstance(data, float) and locationalEligibility is not None:
        data_ = data * locationalEligibility
        data_.sort_index(inplace=True)
        return data_
    elif isinstance(data, int) and locationalEligibility is not None:
        data_ = data * locationalEligibility
        data_.sort_index(inplace=True)
        return data_
    elif isinstance(data, pd.Series):
        data_ = data.sort_index()
        return data_
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
        mdlListFromExcel += [
            cl
            for cl in mdlListFromModel
            if (cl[0:-5] in sheet and cl not in mdlListFromExcel)
        ]
    if set(mdlListFromModel) != set(mdlListFromExcel):
        raise ValueError("Loaded Output does not match the given energy system model.")


def checkComponentsEquality(esM, file):
    compListFromExcel = []
    compListFromModel = list(esM.componentNames.keys())
    for mdl in esM.componentModelingDict.keys():
        dim = esM.componentModelingDict[mdl].dimension
        readSheet = pd.read_excel(
            file, sheet_name=mdl[0:-5] + "OptSummary_" + dim, index_col=[0, 1, 2, 3]
        )
        compListFromExcel += list(readSheet.index.levels[0])
    if not set(compListFromExcel) <= set(compListFromModel):
        raise ValueError("Loaded Output does not match the given energy system model.")


def checkNumberOfConversionFactors(commods):
    if len(commods) > 2:
        if all([isinstance(value, (int, float)) for value in commods.values()]):
            raise ValueError(
                "Currently commodityConversionFactors are overwritten by commodityConversionFactorsPartLoad."
            )
        else:
            raise ValueError(
                "Currently only two commodities are allowed in conversion processes that use commodityConversionFactorsPartLoad."
            )
    else:
        return True


def checkAndSetTimeHorizon(
    startYear, endYear=None, nbOfSteps=None, nbOfRepresentedYears=None
):
    """
    Check if there are enough input parameters given for defining the time horizon for the myopic approach.
    Calculate the number of optimization steps and the number of represented years per each step if not given.
    """
    if (endYear is not None) & (nbOfSteps is None) & (nbOfRepresentedYears is None):
        # endYear is given; determine the nbOfRepresentedYears
        diff = endYear - startYear

        def biggestDivisor(diff):
            for i in [10, 5, 3, 2, 1]:
                if diff % i == 0:
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
        elif (diff % nbOfSteps == 0) & (nbOfRepresentedYears is not None):
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
        else:
            nbOfSteps = int(diff / nbOfRepresentedYears)
    else:
        nbOfSteps = 1
        nbOfRepresentedYears = 1

    return nbOfSteps, nbOfRepresentedYears


def checkStockYears(
    stockCommissioning, startYear, investmentPeriodInterval, ipTechnicalLifetime
):
    if stockCommissioning is None:
        return [], []
    if not isinstance(stockCommissioning, dict):
        raise ValueError(f"stockCommissioning must be None or a dict")

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
    if stockCommissioning is None:
        return stockCommissioning

    # check type of stockCommissioning
    if not isinstance(stockCommissioning, dict):
        raise TypeError("stockCommissioning must be None or a dict")

    # get regions
    if component.dimension == "1dim":
        regions = esM.locations
    if component.dimension == "2dim":
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
            else:  # if there is only one region, convert into pd.series region:stock
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
        if component.processedCapacityMax is not None:
            if installed_sum > component.processedCapacityMax[0][loc]:
                raise ValueError(
                    f"The stock of {installed_sum} for '{component.name}' in region '{loc}' "
                    + f"exceeds its capacityMax of '{component.processedCapacityMax}' in the first year"
                )
        if component.processedCapacityFix is not None:
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
    processedStockCommissioning = stock_df.T.to_dict(orient="series")
    return processedStockCommissioning


def setStockCapacityStartYear(component, esM, dimension):
    if dimension == "1dim":
        regions = esM.locations
    elif dimension == "2dim":
        regions = [
            loc1 + "_" + loc2
            for loc1 in esM.locations
            for loc2 in esM.locations
            if loc1 != loc2
        ]
    if component.processedStockCommissioning is None:
        return pd.Series(index=regions, data=0)
    else:
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
    """
    Check if the CO2 reduction target is either None or the length of the given list equals the number of optimization steps.
    """
    if CO2ReductionTargets is not None:
        if len(CO2ReductionTargets) != nbOfSteps + 1:
            raise ValueError(
                "CO2ReductionTargets has to be None, or the lenght of the given list must equal the number \
 of optimization steps."
            )


def checkSinkCompCO2toEnvironment(esM, CO2ReductionTargets):
    """
    Check if a sink component object called >CO2 to environment< exists.
    This component is required if CO2 reduction targets are given.
    """

    if CO2ReductionTargets is not None:
        if "CO2 to environment" not in esM.componentNames:
            warnings.warn(
                "CO2 emissions are not considered in the current esM. CO2ReductionTargets will be ignored."
            )
            CO2ReductionTargets = None
            return CO2ReductionTargets
        else:
            return CO2ReductionTargets


def checkSimultaneousChargeDischarge(tsCharge, tsDischarge):
    """
    Check if simultaneous charge and discharge occurs for StorageComponent.
    :param tsCharge: Charge time series of component, which is checked. Can be retrieved from
        chargeOperationVariablesOptimum.loc[compName]. Columns are the time steps, index are the regions.
    :type tsCharge: pd.DataFrame
    :param tsDischarge: Discharge time series of component, which is checked. Can be retrieved from
        dischargeOperationVariablesOptimum.loc[compName]. Columns are the time steps, index are the regions.
    :type tsDischarge: pd.DataFrame

    :return: simultaneousChargeDischarge: Boolean with information if simultaneous charge & discharge happens
    :type simultaneousChargeDischarge: bool
    """
    # Merge Charge and Discharge Series
    ts = pd.concat([tsCharge.T, tsDischarge.T], axis=1)
    # If no simultaneous charge and discharge occurs ts[region][ts[region] > 0] will only return nan values. After
    # dropping them the len() is 0 and the check returns False. This is done for all regions in the list comprehension.
    # If any() region returns True the check returns True.
    simultaneousChargeDischarge = any(
        [
            len(ts[region][ts[region] > 0].dropna()) > 0
            for region in set(ts.columns.values)
        ]
    )
    return simultaneousChargeDischarge


def setNewCO2ReductionTarget(esM, CO2Reference, CO2ReductionTargets, step):
    """
    If CO2ReductionTargets are given, set the new value for each iteration.
    """
    if CO2ReductionTargets is not None:
        setattr(
            esM.componentModelingDict["SourceSinkModel"].componentsDict[
                "CO2 to environment"
            ],
            "yearlyLimit",
            CO2Reference * (1 - CO2ReductionTargets[step] / 100),
        )
        setattr(
            esM.componentModelingDict["SourceSinkModel"].componentsDict[
                "CO2 to environment"
            ],
            "processedYearlyLimit",
            {esM.startYear: CO2Reference * (1 - CO2ReductionTargets[step] / 100)},
        )


def checkParamInput(param):
    if isinstance(param, dict):
        for key, value in param.items():
            if value is None:
                raise ValueError(
                    f"Currently a dict containing None values cannot be passed for '{param}'"
                )


def addEmptyRegions(esM, data):
    """
    If data for a region is missing, fill with 0s.
    """

    esM_locations = esM.locations
    data_locations = data.index
    missing_locations = [loc for loc in esM_locations if loc not in data_locations]

    if type(data) == pd.Series:
        for loc in missing_locations:
            tst = pd.Series([0], index=[loc])
            data = pd.concat([data, tst], axis=0)

    elif type(data) == pd.DataFrame:
        for loc in missing_locations:
            if loc not in data.columns:
                data[loc] = 0

    return data


def annuityPresentValueFactor(esM, compName, loc, years):
    # DE:Rentenbarwertfaktor
    interestRate = esM.getComponent(compName).interestRate[loc]
    if interestRate == 0:
        return years
    else:
        return (((1 + interestRate) ** (years)) - 1) / (
            interestRate * (1 + interestRate) ** (years)
        )


def discountFactor(esM, ip, compName, loc):
    return (
        1
        / (1 + esM.getComponent(compName).interestRate[loc])
        ** (ip * esM.investmentPeriodInterval)
        * (1 + esM.getComponent(compName).interestRate[loc])
    )


def checkAndSetCommodityConversionFactor(comp, esM):
    """Set up the full commodity conversion factor, if necessary depending on
    commissioning year and investment period.
    """
    # Check that type is a dict
    if not isinstance(comp.commodityConversionFactors, dict):
        raise ValueError("commodityConversionFactor must be a dict")

    # 0. get a copy of the commodityConversionFactors
    commodityConversionFactors = comp.commodityConversionFactors.copy()

    # 1. check if the commodity conversion variates
    # a) not at all over transformation pathway
    # b) per investment period -> weather dependency
    # c) per commissioning year and investment period
    dictInDict = any(isinstance(x, dict) for x in commodityConversionFactors.values())
    commisInvestmentPeriodTuple = [
        (x, y)
        for x in (comp.stockYears + esM.investmentPeriodNames)
        for y in esM.investmentPeriodNames
        if y >= x and y < x + comp.technicalLifetime.max()
    ]
    dictKeys = sorted(list(commodityConversionFactors.keys()))

    if not dictInDict:  # commodity conversion is not variated over time
        comp.isIpDepending = False
        comp.isCommisDepending = False
        iterationList = esM.investmentPeriodNames
    elif dictInDict and dictKeys == esM.investmentPeriodNames:
        # commodity conversion is not variated between the investment periods
        comp.isIpDepending = True
        comp.isCommisDepending = False
        iterationList = esM.investmentPeriodNames
    elif dictInDict and dictKeys == sorted(commisInvestmentPeriodTuple):
        # input keys of commodity conversion are variated over investment period and commissioning year

        # check if also data is variated over commissioning year
        isDataVariating = False
        commissioningIndependentCommodityConversionFactor = {}
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
                    if type(ccf[commod]) != type(_baseCommodConvFactor[commod]):
                        raise ValueError(
                            f"Unallowed data type variation for commodity {commod} for yearly dependency."
                        )
                    if isinstance(ccf[commod], (pd.Series, pd.DataFrame)):
                        if not ccf[commod].equals(_baseCommodConvFactor[commod]):
                            isDataVariating = True
                            break
                    else:
                        if not ccf[commod] == _baseCommodConvFactor[commod]:
                            isDataVariating = True
                            break
            # if all are same, save the base commodity conversion of the ip in the new dict
            commissioningIndependentCommodityConversionFactor[
                ip
            ] = _baseCommodConvFactor

        # if data is variating set commis depending true
        if isDataVariating:
            comp.isIpDepending = True
            comp.isCommisDepending = True
            iterationList = commisInvestmentPeriodTuple
        # if data is not variating with the commis, set isCommisDepending to
        # False and update the commodityConversionFactors
        else:
            comp.isIpDepending = True
            comp.isCommisDepending = False
            iterationList = esM.investmentPeriodNames
            commodityConversionFactors = (
                commissioningIndependentCommodityConversionFactor
            )
    else:
        raise ValueError(
            f"Wrong format for commodityConversionFactors for {comp.name}. Please check the init"
        )

    # 2. Check and set up commodity conversion factors
    if comp.isIpDepending or comp.isCommisDepending:
        commodities = []
        for key, _commodConv in commodityConversionFactors.items():
            commodities = _commodConv.keys()
            checkCommodities(esM, set(_commodConv.keys()))
        commodities = list(set(commodities))
    else:
        checkCommodities(esM, set(commodityConversionFactors.keys()))
        commodities = list(set(commodityConversionFactors.keys()))

    # 3. check that type of commodity conversion factors is constant over transformation pathway
    # no switch if one commodity has data in type pd.series or pd.dataframe
    if comp.isIpDepending or comp.isCommisDepending:
        for commod in commodities:
            # if there is one pd series or pd
            if any(
                isinstance(comFac[commod], (pd.Series, pd.DataFrame))
                for comFac in commodityConversionFactors.values()
            ):
                commodTypes = [
                    type(comFac[commod])
                    for comFac in commodityConversionFactors.values()
                ]
                if len(set(commodTypes)) != 1:
                    raise ValueError(
                        f"Unallowed data type variation for commodity {commod} for yearly dependency."
                    )

    # 3. Setup of fullCommodityConversionFactor, processedConversionFactor
    # and preprocessedConversionFactor
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

        for commod in _commodityConversionFactors.keys():
            if isinstance(
                _commodityConversionFactors[commod], (pd.Series, pd.DataFrame)
            ):
                fullCommodityConversionFactor[newKeyName][
                    commod
                ] = checkAndSetTimeSeriesConversionFactors(
                    esM,
                    _commodityConversionFactors[commod],
                    comp.locationalEligibility,
                )
                preprocessedCommodityConversionFactor[newKeyName][
                    commod
                ] = fullCommodityConversionFactor[newKeyName][commod]

            elif isinstance(_commodityConversionFactors[commod], (int, float)):
                # fix values do not need a time-series aggregation and are written
                # directly to processedCommodityConversion
                processedCommodityConversionFactor[newKeyName][
                    commod
                ] = _commodityConversionFactors[commod]
                preprocessedCommodityConversionFactor[newKeyName][
                    commod
                ] = processedCommodityConversionFactor[newKeyName][commod]
            else:
                raise ValueError(
                    f"Data type '{_commodityConversionFactors}' for commodity "
                    + f"{commod} in {_key} not accepted."
                )
    return (
        fullCommodityConversionFactor,
        processedCommodityConversionFactor,
        preprocessedCommodityConversionFactor,
    )


def setParamToNoneIfNoneForAllYears(parameter):
    if parameter is None:
        return parameter
    if all(value is None for value in parameter.values()):
        return None
    else:
        return parameter
