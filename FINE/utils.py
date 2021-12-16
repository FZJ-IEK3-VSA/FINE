import warnings

import pandas as pd
import numpy as np
import pwlf
from GPyOpt.methods import BayesianOptimization

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
    costUnit,
    lengthUnit,
    balanceLimit,
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

    # The numberOfTimeSteps and the hoursPerTimeStep have to be strictly positive numbers
    isStrictlyPositiveInt(numberOfTimeSteps), isStrictlyPositiveNumber(hoursPerTimeStep)

    # The costUnit and lengthUnit input parameter have to be strings
    isString(costUnit), isString(lengthUnit)

    # balanceLimit has to be DataFrame with locations as columns or Series, if valid for whole model
    if balanceLimit is not None:
        if (
            not type(balanceLimit) == pd.DataFrame
            and not type(balanceLimit) == pd.Series
        ):
            raise TypeError(
                "The balanceLimit input argument has to be a pandas.DataFrame or a pd.Series."
            )
        if (
            type(balanceLimit) == pd.DataFrame
            and set(balanceLimit.columns) != locations
        ):
            raise ValueError(
                "Location indices in the balanceLimit do not match the input locations.\n"
                + "balanceLimit columns: "
                + str(set(balanceLimit.columns))
                + "\n"
                + "Input regions: "
                + str(locations)
            )


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


def checkRegionalColumnTitles(esM, data):
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
        raise ValueError(
            "Location indices do not match the one of the specified energy system model.\n"
            + "Data columns: "
            + str(set(data.columns))
            + "\n"
            + "Energy system model regions: "
            + str(esM.locations)
        )
    # Sort data according to _locationsOrdered, if not already sorted
    elif not np.array_equal(data.columns, esM._locationsOrdered):
        data.sort_index(inplace=True, axis=1)
    return data


def checkRegionalIndex(esM, data):
    """
    Necessary if the data rows represent the location-dependent data:
    Check if the row-indices match the location indices of the energy system model.
    """
    if set(data.index) != esM.locations:
        raise ValueError(
            "Location indices do not match the one of the specified energy system model.\n"
            + "Data indices: "
            + str(set(data.index))
            + "\n"
            + "Energy system model regions: "
            + str(esM.locations)
        )
    # Sort data according to _locationsOrdered, if not already sorted
    elif not np.array_equal(data.index, esM._locationsOrdered):
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
    elif not np.array_equal(data.index, locationalEligibility.index):
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


def checkCommodityConversionFactorsPartLoad(commodityConversionFactorsPartLoad):
    """
    Check if one of the commodity conversion factors equals 1 and another is either a lambda function or a set of data points.
    Additionally check if the conversion factor that depicts part load behavior
        (1) covers part loads from 0 to 1 and
        (2) includes only conversion factors greater than 0 in the relevant part load range.
    """
    partLoadCommodPresent = False
    nonPartLoadCommodPresent = False

    for conversionFactor in commodityConversionFactorsPartLoad:
        if isinstance(conversionFactor, pd.DataFrame):
            checkDataFrameConversionFactor(conversionFactor)
            partLoadCommodPresent = True
        elif callable(conversionFactor):
            checkCallableConversionFactor(conversionFactor)
            partLoadCommodPresent = True
        elif conversionFactor == 1 or conversionFactor == -1:
            nonPartLoadCommodPresent = True

    if nonPartLoadCommodPresent == False:
        raise TypeError("One conversion factor needs to be either 1 or -1.")
    if partLoadCommodPresent == False:
        raise TypeError(
            "One conversion factor needs to be either a callable function or a list of two-dimensional data points."
        )


def checkAndCorrectDiscretizedPartloads(discretizedPartLoad):
    """Check if the discretized points are >=0 and <=100%"""

    for commod, conversionFactor in discretizedPartLoad.items():
        # ySegments
        if not np.all(
            conversionFactor["ySegments"] == conversionFactor["ySegments"][0]
        ):
            if any(conversionFactor["ySegments"] < 0):
                if sum(conversionFactor["ySegments"] < 0) > 1:
                    raise ValueError(
                        "There is at least two partLoad efficiency values that are < 0. Please check your partLoadEfficiency data or function visually."
                    )
                else:
                    # First element
                    if np.where(conversionFactor["ySegments"] < 0)[0][0] == 0:
                        # Correct efficiency < 0 for index = 0 -> construct line
                        coefficients = np.polyfit(
                            conversionFactor["xSegments"][0:2],
                            conversionFactor["ySegments"][0:2],
                            1,
                        )
                        discretizedPartLoad[commod]["ySegments"][0] = 0
                        discretizedPartLoad[commod]["xSegments"][0] = (
                            -coefficients[1] / coefficients[0]
                        )

                    # Last element
                    elif (
                        np.where(conversionFactor["ySegments"] < 0)[0][0]
                        == len(conversionFactor["ySegments"]) - 1
                    ):
                        # Correct efficiency < for index = 0 -> construct line
                        coefficients = np.polyfit(
                            conversionFactor["xSegments"][-2:],
                            conversionFactor["ySegments"][-2:],
                            1,
                        )
                        discretizedPartLoad[commod]["ySegments"][-1] = 0
                        discretizedPartLoad[commod]["xSegments"][-1] = (
                            -coefficients[1] / coefficients[0]
                        )
                    else:
                        raise ValueError(
                            "PartLoad efficiency value < 0 detected where slope cannot be constructed. Please check your partLoadEfficiency data or function visually."
                        )
        # xSegments
        if any(conversionFactor["xSegments"] < 0):
            if sum(conversionFactor["xSegments"] < 0) > 1:
                raise ValueError(
                    "There is at least two partLoad efficiency values that are < 0. Please check your partLoadEfficiency data or function visually."
                )
            else:
                # First element
                if np.where(conversionFactor["xSegments"] < 0)[0][0] == 0:
                    coefficients = np.polyfit(
                        conversionFactor["xSegments"][0:2],
                        conversionFactor["ySegments"][0:2],
                        1,
                    )
                    discretizedPartLoad[commod]["xSegments"][0] = 0
                    discretizedPartLoad[commod]["ySegments"][0] = coefficients[1]
                else:
                    raise ValueError(
                        "PartLoad efficiency value < 0 detected where slope cannot be constructed. Please check your partLoadEfficiency data or function visually."
                    )
        if any(conversionFactor["xSegments"] > 1):
            if sum(conversionFactor["xSegments"] > 1) > 1:
                raise ValueError(
                    "There is at least two partLoad efficiency values that are > 1. Please check your partLoadEfficiency data or function visually."
                )
            else:
                # Last element
                if (
                    np.where(conversionFactor["xSegments"] > 1)[0][0]
                    == len(conversionFactor["xSegments"]) - 1
                ):
                    coefficients = np.polyfit(
                        conversionFactor["xSegments"][-2:],
                        conversionFactor["ySegments"][-2:],
                        1,
                    )
                    discretizedPartLoad[commod]["xSegments"][0] = 1
                    discretizedPartLoad[commod]["ySegments"][0] = (
                        coefficients[0] + coefficients[1]
                    )
                else:
                    raise ValueError(
                        "PartLoad efficiency value > 1 detected where slope cannot be constructed. Please check your partLoadEfficiency data or function visually."
                    )

    return discretizedPartLoad


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


def getCapitalChargeFactor(interestRate, economicLifetime):
    """Compute and return capital charge factor (inverse of annuity factor)."""
    CCF = 1 / interestRate - 1 / (
        pow(1 + interestRate, economicLifetime) * interestRate
    )
    CCF = CCF.fillna(economicLifetime)
    return CCF


def castToSeries(data, esM):
    if data is None:
        return None
    elif isinstance(data, pd.Series):
        return data
    isPositiveNumber(data)
    return pd.Series(data, index=list(esM.locations))


def getQPbound(QPcostScale, capacityMax, capacityMin):
    """Compute and return lower and upper capacity bounds."""
    index = QPcostScale.index
    QPbound = pd.Series([np.inf] * len(index), index)

    if capacityMin is not None and capacityMax is not None:
        minS = pd.Series(capacityMin.isna(), index)
        maxS = pd.Series(capacityMax.isna(), index)
        for x in index:
            if not minS.loc[x] and not maxS.loc[x]:
                QPbound.loc[x] = capacityMax.loc[x] - capacityMin.loc[x]
    return QPbound


def getQPcostDev(QPcostScale):
    QPcostDev = 1 - QPcostScale
    return QPcostDev


def checkLocationSpecficDesignInputParams(comp, esM):
    if len(esM.locations) == 1:
        comp.capacityMin = castToSeries(comp.capacityMin, esM)
        comp.capacityFix = castToSeries(comp.capacityFix, esM)
        comp.capacityMax = castToSeries(comp.capacityMax, esM)
        comp.locationalEligibility = castToSeries(comp.locationalEligibility, esM)
        comp.isBuiltFix = castToSeries(comp.isBuiltFix, esM)
        comp.QPcostScale = castToSeries(comp.QPcostScale, esM)

    capacityMin, capacityFix, capacityMax, QPcostScale = (
        comp.capacityMin,
        comp.capacityFix,
        comp.capacityMax,
        comp.QPcostScale,
    )
    locationalEligibility, isBuiltFix = comp.locationalEligibility, comp.isBuiltFix
    hasCapacityVariable, hasIsBuiltBinaryVariable = (
        comp.hasCapacityVariable,
        comp.hasIsBuiltBinaryVariable,
    )
    sharedPotentialID = comp.sharedPotentialID
    partLoadMin = comp.partLoadMin
    name = comp.name
    bigM = comp.bigM
    hasCapacityVariable = comp.hasCapacityVariable

    for data in [
        capacityMin,
        capacityFix,
        capacityMax,
        QPcostScale,
        locationalEligibility,
        isBuiltFix,
    ]:
        if data is not None:
            if comp.dimension == "1dim":
                if not isinstance(data, pd.Series):
                    raise TypeError("Input data has to be a pandas Series")
                data = checkRegionalIndex(esM, data)
            elif comp.dimension == "2dim":
                if not isinstance(data, pd.Series):
                    raise TypeError("Input data has to be a pandas DataFrame")
                data = checkConnectionIndex(data, comp.locationalEligibility)
            else:
                raise ValueError(
                    "The dimension parameter has to be either '1dim' or '2dim' "
                )

    if capacityMin is not None and (capacityMin < 0).any():
        raise ValueError("capacityMin values smaller than 0 were detected.")

    if capacityFix is not None and (capacityFix < 0).any():
        raise ValueError("capacityFix values smaller than 0 were detected.")

    if capacityMax is not None and (capacityMax < 0).any():
        raise ValueError("capacityMax values smaller than 0 were detected.")

    if (
        capacityMin is not None or capacityMax is not None or capacityFix is not None
    ) and not hasCapacityVariable:
        raise ValueError(
            "Capacity bounds are given but hasDesignDimensionVar was set to False."
        )

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

    if capacityMin is not None and capacityMax is not None:
        # Test that capacityMin and capacityMax has the same index for comparing.
        # If capacityMin is missing for some locations, it´s set to 0.
        if set(capacityMin.index).issubset(capacityMax.index):
            capacityMin = capacityMin.reindex(capacityMax.index).fillna(0)
        if (capacityMin > capacityMax).any():
            raise ValueError("capacityMin values > capacityMax values detected.")

    if capacityFix is not None and capacityMax is not None:
        if (capacityFix > capacityMax).any():
            raise ValueError("capacityFix values > capacityMax values detected.")

    if capacityFix is not None and capacityMin is not None:
        if (capacityFix < capacityMin).any():
            raise ValueError("capacityFix values < capacityMax values detected.")

    if capacityMax is None or capacityMin is None:
        if (QPcostScale > 0).any():
            raise ValueError(
                "QPcostScale is given but lower or upper capacity bounds are not specified."
            )

    if (QPcostScale < 0).any() or (QPcostScale > 1).any():
        raise ValueError('QPcostScale must ba a number between "0" and "1".')

    if locationalEligibility is not None:
        # Check if values are either one or zero
        if ((locationalEligibility != 0) & (locationalEligibility != 1)).any():
            raise ValueError(
                "The locationalEligibility entries have to be either 0 or 1."
            )
        # Check if given capacities indicate the same eligibility
        if capacityFix is not None:
            data = capacityFix.copy()
            data[data > 0] = 1
            if (data != locationalEligibility).any():
                raise ValueError(
                    "The locationalEligibility and capacityFix parameters indicate different eligibilities."
                )
        if capacityMax is not None:
            data = capacityMax.copy()
            data[data > 0] = 1
            if (data != locationalEligibility).any():
                raise ValueError(
                    "The locationalEligibility and capacityMax parameters indicate different eligibilities."
                )
        if capacityMin is not None:
            data = capacityMin.copy()
            data[data > 0] = 1
            if (data > locationalEligibility).any():
                raise ValueError(
                    "The locationalEligibility and capacityMin parameters indicate different eligibilities."
                )
        if isBuiltFix is not None:
            if (isBuiltFix != locationalEligibility).any():
                raise ValueError(
                    "The locationalEligibility and isBuiltFix parameters indicate different"
                    + "eligibilities."
                )

    if isBuiltFix is not None:
        # Check if values are either one or zero
        if ((isBuiltFix != 0) & (isBuiltFix != 1)).any():
            raise ValueError("The isBuiltFix entries have to be either 0 or 1.")
        # Check if given capacities indicate the same design decisions
        if capacityFix is not None:
            data = capacityFix.copy()
            data[data > 0] = 1
            if (data > isBuiltFix).any():
                raise ValueError(
                    "The isBuiltFix and capacityFix parameters indicate different design decisions."
                )
        if capacityMax is not None:
            data = capacityMax.copy()
            data[data > 0] = 1
            if (data > isBuiltFix).any():
                if esM.verbose < 2:
                    warnings.warn(
                        "The isBuiltFix and capacityMax parameters indicate different design options."
                    )
        if capacityMin is not None:
            data = capacityMin.copy()
            data[data > 0] = 1
            if (data > isBuiltFix).any():
                raise ValueError(
                    "The isBuiltFix and capacityMin parameters indicate different design decisions."
                )

    if partLoadMin is not None:
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
        return locationalEligibility
    else:
        # If the location eligibility is None set it based on other information available
        if not hasCapacityVariable and operationTimeSeries is not None:
            if dimension == "1dim":
                data = operationTimeSeries.copy().sum()
                data[data > 0] = 1
                return data
            elif dimension == "2dim":
                data = operationTimeSeries.copy().sum()
                data.loc[:] = 1
                return data
            else:
                raise ValueError(
                    "The dimension parameter has to be either '1dim' or '2dim' "
                )
        elif (
            (
                capacityFix is None
                or isinstance(capacityFix, float)
                or isinstance(capacityFix, int)
            )
            and (
                capacityMax is None
                or isinstance(capacityMax, float)
                or isinstance(capacityMax, int)
            )
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
            data = capacityFix.copy() if capacityFix is not None else capacityMax.copy()
            data[data > 0] = 1
            return data


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
            checkRegionalColumnTitles(esM, operationTimeSeries)

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
        data = checkRegionalIndex(esM, data)
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

        checkRegionalColumnTitles(esM, fullCommodityConversionFactorsTimeSeries)

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


def checkAndSetFullLoadHoursParameter(
    esM, name, data, dimension, locationalEligibility
):
    if data is None:
        return None
    else:
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
                    + "Full load hours limitations have to be a number or a pandas Series."
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
                    + "Full load hours limitations have to be a number or a pandas Series."
                )
        else:
            raise ValueError(
                "The dimension parameter has to be either '1dim' or '2dim' "
            )

        if dimension == "1dim":
            if isinstance(data, int) or isinstance(data, float):
                if data < 0:
                    raise ValueError(
                        "Value error in "
                        + name
                        + " detected.\n Full load hours limitations have to be positive."
                    )
                return pd.Series(
                    [float(data) for loc in esM.locations], index=esM.locations
                )
            data = checkRegionalIndex(esM, data)
        else:
            if isinstance(data, int) or isinstance(data, float):
                if data < 0:
                    raise ValueError(
                        "Value error in "
                        + name
                        + " detected.\n Full load hours limitations have to be positive."
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


def buildFullTimeSeries(df, periodsOrder, axis=1, esM=None, divide=True):
    # If segmentation is chosen, the segments of each period need to be unravelled to the original number of
    # time steps first
    if esM is not None and esM.segmentation:
        dataAllPeriods = []
        for p in esM.typicalPeriods:
            # Repeat each segment in each period as often as time steps are represented by the corresponding
            # segment
            repList = esM.timeStepsPerSegment.loc[p, :].tolist()
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
    data, varType, dimension, periodsOrder=None, compDict=None, esM=None
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
        df = pd.DataFrame(data, index=[0]).T.swaplevel(i=0, j=-2).sort_index()
        # Unstack the time steps (convert to a two dimensional DataFrame with the time indices being the columns)
        df = df.unstack(level=-1)
        # Get rid of the unnecessary 0 level
        df.columns = df.columns.droplevel()
        # Re-engineer full time series by using Pandas' concat method (only one loop if time series aggregation was not
        # used)
        return buildFullTimeSeries(df, periodsOrder, esM=esM)
    elif varType == "operationVariables" and dimension == "2dim":
        # Convert dictionary to DataFrame, transpose, put the period column first while keeping the order of the
        # regions and sort the index
        # Results in a one dimensional DataFrame
        df = pd.DataFrame(data, index=[0]).T
        indexNew = []
        for tup in df.index.tolist():
            loc1, loc2 = compDict[tup[1]]._mapC[tup[0]]
            indexNew.append((loc1, loc2, tup[1], tup[2], tup[3]))
        df.index = pd.MultiIndex.from_tuples(indexNew)
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
        return buildFullTimeSeries(df, periodsOrder, esM=esM)
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


def pieceWiseLinearization(functionOrRaw, xLowerBound, xUpperBound, nSegments):
    """
    Determine xSegments, ySegments.
    If nSegments is not specified by the user it is either set (e.g. nSegments=5) or nSegements is determined by
    a bayesian optimization algorithm.
    """
    if callable(functionOrRaw):
        nPointsForInputData = 1000
        x = np.linspace(xLowerBound, xUpperBound, nPointsForInputData)
        y = np.array([functionOrRaw(x_i) for x_i in x])
    else:
        x = np.array(functionOrRaw.iloc[:, 0])
        y = np.array(functionOrRaw.iloc[:, 1])
        if not 0.0 in x:
            xMinDefined = np.amin(x)
            xMaxDefined = np.amax(x)
            lenIntervalDefined = xMaxDefined - xMinDefined
            lenIntervalUndefined = xMinDefined
            nPointsUndefined = lenIntervalUndefined * (x.size / lenIntervalDefined)
            xMinIndex = np.argmin(x)
            for i in range(int(nPointsUndefined)):
                x = np.append(x, [i / int(nPointsUndefined + 1) * lenIntervalUndefined])
                y = np.append(y, y[xMinIndex])
        if not 1.0 in x:
            xMinDefined = np.amin(x)
            xMaxDefined = np.amax(x)
            lenIntervalDefined = xMaxDefined - xMinDefined
            lenIntervalUndefined = 1.0 - xMaxDefined
            nPointsUndefined = lenIntervalUndefined * (x.size / lenIntervalDefined)
            xMaxIndex = np.argmax(x)
            for i in range(int(nPointsUndefined)):
                x = np.append(
                    x,
                    [
                        xMaxDefined
                        + (i + 1) / int(nPointsUndefined) * lenIntervalUndefined
                    ],
                )
                y = np.append(y, y[xMaxIndex])

    myPwlf = pwlf.PiecewiseLinFit(x, y)

    if nSegments == None:
        nSegments = 5
    elif nSegments == "optimizeSegmentNumbers":

        bounds = [{"name": "var_1", "type": "discrete", "domain": np.arange(2, 8)}]

        # Define objective function to get optimal number of line segments
        def my_obj(_x):
            # The penalty parameter l is set arbitrarily.
            # It depends upon the noise in your data and the value of your sum of square of residuals
            l = y.mean() * 0.001

            f = np.zeros(_x.shape[0])

            for i, j in enumerate(_x):
                myPwlf.fit(j[0])
                f[i] = myPwlf.ssr + (l * j[0])
            return f

        myBopt = BayesianOptimization(
            my_obj,
            domain=bounds,
            model_type="GP",
            initial_design_numdata=10,
            initial_design_type="latin",
            exact_feval=True,
            verbosity=True,
            verbosity_model=False,
        )

        max_iter = 30

        # Perform bayesian optimization to find the optimum number of line segments
        myBopt.run_optimization(max_iter=max_iter, verbosity=True)
        nSegments = int(myBopt.x_opt)

    xSegments = myPwlf.fit(nSegments)

    # Get the y segments
    ySegments = myPwlf.predict(xSegments)

    # Calcualte the R^2 value
    Rsquared = myPwlf.r_squared()

    # Calculate the piecewise R^2 value
    R2values = np.zeros(nSegments)
    for i in range(nSegments):
        # Segregate the data based on break point locations
        xMin = myPwlf.fit_breaks[i]
        xMax = myPwlf.fit_breaks[i + 1]
        xTemp = myPwlf.x_data
        yTemp = myPwlf.y_data
        indTemp = np.where(xTemp >= xMin)
        xTemp = myPwlf.x_data[indTemp]
        yTemp = myPwlf.y_data[indTemp]
        indTemp = np.where(xTemp <= xMax)
        xTemp = xTemp[indTemp]
        yTemp = yTemp[indTemp]

        # Predict for the new data
        yHatTemp = myPwlf.predict(xTemp)

        # Calcualte ssr
        e = yHatTemp - yTemp
        ssr = np.dot(e, e)

        # Calculate sst
        yBar = np.ones(yTemp.size) * np.mean(yTemp)
        ydiff = yTemp - yBar
        sst = np.dot(ydiff, ydiff)

        R2values[i] = 1.0 - (ssr / sst)

    return {
        "xSegments": xSegments,
        "ySegments": ySegments,
        "nSegments": nSegments,
        "Rsquared": Rsquared,
        "R2values": R2values,
    }


def getDiscretizedPartLoad(commodityConversionFactorsPartLoad, nSegments):
    """Preprocess the conversion factors passed by the user"""
    discretizedPartLoad = {
        commod: None for commod in commodityConversionFactorsPartLoad.keys()
    }
    functionOrRawCommod = None
    nonFunctionOrRawCommod = None
    for commod, conversionFactor in commodityConversionFactorsPartLoad.items():
        if (isinstance(conversionFactor, pd.DataFrame)) or (callable(conversionFactor)):
            discretizedPartLoad[commod] = pieceWiseLinearization(
                functionOrRaw=conversionFactor,
                xLowerBound=0,
                xUpperBound=1,
                nSegments=nSegments,
            )
            functionOrRawCommod = commod
            nSegments = discretizedPartLoad[commod]["nSegments"]
        elif conversionFactor == 1 or conversionFactor == -1:
            discretizedPartLoad[commod] = {
                "xSegments": None,
                "ySegments": None,
                "nSegments": None,
                "Rsquared": 1.0,
                "R2values": 1.0,
            }
            nonFunctionOrRawCommod = commod
    discretizedPartLoad[nonFunctionOrRawCommod]["xSegments"] = discretizedPartLoad[
        functionOrRawCommod
    ]["xSegments"]
    discretizedPartLoad[nonFunctionOrRawCommod]["ySegments"] = np.array(
        [commodityConversionFactorsPartLoad[nonFunctionOrRawCommod]] * (nSegments + 1)
    )
    discretizedPartLoad[nonFunctionOrRawCommod]["nSegments"] = nSegments
    checkAndCorrectDiscretizedPartloads(discretizedPartLoad)
    return discretizedPartLoad, nSegments


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
