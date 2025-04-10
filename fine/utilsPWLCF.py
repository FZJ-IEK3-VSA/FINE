import warnings
import numpy as np
from fine import utils

def checkValuesIncreasing(data):
    return all(data[i] <= data[i+1] for i in range(len(data)-1))

def interpolateFromDataframe(df, xValue, xName, yName):
    df_sorted = df.sort_values(xName)
    df_interp = df_sorted.set_index(xName)
    df_interp = df_interp.reindex(df_interp.index.union([xValue]))
    df_interp = df_interp.sort_index().interpolate(method='linear')
    return df_interp.loc[xValue, yName]

def checkAndSetEosParameters(comp, eosParameters):
    #check that capacity/totalInvest/totalFixOpex grid points are increasing:
    if not checkValuesIncreasing(eosParameters["capacity"]):
        raise ValueError(f"Capacity grid points for economies of scale do not increase for component {comp}.")
    if not checkValuesIncreasing(eosParameters["totalInvest"]):
        raise ValueError(f"totalInvest grid points for economies of scale do not increase with increasing capacity for component {comp}.")
    if not checkValuesIncreasing(eosParameters["totalOpex"]):
        raise ValueError(f"totalInvest grid points for economies of scale do not increase with increasing capacity for component {comp}.")

    #Check capacity variable:
    if not comp.hasCapacityVariable:
        raise ValueError(f"EOS Component ({comp}) must have Capacity Variable")

    return eosParameters

def checkInvestmentPeriods(esM):
    if len(esM.numberOfInvestmentPeriods) != 1: 
        raise NotImplementedError(
            "Economies of Scale are currently only "
            "implemented for single investment period energy system models"
        )

def checkEsmLocations(esM):
    if len(esM.locations) != 1:
        raise NotImplementedError(
            "Piecewise Linear Cost Functions are currently only "
            "implemented for single node energy system models"
        )


def checkStock(comp, initCapacity):
    # TODO: adapt for multi regional
    if comp.stockCapacityStartYear.sum() > initCapacity:
        raise ValueError(
            f"Stock of component {comp.name} must be smaller than "
            "the specified initial pwlcf capacity."
        )


def checkAndSetLearningIndex(learningRate):
    if 1 > learningRate > 0:
        learningIndex = np.log2(1 / (1 - learningRate))
    else:
        raise ValueError("Learning Rate does not match the required format")

    return learningIndex


def checkAndSetInitCost(initCost, comp):
    if initCost is None:
        initCost = comp.processedInvestPerCapacity[0].values[0]
        warnings.warn(
            f"The 'initCost' parameter for {comp.name} is missing. Therefore the investPerCapacity "
            f"specified for the startYear ({initCost}) was chosen."
        )
    else:
        utils.isStrictlyPositiveNumber(initCost)

    return initCost


def checkCapacitiesEtl(initCapacity, maxCapacity, comp):
    if not comp.hasCapacityVariable:
        raise ValueError("ETL Component must have Capacity Variable")

    # check initial Capacity
    utils.isStrictlyPositiveNumber(initCapacity)
    if comp.processedStockCommissioning is not None:
        stock = sum(
            commis.sum() for commis in comp.processedStockCommissioning.values()
        )
        if initCapacity < stock:
            raise ValueError(
                "Initial Capacity of ETL Component must be greater than specified stock."
            )

    # check maximal Capacity
    utils.isStrictlyPositiveNumber(maxCapacity)
    if maxCapacity <= initCapacity:
        raise ValueError("Maximal Capacity must be greater than initial Capacity")

    return initCapacity, maxCapacity

