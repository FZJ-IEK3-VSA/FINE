import warnings
import numpy as np
from fine import utils

def checkValuesIncreasing(data):
    return all(data[i] <= data[i+1] for i in range(len(data)-1))


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
    #Preprocess slope and interception:
    eosParameters["slopeTotalInvest"] = np.nan
    eosParameters["slopeTotalOpex"] = np.nan
    eosParameters["interceptionTotalInvest"] = np.nan
    eosParameters["interceptionTotalOpex"]=np.nan
    for idx in range(len(eosParameters["capacity"])-1):
        eosParameters["slopeTotalInvest"].iloc[idx] = ((eosParameters["totalInvest"].iloc[idx+1] - eosParameters["totalInvest"].iloc[idx])
        / (eosParameters["capacity"].iloc[idx+1] - eosParameters["capacity"].iloc[idx])
        )
        eosParameters["slopeTotalOpex"].iloc[idx] = ((eosParameters["totalOpex"].iloc[idx+1] - eosParameters["totalOpex"].iloc[idx])
        / (eosParameters["capacity"].iloc[idx+1] - eosParameters["capacity"].iloc[idx])
        )
        eosParameters["interceptionTotalInvest"].iloc[idx] = eosParameters["totalInvest"].iloc[idx+1] - eosParameters["slopeTotalInvest"].iloc[idx] * eosParameters["capacity"].iloc[idx+1]
        eosParameters["interceptionTotalOpex"].iloc[idx] = eosParameters["totalOpex"].iloc[idx+1] - eosParameters["slopeTotalOpex"].iloc[idx] * eosParameters["capacity"].iloc[idx+1]

    return eosParameters

def checkInvestmentPeriods(esM):
    if esM.numberOfInvestmentPeriods != 1:
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

def checkEtlCompParams(comp):
    if comp.economicLifetime.nunique() > 1:
        raise ValueError(
            f"Economic Lifetime of ETL Component {comp.name} must be constant for all investment periods."
        )
    if comp.technicalLifetime.nunique() > 1:
        raise ValueError(
            f"Technical Lifetime of ETL Component {comp.name} must be constant for all investment periods."
        )
    if comp.interestRate.nunique() > 1:
        raise ValueError(
            f"Interest Rate of ETL Component {comp.name} must be constant for all investment periods."
        )

def checkStock(comp, initCapacity):
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
