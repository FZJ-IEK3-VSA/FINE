import warnings
import numpy as np
from fine import utils


def checkEsmLocations(esM):
    if len(esM.locations) != 1:
        raise NotImplementedError(
            "Endogenous Technological Learning is currently only "
            "implemented for single node energy system models"
        )

def checkStock(comp, initCapacity):
    #TODO: adapt for multi regional
    if comp.stockCapacityStartYear.sum() > initCapacity:
        raise ValueError(
            f"Stock of component {comp.name} must be smaller than "
            "the specified initial etl capacity."
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
        warnings.warn(f"The 'initCost' parameter for {comp.name} is missing. Therefore the investPerCapacity "
                      f"specified for the startYear ({initCost}) was chosen.")
    else:
        utils.isStrictlyPositiveNumber(initCost)

    return initCost


def checkCapacities(initCapacity, maxCapacity, comp):

    if not comp.hasCapacityVariable:
        raise ValueError("ETL Component must have Capacity Variable")

    # check initial Capacity
    utils.isStrictlyPositiveNumber(initCapacity)
    if comp.processedStockCommissioning is not None:
        stock = sum(commis.sum() for commis in comp.processedStockCommissioning.values())
        if initCapacity < stock:
            raise ValueError("Initial Capacity of ETL Component must be greater than specified stock.")

    # check maximal Capacity
    utils.isStrictlyPositiveNumber(maxCapacity)
    if maxCapacity <= initCapacity:
        raise ValueError("Maximal Capacity must be greater than initial Capacity")

    return initCapacity, maxCapacity


def getTotalCost(etlModul, capacity):
    totalCost = (((etlModul.initCapacity * etlModul.initCost) / (1 - etlModul.learningIndex)) *
                 (capacity / etlModul.initCapacity) ** (1 - etlModul.learningIndex))
    return totalCost


# def linearizeLearningCurve(etlModul):
#     totalCostsPerSegment =