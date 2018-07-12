from abc import ABCMeta, abstractmethod


class Component(metaclass=ABCMeta):
    """
    Doc
    """
    @abstractmethod
    def __init__(self, name):
        pass

    @abstractmethod
    def getDataForTimeSeriesAggregation(self):
        pass

    @abstractmethod
    def setAggregatedTimeSeriesData(self, data):
        pass


class ComponentModeling(metaclass=ABCMeta):
    """
    Doc
    """
    @abstractmethod
    def __init__(self, name):
        pass

    @abstractmethod
    def declareSets(self, esM, pyM):
        pass

    @abstractmethod
    def declareVariables(self, esM, pyM):
        pass

    @abstractmethod
    def declareComponentConstraints(self, esM, pyM):
        pass

    @abstractmethod
    def getSharedPotentialContribution(self, pyM, key, loc):
        pass

    @abstractmethod
    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        pass

    @abstractmethod
    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        pass

    @abstractmethod
    def getObjectiveFunctionContribution(self, esM, pyM):
        pass

    @abstractmethod
    def setOptimalValues(self, esM, pyM):
        pass