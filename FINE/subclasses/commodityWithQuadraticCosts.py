from FINE.sourceSink import Source, SourceSinkModel
from FINE import utils
import pyomo.environ as pyomo
import pandas as pd
from FINE.component import ComponentModel


class CommodityQC(Source):
    """
    A LinearOptimalPowerFlow component shows the behavior of a Transmission component but additionally models a
    linearized power flow.
    """
    def __init__(self, esM, name, commodity, operationRateMax=None, operationRateFix=None, tsaWeight=1,
                 commodityLimitID=None, yearlyLimit=None, locationalEligibility=None, commodityCostA=0,
                 commodityCostB=0, commodityCostC=0, bigMOperation=None):
        """
        Constructor for creating an Conversion class instance.
        The CommodityQC component specific input arguments are described below. The general component
        input arguments are described in the Source class.
        """

        Source.__init__(self, esM, name, commodity, hasCapacityVariable=False,
                         capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                         hasIsBuiltBinaryVariable=False, bigM=None,
                         operationRateMax=operationRateMax, operationRateFix=operationRateFix, tsaWeight=tsaWeight,
                         commodityLimitID=commodityLimitID, yearlyLimit=yearlyLimit,
                        locationalEligibility=locationalEligibility, capacityMin=None,
                         capacityMax=None, sharedPotentialID=None, capacityFix=None, isBuiltFix=None,
                         investPerCapacity=0, investIfBuilt=0, opexPerOperation=0, commodityCost=0,
                         commodityRevenue=0, opexPerCapacity=0, opexIfBuilt=0, interestRate=0.08, economicLifetime=10)

        self.commodityCostA = utils.checkAndSetCostParameter(esM, name, commodityCostA, '1dim',
                                                             locationalEligibility)
        self.commodityCostB = utils.checkAndSetCostParameter(esM, name, commodityCostB, '1dim',
                                                             locationalEligibility)
        self.commodityCostC = utils.checkAndSetCostParameter(esM, name, commodityCostC, '1dim',
                                                             locationalEligibility)

        self.bigMOperation = bigMOperation
        self.modelingClass = CommodityQCModel

    def addToEnergySystemModel(self, esM):
        super().addToEnergySystemModel(esM)


class CommodityQCModel(SourceSinkModel):
    """ Doc """
    def __init__(self):
        self.abbrvName = 'cqc'
        self.dimension = '1dim'
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.binaryCostVariables = None
        self.operationVariablesOptimum, self.phaseAngleVariablesOptimum = None, None
        self.optSummary = None

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareBinCostVarSet(self, pyM):
        compDict, abbrvName = self.componentsDict, self.abbrvName
        # Set for operation variables
        def declareBinCostVarSet(pyM):
            return ((loc, compName) for compName, comp in compDict.items()
                    for loc in comp.locationalEligibility.index
                    if comp.locationalEligibility[loc] == 1 and comp.commodityCostC[loc] != 0)
        setattr(pyM, 'binCostVarSet_' + abbrvName, pyomo.Set(dimen=2, initialize=declareBinCostVarSet))

    def declareSets(self, esM, pyM):
        super().declareSets(esM, pyM)
        self.declareBinCostVarSet(pyM)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareBinaryCostVars(self, pyM):
        """ Declares binary variables [-] indicating if a component is considered at a location or not [-] """
        abbrvName = self.abbrvName
        setattr(pyM, 'binCostVar_' + abbrvName, pyomo.Var(getattr(pyM, 'binCostVarSet_' + abbrvName),
                                                          domain=pyomo.Binary))

    def declareVariables(self, esM, pyM):
        """ Declares design and operation variables """

        super().declareVariables(esM, pyM)
        self.declareBinaryCostVars(pyM)

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def bigMOpCost(self, pyM):
        """ Enforce the consideration of the binary design variables of a component """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        var, binCostVar = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'binCostVar_' + abbrvName)
        binCostVarSet = getattr(pyM, 'binCostVarSet_' + abbrvName)

        def bigMOpCost(pyM, loc, compName):
            return (sum(var[loc, compName, p, t] for p, t in pyM.timeSet) <=
                    binCostVar[loc, compName] * compDict[compName].bigMOperation)
        setattr(pyM, 'ConstrBigMOpCost_' + abbrvName, pyomo.Constraint(binCostVarSet, rule=bigMOpCost))

    def declareComponentConstraints(self, esM, pyM):

        super().declareComponentConstraints(esM, pyM)
        self.bigMOpCost(pyM)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        """ Gets contributions to shared location potential """
        pass

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return super().hasOpVariablesForLocationCommodity(esM, loc, commod)

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        """ Gets contribution to a commodity balance """
        return super().getCommodityBalanceContribution(pyM, commod, loc, p, t)

    def getObjectiveFunctionContribution(self, esM, pyM):
        """ Gets contribution to the objective function """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        indices = getattr(pyM, 'operationVarDict' + '_' + abbrvName).items()
        binCostVarSet = getattr(pyM, 'binCostVarSet_' + abbrvName)
        var, binCostVar = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'binCostVar_' + abbrvName)

        costAB = sum(sum((compDict[compName].commodityCostA[loc] * var[loc, compName, p, t] * var[loc, compName, p, t] +
                          compDict[compName].commodityCostB[loc] * var[loc, compName, p, t]) * esM.periodOccurrences[p]
                         for p, t in pyM.timeSet)/esM.numberOfYears
                     for loc, compNames in indices for compName in compNames)

        costC = sum(compDict[compName].commodityCostC[loc] * binCostVar[loc, compName]/esM.numberOfYears
                    for loc, compName in binCostVarSet)

        return costAB + costC

    def setOptimalValues(self, esM, pyM):
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, binCostVar = getattr(pyM, 'op_' + abbrvName), getattr(pyM, 'binCostVar_' + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = ComponentModel.setOptimalValues(self, esM, pyM, esM.locations, 'commodityUnit')

        # Set optimal operation variables and append optimization summary
        optVal = utils.formatOptimizationOutput(opVar.get_values(), 'operationVariables', '1dim', esM.periodsOrder)
        self.operationVariablesOptimum = optVal

        props = ['operation', 'opexOp', 'commodCosts']
        units = ['[-]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]].commodityUnit + '*h/a]')
                          if x[1] == 'operation' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM.locations)).sort_index()

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name].opexPerOperation[op.index], axis=1)
            cCostAB = opSum.apply(lambda op: compDict[op.name].commodityCostA[op.index] * op * op +
                                           compDict[op.name].commodityCostB[op.index] * op, axis=1)
            cRevenue = opSum.apply(lambda op: op * compDict[op.name].commodityRevenue[op.index], axis=1)
            optSummary.loc[[(ix, 'operation', '[' + compDict[ix].commodityUnit + '*h/a]') for ix in opSum.index],
                            opSum.columns] = opSum.values/esM.numberOfYears
            optSummary.loc[[(ix, 'opexOp', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                ox.values/esM.numberOfYears
            optSummary.loc[[(ix, 'commodCosts', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                (cCostAB-cRevenue).values/esM.numberOfYears

        binCostVar = utils.formatOptimizationOutput(binCostVar.get_values(), 'designVariables', '1dim')
        self.binaryCostVariables = binCostVar

        if binCostVar is not None:
            cCostC = binCostVar.apply(lambda bin: compDict[bin.name].commodityCostC[bin.index] * bin, axis=1)
            optSummary.loc[[(ix, 'commodCosts', '[' + esM.costUnit + '/a]') for ix in cCostC.index], cCostC.columns] = \
                optSummary.loc[[(ix, 'commodCosts', '[' + esM.costUnit + '/a]') for ix in cCostC.index],
                               cCostC.columns] + cCostC.values

        optSummary = optSummary.append(optSummaryBasic).sort_index()
        # Summarize all contributions to the total annual cost
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'TAC') |
                           (optSummary.index.get_level_values(1) == 'opexOp') |
                           (optSummary.index.get_level_values(1) == 'commodCosts')].groupby(level=0).sum().values

        self.optSummary = optSummary

    def getOptimalValues(self, name='all'):
        super().getOptimalValues(name)

    def getLocEconomicsTD(self, pyM, esM, factorNames, varName, loc, compName, getOptValue=False):
        var = getattr(pyM, varName + '_' + self.abbrvName)
        factors = [getattr(self.componentsDict[compName], factorName)[loc] for factorName in factorNames]
        factor = 1.
        for factor_ in factors:
            factor *= factor_
        if not getOptValue:
            return (factor * sum(var[loc, compName, p, t] * esM.periodOccurrences[p]
                                 for p, t in pyM.timeSet)/esM.numberOfYears)
        else:
            return (factor * sum(var[loc, compName, p, t].value * esM.periodOccurrences[p]
                                 for p, t in pyM.timeSet)/esM.numberOfYears)

