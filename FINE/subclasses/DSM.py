from FINE.sourceSink import Sink, Source, SourceSinkModel
from FINE import utils
import FINE as fn
import pyomo.environ as pyomo
import pandas as pd

class DemandSideManagement(Sink):
    """
    TODO
    A DemandSideManagement component
    """
    def __init__(self, esM, name, commodity, hasCapacityVariable, tDown, tUp,
        operationRateFix, **kwargs):
        """
        Constructor for creating an DemandSideManagement class instance.
        TODO

        **Required arguments:**

        """
        self.tUp = tUp
        self.tDown = tDown
        self.tDelta = tDown+tUp+1

        operationRateFix = pd.concat([operationRateFix.iloc[-tUp:], operationRateFix.iloc[:-tUp]]).reset_index(drop=True)
        Sink.__init__(self, esM, name, commodity, hasCapacityVariable,
            operationRateFix=operationRateFix, **kwargs)

        self.modelingClass = DSMModel

        for i in range(self.tDelta):
            SOCmax = pd.concat([operationRateFix[operationRateFix.index % self.tDelta == i]]*self.tDelta).\
                sort_index().reset_index(drop=True)
            SOCmax = pd.concat([SOCmax.iloc[tDown+tUp-i:], SOCmax.iloc[:tDown+tUp-i]]).reset_index(drop=True)

            dischargeFix = operationRateFix.copy()
            dischargeFix[dischargeFix.index % self.tDelta != i] = 0

            esM.add(fn.StorageExt(esM, name + '_' + str(i), commodity, stateOfChargeOpRateMax=SOCmax,
                dischargeOpRateFix=dischargeFix, hasCapacityVariable=False))


class DSMModel(SourceSinkModel):
    """
    A StorageExtModel class instance will be instantly created if a StorageExt class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the StorageExt class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The StorageExtModel class inherits from the StorageModel class.
    """

    def __init__(self):
        """ Constructor for creating a DSMModel class instance """
        self.abbrvName = 'dsm'
        self.dimension = '1dim'
        self.componentsDict = {}
        self.capacityVariablesOptimum, self.isBuiltVariablesOptimum = None, None
        self.operationVariablesOptimum = None
        self.optSummary = None


    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def setOptimalValues(self, esM, pyM):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pym: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pym: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, 'op_' + abbrvName)

        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super(SourceSinkModel, self).setOptimalValues(esM, pyM, esM.locations, 'commodityUnit')

        # Set optimal operation variables and append optimization summary
        chargeOp = getattr(pyM, 'chargeOp_storExt')
        optVal = utils.formatOptimizationOutput(chargeOp.get_values(), 'operationVariables', '1dim', esM.periodsOrder)

        def groupStor(x):
            ix = optVal.loc[x].name
            for compName, comp in self.componentsDict.items():
                if ix[0] in [compName + '_' + str(i) for i in range(comp.tUp+comp.tDown+1)]:
                    return (compName, ix[1])

        optVal = optVal.groupby(lambda x: groupStor(x)).sum()
        optVal.index = pd.MultiIndex.from_tuples(optVal.index)

        self.operationVariablesOptimum = optVal

        props = ['operation', 'opexOp', 'commodCosts', 'commodRevenues']
        units = ['[-]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]', '[' + esM.costUnit + '/a]']
        tuples = [(compName, prop, unit) for compName in compDict.keys() for prop, unit in zip(props, units)]
        tuples = list(map(lambda x: (x[0], x[1], '[' + compDict[x[0]].commodityUnit + '*h/a]')
                          if x[1] == 'operation' else x, tuples))
        mIndex = pd.MultiIndex.from_tuples(tuples, names=['Component', 'Property', 'Unit'])
        optSummary = pd.DataFrame(index=mIndex, columns=sorted(esM.locations)).sort_index()

        if optVal is not None:
            opSum = optVal.sum(axis=1).unstack(-1)
            ox = opSum.apply(lambda op: op * compDict[op.name].opexPerOperation[op.index], axis=1)
            cCost = opSum.apply(lambda op: op * compDict[op.name].commodityCost[op.index], axis=1)
            cRevenue = opSum.apply(lambda op: op * compDict[op.name].commodityRevenue[op.index], axis=1)
            
            optSummary.loc[[(ix, 'operation', '[' + compDict[ix].commodityUnit + '*h/a]') for ix in opSum.index],
                            opSum.columns] = opSum.values/esM.numberOfYears
            optSummary.loc[[(ix, 'opexOp', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                ox.values/esM.numberOfYears
            
            # get empty datframe for resulting time dependent (TD) cost sum
            cRevenueTD = pd.DataFrame(0., index = list(compDict.keys()), columns = opSum.columns)
            cCostTD = pd.DataFrame(0., index = list(compDict.keys()), columns = opSum.columns)

            for compName in compDict.keys():
                if not compDict[compName].commodityCostTimeSeries is None:
                    # in case of time series aggregation rearange clustered cost time series
                    calcCostTD = utils.buildFullTimeSeries(compDict[compName].commodityCostTimeSeries, 
                                                           esM.periodsOrder, axis=0)
                    # multiply with operation values to get the total cost
                    cCostTD.loc[compName,:] = optVal.xs(compName, level=0).T.mul(calcCostTD).sum(axis=0)

                if not compDict[compName].commodityRevenueTimeSeries is None:
                    # in case of time series aggregation rearange clustered revenue time series
                    calcRevenueTD = utils.buildFullTimeSeries(compDict[compName].commodityRevenueTimeSeries,
                                                              esM.periodsOrder, axis=0)
                    # multiply with operation values to get the total revenue
                    cRevenueTD.loc[compName,:] = optVal.xs(compName, level=0).T.mul(calcRevenueTD).sum(axis=0)
                        
            optSummary.loc[[(ix, 'commodCosts', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                (cCostTD.values + cCost.values)/esM.numberOfYears

            optSummary.loc[[(ix, 'commodRevenues', '[' + esM.costUnit + '/a]') for ix in ox.index], ox.columns] = \
                (cRevenueTD.values + cRevenue.values)/esM.numberOfYears
        
        # get discounted investment cost as total annual cost (TAC)
        optSummary = optSummary.append(optSummaryBasic).sort_index()

        # add operation specific contributions to the total annual cost (TAC) and substract revenues
        optSummary.loc[optSummary.index.get_level_values(1) == 'TAC'] = \
            optSummary.loc[(optSummary.index.get_level_values(1) == 'TAC') |
                           (optSummary.index.get_level_values(1) == 'opexOp') |
                           (optSummary.index.get_level_values(1) == 'commodCosts')].groupby(level=0).sum().values \
            - optSummary.loc[(optSummary.index.get_level_values(1) == 'commodRevenues')].groupby(level=0).sum().values

        self.optSummary = optSummary