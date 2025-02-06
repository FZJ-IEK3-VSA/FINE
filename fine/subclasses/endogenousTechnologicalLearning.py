from fine import utils, utilsETL
import math
import pyomo.environ as pyomo
from pyomo.core import Piecewise
import pandas as pd

pwlf = False

class EndogenousTechnologicalLearningModul:

    def __init__(
            self,
            comp,
            esM,
            learningRate,
            initCapacity,
            maxCapacity,
            initCost=None,
            noSegments=None,
    ):
        #utilsETL.checkEsmLocations(esM)
        self.comp = comp
        self.learningRate = learningRate
        self.learningIndex = utilsETL.checkAndSetLearningIndex(learningRate)
        self.initCost = utilsETL.checkAndSetInitCost(initCost, comp)
        self.initCapacity, self.maxCapacity = utilsETL.checkCapacities(initCapacity, maxCapacity, comp)
        utilsETL.checkStock(comp, self.initCapacity)

        if noSegments is None:
            self.noSegments = 4
        else:
            utils.isStrictlyPositiveInt(int(noSegments))
            self.noSegments = int(noSegments)

        self.linEtlParameter = self.linearizeLearningCurve()

        self.commisYears = esM.investmentPeriods
        self.commisYears = comp.processedStockYears + esM.investmentPeriods

    def getTotalCost(self, capacity):
        totalCost = (((self.initCapacity * self.initCost) / (1 - self.learningIndex)) *
                     (capacity / self.initCapacity) ** (1 - self.learningIndex))
        return totalCost

    def linearizeLearningCurve(self):
        linEtlParameter = pd.DataFrame(index=range(self.noSegments + 1),
                                       columns=['experience', 'totalCost', 'slope', 'interception'])

        linEtlParameter['totalCost'].loc[0] = self.getTotalCost(self.initCapacity)
        linEtlParameter['totalCost'].loc[self.noSegments] = self.getTotalCost(self.maxCapacity)
        totalCostDiff = linEtlParameter['totalCost'].loc[self.noSegments] - linEtlParameter['totalCost'].loc[0]

        for segment in range(1, self.noSegments):
            linEtlParameter['totalCost'].loc[segment] = (
                    linEtlParameter['totalCost'].loc[segment - 1] + (2 ** (segment - self.noSegments - 1))
                    * (totalCostDiff / (1 - 0.5 ** self.noSegments))
            )

        linEtlParameter['experience'] = (((1 - self.learningIndex)
                                          / (self.initCost * self.initCapacity ** self.learningIndex)
                                          * linEtlParameter['totalCost']) ** (1 / (1 - self.learningIndex)))

        linEtlParameter['slope'] = linEtlParameter.diff()['totalCost'] / linEtlParameter.diff()['experience']
        linEtlParameter['interception'] = (linEtlParameter['totalCost']
                                           - linEtlParameter['slope'] * linEtlParameter['experience'])

        return linEtlParameter



class EndogenousTechnologicalLearningModel:

    def __init__(self):
        self.abbrvName = "etl"
        self.modulsDict = {}

    def declareSets(self, esM, pyM):
        self.declareEtlDesignSet(pyM, esM)
        if not pwlf:
            self.declareEtlDesignSegmentSet(pyM, esM)

    def declareEtlDesignSet(self, pyM, esM):
        def declareDesignSet(pyM):
            return (
                (modulName, ip)
                for modulName, modul in self.modulsDict.items()
                for ip in esM.investmentPeriods
            )

        pyM.etlDesignSet = pyomo.Set(dimen=2, initialize=declareDesignSet)

    def declareEtlDesignSegmentSet(self, pyM, esM):
        def declareDesignSegmentSet(pyM):
            return (
                (modulName, ip, segment)
                for modulName, modul in self.modulsDict.items()
                for ip in esM.investmentPeriods
                for segment in range(modul.noSegments)
            )

        pyM.etlDesignSegmentSet = pyomo.Set(dimen=3, initialize=declareDesignSegmentSet)

    def declareVariables(self, esM, pyM):
        if not pwlf:
            self.declareBinaryEtlVar(esM, pyM)
            self.declareSegmentCapacityEtlVar(esM, pyM)

    def declareBinaryEtlVar(self, esM, pyM):
        """
        :param esM:
        :param pyM:
        :return:
        """
        pyM.binaryEtlVar = pyomo.Var(
            pyM.etlDesignSegmentSet,
            domain=pyomo.Binary
        )

    def declareSegmentCapacityEtlVar(self, esM, pyM):
        pyM.segmentCapacityEtlVar = pyomo.Var(
            pyM.etlDesignSegmentSet,
            domain=pyomo.NonNegativeReals,
        )

    def declareComponentConstraints(self, esM, pyM):
        if pwlf:
            self.declarePwlfPyomo(esM, pyM)
        else:
            self.declareBinaryEtlConstr(pyM)
            self.declareSegmentCapacityEtlConstr(pyM)
            self.declareCapacityCommissioningEtlConstr(esM, pyM)


    def declareBinaryEtlConstr(self, pyM):

        def binaryEtlConstr(pyM, modulName, ip, segment):
            return (
                    sum(
                        pyM.binaryEtlVar[modulName, ip, segment]
                        for segment in range(self.modulsDict[modulName].noSegments)
                    ) == 1
            )

        pyM.ConstrBinaryEtl = pyomo.Constraint(
            pyM.etlDesignSegmentSet,
            rule=binaryEtlConstr
        )

    def declareSegmentCapacityEtlConstr(self, pyM):

        def lowerSegmentCapacityEtlConstr(pyM, modulName, ip, segment):
            modul = self.modulsDict[modulName]
            maxCapacityPerSegment = modul.linEtlParameter['experience']
            lowerCapacityBound = maxCapacityPerSegment.loc[segment]
            binVar = pyM.binaryEtlVar[modulName, ip, segment]
            capSegmentVar = pyM.segmentCapacityEtlVar[modulName, ip, segment]

            return lowerCapacityBound * binVar <= capSegmentVar

        def upperSegmentCapacityEtlConstr(pyM, modulName, ip, segment):
            modul = self.modulsDict[modulName]
            maxCapacityPerSegment = modul.linEtlParameter['experience']
            upperCapacityBound = maxCapacityPerSegment.loc[segment + 1]
            binVar = pyM.binaryEtlVar[modulName, ip, segment]
            capSegmentVar = pyM.segmentCapacityEtlVar[modulName, ip, segment]

            return capSegmentVar <= upperCapacityBound * binVar

        pyM.ConstrLowerSegmentCapacityEtl = pyomo.Constraint(
            pyM.etlDesignSegmentSet,
            rule=lowerSegmentCapacityEtlConstr
        )

        pyM.ConstrUpperSegmentCapacityEtl = pyomo.Constraint(
            pyM.etlDesignSegmentSet,
            rule=upperSegmentCapacityEtlConstr
        )


    def declarePwlfPyomo(self, esM, pyM):
        """
        https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html#piecewise-linear-expressions
        """
        pyM.totalCost = pyomo.Var(
            pyM.etlDesignSet,
            domain=pyomo.NonNegativeReals,
        )

        def totalCapacityBounds(pyM, modulName, ip):
            return (0, self.modulsDict[modulName].maxCapacity)

        pyM.totalCapacity = pyomo.Var(
            pyM.etlDesignSet,
            domain=pyomo.NonNegativeReals,
            bounds=totalCapacityBounds,
        )

        def fixTotalCapacity(pyM, modulName, ip):
            modul = self.modulsDict[modulName]
            compClass = modul.comp.modelingClass().abbrvName
            commVar = getattr(pyM, "commis_" + compClass)
            commVarSum = sum(
                commVar[loc, modulName, _ip]
                for _ip in range(ip + 1)
                for loc in esM.locations
            )

            return pyM.totalCapacity[modulName, ip] == commVarSum + modul.initCapacity

        pyM.fixTotalCapacity = pyomo.Constraint(
            pyM.etlDesignSet,
            rule=fixTotalCapacity
        )

        xdata = {
            idx: list(self.modulsDict[idx[0]].linEtlParameter['experience'])
            for idx in pyM.etlDesignSet
        }

        ydata = {
            idx: list(self.modulsDict[idx[0]].linEtlParameter['totalCost'])
            for idx in pyM.etlDesignSet
        }

        pyM.pwlf = Piecewise(
            pyM.etlDesignSet,
            pyM.totalCost,
            pyM.totalCapacity,
            pw_pts=xdata,
            pw_constr_type='EQ',
            f_rule=ydata,
            pw_repn='SOS2'
        )


    def declareCapacityCommissioningEtlConstr(self, esM, pyM):

        def capacityCommissioningEtlConstr(pyM, modulName, ip):
            modul = self.modulsDict[modulName]
            compClass = modul.comp.modelingClass().abbrvName
            commVar = getattr(pyM, "commis_" + compClass)
            commVarSum = sum(
                commVar[loc, modulName, _ip]
                for _ip in range(ip + 1)
                for loc in esM.locations
            )
            capSegmentVarSum = sum(
                pyM.segmentCapacityEtlVar[modulName, ip, segment]
                for segment in range(modul.noSegments)
            )

            return capSegmentVarSum == commVarSum + modul.initCapacity


        pyM.ConstrCapacityCommissioningEtl = pyomo.Constraint(
            pyM.etlDesignSet,
            rule=capacityCommissioningEtlConstr
        )

    def getObjectiveFunctionContribution(self, esM, pyM):
        return self.getEconomicsEtl(esM, pyM)

    def getEconomicsEtl(
            self,
            esM,
            pyM,
            getOptValue=False,
            getOptValueCostType='TAC',
    ):
        componentYears = {
            modulName: esM.getComponentAttribute(modulName, "processedStockYears")
                       + esM.investmentPeriods
            for modulName in self.modulsDict.keys()
        }

        costContribution = {
            modulName: {
                (y, i): 0
                for y in modul.commisYears
                for i in esM.investmentPeriods
            }
            for modulName, modul in self.modulsDict.items()
        }

        loc = list(esM.locations)[0]

        for modulName, modul in self.modulsDict.items():

            ipEconomicLifetime = getattr(
                esM.getComponent(modulName), "ipEconomicLifetime"
            )[loc]
            ipTechnicalLifetime = getattr(
                esM.getComponent(modulName), "ipTechnicalLifetime"
            )[loc]

            (fullCostIntervals, costInLastEconInterval,
             costInLastTechInterval) = utils.getParametersForUnevenLifetimes(
                modulName, loc, 'ipEconomicLifetime', esM)

            for commisYear in modul.commisYears:
                annuity = self.getAnnuityEtl(pyM, modulName, commisYear, modul.commisYears, getOptValue)

                for i in range(commisYear, commisYear + fullCostIntervals):
                    costContribution[modulName][
                        (commisYear, i)
                    ] = annuity * utils.annuityPresentValueFactor(
                        esM, modulName, loc, esM.investmentPeriodInterval
                    )

                if costInLastEconInterval:
                    partlyCostInLastEconomicInterval = (
                                                               ipEconomicLifetime % 1
                                                       ) * esM.investmentPeriodInterval
                    costContribution[modulName][
                        (commisYear, commisYear + fullCostIntervals)
                    ] = annuity * utils.annuityPresentValueFactor(
                        esM, modulName, loc, partlyCostInLastEconomicInterval
                    )

                if (
                        costInLastTechInterval
                        and ipTechnicalLifetime % 1 != 0
                ):
                    partlyCostInLastTechnicalInterval = (
                                                                1 - (ipTechnicalLifetime % 1)
                                                        ) * esM.investmentPeriodInterval
                    if commisYear + math.ceil(ipTechnicalLifetime) - 1 in [
                        k[1] for k in costContribution[modulName].keys()
                    ]:
                        costContribution[modulName][
                            (
                                commisYear,
                                commisYear + math.ceil(ipTechnicalLifetime) - 1,
                            )
                        ] = costContribution[modulName][
                                (
                                    commisYear,
                                    commisYear + math.ceil(ipTechnicalLifetime) - 1,
                                )
                            ] + annuity * (
                                    utils.annuityPresentValueFactor(
                                        esM,
                                        modulName,
                                        loc,
                                        partlyCostInLastTechnicalInterval,
                                    )
                                    / (1 + esM.getComponent(modulName).interestRate[loc])
                                    ** (
                                            esM.investmentPeriodInterval
                                            - partlyCostInLastTechnicalInterval
                                    )
                            )

        if getOptValue:
            cost_results = {ip: pd.DataFrame() for ip in esM.investmentPeriods}
            for modulName in self.modulsDict.keys():
                for ip in esM.investmentPeriods:
                    cContrSum = sum(
                        [
                            costContribution[modulName].get((y, ip), 0)
                            for y in componentYears[modulName]
                        ]
                    )
                    if getOptValueCostType == "NPV":
                        cost_results[ip].loc[
                            modulName, loc
                        ] = cContrSum * utils.discountFactor(esM, ip, modulName, loc)
                    elif getOptValueCostType == "TAC":
                        cost_results[ip].loc[
                            modulName, loc
                        ] = cContrSum / utils.annuityPresentValueFactor(
                            esM, modulName, loc, esM.investmentPeriodInterval
                        )
            return cost_results
        else:
            if esM.annuityPerpetuity:
                for modulName in costContribution.keys(): # noqa: PLC0206
                    for y in componentYears[modulName]:
                        costContribution[modulName][
                            (y, esM.investmentPeriods[-1])
                        ] = costContribution[modulName][
                                (y, esM.investmentPeriods[-1])
                            ] / (
                                    utils.annuityPresentValueFactor(
                                        esM, modulName, loc, esM.investmentPeriodInterval
                                    )
                                    * esM.getComponent(modulName).interestRate[loc]
                            )
            return sum(
                sum(
                    [
                        costContribution[modulName].get((y, ip), 0)
                        for y in componentYears[modulName]
                    ]
                )
                * utils.discountFactor(esM, ip, modulName, loc)
                for modulName in self.modulsDict.keys()
                for ip in esM.investmentPeriods
            )

    def getAnnuityEtl(self, pyM, modulName, commisYear, commisYears, getOptValues=False):
        def getIpTotalCost(ip):
            if ip == commisYears[0] - 1:
                totalCost = modul.getTotalCost(
                    modul.initCapacity - modul.comp.stockCapacityStartYear.sum()
                )
            elif ip < 0:
                unbuildStockUntilIp = sum(
                    modul.comp.processedStockCommissioning[i].sum()
                    for i in range(ip + 1, 0)
                )
                totalCost = modul.getTotalCost(
                    modul.initCapacity
                    - unbuildStockUntilIp
                )
            elif pwlf:
                if not getOptValues:
                    totalCost = pyM.totalCost[modulName, ip]
                else:
                    totalCost = pyM.totalCost[modulName, ip].value
            elif not getOptValues:
                totalCost = sum(
                    modul.linEtlParameter['interception'].loc[segment + 1]
                    * pyM.binaryEtlVar[modulName, ip, segment]
                    + modul.linEtlParameter['slope'].loc[segment + 1]
                    * pyM.segmentCapacityEtlVar[modulName, ip, segment]
                    for segment in range(modul.noSegments)
                )
            else:
                totalCost = sum(
                    modul.linEtlParameter['interception'].loc[segment + 1]
                    * pyM.binaryEtlVar[modulName, ip, segment].value
                    + modul.linEtlParameter['slope'].loc[segment + 1]
                    * pyM.segmentCapacityEtlVar[modulName, ip, segment].value
                    for segment in range(modul.noSegments)
                )
            return totalCost

        modul = self.modulsDict[modulName]
        totalCostCommisYear = getIpTotalCost(commisYear)
        totalCostPreCommisYear = getIpTotalCost(commisYear - 1)

        return (totalCostCommisYear - totalCostPreCommisYear) / modul.comp.CCF[commisYear].mean()

    def setOptimalValues(self, esM, pyM):
        loc = list(esM.locations)[0]

        props = [
            "TAC_ETL",
            "NPVcontribution_ETL",
            "knowledgeStock_ETL"
        ]
        units = [
            "[" + esM.costUnit + "/a]",
            "[" + esM.costUnit + "]",
            "[-]",
        ]
        tuples = [
            (modulName, prop, unit)
            for modulName in self.modulsDict.keys()
            for prop, unit in zip(props, units)
        ]

        unitDict = {
            'conv': ('physicalUnit', ''),
            'srcSnk': ('commodityUnit', ''),
            'stor': ('commodityUnit', '*h'),
            'trans': ('commodityUnit', ''),
        }

        tuples = list(
            map(
                lambda x: (
                    x[0],
                    x[1],
                    "[" + getattr(
                        self.modulsDict[x[0]].comp,
                        unitDict[self.modulsDict[x[0]].comp.modelingClass().abbrvName][0]
                    ) + unitDict[self.modulsDict[x[0]].comp.modelingClass().abbrvName][1] + "]",
                )
                if x[1] == "knowledgeStock_ETL"
                else x,
                tuples,
            )
        )
        mIndex = pd.MultiIndex.from_tuples(
            tuples, names=["Component", "Property", "Unit"]
        )

        optSummaryEtl = {
            ip: pd.DataFrame(index=mIndex, columns=list(esM.locations)).sort_index()
            for ip in esM.investmentPeriodNames
        }

        tac = self.getEconomicsEtl(esM, pyM, getOptValue=True, getOptValueCostType='TAC')
        npv = self.getEconomicsEtl(esM, pyM, getOptValue=True, getOptValueCostType='NPV')

        for ip in esM.investmentPeriods:
            for modulName, modul in self.modulsDict.items():
                optSummaryEtl[esM.investmentPeriodNames[ip]].loc[
                    (modulName, 'TAC_ETL', '[' + esM.costUnit + '/a]'),
                    loc
                ] = tac[ip][loc].loc[modulName]

                optSummaryEtl[esM.investmentPeriodNames[ip]].loc[
                    (modulName, 'NPVcontribution_ETL', '[' + esM.costUnit + ']'),
                    loc
                ] = npv[ip][loc].loc[modulName]
                if pwlf:
                    knowledgeStock = pyM.totalCapacity[modulName, ip].value
                else:
                    knowledgeStock = sum(
                        pyM.segmentCapacityEtlVar[modulName, ip, segment]._value
                        for segment in range(modul.noSegments)
                    )
                optSummaryEtl[esM.investmentPeriodNames[ip]].loc[
                    (
                        modulName,
                        'knowledgeStock_ETL',
                        "[" + getattr(
                            modul.comp,
                            unitDict[modul.comp.modelingClass().abbrvName][0]
                        ) + unitDict[modul.comp.modelingClass().abbrvName][1] + "]"
                    ),
                    loc
                ] = knowledgeStock


        for model in esM.componentModelingDict.values():
            optSummary = model._optSummary
            for ipName in esM.investmentPeriodNames:
                etlComps = [comp for comp in model.componentsDict.keys() if comp in self.modulsDict.keys()]
                optSummary[ipName] = pd.concat(
                    [
                        optSummary[ipName],
                        optSummaryEtl[ipName].loc[etlComps, :, :]
                    ],
                    axis=0
                ).sort_index()
            model.optSummary = optSummary[esM.startYear]
