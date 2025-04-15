from fine import utils, utilsPWLCF
import math
import pyomo.environ as pyomo
from pyomo.core import Piecewise
import pandas as pd

pyomo_pwlf = False


class PiecewiseLinearCostFunctionModul:
    def __init__(
        self,
        comp,
        esM,
        etlParameters = None,
        eosParameters = None,
    ):
        self.comp = comp

        if etlParameters and  eosParameters is not None:
            raise NotImplementedError(f"Specifying both, endogenous technology learning (etl) and economies of scale (eos) is not valid. Check component: {self.comp}.")
        if etlParameters:
            self.pwlcf_type = 'etl'
            self.learningRate = etlParameters['learningRate']
            self.learningIndex = utilsPWLCF.checkAndSetLearningIndex(etlParameters['learningRate'])
            self.initCost = utilsPWLCF.checkAndSetInitCost(etlParameters['initCost'], comp)
            self.initCapacity, self.maxCapacity = utilsPWLCF.checkCapacitiesEtl(
                etlParameters['initCapacity'], etlParameters['maxCapacity'], comp
            )
            utilsPWLCF.checkStock(comp, self.initCapacity)

            if etlParameters['noSegments'] is None:
                self.noSegments = 4
            else:
                utils.isStrictlyPositiveInt(int(etlParameters['noSegments']))
                self.noSegments = int(etlParameters['noSegments'])

                self.linEtlParameter = self.linearizeLearningCurveEtl()

        elif eosParameters is not None:
            if pyomo_pwlf:
                raise NotImplementedError("SOS2 Constraints via pyomo.pwlf currently not implemented for economies of scale.")
            self.pwlcf_type = 'eos'
            utilsPWLCF.checkInvestmentPeriods(esM)
            self.eosParameters = utilsPWLCF.checkAndSetEosParameters(comp, eosParameters)
            self.noSegments = len(eosParameters["capacity"]) - 1

        self.commisYears = comp.processedStockYears + esM.investmentPeriods

    def getTotalCostEtl(self, capacity):
        return ((self.initCapacity * self.initCost) / (1 - self.learningIndex)) * (
            capacity / self.initCapacity
        ) ** (1 - self.learningIndex)

    def linearizeLearningCurveEtl(self):
        linEtlParameter = pd.DataFrame(
            index=range(self.noSegments + 1),
            columns=["experience", "totalCost", "slope", "interception"],
        )

        linEtlParameter["totalCost"].loc[0] = self.getTotalCostEtl(self.initCapacity)
        linEtlParameter["totalCost"].loc[self.noSegments] = self.getTotalCostEtl(
            self.maxCapacity
        )
        totalCostDiff = (
            linEtlParameter["totalCost"].loc[self.noSegments]
            - linEtlParameter["totalCost"].loc[0]
        )

        for segment in range(1, self.noSegments):
            linEtlParameter["totalCost"].loc[segment] = linEtlParameter[
                "totalCost"
            ].loc[segment - 1] + (2 ** (segment - self.noSegments - 1)) * (
                totalCostDiff / (1 - 0.5**self.noSegments)
            )

        linEtlParameter["experience"] = (
            (1 - self.learningIndex)
            / (self.initCost * self.initCapacity**self.learningIndex)
            * linEtlParameter["totalCost"]
        ) ** (1 / (1 - self.learningIndex))

        linEtlParameter["slope"] = (
            linEtlParameter.diff()["totalCost"] / linEtlParameter.diff()["experience"]
        )
        linEtlParameter["interception"] = (
            linEtlParameter["totalCost"]
            - linEtlParameter["slope"] * linEtlParameter["experience"]
        )

        return linEtlParameter


class PiecewiseLinearCostFunctionModel:
    def __init__(self):
        self.abbrvName = "pwlcf"
        self.modulsDict = {}

    def declareSets(self, esM, pyM):
        self.declarePwlcfDesignSet(pyM, esM)
        if not pyomo_pwlf:
            self.declarePwlcfDesignSegmentSet(pyM, esM)

    def declarePwlcfDesignSet(self, pyM, esM):
        def declareDesignSet(pyM):
            return (
                (modulName, ip)
                for modulName, modul in self.modulsDict.items()
                for ip in esM.investmentPeriods
            )

        pyM.pwlcfDesignSet = pyomo.Set(dimen=2, initialize=declareDesignSet)

    def declarePwlcfDesignSegmentSet(self, pyM, esM):
        def declareDesignSegmentSet(pyM):
            return (
                (modulName, ip, segment)
                for modulName, modul in self.modulsDict.items()
                for ip in esM.investmentPeriods
                for segment in range(modul.noSegments)
            )

        pyM.pwlcfDesignSegmentSet = pyomo.Set(dimen=3, initialize=declareDesignSegmentSet)

    def declareVariables(self, esM, pyM):
        if not pyomo_pwlf:
            self.declareBinaryPwlcfVar(esM, pyM)
            self.declareSegmentCapacityPwlcfVar(esM, pyM)

    def declareBinaryPwlcfVar(self, esM, pyM):
        pyM.binaryPwlcfVar = pyomo.Var(pyM.pwlcfDesignSegmentSet, domain=pyomo.Binary)

    def declareSegmentCapacityPwlcfVar(self, esM, pyM):
        pyM.segmentCapacityPwlcfVar = pyomo.Var(
            pyM.pwlcfDesignSegmentSet,
            domain=pyomo.NonNegativeReals,
        )

    def declareComponentConstraints(self, esM, pyM):
        if pyomo_pwlf:
            self.declarePwlfPyomo(esM, pyM)
        else:
            self.declareBinaryPwlcfConstr(pyM)
            self.declareSegmentCapacityPwlcfConstr(pyM)
            self.declareCapacityCommissioningPwlcfConstr(esM, pyM)
            self.declareCapacityMaxEosConstr(esM, pyM)

    def declareBinaryPwlcfConstr(self, pyM):
        def binaryPwlcfConstr(pyM, modulName, ip, segment):
            return (
                sum(
                    pyM.binaryPwlcfVar[modulName, ip, segment]
                    for segment in range(self.modulsDict[modulName].noSegments)
                )
                == 1
            )

        pyM.ConstrBinaryPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSegmentSet, rule=binaryPwlcfConstr
        )

    def declareSegmentCapacityPwlcfConstr(self, pyM):
        def lowerSegmentCapacityPwlcfConstr(pyM, modulName, ip, segment):
            modul = self.modulsDict[modulName]
            if modul.pwlcf_type == "etl":
                maxCapacityPerSegment = modul.linEtlParameter["experience"]
            else:
                maxCapacityPerSegment = modul.eosParameters["capacity"]
            lowerCapacityBound = maxCapacityPerSegment.loc[segment]
            binVar = pyM.binaryPwlcfVar[modulName, ip, segment]
            capSegmentVar = pyM.segmentCapacityPwlcfVar[modulName, ip, segment]

            return lowerCapacityBound * binVar <= capSegmentVar

        def upperSegmentCapacityPwlcfConstr(pyM, modulName, ip, segment):
            modul = self.modulsDict[modulName]
            if modul.pwlcf_type == "etl":
                maxCapacityPerSegment = modul.linEtlParameter["experience"]
            else:
                maxCapacityPerSegment = modul.eosParameters["capacity"]
            upperCapacityBound = maxCapacityPerSegment.loc[segment + 1]
            binVar = pyM.binaryPwlcfVar[modulName, ip, segment]
            capSegmentVar = pyM.segmentCapacityPwlcfVar[modulName, ip, segment]

            return capSegmentVar <= upperCapacityBound * binVar

        pyM.ConstrLowerSegmentCapacityPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSegmentSet, rule=lowerSegmentCapacityPwlcfConstr
        )

        pyM.ConstrUpperSegmentCapacityPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSegmentSet, rule=upperSegmentCapacityPwlcfConstr
        )

    def declareCapacityCommissioningPwlcfConstr(self, esM, pyM):
        def capacityCommissioningPwlcfConstr(pyM, modulName, ip):
            modul = self.modulsDict[modulName]
            compClass = modul.comp.modelingClass().abbrvName
            commVar = getattr(pyM, "commis_" + compClass)
            commVarSum = sum(
                commVar[loc, modulName, _ip]
                for _ip in range(ip + 1)
                for loc in esM.locations
            )
            capSegmentVarSum = sum(
                pyM.segmentCapacityPwlcfVar[modulName, ip, segment]
                for segment in range(modul.noSegments)
            )
            if self.modulsDict[modulName].pwlcf_type == "eos":
                return capSegmentVarSum == commVarSum
            elif self.modulsDict[modulName].pwlcf_type == "etl":
                return capSegmentVarSum == commVarSum + modul.initCapacity

        pyM.ConstrCapacityCommissioningPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSet, rule=capacityCommissioningPwlcfConstr
        )

    def declareCapacityMaxEosConstr(self, esM, pyM):
        def capacityMaxEosConstr(pyM, modulName, ip):
            modul = self.modulsDict[modulName]
            if modul.pwlcf_type == "etl":
                return pyomo.Constraint.Skip
            else:
                compClass = modul.comp.modelingClass().abbrvName
                commVar = getattr(pyM, "commis_" + compClass)
                maxCapacity = modul.eosParameters["capacity"].iloc[-1]
                loc = list(esM.locations)[0]
                return commVar[loc, modulName, ip] <= maxCapacity

        pyM.ConstrCapacityMaxEos = pyomo.Constraint(
            pyM.pwlcfDesignSet, rule=capacityMaxEosConstr
        )


    def declarePwlfPyomo(self, esM, pyM):
        """
        https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html#piecewise-linear-expressions
        """
        pyM.totalCost = pyomo.Var(
            pyM.pwlfDesignSet,
            domain=pyomo.NonNegativeReals,
        )

        def totalCapacityBounds(pyM, modulName, ip):
            return (0, self.modulsDict[modulName].maxCapacity)

        pyM.totalCapacity = pyomo.Var(
            pyM.pwlfDesignSet,
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

        pyM.fixTotalCapacity = pyomo.Constraint(pyM.pwlfDesignSet, rule=fixTotalCapacity)

        xdata = {
            idx: list(self.modulsDict[idx[0]].linEtlParameter["experience"])
            for idx in pyM.pwlfDesignSet
        }

        ydata = {
            idx: list(self.modulsDict[idx[0]].linEtlParameter["totalCost"])
            for idx in pyM.pwlfDesignSet
        }

        pyM.pwlf = Piecewise(
            pyM.pwlfDesignSet,
            pyM.totalCost,
            pyM.totalCapacity,
            pw_pts=xdata,
            pw_constr_type="EQ",
            f_rule=ydata,
            pw_repn="SOS2",
        )

    def getObjectiveFunctionContribution(self, esM, pyM):
        return self.getEconomicsPwlcf(esM, pyM)

    def getEconomicsPwlcf(
        self,
        esM,
        pyM,
        getOptValue=False,
        getOptValueCostType="TAC",
    ):
        componentYears = {
            modulName: esM.getComponentAttribute(modulName, "processedStockYears")
            + esM.investmentPeriods
            for modulName in self.modulsDict.keys()
        }

        costContribution = {
            modulName: {
                (y, i): 0 for y in modul.commisYears for i in esM.investmentPeriods
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

            (fullCostIntervals, costInLastEconInterval, costInLastTechInterval) = (
                utils.getParametersForUnevenLifetimes(
                    modulName, loc, "ipEconomicLifetime", esM
                )
            )

            for commisYear in modul.commisYears:

                if self.modulsDict[modulName].pwlcf_type == "eos":
                    opex = self.getOpexEos(pyM, modulName, getOptValue)
                    annuity = self.getAnnuityEos(pyM, modulName, getOptValue)
                else:
                    opex = 0
                    annuity = self.getAnnuityEtl(
                        pyM, modulName, commisYear, modul.commisYears, getOptValue
                    )

                for i in range(commisYear, commisYear + fullCostIntervals):
                    costContribution[modulName][(commisYear, i)] = (
                        (annuity + opex)
                        * utils.annuityPresentValueFactor(
                            esM, modulName, loc, esM.investmentPeriodInterval
                        )
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

                if costInLastTechInterval and ipTechnicalLifetime % 1 != 0:
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
                        cost_results[ip].loc[modulName, loc] = (
                            cContrSum * utils.discountFactor(esM, ip, modulName, loc)
                        )
                    elif getOptValueCostType == "TAC":
                        cost_results[ip].loc[modulName, loc] = (
                            cContrSum
                            / utils.annuityPresentValueFactor(
                                esM, modulName, loc, esM.investmentPeriodInterval
                            )
                        )
                    elif getOptValueCostType == "invest":
                        cost_results[ip].loc[modulName, loc] = (
                            annuity * self.modulsDict[modulName].comp.CCF[0].mean()
                        )

            return cost_results
        if esM.annuityPerpetuity:
            for modulName in costContribution.keys():  # noqa: PLC0206
                for y in componentYears[modulName]:
                    costContribution[modulName][(y, esM.investmentPeriods[-1])] = (
                        costContribution[modulName][(y, esM.investmentPeriods[-1])]
                        / (
                            utils.annuityPresentValueFactor(
                                esM, modulName, loc, esM.investmentPeriodInterval
                            )
                            * esM.getComponent(modulName).interestRate[loc]
                        )
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

    def getAnnuityEos(self, pyM, modulName, getOptValue=False):
        modul = self.modulsDict[modulName]
        if not getOptValue:
            totalCost = sum(
                pyM.binaryPwlcfVar[modulName, 0, segment] * modul.eosParameters["interceptionTotalInvest"].iloc[segment] +
                pyM.segmentCapacityPwlcfVar[modulName, 0, segment] * modul.eosParameters["slopeTotalInvest"].iloc[segment]
                for segment in range(modul.noSegments)
            )
        else:
            totalCost = sum(
                pyM.binaryPwlcfVar[modulName, 0, segment].value * modul.eosParameters["interceptionTotalInvest"].iloc[segment] +
                pyM.segmentCapacityPwlcfVar[modulName, 0, segment].value * modul.eosParameters["slopeTotalInvest"].iloc[segment]
                for segment in range(modul.noSegments)
            )
        return totalCost / self.modulsDict[modulName].comp.CCF[0].mean()

    def getOpexEos(self, pyM, modulName, getOptValue=False):
        modul = self.modulsDict[modulName]
        if not getOptValue:
            totalOpexFix = sum(
                pyM.binaryPwlcfVar[modulName, 0, segment] * modul.eosParameters["interceptionTotalOpex"].iloc[segment] +
                pyM.segmentCapacityPwlcfVar[modulName, 0, segment] * modul.eosParameters["slopeTotalOpex"].iloc[segment]
                for segment in range(modul.noSegments)
            )
        else:
            totalOpexFix = sum(
                pyM.binaryPwlcfVar[modulName, 0, segment].value * modul.eosParameters["interceptionTotalOpex"].iloc[segment] +
                pyM.segmentCapacityPwlcfVar[modulName, 0, segment].value * modul.eosParameters["slopeTotalOpex"].iloc[segment]
                for segment in range(modul.noSegments)
            )
        return totalOpexFix

    def getAnnuityEtl(
        self, pyM, modulName, commisYear, commisYears, getOptValues=False
    ):
        def getIpTotalCost(ip):
            if ip == commisYears[0] - 1:
                totalCost = modul.getTotalCostEtl(
                    modul.initCapacity - modul.comp.stockCapacityStartYear.sum()
                )
            elif ip < 0:
                unbuildStockUntilIp = sum(
                    modul.comp.processedStockCommissioning[i].sum()
                    for i in range(ip + 1, 0)
                )
                totalCost = modul.getTotalCostEtl(modul.initCapacity - unbuildStockUntilIp)
            elif pyomo_pwlf:
                if not getOptValues:
                    totalCost = pyM.totalCost[modulName, ip]
                else:
                    totalCost = pyM.totalCost[modulName, ip].value
            elif not getOptValues:
                totalCost = sum(
                    modul.linEtlParameter["interception"].loc[segment + 1]
                    * pyM.binaryPwlcfVar[modulName, ip, segment]
                    + modul.linEtlParameter["slope"].loc[segment + 1]
                    * pyM.segmentCapacityPwlcfVar[modulName, ip, segment]
                    for segment in range(modul.noSegments)
                )
            else:
                totalCost = sum(
                    modul.linEtlParameter["interception"].loc[segment + 1]
                    * pyM.binaryPwlcfVar[modulName, ip, segment].value
                    + modul.linEtlParameter["slope"].loc[segment + 1]
                    * pyM.segmentCapacityPwlcfVar[modulName, ip, segment].value
                    for segment in range(modul.noSegments)
                )
            return totalCost

        modul = self.modulsDict[modulName]
        totalCostCommisYear = getIpTotalCost(commisYear)
        totalCostPreCommisYear = getIpTotalCost(commisYear - 1)

        return (totalCostCommisYear - totalCostPreCommisYear) / modul.comp.CCF[
            commisYear
        ].mean()

    def setOptimalValues(self, esM, pyM):
        loc = list(esM.locations)[0]

        tac = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType="TAC"
        )
        npv = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType="NPV"
        )
        invest = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType="invest"
        )

        for ip in esM.investmentPeriods:
            for modulName, modul in self.modulsDict.items():

                #initialize different dataframe for ETL/EOS:
                if modul.pwlcf_type == "etl":
                    curPWLCFtype = "ETL"
                    props = ["TAC_ETL", "NPVcontribution_ETL", "invest_ETL", "knowledgeStock_ETL"]
                    units = [
                        "[" + esM.costUnit + "/a]",
                        "[" + esM.costUnit + "]",
                        "[" + esM.costUnit + "]",
                        "[-]",
                    ]
                else: 
                    curPWLCFtype = "EOS"
                    props = ["TAC_EOS", "NPVcontribution_EOS", "invest_EOS"]
                    units = [
                        "[" + esM.costUnit + "/a]",
                        "[" + esM.costUnit + "]",
                        "[" + esM.costUnit + "]",
                    ]

                tuples = [
                    (modName, prop, unit)
                    for modName in self.modulsDict.keys()
                    for prop, unit in zip(props, units)
                ]

                unitDict = {
                    "conv": ("physicalUnit", ""),
                    "srcSnk": ("commodityUnit", ""),
                    "stor": ("commodityUnit", "*h"),
                    "trans": ("commodityUnit", ""),
                }

                tuples = list(
                    map(
                        lambda x: (
                            x[0],
                            x[1],
                            "["
                            + getattr(
                                self.modulsDict[x[0]].comp,
                                unitDict[self.modulsDict[x[0]].comp.modelingClass().abbrvName][
                                    0
                                ],
                            )
                            + unitDict[self.modulsDict[x[0]].comp.modelingClass().abbrvName][1]
                            + "]",
                        )
                        if x[1] == "knowledgeStock_ETL"
                        else x,
                        tuples,
                    )
                )
                mIndex = pd.MultiIndex.from_tuples(
                    tuples, names=["Component", "Property", "Unit"]
                )

                optSummaryPwlcf = {
                    ip: pd.DataFrame(index=mIndex, columns=list(esM.locations)).sort_index()
                    for ip in esM.investmentPeriodNames
                }




                optSummaryPwlcf[esM.investmentPeriodNames[ip]].loc[
                    (modulName, f"TAC_{curPWLCFtype}", "[" + esM.costUnit + "/a]"), loc
                ] = tac[ip][loc].loc[modulName]

                optSummaryPwlcf[esM.investmentPeriodNames[ip]].loc[
                    (modulName, f"NPVcontribution_{curPWLCFtype}", "[" + esM.costUnit + "]"), loc
                ] = npv[ip][loc].loc[modulName]
                
                optSummaryPwlcf[esM.investmentPeriodNames[ip]].loc[
                    (modulName, f"invest_{curPWLCFtype}", "[" + esM.costUnit + "]"), loc
                ] = invest[ip][loc].loc[modulName]
                
                if pyomo_pwlf and curPWLCFtype == "ETL":
                    knowledgeStock = pyM.totalCapacity[modulName, ip].value
                elif curPWLCFtype == "ETL":
                    knowledgeStock = sum(
                        pyM.segmentCapacityPwlcfVar[modulName, ip, segment]._value
                        for segment in range(modul.noSegments)
                    )
                if curPWLCFtype == "ETL":
                    optSummaryPwlcf[esM.investmentPeriodNames[ip]].loc[
                        (
                            modulName,
                            "knowledgeStock_ETL",
                            "["
                            + getattr(
                                modul.comp,
                                unitDict[modul.comp.modelingClass().abbrvName][0],
                            )
                            + unitDict[modul.comp.modelingClass().abbrvName][1]
                            + "]",
                        ),
                        loc,
                    ] = knowledgeStock

        for model in esM.componentModelingDict.values():
            optSummary = model._optSummary
            for ipName in esM.investmentPeriodNames:
                etlComps = [
                    comp
                    for comp in model.componentsDict.keys()
                    if comp in self.modulsDict.keys()
                    if self.modulsDict[comp].pwlcf_type == "etl"
                ]
                eosComps = [
                    comp
                    for comp in model.componentsDict.keys()
                    if comp in self.modulsDict.keys()
                    if self.modulsDict[comp].pwlcf_type == "eos"
                ]
                optSummary[ipName] = pd.concat(
                    [optSummary[ipName], optSummaryPwlcf[ipName].loc[etlComps, :, :]],
                    axis=0,
                ).sort_index()
                optSummary[ipName] = pd.concat(
                    [optSummary[ipName], optSummaryPwlcf[ipName].loc[eosComps, :, :]],
                    axis=0,
                ).sort_index()
                if len(eosComps) > 0:
                    optSummary[ipName].loc[eosComps,'TAC',:] += optSummaryPwlcf[ipName].loc[:,'TAC_EOS',:]
                    optSummary[ipName].loc[eosComps,'NPVcontribution',:] += optSummaryPwlcf[ipName].loc[:,'NPVcontribution_EOS',:]
                    optSummary[ipName].loc[eosComps,'invest',:] += optSummaryPwlcf[ipName].loc[:,'invest_EOS',:]
                if len(etlComps) > 0:
                    optSummary[ipName].loc[etlComps,'TAC',:] += optSummaryPwlcf[ipName].loc[:,'TAC_ETL',:]
                    optSummary[ipName].loc[etlComps,'NPVcontribution',:] += optSummaryPwlcf[ipName].loc[:,'NPVcontribution_ETL',:]
                    optSummary[ipName].loc[eosComps,'invest',:] += optSummaryPwlcf[ipName].loc[:,'invest_ETL',:]          
            model.optSummary = optSummary[esM.startYear]
