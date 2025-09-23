import math

import pandas as pd
import pyomo.environ as pyomo
from pyomo.core import Piecewise

from fine import utils, utilsPWLCF

pyomo_pwlf = False


class PiecewiseLinearCostFunctionModule:
    def __init__(
        self,
        comp,
        esM,
        etlParameters=None,
        eosParameters=None,
    ):
        """
        Constructor for initialization of piecewise linear cost function module. At this stage, either endogenous technology learning or economies of scale (plant/location specific) can be used.

        :param comp: component for which the pwlcf should be added.
        :type comp: Component instance from the FINE package

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param etlParameters: parameters used for the endogenous technology learning approach. Required entries:
            "initCost": float, initial Cost at initial capacity, i.e. 500 €/kW
            "learningRate": float, learning rate for cost reductions
            "initCapacity": float, initial capacity
            "maxCapacity": float, maximum capacity until where learning effects are valid
            "noSegments": float, number of segments used for approximation of nonlinear cost function. defaults to 4.
        :type etlParameters: dict

        "eosParameters": pd.DataFrame(data=np.array([[0,1,2,3],[0,1000, 1800, 2400],[0, 10, 18, 24]]).T, columns=["capacity", "totalInvest", "totalOpex"])
        :param eosParameters: parameters used for economies of scale approach. Required columns:
            "capacity": float, capacity at which the totalInvest/totalOpex are valid
            "totalInvest": float, total Invest at specified capacity
            "totalOpex": float, total opex at specified capacity
            -At each index rising capacities are defined and corresponding invest/opex are defined. Between the defined supporting points the cost is linearily interpolated.
        :type eosParameters: pandas DataFrame
        """
        self.comp = comp

        if etlParameters and eosParameters is not None:
            raise NotImplementedError(
                f"Specifying both, endogenous technology learning (etl) and economies of scale (eos) is not valid. Check component: {self.comp}."
            )
        if etlParameters:
            self.pwlcf_type = "etl"
            self.learningRate = etlParameters["learningRate"]
            self.learningIndex = utilsPWLCF.checkAndSetLearningIndex(
                etlParameters["learningRate"]
            )
            self.initCost = utilsPWLCF.checkAndSetInitCost(
                etlParameters["initCost"], comp
            )
            self.initCapacity, self.maxCapacity = utilsPWLCF.checkCapacitiesEtl(
                etlParameters["initCapacity"], etlParameters["maxCapacity"], comp
            )
            utilsPWLCF.checkStock(comp, self.initCapacity)
            utilsPWLCF.checkMaxCapacity(comp, self.maxCapacity)
            utilsPWLCF.checkEtlCompParams(comp)

            if etlParameters["noSegments"] is None:
                self.noSegments = 4
            else:
                utils.isStrictlyPositiveInt(int(etlParameters["noSegments"]))
                self.noSegments = int(etlParameters["noSegments"])

                self.linEtlParameter = self.linearizeLearningCurveEtl()

        elif eosParameters is not None:
            if pyomo_pwlf:
                raise NotImplementedError(
                    "SOS2 Constraints via pyomo.pwlf currently not implemented for economies of scale."
                )
            self.pwlcf_type = "eos"
            utilsPWLCF.checkInvestmentPeriods(esM)
            self.eosParameters = utilsPWLCF.checkAndSetEosParameters(
                comp, eosParameters
            )
            self.initCapacity = 0
            self.noSegments = len(eosParameters["capacity"]) - 1

        self.commisYears = comp.processedStockYears + esM.investmentPeriods

    def getTotalCostEtl(self, capacity):
        """
        Function to calculate the total cost of a component with ETL.

        :param capacity: The capacity of the component for which the total cost (Invest) is calculated
        :type capacity: float

        :return: total cost at capacity
        :rtype: float
        """
        return ((self.initCapacity * self.initCost) / (1 - self.learningIndex)) * (
            capacity / self.initCapacity
        ) ** (1 - self.learningIndex)

    def linearizeLearningCurveEtl(self):
        """
        Function to linearize the learning curve based on the given initial capacity, cost, and maximal capacity as well as the learning  rate.

        :return: linearized etl parameters:
            cumulative experience, totalCost, slope and interception for each segments linear approximation
        :rtype: pd.DataFrame
        """
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
        self.modulesDict = {}

    def declareSets(self, esM, pyM):
        """
        Function to declare the necessary sets for the variables of the pwlcf model.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        self.declarePwlcfDesignSet(pyM, esM)
        if not pyomo_pwlf:
            self.declarePwlcfDesignSegmentSet(pyM, esM)

    def declarePwlcfDesignSet(self, pyM, esM):
        """
        Function to declare the necessary sets for the variables of the pwlcf model, if the pwlcf approach from pyomo via SOS2 constraints is used. Declares set for each module, investment period.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        def declareDesignSet(pyM):
            return (
                (moduleName, ip)
                for moduleName, module in self.modulesDict.items()
                for ip in esM.investmentPeriods
            )

        pyM.pwlcfDesignSet = pyomo.Set(dimen=2, initialize=declareDesignSet)

    def declarePwlcfDesignSegmentSet(self, pyM, esM):
        """
        Function to declare the necessary sets for the variables of the pwlcf model. Declares set for each module, investment period and segment.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        def declareDesignSegmentSet(pyM):
            return (
                (moduleName, ip, segment)
                for moduleName, module in self.modulesDict.items()
                for ip in esM.investmentPeriods
                for segment in range(module.noSegments)
            )

        pyM.pwlcfDesignSegmentSet = pyomo.Set(
            dimen=3, initialize=declareDesignSegmentSet
        )

    def declareVariables(self, esM, pyM):
        """
        Function to declare the variables of the pwlcf model. Declares binary variables for each segment to indicate which segment is active as well as segment capacity variables which give the exact capacity (for each segment, 0 if the corresponding binary is 0).

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        if not pyomo_pwlf:
            self.declareBinaryPwlcfVar(pyM)
            self.declareSegmentCapacityPwlcfVar(pyM)

    def declareBinaryPwlcfVar(self, pyM):
        """
        Function to add the binary variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        pyM.binaryPwlcfVar = pyomo.Var(pyM.pwlcfDesignSegmentSet, domain=pyomo.Binary)

    def declareSegmentCapacityPwlcfVar(self, pyM):
        """
        Function to add the segment capacity variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        pyM.segmentCapacityPwlcfVar = pyomo.Var(
            pyM.pwlcfDesignSegmentSet,
            domain=pyomo.NonNegativeReals,
        )

    def declareComponentConstraints(self, esM, pyM):
        """
        Function to declare the constraints of the pwlcf model.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        if pyomo_pwlf:
            self.declarePwlfPyomo(esM, pyM)
        else:
            self.declareBinaryPwlcfConstr(pyM)
            self.declareSegmentCapacityPwlcfConstr(pyM)
            self.declareCapacityCommissioningPwlcfConstr(esM, pyM)

    def declareBinaryPwlcfConstr(self, pyM):
        """
        Function to add the binary constraints: for each component exactly one binary has to be 1 and the others 0. The binary indicates, which segment is active.

        .. math::
            \\begin{eqnarray*}
            \\underset{segment}{ \\sum } binVar^{comp}_{ip,segment} = 1
            \\end{eqnarray*}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        def binaryPwlcfConstr(pyM, moduleName, ip, segment):
            return (
                sum(
                    pyM.binaryPwlcfVar[moduleName, ip, segment]
                    for segment in range(self.modulesDict[moduleName].noSegments)
                )
                == 1
            )

        pyM.ConstrBinaryPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSegmentSet, rule=binaryPwlcfConstr
        )

    def declareSegmentCapacityPwlcfConstr(self, pyM):
        """
        Function to add the segment capacity constraints: Each segment capacity variable has to be within the lower and upper bounds of the corresponding segment, if the segment is active (indicated by the binary segment variable). If the segment is not active, the capacity segment variable is zero.

        .. math::
            \\begin{eqnarray*}
            lowerCapacityBound^{comp}_{ip,segment} \\cdot binVar^{comp}_{ip,segment} \\leq capSegmentVar^{comp}_{ip,segment} \\leq  upperCapacityBound^{comp}_{ip,segment} \\cdot binVar^{comp}_{ip,segment}
            \\end{eqnarray*}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        def lowerSegmentCapacityPwlcfConstr(pyM, moduleName, ip, segment):
            module = self.modulesDict[moduleName]
            if module.pwlcf_type == "etl":
                maxCapacityPerSegment = module.linEtlParameter["experience"]
            else:
                maxCapacityPerSegment = module.eosParameters["capacity"]
            lowerCapacityBound = maxCapacityPerSegment.loc[segment]
            binVar = pyM.binaryPwlcfVar[moduleName, ip, segment]
            capSegmentVar = pyM.segmentCapacityPwlcfVar[moduleName, ip, segment]

            return lowerCapacityBound * binVar <= capSegmentVar

        def upperSegmentCapacityPwlcfConstr(pyM, moduleName, ip, segment):
            module = self.modulesDict[moduleName]
            if module.pwlcf_type == "etl":
                maxCapacityPerSegment = module.linEtlParameter["experience"]
            else:
                maxCapacityPerSegment = module.eosParameters["capacity"]
            upperCapacityBound = maxCapacityPerSegment.loc[segment + 1]
            binVar = pyM.binaryPwlcfVar[moduleName, ip, segment]
            capSegmentVar = pyM.segmentCapacityPwlcfVar[moduleName, ip, segment]

            return capSegmentVar <= upperCapacityBound * binVar

        pyM.ConstrLowerSegmentCapacityPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSegmentSet, rule=lowerSegmentCapacityPwlcfConstr
        )

        pyM.ConstrUpperSegmentCapacityPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSegmentSet, rule=upperSegmentCapacityPwlcfConstr
        )

    def declareCapacityCommissioningPwlcfConstr(self, esM, pyM):
        """Function to enforce that capacity segment variable equals totalCommisioning (in all locations over all ips) + initial capacity

        .. math::
            \\begin{eqnarray*}
            \\underset{segment}{ \\sum } segmentCapacityPwlcfVar^{comp}_{ip,segment} = underset{ip,comp}{ \\sum } commVar^{comp}_{ip,segment,loc} + initCapacity^{comp}
            \\end{eqnarray*}

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        def capacityCommissioningPwlcfConstr(pyM, moduleName, ip):
            module = self.modulesDict[moduleName]
            compClass = module.comp.modelingClass().abbrvName
            commVar = getattr(pyM, "commis_" + compClass)
            commVarSum = sum(
                commVar[loc, moduleName, _ip]
                for _ip in range(ip + 1)
                for loc in esM.locations
            )
            capSegmentVarSum = sum(
                pyM.segmentCapacityPwlcfVar[moduleName, ip, segment]
                for segment in range(module.noSegments)
            )
            return capSegmentVarSum == commVarSum + module.initCapacity

        pyM.ConstrCapacityCommissioningPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSet, rule=capacityCommissioningPwlcfConstr
        )

    def declarePwlfPyomo(self, esM, pyM):
        """
        https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html#piecewise-linear-expressions
        """
        pyM.totalCost = pyomo.Var(
            pyM.pwlfDesignSet,
            domain=pyomo.NonNegativeReals,
        )

        def totalCapacityBounds(pyM, moduleName, ip):
            return (0, self.modulesDict[moduleName].maxCapacity)

        pyM.totalCapacity = pyomo.Var(
            pyM.pwlfDesignSet,
            domain=pyomo.NonNegativeReals,
            bounds=totalCapacityBounds,
        )

        def fixTotalCapacity(pyM, moduleName, ip):
            module = self.modulesDict[moduleName]
            compClass = module.comp.modelingClass().abbrvName
            commVar = getattr(pyM, "commis_" + compClass)
            commVarSum = sum(
                commVar[loc, moduleName, _ip]
                for _ip in range(ip + 1)
                for loc in esM.locations
            )

            return pyM.totalCapacity[moduleName, ip] == commVarSum + module.initCapacity

        pyM.fixTotalCapacity = pyomo.Constraint(
            pyM.pwlfDesignSet, rule=fixTotalCapacity
        )

        xdata = {
            idx: list(self.modulesDict[idx[0]].linEtlParameter["experience"])
            for idx in pyM.pwlfDesignSet
        }

        ydata = {
            idx: list(self.modulesDict[idx[0]].linEtlParameter["totalCost"])
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
        """
        Function to get the economic contribution to the cost function of pwlcf components.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param getOptValue: indicator if optimal value extracted from optimization output should be used
        :type getOptValue: binary

        :param getOptValueCostType: indicator which cost type is used, i.e. "TAC" or "NPV"
        :type getOptValueCostType: string
        """
        componentYears = {
            moduleName: esM.getComponentAttribute(moduleName, "processedStockYears")
            + esM.investmentPeriods
            for moduleName in self.modulesDict.keys()
        }

        costContribution = {
            moduleName: {
                (y, i): 0 for y in module.commisYears for i in esM.investmentPeriods
            }
            for moduleName, module in self.modulesDict.items()
        }

        loc = list(esM.locations)[0]

        for moduleName, module in self.modulesDict.items():
            ipEconomicLifetime = getattr(
                esM.getComponent(moduleName), "ipEconomicLifetime"
            ).mean()
            ipTechnicalLifetime = getattr(
                esM.getComponent(moduleName), "ipTechnicalLifetime"
            ).mean()

            (fullCostIntervals, costInLastEconInterval, costInLastTechInterval) = (
                utils.getParametersForUnevenLifetimes(
                    moduleName, loc, "ipEconomicLifetime", esM
                )
            )

            for commisYear in module.commisYears:
                if self.modulesDict[moduleName].pwlcf_type == "eos":
                    opex = self.getCostContributionsPwlcf(
                        pyM,
                        moduleName,
                        self.modulesDict[moduleName].pwlcf_type,
                        "opex",
                        getOptValue=getOptValue,
                    )
                    annuity = self.getCostContributionsPwlcf(
                        pyM,
                        moduleName,
                        self.modulesDict[moduleName].pwlcf_type,
                        "annuity",
                        getOptValue=getOptValue,
                    )
                else:
                    opex = self.getCostContributionsPwlcf(
                        pyM,
                        moduleName,
                        self.modulesDict[moduleName].pwlcf_type,
                        "opex",
                        commisYear=commisYear,
                        getOptValue=getOptValue,
                    )
                    annuity = self.getCostContributionsPwlcf(
                        pyM,
                        moduleName,
                        self.modulesDict[moduleName].pwlcf_type,
                        "annuity",
                        commisYear=commisYear,
                        getOptValue=getOptValue,
                    )

                for i in range(commisYear, commisYear + fullCostIntervals):
                    costContribution[moduleName][(commisYear, i)] = (
                        annuity + opex
                    ) * utils.annuityPresentValueFactor(
                        esM, moduleName, loc, esM.investmentPeriodInterval
                    )

                if costInLastEconInterval:
                    partlyCostInLastEconomicInterval = (
                        ipEconomicLifetime % 1
                    ) * esM.investmentPeriodInterval
                    costContribution[moduleName][
                        (commisYear, commisYear + fullCostIntervals)
                    ] = annuity * utils.annuityPresentValueFactor(
                        esM, moduleName, loc, partlyCostInLastEconomicInterval
                    )

                if costInLastTechInterval and ipTechnicalLifetime % 1 != 0:
                    partlyCostInLastTechnicalInterval = (
                        1 - (ipTechnicalLifetime % 1)
                    ) * esM.investmentPeriodInterval
                    if commisYear + math.ceil(ipTechnicalLifetime) - 1 in [
                        k[1] for k in costContribution[moduleName].keys()
                    ]:
                        costContribution[moduleName][
                            (
                                commisYear,
                                commisYear + math.ceil(ipTechnicalLifetime) - 1,
                            )
                        ] = costContribution[moduleName][
                            (
                                commisYear,
                                commisYear + math.ceil(ipTechnicalLifetime) - 1,
                            )
                        ] + annuity * (
                            utils.annuityPresentValueFactor(
                                esM,
                                moduleName,
                                loc,
                                partlyCostInLastTechnicalInterval,
                            )
                            / (1 + esM.getComponent(moduleName).interestRate[loc])
                            ** (
                                esM.investmentPeriodInterval
                                - partlyCostInLastTechnicalInterval
                            )
                        )

        if getOptValue:
            cost_results = {ip: pd.DataFrame() for ip in esM.investmentPeriods}
            for moduleName in self.modulesDict.keys():
                commis = {
                    ip: esM.getOptimizationSummary(
                        esM.componentNames[moduleName], ip=esM.investmentPeriodNames[ip]
                    )
                    .loc[moduleName, "commissioning"]
                    .iloc[0]
                    for ip in esM.investmentPeriods
                }
                for ip in esM.investmentPeriods:
                    for _loc in esM.locations:
                        cContrSum = sum(
                            costContribution[moduleName].get((y, ip), 0)
                            * commis[y][_loc]
                            / commis[y].sum()
                            if y > 0 and commis[y].sum() != 0
                            else 0
                            if y > 0
                            else costContribution[moduleName].get((y, ip), 0)
                            / len(commis[ip])
                            for y in componentYears[moduleName]
                        )
                        if getOptValueCostType == "NPV":
                            cost_results[ip].loc[moduleName, _loc] = (
                                cContrSum
                                * utils.discountFactor(esM, ip, moduleName, _loc)
                            )
                        elif getOptValueCostType == "TAC":
                            cost_results[ip].loc[moduleName, _loc] = (
                                cContrSum
                                / utils.annuityPresentValueFactor(
                                    esM, moduleName, _loc, esM.investmentPeriodInterval
                                )
                            )
                        elif getOptValueCostType == "invest":
                            if commis[ip].sum() != 0:
                                cost_results[ip].loc[moduleName, _loc] = (
                                    (
                                        annuity
                                        * self.modulesDict[moduleName]
                                        .comp.CCF[0]
                                        .mean()
                                    )
                                    * commis[ip][_loc]
                                    / commis[ip].sum()
                                )
                            else:
                                cost_results[ip].loc[moduleName, _loc] = 0

            return cost_results
        if esM.annuityPerpetuity:
            for moduleName in costContribution.keys():  # noqa: PLC0206
                for y in componentYears[moduleName]:
                    costContribution[moduleName][(y, esM.investmentPeriods[-1])] = (
                        costContribution[moduleName][(y, esM.investmentPeriods[-1])]
                        / (
                            utils.annuityPresentValueFactor(
                                esM, moduleName, loc, esM.investmentPeriodInterval
                            )
                            * esM.getComponent(moduleName).interestRate[loc]
                        )
                    )
        return sum(
            sum(
                [
                    costContribution[moduleName].get((y, ip), 0)
                    for y in componentYears[moduleName]
                ]
            )
            * utils.discountFactor(esM, ip, moduleName, loc)
            for moduleName in self.modulesDict.keys()
            for ip in esM.investmentPeriods
        )

    def getCostContributionsPwlcf(
        self, pyM, moduleName, pwlcf_type, costType, commisYear=None, getOptValue=False
    ):
        """
        Function to extract the cost contribution (opex and annuity) from a specified component and for a specified commisioning year.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param moduleName: name of the module/component
        :type moduleName: string

        :param pwlcf_type: type of the pwlcf, i.e. "etl" or "eos"
        :type pwlcf_type:  string

        :param costType: indicator which cost should be extracted, i.e. "opex" or "annuity"
        :type costType: string

        :param commisYear: for which comissioning year the data should be extracted. defaults to None (if only single IP)
        :type commisYear: float

        :param getOptValue: indicator, if value after optimization is wanted. defaults to False
        :type getOptValue: boolean
        """
        module = self.modulesDict[moduleName]
        commisYears = module.commisYears
        if costType == "opex":
            if pwlcf_type == "eos":
                if not getOptValue:
                    totalOpexFix = sum(
                        pyM.binaryPwlcfVar[moduleName, 0, segment]
                        * module.eosParameters["interceptionTotalOpex"].iloc[segment]
                        + pyM.segmentCapacityPwlcfVar[moduleName, 0, segment]
                        * module.eosParameters["slopeTotalOpex"].iloc[segment]
                        for segment in range(module.noSegments)
                    )
                else:
                    totalOpexFix = sum(
                        pyM.binaryPwlcfVar[moduleName, 0, segment].value
                        * module.eosParameters["interceptionTotalOpex"].iloc[segment]
                        + pyM.segmentCapacityPwlcfVar[moduleName, 0, segment].value
                        * module.eosParameters["slopeTotalOpex"].iloc[segment]
                        for segment in range(module.noSegments)
                    )
            elif pwlcf_type == "etl":
                totalOpexFix = 0  # varying opex not implemented for etl
            return totalOpexFix
        if costType == "annuity":
            if pwlcf_type == "eos":
                if not getOptValue:
                    totalCost = sum(
                        pyM.binaryPwlcfVar[moduleName, 0, segment]
                        * module.eosParameters["interceptionTotalInvest"].iloc[segment]
                        + pyM.segmentCapacityPwlcfVar[moduleName, 0, segment]
                        * module.eosParameters["slopeTotalInvest"].iloc[segment]
                        for segment in range(module.noSegments)
                    )
                else:
                    totalCost = sum(
                        pyM.binaryPwlcfVar[moduleName, 0, segment].value
                        * module.eosParameters["interceptionTotalInvest"].iloc[segment]
                        + pyM.segmentCapacityPwlcfVar[moduleName, 0, segment].value
                        * module.eosParameters["slopeTotalInvest"].iloc[segment]
                        for segment in range(module.noSegments)
                    )
            elif pwlcf_type == "etl":

                def getIpTotalCost(ip):
                    if ip == commisYears[0] - 1:
                        totalCost = module.getTotalCostEtl(
                            module.initCapacity
                            - module.comp.stockCapacityStartYear.sum()
                        )
                    elif ip < 0:
                        unbuildStockUntilIp = sum(
                            module.comp.processedStockCommissioning[i].sum()
                            for i in range(ip + 1, 0)
                        )
                        totalCost = module.getTotalCostEtl(
                            module.initCapacity - unbuildStockUntilIp
                        )
                    elif pyomo_pwlf:
                        if not getOptValue:
                            totalCost = pyM.totalCost[moduleName, ip]
                        else:
                            totalCost = pyM.totalCost[moduleName, ip].value
                    elif not getOptValue:
                        totalCost = sum(
                            module.linEtlParameter["interception"].loc[segment + 1]
                            * pyM.binaryPwlcfVar[moduleName, ip, segment]
                            + module.linEtlParameter["slope"].loc[segment + 1]
                            * pyM.segmentCapacityPwlcfVar[moduleName, ip, segment]
                            for segment in range(module.noSegments)
                        )
                    else:
                        totalCost = sum(
                            module.linEtlParameter["interception"].loc[segment + 1]
                            * pyM.binaryPwlcfVar[moduleName, ip, segment].value
                            + module.linEtlParameter["slope"].loc[segment + 1]
                            * pyM.segmentCapacityPwlcfVar[moduleName, ip, segment].value
                            for segment in range(module.noSegments)
                        )
                    return totalCost

                totalCostCommisYear = getIpTotalCost(commisYear)
                totalCostPreCommisYear = getIpTotalCost(commisYear - 1)
                totalCost = totalCostCommisYear - totalCostPreCommisYear
            return totalCost / module.comp.CCF[0].mean()  # total annuity
        raise NotImplementedError(
            f"Getting cost contribution of a pwlcf component is only defined for opex or annuity and not for {costType}."
        )

    def setOptimalValues(self, esM, pyM):
        """
        Function to set the optimal values into the optimization summary.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        tac = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType="TAC"
        )
        npv = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType="NPV"
        )
        invest = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType="invest"
        )

        optSummaryPwlcf = {}
        for ip in esM.investmentPeriods:
            optSummaryPwlcf[esM.investmentPeriodNames[ip]] = pd.DataFrame()
            for moduleName, module in self.modulesDict.items():
                # initialize different dataframe for ETL/EOS:
                if module.pwlcf_type == "etl":
                    curPWLCFtype = "ETL"
                    props = [
                        "TAC_ETL",
                        "NPVcontribution_ETL",
                        "invest_ETL",
                        "knowledgeStock_ETL",
                    ]
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
                    for modName in self.modulesDict.keys()
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
                                self.modulesDict[x[0]].comp,
                                unitDict[
                                    self.modulesDict[x[0]]
                                    .comp.modelingClass()
                                    .abbrvName
                                ][0],
                            )
                            + unitDict[
                                self.modulesDict[x[0]].comp.modelingClass().abbrvName
                            ][1]
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

                mdlOptSummaryPwlcf = pd.DataFrame(
                    index=mIndex, columns=list(esM.locations)
                ).sort_index()
                # optSummaryPwlcf = {
                #     ip: pd.DataFrame(index=mIndex, columns=list(esM.locations)).sort_index()
                #     for ip in esM.investmentPeriodNames
                # }

                mdlOptSummaryPwlcf.loc[
                    moduleName, f"TAC_{curPWLCFtype}", "[" + esM.costUnit + "/a]"
                ] = tac[ip].loc[moduleName]

                mdlOptSummaryPwlcf.loc[
                    moduleName,
                    f"NPVcontribution_{curPWLCFtype}",
                    "[" + esM.costUnit + "]",
                ] = npv[ip].loc[moduleName]
                mdlOptSummaryPwlcf.loc[
                    moduleName, f"invest_{curPWLCFtype}", "[" + esM.costUnit + "]"
                ] = invest[ip].loc[moduleName]
                if pyomo_pwlf and curPWLCFtype == "ETL":
                    knowledgeStock = pyM.totalCapacity[moduleName, ip].value
                elif curPWLCFtype == "ETL":
                    if ip == 0:
                        stockCapacityStartYear = self.modulesDict[
                            moduleName
                        ].comp.stockCapacityStartYear
                        knowledgeStock_last_ip = stockCapacityStartYear + (
                            (
                                self.modulesDict[moduleName].initCapacity
                                - stockCapacityStartYear.sum()
                            )
                            / len(esM.locations)
                        )
                    else:
                        knowledgeStock_last_ip = (
                            optSummaryPwlcf[esM.investmentPeriodNames[ip - 1]]
                            .loc[moduleName, "knowledgeStock_ETL"]
                            .iloc[0]
                        )
                    commis_ip = (
                        esM.getOptimizationSummary(
                            esM.componentNames[moduleName],
                            ip=esM.investmentPeriodNames[ip],
                        )
                        .loc[moduleName, "commissioning"]
                        .iloc[0]
                    )

                    knowledgeStock = knowledgeStock_last_ip + commis_ip

                if curPWLCFtype == "ETL":
                    mdlOptSummaryPwlcf.loc[
                        (
                            moduleName,
                            "knowledgeStock_ETL",
                            "["
                            + getattr(
                                module.comp,
                                unitDict[module.comp.modelingClass().abbrvName][0],
                            )
                            + unitDict[module.comp.modelingClass().abbrvName][1]
                            + "]",
                        )
                    ] = knowledgeStock
                optSummaryPwlcf[esM.investmentPeriodNames[ip]] = pd.concat(
                    [optSummaryPwlcf[esM.investmentPeriodNames[ip]], mdlOptSummaryPwlcf]
                )
            self._add_pwlcf_summary(esM=esM, optSummaryPwlcf=optSummaryPwlcf)

    def _add_pwlcf_summary(self, esM, optSummaryPwlcf):
        """Add

        Args:
            esM (_type_): _description_
        """
        for model in esM.componentModelingDict.values():
            optSummary = model._optSummary
            for ipName in esM.investmentPeriodNames:
                etlComps = [
                    comp
                    for comp in model.componentsDict.keys()
                    if comp in self.modulesDict.keys()
                    if self.modulesDict[comp].pwlcf_type == "etl"
                ]
                eosComps = [
                    comp
                    for comp in model.componentsDict.keys()
                    if comp in self.modulesDict.keys()
                    if self.modulesDict[comp].pwlcf_type == "eos"
                ]
                optSummary[ipName] = pd.concat(
                    [
                        optSummary[ipName],
                        optSummaryPwlcf[ipName].loc[etlComps, :, :],
                    ],
                    axis=0,
                ).sort_index()
                optSummary[ipName] = pd.concat(
                    [
                        optSummary[ipName],
                        optSummaryPwlcf[ipName].loc[eosComps, :, :],
                    ],
                    axis=0,
                ).sort_index()

                if len(eosComps) > 0:
                    for prop in ["TAC", "NPVcontribution", "invest"]:
                        # add df to df
                        property_slice = optSummaryPwlcf[ipName].loc[
                            (slice(None), [f"{prop}_EOS"], slice(None))
                        ]
                        property_slice = property_slice.rename(
                            index={f"{prop}_EOS": prop}, level="Property"
                        )
                        optSummary[ipName].loc[eosComps, prop, :] += property_slice
                if len(etlComps) > 0:
                    for prop in ["TAC", "NPVcontribution", "invest"]:
                        # add df to df
                        property_slice = optSummaryPwlcf[ipName].loc[
                            (slice(None), [f"{prop}_ETL"], slice(None))
                        ]
                        property_slice = property_slice.rename(
                            index={f"{prop}_ETL": prop}, level="Property"
                        )
                        optSummary[ipName].loc[eosComps, prop, :] += property_slice
            model.optSummary = optSummary[esM.startYear]
