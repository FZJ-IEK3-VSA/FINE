from fine import utils, utilsPWLCF
from fine.enums import ComponentAbbreviation, CostType
import math
import pyomo.environ as pyomo
from pyomo.core import Piecewise
import pandas as pd
import logging

logger = logging.getLogger(__name__)

pyomo_pwlf = False
use_sos2 = False


class PiecewiseLinearCostFunctionModule:
    """Handle the initialization and preprocessing of piecewise linear cost functions."""

    def __init__(
        self,
        comp,
        esM,
        etlParameters=None,
        eosParameters=None,
    ):
        """Initialize the piecewise linear cost function module.
        At this stage, either endogenous technology learning or economies of scale (plant/location specific) can be used.

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

        "eosParameters": pd.DataFrame(
            data=np.array([[0,1,2,3],[0,1000, 1800, 2400],[0, 10, 18, 24]]).T,
            columns=["capacity", "totalInvest", "totalOpex"]
        )
        :param eosParameters: parameters used for economies of scale approach. Required columns:
            "capacity": float, capacity at which the totalInvest/totalOpex are valid
            "totalInvest": float, total Invest at specified capacity
            "totalOpex": float, total opex at specified capacity
            -At each index rising capacities are defined and corresponding invest/opex are defined.
            Between the defined supporting points the cost is linearily interpolated.
        :type eosParameters: pandas DataFrame
        """
        self.comp = comp

        if etlParameters and eosParameters is not None:
            raise NotImplementedError(
                f"Specifying both, endogenous technology learning (etl) and economies of scale "
                f"(eos) is not valid. Check component: {self.comp}."
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
        """Calculate the total cost of a component with ETL.

        :param capacity: The capacity of the component for which the total cost (Invest) is calculated
        :type capacity: float

        :return: total cost at capacity
        :rtype: float
        """
        return ((self.initCapacity * self.initCost) / (1 - self.learningIndex)) * (
            capacity / self.initCapacity
        ) ** (1 - self.learningIndex)

    def linearizeLearningCurveEtl(self):
        """Linearize the learning curve.

        Linearization is based on the given initial capacity, cost, and maximum capacity, as well as the learning rate.

        :return: linearized etl parameters:
            cumulative experience, totalCost, slope and interception for each segments linear approximation
        :rtype: pd.DataFrame
        """
        linEtlParameter = pd.DataFrame(
            index=range(self.noSegments + 1),
            columns=["experience", "totalCost", "slope", "interception"],
        )

        linEtlParameter.loc[0, "totalCost"] = self.getTotalCostEtl(self.initCapacity)
        linEtlParameter.loc[self.noSegments, "totalCost"] = self.getTotalCostEtl(
            self.maxCapacity
        )
        totalCostDiff = (
            linEtlParameter.loc[self.noSegments, "totalCost"]
            - linEtlParameter.loc[0, "totalCost"]
        )

        for segment in range(1, self.noSegments):
            linEtlParameter.loc[segment, "totalCost"] = linEtlParameter.loc[
                segment - 1, "totalCost"
            ] + (2 ** (segment - self.noSegments - 1)) * (
                totalCostDiff / (1 - 0.5**self.noSegments)
            )

        linEtlParameter["experience"] = (
            (1 - self.learningIndex)
            / (self.initCost * self.initCapacity**self.learningIndex)
            * linEtlParameter["totalCost"]
        ) ** (1 / (1 - self.learningIndex))

        linEtlParameter["slope"] = (
            linEtlParameter["totalCost"].diff() / linEtlParameter["experience"].diff()
        )
        linEtlParameter["interception"] = (
            linEtlParameter["totalCost"]
            - linEtlParameter["slope"] * linEtlParameter["experience"]
        )

        return linEtlParameter


class PiecewiseLinearCostFunctionModel:
    """Model to handle piecewise linear cost functions within the energy system optimization.

    This class defines the necessary sets, variables, and constraints to represent
    piecewise linear cost functions in a Pyomo-based formulation. After declaring all
    structural model elements, the class extracts the economic contributions of PWL cost function
    components for a given commissioning year and stores their optimal values in the optimization summary.
    """

    def __init__(self):
        self.abbrvName = ComponentAbbreviation.PWLCF
        self.modulesDict = {}

    def declareSets(self, esM, pyM):
        """Declare the necessary sets for the variables of the pwlcf model.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        self.declarePwlcfDesignSet(pyM, esM)
        if not pyomo_pwlf:
            self.declarePwlcfDesignSegmentSet(pyM, esM)

    def declarePwlcfDesignSet(self, pyM, esM):
        """Declare the necessary sets for the variables of the pwlcf model.

        When using the pwlcf approach from Pyomo via SOS2 constraints:
        define a set for each module and investment period.

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
        """Declare the necessary sets for the variables of the pwlcf model.

        Define a set for each module, investment period, and segment.

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
                for segment in range(
                    module.noSegments if not use_sos2 else module.noSegments + 1
                )
            )

        pyM.pwlcfDesignSegmentSet = pyomo.Set(
            dimen=3, initialize=declareDesignSegmentSet
        )

    def declareVariables(self, esM, pyM):
        """Declare the variables of the pwlcf model.

        Define binary variables for each segment to indicate which segment is active, and
        segment capacity variables to specify the exact capacity for each segment
        (0 if the corresponding binary variable is 0).

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        if not pyomo_pwlf:
            self.declareBinaryPwlcfVar(pyM)
            if not use_sos2:
                self.declareSegmentCapacityPwlcfVar(pyM)

    def declareBinaryPwlcfVar(self, pyM):
        """Add binary variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        if use_sos2:
            pyM.sos2PwlcfVar = pyomo.Var(pyM.pwlcfDesignSegmentSet, bounds=(0, 1))
        else:
            pyM.binaryPwlcfVar = pyomo.Var(
                pyM.pwlcfDesignSegmentSet, domain=pyomo.Binary
            )

    def declareSegmentCapacityPwlcfVar(self, pyM):
        """Add segment capacity variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        pyM.segmentCapacityPwlcfVar = pyomo.Var(
            pyM.pwlcfDesignSegmentSet,
            domain=pyomo.NonNegativeReals,
        )

    def declareComponentConstraints(self, esM, pyM):
        """Declare constraints of the pwlcf model.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        if pyomo_pwlf:
            self.declarePwlfPyomo(esM, pyM)
        elif use_sos2:
            logger.info("Used SOS2 constraints.")
            self.declareBinaryPwlcfConstr(pyM)
            self.declareCapacityCommissioningPwlcfConstr(esM, pyM)
            self.declareSos2PwlcfConstr(pyM)
            self.declareBinarySpeedUpConstr(pyM)
        else:
            logger.info("Used Big-M constraints.")
            self.declareBinaryPwlcfConstr(pyM)
            self.declareSegmentCapacityPwlcfConstr(pyM)
            self.declareCapacityCommissioningPwlcfConstr(esM, pyM)
            self.declareBinarySpeedUpConstr(pyM)

    def declareBinarySpeedUpConstr(self, pyM):
        """Add binary speed up constraints."""
        logger.debug("Used binary speed up constraints.")
        if use_sos2:
            pwlcfVar = pyM.sos2PwlcfVar
        else:
            pwlcfVar = pyM.binaryPwlcfVar

        def binarySpeedUpUpperPwlcfConstr(pyM, moduleName, ip, segment):
            if ip == 0 or self.modulesDict[moduleName].pwlcf_type != "etl":
                return pyomo.Constraint.Skip
            return sum(
                pwlcfVar[moduleName, ip - 1, seg] for seg in range(segment + 1)
            ) >= sum(pwlcfVar[moduleName, ip, seg] for seg in range(segment + 1))

        pyM.ConstrBinarySpeedUpUpperPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSegmentSet, rule=binarySpeedUpUpperPwlcfConstr
        )

        def binarySpeedUpLowerPwlcfConstr(pyM, moduleName, ip, segment):
            if ip == 0 or self.modulesDict[moduleName].pwlcf_type != "etl":
                return pyomo.Constraint.Skip
            if use_sos2:
                seg_range_max = self.modulesDict[moduleName].noSegments + 1
            else:
                seg_range_max = self.modulesDict[moduleName].noSegments
            return sum(
                pwlcfVar[moduleName, ip - 1, seg]
                for seg in range(segment, seg_range_max)
            ) <= sum(
                pwlcfVar[moduleName, ip, seg] for seg in range(segment, seg_range_max)
            )

        pyM.ConstrBinarySpeedUpLowerPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSegmentSet, rule=binarySpeedUpLowerPwlcfConstr
        )

    def declareBinaryPwlcfConstr(self, pyM):
        r"""Add the binary constraints.
        For each component, exactly one binary has to be 1 and the others 0.
        The binary indicates which segment is active.

        .. math::
            \\begin{eqnarray*}
            \\underset{segment}{ \\sum } binVar^{comp}_{ip,segment} = 1
            \\end{eqnarray*}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        def binaryPwlcfConstr(pyM, moduleName, ip):
            if use_sos2:
                pwlcfVar = pyM.sos2PwlcfVar
                seg_range = range(self.modulesDict[moduleName].noSegments + 1)
            else:
                pwlcfVar = pyM.binaryPwlcfVar
                seg_range = range(self.modulesDict[moduleName].noSegments)
            return sum(pwlcfVar[moduleName, ip, segment] for segment in seg_range) == 1

        pyM.ConstrBinaryPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSet, rule=binaryPwlcfConstr
        )

    def declareSegmentCapacityPwlcfConstr(self, pyM):
        r"""Add the segment capacity constraints.

        Each segment capacity variable has to be within the lower and upper bounds of the corresponding segment,
        if the segment is active (indicated by the binary segment variable). If the segment is not active,
        the capacity segment variable is zero.

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
        r"""Enforce that the capacity segment variable equals the total commissioning.

        Constraint ist applied across all locations and investment periods, including also the initial capacity.

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
                if module.comp.processedLocationalEligibility.loc[loc] == 1
            )
            if use_sos2:
                capSegmentVarSum = sum(
                    pyM.sos2PwlcfVar[moduleName, ip, segment]
                    * self.modulesDict[moduleName]
                    .linEtlParameter["experience"]
                    .loc[segment]
                    for segment in range(module.noSegments + 1)
                )
            else:
                capSegmentVarSum = sum(
                    pyM.segmentCapacityPwlcfVar[moduleName, ip, segment]
                    for segment in range(module.noSegments)
                )
            return capSegmentVarSum == commVarSum + module.initCapacity

        pyM.ConstrCapacityCommissioningPwlcf = pyomo.Constraint(
            pyM.pwlcfDesignSet, rule=capacityCommissioningPwlcfConstr
        )

    def declareSos2PwlcfConstr(self, pyM):
        """Declare SOS2 constraints for pwlcf model."""

        def sos2rule(pyM, module_name, ip):
            return [
                pyM.sos2PwlcfVar[module_name, ip, segment]
                for segment in range(self.modulesDict[module_name].noSegments + 1)
            ]

        pyM.sos2Constr = pyomo.SOSConstraint(
            pyM.pwlcfDesignSet,
            rule=sos2rule,
            sos=2,
        )

    def declarePwlfPyomo(self, esM, pyM):
        """https://pyomo.readthedocs.io/en/latest/pyomo_modeling_components/Expressions.html#piecewise-linear-expressions."""
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
                if module.comp.processedLocationalEligibility.loc[loc] == 1
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

    def getObjectiveFunctionContribution(self, esM, pyM):  # noqa D102
        return self.getEconomicsPwlcf(esM, pyM)

    def getEconomicsPwlcf(
        self,
        esM,
        pyM,
        getOptValue=False,
        getOptValueCostType=CostType.TAC,
    ):
        """Get the economic contribution to the cost function of pwlcf components.

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
                # Read from the modeling class' raw results dict, not from the optimization
                # summary: the summary is a view of that dict and must not be used as a
                # derivation input (see :meth:`_commissioningResults`). Reindexing to the
                # sorted locations reproduces the summary's fixed column set, on which the
                # ``len(commis[ip])`` and per-location lookups below rely.
                commis = {
                    ip: self._commissioningResults(
                        esM, moduleName, esM.investmentPeriodNames[ip]
                    ).reindex(sorted(esM.locations))
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
                        if getOptValueCostType == CostType.NPV:
                            cost_results[ip].loc[moduleName, _loc] = (
                                cContrSum
                                * utils.discountFactor(esM, ip, moduleName, _loc)
                            )
                        elif getOptValueCostType == CostType.TAC:
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
        """Extract the cost contribution from a specified component and for a specified commissioning year.

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
                    elif use_sos2:
                        if not getOptValue:
                            totalCost = sum(
                                module.linEtlParameter["totalCost"].loc[segment]
                                * pyM.sos2PwlcfVar[moduleName, ip, segment]
                                for segment in range(module.noSegments + 1)
                            )
                        else:
                            totalCost = sum(
                                module.linEtlParameter["totalCost"].loc[segment]
                                * pyM.sos2PwlcfVar[moduleName, ip, segment].value
                                for segment in range(module.noSegments + 1)
                            )
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

    # Attribute of a module's component holding its plant unit, plus the suffix appended to it,
    # per modeling class. Used for the unit of the knowledgeStock row.
    _plantUnitAttribute = {
        ComponentAbbreviation.CONVERSION: ("physicalUnit", ""),
        ComponentAbbreviation.SOURCE_SINK: ("commodityUnit", ""),
        ComponentAbbreviation.STORAGE: ("commodityUnit", "*h"),
        ComponentAbbreviation.TRANSMISSION: ("commodityUnit", ""),
    }

    def setOptimalValues(self, esM, pyM):
        """Derive the pwlcf results and write them into the modeling classes.

        Mirrors the pipeline of the component modeling classes (see
        :meth:`fine.component.ComponentModel.setOptimalValues`): the results are derived once
        into frames, those frames are published into the modeling classes' raw results dict -
        the single source of truth - and the optimization summary is only a *view* of them.
        Nothing downstream may derive values from the summary.

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        results = self._deriveResults(esM, pyM)
        self._publishResults(esM, results)
        self._buildOptimizationSummary(esM, results)

    def _deriveResults(self, esM, pyM):
        """Derive the pwlcf result frames from the solved model.

        :param esM: EnergySystemModel instance.
        :param pyM: pyomo ConcreteModel.

        :return: ``{ipName: {property: DataFrame}}``, each frame indexed by module name with one
            column per location. Properties are ``TAC_ETL``/``TAC_EOS``,
            ``NPVcontribution_*``, ``invest_*`` and, for etl modules, ``knowledgeStock_ETL``.
        :rtype: dict
        """
        tac = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType=CostType.TAC
        )
        npv = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType=CostType.NPV
        )
        invest = self.getEconomicsPwlcf(
            esM, pyM, getOptValue=True, getOptValueCostType="invest"
        )

        locations = list(esM.locations)
        results = {}
        # knowledge stock of the previous investment period, per module. Formerly read back
        # from the summary of that period - it is kept here so the summary stays a pure view.
        knowledgeStockLastIp = {}
        for ip in esM.investmentPeriods:
            ipName = esM.investmentPeriodNames[ip]
            rows = {}

            def addRow(prop, moduleName, values):
                series = (
                    values.reindex(locations)
                    if isinstance(values, pd.Series)
                    else pd.Series(values, index=locations, dtype=float)
                )
                rows.setdefault(prop, {})[moduleName] = series

            for moduleName, module in self.modulesDict.items():
                pwlcfType = "ETL" if module.pwlcf_type == "etl" else "EOS"
                addRow(f"TAC_{pwlcfType}", moduleName, tac[ip].loc[moduleName])
                addRow(
                    f"NPVcontribution_{pwlcfType}", moduleName, npv[ip].loc[moduleName]
                )
                addRow(f"invest_{pwlcfType}", moduleName, invest[ip].loc[moduleName])
                if pwlcfType == "ETL":
                    knowledgeStock = self._deriveKnowledgeStock(
                        esM, pyM, ip, moduleName, knowledgeStockLastIp
                    )
                    knowledgeStockLastIp[moduleName] = knowledgeStock
                    addRow("knowledgeStock_ETL", moduleName, knowledgeStock)

            results[ipName] = {
                prop: pd.DataFrame(moduleRows).T for prop, moduleRows in rows.items()
            }
        return results

    def _deriveKnowledgeStock(self, esM, pyM, ip, moduleName, knowledgeStockLastIp):
        """Knowledge stock of an etl module in one investment period.

        The commissioning it builds on is read from the modeling class' raw results dict; the
        optimization summary is a view of that dict and must not be used as an input here.

        :param ip: investment period index.
        :param knowledgeStockLastIp: ``{module: knowledge stock}`` of the preceding period,
            filled by :meth:`_deriveResults` as it walks the investment periods in order.

        :return: knowledge stock per location (or a scalar, broadcast by the caller).
        :rtype: pandas.Series or float
        """
        if pyomo_pwlf:
            return pyM.totalCapacity[moduleName, ip].value

        module = self.modulesDict[moduleName]
        if ip == 0:
            stockCapacityStartYear = module.comp.stockCapacityStartYear
            knowledgeStockLast = stockCapacityStartYear + (
                (module.initCapacity - stockCapacityStartYear.sum())
                / module.comp.processedLocationalEligibility.sum().sum()
            )
        else:
            knowledgeStockLast = knowledgeStockLastIp[moduleName]

        commissioning = self._commissioningResults(
            esM, moduleName, esM.investmentPeriodNames[ip]
        )
        return knowledgeStockLast + commissioning

    @staticmethod
    def _commissioningResults(esM, moduleName, ipName):
        """Commissioning of a module's component, read from the raw results dict.

        The single place this module obtains commissioning from. It is deliberately *not* read
        back from the optimization summary: the summary is a pure view of
        ``ComponentModel._rawResults`` (see
        :meth:`fine.component.ComponentModel.getResultSummaryDict`), so deriving values from it
        would make this module depend on a presentation artifact - and on the summary having
        been assembled first, which the ``EnergySystemModel.optimize`` call order happens to
        guarantee but nothing enforces.

        :param esM: EnergySystemModel instance.
        :param moduleName: name of the module's component.
        :param ipName: investment period name (key into ``_rawResults1dim``).

        :return: commissioning per location.
        :rtype: pandas.Series
        """
        model = esM.componentModelingDict[esM.componentNames[moduleName]]
        return model._rawResults1dim[ipName]["commissioning"].loc[moduleName]

    def _resultUnit(self, esM, prop):
        """Unit of a pwlcf result row: a fixed string, or a ``module -> unit`` callable.

        :rtype: string or callable
        """
        if prop == "knowledgeStock_ETL":
            return self._plantUnit
        if prop.startswith("TAC_"):
            return "[" + esM.costUnit + "/a]"
        return "[" + esM.costUnit + "]"

    def _plantUnit(self, moduleName):
        """Plant unit of a module's component, e.g. ``"[GW]"``.

        :rtype: string
        """
        comp = self.modulesDict[moduleName].comp
        attribute, suffix = self._plantUnitAttribute[comp.modelingClass().abbrvName]
        return "[" + getattr(comp, attribute) + suffix + "]"

    def _modulesOfModel(self, model, pwlcfType=None):
        """Module names of a modeling class, optionally restricted to one pwlcf type.

        :param pwlcfType: ``"etl"``, ``"eos"`` or ``None`` for both.

        :rtype: list
        """
        return [
            comp
            for comp in model.componentsDict.keys()
            if comp in self.modulesDict.keys()
            if pwlcfType is None or self.modulesDict[comp].pwlcf_type == pwlcfType
        ]

    def _publishResults(self, esM, results):
        """Hand the derived frames to the modeling classes' raw results dict.

        This keeps that dict the single source of truth: the summary rows built below and the
        xarray/netCDF export (via
        :meth:`fine.component.ComponentModel.getResultSummaryDict`) are both views of the very
        same frames.

        :param esM: EnergySystemModel instance.
        :param results: as returned by :meth:`_deriveResults`.
        """
        for model in esM.componentModelingDict.values():
            modules = self._modulesOfModel(model)
            if not modules:
                continue
            for ipName in esM.investmentPeriodNames:
                rows = []
                for prop, frame in results[ipName].items():
                    ownModules = [name for name in modules if name in frame.index]
                    if ownModules:
                        rows.append(
                            (prop, frame.loc[ownModules], self._resultUnit(esM, prop))
                        )
                model.registerExtraSummaryRows(ipName, rows)
                self._foldContributionsIntoRawResults(
                    model, ipName, results[ipName], modules
                )

    @staticmethod
    def _foldContributionsIntoRawResults(model, ipName, results_ip, modules):
        """Add the pwlcf contributions to the components' base cost frames in the results dict.

        A module's pwlcf costs come on top of the costs its component derived itself, for the
        ``TAC``, ``NPVcontribution`` and ``invest`` rows alike. The summary rows are rebuilt
        from these frames afterwards, so folding here (rather than into the summary) keeps both
        views consistent.

        :param model: modeling class instance holding the components.
        :param ipName: investment period name.
        :param results_ip: ``{property: frame}`` of that investment period.
        :param modules: module names belonging to this modeling class.
        """
        rawResults_ip = model._rawResults.get(ipName)
        if rawResults_ip is None:
            return
        for base in ("TAC", "NPVcontribution", "invest"):
            parts = [
                results_ip[f"{base}_{pwlcfType}"]
                for pwlcfType in ("ETL", "EOS")
                if f"{base}_{pwlcfType}" in results_ip
            ]
            target = rawResults_ip.get(base)
            if not parts or target is None:
                continue
            contribution = pd.concat(parts).reindex(columns=target.columns)
            comps = [
                comp
                for comp in modules
                if comp in contribution.index and comp in target.index
            ]
            if not comps:
                continue
            target = target.copy()
            target.loc[comps] = (
                target.loc[comps].astype(float).fillna(0).values
                + contribution.loc[comps].fillna(0).values
            )
            rawResults_ip[base] = target

    def _buildOptimizationSummary(self, esM, results):
        """Append the pwlcf rows to the components' optimization summary, as a pure view.

        The etl and eos rows are added as their own summary rows, and the base ``TAC``,
        ``NPVcontribution`` and ``invest`` rows are rewritten from the raw results dict, into
        which :meth:`_foldContributionsIntoRawResults` has already folded the pwlcf shares.

        :param esM: EnergySystemModel instance.
        :param results: as returned by :meth:`_deriveResults`.
        """
        for model in esM.componentModelingDict.values():
            optSummary = model._optSummary
            for ipName in esM.investmentPeriodNames:
                for pwlcfType in ("etl", "eos"):
                    modules = self._modulesOfModel(model, pwlcfType)
                    if not modules:
                        continue
                    optSummary[ipName] = pd.concat(
                        [
                            optSummary[ipName],
                            self._summaryRows(esM, results[ipName], modules),
                        ],
                        axis=0,
                    ).sort_index()

                # rewrite the base rows from the (already folded) raw results dict
                modules = self._modulesOfModel(model)
                for base in ("TAC", "NPVcontribution", "invest"):
                    frame = model._rawResults[ipName].get(base)
                    if frame is None:
                        continue
                    comps = [comp for comp in modules if comp in frame.index]
                    if not comps:
                        continue
                    optSummary[ipName].loc[comps, base, :] = (
                        frame.loc[comps]
                        .reindex(columns=optSummary[ipName].columns)
                        .values
                    )
            model.optSummary = optSummary[esM.startYear]

    def _summaryRows(self, esM, results_ip, modules):
        """Build the pwlcf summary rows of one investment period from the derived frames.

        :param results_ip: ``{property: frame}`` of that investment period.
        :param modules: module names to build the rows for.

        :return: frame with the summary's ``(Component, Property, Unit)`` MultiIndex.
        :rtype: pandas.DataFrame
        """
        index = pd.MultiIndex.from_tuples(
            [
                (moduleName, prop, self._resultUnitFor(esM, prop, moduleName))
                for moduleName in modules
                for prop in results_ip
                if moduleName in results_ip[prop].index
            ],
            names=["Component", "Property", "Unit"],
        )
        rows = pd.DataFrame(index=index, columns=list(esM.locations)).sort_index()
        for moduleName, prop, unit in index:
            rows.loc[(moduleName, prop, unit)] = results_ip[prop].loc[moduleName]
        return rows

    def _resultUnitFor(self, esM, prop, moduleName):
        """:meth:`_resultUnit` resolved for one module.

        :rtype: string
        """
        unit = self._resultUnit(esM, prop)
        return unit(moduleName) if callable(unit) else unit
