"""Economic contributions of the component models.

Split out of :class:`fine.component.ComponentModel` so that the class keeps the model
formulation (sets, variables, constraints) as its subject. The methods are unchanged and
still run as methods of the modeling class - they read ``self.componentsDict`` and the
solved pyomo variables - they merely live in their own module.
"""

import math

import pandas as pd

from fine import utils
from fine.enums import CostType, FncType


class ComponentEconomicsMixin:
    """Design and operation cost contributions of a component model.

    Mixed into :class:`fine.component.ComponentModel`; not meant to be used on its own.
    """

    def getEconomicsDesign(
        self,
        pyM,
        esM,
        factorNames,
        lifetimeAttr,
        varName,
        divisorName="",
        QPfactorNames=None,
        QPdivisorNames=None,
        getOptValue=False,
        getOptValueCostType=CostType.TAC,
    ):
        """Set design dependent cost equations for the individual components. The equations will be set
        for all components of a modeling class and all locations.

        **Required arguments**

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package

        :param factorNames: Strings of the parameters that have to be multiplied within the equation.
            (e.g. ['processedInvestPerCapacity'] to multiply the capacity variable with the investment per each capacity unit).
        :type factorNames: list of strings

        :param varName: String of the variable that has to be multiplied within the equation (e.g. 'cap' for capacity variable).
        :type varName: string

        :param divisorName: String of the variable that is used as a divisor within the equation (e.g. 'CCF').
            If the divisorName is an empty string, there is no division within the equation.
            |br| * the default value is "".
        :type divisorName: string

        :param QPfactorNames: Strings of the parameters that have to be multiplied when quadratic programming is used. (e.g. ['processedQPcostScale'])
        :type QPfactorNames: list of strings

        :param QPdivisorNames: Strings of the parameters that have to be used as divisors when quadratic programming is used. (e.g. ['QPbound'])
        :type QPdivisorNames: list of strings

        :param getOptValue: Boolean that defines the output of the function:

            - True: Return the optimal cost values.
            - False: Return the cost equation.

            |br| * the default value is False.
        :type getoptValue: boolean

        :param getOptValueCostType: the cost type can either be TAC (total anualized costs) or NPV (net present value)
            |br| * the default value is None.
        :type getOptValueCostType: string
        """
        if QPfactorNames is None:
            QPfactorNames = []
        if QPdivisorNames is None:
            QPdivisorNames = []

        try:
            getOptValueCostType = CostType(getOptValueCostType)
        except ValueError as exc:
            raise ValueError("The cost types must be 'TAC' or 'NPV'.") from exc

        var = getattr(pyM, varName + "_" + self.abbrvName)
        if esM.stochasticModel:
            if getOptValue:
                cost_results = {}
                for ip in esM.investmentPeriods:
                    cost_results[ip] = pd.DataFrame()
                for loc, compName, ip in var:
                    if ip not in esM.investmentPeriods:
                        continue
                    cost_results[ip].loc[compName, loc] = self.getLocEconomicsDesign(
                        pyM,
                        esM,
                        factorNames,
                        varName,
                        loc,
                        compName,
                        ip,
                        divisorName,
                        QPfactorNames,
                        QPdivisorNames,
                        getOptValue,
                    )
                return cost_results
            return sum(
                self.getLocEconomicsDesign(
                    pyM,
                    esM,
                    factorNames,
                    varName,
                    loc,
                    compName,
                    ip,
                    divisorName,
                    QPfactorNames,
                    QPdivisorNames,
                    getOptValue,
                )
                for loc, compName, ip in var
            )
        # Components can have different investPerCapacity in different years.
        # The capex contribution however only depends on the capex of the
        # commissioning year. Therefore, we initialize a dataframe with index and
        # columns of the investment periods. The rows describe the commissioning
        # years, e.g. a component build in year 2 but with a lifetime of three
        # years would have entries for df.loc[2,2:5]. Afterwards we
        # sum the contributions per column, multiply it with the annuity
        # present value factor to get the npv of the component for
        # different investPerCapacity and several ip for commissioning

        # initialize dict with (loc,comp) as key and df as values
        costContribution = {}
        locCompNamesCombinations = list(set([(x[0], x[1]) for x in var.get_values()]))
        componentYears = {}

        for loc, compName in locCompNamesCombinations:
            # get all years of component with location (also stock years)
            componentYears[compName] = (
                esM.getComponentAttribute(compName, "processedStockYears")
                + esM.investmentPeriods
            )

            costContribution[(loc, compName)] = {
                (y, i): 0
                for y in componentYears[compName]
                for i in esM.investmentPeriods
            }

        # fill the dataframes (per location and compName) with the cost
        # contributions depending on the commissioning year (index) and the
        # investment period (columns)
        for loc, compName, commisYear in var:
            ipEconomicLifetime = getattr(
                esM.getComponent(compName), "ipEconomicLifetime"
            )[loc]
            ipTechnicalLifetime = getattr(
                esM.getComponent(compName), "ipTechnicalLifetime"
            )[loc]

            (fullCostIntervals, costInLastEconInterval, costInLastTechInterval) = (
                utils.getParametersForUnevenLifetimes(compName, loc, lifetimeAttr, esM)
            )

            # calculation of the annuity
            annuity = self.getLocEconomicsDesign(
                pyM,
                esM,
                factorNames,
                varName,
                loc,
                compName,
                commisYear,
                divisorName,
                QPfactorNames,
                QPdivisorNames,
                getOptValue,
            )

            # write costs into dataframe
            # a) costs for complete intervals
            for i in range(commisYear, commisYear + fullCostIntervals):
                costContribution[(loc, compName)][(commisYear, i)] = (
                    annuity
                    * utils.annuityPresentValueFactor(
                        esM, compName, loc, esM.investmentPeriodInterval
                    )
                )

            # b) costs for last economic interval
            # example: interval 5, economic lifetime 7, technical lifetime 10
            # last interval has costs only in year 5 and 6
            if costInLastEconInterval:
                # calculate portion of interval with economic lifetime
                # example: interval 5, economic lifetime 7 leads to partlyCostInLastEconomicInterval of 0.4
                partlyCostInLastEconomicInterval = (
                    ipEconomicLifetime % 1
                ) * esM.investmentPeriodInterval
                costContribution[(loc, compName)][
                    (commisYear, commisYear + fullCostIntervals)
                ] = annuity * utils.annuityPresentValueFactor(
                    esM, compName, loc, partlyCostInLastEconomicInterval
                )

            # c) costs for last technical interval due to additionally required capacity after technical lifetime is over
            # example: interval 5, economic lifetime 5, technical lifetime 7 and is ceiled to 10
            # extra costs for years 8 and 9
            if costInLastTechInterval and ipTechnicalLifetime % 1 != 0:
                partlyCostInLastTechnicalInterval = (
                    1 - (ipTechnicalLifetime % 1)
                ) * esM.investmentPeriodInterval
                if commisYear + math.ceil(ipTechnicalLifetime) - 1 in [
                    k[1] for k in costContribution[(loc, compName)].keys()
                ]:
                    costContribution[(loc, compName)][
                        (
                            commisYear,
                            commisYear + math.ceil(ipTechnicalLifetime) - 1,
                        )
                    ] = costContribution[(loc, compName)][
                        (
                            commisYear,
                            commisYear + math.ceil(ipTechnicalLifetime) - 1,
                        )
                    ] + annuity * (
                        utils.annuityPresentValueFactor(
                            esM,
                            compName,
                            loc,
                            partlyCostInLastTechnicalInterval,
                        )
                        / (1 + esM.getComponent(compName).interestRate[loc])
                        ** (
                            esM.investmentPeriodInterval
                            - partlyCostInLastTechnicalInterval
                        )
                    )

        # create dictionary with ip as key and cost contribution as value
        if getOptValue:
            cost_results = {ip: pd.DataFrame() for ip in esM.investmentPeriods}
            for loc, compName in locCompNamesCombinations:
                for ip in esM.investmentPeriods:
                    cContrSum = sum(
                        [
                            costContribution[(loc, compName)].get((y, ip), 0)
                            for y in componentYears[compName]
                        ]
                    )
                    if getOptValueCostType == CostType.NPV:
                        cost_results[ip].loc[compName, loc] = (
                            cContrSum * utils.discountFactor(esM, ip, compName, loc)
                        )
                    elif getOptValueCostType == CostType.TAC:
                        cost_results[ip].loc[compName, loc] = (
                            cContrSum
                            / utils.annuityPresentValueFactor(
                                esM, compName, loc, esM.investmentPeriodInterval
                            )
                        )
            return cost_results
        if esM.annuityPerpetuity:
            # the last investment period gets the perpetuity cost
            # contribution, implying the system design and operation
            # will remain constant after the time frame of the
            # transformation pathway.
            for loc, compName in costContribution.keys():  # noqa: PLC0206
                for y in componentYears[compName]:
                    costContribution[(loc, compName)][
                        (y, esM.investmentPeriods[-1])
                    ] = costContribution[(loc, compName)][
                        (y, esM.investmentPeriods[-1])
                    ] / (
                        utils.annuityPresentValueFactor(
                            esM, compName, loc, esM.investmentPeriodInterval
                        )
                        * esM.getComponent(compName).interestRate[loc]
                    )
        return sum(
            sum(
                [
                    costContribution[(loc, compName)].get((y, ip), 0)
                    for y in componentYears[compName]
                ]
            )
            * utils.discountFactor(esM, ip, compName, loc)
            for loc, compName, ip in var
            if ip in esM.investmentPeriods
        )

    def getLocEconomicsDesign(  # noqa: PLR0911
        self,
        pyM,
        esM,
        factorNames,
        varName,
        loc,
        compName,
        ip,
        divisorName="",
        QPfactorNames=None,
        QPdivisorNames=None,
        getOptValue=False,
    ):
        """Set time-independent equation specified for one component in one location in one investment period.

        **Required arguments:**

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package

        :param factorNames: Strings of the parameters that have to be multiplied within the equation.
            (e.g. ['processedInvestPerCapacity'] to multiply the capacity variable with the investment per each capacity unit).
        :type factorNames: list of strings

        :param varName: String of the variable that has to be multiplied within the equation (e.g. 'cap' for capacity variable).
        :type varName: string

        :param loc: String of the location for which the equation should be set up.
        :type loc: string

        :param compName: String of the component name for which the equation should be set up.
        :type compName: string

        **Default arguments:**

        :param ip: investment period
        :type ip: int

        :param divisorName: String of the variable that is used as a divisor within the equation (e.g. 'CCF').
            If the divisorName is an empty string, there is no division within the equation.
            |br| * the default value is ''.
        :type divisorName: string

        :param QPfactorNames: Strings of the parameters that have to be multiplied when quadratic programming is used. (e.g. ['processedQPcostScale'])
        :type QPfactorNames: list of strings

        :param QPdivisorNames: Strings of the parameters that have to be used as divisors when quadratic programming is used. (e.g. ['QPbound'])
        :type QPdivisorNames: list of strings

        :param getOptValue: Boolean that defines the output of the function:

            - True: Return the optimal value.
            - False: Return the equation.

            |br| * the default value is False.
        :type getoptValue: boolean
        """
        if QPfactorNames is None:
            QPfactorNames = []
        if QPdivisorNames is None:
            QPdivisorNames = []

        # negative ip (historical data) older than technical lifetime
        # round or ceil technical lifetime to interval
        if self.componentsDict[compName].floorTechnicalLifetime:
            roundedTechnicalLifetime = math.floor(
                self.componentsDict[compName].ipTechnicalLifetime[loc]
            )
        else:
            roundedTechnicalLifetime = math.ceil(
                self.componentsDict[compName].ipTechnicalLifetime[loc]
            )
        if ip < -roundedTechnicalLifetime:
            return 0
        # years where component could have commissioning as it is within the technical
        # lifetime, but does not have commissioning
        if ip < 0 and self.componentsDict[compName].processedStockCommissioning is None:
            return 0
        if (
            ip < 0
            and self.componentsDict[compName].processedStockCommissioning is not None
        ):
            if self.componentsDict[compName].processedStockCommissioning[ip][loc] == 0:
                return 0

        var = getattr(pyM, varName + "_" + self.abbrvName)
        factors = [
            getattr(self.componentsDict[compName], factorName)[ip][loc]
            for factorName in factorNames
        ]
        divisor = (
            getattr(self.componentsDict[compName], divisorName)[ip][loc]
            if not divisorName == ""
            else 1
        )

        factor = 1.0 / divisor
        for factor_ in factors:
            factor *= factor_

        _var = var[loc, compName, ip]

        if self.componentsDict[compName].processedQPcostScale[ip][loc] == 0:
            if not getOptValue:
                return factor * _var
            return factor * _var.value
        QPfactors = [
            getattr(self.componentsDict[compName], QPfactorName)[ip][loc]
            for QPfactorName in QPfactorNames
        ]
        QPdivisors = [
            getattr(self.componentsDict[compName], QPdivisorName)[ip][loc]
            for QPdivisorName in QPdivisorNames
        ]
        QPfactor = 1
        for QPfactor_ in QPfactors:
            QPfactor *= QPfactor_
        for QPdivisor in QPdivisors:
            QPfactor /= QPdivisor
        if not getOptValue:
            return factor * _var + QPfactor * _var * _var
        return factor * _var.value + QPfactor * _var.value * _var.value

    def getEconomicsOperation(
        self,
        pyM,
        esM,
        fncType,
        factorNames,
        varName,
        dictName,
        getOptValue=False,
        getOptValueCostType=CostType.TAC,
    ):
        """Set time-dependent equations for the individual components. The equations will be set for all components of a modeling class
        and all locations as well as for each considered time step.
        In case of a two-dimensional component (e.g. a transmission component), the equations will be set for all possible connections between the
        defined locations.

        **Required arguments:**

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the components should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param fncType: Function type, either "TD" or "TimeSeries"
        :type fncType: string

        :param factorNames: Strings of the time-dependent parameters that have to be multiplied within the equation.
            (e.g. ['opexPerOperation'] to multiply the operation variable with the costs for each operation).
        :type factorNames: list of strings

        :param varName: String of the variable that has to be multiplied within the equation (e.g. 'op' for operation variable).
        :type varName: string

        :param dictName: String of the variable set (e.g. 'operationVarDict')
        :type dictName: string

        **Default arguments:**

        :param getOptValue: Boolean that defines the output of the function:

            - True: Return the optimal value.
            - False: Return the equation.

            |br| * the default value is False.
        :type getoptValue: boolean

        :param getOptValueCostType: the cost type can either be TAC (total annualized costs) or NPV (net present value)
            |br| * the default value is None.
        :type getOptValueCostType: string
        """
        try:
            getOptValueCostType = CostType(getOptValueCostType)
        except ValueError as exc:
            raise ValueError(
                "getOptValueCostType must be either 'TAC' or 'NPV'"
            ) from exc
        try:
            fncType = FncType(fncType)
        except ValueError as exc:
            raise ValueError("fncType must be either 'TD' or 'TimeSeries'") from exc
        if fncType == FncType.TIME_SERIES:
            factorName = factorNames[0]  # noqa: F841

        var = getattr(pyM, varName + "_" + self.abbrvName)
        locCompIpCombinations = list(set([(x[0], x[1], x[2]) for x in var]))
        locCompNamesCombinations = list(set([(x[0], x[1]) for x in var.get_values()]))

        if esM.stochasticModel:
            if getOptValue:
                cost_results = {}
                for ip in esM.investmentPeriods:
                    cost_results[ip] = pd.DataFrame()
                for loc, compName, ip in locCompIpCombinations:
                    if ip not in esM.investmentPeriods:
                        continue
                    cost_results[ip].loc[compName, loc] = self.getLocEconomicsOperation(
                        pyM,
                        esM,
                        fncType,
                        factorNames,
                        varName,
                        loc,
                        compName,
                        ip,
                        getOptValue,
                    )
                return cost_results
            return sum(
                self.getLocEconomicsOperation(
                    pyM,
                    esM,
                    fncType,
                    factorNames,
                    varName,
                    loc,
                    compName,
                    ip,
                    getOptValue,
                )
                for loc, compName, ip in locCompIpCombinations
            )
        # Components can have different investPerCapacity in different
        # years. The capex contribution however only depends on the capex
        # of the commissioning year. Therefore, we initialize a
        # dataframe with index and columns of the investment periods.
        # The rows describe the commissioning years,
        # e.g. a component build in year 2 but with a lifetime of three
        # years would have entries for df.loc[2,2:5]. Afterwards we
        # sum the contributions per column, multiply it with the annuity
        # present value factor to get the npv of the component for
        # different investPerCapacity and several ip for commissioning

        # initialize dict with (loc,comp) as key and df as values
        costContribution = {}
        componentYears = {}
        for loc, compName in locCompNamesCombinations:
            # get all years of component with location (also stock years)
            componentYears[compName] = (
                esM.getComponentAttribute(compName, "processedStockYears")
                + esM.investmentPeriods
            )
            costContribution[(loc, compName)] = {
                (y, i): 0
                for y in componentYears[compName]
                for i in esM.investmentPeriods
            }

        # fill the dataframes (per location and compName) with the cost
        # contributions depending on the commissioning year (index) and the
        # investment period (columns)

        locCompIpCombinations = list(set([(x[0], x[1], x[2]) for x in var]))
        for loc, compName, year in locCompIpCombinations:
            costContribution[(loc, compName)][(year, year)] = (
                self.getLocEconomicsOperation(
                    pyM,
                    esM,
                    fncType,
                    factorNames,
                    varName,
                    loc,
                    compName,
                    year,
                    getOptValue,
                )
            )

        # create dictionary with ip as key and a dataframe with
        # cost contribution per component+location as value
        if getOptValue:
            cost_results = {ip: pd.DataFrame() for ip in esM.investmentPeriods}
            for loc, compName in locCompNamesCombinations:
                for ip in esM.investmentPeriods:
                    cContrSum = sum(
                        [
                            costContribution[(loc, compName)].get((y, ip), 0)
                            for y in componentYears[compName]
                        ]
                    )
                    if getOptValueCostType == CostType.NPV:
                        cost_results[ip].loc[compName, loc] = (
                            cContrSum
                            * utils.annuityPresentValueFactor(
                                esM, compName, loc, esM.investmentPeriodInterval
                            )
                            * utils.discountFactor(esM, ip, compName, loc)
                        )
                    elif getOptValueCostType == CostType.TAC:
                        cost_results[ip].loc[compName, loc] = cContrSum
            return cost_results
        if esM.annuityPerpetuity:
            # the last investment period gets the perpetuity cost
            # contribution, implying the system design and operation
            # will remain constant after the time frame of the
            # transformation pathway.
            for loc, compName in costContribution.keys():  # noqa: PLC0206
                for y in componentYears[compName]:
                    costContribution[(loc, compName)][
                        (y, esM.investmentPeriods[-1])
                    ] = costContribution[(loc, compName)][
                        (y, esM.investmentPeriods[-1])
                    ] / (
                        utils.annuityPresentValueFactor(
                            esM, compName, loc, esM.investmentPeriodInterval
                        )
                        * esM.getComponent(compName).interestRate[loc]
                    )
        return sum(
            sum(
                [
                    costContribution[(loc, compName)].get((y, ip), 0)
                    for y in componentYears[compName]
                ]
            )
            * utils.annuityPresentValueFactor(
                esM, compName, loc, esM.investmentPeriodInterval
            )
            * utils.discountFactor(esM, ip, compName, loc)
            for loc, compName, ip in locCompIpCombinations
            if ip in esM.investmentPeriods
        )

    def getLocEconomicsOperation(
        self,
        pyM,
        esM,
        fncType,
        factorNames,
        varName,
        loc,
        compName,
        ip,
        getOptValue=False,
    ):
        """Set time-dependent cost functions for the individual components. The equations will be set for all components
        of a modeling class and all locations as well as for each considered time step.

        **Required arguments:**

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the components should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param fncType: Function type,  either "TD" or "TimeSeries
        :type fncType: string

        :param factorName: String of the time-dependent parameter that have to be multiplied within the equation.
            (e.g. 'commodityCostTimeSeries' to multiply the operation variable with the costs for each operation).
        :type factorNames: string

        :param varName: String of the variable that has to be multiplied within the equation (e.g. 'op' for operation variable).
        :type varName: string

        :param dictName: String of the variable set (e.g. 'operationVarDict')
        :type dictName: string

        :param loc: String of the location for which the equation should be set up.
        :type loc: string

        :param compName: String of the component name for which the equation should be set up.
        :type compName: string

        :param ip: investment period of transformation path analysis.
        :type ip: int

        **Default arguments:**

        :param getOptValue: Boolean that defines the output of the function:

            - True: Return the optimal value.
            - False: Return the equation.

            |br| * the default value is False.
        :type getoptValue: boolean
        """
        var = getattr(pyM, varName + "_" + self.abbrvName)

        # create new timeSet for current investment period
        timeSet_pt = [(p, t) for ip0, p, t in pyM.timeSet if ip0 == ip]

        # get factor
        if fncType == FncType.TD:
            factors = [
                getattr(self.componentsDict[compName], factorName)[ip][loc]
                for factorName in factorNames
            ]
            # TODO in no function, there is more than one factor, therefore the
            # use case of the following calculation is questioned
            # are the costs per operation calculated correctly for conversions?
            # Shouldnt there be a multiplication with the efficiency?
            factorVal = 1.0
            for factor_ in factors:
                factorVal *= factor_
            # write pd series with constant value for factornames
            mIdx = pd.MultiIndex.from_tuples(timeSet_pt, names=["Period", "TimeStep"])
            factor = pd.Series(factorVal, index=mIdx)
        elif fncType == FncType.TIME_SERIES:
            # if there is not time series, there is not cost contribution
            if getattr(self.componentsDict[compName], factorNames[0])[ip] is None:
                return 0
            factor = getattr(self.componentsDict[compName], factorNames[0])[ip][loc]

        if esM.stochasticModel:
            if not getOptValue:
                return (
                    sum(
                        factor[p, t]
                        * var[loc, compName, ip, p, t]
                        * esM.periodOccurrences[ip][p]
                        for p, t in timeSet_pt
                    )
                    / esM.numberOfYears
                )
            return (
                sum(
                    factor[p, t]
                    * var[loc, compName, ip, p, t].value
                    * esM.periodOccurrences[ip][p]
                    for p, t in timeSet_pt
                )
                / esM.numberOfYears
            )
        if not getOptValue:
            return (
                sum(
                    factor[p, t]
                    * var[loc, compName, ip, p, t]
                    * esM.periodOccurrences[ip][p]
                    for p, t in timeSet_pt
                )
                / esM.numberOfYears
            )
        return (
            sum(
                factor[p, t]
                * var[loc, compName, ip, p, t].value
                * esM.periodOccurrences[ip][p]
                for p, t in timeSet_pt
            )
            / esM.numberOfYears
        )
