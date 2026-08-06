from abc import ABCMeta, abstractmethod
from fine import utils
from fine.enums import CostType, Dimension, FncType, VarType
import fine
import warnings
import pyomo.environ as pyomo
import pandas as pd
import numpy as np
import math


class Component(metaclass=ABCMeta):
    """The Component class includes the general methods and arguments for the components which are add-able to
    the energy system model (e.g. storage component, source component, transmission component). Every of these
    components inherits from the Component class.
    """

    def __init__(
        self,
        esM,
        name,
        dimension,
        hasCapacityVariable,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=1,
        hasIsBuiltBinaryVariable=False,
        bigM=None,
        locationalEligibility=None,
        capacityMin=None,
        capacityMax=None,
        partLoadMin=None,
        sharedPotentialID=None,
        linkedQuantityID=None,
        capacityFix=None,
        commissioningMin=None,
        commissioningMax=None,
        commissioningFix=None,
        isBuiltFix=None,
        investPerCapacity=0,
        investIfBuilt=0,
        opexPerCapacity=0,
        opexIfBuilt=0,
        QPcostScale=0,
        interestRate=0.08,
        economicLifetime=10,
        technicalLifetime=None,
        yearlyFullLoadHoursMin=None,
        yearlyFullLoadHoursMax=None,
        stockCommissioning=None,
        floorTechnicalLifetime=True,
        pwlcfParameters=None,
    ):
        """Create an instance of the Component class.

        **Required arguments:**

        :param esM: energy system model to which the component should be added. Used for unit checks.
        :type esM: EnergySystemModel instance from the FINE package

        :param name: name of the component. Has to be unique (i.e. no other components with that name can
            already exist in the EnergySystemModel instance to which the component is added).
        :type name: string

        :param hasCapacityVariable: specifies if the component should be modeled with a capacity or not. Examples:

            * An electrolyzer has a capacity given in GW_electric -> hasCapacityVariable is True.
            * In the energy system, biogas can, from a model perspective, be converted into methane (and then
              used in conventional power plants which emit CO2) by getting CO2 from the environment. Thus,
              using biogas in conventional power plants is, from a balance perspective, CO2 free. This
              conversion is purely theoretical and does not require a capacity -> hasCapacityVariable
              is False.
            * A electricity cable has a capacity given in GW_electric -> hasCapacityVariable is True.
            * If the transmission capacity of a component is unlimited -> hasCapacityVariable is False.
            * A wind turbine has a capacity given in GW_electric -> hasCapacityVariable is True.
            * Emitting CO2 into the environment is not per se limited by a capacity ->
              hasCapacityVariable is False.

        :type hasCapacityVariable: boolean

        **Default arguments:**

        :param capacityVariableDomain: describes the mathematical domain of the capacity variables, if they are
            specified. By default, the domain is specified as 'continuous' and thus declares the variables as positive
            (>=0) real values. The second input option that is available for this parameter is 'discrete', which
            declares the variables as positive (>=0) integer values.
            |br| * the default value is 'continuous'
        :type capacityVariableDomain: string ('continuous' or 'discrete')

        :param capacityPerPlantUnit: capacity of one plant of the component (in the specified physicalUnit of
            the plant). The default is 1, thus the number of plants is equal to the installed capacity.
            This parameter should be specified when using a 'discrete' capacityVariableDomain.
            It can be specified when using a 'continuous' variable domain.
            |br| * the default value is 1
        :type capacityPerPlantUnit: dict of strictly positive float or strictly positive float

        :param hasIsBuiltBinaryVariable: specifies if binary decision variables should be declared for

            * each eligible location of the component, which indicates if the component is built at that location or
              not (dimension=1dim).
            * each eligible connection of the transmission component, which indicates if the component is built
              between two locations or not (dimension=2dim).

            The binary variables can be used to enforce one-time investment cost or capacity-independent
            annual operation cost. If a minimum capacity is specified and this parameter is set to True,
            the minimum capacities are only considered if a component is built (i.e. if a component is built
            at that location, it has to be built with a minimum capacity of XY GW, otherwise it is set to 0 GW).
            |br| * the default value is False
        :type hasIsBuiltBinaryVariable: boolean

        :param bigM: the bigM parameter is only required when the hasIsBuiltBinaryVariable parameter is set to
            True. In that case, it is set as a strictly positive float, otherwise it can remain a None value.
            If not None and the ifBuiltBinaryVariables parameter is set to True, the parameter enforces an
            artificial upper bound on the maximum capacities which should, however, never be reached. The value
            should be chosen as small as possible but as large as necessary so that the optimal values of the
            designed capacities are well below this value after the optimization.
            |br| * the default value is None
        :type bigM: None or strictly positive float

        :param locationalEligibility:

            * Pandas Series that indicates if a component can be built at a location (=1) or not (=0)
              (dimension=1dim) or
            * Pandas Series or DataFrame that indicates if a component can be built between two
              locations (=1) or not (=0) (dimension=2dim).

            If not specified and a maximum or fixed capacity or time series is given, the parameter will be
            set based on these inputs. If the parameter is specified, a consistency check is done to ensure
            that the parameters indicate the same locational eligibility. If the parameter is not specified,
            and also no other of the parameters is specified, it is assumed that the component is eligible in
            each location and all values are set to 1.
            This parameter is the key part for ensuring small built times of the optimization problem by avoiding the
            declaration of unnecessary variables and constraints.
            |br| * the default value is None
        :type locationalEligibility:

            * None or
            * Pandas Series with values equal to 0 and 1. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with values equal to 0 and 1. The column and row indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param capacityMin: if specified, indicates the minimum capacities. The type of this parameter depends on the
            dimension of the component: If dimension=1dim, it has to be a Pandas Series. If dimension=2dim, it has to
            be a Pandas Series or DataFrame. If binary decision variables are declared, capacityMin is only used
            if the component is built.
            |br| * the default value is None
        :type capacityMin:

            * None or
            * float or
            * int or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations. or
            * Dict with investment periods as keys and one of the options above as values.

        :param capacityMax: if specified, indicates the maximum capacities. The type of this parameter depends on the
            dimension of the component: If dimension=1dim, it has to be a Pandas Series. If dimension=2dim, it has to
            be a Pandas Series or DataFrame.
            |br| * the default value is None
        :type capacityMax:

            * None or
            * float or
            * int or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations. or
            * Dict with investment periods as keys and one of the options above as values.

        :param partLoadMin: If specified, it defines the lowest relative operation
            rate a component must maintain during operation. To still allow the component
            to be completely turned off, a binary variable is introduced for each time
            step. This enables the model to choose between zero operation or operation
            at or above the specified minimum load.
            Note: Adding these binary variables turns the problem into a MILP, which
            can significantly increase computational time.
            |br| * the default value is None
        :type partLoadMin:
            * None or
            * Float value in range ]0;1]
            * Dict with keys of investment periods and float values in range ]0;1]

        :param sharedPotentialID: if specified, indicates that the component has to share its maximum
            potential capacity with other components (e.g. due to space limitations). The shares of how
            much of the maximum potential is used have to add up to less than 100%.
            |br| * the default value is None
        :type sharedPotentialID: string

        :param linkedQuantityID: if specified, indicates that the components with the same ID are built with the same number.
            (e.g. if a vehicle with an engine is built also a storage needs to be built)
            |br| * the default value is None
        :type linkedQuantityID: string

        :param capacityFix: if specified, indicates the fixed capacities. The type of this parameter
            depends on the dimension of the component:
            * If dimension=1dim, it has to be a Pandas Series.
            * If dimension=2dim, it has to be a Pandas Series or DataFrame.
            |br| * the default value is None
        :type capacityFix:
            * None or
            * float or
            * int or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations. or
            * Dict with investment periods as keys and one of the options above as values.

        :param commissioningMin: if specified, indicates the minimum commissioning for the respective
            investment period. The type of this parameter depends on the dimension of the component:
            * If dimension=1dim, it has to be a Pandas Series.
            * If dimension=2dim, it has to be a Pandas Series or DataFrame.
            If binary decision variables are declared, commissioningMin is only used
            if the component is built.
            |br| * the default value is None
        :type commissioningMin:

            * None or
            * float or
            * int or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations. or
            * Dict with investment periods as keys and one of the options above as values.

        :param commissioningMax: if specified, indicates the maximum commissioning for the respective
            investment period. The type of this parameter depends on the dimension of the component:
            * If dimension=1dim, it has to be a Pandas Series.
            * If dimension=2dim, it has to be a Pandas Series or DataFrame.
            |br| * the default value is None
        :type commissioningMax:

            * None or
            * float or
            * int or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations. or
            * Dict with investment periods as keys and one of the options above as values.

        :param commissioningFix: if specified, indicates the fixed commissioning for the respective
            investment period. The type of this parameter depends on the dimension of the component:
            * If dimension=1dim, it has to be a Pandas Series.
            * If dimension=2dim, it has to be a Pandas Series or DataFrame.
            |br| * the default value is None
        :type commissioningFix:
            * None or
            * float or
            * int or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations. or
            * Dict with investment periods as keys and one of the options above as values.

        :param isBuiltFix: if specified, indicates fixed decisions in which or between which locations the component is
            built (i.e. sets the isBuilt binary variables). The type of this parameter
            depends on the dimension of the component:
            * If dimension=1dim, it has to be a Pandas Series.
            * If dimension=2dim, it has to be a Pandas Series or DataFrame.
            |br| * the default value is None
        :type isBuiltFix:
            * None or
            * Pandas Series with values equal to 0 and 1. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with values equal to 0 and 1. The row and column indices of the DataFrame
              have to equal the in the energy system model specified locations.

        :param investPerCapacity: describes the investment costs for one unit of the capacity. The
            invest of a component is obtained by multiplying the commissioned capacities
            of the component (in the physicalUnit of the component) with the investPerCapacity factor
            and is distributed over the components technical lifetime.
            The value has to match the unit costUnit/physicalUnit (e.g. Euro/kW).
            The investPerCapacity can either be given as

            * a float or a Pandas Series with location specific values (dimension=1dim). The cost unit in which the
              parameter is given has to match the one specified in the energy system model (e.g. Euro, Dollar,
              1e6 Euro). The value has to match the unit
              costUnit/physicalUnit (e.g. Euro/kW, 1e6 Euro/GW) or
            * a float or a Pandas Series or DataFrame with location specific values (dimension=2dim). The cost unit
              in which the parameter is given has to match the one specified in the energy system model divided by
              the specified lengthUnit (e.g. Euro/m, Dollar/m, 1e6 Euro/km). The value has to match the unit
              costUnit/(lengthUnit * physicalUnit) (e.g. Euro/(kW * m), 1e6 Euro/(GW * km))
            * a dictionary with years as keys (past years which had stock commissioning and investment periods which
              will be optimized) and one of the two options above as values.
              e.g. {2020: 1000, 2025: 800, 2030: 750}

            |br| * the default value is 0
        :type investPerCapacity:

            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.
            * Dict with years as keys (past years with stock commissioning and investment periods which will be
              optimized) and one of the two options above as values.

        :param investIfBuilt: a capacity-independent invest which only arises in a location if a component
            is built at that location. The investIfBuilt can either be given as

            * a float or a Pandas Series with location specific values (dimension=1dim). The cost unit in which
              the parameter is given has to match the one specified in the energy system model (e.g. Euro, Dollar,
              1e6 Euro) or
            * a float or a Pandas Series or DataFrame with location specific values (dimension=2dim). The cost unit
              in which the parameter is given has to match the one specified in the energy system model divided by
              the specified lengthUnit (e.g. Euro/m, Dollar/m, 1e6 Euro/km)
            * a dictionary with years as keys (past years which had stock commissioning and investment periods which
              will be optimized) and one of the two options above as values.
              e.g. {2020: 1000, 2025: 800, 2030: 750}

            |br| * the default value is 0
        :type investIfBuilt:

            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.
            * Dict with years as keys (past years with stock commissioning and investment periods which will be
              optimized) and one of the two options above as values.

        :param opexPerCapacity: describes the operational cost for one unit of capacity. The annual operational cost,
            which are only a function of the capacity of the component (in the physicalUnit of the component) and not
            of the specific operation itself, are obtained by multiplying the commissioned capacity of the component
            at a location with the opexPerCapacity factor and is distributed over the components technical lifetime.
            The opexPerCapacity factor can either be given as

            * a float or a Pandas Series with location specific values (dimension=1dim). The cost unit in which the
              parameter is given has to match the one specified in the energy system model (e.g. Euro, Dollar,
              1e6 Euro). The value has to match the unit
              costUnit/physicalUnit (e.g. Euro/kW, 1e6 Euro/GW)  or
            * a float or a Pandas Series or DataFrame with location specific values (dimension=2dim). The cost unit
              in which the parameter is given has to match the one specified in the energy system model divided by
              the specified lengthUnit (e.g. Euro/m, Dollar/m, 1e6 Euro/km). The value has to match the unit
              costUnit/(lengthUnit * physicalUnit) (e.g. Euro/(kW * m), 1e6 Euro/(GW * km))
            * a dict with years as keys (past years which had stock commissioning and investment periods which
              will be optimized) and one of the two options above as value.
              e.g. {2020: 1000, 2025: 800, 2030: 750}

            |br| * the default value is 0
        :type opexPerCapacity:

            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.
            * Dict with years as keys (past years with stock commissioning and investment periods which will be
              optimized) and one of the two options above as values.

        :param opexIfBuilt: a capacity-independent annual operational cost which only arises in a location
            if a component is commissioned at that location. The costs are than distributed over the components
            technical lifetime.The opexIfBuilt can either be given as

            * a float or a Pandas Series with location specific values (dimension=1dim) . The cost unit in which
              the parameter is given has to match the one specified in the energy system model (e.g. Euro, Dollar,
              1e6 Euro) or
            * a float or a Pandas Series or DataFrame with location specific values (dimension=2dim). The cost unit
              in which the parameter is given has to match the one specified in the energy system model divided by
              the specified lengthUnit (e.g. Euro/m, Dollar/m, 1e6 Euro/km).
            * a dict with years as keys (past years which had stock commissioning and investment periods which
              will be optimized) and one of the two options above as value.
              e.g. {2020: 1000, 2025: 800, 2030: 750}

            |br| * the default value is 0
        :type opexIfBuilt:

            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.
            * Dict with years as keys (past years with stock commissioning and investment periods which will be
              optimized) and one of the two options above as values.

        :param QPcostScale: describes the absolute deviation of the minimum or maximum cost value from
            the average or weighted average cost value. For further information see
            Lopion et al. (2019): "Cost Uncertainties in Energy System Optimization Models:
            A Quadratic Programming Approach for Avoiding Penny Switching Effects".
            |br| * the default value is 0, i.e. the problem is not quadratic.
        :type QPcostScale:

            * float between 0 and 1
            * Pandas Series with positive (0 <= QPcostScale <= 1) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (0 <= QPcostScale <= 1) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.
            * Dict with years as keys (past years with stock commissioning and investment period which will be
              optimized) and one of the options above as value

        :param interestRate: interest rate which is considered for computing the annuities of the invest
            of the component (depreciates the invests over the economic lifetime).
            A value of 0.08 corresponds to an interest rate of 8%.
            The interest rate is currently constant for all investment periods.
            Warning: The interest must be greater than 0 if annuityPerpetuity is used in the energy system model.
            |br| * the default value is 0.08
        :type interestRate:

            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param economicLifetime: economic lifetime of the component which is considered for computing the
            annuities of the invest of the component (aka depreciation time).
            The economic lifetime is currently constant over the pathway of investment periods.
            |br| * the default value is 10
        :type economicLifetime:

            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param technicalLifetime: technical lifetime of the component which is considered for computing the
            stocks. The technical lifetime is currently constant over the pathway of investment periods.
            |br| * the default value is None
        :type technicalLifetime:

            * None or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim) or
            * Pandas DataFrame with positive (>=0) values. The row and column indices of the DataFrame have
              to equal the in the energy system model specified locations.

        :param yearlyFullLoadHoursMin: if specified, indicates the minimum yearly full load hours.
            |br| * the default value is None
        :type yearlyFullLoadHoursMin:

            * None or
            * Float with positive (>=0) value or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim).
            * Dict with years as keys and one of the two options above as values.

        :param yearlyFullLoadHoursMax: if specified, indicates the maximum yearly full load hours.
            |br| * the default value is None
        :type yearlyFullLoadHoursMax:

            * None or
            * Float with positive (>=0) value or
            * Pandas Series with positive (>=0) values. The indices of the series have to equal the in the
              energy system model specified locations (dimension=1dim) or connections between these locations
              in the format of 'loc1' + '_' + 'loc2' (dimension=2dim).
            * Dict with years as keys and one of the two options above as values.

        :param stockCommissioning: if specified, indictates historical commissioned capacities.
            The parameter describes, how much capacity was commissioned per location in which past
            investment period. The past investment period is not part of the optimized investment periods.

            * e.g. if startYear is 2020:
              {2016:pandas.series(index=["loc1","loc2"],data=[4,3]).
              2018: pandas.series(index=["loc1","loc2"],data=[1,2])}
            * e.g. if startYear is 0:
              {-4:pandas.series(index=["loc1","loc2"],data=[4,3]).
              -2: pandas.series(index=["loc1","loc2"],data=[1,2])}

            Warning: Commissioning years older than the technical lifetime from startYear will be ignored.
            |br| * the default value is None
        :type stockCommissioning:

            * None or
            * Dict with past years as keys and pandas.Series with index of locations as values

        :param modelingClass: to the Component connected modeling class.
            |br| * the default value is ModelingClass
        :type modelingClass: a class inheriting from ComponentModeling

        :param floorTechnicalLifetime: if a technical lifetime is not a multiple of the interval, this
            parameters decides if the technical lifetime is floored to the interval or ceiled to the next interval,
            by default True. The costs will then be applied to the corrected interval.

        :param pwlcfParameters: parameters used for piecewise linear cost function module. Can be used to approximate non-linear cost functions for endogenous technology learning (etl) or economies of scale (eos).
                Enables a standardized endogenous technological learning approach with a fixed learning rate. In that case, the learning is conducted in each investment period and connected throughout.
                Alternatively enables an economies of scale approach. In that case, the cost scaling is indepent in each investment period.

            Example: For etl, the cost reduce with the total cumulative installed capacity via a learning curve approach which is linearized.
            pwlcfParameters = {
                "etlParameters": {
                    "initCost": 1,
                    "learningRate": 0.18,
                    "initCapacity": 10,
                    "maxCapacity": 50,
                    "noSegments": 4,
                }
            Example: For eos, the cost of a specific component (at one location and in one investment period) decreases with increased plant size.
            pwlcfParameters = {
                "eosParameters": pd.DataFrame(data=np.array([[0,1,2,3],[0,1000, 1800, 2400],[0, 10, 18, 24]]).T, columns=["capacity", "totalInvest", "totalOpex"])
            }
        :type pwlcfParameters: dict
        """
        # Set general component data
        utils.isEnergySystemModelInstance(esM)
        self.name = name
        self.dimension = dimension
        self.modelingClass = ComponentModel

        self.hasCapacityVariable = hasCapacityVariable
        self.capacityVariableDomain = capacityVariableDomain
        self.capacityPerPlantUnit = capacityPerPlantUnit
        self.processedCapacityPerPlantUnit = (
            utils.checkAndSetInvestmentPeriodParameters(
                "capacityPerPlantUnit",
                capacityPerPlantUnit,
                esM,
            )
        )

        self.hasIsBuiltBinaryVariable = hasIsBuiltBinaryVariable
        self.bigM = bigM

        # Set design variable modeling parameters
        utils.checkDesignVariableModelingParameters(
            esM,
            capacityVariableDomain,
            hasCapacityVariable,
            self.processedCapacityPerPlantUnit,
            hasIsBuiltBinaryVariable,
            bigM,
        )

        self.partLoadMin = partLoadMin

        # Set economic data
        self.economicLifetime = utils.checkAndSetCostParameter(
            esM, name, economicLifetime, dimension, locationalEligibility
        )
        technicalLifetime = utils.checkTechnicalLifetime(
            esM, technicalLifetime, economicLifetime
        )
        self.technicalLifetime = utils.checkAndSetCostParameter(
            esM, name, technicalLifetime, dimension, locationalEligibility
        )
        utils.checkEconomicAndTechnicalLifetime(
            self.economicLifetime, self.technicalLifetime
        )
        self.floorTechnicalLifetime = utils.checkFlooringParameter(
            floorTechnicalLifetime, self.technicalLifetime, esM.investmentPeriodInterval
        )
        self.ipTechnicalLifetime = utils.checkAndSetLifetimeInvestmentPeriod(
            esM, name, self.technicalLifetime
        )
        self.ipEconomicLifetime = utils.checkAndSetLifetimeInvestmentPeriod(
            esM, name, self.economicLifetime
        )

        self.stockYears, self.processedStockYears = utils.checkStockYears(
            stockCommissioning,
            esM.startYear,
            esM.investmentPeriodInterval,
            self.ipTechnicalLifetime,
        )
        # invest per capacity
        self.investPerCapacity = investPerCapacity
        self.processedInvestPerCapacity = (
            utils.checkAndSetInvestmentPeriodCostParameter(
                esM,
                name,
                investPerCapacity,
                dimension,
                locationalEligibility,
                self.processedStockYears + esM.investmentPeriods,
            )
        )
        # invest if built
        self.investIfBuilt = investIfBuilt
        self.processedInvestIfBuilt = utils.checkAndSetInvestmentPeriodCostParameter(
            esM,
            name,
            investIfBuilt,
            dimension,
            locationalEligibility,
            self.processedStockYears + esM.investmentPeriods,
        )
        # opex per capacity
        self.opexPerCapacity = opexPerCapacity
        self.processedOpexPerCapacity = utils.checkAndSetInvestmentPeriodCostParameter(
            esM,
            name,
            opexPerCapacity,
            dimension,
            locationalEligibility,
            self.processedStockYears + esM.investmentPeriods,
        )
        # opex if built
        self.opexIfBuilt = opexIfBuilt
        self.processedOpexIfBuilt = utils.checkAndSetInvestmentPeriodCostParameter(
            esM,
            name,
            opexIfBuilt,
            dimension,
            locationalEligibility,
            self.processedStockYears + esM.investmentPeriods,
        )
        # QP costscale
        self.QPcostScale = QPcostScale
        self.processedQPcostScale = utils.checkAndSetInvestmentPeriodCostParameter(
            esM,
            name,
            QPcostScale,
            dimension,
            locationalEligibility,
            self.processedStockYears + esM.investmentPeriods,
        )
        # interest rate
        self.interestRate = utils.checkAndSetCostParameter(
            esM, name, interestRate, dimension, locationalEligibility
        )

        self.CCF = utils.getCapitalChargeFactor(
            self.interestRate,
            self.economicLifetime,
            self.processedStockYears + esM.investmentPeriods,
        )

        # Set location-specific design parameters
        self.locationalEligibility = locationalEligibility
        self.sharedPotentialID = sharedPotentialID
        if str(type(self))[-14:-2] != "Transmission":
            self.capacityMin = capacityMin
            self.capacityMax = capacityMax
            self.capacityFix = capacityFix
            self.commissioningMin = commissioningMin
            self.commissioningMax = commissioningMax
            self.commissioningFix = commissioningFix
        (
            self.processedCapacityMin,
            self.processedCapacityMax,
            self.processedCapacityFix,
        ) = utils.checkAndSetBounds(
            esM, name, "capacity", capacityMin, capacityMax, capacityFix
        )
        (
            self.processedCommissioningMin,
            self.processedCommissioningMax,
            self.processedCommissioningFix,
        ) = utils.checkAndSetBounds(
            esM,
            name,
            "commissioning",
            commissioningMin,
            commissioningMax,
            commissioningFix,
        )

        self.linkedQuantityID = linkedQuantityID

        # Set yearly full load hour parameters
        self.yearlyFullLoadHoursMin = yearlyFullLoadHoursMin
        self.yearlyFullLoadHoursMax = yearlyFullLoadHoursMax
        self.processedYearlyFullLoadHoursMin = utils.checkAndSetFullLoadHoursParameter(
            esM, name, yearlyFullLoadHoursMin, dimension, locationalEligibility
        )
        self.processedYearlyFullLoadHoursMax = utils.checkAndSetFullLoadHoursParameter(
            esM, name, yearlyFullLoadHoursMax, dimension, locationalEligibility
        )

        self.isBuiltFix = isBuiltFix

        utils.checkLocationSpecficDesignInputParams(self, esM)

        # Set quadratic capacity bounds and residual cost scale (1-cost scale)
        self.QPbound = utils.getQPbound(
            self.processedStockYears + esM.investmentPeriods,
            self.processedQPcostScale,
            self.processedCapacityMax,
            self.processedCapacityMin,
        )
        self.QPcostDev = utils.getQPcostDev(
            self.processedStockYears + esM.investmentPeriods, self.processedQPcostScale
        )

        # stock commissioning
        self.stockCommissioning = stockCommissioning
        self.processedStockCommissioning = utils.checkAndSetStock(
            self, esM, stockCommissioning
        )
        self.stockCapacityStartYear = utils.setStockCapacityStartYear(
            self, esM, dimension
        )

        # check the capacity development with stock for mismatches
        utils.checkCapacityDevelopmentWithStock(
            esM.investmentPeriods,
            self.processedCapacityMax,
            self.processedCapacityFix,
            self.processedStockCommissioning,
            self.ipTechnicalLifetime,
            self.floorTechnicalLifetime,
        )

        self.pwlcfParameters = pwlcfParameters
        self.pwlcf = None
        if pwlcfParameters and not all(
            param is None for param in pwlcfParameters.values()
        ):
            pwlcfModule = fine.expansionModules.piecewiseLinearCostFunction.PiecewiseLinearCostFunctionModule
            self.pwlcf = pwlcfModule(self, esM, **pwlcfParameters)

    def addToEnergySystemModel(self, esM):
        """Add the component to an EnergySystemModel instance (esM). If the respective component class is not already in
        the esM, it is added as well.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance
        """
        esM.isTimeSeriesDataClustered = False
        if self.name in esM.componentNames:
            if (
                esM.componentNames[self.name] == self.modelingClass.__name__
                and esM.verboseLogLevel < 2
            ):
                warnings.warn(
                    "Component identifier "
                    + self.name
                    + " already exists. Data will be overwritten."
                )
            elif esM.componentNames[self.name] != self.modelingClass.__name__:
                raise ValueError("Component name " + self.name + " is not unique.")
        else:
            esM.componentNames.update({self.name: self.modelingClass.__name__})
        mdl = self.modelingClass.__name__
        if mdl not in esM.componentModelingDict:
            esM.componentModelingDict.update({mdl: self.modelingClass()})
        esM.componentModelingDict[mdl].componentsDict.update({self.name: self})

        if self.sharedPotentialID is not None:
            for ip in esM.investmentPeriods:
                for loc in self.processedLocationalEligibility.index:
                    if self.processedCapacityMax[ip][loc] != 0:
                        esM.sharedPotentialDict.setdefault(
                            (self.sharedPotentialID, loc, ip), []
                        ).append(self.name)

        if self.pwlcf is not None:
            pwlcfModel = fine.expansionModules.piecewiseLinearCostFunction.PiecewiseLinearCostFunctionModel
            if not hasattr(esM, "pwlcfModel"):
                esM.pwlcfModel = pwlcfModel()
            esM.pwlcfModel.modulesDict.update({self.name: self.pwlcf})

    def prepareTSAInput(self, rate, rateName, rateWeight, weightDict, data, ip):
        """Format the time series data of a component to fit the requirements of the time series aggregation package and
        return a list of formatted data.

        :param rate: a fixed/maximum/minimum operation time series or None
        :type rate: Pandas DataFrame or None

        :param rateName: name of the time series (to ensure uniqueness if a component has multiple relevant time series)
        :type rateName: string

        :param rateWeight: weight of the time series in the clustering process
        :type rateWeight: positive float (>=0)

        :param weightDict: dictionary to which the weight is added
        :type weightDict: dict

        :param data: list to which the formatted data is added
        :type data: list of Pandas DataFrames

        :param ip: investment period of transformation path analysis.
        :type ip: int

        :return: data
        :rtype: Pandas DataFrame
        """
        # rate can be passed as a dict with investment periods
        if isinstance(rate, dict):
            rate = rate[ip]
        else:
            pass

        data_ = rate
        if data_ is not None:
            data_ = data_.copy()
            uniqueIdentifiers = [self.name + rateName + loc for loc in data_.columns]
            data_.rename(
                columns={loc: self.name + rateName + loc for loc in data_.columns},
                inplace=True,
            )
            (
                weightDict.update({id: rateWeight for id in uniqueIdentifiers}),
                data.append(data_),
            )
        return weightDict, data

    def getTSAOutput(self, rate, rateName, data, ip):
        """Return a reformatted time series data after applying time series aggregation, if the original time series
        data is not None.

        :param rate: Full (unclustered) time series data or None
        :type rate: Pandas DataFrame or None

        :param rateName: name of the time series (to ensure uniqueness if a component has multiple relevant time series)
        :type rateName: string

        :param data: Pandas DataFrame with the clustered time series data of all components in the energy system
        :type data: Pandas DataFrame

        :param ip: investment period of transformation path analysis.
        :type ip: int

        :return: reformatted data or None
        :rtype: Pandas DataFrame
        """
        if rate is None:
            return None
        if isinstance(rate, dict):
            if rate[ip] is not None:
                uniqueIdentifiers = [
                    self.name + rateName + loc for loc in rate[ip].columns
                ]
                data_ = data[uniqueIdentifiers].copy(deep=True)
                data_.rename(
                    columns={
                        self.name + rateName + loc: loc for loc in rate[ip].columns
                    },
                    inplace=True,
                )
            else:
                return None
        elif isinstance(rate, pd.DataFrame):
            uniqueIdentifiers = [self.name + rateName + loc for loc in rate.columns]
            data_ = data[uniqueIdentifiers].copy(deep=True)
            data_.rename(
                columns={self.name + rateName + loc: loc for loc in rate.columns},
                inplace=True,
            )
        else:
            raise ValueError(f"Wrong type for rate of '{self.name}': {type(rate)}")
        return data_

    @abstractmethod
    def setTimeSeriesData(self, hasTSA):
        """Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises). Sets
        the time series data of a component (either the full time series if hasTSA is false or the aggregated
        time series if hasTSA is True).

        :param hasTSA: indicates if time series aggregation should be considered for modeling
        :type hasTSA: boolean
        """
        raise NotImplementedError

    @abstractmethod
    def getDataForTimeSeriesAggregation(self, ip):
        """Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises). Get
        all time series data of a component for time series aggregation.

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """
        raise NotImplementedError

    @abstractmethod
    def setAggregatedTimeSeriesData(self, data, ip):
        """Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises). Set
        aggregated time series data after applying time series aggregation.

        :param data: time series data
        :type data: Pandas DataFrame

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """
        raise NotImplementedError


class ComponentModel(metaclass=ABCMeta):
    """The ComponentModel class provides the general methods used for modeling the components.
    Every model class of the several component technologies inherits from the ComponentModel class.
    Within the ComponentModel class, general valid sets, variables and constraints are declared.
    """

    def __init__(self):
        """Create a ComponentModel class instance."""
        self.abbrvName = ""
        self.dimension = ""
        self.componentsDict = {}
        self._capacityVariablesOptimum = {}
        self._commissioningVariablesOptimum = {}
        self._decommissioningVariablesOptimum = {}
        self._isBuiltVariablesOptimum = {}
        self._optSummary = {}
        # Raw results dict (single source of truth for the optimization summary and for the
        # xarray/netCDF export accessors), filled by extractRawResults/deriveEconomics during
        # optimize(). It is deliberately NOT restored when an esM is read back from netCDF, so
        # it stays empty until the model is solved.
        self._rawResults = {}
        self._rawResults1dim = {}
        # Additional summary rows contributed by expansion modules that run after
        # setOptimalValues (currently the piecewise linear cost function, see
        # registerExtraSummaryRows). Keyed by investment period name.
        self._extraSummaryRows = {}

    ####################################################################################################################
    #                           Functions for declaring design and operation variables sets                            #
    ####################################################################################################################

    def declareCommissioningVarSet(self, pyM, esM):
        """Declare set for commissioning variables in the pyomo object for a modeling class.

        The commissioning variable must be set for past investment periods
        (stock commissioning) and future/optimized investment periods

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def declareCommisVarSet(pyM):
            return (
                (loc, compName, ip)
                for compName, comp in compDict.items()
                for loc in comp.processedLocationalEligibility.index
                for ip in comp.processedStockYears + esM.investmentPeriods
                if comp.processedLocationalEligibility[loc] == 1
                and comp.hasCapacityVariable
            )

        setattr(
            pyM,
            "designCommisVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareCommisVarSet),
        )

    def declareDesignVarSet(self, pyM, esM):
        """Declare set for capacity variables in the pyomo object for a modeling class.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def declareDesignVarSet(pyM):
            return (
                (loc, compName, ip)
                for compName, comp in compDict.items()
                for loc in comp.processedLocationalEligibility.index
                for ip in esM.investmentPeriods
                if comp.processedLocationalEligibility[loc] == 1
                and comp.hasCapacityVariable
            )

        setattr(
            pyM,
            "designDimensionVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareDesignVarSet),
        )

    def declareLocationComponentSet(self, pyM):
        """Declare set with location and component in the pyomo object for a modeling class.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def initLocationComponentSet(pyM):
            return (
                (loc, compName)
                for compName, comp in compDict.items()
                for loc in comp.processedLocationalEligibility.index
                if comp.processedLocationalEligibility[loc] == 1
                and comp.hasCapacityVariable
            )

        setattr(
            pyM,
            "DesignLocationComponentVarSet_" + abbrvName,
            pyomo.Set(dimen=2, initialize=initLocationComponentSet),
        )

    def declarePathwaySets(self, pyM, esM):
        """Declare set for capacity development in the pyomo object for a modeling class.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def initDesignDevelopmentSet(pyM):
            return (
                (loc, compName, ip)
                for compName, comp in compDict.items()
                for loc in comp.processedLocationalEligibility.index
                for ip in esM.investmentPeriods[:-1]
                if comp.processedLocationalEligibility[loc] == 1
                and comp.hasCapacityVariable
            )

        setattr(
            pyM,
            "designDevelopmentVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=initDesignDevelopmentSet),
        )

    def declareContinuousDesignVarSet(self, pyM):
        """Declare set for continuous number of installed components in the pyomo object for a modeling class.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        abbrvName = self.abbrvName

        def declareContinuousDesignVarSet(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in getattr(
                    pyM, "designDimensionVarSet_" + abbrvName
                )
            )

        setattr(
            pyM,
            "continuousDesignDimensionVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareContinuousDesignVarSet),
        )

    def declareDiscreteDesignVarSet(self, pyM):
        """Declare set for discrete number of installed components in the pyomo object for a modeling class.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def declareDiscreteDesignVarSet(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in getattr(
                    pyM, "designDimensionVarSet_" + abbrvName
                )
                if compDict[compName].capacityVariableDomain == "discrete"
            )

        setattr(
            pyM,
            "discreteDesignDimensionVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareDiscreteDesignVarSet),
        )

    def declareDesignDecisionVarSet(self, pyM):
        """Declare set for design decision variables in the pyomo object for a modeling class.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        def declareDesignDecisionVarSet(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in getattr(pyM, "designCommisVarSet_" + abbrvName)
                if compDict[compName].hasIsBuiltBinaryVariable
            )

        setattr(
            pyM,
            "designDecisionVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareDesignDecisionVarSet),
        )

    def declareOpVarSet(self, esM, pyM):
        """Declare operation related sets (operation variables and mapping sets) in the pyomo object for a
        modeling class.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        # Set for operation variables
        def declareOpVarSet(pyM):
            return (
                (loc, compName, ip)
                for compName, comp in compDict.items()
                for loc in comp.processedLocationalEligibility.index
                for ip in esM.investmentPeriods
                if comp.processedLocationalEligibility[loc] == 1
            )

        setattr(
            pyM,
            "operationVarSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareOpVarSet),
        )

        if self.dimension == Dimension.ONE:
            # Dictionary which lists all components of the modeling class at one location
            setattr(
                pyM,
                "operationVarDict_" + abbrvName,
                {
                    ip: {
                        loc: {
                            compName
                            for compName in compDict
                            if (loc, compName, ip)
                            in getattr(pyM, "operationVarSet_" + abbrvName)
                        }
                        for loc in esM.locations
                    }
                    for ip in esM.investmentPeriods
                },
            )
        elif self.dimension == Dimension.TWO:
            # Dictionaries which list all outgoing and incoming components at a location
            setattr(
                pyM,
                "operationVarDictOut_" + abbrvName,
                {
                    ip: {
                        loc: {
                            loc_: {
                                compName
                                for compName in compDict
                                if (loc + "_" + loc_, compName, ip)
                                in getattr(pyM, "operationVarSet_" + abbrvName)
                            }
                            for loc_ in esM.locations
                        }
                        for loc in esM.locations
                    }
                    for ip in esM.investmentPeriods
                },
            )
            setattr(
                pyM,
                "operationVarDictIn_" + abbrvName,
                {
                    ip: {
                        loc: {
                            loc_: {
                                compName
                                for compName in compDict
                                if (loc_ + "_" + loc, compName, ip)
                                in getattr(pyM, "operationVarSet_" + abbrvName)
                            }
                            for loc_ in esM.locations
                        }
                        for loc in esM.locations
                    }
                    for ip in esM.investmentPeriods
                },
            )

    def declareBinOpVarSet(
        self,
        esM,
        pyM,
        binaryOperationParameter=["partLoadMin"],
        binaryOperationSetName="operationBinVarSet",
    ):
        """Declare binary operation variables.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSet_" + abbrvName)

        # check if any component has binary operation variables
        def _identifyBinaryOperationComponents(compDict, compName):
            return any(
                getattr(compDict[compName], x, None) is not None
                for x in binaryOperationParameter
            )

        binaryOperationComponents = [
            compName
            for (_, compName, _) in varSet
            if _identifyBinaryOperationComponents(compDict, compName)
        ]

        # if components with binary operations exist, set up the corresponding set
        if len(binaryOperationComponents) > 0:

            def declareOpBinVarSet(pyM):
                return (
                    (loc, compName, ip)
                    for compName, comp in compDict.items()
                    if compName in binaryOperationComponents
                    for loc in comp.processedLocationalEligibility.index
                    for ip in esM.investmentPeriods
                    if comp.processedLocationalEligibility[loc] == 1
                )

            setattr(
                pyM,
                binaryOperationSetName + "_" + abbrvName,
                pyomo.Set(dimen=3, initialize=declareOpBinVarSet),
            )

    ####################################################################################################################
    #                                   Functions for declaring operation mode sets                                    #
    ####################################################################################################################

    def declareOpConstrSet1(self, pyM, constrSetName, rateMax, rateFix):
        """Declare set of locations and components for which hasCapacityVariable is set to True and neither the
        maximum nor the fixed operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def declareOpConstrSet1(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in varSet
                if compDict[compName].hasCapacityVariable
                and getattr(compDict[compName], rateMax)[ip] is None
                and getattr(compDict[compName], rateFix)[ip] is None
            )

        setattr(
            pyM,
            constrSetName + "1_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareOpConstrSet1),
        )

    def declareOpConstrSet2(self, pyM, constrSetName, rateFix):
        """Declare set of locations and components for which hasCapacityVariable is set to True and a fixed
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def declareOpConstrSet2(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in varSet
                if compDict[compName].hasCapacityVariable
                and getattr(compDict[compName], rateFix)[ip] is not None
            )

        setattr(
            pyM,
            constrSetName + "2_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareOpConstrSet2),
        )

    def declareOpConstrSet3(self, pyM, constrSetName, rateMax):
        """Declare set of locations and components for which  hasCapacityVariable is set to True and a maximum
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def declareOpConstrSet3(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in varSet
                if compDict[compName].hasCapacityVariable
                and getattr(compDict[compName], rateMax)[ip] is not None
            )

        setattr(
            pyM,
            constrSetName + "3_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareOpConstrSet3),
        )

    def declareOpConstrSet4(self, pyM, constrSetName, rateMin):
        """Declare set of locations and components for which  hasCapacityVariable is set to True and a minimum
        operation rate is given.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def declareOpConstrSet4(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in varSet
                if compDict[compName].hasCapacityVariable
                and getattr(compDict[compName], rateMin)[ip] is not None
            )

        setattr(
            pyM,
            constrSetName + "4_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareOpConstrSet4),
        )

    def declareOpConstrSetMinPartLoad(self, pyM, constrSetName):
        """Declare set of locations and components for which partLoadMin is not None."""
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def declareOpConstrSetMinPartLoad(pyM):
            return (
                (loc, compName, ip)
                for loc, compName, ip in varSet
                if getattr(compDict[compName], "processedPartLoadMin") is not None
            )

        setattr(
            pyM,
            constrSetName + "partLoadMin_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareOpConstrSetMinPartLoad),
        )

    def declareOperationModeSets(
        self, pyM, constrSetName, rateMax, rateFix, rateMin=None
    ):
        """Declare operating mode sets.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param constrSetName: name of the constraint set.
        :type constrSetName: string

        :param rateMax: attribute of the considered component which stores the maximum operation rate data.
        :type rateMax: string

        :param rateMax: attribute of the considered component which stores the minimum operation rate data.
        :type rateMax: string

        :param rateFix: attribute of the considered component which stores the fixed operation rate data.
        :type rateFix: string
        """
        self.declareOpConstrSet1(pyM, constrSetName, rateMax, rateFix)
        self.declareOpConstrSet2(pyM, constrSetName, rateFix)
        self.declareOpConstrSet3(pyM, constrSetName, rateMax)
        if rateMin:
            self.declareOpConstrSet4(pyM, constrSetName, rateMin)

        self.declareOpConstrSetMinPartLoad(pyM, constrSetName)

    def declareYearlyFullLoadHoursMinSet(self, pyM):
        """Declare set of locations and components for which minimum yearly full load hours are given."""
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def declareYearlyFullLoadHoursMinSet():
            return (
                (loc, compName, ip)
                for loc, compName, ip in varSet
                if compDict[compName].processedYearlyFullLoadHoursMin[ip] is not None
            )

        setattr(
            pyM,
            "yearlyFullLoadHoursMinSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareYearlyFullLoadHoursMinSet()),
        )

    def declareYearlyFullLoadHoursMaxSet(self, pyM):
        """Declare set of locations and components for which maximum yearly full load hours are given."""
        compDict, abbrvName = self.componentsDict, self.abbrvName
        varSet = getattr(pyM, "operationVarSet_" + abbrvName)

        def declareYearlyFullLoadHoursMaxSet():
            return (
                (loc, compName, ip)
                for loc, compName, ip in varSet
                if compDict[compName].processedYearlyFullLoadHoursMax[ip] is not None
            )

        setattr(
            pyM,
            "yearlyFullLoadHoursMaxSet_" + abbrvName,
            pyomo.Set(dimen=3, initialize=declareYearlyFullLoadHoursMaxSet()),
        )

    ####################################################################################################################
    #                                         Functions for declaring variables                                        #
    ####################################################################################################################

    def declareCapacityVars(self, pyM):
        r"""Declare capacity variables.

        .. math::

            \\text{capMin}^{comp}_{loc} \\leq cap^{comp}_{loc} \\leq \\text{capMax}^{comp}_{loc}

        If a capacityFix parameter is given, the bounds are set to enforce

        .. math::
            \\text{cap}^{comp}_{loc} = \\text{capFix}^{comp}_{loc}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        abbrvName = self.abbrvName

        def capBounds(pyM, loc, compName, ip):
            """Set the lower and upper capacity bounds."""
            comp = self.componentsDict[compName]
            if (
                comp.processedCapacityFix[ip] is not None
                and loc in comp.processedCapacityFix[ip].index
            ):
                # in utils.py there are checks to ensure that capacityFix is between min and max
                return (
                    comp.processedCapacityFix[ip][loc],
                    comp.processedCapacityFix[ip][loc],
                )
            # the upper bound is only set if the parameter is given and no binary design variable exists
            # In the case of the binary design variable, the bigM-constraint will suffice as upper bound.
            if (comp.processedCapacityMin[ip] is not None) and (
                not comp.hasIsBuiltBinaryVariable
            ):
                capLowerBound = comp.processedCapacityMin[ip][loc]
            else:
                capLowerBound = 0

            if comp.processedCapacityMax[ip] is not None:
                capUpperBound = comp.processedCapacityMax[ip][loc]
            else:
                capUpperBound = None

            return (capLowerBound, capUpperBound)

        setattr(
            pyM,
            "cap_" + abbrvName,
            pyomo.Var(
                getattr(pyM, "designDimensionVarSet_" + abbrvName),
                domain=pyomo.NonNegativeReals,
                bounds=capBounds,
            ),
        )

    def declareCommissioningVars(self, pyM, esM):
        """Declare commissioning variable for capacity development of component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """

        def commisBounds(pyM, loc, compName, ip):
            """Set the lower and upper commissioning bounds."""
            comp = self.componentsDict[compName]
            if ip < 0:
                return None, None
            if (
                comp.processedCommissioningFix[ip] is not None
                and loc in comp.processedCommissioningFix[ip].index
            ):
                # in utils.py there are checks to ensure that CommissioningFix is between min and max
                return (
                    comp.processedCommissioningFix[ip][loc],
                    comp.processedCommissioningFix[ip][loc],
                )
            # the upper bound is only set if the parameter is given and no binary design variable exists
            # In the case of the binary design variable, the bigM-constraint will suffice as upper bound.
            if (
                comp.processedCommissioningMin[ip] is not None
                and not comp.hasIsBuiltBinaryVariable
            ):
                commisLowerBound = comp.processedCommissioningMin[ip][loc]
            else:
                commisLowerBound = 0

            if comp.processedCommissioningMax[ip] is not None:
                commisUpperBound = comp.processedCommissioningMax[ip][loc]
            else:
                commisUpperBound = None

            return commisLowerBound, commisUpperBound

        abbrvName = self.abbrvName
        setattr(
            pyM,
            "commis_" + abbrvName,
            pyomo.Var(
                getattr(pyM, "designCommisVarSet_" + abbrvName),
                domain=pyomo.NonNegativeReals,
                bounds=commisBounds,
            ),
        )

    def declareDecommissioningVars(self, pyM, esM):
        """Declare decommissioning variable for capacity development of component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        abbrvName = self.abbrvName
        setattr(
            pyM,
            "decommis_" + abbrvName,
            pyomo.Var(
                getattr(pyM, "designDimensionVarSet_" + abbrvName),
                domain=pyomo.NonNegativeReals,
            ),
        )

    def declareRealNumbersVars(self, pyM):
        """Declare variables representing the (continuous) number of installed components [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName = self.abbrvName
        setattr(
            pyM,
            "nbReal_" + abbrvName,
            pyomo.Var(
                getattr(pyM, "continuousDesignDimensionVarSet_" + abbrvName),
                domain=pyomo.NonNegativeReals,
            ),
        )

    def declareIntNumbersVars(self, pyM):
        """Declare variables representing the (discrete/integer) number of installed components [-].

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName = self.abbrvName
        setattr(
            pyM,
            "nbInt_" + abbrvName,
            pyomo.Var(
                getattr(pyM, "discreteDesignDimensionVarSet_" + abbrvName),
                domain=pyomo.NonNegativeIntegers,
            ),
        )

    def declareBinaryDesignDecisionVars(self, pyM, relaxIsBuiltBinary):
        r"""Declare binary variables [-] indicating if a component is considered at a location or not [-].

        If a isBuiltFix parameter is given, the bounds are set to enforce

        .. math::
            bin^{comp}_{loc} = \\text{binFix}^{comp}_{loc}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName, compDict = self.abbrvName, self.componentsDict

        def binDomain(pyM, loc, compName, ip):
            """Return the minimal necessary domain for the binary variable depending on the given conditions, e.g., if values are already fixed or if binary variables should be relaxed."""
            if relaxIsBuiltBinary:
                # If binary variables are relaxed, value can take all non negative reals (between 0 and 1)
                return pyomo.NonNegativeReals

            if ip < 0:
                return pyomo.Binary
            if (compDict[compName].isBuiltFix is not None) or (
                compDict[compName].processedCapacityFix[ip] is not None
            ):
                # If isBuiltFix or capacityFix is given, binary variable is already fixed.
                return pyomo.NonNegativeReals
            return pyomo.Binary

        def binBounds(pyM, loc, compName, ip):
            """Return bounds with the minimal necessary freedom for the binary variables (e.g., (0, 0) or (1, 1))."""
            if ip < 0:
                return None, None
            if compDict[compName].isBuiltFix is not None:
                # If isBuiltFix is given, binary variable is set to isBuiltFix
                return (
                    compDict[compName].isBuiltFix[loc],
                    compDict[compName].isBuiltFix[loc],
                )
            if (
                compDict[compName].processedCapacityFix[ip] is not None
                and loc in compDict[compName].processedCapacityFix[ip].index
            ):
                # If capacityFix is given, binary variable is set to 1
                return (
                    (1, 1)
                    if compDict[compName].processedCapacityFix[ip][loc] > 0
                    else (0, 0)
                )
            # Binary Variable between 0 and 1
            return (0, 1)

        if relaxIsBuiltBinary:
            setattr(
                pyM,
                "commisBin_" + abbrvName,
                pyomo.Var(
                    getattr(pyM, "designDecisionVarSet_" + abbrvName),
                    domain=binDomain,
                    bounds=(0, 1),
                ),
            )
        else:
            setattr(
                pyM,
                "commisBin_" + abbrvName,
                pyomo.Var(
                    getattr(pyM, "designDecisionVarSet_" + abbrvName),
                    domain=binDomain,
                    bounds=binBounds,
                ),
            )

    def declareOperationVars(
        self,
        pyM,
        esM,
        opVarName,
        opRateFixName="processedOperationRateFix",
        opRateMaxName="processedOperationRateMax",
        isOperationCommisYearDepending=False,
        flexibleConversion=False,
        relevanceThreshold=None,
    ):
        r"""Declare operation variables.

        The following operation modes are directly handled during variable creation as bounds instead of constraints.

        operation mode 4: If operationRateFix is given for components without a capacity variable,
        the variables are fixed with operationRateFix, i.e. the operation [commodityUnit*h] is equal to a time series.

        .. math::
            op^{comp,opType}_{loc,p,t} = \\text{opRateFix}^{comp,opType}_{loc,p,t}

        operation mode 5: If operationRateMax is given for components without a capacity variable,
        the variables are bounded by operationRateMax, i.e. the operation [commodityUnit*h] is limited by a time series.

        .. math::
            op^{comp,opType}_{loc,p,t} \\leq \\text{opRateMax}^{comp,opType}_{loc,p,t}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None

        :param isOperationCommisYearDepending: defines whether the operation variable is depending on the year
            of commissioning of the component. E.g. relevant if the commodity conversion, for example the efficiency,
            varies over the transformation pathway
        :type isOperationCommisYearDepending: str
        """
        abbrvName, compDict = self.abbrvName, self.componentsDict

        def opBounds(pyM, loc, compName, ip, p, t):  # noqa: PLR0911
            if not getattr(compDict[compName], "hasCapacityVariable"):
                if not pyM.hasSegmentation:
                    if getattr(compDict[compName], opRateMaxName)[ip] is not None:
                        rate = getattr(compDict[compName], opRateMaxName)[ip]
                        if rate is not None:
                            if relevanceThreshold is not None:
                                validThreshold = 0 < relevanceThreshold
                                if validThreshold and (
                                    rate[loc][p, t] < relevanceThreshold
                                ):
                                    return (0, 0)
                            return (0, rate[loc][p, t])
                        return None
                    if getattr(compDict[compName], opRateFixName)[ip] is not None:
                        rate = getattr(compDict[compName], opRateFixName)[ip]
                        if rate is not None:
                            if relevanceThreshold is not None:
                                validThreshold = 0 < relevanceThreshold
                                if validThreshold and (
                                    rate[loc][p, t] < relevanceThreshold
                                ):
                                    return (0, 0)
                            return (rate[loc][p, t], rate[loc][p, t])
                        return None
                    return (0, None)
                if getattr(compDict[compName], opRateMaxName)[ip] is not None:
                    rate = getattr(compDict[compName], opRateMaxName)[ip]
                    if rate is not None:
                        if relevanceThreshold is not None:
                            validThreshold = 0 < relevanceThreshold
                            if validThreshold and (
                                rate[loc][p, t] < relevanceThreshold
                            ):
                                return (0, 0)
                        return (
                            0,
                            rate[loc][p, t]
                            * esM.timeStepsPerSegment[ip].to_dict()[p, t],
                        )
                    return None
                if getattr(compDict[compName], opRateFixName)[ip] is not None:
                    rate = getattr(compDict[compName], opRateFixName)[ip]
                    if rate is not None:
                        if relevanceThreshold is not None:
                            validThreshold = 0 < relevanceThreshold
                            if validThreshold and (
                                rate[loc][p, t] < relevanceThreshold
                            ):
                                return (0, 0)
                        return (
                            rate[loc][p, t]
                            * esM.timeStepsPerSegment[ip].to_dict()[p, t],
                            rate[loc][p, t]
                            * esM.timeStepsPerSegment[ip].to_dict()[p, t],
                        )
                    return None
                return (0, None)
            return (0, None)

        if isOperationCommisYearDepending:
            # if the operation is depending on the year of commissioning, e.g. due to variable efficiencies over the
            # transformation pathway, the operation is additionally depending on commis
            def opBounds_commisDepending(pyM, loc, compName, commis, ip, p, t):
                return opBounds(pyM, loc, compName, ip, p, t)

            setattr(
                pyM,
                opVarName + "_" + abbrvName,
                pyomo.Var(
                    getattr(pyM, "operationCommisVarSet_" + abbrvName),
                    pyM.intraYearTimeSet,
                    domain=pyomo.NonNegativeReals,
                    bounds=opBounds_commisDepending,
                ),
            )
        elif flexibleConversion:
            setattr(
                pyM,
                opVarName + "_" + abbrvName,
                pyomo.Var(
                    getattr(pyM, "operationFlexVarSet_" + abbrvName),
                    pyM.intraYearTimeSet,
                    domain=pyomo.NonNegativeReals,
                ),
            )
        else:
            setattr(
                pyM,
                opVarName + "_" + abbrvName,
                pyomo.Var(
                    getattr(pyM, "operationVarSet_" + abbrvName),
                    pyM.intraYearTimeSet,
                    domain=pyomo.NonNegativeReals,
                    bounds=opBounds,
                ),
            )

    def declareOperationBinaryVars(
        self,
        pyM,
        opVarBinName="op_bin",
        opBinSetName="operationBinVarSet",
    ):
        """Declare binary operation variables.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        abbrvName = self.abbrvName

        # only setup binary operation variables if binary operation is declared, otherwise pyomo
        # always declares binary models which increases run time.

        if hasattr(pyM, opBinSetName + "_" + abbrvName):
            # declare binary operation variables
            setattr(
                pyM,
                opVarBinName + "_" + abbrvName,
                pyomo.Var(
                    getattr(pyM, opBinSetName + "_" + abbrvName),
                    pyM.intraYearTimeSet,
                    domain=pyomo.Binary,
                ),
            )

    ####################################################################################################################
    #                              Functions for declaring time independent constraints                                #
    ####################################################################################################################

    def capToNbReal(self, pyM):
        r"""Determine the components' capacities from the number of installed units.

        .. math::

            cap^{comp}_{loc} = \\text{capPerUnit}^{comp} \\cdot nbReal^{comp}_{loc}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, nbRealVar = (
            getattr(pyM, "cap_" + abbrvName),
            getattr(pyM, "nbReal_" + abbrvName),
        )
        nbRealVarSet = getattr(pyM, "continuousDesignDimensionVarSet_" + abbrvName)

        def capToNbReal(pyM, loc, compName, ip):
            return (
                capVar[loc, compName, ip]
                == nbRealVar[loc, compName, ip]
                * compDict[compName].processedCapacityPerPlantUnit[ip]
            )

        setattr(
            pyM,
            "ConstrCapToNbReal_" + abbrvName,
            pyomo.Constraint(nbRealVarSet, rule=capToNbReal),
        )

    def capToNbInt(self, pyM):
        r"""Determine the components' capacities from the number of installed units.

        .. math::

            cap^{comp}_{loc} = \\text{capPerUnit}^{comp} \\cdot nbInt^{comp}_{loc}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, nbIntVar = (
            getattr(pyM, "cap_" + abbrvName),
            getattr(pyM, "nbInt_" + abbrvName),
        )
        nbIntVarSet = getattr(pyM, "discreteDesignDimensionVarSet_" + abbrvName)

        def capToNbInt(pyM, loc, compName, ip):
            return (
                capVar[loc, compName, ip]
                == nbIntVar[loc, compName, ip]
                * compDict[compName].processedCapacityPerPlantUnit[ip]
            )

        setattr(
            pyM,
            "ConstrCapToNbInt_" + abbrvName,
            pyomo.Constraint(nbIntVarSet, rule=capToNbInt),
        )

    def bigM(self, pyM):
        r"""Enforce the consideration of the binary design variables of a component.

        .. math::

            \\text{M}^{comp} \\cdot bin^{comp}_{loc,ip} \\geq commis^{comp}_{loc,ip}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        commisVar = getattr(pyM, "commis_" + abbrvName)
        commisBinVar = getattr(pyM, "commisBin_" + abbrvName)
        commisBinVarSet = getattr(pyM, "designDecisionVarSet_" + abbrvName)

        def bigM(pyM, loc, compName, ip):
            comp = compDict[compName]
            if ip not in comp.processedStockYears:
                # set bigM for investment periods
                M = (
                    comp.processedCapacityMax[ip][loc]
                    if comp.processedCapacityMax[ip] is not None
                    else comp.bigM
                )
                return (
                    commisVar[loc, compName, ip] <= commisBinVar[loc, compName, ip] * M
                )
            # set binary variables fix for stock years
            hasStockCommissioning = (
                self.componentsDict[compName].processedStockCommissioning[ip].loc[loc]
                > 0
            )
            if hasStockCommissioning:
                return commisBinVar[loc, compName, ip] == 1
            return commisBinVar[loc, compName, ip] == 0

        setattr(
            pyM, "ConstrBigM_" + abbrvName, pyomo.Constraint(commisBinVarSet, rule=bigM)
        )

    def capacityMinDec(self, pyM):
        r"""Enforce the consideration of minimum capacities for components with design decision variables.

        Minimal capacity which needs to be reached for every investment period with commissioning.
        As the commisBinVar is coupled with commissioning var, constraint only sets minimal Capacity if component is commissioned.
        Therefore decommissioning of the component is possible without any constraints.

        .. math::

            \\text{capMin}^{comp}_{loc} \\cdot commisBin^{comp}_{loc,ip} \\leq  cap^{comp}_{loc,ip}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar = getattr(pyM, "cap_" + abbrvName)
        commisBinVar = getattr(pyM, "commisBin_" + abbrvName)
        commisBinVarSet = getattr(pyM, "designDecisionVarSet_" + abbrvName)

        def capacityMinDec(pyM, loc, compName, ip):
            if ip not in compDict[compName].processedStockYears:
                return (
                    capVar[loc, compName, ip]
                    >= compDict[compName].processedCapacityMin[ip][loc]
                    * commisBinVar[loc, compName, ip]
                    if compDict[compName].processedCapacityMin[ip] is not None
                    else pyomo.Constraint.Skip
                )
            # constraint not required for stock years
            return pyomo.Constraint.Skip

        setattr(
            pyM,
            "ConstrCapacityMinDec_" + abbrvName,
            pyomo.Constraint(commisBinVarSet, rule=capacityMinDec),
        )

    def designBinFix(self, pyM):
        r"""Set, if applicable, the installed capacities of a component.

        .. math::

            bin^{comp}_{(loc_1,loc_2),ip} = \\text{binFix}^{comp}_{(loc_1,loc_2)}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        commisBinVar = getattr(pyM, "commisBin_" + abbrvName)
        commisBinVarSet = getattr(pyM, "designDecisionVarSet_" + abbrvName)

        def designBinFix(pyM, loc, compName, ip):
            return (
                commisBinVar[loc, compName, ip] == compDict[compName].isBuiltFix[loc]
                if compDict[compName].isBuiltFix is not None
                else pyomo.Constraint.Skip
            )

        setattr(
            pyM,
            "ConstrDesignBinFix_" + abbrvName,
            pyomo.Constraint(commisBinVarSet, rule=designBinFix),
        )

    ####################################################################################################################
    #                               Functions for declaring pathway dependent constraints                              #
    ####################################################################################################################
    def designDevelopmentConstraint(self, pyM, esM):
        """Link the capacity development between investment periods.

        For stochastic: The capacity design must be equal between the different years.

        .. math::

            cap^{comp}_{loc,ip+1} =  cap^{comp}_{loc,ip}

        For the development pathway, the capacity of an investment period is composed
        of the capacity of the previous investment periods and the commissioning and
        decommissioning in the current investment period.

        .. math::

            cap^{comp}_{loc,ip+1} =  cap^{comp}_{loc,ip} + commis^{comp}_{loc,ip} - decommis^{comp}_{loc,ip}


        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        abbrvName = self.abbrvName
        commisConstrSet = getattr(pyM, "designDevelopmentVarSet_" + abbrvName)
        if esM.stochasticModel:
            capVar = getattr(pyM, "cap_" + abbrvName)

            def capacityDevelopmentStochastic(pyM, loc, compName, ip):
                # all investment periods must have the same capacity
                return capVar[loc, compName, ip + 1] == capVar[loc, compName, ip]

            setattr(
                pyM,
                "ConstrCapacityDevelopment_" + abbrvName,
                pyomo.Constraint(
                    commisConstrSet,
                    rule=capacityDevelopmentStochastic,
                ),
            )
        else:
            capVar = getattr(pyM, "cap_" + abbrvName)
            commisVar = getattr(pyM, "commis_" + abbrvName)
            decommisVar = getattr(pyM, "decommis_" + abbrvName)

            def capacityDevelopmentPerfectForesight(pyM, loc, compName, ip):
                return (
                    capVar[loc, compName, ip + 1]
                    == capVar[loc, compName, ip]
                    + commisVar[loc, compName, ip + 1]
                    - decommisVar[loc, compName, ip + 1]
                )

            setattr(
                pyM,
                "ConstrCapacityDevelopment_" + abbrvName,
                pyomo.Constraint(
                    commisConstrSet, rule=capacityDevelopmentPerfectForesight
                ),
            )

    def stockCapacityConstraint(self, pyM, esM):
        """Set the stock capacity constraint. The stock capacity is the sum of the stock
        commissioning, which do not exceed its technical lifetime.

        For stochastic, the stock of past investment periods is not only valid for ip=0 but for all investment periods.
        .. math::

            cap^{comp}_{loc,ip} =  stockCap^{comp}_{loc} + commis^{comp}_{loc,ip} - decommis^{comp}_{loc,0}

        For capacity development, the stock is only considered for the first investment periods.

        .. math::

            cap^{comp}_{loc,0} =  stockCap^{comp}_{loc} + commis^{comp}_{loc,0} - decommis^{comp}_{loc,0}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        abbrvName = self.abbrvName
        capVar = getattr(pyM, "cap_" + abbrvName)
        commisVar = getattr(pyM, "commis_" + abbrvName)
        decommisVar = getattr(pyM, "decommis_" + abbrvName)
        locCompConstrSet = getattr(pyM, "DesignLocationComponentVarSet_" + abbrvName)
        locCompIpConstrSet = getattr(pyM, "designDimensionVarSet_" + abbrvName)

        if esM.stochasticModel:

            def initialStochastic(pyM, loc, compName, ip):
                stock_cap = self.componentsDict[compName].stockCapacityStartYear[loc]
                return (
                    capVar[loc, compName, ip]
                    == stock_cap
                    + commisVar[loc, compName, ip]
                    - decommisVar[loc, compName, 0]
                )

            setattr(
                pyM,
                "InitialYear_" + abbrvName,
                pyomo.Constraint(locCompIpConstrSet, rule=initialStochastic),
            )
        else:

            def initialYear(pyM, loc, compName):
                stock_cap = self.componentsDict[compName].stockCapacityStartYear[loc]
                return (
                    capVar[loc, compName, 0]
                    == stock_cap
                    + commisVar[loc, compName, 0]
                    - decommisVar[loc, compName, 0]
                )

            setattr(
                pyM,
                "InitialYear_" + abbrvName,
                pyomo.Constraint(locCompConstrSet, rule=initialYear),
            )

    def stockCommissioningConstraint(self, pyM, esM):
        """Set commissioning variable for past investment periods. For past investment periods,
        where no stock commissioning is specified the commissioning variable is set to zero.
        """
        commisConstrSet = getattr(pyM, "designCommisVarSet_" + self.abbrvName)
        commisVar = getattr(pyM, "commis_" + self.abbrvName)

        def stockCommissioning(pyM, loc, compName, ip):
            if (
                ip in esM.investmentPeriods
            ):  # initialize stock commissioning only for stock years
                return pyomo.Constraint.Skip
            if (
                self.componentsDict[compName].processedStockCommissioning is None
            ):  # set 0 if there is no stock
                return commisVar[loc, compName, ip] == 0
            return (
                commisVar[loc, compName, ip]
                == self.componentsDict[compName].processedStockCommissioning[ip][loc]
            )

        setattr(
            pyM,
            "StockCommissioning_" + self.abbrvName,
            pyomo.Constraint(commisConstrSet, rule=stockCommissioning),
        )

    def decommissioningConstraint(self, pyM, esM):
        r"""Declase the decommissioning after the technical lifetime from investment
        period of commissioning.

        .. math::

            decommis^{comp}_{loc,ip} = commis^{comp}_{loc,ip-\\mathrm{ipTechnicalLifetime}}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: energy system model containing general information.
        :type esM: EnergySystemModel instance from the FINE package
        """
        abbrvName = self.abbrvName
        commisVar = getattr(pyM, "commis_" + abbrvName)
        decommisVar = getattr(pyM, "decommis_" + abbrvName)
        decommisConstrSet = getattr(pyM, "designDimensionVarSet_" + abbrvName)

        def capacityDecommissioning(pyM, loc, compName, ip):
            tech_lifetime = self.componentsDict[compName].ipTechnicalLifetime[loc]

            # commissioning date is depending whether technical lifetime ceiled or floored to next interval
            # if technical lifetime is already a multiple of the interval, nothing happens
            if self.componentsDict[compName].floorTechnicalLifetime:
                comm_date = ip - math.floor(tech_lifetime)
            else:
                comm_date = ip - math.ceil(tech_lifetime)
            # if the commissioning date is within the investment periods, the
            # decommissioning and commissioning variables are linked
            if comm_date in esM.investmentPeriods:
                return (
                    decommisVar[loc, compName, ip]
                    == commisVar[loc, compName, comm_date]
                )
            # else the decommissioning is depending on the stockcommissioning
            # or set to 0
            procStockCommissioning = self.componentsDict[
                compName
            ].processedStockCommissioning
            if procStockCommissioning is not None:
                return (
                    decommisVar[loc, compName, ip]
                    == self.componentsDict[compName].processedStockCommissioning[
                        comm_date
                    ][loc]
                )
            return decommisVar[loc, compName, ip] == 0

        setattr(
            pyM,
            "DecommConstrCapacityDevelopment_" + abbrvName,
            pyomo.Constraint(decommisConstrSet, rule=capacityDecommissioning),
        )

    ####################################################################################################################
    #                               Functions for declaring time dependent constraints                                 #
    ####################################################################################################################

    def operationMode1(
        self,
        pyM,
        esM,
        constrName,
        constrSetName,
        opVarName,
        factorName=None,
        *,
        isOperationCommisYearDepending=False,
    ):
        r"""Define operation mode 1. The operation [commodityUnit*h] is limited by the installed capacity in:\n
        * [commodityUnit*h] (for storages) or in
        * [commodityUnit] multiplied by the hours per time step (else).\n
        An additional factor can limited the operation further.

        .. math::

            op^{comp,opType}_{loc,ip,p,t} \\leq \\tau^{hours} \\cdot \\text{opFactor}^{opType} \\cdot cap^{comp}_{loc,ip}

        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)
        commisVar = getattr(pyM, "commis_" + abbrvName)
        constrSet1 = getattr(pyM, constrSetName + "1_" + abbrvName)

        if not pyM.hasSegmentation:
            factor1 = esM.hoursPerTimeStep
            if isOperationCommisYearDepending:

                def op1(pyM, loc, compName, commis, ip, p, t):
                    factor2 = (
                        1
                        if factorName is None
                        else getattr(compDict[compName], factorName)
                    )
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        <= factor1 * factor2 * commisVar[loc, compName, commis]
                    )
            else:

                def op1(pyM, loc, compName, ip, p, t):
                    factor2 = (
                        1
                        if factorName is None
                        else getattr(compDict[compName], factorName)
                    )
                    return (
                        opVar[loc, compName, ip, p, t]
                        <= factor1 * factor2 * capVar[loc, compName, ip]
                    )

            setattr(
                pyM,
                constrName + "1_" + abbrvName,
                pyomo.Constraint(constrSet1, pyM.intraYearTimeSet, rule=op1),
            )
        else:
            if isOperationCommisYearDepending:

                def op1(pyM, loc, compName, commis, ip, p, t):
                    factor1 = esM.hoursPerSegment[ip].to_dict()
                    factor2 = (
                        1
                        if factorName is None
                        else getattr(compDict[compName], factorName)
                    )
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        <= factor1[p, t] * factor2 * commisVar[loc, compName, commis]
                    )  # factor not dependent on ip

            else:

                def op1(pyM, loc, compName, ip, p, t):
                    factor1 = esM.hoursPerSegment[ip].to_dict()
                    factor2 = (
                        1
                        if factorName is None
                        else getattr(compDict[compName], factorName)
                    )
                    return (
                        opVar[loc, compName, ip, p, t]
                        <= factor1[p, t] * factor2 * capVar[loc, compName, ip]
                    )  # factor not dependent on ip

            setattr(
                pyM,
                constrName + "1_" + abbrvName,
                pyomo.Constraint(constrSet1, pyM.intraYearTimeSet, rule=op1),
            )

    def operationMode2(
        self,
        pyM,
        esM,
        constrName,
        constrSetName,
        opVarName,
        opRateName="processedOperationRateFix",
        *,
        isOperationCommisYearDepending=False,
    ):
        r"""Define operation mode 2.

        The operation [commodityUnit*h] is equal to the installed capacity multiplied
        with a time series in:\n
        * [commodityUnit*h] (for storages) or in
        * [commodityUnit] multiplied by the hours per time step (else).\n

        .. math::

            op^{comp,opType}_{loc,ip,p,t} \\leq \\tau^{hours} \\cdot \\text{opRateMax}^{comp,opType}_{loc,ip,p,t} \\cdot cap^{comp}_{loc,ip}

        """
        # additions for perfect foresight
        # operationRate is the same for all ip
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)
        commisVar = getattr(pyM, "commis_" + abbrvName)
        constrSet2 = getattr(pyM, constrSetName + "2_" + abbrvName)

        if not pyM.hasSegmentation:
            factor = esM.hoursPerTimeStep
            if isOperationCommisYearDepending:

                def op2(pyM, loc, compName, commis, ip, p, t):
                    rate = getattr(compDict[compName], opRateName)[ip]
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        == commisVar[loc, compName, commis] * rate[loc][p, t] * factor
                    )  # rate independent from ip

            else:

                def op2(pyM, loc, compName, ip, p, t):
                    rate = getattr(compDict[compName], opRateName)[ip]
                    return (
                        opVar[loc, compName, ip, p, t]
                        == capVar[loc, compName, ip] * rate[loc][p, t] * factor
                    )  # rate independent from ip

            setattr(
                pyM,
                constrName + "2_" + abbrvName,
                pyomo.Constraint(constrSet2, pyM.intraYearTimeSet, rule=op2),
            )
        else:
            if isOperationCommisYearDepending:

                def op2(pyM, loc, compName, commis, ip, p, t):
                    factor = esM.hoursPerSegment[ip].to_dict()
                    rate = getattr(compDict[compName], opRateName)[ip]
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        == commisVar[loc, compName, commis]
                        * rate[loc][p, t]
                        * factor[p, t]
                    )

            else:

                def op2(pyM, loc, compName, ip, p, t):
                    factor = esM.hoursPerSegment[ip].to_dict()
                    rate = getattr(compDict[compName], opRateName)[ip]
                    return (
                        opVar[loc, compName, ip, p, t]
                        == capVar[loc, compName, ip] * rate[loc][p, t] * factor[p, t]
                    )

            setattr(
                pyM,
                constrName + "2_" + abbrvName,
                pyomo.Constraint(constrSet2, pyM.intraYearTimeSet, rule=op2),
            )

    def operationMode3(
        self,
        pyM,
        esM,
        constrName,
        constrSetName,
        opVarName,
        opRateName="processedOperationRateMax",
        *,
        isOperationCommisYearDepending=False,
        relevanceThreshold=None,
    ):
        r"""Define operation mode 3.

        The operation [commodityUnit*h] is limited by an installed capacity multiplied
        with a time series in:\n
        * [commodityUnit*h] (for storages) or in
        * [commodityUnit] multiplied by the hours per time step (else).\n

        .. math::
            op^{comp,opType}_{loc,ip,p,t} = \\tau^{hours} \\cdot \\text{opRateFix}^{comp,opType}_{loc,ip,p,t} \\cdot cap^{comp}_{loc,ip}

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None

        """
        # operationRate is the same for all ip
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)
        commisVar = getattr(pyM, "commis_" + abbrvName)
        constrSet3 = getattr(pyM, constrSetName + "3_" + abbrvName)

        if not pyM.hasSegmentation:
            factor = esM.hoursPerTimeStep
            if isOperationCommisYearDepending:

                def op3(pyM, loc, compName, commis, ip, p, t):
                    rate = getattr(compDict[compName], opRateName)[ip]
                    if relevanceThreshold is not None:
                        validTreshold = 0 < relevanceThreshold
                        if validTreshold and (rate[loc][p, t] <= relevanceThreshold):
                            # operationRate is lower than threshold --> set to 0
                            return opVar[loc, compName, commis, ip, p, t] == 0
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        <= commisVar[loc, compName, commis] * rate[loc][p, t] * factor
                    )

            else:

                def op3(pyM, loc, compName, ip, p, t):
                    rate = getattr(compDict[compName], opRateName)[ip]
                    if relevanceThreshold is not None:
                        validTreshold = 0 < relevanceThreshold
                        if validTreshold and (rate[loc][p, t] <= relevanceThreshold):
                            # operationRate is lower than threshold --> set to 0
                            return opVar[loc, compName, ip, p, t] == 0
                    return (
                        opVar[loc, compName, ip, p, t]
                        <= capVar[loc, compName, ip] * rate[loc][p, t] * factor
                    )

            setattr(
                pyM,
                constrName + "3_" + abbrvName,
                pyomo.Constraint(constrSet3, pyM.intraYearTimeSet, rule=op3),
            )
        else:
            if isOperationCommisYearDepending:

                def op3(pyM, loc, compName, commis, ip, p, t):
                    factor = esM.hoursPerSegment[ip].to_dict()
                    rate = getattr(compDict[compName], opRateName)[ip]
                    if relevanceThreshold is not None:
                        validTreshold = 0 < relevanceThreshold
                        if validTreshold and (rate[loc][p, t] <= relevanceThreshold):
                            # operationRate is lower than threshold --> set to 0
                            return opVar[loc, compName, commis, ip, p, t] == 0
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        <= commisVar[loc, compName, commis]
                        * rate[loc][p, t]
                        * factor[p, t]
                    )  # rate and factor independent from ip

            else:

                def op3(pyM, loc, compName, ip, p, t):
                    factor = esM.hoursPerSegment[ip].to_dict()
                    rate = getattr(compDict[compName], opRateName)[ip]
                    if relevanceThreshold is not None:
                        validTreshold = 0 < relevanceThreshold
                        if validTreshold and (rate[loc][p, t] <= relevanceThreshold):
                            # operationRate is lower than threshold --> set to 0
                            return opVar[loc, compName, ip, p, t] == 0
                    return (
                        opVar[loc, compName, ip, p, t]
                        <= capVar[loc, compName, ip] * rate[loc][p, t] * factor[p, t]
                    )  # rate and factor independent from ip

            setattr(
                pyM,
                constrName + "3_" + abbrvName,
                pyomo.Constraint(constrSet3, pyM.intraYearTimeSet, rule=op3),
            )

    def operationMode4(
        self,
        pyM,
        esM,
        constrName,
        constrSetName,
        opVarName,
        opRateName="processedOperationRateMin",
        *,
        isOperationCommisYearDepending=False,
        relevanceThreshold=None,
    ):
        r"""Define operation mode 4.

        The operation [commodityUnit*h] is limited by an installed capacity
        multiplied with a time series in:\n
        * [commodityUnit*h] (for storages) or in
        * [commodityUnit] multiplied by the hours per time step (else).\n

        .. math::
            op^{comp,opType}_{loc,ip,p,t} = \\tau^{hours} \\cdot \\text{opRateFix}^{comp,opType}_{loc,ip,p,t} \\cdot cap^{comp}_{loc,ip}

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None

        """
        # operationRate is the same for all ip
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)
        commisVar = getattr(pyM, "commis_" + abbrvName)
        constrSet4 = getattr(pyM, constrSetName + "4_" + abbrvName)

        if not pyM.hasSegmentation:
            factor = esM.hoursPerTimeStep
            if isOperationCommisYearDepending:

                def op4(pyM, loc, compName, commis, ip, p, t):
                    rate = getattr(compDict[compName], opRateName)[ip]
                    if relevanceThreshold is not None:
                        validTreshold = 0 < relevanceThreshold
                        if validTreshold and (rate[loc][p, t] <= relevanceThreshold):
                            # operationRate is lower than threshold --> set to 0
                            return opVar[loc, compName, commis, ip, p, t] == 0
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        >= commisVar[loc, compName, commis] * rate[loc][p, t] * factor
                    )

            else:

                def op4(pyM, loc, compName, ip, p, t):
                    rate = getattr(compDict[compName], opRateName)[ip]
                    if relevanceThreshold is not None:
                        validTreshold = 0 < relevanceThreshold
                        if validTreshold and (rate[loc][p, t] <= relevanceThreshold):
                            # operationRate is lower than threshold --> set to 0
                            return opVar[loc, compName, ip, p, t] == 0
                    return (
                        opVar[loc, compName, ip, p, t]
                        >= capVar[loc, compName, ip] * rate[loc][p, t] * factor
                    )

            setattr(
                pyM,
                constrName + "4_" + abbrvName,
                pyomo.Constraint(constrSet4, pyM.intraYearTimeSet, rule=op4),
            )
        else:
            if isOperationCommisYearDepending:

                def op4(pyM, loc, compName, commis, ip, p, t):
                    factor = esM.hoursPerSegment[ip].to_dict()
                    rate = getattr(compDict[compName], opRateName)[ip]
                    if relevanceThreshold is not None:
                        validTreshold = 0 < relevanceThreshold
                        if validTreshold and (rate[loc][p, t] <= relevanceThreshold):
                            # operationRate is lower than threshold --> set to 0
                            return opVar[loc, compName, commis, ip, p, t] == 0
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        >= commisVar[loc, compName, commis]
                        * rate[loc][p, t]
                        * factor[p, t]
                    )  # rate and factor independent from ip

            else:

                def op4(pyM, loc, compName, ip, p, t):
                    factor = esM.hoursPerSegment[ip].to_dict()
                    rate = getattr(compDict[compName], opRateName)[ip]
                    if relevanceThreshold is not None:
                        validTreshold = 0 < relevanceThreshold
                        if validTreshold and (rate[loc][p, t] <= relevanceThreshold):
                            # operationRate is lower than threshold --> set to 0
                            return opVar[loc, compName, ip, p, t] == 0
                    return (
                        opVar[loc, compName, ip, p, t]
                        >= capVar[loc, compName, ip] * rate[loc][p, t] * factor[p, t]
                    )  # rate and factor independent from ip

            setattr(
                pyM,
                constrName + "4_" + abbrvName,
                pyomo.Constraint(constrSet4, pyM.intraYearTimeSet, rule=op4),
            )

    def binaryOperation(
        self,
        pyM,
        constrName,
        constrSetName,
        binaryParameterName,
        opVarName,
        opVarBinName,
        isOperationCommisYearDepending=False,
    ):
        """Create binary operation constraints for component operation.

        Defines two constraints linking a continuous operation variable
        to its corresponding binary variable using the Big-M formulation.
        Handles both standard and commissioning year-dependent cases.

        The binaryOperation1 constraint is used to force the binary variable
        to one if the continuous variable is greater than zero.

        The binaryOperation2 constraint ensures that the continuous
        variable is greater than zero whenever the binary variable is one.
        This is used for the upTimeMin and downTimeMin feature.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        opVarBin = getattr(pyM, opVarBinName + "_" + abbrvName, None)

        # only create constraint when binary operation variable is specified
        if opVarBin is not None:
            constrSetBinary = getattr(
                pyM, constrSetName + binaryParameterName + "_" + abbrvName
            )

            def getBigM(compName):
                return getattr(compDict[compName], "bigM")

            # First constraint
            if isOperationCommisYearDepending:

                def binOperation1(pyM, loc, compName, commis, ip, p, t):
                    return opVar[loc, compName, commis, ip, p, t] <= opVarBin[
                        loc, compName, commis, ip, p, t
                    ] * getBigM(compName)
            else:

                def binOperation1(pyM, loc, compName, ip, p, t):
                    return opVar[loc, compName, ip, p, t] <= opVarBin[
                        loc, compName, ip, p, t
                    ] * getBigM(compName)

            setattr(
                pyM,
                constrName + "binaryOperation1_" + abbrvName,
                pyomo.Constraint(
                    constrSetBinary, pyM.intraYearTimeSet, rule=binOperation1
                ),
            )

            # Second constraint
            if isOperationCommisYearDepending:

                def binOperation2(pyM, loc, compName, commis, ip, p, t):
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        >= opVarBin[loc, compName, commis, ip, p, t] * 1e-4
                    )
            else:

                def binOperation2(pyM, loc, compName, ip, p, t):
                    return (
                        opVar[loc, compName, ip, p, t]
                        >= opVarBin[loc, compName, ip, p, t] * 1e-4
                    )

            setattr(
                pyM,
                constrName + "binaryOperation2_" + abbrvName,
                pyomo.Constraint(
                    constrSetBinary, pyM.intraYearTimeSet, rule=binOperation2
                ),
            )

    def additionalMinPartLoad(
        self,
        pyM,
        esM,
        constrName,
        constrSetName,
        opVarName,
        opVarBinName,
        capVarName,
        isOperationCommisYearDepending=False,
    ):
        """Set, if applicable, the minimal part load of a component.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName

        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        opVarBin = getattr(pyM, opVarBinName + "_" + abbrvName, None)

        # only create constraint when partLoadMin specified
        if opVarBin is not None:
            capVar = getattr(pyM, capVarName + "_" + abbrvName)
            commisVar = getattr(pyM, "commis_" + abbrvName)
            constrSetMinPartLoad = getattr(
                pyM, constrSetName + "partLoadMin_" + abbrvName
            )

            def getPartLoadMin(compDict, compName, ip):
                return getattr(compDict[compName], "processedPartLoadMin")[ip]

            def getBigM(compDict, compName):
                return getattr(compDict[compName], "bigM")

            if not pyM.hasSegmentation:
                if isOperationCommisYearDepending:

                    def opMinPartLoad(pyM, loc, compName, commis, ip, p, t):
                        processedPartLoadMin = getPartLoadMin(compDict, compName, ip)
                        bigM = getBigM(compDict, compName)
                        return (
                            opVar[loc, compName, commis, ip, p, t]
                            >= processedPartLoadMin
                            * commisVar[loc, compName, commis]
                            * esM.hoursPerTimeStep
                            - (1 - opVarBin[loc, compName, commis, ip, p, t]) * bigM
                        )
                else:

                    def opMinPartLoad(pyM, loc, compName, ip, p, t):
                        processedPartLoadMin = getPartLoadMin(compDict, compName, ip)
                        bigM = getBigM(compDict, compName)
                        return (
                            opVar[loc, compName, ip, p, t]
                            >= processedPartLoadMin
                            * capVar[loc, compName, ip]
                            * esM.hoursPerTimeStep
                            - (1 - opVarBin[loc, compName, ip, p, t]) * bigM
                        )
            elif isOperationCommisYearDepending:

                def opMinPartLoad(pyM, loc, compName, commis, ip, p, t):
                    processedPartLoadMin = getPartLoadMin(compDict, compName, ip)
                    bigM = getBigM(compDict, compName)
                    return (
                        opVar[loc, compName, commis, ip, p, t]
                        >= processedPartLoadMin
                        * commisVar[loc, compName, commis]
                        * esM.hoursPerSegment[ip][p, t]
                        - (1 - opVarBin[loc, compName, commis, ip, p, t]) * bigM
                    )
            else:

                def opMinPartLoad(pyM, loc, compName, ip, p, t):
                    processedPartLoadMin = getPartLoadMin(compDict, compName, ip)
                    bigM = getBigM(compDict, compName)
                    return (
                        opVar[loc, compName, ip, p, t]
                        >= processedPartLoadMin
                        * capVar[loc, compName, ip]
                        * esM.hoursPerSegment[ip][p, t]
                        - (1 - opVarBin[loc, compName, ip, p, t]) * bigM
                    )

            setattr(
                pyM,
                constrName + "partLoadMin_2_" + abbrvName,
                pyomo.Constraint(
                    constrSetMinPartLoad, pyM.intraYearTimeSet, rule=opMinPartLoad
                ),
            )

    def yearlyFullLoadHoursMin(
        self,
        pyM,
        esM,
        constrSetName,
        constrName,
        opVarName,
        isOperationCommisYearDepending=False,
    ):
        """Limit the annual full load hours to a minimum value.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param constrName: name for the constraint in esM.pyM
        :type constrName: str

        :param constrSetName: name of the constraint set
        :type constrSetName: str

        :param opVarName: name of the operation variables
        :type opVarName: str

        :param isOperationCommisYearDepending: defines whether the operation variable is depending on the year of commissioning of the component. E.g. relevant if the commodity conversion, for example the efficiency, varies over the transformation pathway
        :type isOperationCommisYearDepending: str
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)
        commisVar = getattr(pyM, "commis_" + abbrvName)
        yearlyFullLoadHoursMinSet = getattr(pyM, constrSetName + "_" + abbrvName)
        if isOperationCommisYearDepending:
            # for technologies which have operations depending on the commissioning year, e.g. by variable commodity conversion factors
            def yearlyFullLoadHoursMinConstraint(pyM, loc, compName, commis, ip):
                full_load_hours = (
                    sum(
                        opVar[loc, compName, commis, ip, p, t]
                        * esM.periodOccurrences[ip][p]
                        for p, t in pyM.intraYearTimeSet
                    )
                    / esM.numberOfYears
                )
                return (
                    full_load_hours
                    >= commisVar[loc, compName, commis]
                    * compDict[compName].processedYearlyFullLoadHoursMin[ip][loc]
                )

        else:

            def yearlyFullLoadHoursMinConstraint(pyM, loc, compName, ip):
                full_load_hours = (
                    sum(
                        opVar[loc, compName, ip, p, t] * esM.periodOccurrences[ip][p]
                        for p, t in pyM.intraYearTimeSet
                    )
                    / esM.numberOfYears
                )
                return (
                    full_load_hours
                    >= capVar[loc, compName, ip]
                    * compDict[compName].processedYearlyFullLoadHoursMin[ip][loc]
                )

        setattr(
            pyM,
            constrName + "_" + abbrvName,
            pyomo.Constraint(
                yearlyFullLoadHoursMinSet, rule=yearlyFullLoadHoursMinConstraint
            ),
        )

    def yearlyFullLoadHoursMax(
        self,
        pyM,
        esM,
        constrSetName,
        constrName,
        opVarName,
        isOperationCommisYearDepending=False,
    ):
        """Limit the annual full load hours to a maximum value.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param constrName: name for the constraint in esM.pyM
        :type constrName: str

        :param constrSetName: name of the constraint set
        :type constrSetName: str

        :param opVarName: name of the operation variables
        :type opVarName: str

        :param isOperationCommisYearDepending: defines whether the operation variable is depending on the year of commissioning of the component. E.g. relevant if the commodity conversion, for example the efficiency, varies over the transformation pathway
        :type isOperationCommisYearDepending: str
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)
        commisVar = getattr(pyM, "commis_" + abbrvName)
        yearlyFullLoadHoursMaxSet = getattr(pyM, constrSetName + "_" + abbrvName)
        if isOperationCommisYearDepending:

            def yearlyFullLoadHoursMaxConstraint(pyM, loc, compName, commis, ip):
                full_load_hours = (
                    sum(
                        opVar[loc, compName, commis, ip, p, t]
                        * esM.periodOccurrences[ip][p]
                        for p, t in pyM.intraYearTimeSet
                    )
                    / esM.numberOfYears
                )
                return (
                    full_load_hours
                    <= commisVar[loc, compName, commis]
                    * compDict[compName].processedYearlyFullLoadHoursMax[ip][loc]
                )

        else:

            def yearlyFullLoadHoursMaxConstraint(pyM, loc, compName, ip):
                full_load_hours = (
                    sum(
                        opVar[loc, compName, ip, p, t] * esM.periodOccurrences[ip][p]
                        for p, t in pyM.intraYearTimeSet
                    )
                    / esM.numberOfYears
                )
                return (
                    full_load_hours
                    <= capVar[loc, compName, ip]
                    * compDict[compName].processedYearlyFullLoadHoursMax[ip][loc]
                )

        setattr(
            pyM,
            constrName + "_" + abbrvName,
            pyomo.Constraint(
                yearlyFullLoadHoursMaxSet, rule=yearlyFullLoadHoursMaxConstraint
            ),
        )

    ####################################################################################################################
    #  Functions for declaring component contributions to basic energy system constraints and the objective function   #
    ####################################################################################################################

    @abstractmethod
    def declareSets(self, esM, pyM):
        """Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Declare sets of components and constraints in the componentModel class.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        raise NotImplementedError

    @abstractmethod
    def declareVariables(self, esM, pyM, relevanceThreshold):
        """Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Declare variables of components in the componentModel class.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None
        """
        raise NotImplementedError

    @abstractmethod
    def declareComponentConstraints(self, esM, pyM):
        """Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Declare constraints of components in the componentModel class.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        raise NotImplementedError

    @abstractmethod
    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        """Check if operation variables exist in the modeling class at a location which are connected to a commodity.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param loc: name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """
        raise NotImplementedError

    @abstractmethod
    def getCommodityBalanceContribution(self, pyM, commod, loc, ip, p, t):
        """Abstract method which has to be implemented by subclasses (otherwise a NotImplementedError raises).
        Get contribution to a commodity balance.
        """
        raise NotImplementedError

    def getObjectiveFunctionContribution(self, esM, pyM):
        """Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        capexCap = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedInvestPerCapacity"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commis",
            divisorName="CCF",
            QPdivisorNames=["QPbound", "CCF"],
        )
        capexDec = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestIfBuilt"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commisBin",
            divisorName="CCF",
        )
        opexCap = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedOpexPerCapacity"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commis",
            QPdivisorNames=["QPbound"],
        )
        opexDec = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexIfBuilt"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commisBin",
        )

        return capexCap + capexDec + opexCap + opexDec

    def getSharedPotentialContribution(self, pyM, key, loc, ip):
        """Get the share which the components of the modeling class have on a shared maximum potential at a location."""
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar = getattr(pyM, "cap_" + abbrvName)
        capVarSet = getattr(pyM, "designDimensionVarSet_" + abbrvName)
        return sum(
            capVar[loc, compName, ip] / compDict[compName].processedCapacityMax[ip][loc]
            for compName in compDict
            if compDict[compName].sharedPotentialID == key
            and (loc, compName, ip) in capVarSet
        )

    def getEconomicsDesign(
        self,
        pyM,
        esM,
        factorNames,
        lifetimeAttr,
        varName,
        divisorName="",
        QPfactorNames=[],
        QPdivisorNames=[],
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
        QPfactorNames=[],
        QPdivisorNames=[],
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

    def extractRawResults(self, esM, pyM):
        """Extract the raw solved variable values into a results dictionary.

        The function is called after a successful optimization. It reads the solved pyomo
        design variables (capacity, commissioning, decommissioning, isBuilt) once per
        investment period and formats them with :func:`utils.formatOptimizationOutput`. It
        deliberately performs *no* economics calculations, big-M warnings or summary
        assembly. Subclass specific variables (e.g. operation, charge/discharge, state of
        charge) are added through the overridable :meth:`_extractSubclassRawResults` hook.

        The existing ``self._*VariablesOptimum`` attributes are populated from the extracted
        values so that the public getter behavior is preserved.

        **Required arguments**

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        The results are stored on the component as ``self._rawResults`` (native dimension)
        and ``self._rawResults1dim`` (1-dim companion used by the economics/summary phase).
        Nothing is returned.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar = getattr(esM.pyM, "cap_" + abbrvName)
        binVar = getattr(esM.pyM, "commisBin_" + abbrvName)
        commisVar = getattr(esM.pyM, "commis_" + abbrvName)
        decommisVar = getattr(esM.pyM, "decommis_" + abbrvName)

        # mapping of dict key -> (pyomo variable, attribute populated from it)
        designVars = [
            ("capacity", capVar, self._capacityVariablesOptimum),
            ("commissioning", commisVar, self._commissioningVariablesOptimum),
            ("decommissioning", decommisVar, self._decommissioningVariablesOptimum),
            ("isBuilt", binVar, self._isBuiltVariablesOptimum),
        ]

        rawResults = {}
        # 1-dim companion frames (used by the economics/summary phase in setOptimalValues)
        self._rawResults1dim = {}
        # drop rows contributed by expansion modules during a previous optimization
        self._extraSummaryRows = {}
        for ip in esM.investmentPeriods:
            ipName = esM.investmentPeriodNames[ip]
            rawResults[ipName] = {}
            self._rawResults1dim[ipName] = {}
            for varName, pyomoVar, optimumAttr in designVars:
                values = pyomoVar.get_values()
                optVal = utils.formatOptimizationOutput(
                    values, VarType.DESIGN, Dimension.ONE, ip
                )
                optVal_ = utils.formatOptimizationOutput(
                    values, VarType.DESIGN, self.dimension, ip, compDict=compDict
                )
                # NOTE (aliasing invariant): the ``self._*VariablesOptimum`` attribute and the
                # raw results dict deliberately share the *same* frame object (avoids doubling
                # the memory of the optima). ``self._rawResults`` is the single source of truth,
                # so these frames must be treated as immutable after optimization: never mutate
                # a ``*VariablesOptimum`` frame in place, or the summary/export read from
                # ``_rawResults`` would silently see the mutation. Readers copy before returning
                # (see :meth:`getResultOptimalValues` / :meth:`getResultSummaryDict`).
                optimumAttr[ipName] = optVal_
                rawResults[ipName][varName] = optVal_
                self._rawResults1dim[ipName][varName] = optVal

        self._rawResults = rawResults
        # let subclasses add their specific variables (operation, charge/discharge, ...)
        self._extractSubclassRawResults(esM, pyM, rawResults)

    def _extractSubclassRawResults(self, esM, pyM, rawResults):
        """Extract subclass specific raw variables (overridable hook).

        Subclasses add their variables (e.g. ``operation``) to ``rawResults`` in place and
        populate their own ``self._*VariablesOptimum`` attributes. The base implementation
        does nothing.

        :param esM: EnergySystemModel instance.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel.
        :type pyM: pyomo ConcreteModel

        :param rawResults: nested dictionary built by :meth:`extractRawResults`, keyed by
            investment period name then variable name. Mutated in place.
        :type rawResults: dict
        """
        pass

    def deriveEconomics(self, esM, pyM):
        """Derive the economic (cost) results and write them into ``self._rawResults``.

        Called after :meth:`extractRawResults` populated ``self._rawResults`` with the solved
        raw variables. It computes the cost rows that are shown in the optimization summary
        (``invest``, ``capexCap``, ``opexCap``, ``capexIfBuilt``, ``opexIfBuilt``,
        ``investLifetimeExtension``, ``revenueLifetimeShorteningResale``, ``TAC`` and
        ``NPVcontribution``) from the component parameters and the raw 1-dim variable frames
        (:attr:`_rawResults1dim`). The derived frames are stored into the *same*
        ``self._rawResults`` dictionary under keys that match the summary property rows, so
        that the later summary assembly is a mechanical write. Subclass specific cost
        contributions (e.g. a storage's charge/discharge opex) are added through the
        overridable :meth:`_deriveSubclassEconomics` hook. Nothing is returned; the results
        dictionary is mutated in place.

        **Required arguments**

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict = self.componentsDict

        # Get the design dependent cost contributions for all components (moved out of
        # setOptimalValues). Each call returns a dict keyed by investment period.
        resultsNPV_cx = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedInvestPerCapacity"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commis",
            divisorName="CCF",
            QPdivisorNames=["QPbound", "CCF"],
            getOptValue=True,
            getOptValueCostType=CostType.NPV,
        )

        resultsTAC_cx = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedInvestPerCapacity"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commis",
            divisorName="CCF",
            QPdivisorNames=["QPbound", "CCF"],
            getOptValue=True,
            getOptValueCostType=CostType.TAC,
        )

        resultsNPV_ox = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedOpexPerCapacity"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commis",
            QPdivisorNames=["QPbound"],
            getOptValue=True,
            getOptValueCostType=CostType.NPV,
        )

        resultsTAC_ox = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedOpexPerCapacity"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commis",
            QPdivisorNames=["QPbound"],
            getOptValue=True,
            getOptValueCostType=CostType.TAC,
        )

        # Get NPV contribution for investmentIfBuilt
        resultsNPV_cx_bin = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestIfBuilt"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commisBin",
            divisorName="CCF",
            getOptValue=True,
            getOptValueCostType=CostType.NPV,
        )

        # Calculate the annualized investment costs cx (CAPEX) if built
        resultsTAC_cx_bin = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestIfBuilt"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commisBin",
            divisorName="CCF",
            getOptValue=True,
            getOptValueCostType=CostType.TAC,
        )

        # Get NPV cost contribution for the annualized operational costs if built ox (OPEX)
        resultsNPV_ox_bin = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexIfBuilt"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commisBin",
            getOptValue=True,
            getOptValueCostType=CostType.NPV,
        )

        # Calculate the annualized operational costs if built ox (OPEX)
        resultTAC_ox_bin = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexIfBuilt"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commisBin",
            getOptValue=True,
            getOptValueCostType=CostType.TAC,
        )

        for ip in esM.investmentPeriods:
            ipName = esM.investmentPeriodNames[ip]
            # write the economic frames into the same results dict
            results_ip = self._rawResults[ipName]

            # Read raw solved design variables (1-dim) extracted by extractRawResults
            commisOptVal = self._rawResults1dim[ipName]["commissioning"]
            capOptVal = self._rawResults1dim[ipName]["capacity"]
            binCapOptVal = self._rawResults1dim[ipName]["isBuilt"]

            if capOptVal is not None:
                # Calculate the investment costs i (proportional to commissioning)
                i = commisOptVal.apply(
                    lambda commis: (
                        commis
                        * compDict[commis.name].processedInvestPerCapacity[ip]
                        * compDict[commis.name].QPcostDev[ip]
                        + (
                            compDict[commis.name].processedInvestPerCapacity[ip]
                            * compDict[commis.name].processedQPcostScale[ip]
                            / (compDict[commis.name].QPbound[ip])
                            * commis
                            * commis
                        )
                    ),
                    axis=1,
                )
                results_ip["invest"] = i

                # Annualized investment (CAPEX) and operational (OPEX) costs due to
                # capacity expansion.
                results_ip["capexCap"] = resultsTAC_cx[ip]
                results_ip["opexCap"] = resultsTAC_ox[ip]

                # add additional costs for lifetime extension or scrapping bonus if lifetime
                # is floored or ceiled to next interval
                investLifetimeExtension = pd.DataFrame(
                    0.0, index=i.index, columns=i.columns
                )
                revenueLifetimeShorteningResale = pd.DataFrame(
                    0.0, index=i.index, columns=i.columns
                )
                for component in i.index:
                    for loc in i.columns:
                        # only relevant if there is any invest
                        if np.isnan(i.loc[component, loc]):
                            val_investLifetimeExtension = 0
                            val_revenueLifetimeShorteningResale = 0
                        else:
                            techLifetime = compDict[component].technicalLifetime[loc]
                            econLifetime = compDict[component].economicLifetime[loc]
                            sameInterval = math.floor(
                                compDict[component].ipTechnicalLifetime[loc]
                            ) == math.floor(compDict[component].ipEconomicLifetime[loc])

                            # investLifetimeExtension
                            if (
                                esM.numberOfInvestmentPeriods > 1
                                and (techLifetime % esM.investmentPeriodInterval != 0)
                                and not compDict[component].floorTechnicalLifetime
                            ):
                                intervalPart = 1 - (
                                    compDict[component].ipTechnicalLifetime[loc] % 1
                                )
                                val_investLifetimeExtension = (
                                    i.loc[component, loc]
                                    * intervalPart
                                    / compDict[component].ipEconomicLifetime[loc]
                                )
                            else:
                                val_investLifetimeExtension = 0

                            # revenueLifetimeShorteningResale
                            if (
                                esM.numberOfInvestmentPeriods > 1
                                and econLifetime % esM.investmentPeriodInterval != 0
                                and compDict[component].floorTechnicalLifetime
                                and sameInterval
                            ):
                                intervalPart = (
                                    compDict[component].ipEconomicLifetime[loc] % 1
                                )
                                val_revenueLifetimeShorteningResale = (
                                    i.loc[component, loc]
                                    * intervalPart
                                    / compDict[component].ipEconomicLifetime[loc]
                                )
                            else:
                                val_revenueLifetimeShorteningResale = 0

                        investLifetimeExtension.loc[component, loc] = (
                            val_investLifetimeExtension
                        )
                        revenueLifetimeShorteningResale.loc[component, loc] = (
                            val_revenueLifetimeShorteningResale
                        )

                results_ip["investLifetimeExtension"] = investLifetimeExtension
                results_ip["revenueLifetimeShorteningResale"] = (
                    revenueLifetimeShorteningResale
                )

            if binCapOptVal is not None:
                # Calculate the investment costs i (fix value if component is built)
                i_bin = binCapOptVal.apply(
                    lambda dec: dec * compDict[dec.name].processedInvestIfBuilt[ip],
                    axis=1,
                )
                # invest is the sum of capacity expansion and isBuilt contributions
                if "invest" in results_ip:
                    results_ip["invest"] = results_ip["invest"].add(i_bin, fill_value=0)
                else:
                    results_ip["invest"] = i_bin

                # Annualized investment (CAPEX) and operational (OPEX) costs if built
                results_ip["capexIfBuilt"] = resultsTAC_cx_bin[ip]
                results_ip["opexIfBuilt"] = resultTAC_ox_bin[ip]

            # Summarize all annualized contributions to the total annual cost. Mirrors the
            # former summary groupby (note: opexIfBuilt is intentionally not part of the TAC,
            # matching the previous behavior).
            tacParts = [
                results_ip[key]
                for key in ("capexCap", "opexCap", "capexIfBuilt")
                if key in results_ip
            ]
            if tacParts:
                results_ip["TAC"] = pd.concat(tacParts).groupby(level=0).sum()

            # Net present value contribution
            npv = pd.DataFrame()
            if capOptVal is not None:
                npv = npv.add(resultsNPV_cx[ip], fill_value=0)
                npv = npv.add(resultsNPV_ox[ip], fill_value=0)
            if binCapOptVal is not None:
                npv = npv.add(resultsNPV_cx_bin[ip], fill_value=0)
                npv = npv.add(resultsNPV_ox_bin[ip], fill_value=0)
            results_ip["NPVcontribution"] = npv

        # let subclasses add their specific cost contributions (e.g. charge/discharge opex)
        self._deriveSubclassEconomics(esM, pyM, self._rawResults)

        # Label the derived economic frames so the raw results dict is self-describing,
        # consistent with the solved-variable frames named in utils.formatOptimizationOutput.
        # All derived economics are flat ``component x location`` (2-dim connections are the
        # 1-dim pseudo-location convention) frames; the raw design/operation frames are already
        # named and either multi-level (2-dim, operation) or time-columned, so this sweep only
        # touches the flat component-indexed cost frames and needs no per-cost-term upkeep.
        for ipResults in self._rawResults.values():
            for frame in ipResults.values():
                if (
                    isinstance(frame, pd.DataFrame)
                    and frame.index.nlevels == 1
                    and frame.columns.nlevels == 1
                    and frame.index.name != "component"
                ):
                    frame.index = frame.index.set_names("component")
                    frame.columns = frame.columns.set_names("location")

    def _deriveSubclassEconomics(self, esM, pyM, rawResults):
        """Add subclass specific economic (cost) contributions (overridable hook).

        Subclasses add their cost frames (e.g. a storage's ``opexCharge``/``opexDischarge``)
        to ``rawResults`` in place and, where applicable, augment the aggregated ``TAC`` and
        ``NPVcontribution`` frames. The base implementation does nothing.

        :param esM: EnergySystemModel instance.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel.
        :type pyM: pyomo ConcreteModel

        :param rawResults: nested dictionary built by :meth:`extractRawResults` and extended
            by :meth:`deriveEconomics`, keyed by investment period name then variable name.
            Mutated in place.
        :type rawResults: dict
        """
        pass

    def setOptimalValues(self, esM, pyM, indexColumns, plantUnit, unitApp=""):
        r"""Set the optimal values for the considered components and return a summary of them.
        The function is called after optimization was successful and an optimal solution was found.
        Each sub class of the component class calls this function for setting the common optimal values,
        e.g. investment and maintenance costs proportional to optimal capacity expansion.

        **Required arguments**

        :param esM: EnergySystemModel instance representing the energy system in which the components are modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param ip: investment period of transformation path analysis.
        :type ip: int

        :param indexColumns: set of strings with the columns indices of the summary. The indices represent the locations
            or connections between the locations are used to call the optimal values of the variables of the components
            in the model class.
        :type indexColumns: set

        :param plantUnit: attribute of the component that describes the unit of the plants to which maximum capacity
            limitations, cost parameters and the operation time series refer to. Depending on the considered component,
            possible inputs are "commodityUnit" (e.g. for transmission components) or "physicalUnit" (e.g. for
            conversion components).
        :type plantUnit: string

        **Default arguments**

        :param unitApp: string which appends the capacity unit in the optimization summary.
            For example, for the StorageModel class, the parameter is set to '\\*h'.
            |br| * the default value is ''.
        :type unitApp: string

        :return: summary of the optimized values.
        :rtype: pandas DataFrame
        """
        # Extract raw solved variables (economics-independent) and populate the
        # *VariablesOptimum attributes, then derive the economic (cost) frames into the
        # results dict. The summary returned below is a mechanical view of those frames.
        self.extractRawResults(esM, pyM)
        self.deriveEconomics(esM, pyM)
        return self._buildOptimizationSummary(esM, indexColumns, plantUnit, unitApp)

    def _convertOptimalValueNames(self, esM):
        """Rename the internal ``_*VariablesOptimum``/``_optSummary`` attributes to their
        public names (e.g. ``_capacityVariablesOptimum`` -> ``capacityVariablesOptimum``).

        For a perfect-foresight run the per-investment-period dict is kept as is; for a
        single-year optimization it is unwrapped to the one dataframe it contains, so that
        models built before the multi-investment-period support was added keep working.

        Called by :meth:`fine.energySystemModel.EnergySystemModel.optimize` once per
        modeling class, after its ``setOptimalValues`` has run and set ``self._optSummary``.
        It is driven from there rather than from the individual ``setOptimalValues``
        implementations so that a class overriding ``setOptimalValues`` cannot omit it.

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance
        """
        optimalValueParameters = [
            "_optSummary",
            "_stateOfChargeOperationVariablesOptimum",
            "_chargeOperationVariablesOptimum",
            "_dischargeOperationVariablesOptimum",
            "_phaseAngleVariablesOptimum",
            "_operationVariablesOptimum",
            "_discretizationPointVariablesOptimum",
            "_discretizationSegmentConVariablesOptimum",
            "_discretizationSegmentBinVariablesOptimum",
            "_capacityVariablesOptimum",
            "_isBuiltVariablesOptimum",
            "_commissioningVariablesOptimum",
            "_decommissioningVariablesOptimum",
        ]

        for key in optimalValueParameters:
            if key not in self.__dict__:
                continue
            # strip only the leading underscore; key.replace("_", "") would also drop
            # underscores inside a name
            publicName = key[1:]
            if esM.numberOfInvestmentPeriods == 1:
                setattr(
                    self,
                    publicName,
                    getattr(self, key)[esM.investmentPeriodNames[0]],
                )
            else:
                setattr(self, publicName, getattr(self, key))

    def _connectionLocationMap(self, esM):
        """Build (and cache) the ``"locIn_locOut" -> (locIn, locOut)`` map for 2-dim connection splitting.

        Built once per (component-model, location set) and rebuilt only if ``esM.locations``
        changes, avoiding the O(locations^2) rebuild on every summary/export call.

        :param esM: EnergySystemModel instance (provides ``locations``).
        :rtype: dict
        """
        cache = getattr(self, "_connLocMapCache", None)
        if cache is None or cache[0] != esM.locations:
            mapC = {
                l1 + "_" + l2: (l1, l2) for l1 in esM.locations for l2 in esM.locations
            }
            cache = (set(esM.locations), mapC)
            self._connLocMapCache = cache
        return cache[1]

    def _economicSummaryUnits(self, esM):
        """Property -> unit string for the derived economic summary rows.

        Single source for these units; consumed by both the optimization summary
        (:meth:`_buildOptimizationSummary`) and the export (:meth:`getResultSummaryDict`).

        :param esM: EnergySystemModel instance (provides ``costUnit``).
        :return: ordered ``{property: unitString}`` mapping.
        :rtype: dict
        """
        perA = "[" + esM.costUnit + "/a]"
        cost = "[" + esM.costUnit + "]"
        return {
            "capexCap": perA,
            "capexIfBuilt": perA,
            "opexCap": perA,
            "opexIfBuilt": perA,
            "TAC": perA,
            "NPVcontribution": cost,
            "invest": cost,
            "investLifetimeExtension": cost,
            "revenueLifetimeShorteningResale": cost,
        }

    def _buildOptimizationSummary(self, esM, indexColumns, plantUnit, unitApp=""):
        r"""Assemble the optimization summary as a view of ``self._rawResults``.

        Called by :meth:`setOptimalValues` after :meth:`extractRawResults` and
        :meth:`deriveEconomics` have populated ``self._rawResults`` (raw solved variables and
        derived economic frames) and ``self._rawResults1dim`` (1-dim companions). This method
        performs no extraction or economics; it only writes the already computed frames into
        the ``(Component, Property, Unit) x Locations`` summary DataFrame, keyed per
        investment period. It also emits the big-M proximity warning for capacities close to
        the chosen big-M value.

        **Required arguments**

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance

        :param indexColumns: set of strings with the column indices of the summary (locations
            or connections between locations).
        :type indexColumns: set

        :param plantUnit: attribute of the component that describes the unit of the plants
            (e.g. "commodityUnit" or "physicalUnit").
        :type plantUnit: string

        **Default arguments**

        :param unitApp: string appended to the capacity unit in the summary (e.g. '\\*h' for
            storage). |br| * the default value is ''.
        :type unitApp: string

        :return: summary of the optimized values, keyed by investment period name.
        :rtype: dict
        """
        compDict = self.componentsDict

        # Single source of truth for the summary's (Property -> Unit) rows. The design rows
        # (capacity/commissioning/decommissioning) carry a per-component plant unit resolved
        # below and are marked with ``None``; every other unit is fixed. The economic units
        # are shared with the export via :meth:`_economicSummaryUnits`. The same mapping
        # drives both the MultiIndex and the economic-frame write loop further down.
        summaryUnits = {
            "capacity": None,
            "commissioning": None,
            "decommissioning": None,
            "isBuilt": "[-]",
            **self._economicSummaryUnits(esM),
        }
        # Design rows are written explicitly below (from the 1-dim frames, with their own
        # conditionals); the remaining rows are the economic frames derived by deriveEconomics.
        designProps = ("capacity", "commissioning", "decommissioning", "isBuilt")

        def resolveUnit(compName, prop):
            # ``None`` marks a capacity-like row whose unit is the component's plant unit.
            unit = summaryUnits[prop]
            if unit is None:
                unit = "[" + getattr(compDict[compName], plantUnit) + unitApp + "]"
            return unit

        mIndex = pd.MultiIndex.from_tuples(
            [
                (compName, prop, resolveUnit(compName, prop))
                for compName in compDict.keys()
                for prop in summaryUnits
            ],
            names=["Component", "Property", "Unit"],
        )

        optSummary = {}
        for ip in esM.investmentPeriods:
            ipName = esM.investmentPeriodNames[ip]
            optSummary_ip = pd.DataFrame(
                index=mIndex, columns=sorted(indexColumns)
            ).sort_index()
            # raw + derived economic frames produced by extractRawResults / deriveEconomics
            results_ip = self._rawResults[ipName]

            # Read raw solved design variables (1-dim) extracted by extractRawResults
            capOptVal = self._rawResults1dim[ipName]["capacity"]
            commisOptVal = self._rawResults1dim[ipName]["commissioning"]
            decommisOptVal = self._rawResults1dim[ipName]["decommissioning"]
            binCapOptVal = self._rawResults1dim[ipName]["isBuilt"]

            if capOptVal is not None:
                # Check if the installed capacities are close to a bigM val
                # ue for components with design decision variables but
                # ignores cases where bigM was substituted by capacityMax parameter (see bigM constraint
                for compName, comp in compDict.items():
                    if (
                        comp.hasIsBuiltBinaryVariable
                        and (comp.processedCapacityMax is None)
                        and capOptVal.loc[compName].max() >= comp.bigM * 0.9
                        and esM.verboseLogLevel < 2
                    ):
                        warnings.warn(
                            "the capacity of component "
                            + compName
                            + " is in one or more locations close "
                            + "or equal to the chosen Big M. Consider rerunning the simulation with a higher"
                            + " Big M."
                        )

                # Fill the optimization summary with the optimal capacities.
                optSummary_ip.loc[
                    [
                        (
                            ix,
                            "capacity",
                            "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                        )
                        for ix in capOptVal.index
                    ],
                    capOptVal.columns,
                ] = capOptVal.values

            # Fill the optimization summary with the isBuilt decisions.
            if binCapOptVal is not None:
                optSummary_ip.loc[
                    [(ix, "isBuilt", "[-]") for ix in binCapOptVal.index],
                    binCapOptVal.columns,
                ] = binCapOptVal.values

            # Get and set optimal values for commissioning and decommissioning
            # not applicable for singleyear optimization, hence dropped from summary
            # either decommissioning or capacity exists
            # (years can have decommissioning, leading to no left capacity)
            if decommisOptVal is not None or capOptVal is not None:
                # commissioning
                optSummary_ip.loc[
                    [
                        (
                            ix,
                            "commissioning",
                            "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                        )
                        for ix in commisOptVal.index
                    ],
                    commisOptVal.columns,
                ] = commisOptVal.values
                # decommissioning
                optSummary_ip.loc[
                    [
                        (
                            ix,
                            "decommissioning",
                            "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                        )
                        for ix in decommisOptVal.index
                    ],
                    decommisOptVal.columns,
                ] = decommisOptVal.values

            # Fill the optimization summary with the derived economic frames (invest,
            # capexCap, opexCap, capexIfBuilt, opexIfBuilt, lifetime corrections, TAC and
            # NPVcontribution) computed by deriveEconomics.
            # The lifetime correction rows are written cell-wise to keep their numpy scalar
            # dtype (as in the former inline implementation).
            perCellProps = (
                "investLifetimeExtension",
                "revenueLifetimeShorteningResale",
            )
            for prop, unit in summaryUnits.items():
                if prop in designProps or prop not in results_ip:
                    continue
                frame = results_ip[prop]
                if frame.empty:
                    continue
                if prop in perCellProps:
                    for component in frame.index:
                        for loc in frame.columns:
                            optSummary_ip.loc[(component, prop, unit), loc] = frame.loc[
                                component, loc
                            ]
                else:
                    optSummary_ip.loc[
                        [(ix, prop, unit) for ix in frame.index],
                        frame.columns,
                    ] = frame.values

            optSummary[ipName] = optSummary_ip

        return optSummary

    def getOptimalValues(self, name="all", ip=0):
        """Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariablesOptimum',
            * 'isBuiltVariablesOptimum',
            * 'operationVariablesOptimum',
            * 'commissioningVariablesOptimum'
            * 'decommissioningVariablesOptimum'
            * 'all' or another input: all variables are returned.

        :type name: string

        The returned frames carry named axes: the index is ``component`` and ``location``
        (1-dim) or ``component``, ``locationIn`` and ``locationOut`` (2-dim), and for
        operation variables the columns are ``time``. Code that relies on the axes being
        unnamed (e.g. ``reset_index()`` producing ``level_0``/``level_1`` columns) has to
        be adapted.

        :param ip: investment period of transformation path analysis.
            |br| * the default value is 0
        :type ip: int

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        timeDependentMapping = {
            "capacityVariablesOptimum": False,
            "isBuiltVariablesOptimum": False,
            "operationVariablesOptimum": True,
            "commissioningVariablesOptimum": False,
            "decommissioningVariablesOptimum": False,
        }

        if name in timeDependentMapping:
            return {
                "values": getattr(self, f"_{name}")[ip],
                "timeDependent": timeDependentMapping[name],
                "dimension": self.dimension,
            }
        return {
            valName: {
                "values": getattr(self, f"_{valName}")[ip],
                "timeDependent": timeDependentMapping[valName],
                "dimension": self.dimension,
            }
            for valName in timeDependentMapping
        }

    # ------------------------------------------------------------------
    # Results-dict accessors used by the xarray/netCDF export.
    # They read directly from the raw results dict (``self._rawResults`` /
    # ``self._rawResults1dim``), so
    # :func:`fine.IOManagement.xarrayIO.convertOptimizationOutputToDatasets` neither reparses
    # the summary DataFrame nor calls getOptimalValues - the dict is the single source of
    # truth for both the summary and the export.
    # ------------------------------------------------------------------

    def _requireRawResults(self, ip):
        """Return ``self._rawResults[ip]``, with an explanatory error if it is not available.

        The raw results dict is populated by :meth:`extractRawResults` /
        :meth:`deriveEconomics` during ``esM.optimize()``. It is deliberately *not*
        reconstructed when an EnergySystemModel is read back from a netCDF file
        (:func:`fine.IOManagement.xarrayIO.readNetCDFtoEnergySystemModel` restores the
        optimization summary and the ``*VariablesOptimum`` attributes only), so results loaded
        from a file cannot be exported again without re-optimizing.

        :param ip: investment period name (key into ``self._rawResults``).
        :type ip: string

        :return: the raw results of that investment period.
        :rtype: dict
        """
        if not self._rawResults:
            raise RuntimeError(
                f"No raw optimization results available for '{type(self).__name__}'. They are "
                "created by esM.optimize() and are not restored when an EnergySystemModel is "
                "read from a netCDF file - re-optimize the model before exporting its results."
            )
        if ip not in self._rawResults:
            raise KeyError(
                f"No raw optimization results for investment period '{ip}' in "
                f"'{type(self).__name__}'. Available: {sorted(self._rawResults)}."
            )
        return self._rawResults[ip]

    def _exportOptimumVarMap(self):
        """Map raw result keys to the optimum variable names used by the export.

        Each entry carries an explicit ``dimension`` so a variable can be shaped
        independently of the component's own dimension (e.g. the LOPF phase angle is
        1-dim on a 2-dim component). Subclasses extend this list with their own
        variables (see :meth:`fine.storage.StorageModel._exportOptimumVarMap`).

        :return: list of ``(rawResultsKey, optimumVariableName, timeDependent, dimension)``
            tuples in the same order/meaning as :meth:`getOptimalValues`.
        :rtype: list
        """
        d = self.dimension
        return [
            ("capacity", "capacityVariablesOptimum", False, d),
            ("isBuilt", "isBuiltVariablesOptimum", False, d),
            ("operation", "operationVariablesOptimum", True, d),
            ("commissioning", "commissioningVariablesOptimum", False, d),
            ("decommissioning", "decommissioningVariablesOptimum", False, d),
        ]

    def getResultOptimalValues(self, ip):
        """Return the design/operation optima for the export, read from ``self._rawResults``.

        Per component, each optimum variable is shaped into a ``Series`` ready for
        ``.to_xarray()`` (same structure / dimension names the export produced before). The
        values are the exact same frame objects the optimum attributes hold (both are populated
        from the same object in :meth:`extractRawResults`), so the export output is unchanged.
        These optima carry no unit, so the entries pair the series with ``None``.

        :param ip: investment period name (key into ``self._rawResults``).
        :type ip: string

        :return: ``{componentName: {optimumVariableName: (values, None)}}``.
        :rtype: dict
        """
        results_ip = self._requireRawResults(ip)
        out = {compName: {} for compName in self.componentsDict}
        for rawKey, optName, timeDependent, dimension in self._exportOptimumVarMap():
            frame = results_ip.get(rawKey)
            if frame is None:
                continue
            for compName in self.componentsDict:
                if compName not in frame.index.get_level_values(0):
                    continue
                series = self._shapeOptimumResult(
                    frame.loc[compName], optName, timeDependent, dimension
                )
                out[compName][optName] = (series, None)
        return out

    def _shapeOptimumResult(self, sub, name, timeDependent, dimension):
        """Shape a single component's optimum frame into a ``to_xarray``-ready ``Series``.

        Reproduces the per-case index handling the export applied to ``getOptimalValues`` output:
        time-dependent rows gain a ``time`` dimension; 2-dim rows are split into
        ``(locationIn, locationOut)`` (the time-independent 2-dim case keeps the historical
        transpose). The shaping uses the variable's own ``dimension`` (from
        :meth:`_exportOptimumVarMap`), which may differ from ``self.dimension`` (e.g. the LOPF
        phase angle is 1-dim on a 2-dim component).

        Variables that carry an extra index level beyond ``location`` (e.g. the part-load
        discretization point/segment variables, indexed by ``(discretizationIndex, location)``
        per component) keep that level so each variable exports under its own name instead of
        colliding on an anonymous stacked column. The extra level names are propagated from the
        frame (labelled in :func:`utils.formatOptimizationOutput`) rather than re-derived here.

        :param sub: the component slice ``frame.loc[component]``.
        :param name: variable name (becomes the data variable name).
        :param timeDependent: whether the variable carries a ``time`` dimension.
        :param dimension: ``"1dim"`` or ``"2dim"`` shaping to apply to this variable.

        :rtype: pandas.Series
        """
        if timeDependent and dimension == Dimension.ONE:
            subT = sub.T
            if subT.columns.nlevels == 1:
                series = subT.stack()
                series.index = series.index.rename(["time", "location"])
            else:
                # extra index levels (e.g. discretizationIndex) sit before location; stack
                # every column level so nothing is lost or collides on export, keeping the
                # level names set by formatOptimizationOutput.
                series = subT.stack(list(range(subT.columns.nlevels)))
                extraNames = list(sub.index.names[:-1])
                series.index = series.index.rename(["time", *extraNames, "location"])
        elif timeDependent and dimension == Dimension.TWO:
            series = sub.stack()
            series.index = series.index.rename(["locationIn", "locationOut", "time"])
            series = series.reorder_levels(["time", "locationIn", "locationOut"])
        elif not timeDependent and dimension == Dimension.ONE:
            series = sub.rename_axis("location")
        else:  # time-independent 2-dim
            series = sub.T.stack()
            series.index = series.index.rename(["locationIn", "locationOut"])
        series = series.copy()
        series.name = name
        return series

    def _summaryPlantUnit(self):
        """(plant unit attribute name, capacity unit suffix) for the design summary rows.

        :rtype: tuple(str, str)
        """
        return "commodityUnit", ""

    def _subclassSummaryFrames(self, esM, ip):
        """Operation summary rows (per subclass) derived from ``self._rawResults``.

        These frames are the single source of the aggregated operation rows: they feed the
        optimization summary (written into the summary skeleton by
        :meth:`_writeOperationSummaryRows`) and the raw-results export accessor
        (:meth:`getResultSummaryDict`).

        :return: ordered list of ``(property, frame, unitFn)`` where ``frame`` is indexed by
            component with locations (1dim) / connections (2dim) as columns, and ``unitFn`` maps
            a component name to its unit string. The base class has no operation rows.
        :rtype: list
        """
        return []

    def _writeOperationSummaryRows(self, optSummary, esM, ipName):
        """Write the subclass operation rows (:meth:`_subclassSummaryFrames`) into the summary.

        Shared by every subclass' ``_buildSubclassOptimizationSummary`` so the operation
        aggregation is computed once (in :meth:`_subclassSummaryFrames`) and reused by the
        summary and the raw-results export accessor.

        :param optSummary: summary skeleton with a ``(Component, Property, Unit) x columns``
            MultiIndex; filled in place.
        :param esM: EnergySystemModel instance.
        :param ipName: investment period name (key into ``self._rawResults``).

        :return: the ``{property: frame}`` mapping (so callers can reuse the aggregated frames,
            e.g. for the storage charge/discharge warning).
        :rtype: dict
        """
        framesByProp = {}
        for prop, frame, unit in self._subclassSummaryFrames(esM, ipName):
            framesByProp[prop] = frame
            if frame is None or frame.empty:
                continue
            optSummary.loc[
                [
                    (ix, prop, unit(ix) if callable(unit) else unit)
                    for ix in frame.index
                ],
                frame.columns,
            ] = frame.values
        return framesByProp

    def getResultSummaryDict(self, esM, ip):
        """Assemble the time-independent summary results for the export from ``self._rawResults``.

        Reproduces, per component, the variable -> (values, unit) entries the export previously
        obtained by re-parsing ``getOptimizationSummary``. Design rows come from
        ``self._rawResults1dim`` and the derived economic rows from ``self._rawResults``; the
        subclass operation rows are added through :meth:`_subclassSummaryFrames`. For 1-dim
        components a value is a per-location ``Series`` (NaN-filled where absent, matching the
        summary's fixed columns); for 2-dim components it is a ``Series`` indexed by
        ``(locationIn, locationOut)`` with absent/NaN connections dropped (matching the summary's
        ``stack``), or ``None`` when the whole property is absent.

        :param esM: EnergySystemModel instance.
        :type esM: EnergySystemModel instance

        :param ip: investment period name (key into ``self._rawResults``).
        :type ip: string

        :return: ``{componentName: {property: (values, unit)}}``.
        :rtype: dict
        """
        compDict = self.componentsDict
        results_ip = self._requireRawResults(ip)
        results1dim_ip = self._rawResults1dim[ip]
        plantUnit, unitApp = self._summaryPlantUnit()

        def plantUnitFn(compName, suffix):
            return "[" + getattr(compDict[compName], plantUnit) + suffix + "]"

        # (property, frame, unit) in the order the summary lists them. ``unit`` is either a
        # fixed string or a callable ``comp -> unit`` for the per-component plant units. Design
        # rows use the 1-dim companion frames; economic rows use the derived frames. The
        # economic units are shared with the summary via :meth:`_economicSummaryUnits`.
        designRows = [
            (
                "capacity",
                results1dim_ip.get("capacity"),
                lambda c: plantUnitFn(c, unitApp),
            ),
            (
                "commissioning",
                results1dim_ip.get("commissioning"),
                lambda c: plantUnitFn(c, unitApp),
            ),
            (
                "decommissioning",
                results1dim_ip.get("decommissioning"),
                lambda c: plantUnitFn(c, unitApp),
            ),
            ("isBuilt", results1dim_ip.get("isBuilt"), "[-]"),
        ]
        econRows = [
            (prop, results_ip.get(prop), unit)
            for prop, unit in self._economicSummaryUnits(esM).items()
        ]
        rows = designRows + econRows + self._subclassSummaryFrames(esM, ip)

        mapC = self._connectionLocationMap(esM)
        out = {compName: {} for compName in compDict}
        for compName in compDict:
            for prop, frame, unit in rows:
                values = self._extractComponentResult(frame, compName, esM, mapC)
                if values is None:
                    continue
                series = self._nameResultSeries(pd.to_numeric(values), prop)
                out[compName][prop] = (
                    series,
                    unit(compName) if callable(unit) else unit,
                )

        # Rows registered by expansion modules exist only for the components they apply to
        # (unlike the fixed property set above, which is NaN-filled for 1-dim components), so
        # they are only emitted where the component is actually present in the frame.
        for prop, frame, unit in self._extraSummaryRows.get(ip, []):
            for compName in compDict:
                if frame is None or compName not in frame.index:
                    continue
                values = self._extractComponentResult(frame, compName, esM, mapC)
                if values is None:
                    continue
                series = self._nameResultSeries(pd.to_numeric(values), prop)
                out[compName][prop] = (
                    series,
                    unit(compName) if callable(unit) else unit,
                )
        return out

    def registerExtraSummaryRows(self, ip, rows):
        """Publish result frames produced *after* :meth:`setOptimalValues` into the results dict.

        Expansion modules that run once the component models are done (currently
        :class:`fine.expansionModules.piecewiseLinearCostFunction.PiecewiseLinearCostFunctionModel`,
        called from ``EnergySystemModel.optimize``) contribute additional result rows. They are
        stored in ``self._rawResults[ip]`` so the dict stays the single source of truth, and
        registered in ``self._extraSummaryRows`` so :meth:`getResultSummaryDict` exports them -
        the caller builds its optimization summary rows from the very same frames.

        :param ip: investment period name (key into ``self._rawResults``).
        :type ip: string

        :param rows: ``(property, frame, unit)`` triples - the same shape as
            :meth:`_subclassSummaryFrames` - where ``frame`` is indexed by component with one
            column per location and ``unit`` is a unit string or a ``component -> unit`` callable.
        :type rows: list

        :return: nothing; the frames are stored on the modeling class.
        """
        if not self._rawResults or ip not in self._rawResults or not rows:
            return
        if self.dimension != Dimension.ONE:
            # The frames are indexed by location, while a 2-dim model's result frames are
            # indexed by connection - there is no meaningful mapping, so nothing is published.
            warnings.warn(
                f"Extra result rows for the 2-dimensional modeling class "
                f"'{type(self).__name__}' cannot be added to the result export.",
                UserWarning,
            )
            return

        for prop, frame, _ in rows:
            self._rawResults[ip][prop] = frame
        self._extraSummaryRows.setdefault(ip, []).extend(rows)

    def _nameResultSeries(self, series, name):
        """Set the variable name and dimension-specific index names for the export.

        :param series: per-component result series (1dim: index = locations; 2dim: index =
            ``(locationIn, locationOut)`` tuples).
        :param name: variable name (becomes the data variable name after ``to_xarray``).

        :return: the same series, ready for ``.to_xarray()``.
        :rtype: pandas.Series
        """
        series = series.copy()
        series.name = name
        if self.dimension == Dimension.ONE:
            series.index = series.index.rename("location")
        else:
            series.index = series.index.rename(["locationIn", "locationOut"])
        return series

    def _extractComponentResult(self, frame, compName, esM, mapC):
        """Shape a class-level result frame into the per-component export values.

        :param frame: frame indexed by component, columns are locations (1dim) or connections
            (2dim); may be ``None``.
        :param compName: component name to extract.
        :param mapC: mapping ``"locIn_locOut" -> (locIn, locOut)`` for the 2-dim split.

        :return: per-location ``Series`` (1dim, NaN-filled), ``(locationIn, locationOut)``
            ``Series`` (2dim, NaN dropped) or ``None`` to skip the variable.
        :rtype: pandas.Series or None
        """
        if self.dimension == Dimension.ONE:
            locations = sorted(esM.locations)
            if frame is None or compName not in frame.index:
                return pd.Series(np.nan, index=locations)
            return frame.loc[compName].reindex(locations)
        # 2dim: split connection columns into (locationIn, locationOut), dropping NaN
        if frame is None or compName not in frame.index:
            return None
        row = frame.loc[compName]
        index, values = [], []
        for connection, value in row.items():
            if pd.isna(value):
                continue
            index.append(mapC[connection])
            values.append(value)
        if not index:
            return None
        return pd.Series(values, index=pd.MultiIndex.from_tuples(index))
