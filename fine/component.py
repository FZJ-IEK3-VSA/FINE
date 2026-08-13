from abc import ABCMeta, abstractmethod
from fine import utils
from fine.enums import Dimension
from fine.results.economics import ComponentEconomicsMixin
from fine.results.mixin import ComponentResultsMixin
import fine
import warnings
import pyomo.environ as pyomo
import pandas as pd
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


class ComponentModel(ComponentEconomicsMixin, ComponentResultsMixin, metaclass=ABCMeta):
    """The ComponentModel class provides the general methods used for modeling the components.
    Every model class of the several component technologies inherits from the ComponentModel class.
    Within the ComponentModel class, general valid sets, variables and constraints are declared.

    The economic contributions and the post-processing of a solved model are contributed by
    :class:`fine.results.economics.ComponentEconomicsMixin` and
    :class:`fine.results.mixin.ComponentResultsMixin`; a modeling class inherits them
    through this class and overrides their hooks as before.
    """

    def __init__(self):
        """Create a ComponentModel class instance."""
        super().__init__()
        self.abbrvName = ""
        self.dimension = ""
        self.componentsDict = {}

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
