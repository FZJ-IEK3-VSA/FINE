from typing import Type
from FINE.component import Component, ComponentModel
from FINE import utils
import warnings
import pyomo.environ as pyomo
import pandas as pd


class Transmission(Component):
    """
    A Transmission component can transmit a commodity between locations of the energy system.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
        losses=0,
        distances=None,
        hasCapacityVariable=True,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=1,
        hasIsBuiltBinaryVariable=False,
        bigM=None,
        operationRateMax=None,
        operationRateFix=None,
        tsaWeight=1,
        locationalEligibility=None,
        capacityMin=None,
        capacityMax=None,
        partLoadMin=None,
        sharedPotentialID=None,
        linkedQuantityID=None,
        capacityFix=None,
        isBuiltFix=None,
        investPerCapacity=0,
        investIfBuilt=0,
        opexPerOperation=0,
        opexPerCapacity=0,
        opexIfBuilt=0,
        QPcostScale=0,
        interestRate=0.08,
        economicLifetime=10,
        technicalLifetime=None,
        floorTechnicalLifetime=True,
        balanceLimitID=None,
        pathwayBalanceLimitID=None,
        stockCommissioning=None,
    ):
        """
        Constructor for creating an Transmission class instance.
        The Transmission component specific input arguments are described below. The general component
        input arguments are described in the Component class.

        **Required arguments:**

        :param commodity: to the component related commodity.
        :type commodity: string

        **Default arguments:**

        :param losses: relative losses per lengthUnit (lengthUnit as specified in the energy system model) in
            percentage of the commodity flow. This loss factor can capture simple linear losses

            .. math::
                trans_{in, ij} = (1 - \\text{losses} \\cdot \\text{distances}) \\cdot trans_{out, ij}

            (with trans being the commodity flow at a certain point in
            time and i and j being locations in the energy system). The losses can either be given as a float or a
            Pandas DataFrame with location specific values.
            |br| * the default value is 0
        :type losses: positive float (0 <= float <= 1) or Pandas DataFrame with positive values
            (0 <= float <= 1). The row and column indices of the DataFrame have to equal the in the energy
            system model specified locations.

        :param distances: distances between locations given in the lengthUnit (lengthUnit as specified in
            the energy system model).
            |br| * the default value is None
        :type distances: positive float (>= 0) or Pandas DataFrame with positive values (>= 0). The row and
            column indices of the DataFrame have to equal the in the energy system model specified locations.

        :param operationRateMax: if specified, indicates a maximum operation rate for all possible connections
            (both directions) of the transmission component at each time step, if required also for each investment period, by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateMax:
            * None
            * Pandas DataFrame with positive (>= 0) entries. The row indices have
              to match the in the energy system model specified time steps. The column indices are combinations
              of locations (as defined in the energy system model), separated by a underscore (e.g.
              "location1_location2"). The first location indicates where the commodity is coming from. The second
              location indicates where the commodity is going too. If a flow is specified from location i to
              location j, it also has to be specified from j to i.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param operationRateFix: if specified, indicates a fixed operation rate for all possible connections
            (both directions) of the transmission component at each time step, if required also for each investment period, by a positive float. If
            hasCapacityVariable is set to True, the values are given relative to the installed capacities (i.e.
            a value of 1 indicates a utilization of 100% of the capacity). If hasCapacityVariable
            is set to False, the values are given as absolute values in form of the commodityUnit,
            referring to the transmitted commodity (before considering losses) during one time step.
            |br| * the default value is None
        :type operationRateFix:
            * None
            * Pandas DataFrame with positive (>= 0). The row indices have
              to match the in the energy system model specified time steps. The column indices are combinations
              of locations (as defined in the energy system model), separated by a underscore (e.g.
              "location1_location2"). The first location indicates where the commodity is coming from. The second
              one location indicates where the commodity is going too. If a flow is specified from location i to
              location j, it also has to be specified from j to i.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param tsaWeight: weight with which the time series of the component should be considered when applying
            time series aggregation.
            |br| * the default value is 1
        :type tsaWeight: positive (>= 0) float

        :param opexPerOperation: describes the cost for one unit of the operation.
            The cost which is directly proportional to the operation of the component is obtained by multiplying
            the opexPerOperation parameter with the annual sum of the operational time series of the components.
            The opexPerOperation can either be given as a float or a Pandas DataFrame with location specific values or a dictionary per investment period with one of the previous options.
            The cost unit in which the parameter is given has to match the one specified in the energy
            system model (e.g. Euro, Dollar, 1e6 Euro). The value has to match the unit costUnit/operationUnit
            (e.g. Euro/kWh, Dollar/kWh).
            |br| * the default value is 0
        :type opexPerOperation:
            * positive (>=0) float
            * Pandas DataFrame with positive (>=0).The row and column indices of the DataFrame have to equal the in the energy system model
              specified locations.
            * a dictionary with investment periods as keys and one of the two options above as values.

        :param balanceLimitID: ID for the respective balance limit (out of the balance limits introduced in the esM).
            Should be specified if the respective component of the TransmissionModel is supposed to be included in
            the balance analysis. If the commodity is transported out of the region, it is counted as a negative, if
            it is imported into the region it is considered positive.
            |br| * the default value is None
        :type balanceLimitID: string

        :param pathwayBalanceLimitID: similar to balanceLimitID just as restriction over the entire pathway.
            |br| * the default value is None
        :type pathwayBalanceLimitID: string
        """
        # TODO add unit checks
        self.capacityMax = capacityMax
        self.capacityMin = capacityMin
        self.capacityFix = capacityFix
        # Preprocess two-dimensional data
        self.locationalEligibility = utils.preprocess2dimData(locationalEligibility)
        preprocessedCapacityMax = utils.process2dimCapacityData(
            esM,
            "capacityMax",
            capacityMax,
            esM.investmentPeriods,
        )
        preprocessedCapacityFix = utils.process2dimCapacityData(
            esM,
            "capacityFix",
            capacityFix,
            esM.investmentPeriods,
        )
        self.isBuiltFix = utils.preprocess2dimData(
            isBuiltFix, locationalEligibility=locationalEligibility
        )

        # Set locational eligibility
        if operationRateFix is None:
            operationTimeSeries = operationRateMax
        elif not isinstance(operationRateFix, dict):
            operationTimeSeries = operationRateFix
        elif isinstance(operationRateFix, dict) and any(
            x is not None for x in operationRateFix.values()
        ):
            if not all(x is not None for x in operationRateFix.values()):
                raise ValueError()
            operationTimeSeries = operationRateFix
        else:
            operationTimeSeries = operationRateMax

        if not isinstance(operationTimeSeries, dict):
            operationTimeSeries = dict.fromkeys(
                esM.investmentPeriods, operationTimeSeries
            )
        if all(x is None for x in operationTimeSeries.values()):
            operationTimeSeries = None

        self.locationalEligibility = utils.setLocationalEligibility(
            esM,
            self.locationalEligibility,
            preprocessedCapacityMax,
            preprocessedCapacityFix,
            self.isBuiltFix,
            hasCapacityVariable,
            operationTimeSeries,
            "2dim",
        )

        self._mapC, self._mapL, self._mapI = {}, {}, {}
        for loc1 in esM.locations:
            for loc2 in esM.locations:
                if loc1 + "_" + loc2 in self.locationalEligibility.index:
                    if self.locationalEligibility[loc1 + "_" + loc2] == 0:
                        self.locationalEligibility.drop(
                            labels=loc1 + "_" + loc2, inplace=True
                        )
                    self._mapC.update({loc1 + "_" + loc2: (loc1, loc2)})
                    self._mapL.setdefault(loc1, {}).update({loc2: loc1 + "_" + loc2})
                    self._mapI.update({loc1 + "_" + loc2: loc2 + "_" + loc1})

        # capacity parameter
        preprocessedCapacityMax = utils.preprocess2dimData(
            capacityMax, self._mapC, locationalEligibility=self.locationalEligibility
        )
        preprocessedCapacityFix = utils.preprocess2dimData(
            capacityFix, self._mapC, locationalEligibility=self.locationalEligibility
        )
        preprocessedCapacityMin = utils.preprocess2dimData(
            capacityMin, self._mapC, locationalEligibility=self.locationalEligibility
        )
        # stockCommissioning
        if stockCommissioning is None:
            self.stockCommissioning = stockCommissioning
        elif isinstance(stockCommissioning, dict):
            self.stockCommissioning = {}
            for potential_ip in stockCommissioning.keys():
                self.stockCommissioning[potential_ip] = utils.preprocess2dimData(
                    stockCommissioning[potential_ip],
                    locationalEligibility=locationalEligibility,
                )
        else:
            raise ValueError("stockCommissioning must be None or a dict.")

        self.isBuiltFix = utils.preprocess2dimData(
            isBuiltFix, self._mapC, locationalEligibility=self.locationalEligibility
        )

        self.interestRate = utils.preprocess2dimData(interestRate, self._mapC)
        self.economicLifetime = utils.preprocess2dimData(economicLifetime, self._mapC)
        self.technicalLifetime = utils.preprocess2dimData(technicalLifetime, self._mapC)
        self.balanceLimitID = balanceLimitID
        self.pathwayBalanceLimitID = pathwayBalanceLimitID

        Component.__init__(
            self,
            esM,
            name,
            dimension="2dim",
            hasCapacityVariable=hasCapacityVariable,
            capacityVariableDomain=capacityVariableDomain,
            capacityPerPlantUnit=capacityPerPlantUnit,
            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable,
            bigM=bigM,
            locationalEligibility=self.locationalEligibility,
            capacityMin=preprocessedCapacityMin,
            capacityMax=preprocessedCapacityMax,
            partLoadMin=partLoadMin,
            sharedPotentialID=sharedPotentialID,
            linkedQuantityID=linkedQuantityID,
            capacityFix=preprocessedCapacityFix,
            isBuiltFix=self.isBuiltFix,
            investPerCapacity=0,
            investIfBuilt=0,
            opexPerCapacity=0,
            opexIfBuilt=0,
            interestRate=self.interestRate,
            QPcostScale=QPcostScale,
            economicLifetime=self.economicLifetime,
            technicalLifetime=self.technicalLifetime,
            floorTechnicalLifetime=floorTechnicalLifetime,
            stockCommissioning=self.stockCommissioning,
        )
        # Set general component data
        utils.checkCommodities(esM, {commodity})
        self.commodity, self.commodityUnit = (
            commodity,
            esM.commodityUnitsDict[commodity],
        )
        self.distances = utils.preprocess2dimData(
            distances, self._mapC, locationalEligibility=self.locationalEligibility
        )
        self.losses = utils.preprocess2dimData(losses, self._mapC)
        self.distances = utils.checkAndSetDistances(
            self.distances, self.locationalEligibility, esM
        )
        self.losses = utils.checkAndSetTransmissionLosses(
            self.losses, self.distances, self.locationalEligibility
        )
        self.modelingClass = TransmissionModel

        # these are initialized with 0 in the component.__init__ and overwritten here,
        # due to its different structure otherwise the tests fail in the component
        self.investPerCapacity = investPerCapacity
        self.preprocessedInvestPerCapacity = utils.preprocess2dimInvestmentPeriodData(
            esM,
            "investPerCapacity",
            investPerCapacity,
            self.processedStockYears + esM.investmentPeriods,
            mapC=self._mapC,
        )

        self.investIfBuilt = investIfBuilt
        self.preprocessedInvestIfBuilt = utils.preprocess2dimInvestmentPeriodData(
            esM,
            "investIfBuilt",
            investIfBuilt,
            self.processedStockYears + esM.investmentPeriods,
            mapC=self._mapC,
        )

        self.opexPerCapacity = opexPerCapacity
        self.preprocessedOpexPerCapacity = utils.preprocess2dimInvestmentPeriodData(
            esM,
            "opexPerCapacity",
            opexPerCapacity,
            self.processedStockYears + esM.investmentPeriods,
            mapC=self._mapC,
        )

        self.opexIfBuilt = opexIfBuilt
        self.preprocessedOpexIfBuilt = utils.preprocess2dimInvestmentPeriodData(
            esM,
            "opexIfBuilt",
            opexIfBuilt,
            self.processedStockYears + esM.investmentPeriods,
            mapC=self._mapC,
        )

        # Set distance related costs data
        self.processedInvestPerCapacity = {}
        self.processedInvestIfBuilt = {}
        self.processedOpexPerCapacity = {}
        self.processedOpexIfBuilt = {}
        for year in self.processedStockYears + esM.investmentPeriods:
            self.processedInvestPerCapacity[year] = (
                utils.preprocess2dimData(
                    self.preprocessedInvestPerCapacity[year],
                    self._mapC,
                    self.locationalEligibility,
                )
                * self.distances
                * 0.5
            )
            self.processedInvestIfBuilt[year] = (
                utils.preprocess2dimData(
                    self.preprocessedInvestIfBuilt[year],
                    self._mapC,
                    self.locationalEligibility,
                )
                * self.distances
                * 0.5
            )
            self.processedOpexPerCapacity[year] = (
                utils.preprocess2dimData(
                    self.preprocessedOpexPerCapacity[year],
                    self._mapC,
                    self.locationalEligibility,
                )
                * self.distances
                * 0.5
            )
            self.processedOpexIfBuilt[year] = (
                utils.preprocess2dimData(
                    self.preprocessedOpexIfBuilt[year],
                    self._mapC,
                    self.locationalEligibility,
                )
                * self.distances
                * 0.5
            )

        # Set additional economic data
        # opexPerOperation
        self.opexPerOperation = utils.preprocess2dimData(opexPerOperation, self._mapC)
        self.processedOpexPerOperation = utils.checkAndSetInvestmentPeriodCostParameter(
            esM,
            name,
            self.opexPerOperation,
            "2dim",
            self.locationalEligibility,
            esM.investmentPeriods,
        )

        # operationRateMax
        self.operationRateMax = operationRateMax
        self.fullOperationRateMax = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, operationRateMax, self.locationalEligibility
        )
        self.aggregatedOperationRateMax = dict.fromkeys(esM.investmentPeriods)
        self.processedOperationRateMax = dict.fromkeys(esM.investmentPeriods)

        # operationRateFix
        self.operationRateFix = operationRateFix
        self.fullOperationRateFix = utils.checkAndSetInvestmentPeriodTimeSeries(
            esM, name, operationRateFix, self.locationalEligibility
        )
        self.aggregatedOperationRateFix = dict.fromkeys(esM.investmentPeriods)
        self.processedOperationRateFix = dict.fromkeys(esM.investmentPeriods)

        # partLoadMin
        self.processedPartLoadMin = utils.checkAndSetPartLoadMin(
            esM,
            name,
            partLoadMin,
            self.fullOperationRateMax,
            self.fullOperationRateFix,
            self.bigM,
            self.hasCapacityVariable,
        )

        utils.isPositiveNumber(tsaWeight)
        self.tsaWeight = tsaWeight

        # set parameter to None if all years have None values
        self.fullOperationRateFix = utils.setParamToNoneIfNoneForAllYears(
            self.fullOperationRateFix
        )
        self.fullOperationRateMax = utils.setParamToNoneIfNoneForAllYears(
            self.fullOperationRateMax
        )

        # set processed location eligiblity # TODO implement check and set
        self.processedLocationalEligibility = self.locationalEligibility

    def setTimeSeriesData(self, hasTSA):
        """
        Function for setting the maximum operation rate and fixed operation rate depending on whether a time series
        analysis is requested or not.

        :param hasTSA: states whether a time series aggregation is requested (True) or not (False).
        :type hasTSA: boolean
        """
        self.processedOperationRateMax = (
            self.aggregatedOperationRateMax if hasTSA else self.fullOperationRateMax
        )
        self.processedOperationRateFix = (
            self.aggregatedOperationRateFix if hasTSA else self.fullOperationRateFix
        )

    def getDataForTimeSeriesAggregation(self, ip):
        """Function for getting the required data if a time series aggregation is requested.

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """
        weightDict, data = {}, []
        weightDict, data = self.prepareTSAInput(
            self.fullOperationRateFix,
            self.fullOperationRateMax,
            "_operationRate_",
            self.tsaWeight,
            weightDict,
            data,
            ip,
        )
        return (pd.concat(data, axis=1), weightDict) if data else (None, {})

    def setAggregatedTimeSeriesData(self, data, ip):
        """
        Function for determining the aggregated maximum rate and the aggregated fixed operation rate.

        :param data: Pandas DataFrame with the clustered time series data of the conversion component
        :type data: Pandas DataFrame

        :param ip: investment period of transformation path analysis.
        :type ip: int
        """

        self.aggregatedOperationRateFix[ip] = self.getTSAOutput(
            self.fullOperationRateFix, "_operationRate_", data, ip
        )
        self.aggregatedOperationRateMax[ip] = self.getTSAOutput(
            self.fullOperationRateMax, "_operationRate_", data, ip
        )

    def checkProcessedDataSets(self):
        """
        Check processed time series data after applying time series aggregation. If all entries of dictionary are None
        the parameter itself is set to None.
        """
        for parameter in ["processedOperationRateFix", "processedOperationRateMax"]:
            setattr(
                self,
                parameter,
                utils.setParamToNoneIfNoneForAllYears(getattr(self, parameter)),
            )


class TransmissionModel(ComponentModel):
    """
    A TransmissionModel class instance will be instantly created if a Transmission class instance is initialized.
    It is used for the declaration of the sets, variables and constraints which are valid for the Transmission class
    instance. These declarations are necessary for the modeling and optimization of the energy system model.
    The TransmissionModel class inherits from the ComponentModel class.
    """

    def __init__(self):
        """ " Constructor for creating a TransmissionModel class instance"""
        super().__init__()
        self.abbrvName = "trans"
        self.dimension = "2dim"
        self._operationVariablesOptimum = {}
        self._isBuiltVariablesOptimum = {}

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """
        Declare sets: design variable sets, operation variable set and operation mode sets.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        # # Declare design variable sets
        self.declareDesignVarSet(pyM, esM)
        self.declareCommissioningVarSet(pyM, esM)
        self.declareContinuousDesignVarSet(pyM)
        self.declareDiscreteDesignVarSet(pyM)
        self.declareDesignDecisionVarSet(pyM)

        # Declare design pathway sets
        self.declarePathwaySets(pyM, esM)
        self.declareLocationComponentSet(pyM)

        # Declare operation variable set
        self.declareOpVarSet(esM, pyM)

        # Declare operation mode sets
        self.declareOperationModeSets(
            pyM, "opConstrSet", "operationRateMax", "operationRateFix"
        )

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM, relaxIsBuiltBinary, relevanceThreshold):
        """
        Declare design and operation variables

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param relaxIsBuiltBinary: states if the optimization problem should be solved as a relaxed LP to get the lower
            bound of the problem.
            |br| * the default value is False
        :type declaresOptimizationProblem: boolean

        :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
            |br| * the default value is None
        :type relevanceThreshold: float (>=0) or None
        """

        # Capacity variables [commodityUnit]
        self.declareCapacityVars(pyM)
        # (Continuous) numbers of installed components [-]
        self.declareRealNumbersVars(pyM)
        # (Discrete/integer) numbers of installed components [-]
        self.declareIntNumbersVars(pyM)
        # Binary variables [-] indicating if a component is considered at a location or not
        self.declareBinaryDesignDecisionVars(pyM, relaxIsBuiltBinary)
        # Operation of component [commodityUnit]
        self.declareOperationVars(pyM, esM, "op", relevanceThreshold=relevanceThreshold)
        # Operation of component as binary [1/0]
        self.declareOperationBinaryVars(pyM, "op_bin")
        # Capacity development variables [physicalUnit]
        self.declareCommissioningVars(pyM, esM)
        self.declareDecommissioningVars(pyM, esM)

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def symmetricalCapacity(self, pyM):
        """
        Ensure that the capacity between location_1 and location_2 is the same as the one
        between location_2 and location_1.

        .. math::

            cap^{comp}_{(loc_1,loc_2)} = cap^{comp}_{(loc_2,loc_1)}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar, capVarSet = (
            getattr(pyM, "cap_" + abbrvName),
            getattr(pyM, "designDimensionVarSet_" + abbrvName),
        )

        def symmetricalCapacity(pyM, loc, compName, ip):
            return (
                capVar[loc, compName, ip]
                == capVar[compDict[compName]._mapI[loc], compName, ip]
            )

        setattr(
            pyM,
            "ConstrSymmetricalCapacity_" + abbrvName,
            pyomo.Constraint(capVarSet, rule=symmetricalCapacity),
        )

    def operationMode1_2dim(self, pyM, esM, constrName, constrSetName, opVarName):
        """
        Declare the constraint that the operation [commodityUnit*hour] is limited by the installed
        capacity [commodityUnit] multiplied by the hours per time step.
        Since the flow should either go in one direction or the other, the limitation can be enforced on the sum
        of the forward and backward flow over the line. This leads to one of the flow variables being set to zero
        if a basic solution is obtained during optimization.

        .. math::

            op^{comp,op}_{(loc_1,loc_2),ip,p,t} + op^{op}_{(loc_2,loc_1),ip,p,t} \leq \\tau^{hours} \cdot \\text{cap}^{comp}_{(loc_{in},loc_{out})}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, capVar = (
            getattr(pyM, opVarName + "_" + abbrvName),
            getattr(pyM, "cap_" + abbrvName),
        )
        constrSet1 = getattr(pyM, constrSetName + "1_" + abbrvName)

        if not pyM.hasSegmentation:

            def op1(pyM, loc, compName, ip, p, t):
                return (
                    opVar[loc, compName, ip, p, t]
                    + opVar[compDict[compName]._mapI[loc], compName, ip, p, t]
                    <= capVar[loc, compName, ip] * esM.hoursPerTimeStep
                )

            setattr(
                pyM,
                constrName + "_" + abbrvName,
                pyomo.Constraint(constrSet1, pyM.intraYearTimeSet, rule=op1),
            )
        else:

            def op1(pyM, loc, compName, ip, p, t):
                return (
                    opVar[loc, compName, ip, p, t]
                    + opVar[compDict[compName]._mapI[loc], compName, ip, p, t]
                    <= capVar[loc, compName, ip]
                    * esM.hoursPerSegment[ip].to_dict()[p, t]
                )

            setattr(
                pyM,
                constrName + "_" + abbrvName,
                pyomo.Constraint(constrSet1, pyM.intraYearTimeSet, rule=op1),
            )

    def declareComponentConstraints(self, esM, pyM):
        """
        Declare time independent and dependent constraints

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        ################################################################################################################
        #                                    Declare time independent constraints                                      #
        ################################################################################################################

        # Determine the components' capacities from the number of installed units
        self.capToNbReal(pyM)
        # Determine the components' capacities from the number of installed units
        self.capToNbInt(pyM)
        # Enforce the consideration of the binary design variables of a component
        self.bigM(pyM)
        # Enforce the consideration of minimum capacities for components with design decision variables
        self.capacityMinDec(pyM)
        # Set, if applicable, the installed capacities of a component
        self.capacityFix(pyM, esM)
        # Set, if applicable, the binary design variables of a component
        self.designBinFix(pyM)
        # Enforce the equality of the capacities cap_loc1_loc2 and cap_loc2_loc1
        self.symmetricalCapacity(pyM)

        ################################################################################################################
        #                                    Declare pathway constraints                                               #
        ################################################################################################################
        # Set capacity development constraints over investment periods
        self.designDevelopmentConstraint(pyM, esM)
        self.decommissioningConstraint(pyM, esM)
        self.stockCapacityConstraint(pyM, esM)
        self.stockCommissioningConstraint(pyM, esM)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [commodityUnit*h] is limited by the installed capacity [commodityUnit] multiplied by the hours per
        # time step [h]
        self.operationMode1_2dim(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] is equal to the installed capacity [commodityUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode2(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [commodityUnit*h] is limited by the installed capacity [commodityUnit] multiplied by operation time
        # series [-] and the hours per time step [h]
        self.operationMode3(pyM, esM, "ConstrOperation", "opConstrSet", "op")
        # Operation [physicalUnit*h] is limited by minimum part Load
        self.additionalMinPartLoad(
            pyM, esM, "ConstrOperation", "opConstrSet", "op", "op_bin", "cap"
        )

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        """
        Check if the commodity´s transfer between a given location and the other locations of the energy system model
        is eligible.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param commod: Name of the regarded commodity (commodities are defined in the EnergySystemModel instance)
        :param commod: string
        """

        return any(
            [
                comp.commodity == commod
                and (
                    loc + "_" + loc_ in comp.processedLocationalEligibility.index
                    or loc_ + "_" + loc in comp.processedLocationalEligibility.index
                )
                for comp in self.componentsDict.values()
                for loc_ in esM.locations
            ]
        )

    def getCommodityBalanceContribution(self, pyM, commod, loc, ip, p, t):
        """ Get contribution to a commodity balance. 
        
            .. math::
                :nowrap:

                \\begin{eqnarray*}
                \\text{C}^{comp,comm}_{loc,ip,p,t} = & & \\underset{\substack{(loc_{in},loc_{out}) \in \\ \mathcal{L}^{tans}: loc_{in}=loc}}{ \sum } \left(1-\eta_{(loc_{in},loc_{out})} \cdot I_{(loc_{in},loc_{out})} \\right) \cdot op^{comp,op}_{(loc_{in},loc_{out}),ip,p,t} \\\\
                & - & \\underset{\substack{(loc_{in},loc_{out}) \in \\ \mathcal{L}^{tans}:loc_{out}=loc}}{ \sum } op^{comp,op}_{(loc_{in},loc_{out}),ip,p,t}
                \\end{eqnarray*}
            
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar, opVarDictIn = (
            getattr(pyM, "op_" + abbrvName),
            getattr(pyM, "operationVarDictIn_" + abbrvName),
        )
        opVarDictOut = getattr(pyM, "operationVarDictOut_" + abbrvName)
        return sum(
            opVar[loc_ + "_" + loc, compName, ip, p, t]
            * (
                1
                - compDict[compName].losses[loc_ + "_" + loc]
                * compDict[compName].distances[loc_ + "_" + loc]
            )
            for loc_ in opVarDictIn[ip][loc].keys()
            for compName in opVarDictIn[ip][loc][loc_]
            if commod == compDict[compName].commodity
        ) - sum(
            opVar[loc + "_" + loc_, compName, ip, p, t]
            for loc_ in opVarDictOut[ip][loc].keys()
            for compName in opVarDictOut[ip][loc][loc_]
            if commod == compDict[compName].commodity
        )

    def getBalanceLimitContribution(
        self, esM, pyM, ID, ip, loc, timeSeriesAggregation, componentNames
    ):
        """
        Get contribution to balanceLimitConstraint (Further read in EnergySystemModel).
        Sum of the operation time series of a Transmission component is used as the balanceLimit contribution:

        - If commodity is transferred out of region a negative sign is used.
        - If commodity is transferred into region a positive sign is used and losses are considered.

        Sum of the operation time series of a Transmission component is used as the balanceLimit contribution:

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pym: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pym: pyomo ConcreteModel

        :param ip: investment period of transformation path analysis.
        :type ip: int

        :param ID: ID of the regarded balanceLimitConstraint
        :param ID: string

        :param timeSeriesAggregation: states if the optimization of the energy system model should be done with

            (a) the full time series (False) or
            (b) clustered time series data (True).

        :type timeSeriesAggregation: boolean

        :param loc: Name of the regarded location (locations are defined in the EnergySystemModel instance)
        :type loc: string

        :param componentNames: Names of components which contribute to the balance limit
        :type componentNames: list
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)
        opVarDictIn = getattr(pyM, "operationVarDictIn_" + abbrvName)
        opVarDictOut = getattr(pyM, "operationVarDictOut_" + abbrvName)

        if timeSeriesAggregation:
            periods = esM.typicalPeriods
            if esM.segmentation:
                timeSteps = esM.segmentsPerPeriod
            else:
                timeSteps = esM.timeStepsPerPeriod

        else:
            periods = esM.periods
            timeSteps = esM.totalTimeSteps
        aut = sum(
            opVar[loc_ + "_" + loc, compName, ip, p, t]
            * (
                1
                - compDict[compName].losses[loc_ + "_" + loc]
                * compDict[compName].distances[loc_ + "_" + loc]
            )
            * esM.periodOccurrences[ip][p]
            for loc_ in opVarDictIn[ip][loc].keys()
            for compName in opVarDictIn[ip][loc][loc_]
            if compName in componentNames
            for p in periods
            for t in timeSteps
        ) - sum(
            opVar[loc + "_" + loc_, compName, ip, p, t] * esM.periodOccurrences[ip][p]
            for loc_ in opVarDictOut[ip][loc].keys()
            for compName in opVarDictOut[ip][loc][loc_]
            if compName in componentNames
            for p in periods
            for t in timeSteps
        )
        return aut

    def getObjectiveFunctionContribution(self, esM, pyM):
        """
        Get contribution to the objective function.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

        opexOp = self.getEconomicsOperation(
            pyM, esM, "TD", ["processedOpexPerOperation"], "op", "operationVarDictOut"
        )

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

        return opexOp + capexCap + capexDec + opexCap + opexDec

    def setOptimalValues(self, esM, pyM):
        """
        Set the optimal values of the components.

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        opVar = getattr(pyM, "op_" + abbrvName)
        mapC = {
            loc1 + "_" + loc2: (loc1, loc2)
            for loc1 in esM.locations
            for loc2 in esM.locations
        }
        # Set optimal design dimension variables and get basic optimization summary
        optSummaryBasic = super().setOptimalValues(
            esM, pyM, mapC.keys(), "commodityUnit"
        )

        # Get class related results
        resultsTAC_opexOp = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerOperation"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="TAC",
        )
        resultsNPV_opexOp = self.getEconomicsOperation(
            pyM,
            esM,
            "TD",
            ["processedOpexPerOperation"],
            "op",
            "operationVarDict",
            getOptValue=True,
            getOptValueCostType="NPV",
        )
        for ip in esM.investmentPeriods:
            for compName, comp in compDict.items():
                for cost in [
                    "invest",
                    "capexCap",
                    "capexIfBuilt",
                    "opexCap",
                    "opexIfBuilt",
                    "TAC",
                ]:
                    data = optSummaryBasic[esM.investmentPeriodNames[ip]].loc[
                        compName, cost
                    ]
                    optSummaryBasic[esM.investmentPeriodNames[ip]].loc[
                        compName, cost
                    ] = (data).values

            # Set optimal operation variables and append optimization summary
            optVal = utils.formatOptimizationOutput(
                opVar.get_values(),
                "operationVariables",
                "1dim",
                ip,
                esM.periodsOrder[ip],
                esM=esM,
            )
            optVal_ = utils.formatOptimizationOutput(
                opVar.get_values(),
                "operationVariables",
                "2dim",
                ip,
                esM.periodsOrder[ip],
                compDict=compDict,
                esM=esM,
            )
            self._operationVariablesOptimum[esM.investmentPeriodNames[ip]] = optVal_

            props = ["operation", "opexOp", "NPV_opexOp"]
            # Unit dict: Specify units for props
            units = {
                props[0]: ["[-*h]", "[-*h/a]"],
                props[1]: ["[" + esM.costUnit + "/a]"],
                props[2]: ["[" + esM.costUnit + "/a]"],
            }
            # Create tuples for the optSummary's multiIndex. Combine component with the respective properties and units.
            tuples = [
                (compName, prop, unit)
                for compName in compDict.keys()
                for prop in props
                for unit in units[prop]
            ]
            # Replace placeholder with correct unit of component
            tuples = list(
                map(
                    lambda x: (
                        x[0],
                        x[1],
                        x[2].replace("-", compDict[x[0]].commodityUnit),
                    )
                    if x[1] == "operation"
                    else x,
                    tuples,
                )
            )
            mIndex = pd.MultiIndex.from_tuples(
                tuples, names=["Component", "Property", "Unit"]
            )
            optSummary = pd.DataFrame(
                index=mIndex, columns=sorted(mapC.keys())
            ).sort_index()

            if optVal is not None:
                opSum = optVal.sum(axis=1).unstack(-1)

                optSummary.loc[
                    [
                        (ix, "operation", "[" + compDict[ix].commodityUnit + "*h/a]")
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = (
                    opSum.values / esM.numberOfYears
                )
                optSummary.loc[
                    [
                        (ix, "operation", "[" + compDict[ix].commodityUnit + "*h]")
                        for ix in opSum.index
                    ],
                    opSum.columns,
                ] = opSum.values

                tac_ox = resultsTAC_opexOp[ip]
                optSummary.loc[
                    [(ix, "opexOp", "[" + esM.costUnit + "/a]") for ix in tac_ox.index],
                    tac_ox.columns,
                ] = tac_ox.values

                npv_ox = resultsNPV_opexOp[ip]
                optSummary.loc[
                    [(ix, "opexOp", "[" + esM.costUnit + "/a]") for ix in npv_ox.index],
                    npv_ox.columns,
                ] = npv_ox.values

            optSummaryBasic_frame = optSummaryBasic[esM.investmentPeriodNames[ip]]
            if isinstance(optSummaryBasic_frame, pd.Series):
                optSummaryBasic_frame = optSummaryBasic_frame.to_frame().T

            optSummary = pd.concat(
                [
                    optSummary,
                    optSummaryBasic_frame,
                ],
                axis=0,
            ).sort_index()

            # Summarize all contributions to the total annual cost
            optSummary.loc[optSummary.index.get_level_values(1) == "TAC"] = (
                optSummary.loc[
                    (optSummary.index.get_level_values(1) == "TAC")
                    | (optSummary.index.get_level_values(1) == "opexOp")
                ]
                .groupby(level=0)
                .sum()
                .values
            )

            # Update the NPV contribution
            optSummary.loc[
                optSummary.index.get_level_values(1) == "NPVcontribution"
            ] = (
                optSummary.loc[
                    (optSummary.index.get_level_values(1) == "NPVcontribution")
                    | (optSummary.index.get_level_values(1) == "NPV_opexOp")
                ]
                .groupby(level=0)
                .sum()
                .values
            )
            # Delete details of NPV contribution
            optSummary = optSummary.drop("NPV_opexOp", level=1)

            # Split connection indices to two location indices
            optSummary = optSummary.stack()
            indexNew = []
            for tup in optSummary.index.tolist():
                loc1, loc2 = mapC[tup[3]]
                indexNew.append((tup[0], tup[1], tup[2], loc1, loc2))
            optSummary.index = pd.MultiIndex.from_tuples(indexNew)
            optSummary = optSummary.unstack(level=-1)
            names = list(optSummaryBasic[esM.investmentPeriodNames[ip]].index.names)
            names.append("LocationIn")
            optSummary.index.set_names(names, inplace=True)
            self._optSummary[esM.investmentPeriodNames[ip]] = optSummary

    def getOptimalValues(self, name="all", ip=0):
        """
        Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

        * '_capacityVariables',
        * '_isBuiltVariables',
        * '_operationVariablesOptimum',
        * 'all' or another input: all variables are returned.

        |br| * the default value is 'all'
        :type name: string

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        return super().getOptimalValues(name, ip=ip)
