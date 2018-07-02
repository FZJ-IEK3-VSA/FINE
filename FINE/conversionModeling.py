from FINE.component import Component, ComponentModeling
from FINE import utils
import pyomo.environ as pyomo
import warnings
import pandas as pd


class Conversion(Component):
    """
    Doc
    """
    def __init__(self, esM, name, commodityConversionFactors, hasDesignDimensionVariables=True,
                 designDimensionVariableDomain='continuous', capacityPerUnit=1,
                 hasDesignDecisionVariables=False, bigM=None,
                 operationRateMax=None, operationRateFix=None, tsaWeight=1,
                 locationalEligibility=None, capacityMin=None, capacityMax=None, sharedPotentialID=None,
                 capacityFix=None, designDecisionFix=None,
                 capexPerDesignDimension=0, capexForDesignDecision=0, opexPerOperation=0, opexPerDesignDimension=0,
                 opexForDesignDecision=0, interestRate=0.08, economicLifetime=10):
        # Set general component data
        utils.checkCommodities(esM, set(commodityConversionFactors.keys()))
        self._name, self._commodityConversionFactors = name, commodityConversionFactors

        # Set design variable modeling parameters
        utils.checkDesignVariableModelingParameters(designDimensionVariableDomain, hasDesignDimensionVariables,
                                                    hasDesignDecisionVariables, bigM)
        self._hasDesignDimensionVariables = hasDesignDimensionVariables
        self._designDimensionVariableDomain = designDimensionVariableDomain
        self._capacityPerUnit = capacityPerUnit
        self._hasDesignDecisionVariables = hasDesignDecisionVariables
        self._bigM = bigM

        # Set economic data
        self._capexPerDesignDimension = utils.checkAndSetCostParameter(esM, name, capexPerDesignDimension)
        self._capexForDesignDecision = utils.checkAndSetCostParameter(esM, name, capexForDesignDecision)
        self._opexPerOperation = utils.checkAndSetCostParameter(esM, name, opexPerOperation)
        self._opexPerDesignDimension = utils.checkAndSetCostParameter(esM, name, opexPerDesignDimension)
        self._opexForDesignDecision = utils.checkAndSetCostParameter(esM, name, opexForDesignDecision)
        self._interestRate = utils.checkAndSetCostParameter(esM, name, interestRate)
        self._economicLifetime = utils.checkAndSetCostParameter(esM, name, economicLifetime)
        self._CCF = self.getCapitalChargeFactor()

        # Set location-specific operation parameters
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            warnings.warn('If operationRateFix is specified, the operationRateMax parameter is not required.\n' +
                          'The operationRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateMax, locationalEligibility)
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateFix, locationalEligibility)

        self._fullOperationRateMax = utils.setFormattedTimeSeries(operationRateMax)
        self._aggregatedOperationRateMax = None
        self._operationRateMax = None

        self._fullOperationRateFix = utils.setFormattedTimeSeries(operationRateFix)
        self._aggregatedOperationRateFix = None
        self._operationRateFix = None
        self._tsaWeight = tsaWeight

        # Set location-specific design parameters
        self._sharedPotentialID = sharedPotentialID
        utils.checkLocationSpecficDesignInputParams(esM, hasDesignDimensionVariables, hasDesignDecisionVariables,
                                                    capacityMin, capacityMax, capacityFix,
                                                    locationalEligibility, designDecisionFix, sharedPotentialID,
                                                    dimension='1dim')
        self._capacityMin, self._capacityMax, self._capacityFix = capacityMin, capacityMax, capacityFix
        self._designDecisionFix = designDecisionFix

        # Set locational eligibility
        operationTimeSeries = operationRateFix if operationRateFix is not None else operationRateMax
        self._locationalEligibility = utils.setLocationalEligibility(esM, locationalEligibility, capacityMax,
                                                                     capacityFix, designDecisionFix,
                                                                     hasDesignDimensionVariables, operationTimeSeries)

    def getCapitalChargeFactor(self):
        """ Computes and returns capital charge factor (inverse of annuity factor) """
        return 1 / self._interestRate - 1 / (pow(1 + self._interestRate, self._economicLifetime) * self._interestRate)

    def addToESM(self, esM):
        esM._isTimeSeriesDataClustered = False
        if self._name in esM._componentNames:
            if esM._componentNames[self._name] == ConversionModeling.__name__:
                warnings.warn('Component identifier ' + self._name + ' already exists. Data will be overwritten.')
            else:
                raise ValueError('Component name ' + self._name + ' is not unique.')
        else:
            esM._componentNames.update({self._name: ConversionModeling.__name__})
        mdl = ConversionModeling.__name__
        if mdl not in esM._componentModelingDict:
            esM._componentModelingDict.update({mdl: ConversionModeling()})
        esM._componentModelingDict[mdl]._componentsDict.update({self._name: self})

    def setTimeSeriesData(self, hasTSA):
        self._operationRateMax = self._aggregatedOperationRateMax if hasTSA else self._fullOperationRateMax
        self._operationRateFix = self._aggregatedOperationRateFix if hasTSA else self._fullOperationRateFix

    def getDataForTimeSeriesAggregation(self):
        data = self._fullOperationRateFix if self._fullOperationRateFix is not None else self._fullOperationRateMax
        if data is not None:
            data, compDict = data.copy(), {}
            for location in data.columns:
                uniqueIdentifier = self._name + "_operationRate_" + location
                data[uniqueIdentifier] = data.pop(location)
                compDict.update({uniqueIdentifier: self._tsaWeight})
            return data, compDict
        else:
            return None, {}

    def setAggregatedTimeSeriesData(self, data):
        fullOperationRate = self._fullOperationRateFix if self._fullOperationRateFix is not None \
            else self._fullOperationRateMax
        if fullOperationRate is not None:
            uniqueIdentifiers = [self._name + "_operationRate_" + location for location in fullOperationRate.columns]
            compData = data[uniqueIdentifiers].copy()
            compData = compData.rename(columns={self._name + "_operationRate_" + location: location
                                                for location in fullOperationRate.columns})
            if self._fullOperationRateFix is not None:
                self._aggregatedOperationRateFix = compData
            else:
                self._aggregatedOperationRateMax = compData


class ConversionModeling(ComponentModeling):
    """ Doc """
    def __init__(self):
        self._componentsDict = {}

    ####################################################################################################################
    #                                            Declare sparse index sets                                             #
    ####################################################################################################################

    def declareSets(self, esM, pyM):
        """ Declares sets and dictionaries """
        compDict = self._componentsDict

        ################################################################################################################
        #                                        Declare design variables sets                                         #
        ################################################################################################################

        def initDesignVarSet(pyM):
            return ((loc, compName) for loc in esM._locations for compName, comp in compDict.items()
                    if comp._locationalEligibility[loc] == 1 and comp._hasDesignDimensionVariables)
        pyM.designDimensionVarSet_conv = pyomo.Set(dimen=2, initialize=initDesignVarSet)

        def initContinuousDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_conv
                    if compDict[compName]._designDimensionVariableDomain == 'continuous')
        pyM.continuousDesignDimensionVarSet_conv = pyomo.Set(dimen=2, initialize=initContinuousDesignVarSet)

        def initDiscreteDesignVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_conv
                    if compDict[compName]._designDimensionVariableDomain == 'discrete')
        pyM.discreteDesignDimensionVarSet_conv = pyomo.Set(dimen=2, initialize=initDiscreteDesignVarSet)

        def initDesignDecisionVarSet(pyM):
            return ((loc, compName) for loc, compName in pyM.designDimensionVarSet_conv
                    if compDict[compName]._hasDesignDecisionVariables)
        pyM.designDecisionVarSet_conv = pyomo.Set(dimen=2, initialize=initDesignDecisionVarSet)

        ################################################################################################################
        #                                     Declare operation variables sets                                         #
        ################################################################################################################

        def initOpVarSet(pyM):
            return ((loc, compName) for loc in esM._locations for compName, comp in compDict.items()
                    if comp._locationalEligibility[loc] == 1)
        pyM.operationVarSet_conv = pyomo.Set(dimen=2, initialize=initOpVarSet)
        pyM.operationVarDict_conv = {loc: {compName for compName in compDict if (loc, compName)
                                           in pyM.operationVarSet_conv} for loc in esM._locations}

        ################################################################################################################
        #                           Declare sets for case differentiation of operating modes                           #
        ################################################################################################################

        def initOpConstrSet1(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if
                    compDict[compName]._hasDesignDimensionVariables and compDict[compName]._operationRateMax is None
                    and compDict[compName]._operationRateFix is None)
        pyM.opConstrSet1_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet1)

        def initOpConstrSet2(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if
                    compDict[compName]._hasDesignDimensionVariables and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet2_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet2)

        def initOpConstrSet3(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if
                    compDict[compName]._hasDesignDimensionVariables and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet3_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet3)

        def initOpConstrSet4(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if not
                    compDict[compName]._hasDesignDimensionVariables and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet4_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet4)

        def initOpConstrSet5(pyM):
            return ((loc, compName) for loc, compName in pyM.operationVarSet_conv if not
                    compDict[compName]._hasDesignDimensionVariables and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet5_conv = pyomo.Set(dimen=2, initialize=initOpConstrSet5)

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """ Declares design and operation variables """
        # Function for setting lower and upper capacity bounds
        def capBounds(pyM, loc, compName):
            comp = self._componentsDict[compName]
            return (comp._capacityMin[loc] if (comp._capacityMin is not None and not comp._hasDesignDecisionVariables)
                    else 0, comp._capacityMax[loc] if comp._capacityMax is not None else None)

        # Capacity of components [powerUnit]
        pyM.cap_conv = pyomo.Var(pyM.designDimensionVarSet_conv, domain=pyomo.NonNegativeReals, bounds=capBounds)
        # Number of components [-]
        pyM.nbReal_conv = pyomo.Var(pyM.continuousDesignDimensionVarSet_conv, domain=pyomo.NonNegativeReals)
        # Number of components [-]
        pyM.nbInt_conv = pyomo.Var(pyM.discreteDesignDimensionVarSet_conv, domain=pyomo.NonNegativeIntegers)
        # Binary variables [-], indicate if a component is considered at a location or not
        pyM.designBin_conv = pyomo.Var(pyM.designDecisionVarSet_conv, domain=pyomo.Binary)
        # Operation of component [energyUnit]
        pyM.op_conv = pyomo.Var(pyM.operationVarSet_conv, pyM.timeSet, domain=pyomo.NonNegativeReals)

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def declareComponentConstraints(self, esM, pyM):
        """ Declares time independent and dependent constraints"""
        compDict = self._componentsDict

        ################################################################################################################
        #                                    Declare time independent constraints                                      #
        ################################################################################################################

        # Determine the components' capacities from the number of installed units
        def capToNbReal_conv(pyM, loc, compName):
            return pyM.cap_conv[loc, compName] == pyM.nbReal_conv[loc, compName] * compDict[compName]._capacityPerUnit
        pyM.ConstrCapToNbReal_conv = pyomo.Constraint(pyM.continuousDesignDimensionVarSet_conv, rule=capToNbReal_conv)

        # Determine the components' capacities from the number of installed units
        def capToNbInt_conv(pyM, loc, compName):
            return pyM.cap_conv[loc, compName] == pyM.nbInt_conv[loc, compName] * compDict[compName]._capacityPerUnit
        pyM.ConstrCapToNbInt_conv = pyomo.Constraint(pyM.discreteDesignDimensionVarSet_conv, rule=capToNbInt_conv)

        # Enforce the consideration of the binary design variables of a component
        def bigM_conv(pyM, loc, compName):
            return pyM.cap_conv[loc, compName] <= pyM.designBin_conv[loc, compName] * compDict[compName]._bigM
        pyM.ConstrBigM_conv = pyomo.Constraint(pyM.designDecisionVarSet_conv, rule=bigM_conv)

        # Enforce the consideration of minimum capacities for components with design decision variables
        def capacityMinDec_conv(pyM, loc, compName):
            return (pyM.cap_conv[loc, compName] >= compDict[compName]._capacityMin[loc] *
                    pyM.designBin_conv[loc, compName] if compDict[compName]._capacityMin is not None
                    else pyomo.Constraint.Skip)
        pyM.ConstrCapacityMinDec_conv = pyomo.Constraint(pyM.designDecisionVarSet_conv, rule=capacityMinDec_conv)

        # Sets, if applicable, the installed capacities of a component
        def capacityFix_conv(pyM, loc, compName):
            return (pyM.cap_conv[loc, compName] == compDict[compName]._capacityFix[loc]
                    if compDict[compName]._capacityFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrCapacityFix_conv = pyomo.Constraint(pyM.designDimensionVarSet_conv, rule=capacityFix_conv)

        # Sets, if applicable, the binary design variables of a component
        def designBinFix_conv(pyM, loc, compName):
            return (pyM.designBin_conv[loc, compName] == compDict[compName]._designDecisionFix[loc]
                    if compDict[compName]._designDecisionFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrDesignBinFix_conv = pyomo.Constraint(pyM.designDecisionVarSet_conv, rule=designBinFix_conv)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        def op1_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] <= pyM.cap_conv[loc, compName] * esM._hoursPerTimeStep
        pyM.ConstrOperation1_conv = pyomo.Constraint(pyM.opConstrSet1_conv, pyM.timeSet, rule=op1_conv)

        # Operation [energyUnit] equal to the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op2_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] == pyM.cap_conv[loc, compName] * \
                   compDict[compName]._operationRateFix[loc][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation2_conv = pyomo.Constraint(pyM.opConstrSet2_conv, pyM.timeSet, rule=op2_conv)

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op3_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] <= pyM.cap_conv[loc, compName] * \
                   compDict[compName]._operationRateMax[loc][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation3_conv = pyomo.Constraint(pyM.opConstrSet3_conv, pyM.timeSet, rule=op3_conv)

        # Operation [energyUnit] equal to the operation time series [energyUnit]
        def op4_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] == compDict[compName]._operationRateFix[loc][p, t]
        pyM.ConstrOperation4_conv = pyomo.Constraint(pyM.opConstrSet4_conv, pyM.timeSet, rule=op4_conv)

        # Operation [energyUnit] limited by the operation time series [energyUnit]
        def op5_conv(pyM, loc, compName, p, t):
            return pyM.op_conv[loc, compName, p, t] <= compDict[compName]._operationRateMax[loc][p, t]
        pyM.ConstrOperation5_conv = pyomo.Constraint(pyM.opConstrSet5_conv, pyM.timeSet, rule=op5_conv)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        return sum(pyM.cap_conv[loc, compName] / self._componentsDict[compName]._capacityMax[loc]
                   for compName in self._componentsDict if self._componentsDict[compName]._sharedPotentialID == key and
                   (loc, compName) in pyM.designDimensionVarSet_conv)

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([(commod in comp._commodityConversionFactors and comp._commodityConversionFactors[commod] != 0)
                    and comp._locationalEligibility[loc] == 1 for comp in self._componentsDict.values()])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t):
        return sum(pyM.op_conv[loc, compName, p, t] * self._componentsDict[compName]._commodityConversionFactors[commod]
                   for compName in pyM.operationVarDict_conv[loc]
                   if commod in self._componentsDict[compName]._commodityConversionFactors)

    def getObjectiveFunctionContribution(self, esM, pyM):
        compDict = self._componentsDict

        capexDim = sum(compDict[compName]._capexPerDesignDimension[loc] * pyM.cap_conv[loc, compName] /
                       compDict[compName]._CCF[loc] for loc, compName in pyM.cap_conv)

        capexDec = sum(compDict[compName]._capexForDesignDecision[loc] * pyM.designBin_conv[loc, compName] /
                       compDict[compName]._CCF[loc] for loc, compName in pyM.designBin_conv)

        opexDim = sum(compDict[compName]._opexPerDesignDimension[loc] * pyM.cap_conv[loc, compName]
                      for loc, compName in pyM.cap_conv)

        opexDec = sum(compDict[compName]._opexForDesignDecision[loc] * pyM.designBin_conv[loc, compName]
                      for loc, compName in pyM.designBin_conv)

        opexOp = sum(compDict[compName]._opexPerOperation[loc] *
                     sum(pyM.op_conv[loc, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet)
                     for loc, compNames in pyM.operationVarDict_conv.items()
                     for compName in compNames) / esM._years

        return capexDim + capexDec + opexDim + opexDec + opexOp

    ####################################################################################################################
    #                                  Return optimal values of the component class                                    #
    ####################################################################################################################

    def getOptimalValues(self, pyM):
        pass
