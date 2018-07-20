from FINE.component import Component, ComponentModeling
from FINE import utils
import warnings
import pyomo.environ as pyomo
import pandas as pd


class Transmission(Component):
    """
    Doc
    """
    def __init__(self, esM, name, commodity, losses=0, distances=None,
                 hasCapacityVariable=True, capacityVariableDomain='continuous', capacityPerPlantUnit=1,
                 hasIsBuiltBinaryVariable=False, bigM=None,
                 operationRateMax=None, operationRateFix=None, tsaWeight=1,
                 locationalEligibility=None, capacityMin=None, capacityMax=None, sharedPotentialID=None,
                 capacityFix=None, isBuiltFix=None,
                 investPerCapacity=0, investIfBuilt=0, opexPerOperation=0, opexPerCapacity=0,
                 opexIfBuilt=0, interestRate=0.08, economicLifetime=10):
        # TODO add unit checks
        # Set general component data
        utils.checkCommodities(esM, {commodity})
        self._name, self._commodity = name, commodity
        self._distances = utils.checkAndSetDistances(esM, distances)
        self._losses = utils.checkAndSetTransmissionLosses(esM, losses, distances)

        # Set design variable modeling parameters
        utils.checkDesignVariableModelingParameters(capacityVariableDomain, hasCapacityVariable,
                                                    hasIsBuiltBinaryVariable, bigM)
        self._hasCapacityVariable = hasCapacityVariable
        self._capacityVariableDomain = capacityVariableDomain
        self._capacityPerPlantUnit = capacityPerPlantUnit
        self._hasIsBuiltBinaryVariable = hasIsBuiltBinaryVariable
        self._bigM = bigM

        # Set economic data
        self._investPerCapacity = utils.checkAndSetCostParameter(esM, name, investPerCapacity, '2dim')
        self._investIfBuilt = utils.checkAndSetCostParameter(esM, name, investIfBuilt, '2dim')
        self._opexPerOperation = utils.checkAndSetCostParameter(esM, name, opexPerOperation, '2dim')
        self._opexPerCapacity = utils.checkAndSetCostParameter(esM, name, opexPerCapacity, '2dim')
        self._opexIfBuilt = utils.checkAndSetCostParameter(esM, name, opexIfBuilt, '2dim')
        self._interestRate = utils.checkAndSetCostParameter(esM, name, interestRate, '2dim')
        self._economicLifetime = utils.checkAndSetCostParameter(esM, name, economicLifetime, '2dim')
        self._CCF = self.getCapitalChargeFactor()

        # Set location-specific operation parameters
        if operationRateMax is not None and operationRateFix is not None:
            operationRateMax = None
            warnings.warn('If operationRateFix is specified, the operationRateMax parameter is not required.\n' +
                          'The operationRateMax time series was set to None.')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateMax, locationalEligibility, '2dim')
        utils.checkOperationTimeSeriesInputParameters(esM, operationRateFix, locationalEligibility, '2dim')

        self._fullOperationRateMax = utils.setFormattedTimeSeries(operationRateMax)
        self._aggregatedOperationRateMax = None
        self._operationRateMax = utils.setFormattedTimeSeries(operationRateMax)

        self._fullOperationRateFix = utils.setFormattedTimeSeries(operationRateFix)
        self._aggregatedOperationRateFix = None
        self._operationRateFix = utils.setFormattedTimeSeries(operationRateFix)

        self._tsaWeight = tsaWeight

        # Set location-specific design parameters
        self._sharedPotentialID = sharedPotentialID
        utils.checkLocationSpecficDesignInputParams(esM, hasCapacityVariable, hasIsBuiltBinaryVariable,
                                                    capacityMin, capacityMax, capacityFix,
                                                    locationalEligibility, isBuiltFix, sharedPotentialID,
                                                    '2dim')
        self._capacityMin, self._capacityMax, self._capacityFix = capacityMin, capacityMax, capacityFix
        self._isBuiltFix = isBuiltFix

        # Set locational eligibility
        operationTimeSeries = operationRateFix if operationRateFix is not None else operationRateMax
        self._locationalEligibility = utils.setLocationalEligibility(esM, locationalEligibility, capacityMax,
                                                                     capacityFix, isBuiltFix,
                                                                     hasCapacityVariable, operationTimeSeries,
                                                                     '2dim')

        # Variables at optimum (set after optimization)
        self._capacityVariablesOptimum = None
        self._isBuiltVariablesOptimum = None
        self._operationVariablesOptimum = None

    def getCapitalChargeFactor(self):
        """ Computes and returns capital charge factor (inverse of annuity factor) """
        return 1 / self._interestRate - 1 / (pow(1 + self._interestRate, self._economicLifetime) * self._interestRate)

    def addToEnergySystemModel(self, esM):
        esM._isTimeSeriesDataClustered = False
        if self._name in esM._componentNames:
            if esM._componentNames[self._name] == TransmissionModeling.__name__:
                warnings.warn('Component identifier ' + self._name + ' already exists. Data will be overwritten.')
            else:
                raise ValueError('Component name ' + self._name + ' is not unique.')
        else:
            esM._componentNames.update({self._name: TransmissionModeling.__name__})
        mdl = TransmissionModeling.__name__
        if mdl not in esM._componentModelingDict:
            esM._componentModelingDict.update({mdl: TransmissionModeling()})
        esM._componentModelingDict[mdl]._componentsDict.update({self._name: self})

    def setTimeSeriesData(self, hasTSA):
        self._operationRateMax = self._aggregatedOperationRateMax if hasTSA else self._fullOperationRateMax
        self._operationRateFix = self._aggregatedOperationRateFix if hasTSA else self._fullOperationRateFix

    def getDataForTimeSeriesAggregation(self):
        fullOperationRate = self._fullOperationRateFix if self._fullOperationRateFix is not None \
            else self._fullOperationRateMax
        if fullOperationRate is not None:
            fullOperationRate = fullOperationRate.copy()
            uniqueIdentifiers = [self._name + "_operationRate_" + locationIn + '_' + locationOut
                                 for locationIn, locationOut in fullOperationRate.columns]
            compData = pd.DataFrame(index=fullOperationRate.index, columns=uniqueIdentifiers)
            compDict = {}
            for locationIn, locationOut in fullOperationRate.columns:
                uniqueIdentifier = self._name + "_operationRate_" + locationIn + '_' + locationOut
                compData[uniqueIdentifier] = fullOperationRate.pop((locationIn, locationOut))
                compDict.update({uniqueIdentifier: self._tsaWeight})
            return compData, compDict
        else:
            return None, {}

    def setAggregatedTimeSeriesData(self, data):
        fullOperationRate = self._fullOperationRateFix if self._fullOperationRateFix is not None \
            else self._fullOperationRateMax
        if fullOperationRate is not None:
            uniqueIdentifiers = [self._name + "_operationRate_" + locationIn + '_' + locationOut
                                 for locationIn, locationOut in fullOperationRate.columns]
            compData = data[uniqueIdentifiers].copy()
            compData = pd.DataFrame(index=data.index, columns=fullOperationRate.columns)
            for locationIn, locationOut in compData.columns:
                compData.loc[:, (locationIn, locationOut)] = \
                    data.loc[:, self._name + "_operationRate_" + locationIn + '_' + locationOut]
            if self._fullOperationRateFix is not None:
                self._aggregatedOperationRateFix = compData
            else:
                self._aggregatedOperationRateMax = compData


class TransmissionModeling(ComponentModeling):
    """ Doc """
    def __init__(self):
        self._componentsDict = {}
        self._capacityVariablesOptimum = None
        self._isBuiltVariablesOptimum = None
        self._operationVariablesOptimum = None

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
            return ((loc, loc_, compName) for loc in esM._locations for loc_ in esM._locations
                    for compName, comp in compDict.items()
                    if comp._locationalEligibility[loc][loc_] == 1 and comp._hasCapacityVariable)
        pyM.designDimensionVarSet_trans = pyomo.Set(dimen=3, initialize=initDesignVarSet)

        def initContinuousDesignVarSet(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName, in pyM.designDimensionVarSet_trans
                    if compDict[compName]._capacityVariableDomain == 'continuous')
        pyM.continuousDesignDimensionVarSet_trans = pyomo.Set(dimen=3, initialize=initContinuousDesignVarSet)

        def initDiscreteDesignVarSet(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.designDimensionVarSet_trans
                    if compDict[compName]._capacityVariableDomain == 'discrete')
        pyM.discreteDesignDimensionVarSet_trans = pyomo.Set(dimen=3, initialize=initDiscreteDesignVarSet)

        def initDesignDecisionVarSet(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.designDimensionVarSet_trans
                    if compDict[compName]._hasIsBuiltBinaryVariable)
        pyM.designDecisionVarSet_trans = pyomo.Set(dimen=3, initialize=initDesignDecisionVarSet)

        ################################################################################################################
        #                                     Declare operation variables sets                                         #
        ################################################################################################################

        def initOpVarSet(pyM):
            return ((loc, loc_, compName) for loc in esM._locations for loc_ in esM._locations
                    for compName, comp in compDict.items() if comp._locationalEligibility[loc][loc_] == 1)
        pyM.operationVarSet_trans = pyomo.Set(dimen=3, initialize=initOpVarSet)
        pyM.operationVarDict_transOut = {loc: {loc_: {compName for compName in compDict
                                                      if (loc, loc_, compName) in pyM.operationVarSet_trans}
                                               for loc_ in esM._locations} for loc in esM._locations}
        pyM.operationVarDict_transIn = {loc: {loc_: {compName for compName in compDict
                                                     if (loc_, loc, compName) in pyM.operationVarSet_trans}
                                              for loc_ in esM._locations} for loc in esM._locations}

        ################################################################################################################
        #                           Declare sets for case differentiation of operating modes                           #
        ################################################################################################################

        def initOpConstrSet1(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is None
                    and compDict[compName]._operationRateFix is None)
        pyM.opConstrSet1_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet1)

        def initOpConstrSet2(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet2_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet2)

        def initOpConstrSet3(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet3_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet3)

        def initOpConstrSet4(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if not
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateFix is not None)
        pyM.opConstrSet4_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet4)

        def initOpConstrSet5(pyM):
            return ((loc, loc_, compName) for loc, loc_, compName in pyM.operationVarSet_trans if not
                    compDict[compName]._hasCapacityVariable and compDict[compName]._operationRateMax is not None)
        pyM.opConstrSet5_trans = pyomo.Set(dimen=3, initialize=initOpConstrSet5)

        potentialDict = {} # TODO adapt for 2dim components
        for compName, comp in compDict.items():
            if comp._sharedPotentialID is not None:
                potentialDict.setdefault(comp._sharedPotentialID, []).append(compName)
        pyM.sharedPotentialTransmissionDict = potentialDict

    ####################################################################################################################
    #                                                Declare variables                                                 #
    ####################################################################################################################

    def declareVariables(self, esM, pyM):
        """ Declares design and operation variables """
        # Function for setting lower and upper capacity bounds
        def capBounds(pyM, loc, loc_, compName):
            comp = self._componentsDict[compName]
            return (comp._capacityMin[loc][loc_]
                    if (comp._capacityMin is not None and not comp._hasIsBuiltBinaryVariable) else 0,
                    comp._capacityMax[loc][loc_] if comp._capacityMax is not None else None)

        # Capacity of components [powerUnit]
        pyM.cap_trans = pyomo.Var(pyM.designDimensionVarSet_trans, domain=pyomo.NonNegativeReals, bounds=capBounds)
        # Number of components [-]
        pyM.nbReal_trans = pyomo.Var(pyM.continuousDesignDimensionVarSet_trans, domain=pyomo.NonNegativeReals)
        # Number of components [-]
        pyM.nbInt_trans = pyomo.Var(pyM.discreteDesignDimensionVarSet_trans, domain=pyomo.NonNegativeIntegers)
        # Binary variables [-], indicate if a component is considered at a location or not
        pyM.designBin_trans = pyomo.Var(pyM.designDecisionVarSet_trans, domain=pyomo.Binary)
        # Operation of component [energyUnit]
        pyM.op_trans = pyomo.Var(pyM.operationVarSet_trans, pyM.timeSet, domain=pyomo.NonNegativeReals)

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
        def capToNbReal_trans(pyM, loc, loc_, compName):
            return pyM.cap_trans[loc, loc_, compName] == \
                   pyM.nbReal_trans[loc, loc_, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbReal_trans = pyomo.Constraint(pyM.continuousDesignDimensionVarSet_trans,
                                                         rule=capToNbReal_trans)

        # Determine the components' capacities from the number of installed units
        def capToNbInt_trans(pyM, loc, loc_, compName):
            return pyM.cap_trans[loc, loc_, compName] == \
                   pyM.nbInt_trans[loc, loc_, compName] * compDict[compName]._capacityPerPlantUnit
        pyM.ConstrCapToNbInt_trans = pyomo.Constraint(pyM.discreteDesignDimensionVarSet_trans,
                                                       rule=capToNbInt_trans)

        # Enforce the consideration of the binary design variables of a component
        def bigM_trans(pyM, loc, loc_, compName):
            return pyM.cap_trans[loc, loc_, compName] <= \
                   compDict[compName]._bigM * pyM.designBin_trans[loc, loc_, compName]
        pyM.ConstrBigM_trans = pyomo.Constraint(pyM.designDecisionVarSet_trans, rule=bigM_trans)

        # Enforce the consideration of minimum capacities for components with design decision variables
        def capacityMinDec_trans(pyM, loc, loc_, compName):
            return (pyM.cap_trans[loc, loc_, compName] >= compDict[compName]._capacityMin[loc][loc_] *
                    pyM.designBin_trans[loc, loc_, compName] if compDict[compName]._capacityMin is not None
                    else pyomo.Constraint.Skip)
        pyM.ConstrCapacityMinDec_trans = pyomo.Constraint(pyM.designDecisionVarSet_trans, rule=capacityMinDec_trans)

        # Sets, if applicable, the installed capacities of a component
        def capacityFix_trans(pyM, loc, loc_, compName):
            return (pyM.cap_trans[loc, loc_, compName] == compDict[compName]._capacityFix[loc][loc_]
                    if compDict[compName]._capacityFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrCapacityFix_trans = pyomo.Constraint(pyM.designDimensionVarSet_trans, rule=capacityFix_trans)

        # Sets, if applicable, the binary design variables of a component
        def designBinFix_trans(pyM, loc, loc_, compName):
            return (pyM.designBin_trans[loc, loc_, compName] == compDict[compName]._isBuiltFix[loc][loc_]
                    if compDict[compName]._isBuiltFix is not None else pyomo.Constraint.Skip)
        pyM.ConstrDesignBinFix_trans = pyomo.Constraint(pyM.designDecisionVarSet_trans, rule=designBinFix_trans)

        def sharedPotentialTransmission(pyM, key, loc, loc_):
            return sum(pyM.cap_trans[loc, loc_, compName] / compDict[compName].capacityMax[loc][loc_]
                       for compName in compDict if compDict[compName]._sharedPotentialID == key
                       and (loc, loc_, compName) in pyM.designDimensionVarSet_trans)
        pyM.ConstSharedPotential_trans = \
            pyomo.Constraint(pyM.sharedPotentialTransmissionDict.keys(), esM._locations, esM._locations,
                             rule=sharedPotentialTransmission)

        def symmetricalCapacity_trans(pyM, loc, loc_, compName):
            return pyM.cap_trans[loc, loc_, compName] == pyM.cap_trans[loc_, loc, compName]
        pyM.ConstrSymmetricalCapacity_trans = \
            pyomo.Constraint(pyM.designDimensionVarSet_trans, rule=symmetricalCapacity_trans)

        ################################################################################################################
        #                                      Declare time dependent constraints                                      #
        ################################################################################################################

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by the hours per time step
        def op1_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] <= \
                   pyM.cap_trans[loc, loc_, compName] * esM._hoursPerTimeStep
        pyM.ConstrOperation1_trans = pyomo.Constraint(pyM.opConstrSet1_trans, pyM.timeSet, rule=op1_trans)

        # Operation [energyUnit] equal to the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op2_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] == pyM.cap_trans[loc, loc_, compName] * \
                   compDict[compName]._operationRateFix[loc, loc_][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation2_trans = pyomo.Constraint(pyM.opConstrSet2_trans, pyM.timeSet, rule=op2_trans)

        # Operation [energyUnit] limited by the installed capacity [powerUnit] multiplied by operation time series
        # [powerUnit/powerUnit] and the hours per time step [h])
        def op3_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] <= pyM.cap_trans[loc, loc_, compName] * \
                   compDict[compName]._operationRateMax[loc, loc_][p, t] * esM._hoursPerTimeStep
        pyM.ConstrOperation3_trans = pyomo.Constraint(pyM.opConstrSet3_trans, pyM.timeSet, rule=op3_trans)

        # Operation [energyUnit] equal to the operation time series [energyUnit]
        def op4_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] == compDict[compName]._operationRateFix[loc, loc_][p, t]
        pyM.ConstrOperation4_trans = pyomo.Constraint(pyM.opConstrSet4_trans, pyM.timeSet, rule=op4_trans)

        # Operation [energyUnit] limited by the operation time series [energyUnit]
        def op5_trans(pyM, loc, loc_, compName, p, t):
            return pyM.op_trans[loc, loc_, compName, p, t] <= compDict[compName]._operationRateMax[loc, loc_][p, t]
        pyM.ConstrOperation5_trans = pyomo.Constraint(pyM.opConstrSet5_trans, pyM.timeSet, rule=op5_trans)

    ####################################################################################################################
    #        Declare component contributions to basic EnergySystemModel constraints and its objective function         #
    ####################################################################################################################

    def getSharedPotentialContribution(self, pyM, key, loc):
        return 0

    def hasOpVariablesForLocationCommodity(self, esM, loc, commod):
        return any([comp._commodity == commod and
                    (comp._locationalEligibility[loc][loc_] == 1 or comp._locationalEligibility[loc_][loc] == 1)
                    for comp in self._componentsDict.values() for loc_ in esM._locations])

    def getCommodityBalanceContribution(self, pyM, commod, loc, p, t): # TODO losses connected to distances
        return sum(pyM.op_trans[loc_, loc, compName, p, t] *
                   (1 - self._componentsDict[compName]._losses[loc_][loc] *
                    self._componentsDict[compName]._distances[loc_][loc])
                   for loc_ in pyM.operationVarDict_transIn[loc].keys()
                   for compName in pyM.operationVarDict_transIn[loc][loc_]
                   if commod in self._componentsDict[compName]._commodity) - \
               sum(pyM.op_trans[loc, loc_, compName, p, t]
                   for loc_ in pyM.operationVarDict_transOut[loc].keys()
                   for compName in pyM.operationVarDict_transOut[loc][loc_]
                   if commod in self._componentsDict[compName]._commodity)

    def getObjectiveFunctionContribution(self, esM, pyM):
        # TODO replace 0.5 with factor which is one when non-directional and 0.5 when bi-directional
        compDict = self._componentsDict

        capexDim = sum(compDict[compName]._investPerCapacity[loc][loc_] * pyM.cap_trans[loc, loc_, compName] *
                       compDict[compName]._distances[loc][loc_] /
                       compDict[compName]._CCF[loc][loc_] for loc, loc_, compName in pyM.cap_trans) * 0.5

        capexDec = sum(compDict[compName]._investIfBuilt[loc][loc_] *
                       pyM.designBin_trans[loc, loc_, compName] * compDict[compName]._distances[loc][loc_] /
                       compDict[compName]._CCF[loc][loc_] for loc, loc_, compName in pyM.designBin_trans) * 0.5

        opexDim = sum(compDict[compName]._opexPerCapacity[loc][loc_] * pyM.cap_trans[loc, loc_, compName] *
                      compDict[compName]._distances[loc][loc_] for loc, loc_, compName in pyM.cap_trans) * 0.5

        opexDec = sum(compDict[compName]._opexIfBuilt[loc][loc_] * pyM.designBin_trans[loc, loc_, compName] *
                      compDict[compName]._distances[loc][loc_] for loc, loc_, compName in pyM.designBin_trans) * 0.5

        opexOp = sum(compDict[compName]._opexPerOperation[loc][loc_] *
                     sum(pyM.op_trans[loc, loc_, compName, p, t] * esM._periodOccurrences[p] for p, t in pyM.timeSet)
                     for loc, subDict in pyM.operationVarDict_transOut.items()
                     for loc_, compNames in subDict.items()
                     for compName in compNames) / esM._numberOfYears

        return capexDim + capexDec + opexDim + opexDec + opexOp

    def setOptimalValues(self, esM, pyM):
        optVal = utils.formatOptimizationOutput(pyM.cap_trans.get_values(), 'designVariables', '1dim')
        self._capacityVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_capacityVariablesOptimum', self._componentsDict)

        optVal = utils.formatOptimizationOutput(pyM.designBin_trans.get_values(), 'designVariables', '1dim')
        self._isBuiltVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_isBuiltVariablesOptimum', self._componentsDict)

        optVal = utils.formatOptimizationOutput(pyM.op_trans.get_values(), 'operationVariables', '1dim',
                                                esM._periodsOrder)
        self._operationVariablesOptimum = optVal
        utils.setOptimalComponentVariables(optVal, '_operationVariablesOptimum', self._componentsDict)

    def getOptimalCapacities(self):
        return self._capacitiesOpt