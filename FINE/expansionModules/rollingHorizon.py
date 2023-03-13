import FINE as fn
from FINE import utils
from FINE.IOManagement.standardIO import writeOptimizationOutputToExcel
import os
import copy

def rollingHorizonOptimization(
    esM,
    scenario_name,
    resultExportPath,
    numberOfInvestmentPeriodsForRollingHorizon,
    timeSeriesAggregation=True,
    numberOfTypicalPeriods=7,
    numberOfTimeStepsPerPeriod=24,
    numberOfSegments=16,
    clusterMethod="hierarchical",
    solver="gurobi",
    optimizationSpecs="",
):
    """
    If numberOfInvestmentPeriodsForRollingHorizon == numberOfInvestmentPeriods -> Perfect Foresight,
    If numberOfInvestmentPeriodsForRollingHorizon == 1 -> Foresight, else Rolling Horizon
    """

    # checks for data input
    if esM.numberOfInvestmentPeriods < 2:
        raise ValueError("At least two investmentperiods required for rolling horizon.")
    if esM.numberOfInvestmentPeriods <= numberOfInvestmentPeriodsForRollingHorizon:
        raise ValueError("There must be at least one more investment period in the transformation pathway than in the rolling horizon interval")

    utils.isStrictlyPositiveInt(numberOfInvestmentPeriodsForRollingHorizon)
    utils.isStrictlyPositiveNumber(numberOfInvestmentPeriodsForRollingHorizon)

    # 0. set up rolling horizon intervals
    interval=esM.investmentPeriodInterval
    rollingHorizonIntervals= \
        [list(
            range(
                start,
                start+interval*numberOfInvestmentPeriodsForRollingHorizon,
                interval
            )
        ) 
        for start in esM.investmentPeriodNames 
        if start+interval*(numberOfInvestmentPeriodsForRollingHorizon-1) in esM.investmentPeriodNames] 

    # extract all information of original esM
    esmDict, compDict = fn.dictIO.exportToDict(esM)

    print("Starting rolling horizon optimization.")

    esM_results={}

    for rollingHorizonYears in rollingHorizonIntervals:
        print(f"Initizializing rolling horizon optimization for {rollingHorizonYears}...")
        # 1. Analyse components and create dicts for adding them
        rollingHorizonCompDict=copy.deepcopy(dict(compDict))
        for classname in rollingHorizonCompDict:
            for comp in rollingHorizonCompDict[classname]:
                # get data for rolling horizon years from perfect foresight model
                parameterYears=rollingHorizonYears # TODO change!
                for parameter_name, parameter_value in rollingHorizonCompDict[classname][comp].items():
                    # 1.1 stock commissioning 
                    if parameter_name == "stockCommissioning":
                        # first rolling horizon requires no changes, just external stock,
                        # further years needs internal optimization results
                        if rollingHorizonYears != rollingHorizonIntervals[0]:
                            # get previous year
                            previousYear=rollingHorizonYears[0]-interval
                            # get model class from previous rolling horizon optimization
                            mdl_class=esM_results[previousYear].componentNames[comp]
                            # get commissioning results of previous rolling horizon
                            previousCommissioning=esM_results[previousYear].getOptimizationSummary(mdl_class,ip=previousYear).loc[comp,'commissioning'].squeeze()

                            # add commissioning of previous runs as stock, if there was commissioning
                            if previousCommissioning.sum() > 0: 
                                # a) if no stock previously existed, create new structure
                                if rollingHorizonCompDict[classname][comp]["stockCommissioning"] is None:
                                    rollingHorizonCompDict[classname][comp]["stockCommissioning"]={}
                                    rollingHorizonCompDict[classname][comp]["stockCommissioning"][previousYear]=previousCommissioning
                                # b) else add to structure
                                else:
                                    rollingHorizonCompDict[classname][comp]["stockCommissioning"][previousYear] = previousCommissioning
                    # 1.2 commodity conversion factors
                    elif parameter_name == "commodityConversionFactors":
                        # check for ip dependendy
                        if parameter_value.keys()[0] in esM.investmentPeriods:
                            # filter for years of rolling horizon time frame
                            new_parameter_value={key:value for (key,value) in parameter_value.items() if key in parameterYears}
                            rollingHorizonCompDict[classname][comp][parameter_name]=new_parameter_value
                        else:
                            pass
                    # 1.3 other parameter which are yearly dependent
                    elif isinstance(parameter_value,dict):
                        # filter for years of rolling horizon time frame
                        new_parameter_value={key:value for (key,value) in parameter_value.items() if key in parameterYears}
                        rollingHorizonCompDict[classname][comp][parameter_name]=new_parameter_value
                    # 1.4 other parameters, which do not change over time
                    else:
                        pass

                
        # 2. init esm with new startYear and numberOfInvestmentPeriods and add components
        rollingHorizonEsmDict=esmDict.copy()
        rollingHorizonEsmDict["startYear"]=rollingHorizonYears[0]
        rollingHorizonEsmDict["numberOfInvestmentPeriods"]=numberOfInvestmentPeriodsForRollingHorizon
        rollingHorizonEsm = fn.EnergySystemModel(**rollingHorizonEsmDict)
        # add components per class
        for classname in rollingHorizonCompDict:
            for comp in rollingHorizonCompDict[classname]:
                rollingHorizonEsm.add(
                    getattr(fn, classname)(
                        esM=rollingHorizonEsm, 
                        **rollingHorizonCompDict[classname][comp] # information of component
                    )
                )

        # 3. optimize the rolling horizon esM
        if timeSeriesAggregation:
            rollingHorizonEsm.aggregateTemporally(
                numberOfTypicalPeriods=numberOfTypicalPeriods,
                numberOfTimeStepsPerPeriod=numberOfTimeStepsPerPeriod,
                numberOfSegmentsPerPeriod=numberOfSegments,
                segmentation=True,
                clusterMethod=clusterMethod,
                solver=solver,
                sortValues=True,
                rescaleClusterPeriods=True,
                representationMethod=None,
            )

        rollingHorizonEsm.optimize(
            declaresOptimizationProblem=True,
            timeSeriesAggregation=timeSeriesAggregation,
            solver=solver,
            optimizationSpecs=optimizationSpecs,
        )

        # 4. export optimization summaries
        # For all optimization than the last rolling horizon optimization, export only first year.
        # For the last rolling horizon interval, export all years.
        if rollingHorizonYears != rollingHorizonIntervals[-1]:
            exportYears=[rollingHorizonYears[0]]
        else:
            exportYears=rollingHorizonYears

        for year in exportYears:
            writeOptimizationOutputToExcel(
                rollingHorizonEsm,
                outputFileName=os.path.join(
                    resultExportPath,
                    f"{scenario_name}_rollingHorizon"
                    ),
                optSumOutputLevel={
                    "SourceSinkModel": 0,
                    "ConversionModel": 0,
                    "StorageModel": 0,
                    "TransmissionModel": 0,
                    "LOPFModel": 0,
                },
                optValOutputLevel={
                    "SourceSinkModel": 0,
                    "ConversionModel": 0,
                    "StorageModel": 0,
                    "TransmissionModel": 0,
                    "LOPFModel": 0,
                },
                investmentPeriod=year
            )

        # 5. save esM's
        esM_results[rollingHorizonYears[0]] = rollingHorizonEsm

        print(f"Finished rolling horizon optimization for {rollingHorizonYears}")

    print("Finished rolling horizon optimization.")
    return esM_results