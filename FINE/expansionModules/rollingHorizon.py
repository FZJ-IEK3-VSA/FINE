import FINE as fn
from FINE import utils
from FINE.IOManagement.standardIO import writeOptimizationOutputToExcel
import os

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
"""If numberOfInvestmentPeriodsForRollingHorizon == numberOfInvestmentPeriods -> Forecasting, else Rolling Horizon"""

    # checks
    if esM.numberOfInvestmentPeriods < 3:
        raise ValueError("At least three investmentperiods required for rolling horizon.")
    if esM.numberOfInvestmentPeriods < numberOfInvestmentPeriodsForRollingHorizon:
        raise ValueError("There must be at least on more investment period in the rolling horizon interval than in the transformation pathway")

    # 0. set up rolling horizon intervals
    rollingHorizonIntervals= \
        [list(range(
            start,
            start+esM.investmentPeriodInterval*numberOfInvestmentPeriodsForRollingHorizon,
            esM.investmentPeriodInterval)) 
        for start in esM.investmentPeriodNames 
        if start+esM.investmentPeriodInterval*numberOfInvestmentPeriodsForRollingHorizon in esM.investmentPeriodNames # filter out if years after transformation pathway
        ]

    # extract all information of original esM
    esmDict, compDict = fn.dictIO.exportToDict(esM)

    print("Starting rolling horizon optimization.")

    for rollingHorizonYears in rollingHorizonIntervals:
        print(f"Initizializing rolling horizon optimization for {rollingHorizonYears}...")
        # 1. Analyse components and create dicts for adding them
        rollingHorizonCompDict=compDict.copy()
        for classname in rollingHorizonCompDict:
            for comp in rollingHorizonCompDict[classname]:
                # first thing: stockCommissioning!
                # for first rolling horizon there is no internally created stock
                if rollingHorizonYears == rollingHorizonIntervals[0]:
                    if esM.getComponent(comp).stockCommissioning is None:
                        stockYears=[]
                    else:
                        stockYears=list(esM.getComponent(comp).stockCommissioning.index) # years of initial esm
                    pass # use stock years
                # all further years have the stock commissioning of the previous run plus the optimization output 
                # as stock (if something was commissioned) - oder ist das egal? dann einfach als 0??
                else:
                    if rollingHorizonEsm.getComponent(comp).stockCommissioning is None:
                        _historicalStock=[]
                    else:
                        _historicalStock=list(rollingHorizonEsm.getComponent(comp).stockCommissioning.index)

                    # get results of previous run
                    rollingHorizonEsm
                    if commissioing in previous run:

                    stockYears= (
                        _historicalStock+ 
                        [rollingHorizonYears[0]-rollingHorizonYears.investmentPeriodInterval]
                        )
                    pass # TODO add internal stock
                # check if stock exists (historical) or new stock was added
                

                parameterYears=stockYears+rollingHorizonYears
                for parameter_name, parameter_value in rollingHorizonCompDict[classname][comp].items():
                    if parameter_name == "stockCommissioning":
                        # already done previously
                        pass
                    elif parameter_name == "commodityConversionFactors":
                        # check for ip dependendy
                        if parameter_value.keys()[0] in esM.investmentPeriods:
                            # filter for years of rolling horizon time frame
                            new_parameter_value={key:value for (key,value) in parameter_value.items() if key in parameterYears}
                            rollingHorizonCompDict[classname][comp][parameter_name]=new_parameter_value
                        else:
                            pass
                    elif isinstance(parameter_value,dict):
                        # filter for years of rolling horizon time frame
                        new_parameter_value={key:value for (key,value) in parameter_value.items() if key in parameterYears}
                        rollingHorizonCompDict[classname][comp][parameter_name]=new_parameter_value
                    else:
                        pass

                
        # 2. init esm with new startYear and numberOfInvestmentPeriods and add components
        rollingHorizonEsmDict=esmDict.copy()
        rollingHorizonEsmDict["startYear"]=rollingHorizonYears[0]
        rollingHorizonEsmDict["numberOfInvestmentPeriods"]=numberOfInvestmentPeriodsForRollingHorizon
        rollingHorizonEsm = fn.EnergySystemModel(**rollingHorizonEsmDict)
        # add components per class
        for classname in rollingHorizonCompDict:
            rollingHorizonEsm.add(getattr(fn, classname)(rollingHorizonEsm, **rollingHorizonCompDict[classname][comp]))

        # 3. optimize the rolling horizon esM
        # Optimization
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

        # 4. export optimization summary of initial year
        writeOptimizationOutputToExcel(
            rollingHorizonEsm,
            outputFileName=os.path.join(
                resultExportPath,
                f"{scenario_name}_rollingHorizon_{rollingHorizonYears[0]}"
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
            investmentPeriod=rollingHorizonYears[0]
        )  

        print(f"Finished rolling horizon optimization for {rollingHorizonYears}")

    print("Finished rolling horizon optimization.")