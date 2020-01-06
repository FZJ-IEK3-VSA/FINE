"""
Last edited: January 6 2020

@author: Lara Welder
"""
import pandas as pd

def getNetworkData(componentName, operationVariablesOptimumData=None, connections=None, distances=None, esM=None):
    """[summary]
    
    :param componentName: [description]
    :type componentName: [type]

    :param operationVariablesOptimumData: [description], defaults to None
    :type operationVariablesOptimumData: [type], optional

    :param connections: [description], defaults to None
    :type connections: [type], optional

    :param distances: [description], defaults to None
    :type distances: [type], optional

    :param esM: [description], defaults to None
    :type esM: [type], optional

    :return: [description]
    :rtype: [type]
    """

    #TODO adapt docstring and check type and value correctness

    ### PART I: Determine considered connections as well as injection and withdrawal rates ###

    # Get the original optimal operation variables
    if operationVariablesOptimumData is not None:
        op = operationVariablesOptimumData
    else:
        op = esM.componentModelingDict[esM.componentNames[componentName]].\
            getOptimalValues('operationVariablesOptimum')['values'].loc[componentName]

    # Get a map of the component's network
    if connections is not None:
        mapN = {}
        for conn in connections:
            loc, loc_ = conn
            mapN.setdefault(loc, {}).update({loc_: loc + '_' + loc_})
            mapN.setdefault(loc_, {}).update({loc: loc_ + '_' + loc})
    else:
        mapN = esM.getComponent(componentName)._mapL


    # Initialize list for nodal injection and withdrawal time series data
    sourceSinkTimeSeries, nodeIx = [], []

    # Reset connections set (not all indices might be in the operationVariablesOptimumData data)
    connections = set()

    # For each node loc, compute the injection and withdrawal rates 
    for loc, locConn in mapN.items():
        # As in a few cases zero columns/ rows are dropped from data frames, two lists
        # of eligible connection indices are created.
        ixIn, ixOut = [], []
        for loc_, conn in locConn.items():
            if (loc, loc_) in op.index:
                ixOut.append((loc, loc_)), connections.add((loc, loc_))
            if (loc_, loc) in op.index:
                ixIn.append((loc_, loc)), connections.add((loc_, loc))
        
        # If either list has at least one entry, the incoming and outgoing flows are selected
        # from the original optimal flow variables and aggregated. The resulting commodity
        # withdrawals from the network are positive while injections are negative.
        if (len(ixIn) != 0) | (len(ixOut) != 0):
            sourceSinkTimeSeries.append(op.loc[ixIn].sum()-op.loc[ixOut].sum())
            nodeIx.append(loc)
            
    # Concat data to a pandas dataframe
    sourceSinkTimeSeries = pd.concat(sourceSinkTimeSeries, keys=nodeIx, axis=1)

    ### PART II: Set or get distances ###
    if distances is not None:
        distances = distances
    else:
        distances = esM.getComponent(componentName).distances.copy()
        indexMap = esM.getComponent(componentName)._mapC
        distances.index = [indexMap[ix] for ix in distances.index if indexMap[ix] in connections]

    return connections, distances, sourceSinkTimeSeries


def createMinimumSpanningTree():
    #TODO
    return


def generateRobustScenarios():
    #TODO
    return

def determineOptimalDiscretePipelineSelection():
    #TODO
    return


def determineRobustDiscretePipelineDesign(componentName, diameters=[], diameterCost=[],
    pressureMin=70, pressureMax=100, physicalParameters={}, inputDataConversionFactor=1,
    operationVariablesOptimumData=None, connections=None, distances=None, esM=None, nDigits=6):
    """[summary]
    
    :param componentName: [description]
    :type componentName: [type]

    :param diameters: [description], defaults to []
    :type diameters: list, optional

    :param diameterCost: [description], defaults to []
    :type diameterCost: list, optional

    :param pressureMin: [description], defaults to 70
    :type pressureMin: int, optional

    :param pressureMax: [description], defaults to 100
    :type pressureMax: int, optional

    :param physicalParameters: [description], defaults to {}
    :type physicalParameters: dict, optional

    :param inputDataConversionFactor: [description], defaults to 1
    :type inputDataConversionFactor: int, optional

    :param operationVariablesOptimumData: [description], defaults to None
    :type operationVariablesOptimumData: [type], optional

    :param connections: [description], defaults to None
    :type connections: [type], optional

    :param distances: [description], defaults to None
    :type distances: [type], optional

    :param esM: [description], defaults to None
    :type esM: [type], optional

    :param nDigits: [description], defaults to 6
    :type nDigits: int, optional

    :return: [description]
    :rtype: [type]
    """

    #TODO adapt docstring and check type and value correctness

    # Get the connections of the network and its commodity injection (<= 0) and withdrawal
    # (>= 0) rates
    connections, distances, sourceSinkTimeSeries = \
        getNetworkData(componentName, operationVariablesOptimumData, connections, distances, esM)

    sourceSinkTimeSeries = inputDataConversionFactor * sourceSinkTimeSeries.round(nDigits)

    # Create a minimum spanning tree of the network with a reasonable logic
    # TODO
    print('Creating a minimum spanning tree...')
    createMinimumSpanningTree()

    # Generate robust set of operation scenarios
    # TODO I believe that, in the long run, we could also call FINE here but for now we can
    # just use the model you created.
    print('Creating a robust set of operation scenarios...')
    generateRobustScenarios()
    fluidInjectionWithdrawalScenarioSet = {}
    fluidFlowScenarioSet = {}

    # Determine optimal discrete pipeline selection
    # TODO
    print('Determining optimal discrete pipeline design under the consideration of pressure losses...')
    optimalDiameters = None
    pipelinePressureScenarioSet = {}
    determineOptimalDiscretePipelineSelection()

    # Return output of interest
    # TODO
    return fluidInjectionWithdrawalScenarioSet, fluidFlowScenarioSet, optimalDiameters, \
        pipelinePressureScenarioSet
