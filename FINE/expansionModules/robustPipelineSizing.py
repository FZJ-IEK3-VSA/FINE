"""
Last edited: January 6 2020

@author: Lara Welder
"""
import pandas as pd

def getInjectionWithdrawalRates(componentName='', esM=None, operationVariablesOptimumData=None):
    """
    Determines the injection and withdrawal rates into a network from a component in an
    EnergySystemModel object or based on the fluid flow data.

    :param componentName: name of the network component in the EnergySystemModel class
        (only required the fluid flows are to be obtained from the EnergySystemModel class)
        |br| * the default value is ''
    :type componentName: string

    :param esM: EnergySystemModel object with an optimized Pyomo instance (only needs to be
        specified if the operationVariablesOptimumData are to be obtained from the
        EnergySystemModel object) 
        |br| * the default value is None
    :type esM: FINE EnergySystemModel

    :param operationVariablesOptimumData: the injection and withdrawal rates into and out of the
        network can either be obtained from a DataFrame with the original fluid flows or an
        EnergySystemModel with an optimized Pyomo instance.
        In the former case, the argument is a pandas DataFrame with two index columns (specifying
        the names of the start and end node of a pipeline) and one index row (for the time steps).
        The data in the DataFrame denotes the flow coming from the start node and going to the end
        node [e.g. in kWh or Nm^3]. Example:

                     0   1  ... 8759
        node1 node2 0.1 0.0 ... 0.9
        node2 node3 0.0 0.3 ... 0.4
        node2 node1 0.9 0.9 ... 0.2
        node3 node2 1.1 0.2 ... 0.9

        |br| * the default value is None
    :type operationVariablesOptimumData: pandas DataFrame with non-negative floats

    :return: injection and withdrawal rates (withdrawals from the network are positive while
        injections are negative)
    :rtype: pandas DataFrame
    """
    #TODO check type and value correctness

    # Get the original optimal operation variables
    if operationVariablesOptimumData is not None:
        op = operationVariablesOptimumData
    else:
        op = esM.componentModelingDict[esM.componentNames[componentName]].\
            getOptimalValues('operationVariablesOptimum')['values'].loc[componentName]

    # Get a map of the component's network
    if esM is None:
        mapN = {}
        for conn in operationVariablesOptimumData.index:
            loc, loc_ = conn
            mapN.setdefault(loc, {}).update({loc_: loc + '_' + loc_})
            mapN.setdefault(loc_, {}).update({loc: loc_ + '_' + loc})
    else:
        mapN = esM.getComponent(componentName)._mapL


    # Initialize list for nodal injection and withdrawal time series data
    injectionWithdrawalRates, nodeIx = [], []

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
            injectionWithdrawalRates.append(op.loc[ixIn].sum()-op.loc[ixOut].sum())
            nodeIx.append(loc)
            
    # Concat data to a pandas dataframe
    injectionWithdrawalRates = pd.concat(injectionWithdrawalRates, keys=nodeIx, axis=1)

    return injectionWithdrawalRates

def getNetworkLengthsFromESM(componentName, esM):
    """
    Obtains the pipeline lengths of a transmission component in an EnergySystemModel class.
    
    :param componentName: name of the network component in the EnergySystemModel class
        (only required if the fluid flows are to be obtained from the EnergySystemModel class)
        |br| * the default value is ''
    :type componentName: string

    :param esM: EnergySystemModel object with an optimized Pyomo instance (only needs to be
        specified if the operationVariablesOptimumData are to be obtained from the
        EnergySystemModel object) 
        |br| * the default value is None
    :type esM: FINE EnergySystemModel

    :return: pipeline distances in the length unit specified in the esM object
    :rtype: pandas series
    """
    #TODO add type and value correctness check

    distances = esM.getComponent(componentName).distances.copy() 
    indexMap = esM.getComponent(componentName)._mapC
    distances.index = [indexMap[ix] for ix in distances.index]

    return distances


def createMinimumSpanningTree():
    #TODO
    return

def determineFullFluidFlowScenarioSet():
    #TODO
    return

def generateRobustScenarios():
    #TODO
    return

def determineOptimalDiscretePipelineSelection():
    #TODO
    return


def determineRobustDiscretePipelineDesign(injectionWithdrawalRates, lengths,
    diameters=[], investForDiameters=[], opexForDiameters=[],
    economicLifetime=30, interestRate=0.08, costUnit = '€', 
    pressureMin=70, pressureMax=100, physicalParameters={},
    originalFluidFlows=None, nDigits=6):
    """[summary]

    :param injectionWithdrawalRates: the argument is a pandas DataFrame with the index column
        denoting the timesteps and the index row denoting the name of the network's nodes.
        Injection are denoted with negative floats and withdrawal with positive floats
        in [Nm^3]. Example:

              node1 node2 node3
        0      -4     2     2
        1       3   -1.5  -1.5
        ...    ...   ...   ...
        8759    0    -1     1.

    :type injectionWithdrawalRates: pandas DataFrame with floats

    :param lengths: the parameter is a pandas Series with the indices being tuples of the
        network's nodes and the values being the lengths of the pipelines in [m]. Example:

        (node1, node2)   1000
        (node2, node3)  50000
        (node2, node1)   1000
        (node3, node2)  50000

    :type lengths: pandas Series, optional
    
    :param diameters: list of eligible pipeline diameters in mm
        |br| * the default value is []
    :type diameters: list

    :param investForDiameters: investments of the individual pipeline diameters, specified
        in costUnit/m (--> default €/m)
        |br| * the default value is []
    :type investForDiameters: list

    :param opexForDiameters: opex of the individual pipeline diameter classes, specified
        in costUnit/m/a (--> default €/m/a)
        |br| * the default value is []
    :type opexForDiameters: list

    :param economicLifetime: economic lifetime of the pipeline network in years
        |br| * the default value is 30
    :type economicLifetime: integer > 0

    :param interestRate: considered interest rate
        |br| * the default value is 0.08
    :type interestRate: float (0,1)

    :param costUnit: cost unit of the invest
        |br| * the default value is '€
    :type costUnit: string    

    :param pressureMin: minimum pressure in the pipeline network
        |br| * the default value is 70
    :type pressureMin: non-negative float

    :param pressureMax: maximum pressure in the pipeline network
        |br| * the default value is 100
    :type pressureMax: non-negative float

    :param physicalParameters: [description]
        #TODO required parameters could either be added as a dictionary or added as separate key
        word arguments
        |br| * the default value is {}
    :type physicalParameters: dict, optional

    :param nDigits: number of digits used in the pandas round function. Is applied to the
        specified or determined injection and withdrawal rates.
        |br| * the default value is 6
    :type nDigits: int

    :return: [description]
        #TODO
    :rtype: [type]

    """
    #TODO adapt docstring and check type and value correctness

    # Create a minimum spanning tree of the network with a reasonable logic
    # TODO
    print('Creating a minimum spanning tree...')
    createMinimumSpanningTree()

    # Determine flows on unrestricted minimum spanning trees
    # TODO (this would be nice to have for all operation scenarios)
    print('Determining fluid flow scenarios on minimum spanning tree...')
    determineFullFluidFlowScenarioSet()
    fluidFlowScenarioSet = {}

    # Generate robust set of operation scenarios
    # TODO I believe that, in the long run, we could also call FINE here but for now we can
    # just use the model you created.
    print('Creating a robust set of operation scenarios...')
    generateRobustScenarios()
    robustFluidInjectionWithdrawalScenarioSet = {}
    robustFluidFlowScenarioSet = {}

    # Determine optimal discrete pipeline selection
    # TODO
    print('Determining optimal discrete pipeline design under the consideration of pressure losses...')
    optimalDiameters = None
    pipelinePressureScenarioSet = {}
    determineOptimalDiscretePipelineSelection()

    # Add some output which somehow quantifies the difference between the original and the new
    # pipeline design (for this additional input argument are required)
    #TODO

    # Return output of interest
    # TODO
    return None