"""
Last edited: January 20 2020

@author: Lara Welder, Johannes Thürauf
the approaches used are described in
Robinius et. al. (2019) "Robust Optimal Discrete Arc Sizing for Tree-Shaped Potential Networks"
and they are further developed with the help of
Theorem 10 of Labbé et. al. (2019) "Bookings in the European gas market: characterisation of feasibility and
computational complexity results"
and Lemma 3.4 and 3.5 of Schewe et. al. (preprint 2020) "Computing Technical Capacities in the European Entry-Exit
Gas Market is NP-Hard"
"""
import pandas as pd
from FINE import utils
import networkx as nx
import math
import pyomo.environ as py
import warnings
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import numpy as np
import copy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# local type und value checker

def isPandasDataFrameNumber(dataframe):
    # check if dataframe is a pandas dataframe and if each value is float or int
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("The input argument has to be a pandas DataFrame")
    else:
        if not dataframe.select_dtypes(exclude=["float", "int"]).empty:
            raise ValueError("The input pandas DataFrame has to contain only floats or ints")


def isPandasSeriesPositiveNumber(pandasSeries):
    # Check if the input argument is a pandas series and it contains only positive numbers
    if not isinstance(pandasSeries, pd.Series):
        raise TypeError("The input argument has to be a pandas series")
    else:
        for index in pandasSeries.index:
            utils.isPositiveNumber(pandasSeries[index])


def isNetworkxGraph(graph):
    # Check if the input argument is a networkx graph
    if not isinstance(graph, nx.Graph):
        raise TypeError("The input argument has to be a networkx graph")


def isDictionaryPositiveNumber(dictionary):
    # Check if the input argument is a dictionary with positive numbers as values
    if not isinstance(dictionary, dict):
        raise TypeError("The input argument has to be a dictionary")
    else:
        for key in dictionary.keys():
            utils.isPositiveNumber(dictionary[key])


def checkLowerUpperBoundsOfDicts(lowerDict, upperDict):
    # check if lowerDict and upperDict have the same keys and if lowerDict[key] <= upperDict[key] holds
    if not (lowerDict.keys() == upperDict.keys()):
        raise ValueError("The input arguments have to have the same keys")
    else:
        for key in lowerDict.keys():
            if lowerDict[key] > upperDict[key]:
                raise ValueError("The lower bound has to be the smaller than the upper bound")


def isListOfStrings(strings):
    # check if strings is list of strings
    if not isinstance(strings, list):
        raise TypeError("The input argument has to be a list")
    else:
        for string in strings:
            utils.isString(string)


def isBool(boolean):
    # check if boolean is a bool
    if not isinstance(boolean, bool):
        raise TypeError("The input argument has to be a bool")


# End utils checks


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
    # TODO check type and value correctness @Juelich

    # Get the original optimal operation variables
    if operationVariablesOptimumData is not None:
        op = operationVariablesOptimumData
    else:
        op = esM.componentModelingDict[esM.componentNames[componentName]]. \
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
            injectionWithdrawalRates.append(op.loc[ixIn].sum() - op.loc[ixOut].sum())
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
    # TODO add type and value correctness check @Juelich

    distances = esM.getComponent(componentName).distances.copy()
    indexMap = esM.getComponent(componentName)._mapC
    distances.index = [indexMap[ix] for ix in distances.index]

    return distances


def createNetwork(distances):
    """
    Creates undirected network/graph from given distances; updates distances such that
    either (u,v) or (v,u) are contained

    :param distances: pipeline distances in the length unit specified in the esM object
    :type distances: pandas series

    :return: graph of the network corresponding to the distances
    :rtype: graph object of networkx

    :return: pipeline distances in the length unit specified in the esM object
    :rtype: pandas series
    """
    # type and value check
    isPandasSeriesPositiveNumber(distances)
    for index in distances.index:
        if not isinstance(index, tuple):
            raise TypeError("Index of pandas series has to be a tuple")

    # first check if distances are consistent, i.e. if (u,v) and (v,u) are in distances they have to have the same
    # length and we will delete one of them
    # tmp list for reversed edges that we will be delete
    tmp_edges = []
    for edge in distances.index:
        if (edge[1], edge[0]) in distances.index and (edge[1], edge[0]) not in tmp_edges:
            assert (distances[edge] == distances[(edge[1], edge[0])])
            tmp_edges.append(edge)
    # delete tmp_edges because reversed edges are already contained and we consider an undirected graph
    distances = distances.drop(tmp_edges)

    # get edges for graph
    edges = distances.index
    # create empty graph
    G = nx.Graph()
    # create graph from given edges and add length as edge attribute
    for edge in edges:
        G.add_edge(edge[0], edge[1], length=distances[edge])
    return G, distances


def createMinimumSpanningTree(graph, distances):
    """
    Computes a minimal spanning tree with minimal sum of pipeline lengths;
    updates distances such that only arcs of the spanning tree are contained with corresponding length

    :param graph: an undirected networkx graph: Its edges have the attribute length which is the pipeline length in [m]
    :type graph: networkx graph object

    :param distances: pipeline distances in the length unit specified in the esM object
    :type distances: pandas series

    :return spanning tree with sum of lengths of pipelines is minimal
    :rtype: graph object of networkx
    """
    # type and value check
    isNetworkxGraph(graph)
    isPandasSeriesPositiveNumber(distances)

    # compute spanning tree with minimal sum of pipeline lengths
    S = nx.minimum_spanning_tree(graph, weight='length')
    # delete edges that are in graph but not in minimal spanning tree from the distance matrix
    edgesToDelete = []
    for edge in distances.index:
        # check if edge or its reversed edge are contained in minimal spanning tree
        # you have to check both directions because we have an undirected graph
        if edge not in S.edges and (edge[1], edge[0]) not in S.edges:
            edgesToDelete.append(edge)
    distances = distances.drop(edgesToDelete)

    return S, distances


def networkRefinement(distances, maxPipeLength, dic_node_minPress, dic_node_maxPress):
    """
    If a pipe is longer than maxPipeLength than it will be split into several pipes with equidistant length,
    i.e., replace arc (u,v) by (u,v_1), (v_1,v_2),..., (v_n,v) with n = ceil(lengthOfPipe/maxPipeLength) -1

    :param distances: pipeline distances in the length unit specified in the esM object
    :type distances: pandas series

    :param maxPipeLength: determines the maximal length of a pipe in [m].
    :type maxPipeLength: positive number

    :param dic_node_minPress: dictionary that contains for every node of the network its lower pressure bound in [bar]
    :type dic_node_minPress: dictionary: key: node of the network, value: non-negative float

    :param dic_node_maxPress: dictionary that contains for every node of the network its upper pressure bound in [bar]
    :type dic_node_maxPress: dictionary key: node of the network, value: non-negative float

    It holds dic_node_minPress[index] <= dic_node_maxPress[index]

    :return: graph of the network corresponding to the distances
    :rtype: graph object of networkx

    :return: pipeline distances in the length unit specified in the esM object
    :rtype: pandas series

    :return: dic_node_minPress dictionary that contains for every node of the network its lower pressure  bound in [bar]
    :rtype: dictionary key: node of the network, value: non-negative float

    :return dic_node_maxPress dictionary that contains for every node of the network its upper pressure bound in [bar]
    :rtype: dictionary key: node of the network, value: non-negative float
    """
    # type and value check
    isPandasSeriesPositiveNumber(distances)
    isDictionaryPositiveNumber(dic_node_minPress)
    isDictionaryPositiveNumber(dic_node_maxPress)
    checkLowerUpperBoundsOfDicts(dic_node_minPress, dic_node_maxPress)
    if maxPipeLength is not None:
        utils.isStrictlyPositiveNumber(maxPipeLength)

    # if maximal pipeline length is a positive number we apply the refinement
    if maxPipeLength is not None:
        # we have to check if pipes satisfy maximal pipeline length
        # list of new arcs that will be added
        newPipes = []
        # list of lengths of new added pipes
        newPipesLengths = []
        # list of split original pipes
        splitEdges = []
        for edge in distances.index:
            # get length of pipeline
            pipeLength = distances[edge]
            if pipeLength > maxPipeLength:
                # compute number of necessary artificial nodes
                nArtificialNodes = math.ceil(pipeLength / maxPipeLength) - 1
                # compute length of new pipelines
                newPipeLength = float(pipeLength / (math.ceil(pipeLength / maxPipeLength)))
                # lower and upper pressure bound for new nodes computed by average of nodes of original edge
                lowPress = (dic_node_minPress[edge[0]] + dic_node_minPress[edge[1]]) / 2
                maxPress = (dic_node_maxPress[edge[0]] + dic_node_maxPress[edge[1]]) / 2
                # add first new pipe and its length
                newPipes.append((edge[0], "v" + str(1) + "_" + str(edge[0]) + "_" + str(edge[1])))
                # add length of first new pipe
                newPipesLengths.append(newPipeLength)
                # add lower and upper bound for new artificial node
                dic_node_minPress["v" + str(1) + "_" + str(edge[0]) + "_" + str(edge[1])] = lowPress
                dic_node_maxPress["v" + str(1) + "_" + str(edge[0]) + "_" + str(edge[1])] = maxPress
                # add intermediate artificial pipes, its length, and lower/upper pressure bounds
                for index in range(1, nArtificialNodes):
                    newPipes.append(("v" + str(index) + "_" + str(edge[0]) + "_" + str(edge[1]),
                                     "v" + str(index + 1) + "_" + str(edge[0]) + "_" + str(edge[1])))
                    newPipesLengths.append(newPipeLength)
                    dic_node_minPress["v" + str(index + 1) + "_" + str(edge[0]) + "_" + str(edge[1])] = lowPress
                    dic_node_maxPress["v" + str(index + 1) + "_" + str(edge[0]) + "_" + str(edge[1])] = maxPress
                # add last new pipe and its length
                newPipes.append(("v" + str(nArtificialNodes) + "_" + str(edge[0]) + "_" + str(edge[1]),
                                 edge[1]))
                newPipesLengths.append(newPipeLength)
                # add edge to split edges
                splitEdges.append(edge)

        # Now delete edges that have been split
        distances = distances.drop(splitEdges)
        # Add new edges
        distances = distances.append(pd.Series(newPipesLengths, index=newPipes))

    # get edges for graph
    edges = distances.index
    # create empty graph
    G = nx.Graph()
    # create graph from given edges and add length as edge attribute
    for edge in edges:
        G.add_edge(edge[0], edge[1], length=distances[edge])

    return G, distances, dic_node_minPress, dic_node_maxPress


def generateRobustScenarios(injectionWithdrawalRates, graph, distances):
    """
    Compute for every node combination a special robust scenario according to Robinius et. al. (2019)
    and Labbé et. al. (2019)

    :param injectionWithdrawalRates: injection and withdrawal rates (withdrawals from the network are positive while
        injections are negative) for every time step and node; unit [kg/s]
    :type: pandas dataframe

    :param graph: an undirected networkx graph: Its edges have the attribute length which is the pipeline length in [m]
    :type graph: networkx graph object

    :param distances: pipeline distances in the length unit specified in the esM object
    :type distances: pandas series

    :return dictionary that contains for every node pair a dictionary containing all arc flows of the corresponding
    special scenario
    :rtype: dictionary key: (node1,node2), value: dictionary: key: arc, value: arc flow in [kg/s]

    :return list of entry node
    :rtype: list of strings

    :return list of exit node
    :rtype: list of strings
    """
    # Type and value checks
    isPandasDataFrameNumber(injectionWithdrawalRates)
    isNetworkxGraph(graph)
    isPandasSeriesPositiveNumber(distances)

    # get for every  entry/exit node the minimal and maximal injection rate and save it in a
    # dictionary: key: node, value: min Rate; respectively max Rate in [kg/s]
    # we note that inner nodes a handled separately in the computation of the special scenario
    dic_nodes_MinCapacity = {}
    dic_nodes_MaxCapacity = {}
    # list of entry nodes and exit nodes; note node can be in both for example storages
    entries = []
    exits = []
    for node in list(injectionWithdrawalRates.columns.values):
        minRate = injectionWithdrawalRates[node].min()
        maxRate = injectionWithdrawalRates[node].max()
        assert (minRate <= maxRate)
        dic_nodes_MinCapacity[node] = minRate
        dic_nodes_MaxCapacity[node] = maxRate
        # if minRate is negative, then node is an entry; if maxRate is positive, then node is an exit
        if minRate < 0.0:
            entries.append(node)
            if maxRate > 0.0:
                exits.append(node)
        elif maxRate > 0:
            exits.append(node)

    # compute special scenario for each node combination; see Paper Robinius et. al.(2019); Labbé et. al. (2019)
    # save arc flows of special scenarios for each node combination;
    # dictionary: key: node pair, value: dictionary: key: arc, value: arc flow
    dic_nodePair_flows = {}
    for startNode in graph.nodes:
        for endNode in graph.nodes:
            # only compute special scenario if start and endnode are different
            if startNode != endNode:
                dic_nodePair_flows[(startNode, endNode)] = computeSingleSpecialScenario(graph, distances, entries,
                                                                                        exits, startNode, endNode,
                                                                                        dic_nodes_MinCapacity,
                                                                                        dic_nodes_MaxCapacity)

    return dic_nodePair_flows, entries, exits


def computeSingleSpecialScenario(graph, distances, entries, exits, startNode, endNode, dic_nodes_MinCapacity,
                                 dic_nodes_MaxCapacity, specialScenario=True):
    """
    Compute special robust scenario for given node combination according to Robinius et. al. (2019)
    and Labbé et. al. (2019)

    :param graph: an undirected networkx graph: Its edges have the attribute length which is the pipeline length in [m]
    :type graph: networkx graph object

    :param distances: pipeline distances in the length unit specified in the esM object
    :type distances: pandas series

    :param entries: list of entry nodes of the network
    :type entries: list of strings

    :param exits: list of exit nodes of the network
    :type exits: list of strings

    :param startNode: node of the network (starting node of the special scenario)
    :type startNode: string

    :param endNode: node of the network (end node of special scenario)
    :type endNode: string

    :param dic_nodes_MinCapacity: dictionary containing minimal capacity for each node
    :type dic_nodes_MinCapacity: dictionary: key: node of the network, value: float

    :param dic_nodes_MaxCapacity: dictionary containing maximal capacity for each node
    :type dic_nodes_MaxCapacity: dictionary: key: node of the network, value: float

    :param specialScenario: bool: True if we compute special robust scenario; False if we compute scenario for fixed
    demand vector, e.g., for scenario of a time step
    :type specialScenario: bool

    :return dictionary that contains for every arc the corresponding arc flows of the (special) scenario
    :rtype: dictionary key: arc, value: arc flow
    """
    # Type and value check
    isNetworkxGraph(graph)
    isPandasSeriesPositiveNumber(distances)
    isListOfStrings(entries)
    isListOfStrings(exits)
    utils.isString(startNode)
    utils.isString(endNode)
    if isinstance(dic_nodes_MinCapacity, dict) and isinstance(dic_nodes_MaxCapacity, dict):
        if not (dic_nodes_MinCapacity.keys() == dic_nodes_MaxCapacity.keys()):
            raise TypeError("Dictionaries for min and max capacity need same keys")
        for node in dic_nodes_MinCapacity.keys():
            if not (isinstance(dic_nodes_MinCapacity[node], float) or isinstance(dic_nodes_MinCapacity[node], int)):
                raise TypeError("The input argument has to be an number")
            if not (isinstance(dic_nodes_MaxCapacity[node], float) or isinstance(dic_nodes_MaxCapacity[node], int)):
                raise TypeError("The input argument has to be an number")
            if dic_nodes_MaxCapacity[node] < dic_nodes_MinCapacity[node]:
                raise ValueError("minimal node capacity has to be equal or smaller than maximal node capacity")
    else:
        raise TypeError("dic_nodes_MinCapacity and dic_nodes_MinCapacity have to be dictionaries")
    isBool(specialScenario)

    # we build concrete Pyomo Model
    model = py.ConcreteModel()

    # Description model: we have a simple directed graph. We allow negative flows because a pipe can be used in both
    # directions by the flows
    model.Nodes = py.Set(initialize=graph.nodes)
    # important to use distances.keys() instead of graph.edges such that we do not have key errors later on because
    # the edges in graph are undirected and in distances.keys() directed
    model.Arcs = py.Set(initialize=distances.keys(), dimen=2)

    # create demand variables for every node;
    # if specialScenario is true, then we compute special scenario, i.e. entry/exit demand variables are bounded by
    # min(0,minimal_capacity) <= demandVariable <= max(0, maximal_capacity)
    # demand variables for inner nodes are set to zero
    # if specialScenario is false, the demand variable is just bounded by the minimal and maximal capacity
    if specialScenario:
        def demandCapacities(model, node):
            if node in entries or node in exits:
                return min(0, dic_nodes_MinCapacity[node]), max(0, dic_nodes_MaxCapacity[node])
            else:
                return 0, 0

        model.Demand = py.Var(model.Nodes, bounds=demandCapacities)
    else:
        # we do not compute special scenarios; we just compute flows for given, possibly fixed, demands
        def demandCapacities(model, node):
            return dic_nodes_MinCapacity[node], dic_nodes_MaxCapacity[node]

        model.Demand = py.Var(model.Nodes, bounds=demandCapacities)

    # create arc flow variables for every arc of the network
    model.Flow = py.Var(model.Arcs)

    # compute NodesOut, i.e., set of nodes that are connected to considered node by outgoing arc
    def nodes_out_init(model, node):
        retval = []
        for (i, j) in model.Arcs:
            if i == node:
                retval.append(j)
        return retval

    model.NodesOut = py.Set(model.Nodes, initialize=nodes_out_init)

    # compute NodesIn, i.e., set of nodes connected to considered node by ingoing arc
    def nodes_in_init(model, node):
        retval = []
        for (i, j) in model.Arcs:
            if j == node:
                retval.append(i)
        return retval

    model.NodesIn = py.Set(model.Nodes, initialize=nodes_in_init)

    # add flow balance constraints corresponding to the node demands
    def flow_balance_rule(model, node):
        return sum(model.Flow[i, node] for i in model.NodesIn[node]) \
               - sum(model.Flow[node, j] for j in model.NodesOut[node]) \
               == model.Demand[node]

    model.FlowBalance_cons = py.Constraint(model.Nodes, rule=flow_balance_rule)

    # compute unique flow-path P(startNode,endNode) from entry to exit; given by list of nodes of the path
    pathNodes = nx.shortest_path(graph, source=startNode, target=endNode)
    # non zero coefficients of objective function
    dic_arc_coef = {}
    # determine coefficients for objective function
    # if for an arc (u,v), u, respectively v, are not in pathNodes, then the coefficient is 0
    # if arc (u,v) of pathNodes satisfies P(startNode, u) subset P(startNode,v), then coefficient is 1, otherwise -1
    for index in range(0, len(pathNodes) - 1):
        # check which direction of the arc is contained in the graph
        if (pathNodes[index], pathNodes[index + 1]) in model.Arcs:
            dic_arc_coef[(pathNodes[index], pathNodes[index + 1])] = 1
        else:
            dic_arc_coef[(pathNodes[index + 1], pathNodes[index])] = -1

    # we set objective
    def obj_rule(model):
        return sum(dic_arc_coef[arc] * model.Flow[arc] for arc in dic_arc_coef.keys())

    model.Obj = py.Objective(rule=obj_rule, sense=py.maximize)

    # Create a solver
    opt = SolverFactory('gurobi')
    # Solve optimization model
    results = opt.solve(model)
    # status of solver
    status = results.solver.status
    # termination condition
    termCondition = results.solver.termination_condition

    # save the solution of the flows in a dictionary key: arcs, values: flow
    dic_scenario_flow = {}

    if status == SolverStatus.error or status == SolverStatus.aborted or status == SolverStatus.unknown:
        utils.output('Solver status:  ' + str(status) + ', termination condition:  ' + str(termCondition) +
                     '. No output is generated.', 0, 0)
    elif termCondition == TerminationCondition.infeasibleOrUnbounded or \
            termCondition == TerminationCondition.infeasible or \
            termCondition == TerminationCondition.unbounded:
        utils.output('Optimization problem is ' + str(termCondition) +
                     '. No output is generated.', 0, 0)
    else:
        # If the solver status is not okay (hence either has a warning, an error, was aborted or has an unknown
        # status), show a warning message.
        if not termCondition == TerminationCondition.optimal:
            warnings.warn('Output is generated for a non-optimal solution.')

        # dic_arcScenario has key (v,w,scenario) and value flow will be needed for MIP
        for arc in model.Arcs:
            dic_scenario_flow[arc] = model.Flow[arc].value

    return dic_scenario_flow


def computeLargeMergedDiameters(dic_subSetDiam_costs, nDigits=6):
    """
    Compute merged diameters, i.e. compute equivalent single diameter for two looped pipes.

    :param dic_subSetDiam_costs: dictionary containing diameters in [m] and costs in [Euro/m]
    :type: dictionary: key: diameter, value: costs

    :param nDigits: number of digits used in the round function
        |br| * the default value is 6
    :type nDigits: positive int

    :return dic_newDiam_costs: dictionary containing merged diameters in [m] and costs in [Euro/m]
    :rtype: dictionary: key: diameter, value: costs

    :return dic_newDiam_oldDiam: dictionary matching new diameters to old diameters
    :rtype: dictionary: key: new diameter, value: corresponding old diameter, which will be used in the looped pipe

    """
    # Type and value check
    if isinstance(dic_subSetDiam_costs, dict):
        for diam in dic_subSetDiam_costs.keys():
            utils.isStrictlyPositiveNumber(diam)
            utils.isStrictlyPositiveNumber(dic_subSetDiam_costs[diam])
    else:
        raise TypeError("The input has to be a dictionary")
    utils.isStrictlyPositiveInt(nDigits)

    dic_newDiam_costs = {}
    dic_newDiam_oldDiam = {}

    for diam in dic_subSetDiam_costs.keys():
        # compute new diameter in [m] and its costs in [Euro/m]
        # for Formula see (1) in Paper Reuß et. al.
        # since at current state we consider the diameter for a looped pipe the above is
        # equivalent to 2^(2/5) * diam and thus, we do not have to transform diam from [m] to [mm]
        newDiam = ((diam ** (5 / 2) + diam ** (5 / 2)) ** (2 / 5)).__round__(nDigits)
        # costs are two times costs of diam because newDiam represents two looped pipe with diameter diam
        newCosts = 2 * dic_subSetDiam_costs[diam]
        dic_newDiam_costs[newDiam] = newCosts
        dic_newDiam_oldDiam[newDiam] = diam

    return dic_newDiam_costs, dic_newDiam_oldDiam


def determinePressureDropCoef(dic_scenario_flows, distances, dic_node_minPress, dic_node_maxPress,
                              diameters, ir=0.2, rho_n=0.089882, T_m=20 + 273.15, T_n=273.15, p_n=1.01325,
                              Z_n=1.00062387922965, nDigits=6):
    """
    Compute for each scenario, diameter, and each arc the corresponding pressure drop

    :param dic_scenario_flows: dictionary that contains for every node pair a dictionary containing all
    arc flows in [kg/s] of the corresponding (special) scenario
    :type dic_scenario_flows: dictionary key: scenarioName (node1,node2), value: dictionary: key: arc, value: arc flow

    :param distances: pipeline distances in the length unit specified in the esM object ([m])
    :type distances: pandas series

    :param dic_node_minPress: dictionary that contains for every node of the network its lower pressure bound in [bar]
    :type dic_node_minPress: dictionary: key: node of the network, value: non-negative float

    :param dic_node_maxPress: dictionary that contains for every node of the network its upper pressure bound in [bar]
    :type dic_node_maxPress: dictionary key: node of the network, value: non-negative float

    It holds dic_node_minPress[index] <= dic_node_maxPress[index]

    :param diameters: list of diameters in [m]
    :type: list of strictly positive numbers

    :param ir: integral roughness of pipe in [mm]
        |br| * the default value is 0.2 (hydrogen, this value can also be used for methane)
    :type ir: positive float; optional

    :param rho_n: density at standard state in [kg/m^3]
        |br| * the default value is 0.089882 (hydrogen, you can use 0.71745877 for methane)
    :type rho_n: positive float; optional

    :param T_m: constant temperature in [kelvin]
        |br| * the default value is 20 + 273.15 (hydrogen, you can use 281.15 for methane)
    :type T_m: float; optional

    :param T_n: temperature in standard state in [kelvin]
        |br| * the default value is 273.15 (hydrogen, this value can also be used for methane)
    :type T_n: float; optional

    :param p_n: pressure at standard state in [bar]
        |br| * the default value is 1.01325 (hydrogen, this value can also be used for methane)
    :type p_n: non-negative float; optional

    :param Z_n: realgasfactor of hydrogen at standard state
        |br| * the default value is 1.00062387922965 (hydrogen, you can use 0.997612687740414 for methane)
    :type Z_n: non-negative float; optional

    :param nDigits: number of digits used in the round function
        |br| * the default value is 6
    :type nDigits: positive int; optional

    :return dictionary that contains for every scenario and diameter the corresponding pressure drops
    :rtype: dictionary key: (diameter, scenario Name), value: dic: key: arc, value: pressure drop
    """
    # check type and value
    if not isinstance(dic_scenario_flows, dict):
        raise TypeError("The input has to be a dictionary")
    isPandasSeriesPositiveNumber(distances)
    isDictionaryPositiveNumber(dic_node_minPress)
    isDictionaryPositiveNumber(dic_node_maxPress)
    checkLowerUpperBoundsOfDicts(dic_node_minPress, dic_node_maxPress)
    if isinstance(diameters, list):
        for diam in diameters:
            utils.isPositiveNumber(diam)
    else:
        raise TypeError("Diameters has to be a list")
    utils.isStrictlyPositiveNumber(ir)
    utils.isStrictlyPositiveNumber(rho_n)
    if not isinstance(T_m, float):
        raise TypeError("The input argument has to be an number")
    if not isinstance(T_n, float):
        raise TypeError("The input argument has to be an number")
    utils.isPositiveNumber(p_n)
    utils.isPositiveNumber(Z_n)
    utils.isStrictlyPositiveInt(nDigits)

    # compute for each diameter, scenario, and arc its pressure drop
    # save results in dic: key: (diameter, scenario Name), value: dic: key: arc, value: pressure drop
    dic_pressureDropCoef = {}
    for diameter in diameters:
        for nodePair in dic_scenario_flows.keys():
            # initialize dictionary
            dic_pressureDropCoef[(diameter, nodePair)] = {}
            # compute cross section of considered pipe and diameter
            tmpvalue_A = 0.25 * np.pi * diameter ** 2
            for arc in dic_scenario_flows[nodePair].keys():
                # check if flow is unequal to zero
                if dic_scenario_flows[nodePair][arc] != 0.0:
                    # Compute approximation of average pressure flow in pipe (u,v) by
                    # if flow((u,v)) is positive then set p_min to lower pressure bound of v and p_max to
                    # upper pressure bound u
                    # if flow((u,v)) is negative then set p_min to lower pressure bound of u and p_max to
                    # upper pressure bound v
                    if dic_scenario_flows[nodePair][arc] > 0:
                        p_min = dic_node_minPress[arc[1]]
                        p_max = dic_node_maxPress[arc[0]]
                    else:
                        p_min = dic_node_minPress[arc[0]]
                        p_max = dic_node_maxPress[arc[1]]
                    # compute approximation of average pressure
                    p_m = (2 / 3) * (p_max + p_min - (p_max * p_min) / (p_max + p_min))
                    # approximation for density
                    rho = 0.11922 * p_m ** 0.91192 - 0.17264
                    # approximation of the realgasfactor
                    Z_m = 5.04421 * 10 ** (-4) * p_m ** 1.03905 + 1.00050
                    K_m = Z_m / Z_n
                    # approximation of the dynamic viscosity
                    eta = 1.04298 * 10 ** (-10) * p_m ** 1.53560 + 8.79987 * 10 ** (-6)
                    nue = eta / rho
                    # compute velocity
                    tmpvalue_w = (abs(dic_scenario_flows[nodePair][arc]) / rho) / tmpvalue_A
                    # compute reynolds number
                    tmpvalue_Re = tmpvalue_w * (diameter / nue)
                    tmpvalue_alpha = np.exp(-np.exp(6.75 - 0.0025 * tmpvalue_Re))
                    tmpvalue_Lambda = (64 / tmpvalue_Re) * (1 - tmpvalue_alpha) + tmpvalue_alpha * (
                            -2 * np.log10(2.7 * (np.log10(tmpvalue_Re) ** 1.2 / tmpvalue_Re) + ir / (3.71 * 1000 *
                                                                                                     diameter))) ** (-2)
                    # note p_n is in [bar] instead of [PA], thus we divide tmpvalue_C by 10**5
                    # explanation: we have p_i^2-p_j^2=C. If p_i is in [PA] and we want p_i in [bar] then this leads to
                    # (p_i/10^5)^2-(p_j/10^5)^2=C/10^10
                    # but we changed p_n in computation C from [PA] to [bar] hence we only divide C by 10^5
                    tmpvalue_C_bar = tmpvalue_Lambda * 16 * rho_n * T_m * p_n * K_m / (np.pi ** 2 * T_n * 10 ** 5)
                    # compute final pressure drop coefficient depending on the flow
                    tmp_value_C_coef = (distances[arc] / rho_n ** 2) * \
                                       (tmpvalue_C_bar * dic_scenario_flows[nodePair][arc] *
                                        abs(dic_scenario_flows[nodePair][arc]) / diameter ** 5)
                    # save pressure drop for considered diameter, scenario, and arc
                    dic_pressureDropCoef[(diameter, nodePair)][arc] = tmp_value_C_coef
                else:
                    dic_pressureDropCoef[(diameter, nodePair)][arc] = 0

    return dic_pressureDropCoef


def determineOptimalDiscretePipelineSelection(graph, distances, dic_pressureDropCoef, specialScenarioNames,
                                              dic_node_minPress, dic_node_maxPress, dic_diam_costs, robust=True):
    """
    Model of optimal pipeline sizing (diameter selection) w.r.t. to the given scenarios

    :param graph: an undirected networkx graph: Its edges have the attribute length which is the pipeline length in [m]
    :type graph: networkx graph object

    :param distances: pipeline distances in the length unit specified in the esM object ([m])
    :type distances: pandas series

    :param dic_pressureDropCoef: dictionary that contains for every scenario and diameter the
     corresponding pressure drops in [bar]
    :type  dic_pressureDropCoef: dictionary: keys: scenarioName; value: dict: key: arc, value: pressure drop in [bar]

    :param specialScenarioNames: list of names of scenarios. In robust case tuples (startNode, endNode).
    :type specialScenarioNames: list of tuples in the robust case, otherwise list of time Steps

    :param dic_node_minPress: dictionary that contains for every node of the network its lower pressure bound in [bar]
    :type dic_node_minPress: dictionary: key: node of the network, value: non-negative float

    :param dic_node_maxPress: dictionary that contains for every node of the network its upper pressure bound in [bar]
    :type dic_node_maxPress: dictionary key: node of the network, value: non-negative float

    It holds dic_node_minPress[index] <= dic_node_maxPress[index]

    :param dic_diam_costs: dictionary that contains for every diameter in [m] its costs [Euro/m]
    :type dic_diam_costs: dictionary key: diameter, value: non-negative float

    :param robust: Bool that is true, if we optimize w.r.t. robust scenarios, otherwise False.
    :type robust: bool

    :return dictionary that contains for every arc the optimal diameter in [m]
    :rtype dictionary: key: arc, value: optimal diameter

    :return dictionary that contains for every scenario the corresponding pressure levels
    :rtype dictionary: key: scenarioName, value: dict: key: node, value: pressure level of node
    """
    # type and value checks
    isNetworkxGraph(graph)
    isPandasSeriesPositiveNumber(distances)
    if not isinstance(dic_pressureDropCoef, dict):
        raise TypeError("The input has to be a dictionary")

    if isinstance(specialScenarioNames, list):
        if robust:
            for scenario in specialScenarioNames:
                isinstance(scenario, tuple)
    else:
        raise TypeError("The input argument has to be a list")
    isDictionaryPositiveNumber(dic_node_minPress)
    isDictionaryPositiveNumber(dic_node_maxPress)
    checkLowerUpperBoundsOfDicts(dic_node_minPress, dic_node_maxPress)
    if isinstance(dic_diam_costs, dict):
        for diam in dic_diam_costs.keys():
            utils.isStrictlyPositiveNumber(diam)
            utils.isStrictlyPositiveNumber(dic_diam_costs[diam])
    else:
        raise TypeError("The input has to be a dictionary")
    if not isinstance(robust, bool):
        raise TypeError("The input has to be a bool")

    # set list of available diameters
    diameters = dic_diam_costs.keys()

    # build concrete pyomo model
    model = py.ConcreteModel()

    # sets for nodes, arcs, diameters, scenarios
    model.nodes = py.Set(initialize=graph.nodes)
    model.arcs = py.Set(initialize=list(distances.keys()), dimen=2)
    # diameters assuming that each pipe has the same diameter options
    model.diameters = py.Set(initialize=diameters)
    # if we have special scenarios, scenario names are tuples, otherwise not
    if robust:
        # set indices for each scenario by its nodePair = (startnode, endnode)
        model.scenarios = py.Set(initialize=specialScenarioNames, dimen=2)
    else:
        # set indices for each timeStep number
        model.scenarios = py.Set(initialize=specialScenarioNames, dimen=1)

    # create variables binaries x are the same for each scenario
    # pressure variables are different for each scenario
    model.x = py.Var(model.arcs, model.diameters, domain=py.Binary)
    if robust:
        def pressureBounds(model, node, startnode, endnode):
            return dic_node_minPress[node] ** 2, dic_node_maxPress[node] ** 2

        model.pi = py.Var(model.nodes, model.scenarios, bounds=pressureBounds)
    else:
        def pressureBounds(model, node, timeStep):
            return dic_node_minPress[node] ** 2, dic_node_maxPress[node] ** 2

        model.pi = py.Var(model.nodes, model.scenarios, bounds=pressureBounds)

    # objective: minimize the costs
    def obj_rule(model):
        return sum(
            sum(dic_diam_costs[diam] * distances[arc] * model.x[arc, diam] for diam in model.diameters)
            for arc in model.arcs)

    model.Obj = py.Objective(rule=obj_rule)

    # pressure drop for each cons and each scenario
    if robust:
        def pressure_drop(model, arc0, arc1, scenarioStart, scenarioEnd):
            return model.pi[arc1, (scenarioStart, scenarioEnd)] - model.pi[arc0, (scenarioStart, scenarioEnd)] == \
                   -sum(dic_pressureDropCoef[(diam, (scenarioStart, scenarioEnd))][(arc0, arc1)] *
                        model.x[arc0, arc1, diam] for diam in model.diameters)

        model.PressureDrop_cons = py.Constraint(model.arcs, model.scenarios, rule=pressure_drop)
    else:
        def pressure_dropNotRobust(model, arc0, arc1, timeStep):
            return model.pi[arc1, timeStep] - model.pi[arc0, timeStep] == \
                   -sum(dic_pressureDropCoef[(diam, timeStep)][(arc0, arc1)] *
                        model.x[arc0, arc1, diam] for diam in model.diameters)

        model.PressureDrop_cons = py.Constraint(model.arcs, model.scenarios, rule=pressure_dropNotRobust)

    # ensure that a single diameter per arc is chosen
    def selection_diameter(model, arc0, arc1):
        return sum(model.x[arc0, arc1, diam] for diam in model.diameters) == 1

    model.SelectionDiameter_cons = py.Constraint(model.arcs, rule=selection_diameter)

    # Create a solver
    opt = SolverFactory('gurobi')
    results = opt.solve(model, tee=True, report_timing=True, keepfiles=False)
    # status of solver
    status = results.solver.status
    # termination condition
    termCondition = results.solver.termination_condition
    # write diameter solution to dictionary: key: arc, value: optimal diameter
    # write pressure solutions to dictionary; key: scenarioName, value: dict: key: node, value: pressure level in [bar]
    dic_arc_diam = {}
    dic_scen_node_press = {}

    if status == SolverStatus.error or status == SolverStatus.aborted or status == SolverStatus.unknown:
        utils.output('Solver status:  ' + str(status) + ', termination condition:  ' + str(termCondition) +
                     '. No output is generated.', 0, 0)
    elif termCondition == TerminationCondition.infeasibleOrUnbounded or \
            termCondition == TerminationCondition.infeasible or \
            termCondition == TerminationCondition.unbounded:
        utils.output('Optimization problem is ' + str(termCondition) +
                     '. No output is generated.', 0, 0)
    else:
        # If the solver status is not okay (hence either has a warning, an error, was aborted or has an unknown
        # status), show a warning message.
        if not termCondition == TerminationCondition.optimal:
            warnings.warn('Output is generated for a non-optimal solution.')

        # initialize dict with empty dict
        for scenario in specialScenarioNames:
            dic_scen_node_press[scenario] = {}

        for v in model.component_objects(py.Var, active=True):
            varobject = getattr(model, str(v))
            for index in varobject:
                # round because sometimes we are nearly one
                if str(varobject) == 'x' and round(varobject[index].value) == 1:
                    dic_arc_diam.update({(index[0], index[1]): index[2]})
                elif str(varobject) == 'pi':
                    if robust:
                        # need sqrt() because in model pressure is quadratic because of the transformation
                        dic_scen_node_press[(index[1], index[2])].update({index[0]: np.sqrt(varobject[index].value)})
                    else:
                        # need sqrt() because in model pressure is quadratic because of the transformation
                        dic_scen_node_press[(index[1])].update({index[0]: np.sqrt(varobject[index].value)})

    return dic_arc_diam, dic_scen_node_press


def postprocessing(graph, distances, dic_arc_diam, dic_scenario_flows, dic_node_minPress, dic_node_maxPress):
    """"
    Compute "more" accurate pressure levels for the considered scenarios in the network with optimal diameters
    Apply postprocessing of Master's thesis with adaption that we possibly consider every node for fixing its
    pressure level to the upper pressure bound.

    :param graph: an undirected networkx graph: Its edges have the attribute length which is the pipeline length in [m]
    :type graph: networkx graph object

    :param distances: pipeline distances in the length unit specified in the esM object ([m])
    :type distances: pandas series

    :param dic_arc_diam: dictionary containing for each arc the optimal diameter in [m]
    :type: dictionary: key: arc, value: optimal diameter

    :param dic_scenario_flows: dictionary that contains for every node pair a dictionary containing all
    arc flows in [kg/s] of the corresponding (special) scenario
    :type dic_scenario_flows: dictionary key: scenarioName (node1,node2), value: dictionary: key: arc, value: arc flow

    :param dic_node_minPress: dictionary that contains for every node of the network its lower pressure bound in [bar]
    :type dic_node_minPress: dictionary: key: node of the network, value: non-negative float

    :param dic_node_maxPress: dictionary that contains for every node of the network its upper pressure bound in [bar]
    :type dic_node_maxPress: dictionary key: node of the network, value: non-negative float

    It holds dic_node_minPress[index] <= dic_node_maxPress[index]

    :return dictionary that contains for every scenario the corresponding pressure levels in [bar]
    :rtype: dictionary key: scenarioName, value: dic: key: arc, value pressure level

    :return dictionary that contains for every scenario the maximal pressure bound violation in [bar]
    :rtype: dictionary key: scenarioName, value: float = maximal pressure bound violation
    """
    # Type and value check
    isNetworkxGraph(graph)
    isPandasSeriesPositiveNumber(distances)
    if not isinstance(dic_scenario_flows, dict):
        raise TypeError("The input has to be a dictionary")
    if isinstance(dic_arc_diam, dict):
        for diam in dic_arc_diam.keys():
            utils.isStrictlyPositiveNumber(dic_arc_diam[diam])
    else:
        raise TypeError("The input has to be a dictionary")
    isDictionaryPositiveNumber(dic_node_minPress)
    isDictionaryPositiveNumber(dic_node_maxPress)
    checkLowerUpperBoundsOfDicts(dic_node_minPress, dic_node_maxPress)

    # best found pressure levels for scenarios; dic key: scenario, value: dic: key: node, value: pressure level in [bar]
    dic_scen_PressLevel = {}
    # maximal violation of pressure bounds; zero if no violation exists; dic: key: scenario, value: pressure violation
    dic_scen_MaxViolPress = {}
    # we compute "precise" pressure levels for every scenarios
    for scenario in dic_scenario_flows.keys():
        dic_scen_PressLevel[scenario] = {}
        dic_scen_MaxViolPress[scenario] = math.inf
        # copy a list of nodes
        tmp_nodes = copy.deepcopy(list(graph.nodes))
        # we now set iteratively the pressure level of a single node to its upper pressure bound and then compute the
        # unique pressure levels until we find valid pressure levels or have tested all nodes
        while tmp_nodes:
            # we have not found valid pressure levels for this scenario
            # temporary pressure levels
            dic_tmp_pressure = {}
            for node in list(graph.nodes):
                dic_tmp_pressure[node] = None
            # choose the node which pressure level is fixed to the upper pressure bound
            current_node = tmp_nodes[0]
            validation, tmp_viol = computePressureAtNode(True, current_node, current_node, graph,
                                                         dic_arc_diam, distances, dic_scenario_flows[scenario],
                                                         dic_node_minPress, dic_node_maxPress, 0, dic_tmp_pressure)
            # if validation true, then we have feasible pressure levels; empty list of nodes that have to be
            # considered
            if validation:
                tmp_nodes = []
                # we have feasible pressure level and save them
                dic_scen_PressLevel[scenario] = dic_tmp_pressure
                dic_scen_MaxViolPress[scenario] = tmp_viol
            else:
                # remove considered entry from list of nodes that will be considered for fixing the pressure level
                tmp_nodes.remove(tmp_nodes[0])
                # we update the maximal pressure level violation
                if tmp_viol < dic_scen_MaxViolPress[scenario]:
                    # save currently best pressure levels
                    dic_scen_PressLevel[scenario] = copy.deepcopy(dic_tmp_pressure)
                    dic_scen_MaxViolPress[scenario] = tmp_viol

    return dic_scen_PressLevel, dic_scen_MaxViolPress


def computePressureAtNode(validation, node, nodeUpperBound, graph, dic_arc_diam, distances, dic_scenario_flows,
                          dic_node_minPress, dic_node_maxPress, tmp_violation, dic_node_pressure,
                          ir=0.2, rho_n=0.089882, T_m=20 + 273.15, T_n=273.15, p_n=1.01325,
                          Z_n=1.00062387922965, nDigits=6):
    """"
    Compute pressure levels recursive for given scenario and node that is fixed to its upper pressure level

    :param validation: boolean that is False, if the computed pressure levels are infeasible
    :rtype validation: bool

    :param node: node of the network for which we currently consider for computing the pressure levels
    :type node: str

    :param nodeUpperBound: node which pressure level is fixed to the upper bound
    :type node: str

    :param graph: an undirected networkx graph: Its edges have the attribute length which is the pipeline length in [m]
    :type graph: networkx graph object

    :param dic_arc_diam: dictionary containing for each arc the optimal diameter in [m]
    :type: dictionary: key: arc, value: optimal diameter

    :param distances: pipeline distances in the length unit specified in the esM object ([m])
    :type distances: pandas series

    :param dic_scenario_flows: dictionary scenario and corresponding flows in [kg/s]
    :type: dictionary: key: arc, value: arc flow

    :param dic_node_minPress: dictionary that contains for every node of the network its lower pressure bound in [bar]
    :type dic_node_minPress: dictionary: key: node of the network, value: non-negative float

    :param dic_node_maxPress: dictionary that contains for every node of the network its upper pressure bound in [bar]
    :type dic_node_maxPress: dictionary key: node of the network, value: non-negative float

    It holds dic_node_minPress[index] <= dic_node_maxPress[index]

    :param tmp_violation: violation of the current pressure bounds in [bar]
    :type tmp_violation: float

    :param dic_node_pressure: dictionary that contains node pressure levels in [bar]
    :type dic_node_pressure: dictionary key: node of the network, value: non-negative float


    :param ir: integral roughness of pipe in [mm]
        |br| * the default value is 0.2 (hydrogen, this value can also be used for methane)
    :type ir: positive float

    :param rho_n: density at standard state in [kg/m^3]
        |br| * the default value is 0.089882 (hydrogen, you can use 0.71745877 for methane)
    :type rho_n: positive float

    :param T_m: constant temperature in [kelvin]
        |br| * the default value is 20 + 273.15 (hydrogen, you can use 281.15 for methane)
    :type T_m: float

    :param T_n: temperature in standard state in [kelvin]
        |br| * the default value is 273.15 (hydrogen, this value can also be used for methane)
    :type T_n: float

    :param p_n: pressure at standard state in [bar]
        |br| * the default value is 1.01325 (hydrogen, this value can also be used for methane)
    :type p_n: non-negative float

    :param Z_n: realgasfactor of hydrogen at standard state
        |br| * the default value is 1.00062387922965 (hydrogen, you can use 0.997612687740414 for methane)
    :type Z_n: non-negative float

    :param nDigits: number of digits used in the pandas round function. Is applied to the
        specified or determined injection and withdrawal rates.
        |br| * the default value is 6
    :type nDigits: positive int

    :return validation: boolean that is true, if the computed pressure levels are feasible
    :rtype: bool

    :return maximal violation of the pressure bounds w.r.t. the computed pressure levels in [bar]
    :rtype: float
    """
    # Type and value check
    isBool(validation)
    utils.isString(node)
    utils.isString(nodeUpperBound)
    isNetworkxGraph(graph)
    isPandasSeriesPositiveNumber(distances)
    if not isinstance(dic_scenario_flows, dict):
        raise TypeError("The input has to be a dictionary")
    if isinstance(dic_arc_diam, dict):
        for diam in dic_arc_diam.keys():
            utils.isStrictlyPositiveNumber(dic_arc_diam[diam])
    else:
        raise TypeError("The input has to be a dictionary")
    isDictionaryPositiveNumber(dic_node_minPress)
    isDictionaryPositiveNumber(dic_node_maxPress)
    checkLowerUpperBoundsOfDicts(dic_node_minPress, dic_node_maxPress)
    utils.isPositiveNumber(tmp_violation)
    if not isinstance(dic_node_pressure, dict):
        raise TypeError("The Input has to a dictionary")
    utils.isStrictlyPositiveNumber(ir)
    utils.isStrictlyPositiveNumber(rho_n)

    if not isinstance(T_m, float):
        raise TypeError("The input argument has to be an number")

    if not isinstance(T_n, float):
        raise TypeError("The input argument has to be an number")
    utils.isPositiveNumber(p_n)
    utils.isPositiveNumber(Z_n)
    utils.isStrictlyPositiveInt(nDigits)

    # if node is equal to nodeUpperBound, we fix its pressure level to the upper bound; base case in recursion
    if node == nodeUpperBound:
        dic_node_pressure[node] = dic_node_maxPress[node]
    # list of arcs
    arcs = list(distances.keys())
    # we now compute the neighbours of the considered node
    neighbors = graph.neighbors(node)
    # compute pressure levels for neighbor nodes
    for neighbor in neighbors:
        # check if pressure is already computed
        if dic_node_pressure[neighbor] is None:
            # check if (node,neighbor) or (neighbor,node) is in graph
            if (node, neighbor) in arcs:
                # check flow direction for arc (node,neighbor)
                if dic_scenario_flows[(node, neighbor)] >= 0.0:
                    # we know pressure level of beginning node of arc; compute pressure level for end node of arc
                    dic_node_pressure[neighbor] = computePressureEndnodeArc((node, neighbor), dic_node_pressure[node],
                                                                            dic_scenario_flows, dic_arc_diam, distances,
                                                                            ir, rho_n, T_m, T_n, p_n, Z_n)
                else:
                    # we know pressure level of endnode
                    dic_node_pressure[neighbor] = computePressureStartnodeArc((node, neighbor), dic_node_pressure[node],
                                                                              dic_scenario_flows, dic_arc_diam,
                                                                              distances,
                                                                              ir, rho_n, T_m, T_n, p_n, Z_n,
                                                                              tol=10 ** (- nDigits))
            else:
                # we know that arc (neighbor,node) is contained in the graph
                # check flow direction
                if dic_scenario_flows[(neighbor, node)] <= 0.0:
                    # we know pressure of start node
                    dic_node_pressure[neighbor] = computePressureEndnodeArc((neighbor, node), dic_node_pressure[node],
                                                                            dic_scenario_flows, dic_arc_diam, distances,
                                                                            ir, rho_n, T_m, T_n, p_n, Z_n)
                else:
                    # we know pressure level of end node
                    dic_node_pressure[neighbor] = computePressureStartnodeArc((neighbor, node), dic_node_pressure[node],
                                                                              dic_scenario_flows, dic_arc_diam,
                                                                              distances,
                                                                              ir, rho_n, T_m, T_n, p_n, Z_n,
                                                                              tol=10 ** (- nDigits))
            # check if new computed pressure level is feasible
            if dic_node_pressure[neighbor] == - math.inf:
                # pressure violation is really high
                tmp_violation = math.inf
                return False, tmp_violation
            # check if we violate pressure bounds for neighbor node
            if dic_node_pressure[neighbor] < dic_node_minPress[neighbor] \
                    or dic_node_pressure[neighbor] > dic_node_maxPress[neighbor]:
                # pressure level is not valid
                validation = False
                # update pressure bound violation
                if dic_node_pressure[neighbor] < dic_node_minPress[neighbor]:
                    # update violation and violation node if it is bigger
                    if tmp_violation is None or \
                            abs(dic_node_minPress[neighbor] - dic_node_pressure[neighbor]) > tmp_violation:
                        tmp_violation = abs(dic_node_minPress[neighbor] - dic_node_pressure[neighbor])
                else:
                    if tmp_violation is None or \
                            abs(dic_node_pressure[neighbor] - dic_node_maxPress[neighbor]) > tmp_violation:
                        tmp_violation = abs(dic_node_pressure[neighbor] - dic_node_maxPress[neighbor])

            # compute value for neighbour of tmp
            validation, tmp_violation = computePressureAtNode(validation, neighbor, nodeUpperBound, graph, dic_arc_diam,
                                                              distances,
                                                              dic_scenario_flows, dic_node_minPress, dic_node_maxPress,
                                                              tmp_violation, dic_node_pressure)

    return validation, tmp_violation


def computePressureStartnodeArc(arc, pressureEndNode, dic_scenario_flows, dic_arc_diam, distances, ir=0.2,
                                rho_n=0.089882, T_m=20 + 273.15, T_n=273.15, p_n=1.01325,
                                Z_n=1.00062387922965, tol=10 ** (-4)):
    """"
    For given arc and pressure level of endNode compute the pressure of the startNode by solving the corresponding
    equation system

    :param arc: arc of the network for which we know the pressure at the endNode, i.e. the node which receives gas
    :type arc: tuple

    :param pressureEndNode: pressure level of endNode
    :type pressureEndNode: non-negative float

    :param dic_scenario_flows: dictionary scenario and corresponding flows in [kg/s]; note arc flow of arc has to be
    positive
    :type: dictionary: key: arc, value: arc flow

    :param dic_arc_diam: dictionary containing for each arc the optimal diameter in [m]
    :type: dictionary: key: arc, value: optimal diameter

    :param distances: pipeline distances in the length unit specified in the esM object ([m])
    :type distances: pandas series

    :param ir: integral roughness of pipe in [mm]
        |br| * the default value is 0.2 (hydrogen, this value can also be used for methane)
    :type ir: positive float

    :param rho_n: density at standard state in [kg/m^3]
        |br| * the default value is 0.089882 (hydrogen, you can use 0.71745877 for methane)
    :type rho_n: positive float

    :param T_m: constant temperature in [kelvin]
        |br| * the default value is 20 + 273.15 (hydrogen, you can use 281.15 for methane)
    :type T_m: float

    :param T_n: temperature in standard state in [kelvin]
        |br| * the default value is 273.15 (hydrogen, this value can also be used for methane)
    :type T_n: float

    :param p_n: pressure at standard state in [bar]
        |br| * the default value is 1.01325 (hydrogen, this value can also be used for methane)
    :type p_n: non-negative float

    :param Z_n: realgasfactor of hydrogen at standard state
        |br| * the default value is 1.00062387922965 (hydrogen, you can use 0.997612687740414 for methane)
    :type Z_n: non-negative float

    :param tol: tolerance to which accuracy we solve the equation system
        |br| * the default value is 10^-4
    :type tol: non-negative float

    :return: pressure level of startNode in [bar]
    :rtype: float
    """
    # Type and Value check
    if not isinstance(arc, tuple):
        raise TypeError("The input has to be a tuple")
    utils.isStrictlyPositiveNumber(pressureEndNode)
    if not isinstance(dic_scenario_flows, dict):
        raise TypeError("The input has to be a dictionary")
    if isinstance(dic_arc_diam, dict):
        for diam in dic_arc_diam.keys():
            utils.isStrictlyPositiveNumber(dic_arc_diam[diam])
    isPandasSeriesPositiveNumber(distances)
    utils.isStrictlyPositiveNumber(ir)
    utils.isStrictlyPositiveNumber(rho_n)
    if not isinstance(T_m, float):
        raise TypeError("The input argument has to be an number")
    if not isinstance(T_n, float):
        raise TypeError("The input argument has to be an number")
    utils.isPositiveNumber(p_n)
    utils.isPositiveNumber(Z_n)
    utils.isStrictlyPositiveNumber(tol)

    if dic_scenario_flows[arc] == 0.0:
        return pressureEndNode

    # define function of nonlinear equation system f(x) = pressure_start^2-pressure_end^2-C
    # because then root is our valid pressure level solution, because we know pressure_end

    def f(pressure_start):
        d = dic_arc_diam[arc]
        A = 0.25 * math.pi * d ** 2
        rho_in = 0.11922 * pressure_start ** 0.91192 - 0.17264
        V_in = abs(dic_scenario_flows[arc]) / rho_in
        w_in = V_in / A
        eta_in = 1.04298 * 10 ** (-10) * pressure_start ** 1.53560 + 8.79987 * 10 ** (-6)
        nue_in = eta_in / rho_in
        Re_in = w_in * (d / nue_in)
        alpha = math.exp(-math.exp(6.75 - 0.0025 * Re_in))
        Lambda = (64 / Re_in) * (1 - alpha) + alpha * (-2 * math.log10(
            (2.7 * (math.log10(Re_in)) ** 1.2) / Re_in +
            ir / (3.71 * 1000 * d))) ** (-2)
        C_tilde = (Lambda * distances[arc] * rho_in * w_in ** 2) / (2 * d)
        # note pressure_start is in bar
        p_m = pressure_start - C_tilde / 10 ** 5
        if p_m < 0.0:
            # pressure drop too large no valid pressure assignment possible
            return -math.inf
        Z_m = 5.04421 * 10 ** (-4) * p_m ** 1.03905 + 1.00050
        K_m = Z_m / Z_n
        # note flow direction is given by startnode endnode so we square the arcflow
        C = (Lambda * 16 * distances[arc] * T_m * p_n * K_m) / (
                math.pi ** 2 * T_n * rho_n * 10 ** 5 * dic_arc_diam[arc] ** 5) * dic_scenario_flows[arc] ** 2
        return pressure_start ** 2 - pressureEndNode ** 2 - C

    # find root of f, start value pressure_end + 0.5(bar)
    # x = fsolve(f, pressureEndNode + 0.5)
    # pressureEndnode + guess for solution depending on flow; you can replace this guess by the approximation of the
    # pressure drop of the MIP to probably achieve better results
    x = fsolve(f, pressureEndNode + 0.5 * (dic_scenario_flows[arc] ** 2) / (dic_arc_diam[arc] ** 5))
    # check if tolerance is ok
    assert isinstance(tol, float)
    # check tolerance of first solution
    if f(x[0]) <= tol:
        # value is ok
        # because x is an array return first entry, we only have one solution for the nonlinear equation system
        return x[0]
    else:
        print('nonlinear equation system failed')
        # this warning means we could not solve the system, this could be the case if the pressure drop is too large
        # or when the start value for the nonlinear equation solver is too far away from the solution
        print("Nonlinear equation system in Postprocessing failed. Try another node which pressure level is"
              " set to the upper bound")
        return -math.inf


def computePressureEndnodeArc(arc, pressureStartNode, dic_scenario_flows, dic_arc_diam, distances,
                              ir=0.2, rho_n=0.089882, T_m=20 + 273.15, T_n=273.15, p_n=1.01325,
                              Z_n=1.00062387922965):
    """"
    For given arc and pressure level of startNode compute the pressure of the endNode

    :param arc: arc of the network for which we know the pressure at the endNode, i.e. the node which receives gas
    :type arc: tuple

    :param pressureStartNode: pressure level of endNode
    :type pressureStartNode: non-negative float

    :param dic_scenario_flows: dictionary scenario and corresponding flows in [kg/s]
    :type: dictionary: key: arc, value: arc flow

    :param dic_arc_diam: dictionary containing for each arc the optimal diameter in [m]
    :type: dictionary: key: arc, value: optimal diameter

    :param distances: pipeline distances in the length unit specified in the esM object ([m])
    :type distances: pandas series

    :param ir: integral roughness of pipe in [mm]
        |br| * the default value is 0.2 (hydrogen, this value can also be used for methane)
    :type ir: positive float

    :param rho_n: density at standard state in [kg/m^3]
        |br| * the default value is 0.089882 (hydrogen, you can use 0.71745877 for methane)
    :type rho_n: positive float

    :param T_m: constant temperature in [kelvin]
        |br| * the default value is 20 + 273.15 (hydrogen, you can use 281.15 for methane)
    :type T_m: float

    :param T_n: temperature in standard state in [kelvin]
        |br| * the default value is 273.15 (hydrogen, this value can also be used for methane)
    :type T_n: float

    :param p_n: pressure at standard state in [bar]
        |br| * the default value is 1.01325 (hydrogen, this value can also be used for methane)
    :type p_n: non-negative float

    :param Z_n: realgasfactor of hydrogen at standard state
        |br| * the default value is 1.00062387922965 (hydrogen, you can use 0.997612687740414 for methane)
    :type Z_n: non-negative float

    :return: pressure level of endNode in [bar]
    :rtype: float
    """
    # Type and Value check
    if not isinstance(arc, tuple):
        raise TypeError("The input has to be a tuple")
    utils.isStrictlyPositiveNumber(pressureStartNode)
    if not isinstance(dic_scenario_flows, dict):
        raise TypeError("The input has to be a dictionary")
    if isinstance(dic_arc_diam, dict):
        for diam in dic_arc_diam.keys():
            utils.isStrictlyPositiveNumber(dic_arc_diam[diam])
    isPandasSeriesPositiveNumber(distances)
    utils.isStrictlyPositiveNumber(ir)
    utils.isStrictlyPositiveNumber(rho_n)
    if not isinstance(T_m, float):
        raise TypeError("The input argument has to be an number")
    if not isinstance(T_n, float):
        raise TypeError("The input argument has to be an number")
    utils.isPositiveNumber(p_n)
    utils.isPositiveNumber(Z_n)

    arcFlow = dic_scenario_flows[arc]
    if arcFlow != 0:
        d = dic_arc_diam[arc]
        A = 0.25 * math.pi * d ** 2
        rho_in = 0.11922 * pressureStartNode ** 0.91192 - 0.17264
        V_in = abs(arcFlow) / rho_in
        w_in = V_in / A
        eta_in = 1.04298 * 10 ** (-10) * pressureStartNode ** 1.53560 + 8.79987 * 10 ** (-6)
        nue_in = eta_in / rho_in
        Re_in = w_in * (d / nue_in)
        alpha = math.exp(-math.exp(6.75 - 0.0025 * Re_in))
        Lambda = (64 / Re_in) * (1 - alpha) + alpha * (-2 * math.log10(
            (2.7 * (math.log10(Re_in)) ** 1.2) / Re_in +
            ir / (3.71 * 1000 * d))) ** (-2)
        C_tilde = (Lambda * distances[arc] * rho_in * w_in ** 2) / (2 * d)
        # note pressure_start is in bar
        p_m = pressureStartNode - C_tilde / 10 ** 5
        if p_m < 0.0:
            # pressure drop too large no valid pressure assignment possible
            return -math.inf
        Z_m = 5.04421 * 10 ** (-4) * p_m ** 1.03905 + 1.00050
        K_m = Z_m / Z_n
        # note flow direction is given by startnode endnode so we square the arcflow
        C = (Lambda * 16 * distances[arc] * T_m * p_n * K_m) / (math.pi ** 2 * T_n * rho_n * 10 ** 5 *
                                                                dic_arc_diam[arc] ** 5) * arcFlow ** 2
    else:
        # flow is zero therefore pressure drop is zero
        C = 0

    if pressureStartNode ** 2 - C >= 0:
        return math.sqrt(pressureStartNode ** 2 - C)
    else:
        # pressure drop is too big return negative value, which is a invalid pressure value
        return -math.inf


def computeTimeStepFlows(injectionWithdrawalRates, distances, graph, entries, exits):
    """"
    Compute for each timeStep and demands given by injectionWithdrawalRates the corresponding flow values

    :param: injectionWithdrawalRates: injection and withdrawal rates (withdrawals from the network are positive while
        injections are negative) in [kg^3/s]
    :type injectionWithdrawalRates: pandas DataFrame

    :param distances: pipeline distances in the length unit specified in the esM object ([m])
    :type distances: pandas series

    :param graph: an undirected networkx graph: Its edges have the attribute length which is the pipeline length in [m]
    :type graph: networkx graph object

    :param entries: list of entry nodes of the network
    :type entries: list of str

    :param exits: list of exit nodes of the network
    :type exits: list of str

    :return dictionary that contains for every time step the corresponding flows in [kg/s]
    :rtype : dictionary key: timeStep, value: dict: key: arc, value: arc flow
    """
    # Type and value check
    isPandasDataFrameNumber(injectionWithdrawalRates)
    isPandasSeriesPositiveNumber(distances)
    isNetworkxGraph(graph)
    isListOfStrings(entries)
    isListOfStrings(exits)

    # compute for every time step the corresponding flows; dict: key: timeStep, value: dict: key: arc, value: flow
    dic_timeStep_flows = {}
    # nodes with nonzero demand are given by columns of dataframe
    activeNodes = injectionWithdrawalRates.columns
    for index in injectionWithdrawalRates.index:
        # compute flows corresponding to demand by fixing demand for every node to given value and then compute
        # flows by LP
        dic_nodes_MinCapacity = {}
        dic_nodes_MaxCapacity = {}
        for node in graph.nodes:
            if node in activeNodes:
                dic_nodes_MinCapacity[node] = injectionWithdrawalRates.at[index, node]
                dic_nodes_MaxCapacity[node] = injectionWithdrawalRates.at[index, node]
            else:
                dic_nodes_MinCapacity[node] = 0
                dic_nodes_MaxCapacity[node] = 0
        # compute flows
        dic_timeStep_flows[index] = computeSingleSpecialScenario(graph, distances, entries, exits, activeNodes[0],
                                                                 activeNodes[1], dic_nodes_MinCapacity,
                                                                 dic_nodes_MaxCapacity, False)

    return dic_timeStep_flows


def determineDiscretePipelineDesign(robust, injectionWithdrawalRates, distances, dic_node_minPress, dic_node_maxPress,
                                    maxLengthPipe=None, dic_diameter_costs=None, dic_candidateMergedDiam_costs=None,
                                    opexForDiameters=None, economicLifetime=30, interestRate=0.08, costUnit='€', ir=0.2,
                                    rho_n=0.089882, T_m=20 + 273.15, T_n=273.15, p_n=1.01325, Z_n=1.00062387922965,
                                    originalFluidFlows=None, nDigits=6):
    """
    We compute a robust (depending on parameter robust) optimal pipeline design,
    i.e. for a given network, we compute a minimal spanning tree w.r.t. its total length.
    Afterward, we compute our robust (special) scenarios, see Robinius et. al..
    Also we compute for every timeStep of injectionWithdrawalRates the corresponding flows.
    We compute merged diameters according to list candidatesMergedDiameter, i.e. we compute a equivalent single diameter
    for two parallel pipes with the same diameter
    If robust is True, then we compute the corresponding pressure drops for every diameter and robust scenario.
    If robust is False, then we compute for every timeStep the corresponding pressure drops for every diameter and
    timeStep.
    If robust is True, then we compute optimal diameters by a MIP for the robust scenarios.
    If robust is False, then we compute optimal diameters by a MIP for the timeStep scenarios. Not Robust Version!
    In a postprocessing step, we compute "precise" pressure levels for the robust scenarios and the timeStep scenarios.

    Note that if robust is False, then the network may be infeasible for robust scenarios
    which can occur in the network!

    :param robust: Bool that is true, we build a robust pipeline network, otherwise not
    :type robust: bool

    :param injectionWithdrawalRates: the argument is a pandas DataFrame with the index column
        denoting the timesteps and the index row denoting the name of the network's nodes.
        Injection are denoted with negative floats and withdrawal with positive floats
        in [kg/s]. Example:

              node1 node2 node3
        0      -4     2     2
        1       3   -1.5  -1.5
        ...    ...   ...   ...
        8759    0    -1     1.

    :type injectionWithdrawalRates: pandas DataFrame with floats

    :param distances: the parameter is a pandas Series with the indices being tuples of the
        network's nodes and the values being the lengths of the pipelines in [m]. Example:

        (node1, node2)   1000
        (node2, node3)  50000
        (node2, node1)   1000
        (node3, node2)  50000

    :type distances: pandas Series

    :param dic_node_minPress: dictionary that contains for every node of the network its lower pressure bound in [bar]
    :type dic_node_minPress: dictionary: key: node of the network, value: non-negative float

    :param dic_node_maxPress: dictionary that contains for every node of the network its upper pressure bound in [bar]
    :type dic_node_maxPress: dictionary key: node of the network, value: non-negative float

    It holds dic_node_minPress[index] <= dic_node_maxPress[index]

    :param maxLengthPipe: parameter that is upper limit for pipeline length in [m].
     If a pipe is longer then maxLengthPipe,
        then we will split this pipe into several pipes with equidistant length with the help of networkRefinement.
        |br| * the default value is None, i.e., no network refinement will be applied
    :type maxLengthPipe: positive float; optional

    :param dic_diameter_costs: dictionary that contains all diameters in [m] as keys and the values are the
    corresponding costs in [Euro/m]. Default Value is a preselection of diameters and its costs.
    if None, then we chose the following preselection of diameters and costs
    dic_diameter_costs = {0.1063: 37.51, 0.1307: 38.45, 0.1593: 39.64, 0.2065: 42.12, 0.2588: 45.26, 0.3063: 48.69,
    0.3356: 51.07, 0.3844: 55.24, 0.432: 59.86, 0.4796: 64.98, 0.527: 70.56, 0.578: 76.61,
    0.625: 82.99, 0.671: 89.95, 0.722: 97.38, 0.7686: 105.28, 0.814: 113.63, 0.864: 122.28,
    0.915: 131.56, 0.96: 141.3, 1.011: 151.5, 1.058: 162.17, 1.104: 173.08, 1.155: 184.67,
    1.249: 209.24, 1.342: 235.4, 1.444: 263.66, 1.536: 293.78}
    :type dic_diameter_costs: dict with keys: diameters, values: cost for pipeline; optional

    :param dic_candidateMergedDiam_costs: dictionary that contains a set of diameters in [m] as keys and
     the values are the corresponding costs in [Euro/m]. This diameters are then used to compute a single equivalent
     diameter for two looped (parallel) pipes with the considered diameter.
        |br| * the default value is empty dictionary {}
    :type dic_candidateMergedDiam_costs: dict with keys: diameters, values: cost for pipeline; optional

    :TODO not used @Juelich where to use
    param opexForDiameters: opex of the individual pipeline diameter classes, specified
        in costUnit/m/a (--> default €/m/a)
        |br| * the default value is []
    :type opexForDiameters: list; optional

    :TODO not used @Juelich where to use
    param economicLifetime: economic lifetime of the pipeline network in years
        |br| * the default value is 30
    :type economicLifetime: integer > 0; optional

    :TODO not used @Juelich where to use
    param interestRate: considered interest rate
        |br| * the default value is 0.08
    :type interestRate: float (0,1); optional

    :TODO not used @Juelich where to use
    param costUnit: cost unit of the invest
        |br| * the default value is '€
    :type costUnit: string; optional

    :param ir: integral roughness of pipe in [mm]
        |br| * the default value is 0.2 (hydrogen, this value can also be used for methane)
    :type ir: positive float

    :param rho_n: density at standard state in [kg/m^3]
        |br| * the default value is 0.089882 (hydrogen, you can use 0.71745877 for methane)
    :type rho_n: positive float

    :param T_m: constant temperature in [kelvin]
        |br| * the default value is 20 + 273.15 (hydrogen, you can use 281.15 for methane)
    :type T_m: float

    :param T_n: temperature in standard state in [kelvin]
        |br| * the default value is 273.15 (hydrogen, this value can also be used for methane)
    :type T_n: float

    :param p_n: pressure at standard state in [bar]
        |br| * the default value is 1.01325 (hydrogen, this value can also be used for methane)
    :type p_n: non-negative float

    :param Z_n: realgasfactor of hydrogen at standard state
        |br| * the default value is 1.00062387922965 (hydrogen, you can use 0.997612687740414 for methane)
    :type Z_n: non-negative float

    # TODO @Juelich where to use
    param originalFluidFlows: string that specifies the considered fluid
        |br| * the default value is None
    :type originalFluidFlows: str; optional

    :param nDigits: number of digits used in the round function
        |br| * the default value is 6
    :type nDigits: positive int

    :return: dictionary: key: arcs of the network, value: (number of pipes, optimal diameter of arc in [m]);
    here number of pipe is usually 1, but if we have a looped pipe (two parallel pipes) with the same diameter, then it
    is 2
    :rtype: dictionary key: arc, value: tuple

    :return: dictionary containing the pressure levels in [bar] of the postprocessing for the robust (special) scenarios
    :rtype: dictionary: key: nodePair, value: dict: key: node, value: pressure level of node

    :return: dictionary containing the pressure levels in [bar] of the postprocessing for each timeStep
    :rtype: dictionary: key: timeStep, value: dict: key: node, value: pressure level of node
    """
    # Do type and value check of input data:
    isBool(robust)
    isPandasDataFrameNumber(injectionWithdrawalRates)
    isPandasSeriesPositiveNumber(distances)
    isDictionaryPositiveNumber(dic_node_minPress)
    isDictionaryPositiveNumber(dic_node_maxPress)
    checkLowerUpperBoundsOfDicts(dic_node_minPress, dic_node_maxPress)
    if maxLengthPipe is not None:
        utils.isPositiveNumber(maxLengthPipe)
    # extract diameters for the optimization
    if dic_diameter_costs is not None:
        if isinstance(dic_diameter_costs, dict):
            diameters = list(dic_diameter_costs.keys())
            if isinstance(diameters, list):
                for diam in diameters:
                    utils.isStrictlyPositiveNumber(diam)
        else:
            raise TypeError("The input argument has to be a list")
    isDictionaryPositiveNumber(dic_diameter_costs)
    if dic_candidateMergedDiam_costs is not None:
        if isinstance(dic_candidateMergedDiam_costs, dict):
            for diam in dic_candidateMergedDiam_costs.keys():
                utils.isStrictlyPositiveNumber(diam)
                utils.isPositiveNumber(dic_candidateMergedDiam_costs[diam])
        else:
            raise TypeError("The input argument has to be a list")
    if opexForDiameters is not None:
        if isinstance(opexForDiameters, list):
            for opex in opexForDiameters:
                utils.isPositiveNumber(opex)
        else:
            raise TypeError("The input argument has to be a list")
    utils.isPositiveNumber(interestRate)
    utils.isStrictlyPositiveNumber(economicLifetime)
    utils.isString(costUnit)
    utils.isStrictlyPositiveNumber(ir)
    utils.isStrictlyPositiveNumber(rho_n)
    if not isinstance(T_m, float):
        raise TypeError("The input argument has to be an number")

    if not isinstance(T_n, float):
        raise TypeError("The input argument has to be an number")
    utils.isPositiveNumber(p_n)
    utils.isPositiveNumber(Z_n)
    if originalFluidFlows is not None:
        utils.isString(originalFluidFlows)
    utils.isStrictlyPositiveInt(nDigits)

    if dic_diameter_costs is None:
        print("There are no diameters to choose in the optimization. Thus, we consider the diameters and costs:")
        dic_diameter_costs = {0.1063: 37.51, 0.1307: 38.45, 0.1593: 39.64, 0.2065: 42.12, 0.2588: 45.26, 0.3063: 48.69,
                              0.3356: 51.07, 0.3844: 55.24, 0.432: 59.86, 0.4796: 64.98, 0.527: 70.56, 0.578: 76.61,
                              0.625: 82.99, 0.671: 89.95, 0.722: 97.38, 0.7686: 105.28, 0.814: 113.63, 0.864: 122.28,
                              0.915: 131.56, 0.96: 141.3, 1.011: 151.5, 1.058: 162.17, 1.104: 173.08, 1.155: 184.67,
                              1.249: 209.24, 1.342: 235.4, 1.444: 263.66, 1.536: 293.78}
        print(dic_diameter_costs)

    # create graph with respect to distances
    print('Creating graph with respect to given distances')
    graph, distances = createNetwork(distances)

    # plot graph
    print("Original Network Graph:")
    nx.draw(graph, with_labels=True)
    plt.show()

    # Create a minimum spanning tree of the network with a reasonable logic
    print('Creating a minimum spanning tree with minimal total length')
    graph, distances = createMinimumSpanningTree(graph, distances)

    print("Minimal Spanning Tree:")
    nx.draw(graph, with_labels=True)
    plt.show()

    # Apply a network refinement, if maxLengthPipe is not None
    if maxLengthPipe is not None:
        # apply network refinement:
        print("Apply network refinement")
        graph, distances, dic_node_minPress, dic_node_maxPress = networkRefinement(distances, maxLengthPipe,
                                                                                   dic_node_minPress, dic_node_maxPress)
        print("Network after pipeline refinement")
        nx.draw(graph, with_labels=True)
        plt.show()

    # Compute robust scenarios for spanning tree network
    print("Compute robust scenario set for spanning tree network")

    dic_nodePair_flows, entries, exits = generateRobustScenarios(injectionWithdrawalRates, graph, distances)

    # Compute scenarios for timeSteps
    print("Compute scenarios for timeStep. May take a little longer. Number Time Steps is "
          + str(injectionWithdrawalRates.shape[0]))
    dic_timeStep_flows = computeTimeStepFlows(injectionWithdrawalRates, distances, graph, entries, exits)

    # Compute equivalent single diameters for looped (parallel) pipes
    print("Compute equivalent single diameters for looped (parallel) pipes")
    # dic_LoopedDiam_costs contains the new computed diameters and its costs
    dic_LoopedDiam_costs = None
    # dic_newDiam_oldDiam merges new and old diameters
    dic_newDiam_oldDiam = None
    if dic_candidateMergedDiam_costs is not None:
        dic_LoopedDiam_costs, dic_newDiam_oldDiam = computeLargeMergedDiameters(dic_candidateMergedDiam_costs)

        # merge all diameters to one dictionary for the optimization model
        dic_diameter_costs.update(dic_LoopedDiam_costs)

    # Compute pressure drops for each scenario and diameter and the compute optimal diameters
    # depending on robust, we do this w.r.t. robust scenarios or every timeStep
    # dictionary for the pressure coefficients
    dic_pressureCoef = {}
    # dictionary for the optimal diameters
    dic_arc_diam = {}
    if robust:
        # we compute the pressure drops for the robust scenarios
        print("Pressure drop coefficients for diameters with respect to robust scenarios")
        dic_pressureCoef = determinePressureDropCoef(dic_nodePair_flows, distances, dic_node_minPress,
                                                     dic_node_maxPress, list(dic_diameter_costs.keys()))
        specialScenarionames = list(dic_nodePair_flows.keys())

        # Determine optimal discrete pipeline selection by solving a MIP w.r.t. the robust scenarios
        print('Determining optimal robust pipeline design under the consideration of pressure losses and robust '
              'scenarios')
        # returns dict: key: arc, value: optimal diameter
        # returns dict: key: nodePair, value: dic: key: node, value: pressure level
        dic_arc_diam, dic_scen_node_press = determineOptimalDiscretePipelineSelection(graph, distances,
                                                                                      dic_pressureCoef,
                                                                                      specialScenarionames,
                                                                                      dic_node_minPress,
                                                                                      dic_node_maxPress,
                                                                                      dic_diameter_costs, robust)
    else:
        # we compute pressure drops for every timeStep scenario. Not robust version!
        # we compute the pressure drops for the robust scenarios and optimize
        print("Pressure drop coefficients for diameters with respect to robust scenarios")
        dic_pressureCoef = determinePressureDropCoef(dic_timeStep_flows, distances, dic_node_minPress,
                                                     dic_node_maxPress, list(dic_diameter_costs.keys()))
        timeSteps = list(dic_timeStep_flows.keys())

        # Determine optimal discrete pipeline selection by solving a MIP w.r.t. the timeStep scenarios
        print('Determining optimal pipeline design under the consideration of pressure losses and every time step')
        print('This network design is necessarily robust!')
        # returns dict: key: arc, value: optimal diameter
        # returns dict: key: timeStep, value: dic: key: node, value: pressure level
        dic_arc_diam, dic_scen_node_press = determineOptimalDiscretePipelineSelection(graph, distances,
                                                                                      dic_pressureCoef, timeSteps,
                                                                                      dic_node_minPress,
                                                                                      dic_node_maxPress,
                                                                                      dic_diameter_costs, False)

    if not dic_arc_diam:
        print("No feasible diameter selections exits")
        return None

    # Do postprocessing: Use a "more" accurate pressure model and apply Postprocessing of master's thesis:
    # first do postprocessing for special scenarios
    print("Do postprocessing for robust (special) scenarios")
    dic_scen_PressLevels, dic_scen_MaxViolPress = postprocessing(graph, distances, dic_arc_diam, dic_nodePair_flows,
                                                                 dic_node_minPress, dic_node_maxPress)
    # print if some of these scenarios are not feasible for the "more" precise pressure model
    for scenario in dic_scen_MaxViolPress.keys():
        if dic_scen_MaxViolPress[scenario] > 0:
            print("Robust Scenario " + str(scenario) + " violates pressure bounds by " +
                  str(dic_scen_MaxViolPress[scenario]))

    # compute pressure levels for each time step
    print("Do postprocessing for timeStep scenarios. May take a little longer. Number of time steps is " +
          str(injectionWithdrawalRates.shape[0]))
    dic_timeStep_PressLevels, dic_timeStep_MaxViolPress = postprocessing(graph, distances, dic_arc_diam,
                                                                         dic_timeStep_flows, dic_node_minPress,
                                                                         dic_node_maxPress)

    for timeStep in dic_timeStep_MaxViolPress.keys():
        if dic_timeStep_MaxViolPress[timeStep] > 0:
            print("Time Step " + str(timeStep) + " violates pressure bounds by " +
                  str(dic_timeStep_MaxViolPress[timeStep]))

    # now determine final output, i.e. dictionary: key: arcs, values: (numberOfPipes, diameter)
    # note usually numberOfPipes is 1, but if we have chosen a merged diameter, then we have two parallel pipes with
    # the same diameter, i.e. numberOfPipes is 2.
    dic_arc_optimalDiameters = {}
    for arc in dic_arc_diam.keys():
        if dic_LoopedDiam_costs is not None:
            if dic_arc_diam[arc] in dic_LoopedDiam_costs.keys():
                dic_arc_optimalDiameters[arc] = (2, dic_newDiam_oldDiam[dic_arc_diam[arc]])
            else:
                dic_arc_optimalDiameters[arc] = (1, dic_arc_diam[arc])
        else:
            dic_arc_optimalDiameters[arc] = (1, dic_arc_diam[arc])

    print("Optimal Diameters: arc: (number of pipes, diameter)")
    print(dic_arc_optimalDiameters)

    # plot network with new diameters
    print("Network with optimized diameters")
    print("Looped pipes are indicated by two colored edges")
    print("Thicker edge means larger diameter")

    finalG = nx.MultiGraph()

    for arc in dic_arc_optimalDiameters.keys():
        if dic_arc_optimalDiameters[arc][0] == 1:
            # we have a single not looped pipe
            finalG.add_edge(arc[0], arc[1], color='black', weight=5 * dic_arc_optimalDiameters[arc][1])
        else:
            # we have a looped pipe
            finalG.add_edge(arc[0], arc[1], color='r',
                            weight=10 * dic_arc_optimalDiameters[arc][1])
            finalG.add_edge(arc[0], arc[1], color='b',
                            weight=5 * dic_arc_optimalDiameters[arc][1])
    # pos = nx.circular_layout(finalG)

    edges = finalG.edges()

    colors = []
    weight = []

    for (u, v, attrib_dict) in list(finalG.edges.data()):
        colors.append(attrib_dict['color'])
        weight.append(attrib_dict['weight'])

    nx.draw(finalG, edges=edges, edge_color=colors, width=weight, with_labels=True)
    plt.show()

    # TODO use OUTPUT for nice visualization @Juelich
    # dic_arc_optimalDiameters dictionary: key: arcs, values: (numberOfPipes, diameter)
    # note usually numberOfPipes is 1, but if we have chosen a merged diameter, then we have two parallel pipes with
    # the same diameter, i.e. numberOfPipes is 2.

    # pressure levels of postprocessing  of robust scenarios dic_scen_PressLevels: key: nodePair,
    # value: dict: key: arc, value: pressure level in [bar]

    # violation of pressure bounds of robust scenarios in optimized network determined by postprocessing
    # dic_scen_MaxViolPress: key: nodePair, value: dict: key: arc, value: non-negative number
    # (zero means no pressure violation)

    # pressure levels of postprocessing  of timeSteps dic_timeStep_PressLevels: key: timeStep,
    # value: dict: key: arc, value: pressure level in [bar]

    # violation of pressure bounds of timeStep scenarios in optimized network determined by postprocessing
    # dic_timeStep_MaxViolPress: key: nodePair, value: dict: key: arc, value: non-negative number
    # (zero means no pressure violation)

    # distances: pandas series with the arcs of the network and their lengths in [m]

    # Add some output which somehow quantifies the difference between the original and the new
    # pipeline design (for this additional input argument are required)
    # TODO @ Juelich just compare original solution to solution dic_arc_optimalDiameters

    return dic_arc_optimalDiameters, dic_scen_PressLevels, dic_scen_MaxViolPress, dic_timeStep_PressLevels, \
           dic_timeStep_MaxViolPress
