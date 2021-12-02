"""
Last edited: January 22 2020

|br| @author: FINE Developer Team (FZJ IEK-3)
# here we provide unit tests of our main functions in robustPipelineSizing
"""
from FINE.expansionModules import robustPipelineSizing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def test_robustPipelineDesign():
    # write tests for function createNetwork()
    # check if returned graph has the edges of the input argument:
    def test_createNetwork(distances):
        graph, newdistances = robustPipelineSizing.createNetwork(distances)
        # check that every arc of the graph is in the original distances matrix
        # check that every arc of the graph is in the new distances matrix
        # check that in the new distance matrix only one direction either (u,v) or (v,u) is contained
        for arcIndex in list(graph.edges):
            assert (
                arcIndex in distances.index
                or (arcIndex[1], arcIndex[0]) in distances.index
            )
            assert (
                arcIndex in newdistances.index
                or (arcIndex[1], arcIndex[0]) in newdistances.index
            )
            assert not (
                arcIndex in newdistances.index
                and (arcIndex[1], arcIndex[0]) in newdistances.index
            )
        # check that every arcIndex of the original distance matrix is in the graph
        for arcIndex in distances.index:
            assert arcIndex in graph.edges or (arcIndex[1], arcIndex[0]) in graph.edges
        # check that every arcIndex of new distances matrix is in the graph
        for arcIndex in newdistances.index:
            assert arcIndex in graph.edges or (arcIndex[1], arcIndex[0]) in graph.edges
        # check lengths of the graph
        for arcIndex in nx.get_edge_attributes(graph, "length").keys():
            if arcIndex in newdistances.index:
                assert (
                    newdistances[arcIndex]
                    == nx.get_edge_attributes(graph, "length")[arcIndex]
                )
            else:
                assert (
                    newdistances[(arcIndex[1], arcIndex[0])]
                    == nx.get_edge_attributes(graph, "length")[arcIndex]
                )

        return graph, newdistances

    # we test the function createSteinerTree
    def test_createSteinerTree(graph, distances):
        stTree, newdistances = robustPipelineSizing.createSteinerTree(
            graph, distances, list(graph.nodes)
        )
        # check if in newDistances only arcs of the spanning tree are contained
        for arc in stTree.edges:
            # first check that only one direction of the arc is contained
            assert not (arc in newdistances.index and (arc[1], arc[0]))
            # check that the arc or its reversered arc is contained
            assert arc in newdistances.index or (arc[1], arc[0]) in newdistances.index
        # check that all arcs, respectively its reversed arc is contained in MinSpannTree
        for arc in newdistances.index:
            assert arc in graph.edges or (arc[1], arc[0]) in graph.edges
        # check weights of the graph
        for arc in nx.get_edge_attributes(graph, "length").keys():
            if arc in newdistances.index:
                assert newdistances[arc] == nx.get_edge_attributes(graph, "length")[arc]
            else:
                assert (
                    newdistances[arc]
                    == nx.get_edge_attributes(graph, "length")[(arc[1], arc[0])]
                )
        return stTree, newdistances

    def test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    ):
        (
            G,
            newdistances,
            dic_node_minPress,
            dic_node_maxPress,
        ) = robustPipelineSizing.networkRefinement(
            distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
        )
        # check that every arc of the graph is in the distance matrix, respectively its reversed arc
        # check that not the arc and its reversed are contained in the distance matrix
        for arc in list(G.edges):
            assert arc in newdistances.index or (arc[1], arc[0]) in newdistances.index
            assert not (
                arc in newdistances.index and (arc[1], arc[0]) in newdistances.index
            )
        # check that every arc of new distances matrix is in the graph
        for arc in newdistances.index:
            assert arc in G.edges or (arc[1], arc[0]) in G.edges
        # check that we have a minimal and maximal pressure for every node of the graph
        for node in G.nodes:
            assert node in dic_node_minPress.keys() and node in dic_node_maxPress.keys()

        return G, newdistances, dic_node_minPress, dic_node_maxPress

    def test_computeSingleSpecialScenario(
        graph,
        distances,
        entries,
        exits,
        startNode,
        endNode,
        dic_nodes_MinCapacity,
        dic_nodes_MaxCapacity,
        specialScenario=True,
    ):
        dic_scenario_flow = robustPipelineSizing.computeSingleSpecialScenario(
            graph,
            distances,
            entries,
            exits,
            startNode,
            endNode,
            dic_nodes_MinCapacity,
            dic_nodes_MaxCapacity,
            specialScenario,
        )
        # check if each arc has a flow
        for arc in distances.index:
            assert not (
                arc in dic_scenario_flow.keys()
                and (arc[1], arc[0]) in dic_scenario_flow.keys()
            )
            assert (
                arc in dic_scenario_flow.keys()
                or (arc[1], arc[0]) in dic_scenario_flow.keys()
            )

        return dic_scenario_flow

    def test_generateRobustScenarios(
        injectionWithdrawalRates, graph, distances, dic_node_minPress, dic_node_maxPress
    ):
        (
            dic_robustScenarios,
            entries,
            exits,
        ) = robustPipelineSizing.generateRobustScenarios(
            injectionWithdrawalRates,
            graph,
            distances,
            dic_node_minPress,
            dic_node_maxPress,
        )

        # we compute optimal values for each nodePair and save them in dictionary
        dic_nodePair_optvalue = {}
        for nodePair in dic_robustScenarios.keys():
            # compute path between this two nodes
            shortestPath = nx.shortest_path(graph, nodePair[0], nodePair[1])
            obj = 0.0
            for i in range(0, len(shortestPath) - 1):
                if (shortestPath[i], shortestPath[i + 1]) in list(
                    dic_robustScenarios[nodePair].keys()
                ):
                    obj = (
                        obj
                        + dic_robustScenarios[nodePair][
                            (shortestPath[i], shortestPath[i + 1])
                        ]
                    )
                else:
                    obj = (
                        obj
                        - dic_robustScenarios[nodePair][
                            (shortestPath[i + 1], shortestPath[i])
                        ]
                    )

            dic_nodePair_optvalue[nodePair] = obj

        return dic_robustScenarios, entries, exits, dic_nodePair_optvalue

    def checkGraphDistanceMatrixCoincide(graph, distances):
        # check that for every arc in the graph either the arc or its reversed arc are contained in the distance matrix
        for arcIndex in list(graph.edges):
            assert (
                arcIndex in distances.index
                or (arcIndex[1], arcIndex[0]) in distances.index
            )
            assert not (
                arcIndex in distances.index
                and (arcIndex[1], arcIndex[0]) in distances.index
            )
        # check that for every arc in the distance matrix either the arc or its reversed are contained in the graph
        for arcIndex in distances.index:
            assert arcIndex in list(graph.edges) or (arcIndex[1], arcIndex[0]) in list(
                graph.edges
            )
            assert not (
                arcIndex in list(graph.edges)
                and (arcIndex[1], arcIndex[0]) in list(graph.edges)
            )
        # check lengths of the graph
        for arcIndex in nx.get_edge_attributes(graph, "length").keys():
            if arcIndex in distances.index:
                assert (
                    distances[arcIndex]
                    == nx.get_edge_attributes(graph, "length")[arcIndex]
                )
            else:
                assert (
                    distances[(arcIndex[1], arcIndex[0])]
                    == nx.get_edge_attributes(graph, "length")[arcIndex]
                )

        return

    # ######################################################################################################################
    # unit test: createNetwork
    # create input data and call tests
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    invalidData = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 450.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    invalidDistances = pd.Series(invalidData, index=keys)
    # should raise an assertion error because an arc and its reversed arc with different length exist
    try:
        graph, distances = test_createNetwork(invalidDistances)
    except AssertionError as error:
        print(error)
    # We create the new network
    try:
        graph, distances = test_createNetwork(distances)
    except AssertionError as error:
        print(error)
        print(
            "Something went wrong in createNetwork; check comment above assertion for error"
        )

    # # uncomment this block if you want the graph being plotted
    # nx.draw(graph, with_labels=True)
    # plt.show()

    #######################################################################################################################
    # unit test function: createSteinerTree
    # create input data for testing and the optimal solution of the minimal spanning tree
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    optimalSolution = {
        ("w2", "w1"): 1200.0,
        ("w2", "w3"): 750.0,
        ("w4", "w3"): 500.0,
        ("w3", "w5"): 600.0,
    }

    # test the minimum spanning tree function
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )

    # check if spanning tree is optimal
    for arc in stTree.edges:
        if arc in optimalSolution.keys():
            assert optimalSolution[arc] == nx.get_edge_attributes(stTree, "length")[arc]
        elif (arc[1], arc[0]) in optimalSolution.keys():
            assert (
                optimalSolution[(arc[1], arc[0])]
                == nx.get_edge_attributes(stTree, "length")[arc]
            )
        else:
            print("Something went wrong in computation of minimum spanning tree")
            raise ()
    # check that the distance matrix and the graph of the graph (spanning tree) coincide
    checkGraphDistanceMatrixCoincide(stTree, distances)

    # # uncomment the following block if you want the minimal spanning tree to be plotted
    # nx.draw(stTree, with_labels=True)
    # plt.show()

    #######################################################################################################################
    # unit test function networkRefinement
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    # optimal solution
    dic_node_minPressTruth = {
        "w1": 50,
        "w2": 45,
        "w3": 60,
        "w4": 60,
        "w5": 50,
        "v1_w2_w1": 47.5,
        "v2_w2_w1": 47.5,
        "v1_w2_w3": 52.5,
        "v1_w3_w5": 55,
    }
    dic_node_maxPressTruth = {
        "w1": 100,
        "w2": 90,
        "w3": 100,
        "w4": 80,
        "w5": 95,
        "v1_w2_w1": 95,
        "v2_w2_w1": 95,
        "v1_w2_w3": 95,
        "v1_w3_w5": 97.5,
    }
    nodesTruth = list(dic_node_minPressTruth.keys())
    edgesTruth = [
        ("w2", "v1_w2_w1"),
        ("v1_w2_w1", "v2_w2_w1"),
        ("v2_w2_w1", "w1"),
        ("w4", "w3"),
        ("w2", "v1_w2_w3"),
        ("w3", "v1_w2_w3"),
        ("w3", "v1_w3_w5"),
        ("v1_w3_w5", "w5"),
    ]

    # test function networkRefinement
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )

    # check solution
    assert sorted(edgesTruth) == sorted(list(graph.edges))
    assert sorted(nodesTruth) == sorted(list(graph.nodes))
    assert dic_node_minPress == dic_node_minPressTruth
    assert dic_node_maxPress == dic_node_maxPressTruth

    # uncomment this block if you want the graph to be plotted
    # nx.draw(graph, with_labels=True)
    # plt.show()

    ######################################################################################################################
    # unit test function computeSingleSpecialScenario Case: SpecialScenario = true
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    dic_nodes_minCapacity = {"w1": -2, "w2": 0, "w3": -2, "w4": 0, "w5": 0}
    dic_nodes_maxCapacity = {"w1": 0, "w2": 1, "w3": 2, "w4": 0, "w5": 4}
    startNode = "w1"
    endNode = "w3"
    # create optimal solution
    entries = ["w1", "w3"]
    exits = ["w2", "w3", "w5"]
    optSol_W1_W2 = {
        ("w2", "v1_w2_w1"): -2.0,
        ("v1_w2_w1", "v2_w2_w1"): -2.0,
        ("v2_w2_w1", "w1"): -2.0,
        ("w2", "v1_w2_w3"): 2.0,
        ("v1_w2_w3", "w3"): 2.0,
    }

    # test function computeSingleSpecialScenario
    dic_scenario_flow_W1_W3 = test_computeSingleSpecialScenario(
        graph,
        distances,
        entries,
        exits,
        startNode,
        endNode,
        dic_nodes_minCapacity,
        dic_nodes_maxCapacity,
        True,
    )
    # check solution: since flow values on arcs not part of the unique path between start and endNode are not unique,
    # we only check the unique flow values on the path between start and endNode
    for arc in optSol_W1_W2.keys():
        assert dic_scenario_flow_W1_W3[arc] == optSol_W1_W2[arc]

    #######################################################################################################################
    # unit test function computeSingleSpecialScenario Case: SpecialScenario = false
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    dic_nodes_minCapacity = {
        "w1": -1,
        "w2": 0,
        "w3": -1,
        "w4": 0,
        "w5": 2,
        "v1_w2_w1": 0,
        "v2_w2_w1": 0,
        "v1_w2_w3": 0,
        "v1_w3_w5": 0,
    }
    dic_nodes_maxCapacity = dic_nodes_minCapacity
    startNode = "w1"
    endNode = "w3"
    # create optimal solution
    entries = []
    exits = []
    optSol_W1_W2 = {
        ("w2", "v1_w2_w1"): -1.0,
        ("v1_w2_w1", "v2_w2_w1"): -1.0,
        ("v2_w2_w1", "w1"): -1.0,
        ("w2", "v1_w2_w3"): 1.0,
        ("v1_w2_w3", "w3"): 1.0,
        ("w3", "v1_w3_w5"): 2.0,
        ("w3", "v1_w3_w5"): 2.0,
    }

    # test function computeSingleSpecialScenario
    dic_scenario_flow_W1_W3 = test_computeSingleSpecialScenario(
        graph,
        distances,
        entries,
        exits,
        startNode,
        endNode,
        dic_nodes_minCapacity,
        dic_nodes_maxCapacity,
        False,
    )
    # check solution: since demands are fixed, the solution is unique
    for arc in optSol_W1_W2.keys():
        assert dic_scenario_flow_W1_W3[arc] == optSol_W1_W2[arc]

    ######################################################################################################################
    # unit test of function generateRobustScenarios
    # create input data and optimal solution (only parts because inner function computeSingleSpecialScenario already tested
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    injectionRates = {
        "w1": [-2.0, 0.0, -2.0, -2.0],
        "w2": [1.0, 0.0, 0.0, 0.0],
        "w3": [1.0, -2.0, 2.0, -2.0],
        "w4": [0.0, 0.0, 0.0, 0.0],
        "w5": [0.0, 2.0, 0.0, 4.0],
    }
    injectionWithdrawal = pd.DataFrame(data=injectionRates)
    # optimal solution
    entriesTruth = ["w1", "w3"]
    exitsTruth = ["w2", "w3", "w5"]
    # optimal value of each scenario
    dic_nodePair_optValueTruth = {
        ("w4", "w3"): 0.0,
        ("w4", "w2"): 2.0,
        ("w4", "v1_w2_w1"): 2.0,
        ("w4", "v2_w2_w1"): 2.0,
        ("w4", "w1"): 2.0,
        ("w4", "v1_w2_w3"): 1.0,
        ("w4", "v1_w3_w5"): 4.0,
        ("w4", "w5"): 8.0,
        ("w3", "w4"): 0.0,
        ("w3", "w2"): 2.0,
        ("w3", "v1_w2_w1"): 2.0,
        ("w3", "v2_w2_w1"): 2.0,
        ("w3", "w1"): 2.0,
        ("w3", "v1_w2_w3"): 1.0,
        ("w3", "v1_w3_w5"): 4.0,
        ("w3", "w5"): 8.0,
        ("w2", "w4"): 4.0,
        ("w2", "w3"): 4.0,
        ("w2", "v1_w2_w1"): 0.0,
        ("w2", "v2_w2_w1"): 0.0,
        ("w2", "w1"): 0.0,
        ("w2", "v1_w2_w3"): 2.0,
        ("w2", "v1_w3_w5"): 8.0,
        ("w2", "w5"): 12.0,
        ("v1_w2_w1", "w4"): 6.0,
        ("v1_w2_w1", "w3"): 6.0,
        ("v1_w2_w1", "w2"): 2.0,
        ("v1_w2_w1", "v2_w2_w1"): 0.0,
        ("v1_w2_w1", "w1"): 0.0,
        ("v1_w2_w1", "v1_w2_w3"): 4.0,
        ("v1_w2_w1", "v1_w3_w5"): 10.0,
        ("v1_w2_w1", "w5"): 14.0,
        ("v2_w2_w1", "w4"): 8.0,
        ("v2_w2_w1", "w3"): 8.0,
        ("v2_w2_w1", "w2"): 4.0,
        ("v2_w2_w1", "v1_w2_w1"): 2.0,
        ("v2_w2_w1", "w1"): 0.0,
        ("v2_w2_w1", "v1_w2_w3"): 6.0,
        ("v2_w2_w1", "v1_w3_w5"): 12.0,
        ("v2_w2_w1", "w5"): 16.0,
        ("w1", "w4"): 10.0,
        ("w1", "w3"): 10.0,
        ("w1", "w2"): 6.0,
        ("w1", "v1_w2_w1"): 4.0,
        ("w1", "v2_w2_w1"): 2.0,
        ("w1", "v1_w2_w3"): 8.0,
        ("w1", "v1_w3_w5"): 14.0,
        ("w1", "w5"): 18.0,
        ("v1_w2_w3", "w4"): 2.0,
        ("v1_w2_w3", "w3"): 2.0,
        ("v1_w2_w3", "w2"): 1.0,
        ("v1_w2_w3", "v1_w2_w1"): 1.0,
        ("v1_w2_w3", "v2_w2_w1"): 1.0,
        ("v1_w2_w3", "w1"): 1.0,
        ("v1_w2_w3", "v1_w3_w5"): 6.0,
        ("v1_w2_w3", "w5"): 10.0,
        ("v1_w3_w5", "w4"): 0.0,
        ("v1_w3_w5", "w3"): 0.0,
        ("v1_w3_w5", "w2"): 2.0,
        ("v1_w3_w5", "v1_w2_w1"): 2.0,
        ("v1_w3_w5", "v2_w2_w1"): 2.0,
        ("v1_w3_w5", "w1"): 2.0,
        ("v1_w3_w5", "v1_w2_w3"): 1.0,
        ("v1_w3_w5", "w5"): 4.0,
        ("w5", "w4"): 0.0,
        ("w5", "w3"): 0.0,
        ("w5", "w2"): 2.0,
        ("w5", "v1_w2_w1"): 2.0,
        ("w5", "v2_w2_w1"): 2.0,
        ("w5", "w1"): 2.0,
        ("w5", "v1_w2_w3"): 1.0,
        ("w5", "v1_w3_w5"): 0.0,
    }
    nodes = list(dic_node_minPress.keys())
    nodePair = []
    for startnode in nodes:
        for endnode in nodes:
            if startnode is not endnode:
                nodePair.append((startnode, endnode))

    # test function generateRobustScenarios
    (
        dic_robustScenarios,
        entries,
        exits,
        dic_nodePair_optValue,
    ) = test_generateRobustScenarios(
        injectionWithdrawal, graph, distances, dic_node_minPress, dic_node_maxPress
    )

    assert sorted(entriesTruth) == sorted(entries)
    assert sorted(exitsTruth) == sorted(exits)
    assert sorted(dic_robustScenarios.keys()) == sorted(nodePair)
    assert dic_nodePair_optValue == dic_nodePair_optValueTruth

    ######################################################################################################################
    # unit test of function computeLargeMergedDiameters
    # create input data and optimal solution
    dic_diamToMerge_costs = {0.144: 10, 1.500: 20}
    # optimal solution
    dic_mergedDiamTruth = {0.190009: 20, 1.979262: 40}
    dic_reversed_diamsTruth = {0.190009: 0.144, 1.979262: 1.500}
    (
        dic_mergedDiam,
        dic_reversed_diams,
    ) = robustPipelineSizing.computeLargeMergedDiameters(dic_diamToMerge_costs, 6)

    assert dic_mergedDiam == dic_mergedDiamTruth
    assert dic_reversed_diams == dic_reversed_diamsTruth

    #######################################################################################################################
    # unit function test of function determinePressureDropCoef
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    diameters = [0.1063, 1.536]
    testscenarios = {
        ("w5", "w1"): {
            ("w4", "w3"): 0.0,
            ("w2", "v1_w2_w1"): 0.0,
            ("v1_w2_w1", "v2_w2_w1"): 0.0,
            ("v2_w2_w1", "w1"): 0.0,
            ("w2", "v1_w2_w3"): -1.0,
            ("v1_w2_w3", "w3"): -1.0,
            ("w3", "v1_w3_w5"): 0.0,
            ("v1_w3_w5", "w5"): 0.0,
        },
        ("w2", "v1_w3_w5"): {
            ("w4", "w3"): 0.0,
            ("w2", "v1_w2_w1"): -2.0,
            ("v1_w2_w1", "v2_w2_w1"): -2.0,
            ("v2_w2_w1", "w1"): -2.0,
            ("w2", "v1_w2_w3"): 2.0,
            ("v1_w2_w3", "w3"): 2.0,
            ("w3", "v1_w3_w5"): 4.0,
            ("v1_w3_w5", "w5"): 4.0,
        },
    }
    # optimal solution
    dic_pressure_coefTruth = {
        (0.1063, ("w5", "w1")): {
            ("w4", "w3"): 0,
            ("w2", "v1_w2_w1"): 0,
            ("v1_w2_w1", "v2_w2_w1"): 0,
            ("v2_w2_w1", "w1"): 0,
            ("w2", "v1_w2_w3"): -131.38307913282281,
            ("v1_w2_w3", "w3"): -131.83249054911866,
            ("w3", "v1_w3_w5"): 0,
            ("v1_w3_w5", "w5"): 0,
        },
        (0.1063, ("w2", "v1_w3_w5")): {
            ("w4", "w3"): 0,
            ("w2", "v1_w2_w1"): -558.2111338732643,
            ("v1_w2_w1", "v2_w2_w1"): -558.5140246655361,
            ("v2_w2_w1", "w1"): -559.5025149566926,
            ("w2", "v1_w2_w3"): 523.2976028753353,
            ("v1_w2_w3", "w3"): 525.1425774408543,
            ("w3", "v1_w3_w5"): 1677.5126669798137,
            ("v1_w3_w5", "w5"): 1674.151293128955,
        },
        (1.536, ("w5", "w1")): {
            ("w4", "w3"): 0,
            ("w2", "v1_w2_w1"): 0,
            ("v1_w2_w1", "v2_w2_w1"): 0,
            ("v2_w2_w1", "w1"): 0,
            ("w2", "v1_w2_w3"): -0.0001700642888961138,
            ("v1_w2_w3", "w3"): -0.00017067718965527387,
            ("w3", "v1_w3_w5"): 0,
            ("v1_w3_w5", "w5"): 0,
        },
        (1.536, ("w2", "v1_w3_w5")): {
            ("w4", "w3"): 0,
            ("w2", "v1_w2_w1"): -0.0006460078899527932,
            ("v1_w2_w1", "v2_w2_w1"): -0.0006463741986205677,
            ("v2_w2_w1", "w1"): -0.000647570434768236,
            ("w2", "v1_w2_w3"): 0.0006056017622910718,
            ("v1_w2_w3", "w3"): 0.0006078348401792327,
            ("w3", "v1_w3_w5"): 0.0017698676222416114,
            ("v1_w3_w5", "w5"): 0.0017661900309253952,
        },
    }

    dic_pressure_coef = robustPipelineSizing.determinePressureDropCoef(
        testscenarios, distances, dic_node_minPress, dic_node_maxPress, diameters
    )

    assert dic_pressure_coef == dic_pressure_coefTruth

    ######################################################################################################################
    # unit test of function computeTimeStepFlows
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    injectionRates = {
        "w1": [-20.0, -20.0],
        "w2": [10.0, 0.0],
        "w3": [10.0, -20.0],
        "w4": [0.0, 0.0],
        "w5": [0.0, 40.0],
    }
    injectionWithdrawal = pd.DataFrame(data=injectionRates)
    entries = []
    exits = []
    # create optimal solution
    dic_timeStep_flowsTruth = {
        0: {
            ("w4", "w3"): 0.0,
            ("w2", "v1_w2_w1"): -20.0,
            ("v1_w2_w1", "v2_w2_w1"): -20.0,
            ("v2_w2_w1", "w1"): -20.0,
            ("w2", "v1_w2_w3"): 10.0,
            ("v1_w2_w3", "w3"): 10.0,
            ("w3", "v1_w3_w5"): 0.0,
            ("v1_w3_w5", "w5"): 0.0,
        },
        1: {
            ("w4", "w3"): 0.0,
            ("w2", "v1_w2_w1"): -20.0,
            ("v1_w2_w1", "v2_w2_w1"): -20.0,
            ("v2_w2_w1", "w1"): -20.0,
            ("w2", "v1_w2_w3"): 20.0,
            ("v1_w2_w3", "w3"): 20.0,
            ("w3", "v1_w3_w5"): 40.0,
            ("v1_w3_w5", "w5"): 40.0,
        },
    }

    dic_timeStep_flows = robustPipelineSizing.computeTimeStepFlows(
        injectionWithdrawal, distances, graph, entries, exits
    )

    assert dic_timeStep_flows == dic_timeStep_flowsTruth

    ######################################################################################################################
    # unit test of function determineOptimalDiscretePipelineSelection case Robust = True
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    injectionRates = {
        "w1": [-20.0, 0.0, -20.0, -20.0],
        "w2": [10.0, 0.0, 0.0, 0.0],
        "w3": [10.0, -20.0, 20.0, -20.0],
        "w4": [0.0, 0.0, 0.0, 0.0],
        "w5": [0.0, 20.0, 0.0, 40.0],
    }
    injectionWithdrawal = pd.DataFrame(data=injectionRates)
    dic_robustScenarios, entries, exits = robustPipelineSizing.generateRobustScenarios(
        injectionWithdrawal, graph, distances, dic_node_minPress, dic_node_maxPress
    )
    diameters = [0.1063, 1.536]
    # for debugging reason we consider only two special scenarios
    dic_robustTestScenarios = {}
    dic_robustTestScenarios[("w1", "w3")] = dic_robustScenarios[("w1", "w3")]
    dic_robustTestScenarios[("w5", "w1")] = dic_robustScenarios[("w5", "w1")]
    dic_pressure_coef = robustPipelineSizing.determinePressureDropCoef(
        dic_robustTestScenarios,
        distances,
        dic_node_minPress,
        dic_node_maxPress,
        diameters,
    )
    dic_diameter_costs = {0.1063: 10, 1.536: 30}
    specialScenarionames = [("w1", "w3"), ("w5", "w1")]

    # optimal solution
    dic_arc_diamTruth = {
        ("w4", "w3"): 0.1063,
        ("w2", "v1_w2_w1"): 1.536,
        ("v1_w2_w1", "v2_w2_w1"): 1.536,
        ("v2_w2_w1", "w1"): 1.536,
        ("w2", "v1_w2_w3"): 1.536,
        ("v1_w2_w3", "w3"): 1.536,
        ("w3", "v1_w3_w5"): 1.536,
        ("v1_w3_w5", "w5"): 1.536,
    }

    (
        dic_arc_diam,
        dic_scen_node_press,
    ) = robustPipelineSizing.determineOptimalDiscretePipelineSelection(
        graph,
        distances,
        dic_pressure_coef,
        specialScenarionames,
        dic_node_minPress,
        dic_node_maxPress,
        dic_diameter_costs,
        robust=True,
        verbose=2,
    )

    assert sorted(dic_arc_diam) == sorted(dic_arc_diamTruth)

    #######################################################################################################################
    # unit test of function determineOptimalDiscretePipelineSelection case Robust = False
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    injectionRates = {
        "w1": [-5.0, -5.0],
        "w2": [3.0, 0.0],
        "w3": [2.0, 0.0],
        "w4": [0.0, 0.0],
        "w5": [0.0, 5.0],
    }
    injectionWithdrawal = pd.DataFrame(data=injectionRates)
    entries = []
    exits = []
    diameters = [0.1063, 1.536]
    dic_timeStep_flows = robustPipelineSizing.computeTimeStepFlows(
        injectionWithdrawal, distances, graph, entries, exits
    )
    dic_pressure_coef = robustPipelineSizing.determinePressureDropCoef(
        dic_timeStep_flows, distances, dic_node_minPress, dic_node_maxPress, diameters
    )
    dic_diameter_costs = {0.1063: 10, 1.536: 30}
    specialScenarionames = list(dic_timeStep_flows.keys())
    # create optimal solution
    dic_arc_diamTruth = {
        ("w4", "w3"): 0.1063,
        ("w2", "v1_w2_w1"): 1.536,
        ("v1_w2_w1", "v2_w2_w1"): 1.536,
        ("v2_w2_w1", "w1"): 0.1063,
        ("w2", "v1_w2_w3"): 1.536,
        ("v1_w2_w3", "w3"): 1.536,
        ("w3", "v1_w3_w5"): 1.536,
        ("v1_w3_w5", "w5"): 0.1063,
    }

    (
        dic_arc_diam,
        dic_scen_node_press,
    ) = robustPipelineSizing.determineOptimalDiscretePipelineSelection(
        graph,
        distances,
        dic_pressure_coef,
        specialScenarionames,
        dic_node_minPress,
        dic_node_maxPress,
        dic_diameter_costs,
        robust=False,
        verbose=2,
    )

    assert sorted(dic_arc_diam) == sorted(dic_arc_diamTruth)

    #######################################################################################################################
    # unit test of function computePressureStartnodeArc
    # create input data and optimal solution
    arc = ("w1", "w2")
    pressureEndNode = 50.0
    dic_arc_diam = {arc: 0.1063}
    distances = pd.Series(1000.0, index=[arc])
    dic_scenario_flows = {arc: 5.0}
    pressStartnode = robustPipelineSizing.computePressureStartnodeArc(
        arc, pressureEndNode, dic_scenario_flows, dic_arc_diam, distances
    )

    assert np.round(pressStartnode, 3) == 105.59
    # for the reversed arc flow we have the same result because the arc flow direction is handled by computePressureAtNode
    dic_scenario_flows = {arc: -5.0}
    pressStartnode = robustPipelineSizing.computePressureStartnodeArc(
        arc, pressureEndNode, dic_scenario_flows, dic_arc_diam, distances
    )
    assert np.round(pressStartnode, 3) == 105.59

    dic_scenario_flows = {arc: 0.0}
    pressStartnode = robustPipelineSizing.computePressureStartnodeArc(
        arc, pressureEndNode, dic_scenario_flows, dic_arc_diam, distances
    )
    assert np.round(pressStartnode, 3) == np.round(pressureEndNode, 3)

    #######################################################################################################################
    # unit test of function computePressureEndnodeArc
    # create input data and optimal solution
    arc = ("w1", "w2")
    pressStartNode = 100.0
    dic_arc_diam = {arc: 0.1063}
    distances = pd.Series(1000.0, index=[arc])
    dic_scenario_flows = {arc: 5.0}
    pressureEndNode = robustPipelineSizing.computePressureEndnodeArc(
        arc, pressStartNode, dic_scenario_flows, dic_arc_diam, distances
    )
    assert np.round(pressureEndNode, 3) == 37.29
    # results should be the same for reversed arc flow since the flow direction is handled by computePressureAtNode
    dic_scenario_flows = {arc: -5.0}
    pressureEndNode = robustPipelineSizing.computePressureEndnodeArc(
        arc, pressStartNode, dic_scenario_flows, dic_arc_diam, distances
    )
    assert np.round(pressureEndNode, 3) == 37.29

    dic_scenario_flows = {arc: 0.0}
    pressureEndNode = robustPipelineSizing.computePressureEndnodeArc(
        arc, pressStartNode, dic_scenario_flows, dic_arc_diam, distances
    )
    assert np.round(pressureEndNode, 3) == np.round(pressStartNode, 3)

    #######################################################################################################################
    # unit test of function computePressureAtNode
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    injectionRates = {
        "w1": [-5.0, -5.0],
        "w2": [3.0, 0.0],
        "w3": [2.0, 0.0],
        "w4": [0.0, 0.0],
        "w5": [0.0, 5.0],
    }
    injectionWithdrawal = pd.DataFrame(data=injectionRates)
    entries = []
    exits = []
    diameters = [0.1063, 1.536]
    dic_timeStep_flows = robustPipelineSizing.computeTimeStepFlows(
        injectionWithdrawal, distances, graph, entries, exits
    )
    dic_pressure_coef = robustPipelineSizing.determinePressureDropCoef(
        dic_timeStep_flows, distances, dic_node_minPress, dic_node_maxPress, diameters
    )
    dic_diameter_costs = {0.1063: 10, 1.536: 30}
    specialScenarionames = list(dic_timeStep_flows.keys())
    (
        dic_arc_diam,
        dic_scen_node_press,
    ) = robustPipelineSizing.determineOptimalDiscretePipelineSelection(
        graph,
        distances,
        dic_pressure_coef,
        specialScenarionames,
        dic_node_minPress,
        dic_node_maxPress,
        dic_diameter_costs,
        robust=False,
    )
    validation = True
    node = "w1"
    upperPressNode = "w1"
    tmp_violation = 0.0
    dic_node_Pressure = {}
    for nodeindex in graph.nodes:
        dic_node_Pressure[nodeindex] = None
    dic_timeStep_flow = dic_timeStep_flows[0]
    # optimal solution
    dic_nodePressTruth = {
        "w4": 80.6313893875461,
        "w3": 80.6313893875461,
        "w2": 80.6313969344965,
        "v1_w2_w1": 80.63141923180355,
        "v2_w2_w1": 80.63144152910475,
        "w1": 100,
        "v1_w2_w3": 80.63139316102138,
        "v1_w3_w5": 80.6313893875461,
        "w5": 80.6313893875461,
    }

    validation, tmp_violation = robustPipelineSizing.computePressureAtNode(
        validation,
        node,
        upperPressNode,
        graph,
        dic_arc_diam,
        distances,
        dic_timeStep_flow,
        dic_node_minPress,
        dic_node_maxPress,
        tmp_violation,
        dic_node_Pressure,
    )
    assert not validation
    assert tmp_violation == 0.6313893875460934
    assert dic_node_Pressure == dic_nodePressTruth

    validation = True
    node = "w4"
    upperPressNode = "w4"
    tmp_violation = 0.0
    dic_node_Pressure = {}
    for nodeindex in graph.nodes:
        dic_node_Pressure[nodeindex] = None
    dic_timeStep_flow = dic_timeStep_flows[0]
    # optimal solution
    dic_nodePressTruth = {
        "w4": 80,
        "w3": 80.0,
        "w2": 80.00000760352876,
        "v1_w2_w1": 80.00003006810435,
        "v2_w2_w1": 80.00005253267393,
        "w1": 99.4853419184688,
        "v1_w2_w3": 80.00000380176446,
        "v1_w3_w5": 80.0,
        "w5": 80.0,
    }

    validation, tmp_violation = robustPipelineSizing.computePressureAtNode(
        validation,
        node,
        upperPressNode,
        graph,
        dic_arc_diam,
        distances,
        dic_timeStep_flow,
        dic_node_minPress,
        dic_node_maxPress,
        tmp_violation,
        dic_node_Pressure,
    )
    assert validation
    assert tmp_violation == 0.0
    assert dic_node_Pressure == dic_nodePressTruth

    #####################################################################################################################
    # unit test of function postprocessing
    # create input data and optimal solution
    data = [1200.0, 1200.0, 1500.0, 750.0, 1500.0, 500.0, 500.0, 600.0, 2000.0]
    keys = [
        ("w1", "w2"),
        ("w2", "w1"),
        ("w1", "w5"),
        ("w2", "w3"),
        ("w2", "w4"),
        ("w3", "w4"),
        ("w4", "w3"),
        ("w3", "w5"),
        ("w4", "w5"),
    ]
    distances = pd.Series(data, index=keys)
    graph, distances = robustPipelineSizing.createNetwork(distances)
    stTree, distances = robustPipelineSizing.createSteinerTree(
        graph, distances, list(graph.nodes)
    )
    dic_node_maxPress = {"w1": 100, "w2": 90, "w3": 100, "w4": 80, "w5": 95}
    dic_node_minPress = {"w1": 50, "w2": 45, "w3": 60, "w4": 60, "w5": 50}
    maxPipeLength = 500
    graph, distances, dic_node_minPress, dic_node_maxPress = test_networkRefinement(
        distances, maxPipeLength, dic_node_minPress, dic_node_maxPress
    )
    dic_arc_diam = {
        ("w4", "w3"): 0.1063,
        ("w2", "v1_w2_w1"): 1.536,
        ("v1_w2_w1", "v2_w2_w1"): 1.536,
        ("v2_w2_w1", "w1"): 1.536,
        ("w2", "v1_w2_w3"): 1.536,
        ("v1_w2_w3", "w3"): 1.536,
        ("w3", "v1_w3_w5"): 1.536,
        ("v1_w3_w5", "w5"): 1.536,
    }

    testScen = {
        ("w2", "v1_w3_w5"): {
            ("w4", "w3"): 0.0,
            ("w2", "v1_w2_w1"): -200.0,
            ("v1_w2_w1", "v2_w2_w1"): -200.0,
            ("v2_w2_w1", "w1"): -200.0,
            ("w2", "v1_w2_w3"): 200.0,
            ("v1_w2_w3", "w3"): 200.0,
            ("w3", "v1_w3_w5"): 400.0,
            ("v1_w3_w5", "w5"): 400.0,
        }
    }
    # optimal solution
    dic_scenPressTruth = {
        ("w2", "v1_w3_w5"): {
            "w4": 80,
            "w3": 80.0,
            "w2": 80.05723929028916,
            "v1_w2_w1": 80.08775096325076,
            "v2_w2_w1": 80.11825156968065,
            "w1": 80.14874112202088,
            "v1_w2_w3": 80.02862451899747,
            "v1_w3_w5": 79.9087044007704,
            "w5": 79.81730934184435,
        }
    }

    dic_scen_PressLevel, dic_scen_MaxViolPress = robustPipelineSizing.postprocessing(
        graph, distances, dic_arc_diam, testScen, dic_node_minPress, dic_node_maxPress
    )

    assert dic_scenPressTruth == dic_scen_PressLevel
    assert dic_scen_MaxViolPress[("w2", "v1_w3_w5")] == 0.0
    # second testcase in which a violation exists
    dic_arc_diam = {
        ("w4", "w3"): 0.1063,
        ("w2", "v1_w2_w1"): 1.536,
        ("v1_w2_w1", "v2_w2_w1"): 1.536,
        ("v2_w2_w1", "w1"): 0.3063,
        ("w2", "v1_w2_w3"): 1.536,
        ("v1_w2_w3", "w3"): 1.536,
        ("w3", "v1_w3_w5"): 1.536,
        ("v1_w3_w5", "w5"): 1.536,
    }
    # optimal solution
    dic_scenPressTruth = {
        ("w2", "v1_w3_w5"): {
            "w4": 80,
            "w3": 80.0,
            "w2": 80.05723929028916,
            "v1_w2_w1": 80.08775096325076,
            "v2_w2_w1": 80.11825156968065,
            "w1": 168.2934447139534,
            "v1_w2_w3": 80.02862451899747,
            "v1_w3_w5": 79.9087044007704,
            "w5": 79.81730934184435,
        }
    }
    dic_scen_MaxViol = {("w2", "v1_w3_w5"): 68.29344471395339}

    dic_scen_PressLevel, dic_scen_MaxViolPress = robustPipelineSizing.postprocessing(
        graph, distances, dic_arc_diam, testScen, dic_node_minPress, dic_node_maxPress
    )
    assert dic_scen_PressLevel == dic_scenPressTruth
    assert dic_scen_MaxViolPress == dic_scen_MaxViol

    print("All Unit tests worked as expected")
