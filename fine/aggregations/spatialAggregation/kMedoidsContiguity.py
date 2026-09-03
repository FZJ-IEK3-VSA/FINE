"""Contiguity constrained k-medoids clustering for spatial aggregation.

The algorithm is based on Oehrlein and Haunert (2017): *A cutting-plane method
for contiguity-constrained spatial aggregation*. It was previously provided by
``tsam.utils.k_medoids_contiguity``; ETHOS.TSAM 4.0 dropped it together with the
``utils`` package and its networkx dependency, so it is maintained here (the
original is MIT licensed, cf. https://github.com/FZJ-IEK3-VSA/tsam).

Only the spatial aggregation uses this: regions may only be merged with regions
they share a border with, which the plain k-medoids formulation cannot express.
"""

import logging
import time

import networkx as nx
import numpy as np
import pyomo.environ as pyomo
from pyomo import opt
from pyomo.contrib.appsi.solvers import Highs

logger_kmedoids = logging.getLogger("spatial_grouping")


def k_medoids_contiguity(
    distances, n_clusters, adjacency, max_iter=500, solver="highs"
):
    """Cluster with k-medoids while enforcing that every cluster stays contiguous.

    A relaxed k-medoids problem is solved repeatedly; after each solve, every
    cluster that fell apart into disconnected parts contributes a cutting plane,
    until all clusters are connected.

    **Required arguments:**

    :param distances: Symmetric matrix of the distances between all candidates.
    :type distances: numpy.ndarray

    :param n_clusters: Number of clusters to group the candidates into.
    :type n_clusters: strictly positive integer

    :param adjacency: Symmetric matrix marking with a 1 which candidates are
        adjacent to each other. Must describe a connected graph.
    :type adjacency: numpy.ndarray

    **Default arguments:**

    :param max_iter: Maximum number of cutting plane iterations.
        |br| * the default value is 500
    :type max_iter: strictly positive integer

    :param solver: Solver used for the k-medoids MILP.
        |br| * the default value is 'highs'
    :type solver: string

    :returns: Tuple of the medoid selection vector, the transposed assignment
        matrix and the objective value.
    :rtype: tuple
    """
    # First transform the network to a networkx instance which is required for cut generation
    G = _contiguity_to_graph(adjacency, distances=distances)

    # check if inputs are correct
    if np.size(distances) != np.size(adjacency):
        raise ValueError("distances and adjacency must have the same size")

    # and test for connectivity
    if not nx.is_connected(G):
        raise ValueError("The given adjacency matrix is not connected.")

    # Initial setup of k medoids
    M = _setup_k_medoids(distances, n_clusters)

    M.adjacency = adjacency

    # Add constraintlist for the cuts later added
    M.cuts = pyomo.ConstraintList()

    # Loop over the relaxed k-medoids problem and add cuts until the problem fits
    _all_cluster_connected = False
    _iter = 0
    _cuts_added = []
    while not _all_cluster_connected and _iter < max_iter:
        # first solve instance
        t_presolve = time.time()
        r_x, r_y, obj = _solve_given_pyomo_model(M, solver=solver)
        t_aftersolve = time.time()
        logger_kmedoids.debug(
            "%s. iteration: total distance %s with solving time %s",
            _iter,
            obj,
            t_aftersolve - t_presolve,
        )

        _candidates, labels = np.where(r_x == 1)
        # claim that the resulting clusters are connected
        _all_cluster_connected = True
        _new_cuts_added = []
        for label in np.unique(labels):
            # extract the cluster
            cluster = G.subgraph(np.where(labels == label)[0])
            # Identify if the cluster is contiguous, instead of validating the constraints such as Validi and Oehrlein.
            if not nx.is_connected(cluster):
                _all_cluster_connected = False
                # if not add contiguity constraints based on c-v (Oehrlein) or a-b (Validi) separators
                for candidate in cluster.nodes:
                    # It is not clear in Validi and Oehrlein, if cuts between all cluster candidates or just the center
                    # and the candidates shall be made. The latter one does not converge for the test system wherefore
                    # the first one is chosen.
                    for node in cluster.nodes:
                        # different to Validi et al. (2021) and Oehrlein and Haunert (2017), check first and just add
                        # continuity constraints for the not connected candidates to increase performance
                        if nx.node_connectivity(cluster, node, candidate) == 0:
                            # check that the cut was not added so far for the cluster
                            if (label, candidate, node) in _cuts_added:
                                raise ValueError(
                                    "Minimal cluster/candidate separation/minimum cut does not seem sufficient. "
                                    "Adding additional separators could help."
                                )
                            # include the cut in the cut list
                            _new_cuts_added.append((label, candidate, node))
                            # Cuts to Separators - Appendix A Minimum-weight vertex separators (Oehrlein and
                            # Haunert, 2017). Validi uses an own cut generator and Oehrlein falls back to a Java
                            # library, here we use simple max flow cutting.
                            cut_set = nx.minimum_node_cut(G, node, candidate)
                            # (Eq. 13 - Oehrlein and Haunert, 2017)
                            M.cuts.add(
                                sum(M.z[u, node] for u in cut_set)
                                >= M.z[candidate, node]
                            )
        # Total cuts
        _cuts_added.extend(_new_cuts_added)
        _iter += 1
        t_afteradding = time.time()

        logger_kmedoids.debug(
            "%s contiguity constraints/cuts added, adding to a total number of %s cuts within time %s",
            len(_new_cuts_added),
            len(_cuts_added),
            t_afteradding - t_aftersolve,
        )

    return (r_y, r_x.T, obj)


def _contiguity_to_graph(adjacency, distances=None):
    """Transform an adjacency matrix into a :class:`networkx.Graph`.

    :param adjacency: 2-dimensional adjacency matrix.
    :type adjacency: numpy.ndarray

    :param distances: If provided, delivers the distances between the nodes,
        which are turned into edge weights.
        |br| * the default value is None
    :type distances: numpy.ndarray

    :returns: Graph with every index as node name.
    :rtype: networkx.Graph
    """
    rows, cols = np.where(adjacency == 1)
    G = nx.Graph()
    if distances is None:
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)
    else:
        normed_distances = distances / np.max(distances)
        weights = 1 - normed_distances
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("Weight calculation went wrong.")

        edge_weights = weights[rows, cols]
        edges = zip(rows.tolist(), cols.tolist(), edge_weights.tolist())
        G.add_weighted_edges_from(edges)
    return G


def _setup_k_medoids(distances, n_clusters):
    """Define the k-medoids model with pyomo.

    In the spatial aggregation community, it is referred to as Hess model for
    political districting with an additional constraint of cluster
    sizes/populations (W Hess, JB Weaver, HJ Siegfeldt, JN Whelan, and PA
    Zitlau. Nonpartisan political redistricting by computer. Operations
    Research, 13(6):998-1006, 1965.).

    :param distances: Symmetric matrix of the distances between all candidates.
    :type distances: numpy.ndarray

    :param n_clusters: Number of clusters.
    :type n_clusters: strictly positive integer

    :returns: The k-medoids model.
    :rtype: pyomo.environ.ConcreteModel
    """
    # Create model
    M = pyomo.ConcreteModel()

    # get distance matrix
    M.d = distances

    # set number of clusters
    M.no_k = n_clusters

    # Distances is a symmetrical matrix, extract its length
    length = distances.shape[0]

    # get indices
    M.i = list(range(length))
    M.j = list(range(length))

    # initialize vars
    # Decision every candidate to every possible other candidate as cluster center
    M.z = pyomo.Var(M.i, M.j, within=pyomo.Binary)

    # get objective
    # Minimize the distance of every candidate to the cluster center
    def objRule(M):
        return sum(sum(M.d[i, j] * M.z[i, j] for j in M.j) for i in M.i)

    M.obj = pyomo.Objective(rule=objRule)

    # s.t.
    # Assign all candidates to one clusters
    def candToClusterRule(M, j):
        return sum(M.z[i, j] for i in M.i) == 1

    M.candToClusterCon = pyomo.Constraint(M.j, rule=candToClusterRule)

    # Predefine the number of clusters
    def noClustersRule(M):
        return sum(M.z[i, i] for i in M.i) == M.no_k

    M.noClustersCon = pyomo.Constraint(rule=noClustersRule)

    # Describe the choice of a candidate to a cluster
    def clusterRelationRule(M, i, j):
        return M.z[i, j] <= M.z[i, i]

    M.clusterRelationCon = pyomo.Constraint(M.i, M.j, rule=clusterRelationRule)
    return M


def _solve_given_pyomo_model(M, solver="highs"):
    """Solve a given pyomo clustering model and return the clusters.

    :param M: Concrete model instance that gets solved.
    :type M: pyomo.environ.ConcreteModel

    :param solver: Defines the solver for the pyomo model.
        |br| * the default value is 'highs'
    :type solver: string

    :returns: Tuple of the assignment matrix, the medoid selection vector and
        the objective value.
    :rtype: tuple
    """
    # create optimization problem
    if solver == "highs":
        solver_instance = Highs()
    else:
        solver_instance = opt.SolverFactory(solver)
    solver_instance.solve(M)  # results checked via model state

    # Get results
    r_x = np.array([[round(M.z[i, j].value) for i in M.i] for j in M.j])

    r_y = np.array([round(M.z[j, j].value) for j in M.j])

    r_obj = pyomo.value(M.obj)

    return (r_x, r_y, r_obj)
