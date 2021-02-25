def test_solver_not_specified(minimal_test_esM):
    """
    Test solver not specified. The first available solver
    in `solverList` will be chosen.

    """
    esM = minimal_test_esM
    esM.optimize()
    assert esM.solverSpecs["terminationCondition"] == "optimal"


def test_solver_glpk(minimal_test_esM):
    """
    Test glpk solver. If not available the first available solver
    in `solverList` will be chosen.
    """
    esM = minimal_test_esM
    esM.optimize(solver="glpk")
    assert esM.solverSpecs["terminationCondition"] == "optimal"


def test_solver_coincbc(minimal_test_esM):
    """
    Test coincbc solver. If not available the first available solver
    in `solverList` will be chosen.
    """
    esM = minimal_test_esM
    esM.optimize(solver="coincbc")
    assert esM.solverSpecs["terminationCondition"] == "optimal"


def test_solver_gurobi(minimal_test_esM):
    """
    Test Gurobi solver. If not available the first available solver
    in `solverList` will be chosen.
    """
    esM = minimal_test_esM
    esM.optimize(solver="gurobi")
    assert esM.solverSpecs["terminationCondition"] == "optimal"


def test_solver_not_available(minimal_test_esM):
    """
    Test solver that is not available. The first available solver
    in `solverList` will be chosen.
    """
    esM = minimal_test_esM
    esM.optimize(solver="not_available_solver")
    assert esM.solverSpecs["terminationCondition"] == "optimal"