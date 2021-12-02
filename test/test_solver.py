def test_solver_not_specified(minimal_test_esM):
    """
    Test solver not specified. The first available solver
    in `solverList` will be chosen.

    """
    esM = minimal_test_esM
    esM.optimize()
    assert esM.solverSpecs["terminationCondition"] == "optimal"
