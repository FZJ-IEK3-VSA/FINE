import FINE as fn
import numpy as np
import pandas as pd
import pytest
import copy
from FINE.IOManagement import standardIO


# This is the test for multi objective optimization for the use case of raw material demands


def test_multi_objective_optimization(multi_objective_test_esm):
    test_esm = multi_objective_test_esm
    # Assert if the esm has the attribute "objective", if its a list and if the elements are "costs" and "material_demand"
    assert hasattr(test_esm, "objective")
    assert type(test_esm.objective) == list
    assert test_esm.objective.sort() == ["costs", "material_demand"].sort()
    # Declare the optimization problem
    test_esm.declareOptimizationProblem()
    # Assert if the pyomo model has the attribute "obj_list"
    assert hasattr(test_esm.pyM, "obj_list")
    # Make sure both of the objective functions are in the obj_list
    assert len(test_esm.pyM.obj_list) == 2

    # Make sure all Conversions have the attribute material demand -> Future all Components!
    assert all(
        hasattr(
            test_esm.componentModelingDict["ConversionModel"].componentsDict[comp_name],
            "processed_materialDemandPerCapacity",
        )
        for comp_name in test_esm.componentModelingDict[
            "ConversionModel"
        ].componentsDict
    )
    # USE THIS LATER!
    # assert all(
    #     hasattr(
    #         test_esm.componentModelingDict[model].componentsDict[comp_name],
    #         "processed_materialDemandPerCapacity",
    #     )
    #     for model in test_esm.componentModelingDict
    #     for comp_name in test_esm.componentModelingDict[model].componentsDict
    # )

    opts = {
        "grid_points": 5,
        "output_excel": True,
        # "cpu_count": 2,
        "process_logging": True,
        "export_solved_pyomo_models": True,
    }
    pyaugmecon = test_esm.optimize_moo(pyaugmeconOptions=opts)


def test_moo_1node_1year(moo_single_region_single_year_test_esM):
    test_esm = moo_single_region_single_year_test_esM
    test_esm.declareOptimizationProblem()
    opts = {
        "grid_points": 5,
        "output_excel": True,
        "process_logging": True,
        "export_solved_pyomo_models": True,
    }
    pyaugmecon_instance = test_esm.optimize_moo(pyaugmeconOptions=opts)


def test_obj_value_single_vs_multi(
    temp_single_objective_test_model, moo_single_region_single_year_test_esM
):
    """Compare the objective value for the default single objective optimization (costs)
    with the objective value for the same objective from the multi-objective optpimization (costs and material demand )

    Args:
        temp_single_objective_test_model (_type_): _description_
    """
    # Optimize ESM with only one objective (costs) using the existing esm.optimize() methode of FINE
    test_esm_single_objective = temp_single_objective_test_model
    test_esm_single_objective.optimize(solver="glpk")
    # Optimize ESM with multiple objectives using the esm.optimize_moo() methode based on the pyaugmecon package
    test_esm_multi_objective = moo_single_region_single_year_test_esM
    opts = {
        "grid_points": 5,
        "output_excel": False,
        "process_logging": False,
    }
    pareto_solutions_moo = test_esm_multi_objective.optimize_moo(
        pyaugmeconOptions=opts
    ).get_pareto_solutions()

    # Assert whether the objective value of the single objective optimziation is
    # almost equal to the value of the first objective of the multi-objective
    # optimizations first pareto point
    # IMPORTANT: This does only work if the costs objective is the first objective
    # Future TODO: Find a way to make sure that this is always the case...
    np.testing.assert_almost_equal(
        test_esm_single_objective.pyM.Obj(), pareto_solutions_moo[0][0]
    )


def test_second_objective_minimization(multi_objective_test_esm):
    """Test if a hydrogen source with no material demand is added if this leads to the fact,
    that no other conversions (that would induce a material demand) are used

    """
    test_esm = multi_objective_test_esm
    opts = {
        "grid_points": 5,
        "output_excel": True,
        # "cpu_count": 2,
        "process_logging": True,
    }
    # 1. Optimize ESM without additional hydrogen source
    pyaugmecon_instance_no_h2_source = test_esm.optimize_moo(pyaugmeconOptions=opts)
    # 2. Add hydrogen source - no materialDemandPerCapacity
    test_esm.add(
        fn.Source(
            esM=test_esm,
            name="Hydrogen_source",
            commodity="hydrogen",
            hasCapacityVariable=False,
            commodityCost=10,
        )
    )
    # 3. Optimize ESM again
    pyaugmecon_instance_added_h2_source = test_esm.optimize_moo(pyaugmeconOptions=opts)
    # 4. Compare operation of conversions (that produce h2) between the two ESM
    #   Future TODO -> if the way of selecting the material-optimal solution proofs to be faulty in the future,
    #   one needs to think about a different method to select the solution (e.g. via the payoff table)
    # 4.1 Assert whether the operation of conversions is greater 0 in the solution WITHOUT the additional h2 source
    assert (
        pyaugmecon_instance_no_h2_source.unique_pareto_sols[
            list(pyaugmecon_instance_no_h2_source.unique_pareto_sols)[-1]
        ]["op_conv"].sum()
        > 0
    )
    # 4.2 Assert whether the operation of the conversions is equal to 0 in the solution WITH the additional h2 source
    assert (
        pyaugmecon_instance_added_h2_source.unique_pareto_sols[
            list(pyaugmecon_instance_added_h2_source.unique_pareto_sols)[-1]
        ]["op_conv"].sum()
        == 0
    )
    # 4.3 Assert whether the added h2 source completely substitutes both conversions
    # (which should be the case because we are looking at the "material-optimal" solution and the added h2 source has no material demand!)
    assert (
        pyaugmecon_instance_added_h2_source.unique_pareto_sols[
            list(pyaugmecon_instance_added_h2_source.unique_pareto_sols)[-1]
        ]["op_srcSnk"]["Location"]["Hydrogen_source"].sum()
        / 0.7
        == pyaugmecon_instance_no_h2_source.unique_pareto_sols[
            list(pyaugmecon_instance_no_h2_source.unique_pareto_sols)[-1]
        ]["op_conv"].sum()
    )


def test_multi_objective_optimization_perfect_foresight(
    multi_objective_optimization_test_esM,
):
    test_esm = multi_objective_optimization_test_esM
    test_esm.declareOptimizationProblem()
    opts = {
        "grid_points": 5,
        "output_excel": True,
        # "cpu_count": 2,
        "process_logging": True,
    }
    pyaugmecon_instance = test_esm.optimize_moo(pyaugmeconOptions=opts)
