import numpy as np
import pandas as pd
import pytest
from fine.utils import ImplementedSolvers


# Deliberately compares the results of all solvers, independent of the selected one
@pytest.mark.multi_solver
@pytest.mark.parametrize(
    "solver",
    [
        ImplementedSolvers.GUROBI.value,
        ImplementedSolvers.HIGHS.value,
        ImplementedSolvers.GLPK.value,
    ],
)
def test_miniSystem(minimal_test_esM, solver):
    if (
        solver == ImplementedSolvers.GUROBI.value
        and not ImplementedSolvers._gurobi_available()
    ):
        pytest.skip("gurobi not available")

    minimal_test_esM.optimize(
        timeSeriesAggregation=False,
        solver=solver,
    )

    # test if solve fits to the original results
    testresults = minimal_test_esM.componentModelingDict[
        "SourceSinkModel"
    ].operationVariablesOptimum.xs("Electricity market")
    np.testing.assert_array_almost_equal(
        testresults.values,
        [
            np.array([1.877143e07, 3.754286e07, 0.0, 1.877143e07]),
        ],
        decimal=-3,
    )

    # test if the summary fits to the expected summary
    summary = minimal_test_esM.getOptimizationSummary("SourceSinkModel")

    # of cost
    costs = pd.DataFrame(
        [
            np.array(
                [
                    0.05,
                    0.0,
                    0.1,
                    0.051,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T

    np.testing.assert_almost_equal(
        summary.loc[
            ("Electricity market", "commodCosts", "[1 Euro/a]"), "ElectrolyzerLocation"
        ],
        costs["ElectrolyzerLocation"]
        .mul(np.array([1.877143e07, 3.754286e07, 0.0, 1.877143e07]))
        .sum(),
        decimal=0,
    )

    # and of revenues
    revenues = pd.DataFrame(
        [
            np.array(
                [
                    0.0,
                    0.01,
                    0.0,
                    0.0,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T
    np.testing.assert_almost_equal(
        summary.loc[
            ("Electricity market", "commodRevenues", "[1 Euro/a]"),
            "ElectrolyzerLocation",
        ],
        revenues["ElectrolyzerLocation"]
        .mul(np.array([1.877143e07, 3.754286e07, 0.0, 1.877143e07]))
        .sum(),
        decimal=0,
    )
