#!/usr/bin/env python
# coding: utf-8

# # Workflow for a multi-regional energy system
#
# In this application of the FINE framework, a multi-regional energy system is modeled and optimized.
#
# All classes which are available to the user are utilized and examples of the selection of different parameters within these classes are given.
#
# The workflow is structures as follows:
# 1. Required packages are imported and the input data path is set
# 2. An energy system model instance is created
# 3. Commodity sources are added to the energy system model
# 4. Commodity conversion components are added to the energy system model
# 5. Commodity storages are added to the energy system model
# 6. Commodity transmission components are added to the energy system model
# 7. Commodity sinks are added to the energy system model
# 8. The energy system model is optimized
# 9. Selected optimization results are presented
#

# 1. Import required packages and set input data path

import FINE as fn
import numpy as np
import pandas as pd


def test_miniSystem(minimal_test_esM):
    minimal_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")

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
