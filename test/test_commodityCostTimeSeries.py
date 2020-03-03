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
import os
import pandas as pd
import numpy as np

def test_commodityCostTimeSeries(multi_node_test_esM_optimized):

    # read in original results
    expected_results = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples',
                                       'Multi-regional Energy System Workflow', 'totalBiogasPurchase.csv'),
                          index_col=0, header=None, squeeze=True)

    # test if here solved fits with original results
    testresults = multi_node_test_esM_optimized.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('Biogas purchase').sum(axis=1)
    np.testing.assert_array_almost_equal(testresults.values, expected_results.values, decimal=2)
