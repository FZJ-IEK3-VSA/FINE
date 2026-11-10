"""Compare the LP files of test fixtures with stored references.

A refactoring that must not change the optimization model keeps these files
identical. Set UPDATE_GOLDEN=1 to regenerate the references intentionally.
"""

import numpy as np
import pytest


@pytest.mark.parametrize(
    "fixture_name, reference_name",
    [
        ("minimal_test_esM", "minimal_test_esM.lp"),
        ("single_node_test_esM", "single_node_test_esM.lp"),
        ("dsm_test_esM", "dsm_test_esM.lp"),
        ("balanceLimitConstraint_test_esM", "balanceLimitConstraint_test_esM.lp.gz"),
        ("perfectForesight_test_esM", "perfectForesight_test_esM.lp"),
    ],
)
def test_lp_snapshot(fixture_name, reference_name, request, assert_lp_snapshot):
    # balanceLimitConstraint_test_esM draws random input data.
    np.random.seed(0)
    esM = request.getfixturevalue(fixture_name)
    if isinstance(esM, tuple):
        esM = esM[0]
    assert_lp_snapshot(esM, reference_name)
