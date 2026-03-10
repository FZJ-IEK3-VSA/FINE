import pytest
from fine import utils

from fine.utils import ImplementedSolvers


def test_userWarnings_esm(minimal_test_esM):
    """Tests if the warnings are supressed only when intended and shown otherwise in energySystemModel.py."""
    with pytest.warns(
        UserWarning, match="Invalid input. An outputLevel parameter of 2 is assumed."
    ):
        minimal_test_esM.optimize(
            timeSeriesAggregation=False,
            solver=ImplementedSolvers.STANDARD_OPEN_SOURCE_SOLVER.value,
        )
        minimal_test_esM.getOptimizationSummary("SourceSinkModel", outputLevel=5)

    # TODO: test also if DeprecationWarning and FutureWarning are ignored


def test_userWarnings_utils(minimal_test_esM):
    """Tests if the warnings are shown in utils.py."""
    with pytest.warns(
        UserWarning,
        match="CO2 emissions are not considered in the current esM. CO2ReductionTargets will be ignored.",
    ):
        utils.checkSinkCompCO2toEnvironment(minimal_test_esM, CO2ReductionTargets=1)
