import fine as fn
from fine.utils import ImplementedSolvers


def test_fullloadhours_above(minimal_test_esM):
    """Get the minimal test system, and check if the fulllload hours of electrolyzer are above 4000."""
    esM = minimal_test_esM

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    # Plot the operational heat map
    fig, ax = fn.plotOperationColorMap(
        esM,
        "Electrolyzers",
        "ElectrolyzerLocation",
        figsize=(4, 3),
        nbTimeStepsPerPeriod=1,
        nbPeriods=4,
        yticks=[0, 1],
    )
