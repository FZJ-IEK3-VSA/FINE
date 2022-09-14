import FINE as fn
import pandas as pd


def test_TSAmultiStage(minimal_test_esM):
    """
    Get the minimal test system, and check if the Error-Bounding-Approach works for it
    """

    # modify the minimal LP and change it to a MILP
    esM = minimal_test_esM

    # get the components with capacity variables
    electrolyzers = esM.getComponent("Electrolyzers")
    pressureTank = esM.getComponent("Pressure tank")
    pipelines = esM.getComponent("Pipelines")

    # set binary variables and define bigM
    electrolyzers.hasIsBuiltBinaryVariable = True
    pressureTank.hasIsBuiltBinaryVariable = True
    pipelines.hasIsBuiltBinaryVariable = True

    electrolyzers.processedInvestIfBuilt[0] = pd.Series(2e5, index=esM.locations)
    pressureTank.processedInvestIfBuilt[0] = pd.Series(1e5, index=esM.locations)
    pipelines.processedInvestIfBuilt[0].loc[:] = 100

    electrolyzers.bigM = 30e4
    pressureTank.bigM = 30e6
    pipelines.bigM = 30e3

    electrolyzers.processedInvestIfBuilt[0] = pd.Series(2e5, index=esM.locations)
    pressureTank.processedInvestIfBuilt[0] = pd.Series(1e5, index=esM.locations)

    electrolyzers.bigM = 30e4
    pressureTank.bigM = 30e6

    # optimize with 2 Stage Approach
    fn.optimizeTSAmultiStage(
        esM,
        relaxIsBuiltBinary=True,
        solver="glpk",
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=1,
    )

    # get gap
    gap = esM.gap

    assert gap > 0.1078 and gap < 0.1079
