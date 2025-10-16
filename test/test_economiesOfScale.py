# %%
import numpy as np
import pandas as pd

import fine as fn


# %%
def test_eos_NPV():
    """
    Test case for basic npv calculation with eos module.
    """
    # %%
    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=4,
        hoursPerTimeStep=2190,
        costUnit="1 Euro",
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            economicLifetime=10,
            interestRate=0.1,
            pwlcfParameters={
                "eosParameters": pd.DataFrame(
                    data=np.array(
                        [[0, 1, 2, 3], [0, 1000, 1800, 2400], [0, 10, 18, 24]]
                    ).T,
                    columns=["capacity", "totalInvest", "totalOpex"],
                )
            },
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="electricity_sink",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.Series([2190 * 1.8] * 4),
        )
    )

    esM.declareOptimizationProblem()

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # %%
    # correct comissioning/capacity:
    commissioning = [
        esM.getOptimizationSummary("SourceSinkModel", ip=0).loc[
            "PV", "commissioning", "[kW$_{el}$]"
        ]["loc1"]
    ]
    capacity = [
        esM.getOptimizationSummary("SourceSinkModel", ip=0).loc[
            "PV", "capacity", "[kW$_{el}$]"
        ]["loc1"]
    ]
    np.testing.assert_almost_equal(commissioning, 1.8)
    np.testing.assert_almost_equal(capacity, 1.8)

    # correct obj:
    np.testing.assert_almost_equal(actual=esM.pyM.Obj(), desired=283.3024476)

    # check for correct invest/TAC/NPV:
    invest = esM.getOptimizationSummary("SourceSinkModel", ip=0).loc[
        "PV", "invest", "[1 Euro]"
    ]["loc1"]
    investEOS = esM.getOptimizationSummary("SourceSinkModel", ip=0).loc[
        "PV", "invest_EOS", "[1 Euro]"
    ]["loc1"]
    np.testing.assert_almost_equal(actual=invest, desired=investEOS)

    tac = esM.getOptimizationSummary("SourceSinkModel", ip=0).loc[
        "PV", "TAC", "[1 Euro/a]"
    ]["loc1"]
    tacEOS = esM.getOptimizationSummary("SourceSinkModel", ip=0).loc[
        "PV", "TAC_EOS", "[1 Euro/a]"
    ]["loc1"]
    np.testing.assert_almost_equal(actual=tac, desired=tacEOS)

    npv = esM.getOptimizationSummary("SourceSinkModel", ip=0).loc[
        "PV", "NPVcontribution", "[1 Euro]"
    ]["loc1"]
    npvEOS = esM.getOptimizationSummary("SourceSinkModel", ip=0).loc[
        "PV", "NPVcontribution_EOS", "[1 Euro]"
    ]["loc1"]
    np.testing.assert_almost_equal(actual=npv, desired=npvEOS)


if __name__ == "__main__":
    test_eos_NPV()
