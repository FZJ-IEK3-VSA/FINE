import FINE as fn
import numpy as np
import pytest
import pandas as pd

def test_perfectForesight_mini(perfectForesight_test_esM):
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    np.testing.assert_almost_equal(perfectForesight_test_esM.pyM.Obj(), 11861.771783274202)

def test_perfectForesight_stock(perfectForesight_test_esM):

    # with pytest.raises(ValueError, match=r".*stockCommissioning was initialized for.*"):
    #     fn.Source(
    #         esM=perfectForesight_test_esM,
    #         name="PV",
    #         commodity="electricity",
    #         hasCapacityVariable=True,
    #         capacityMax=4e6,
    #         investPerCapacity=2 * 2190,
    #         opexPerCapacity=0,
    #         interestRate=0.02,
    #         opexPerOperation= 0.01,
    #         economicLifetime=5,
    #         stockCommissioning={
    #             2005: pd.Series([10,5],index=perfectForesight_test_esM.locations),
    #             2012: pd.Series([10,5],index=perfectForesight_test_esM.locations),
    #             2015: pd.Series([0.5,0.25],index=perfectForesight_test_esM.locations),
    #         }
    #     )
    PVoperationRateMax = pd.DataFrame(
        [
            np.array(
                [
                    0.5,
                    0.25,
                ]
            ),
            np.array(
                [
                    0.25,
                    0.5,
                ]
            )
        ],
        index=["PerfectLand", "ForesightLand"],
    ).T

    perfectForesight_test_esM.add(
        fn.Source(
            esM=perfectForesight_test_esM,
            name="PV1",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            capacityMax=4e6,
            investPerCapacity=1e3,
            opexPerCapacity=1,
            interestRate=0.02,
            opexPerOperation=0.01,
            economicLifetime=10,#pd.Series([10,5],index=perfectForesight_test_esM.locations),
            # stockCommissioning={
            #     2005: pd.Series([10,5],index=perfectForesight_test_esM.locations),
            #     2010: pd.Series([10,5], index=perfectForesight_test_esM.locations),
            #     2015: pd.Series([0.5,0.25],index=perfectForesight_test_esM.locations),
            # }
        )
    )

    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="gurobi")