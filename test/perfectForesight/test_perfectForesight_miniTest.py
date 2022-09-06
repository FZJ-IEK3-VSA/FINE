import FINE as fn
import numpy as np
import pytest
import pandas as pd
import math

def perfectForesight_test_esM():

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand", "ForesightLand"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=5,
        yearsPerInvestmentPeriod=5,
        startYear=2020,
        mode="perfectForesight",
        lengthUnit="km",
        verboseLogLevel=2,
    )

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

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            capacityMax=4e6,
            investPerCapacity=1e3,
            opexPerCapacity=1,
            interestRate=0.02,
            opexPerOperation=0.01,
            economicLifetime=10,
        )
    )

    demand = {}
    demand[2020] = pd.DataFrame(
        [
            np.array(
                [
                    4380,
                    1e3,
                ]
            ),
            np.array(
                [
                    2190,
                    1e3,
                ]
            ),
        ],
        index=["PerfectLand", "ForesightLand"],
    ).T  # first investmentperiod
    demand[2025]=demand[2020]
    demand[2030] = pd.DataFrame(
        [
            np.array(
                [
                    2190,
                    1e3,
                ]
            ),
            np.array(
                [
                    4380,
                    1e3,
                ]
            )
        ],
        index=["PerfectLand", "ForesightLand"],
    ).T
    demand[2035]=demand[2030]
    demand[2040]=demand[2030]

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    return esM

def test_perfectForesight_mini(perfectForesight_test_esM):
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    np.testing.assert_almost_equal(perfectForesight_test_esM.pyM.Obj(), 11861.771783274202)

def test_stock_wrongStockYear(perfectForesight_test_esM):
    
    with pytest.raises(ValueError, match=r".*stockCommissioning was initialized for.*"):
        fn.Source(
            esM=perfectForesight_test_esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=4e6,
            investPerCapacity=2 * 2190,
            opexPerCapacity=0,
            interestRate=0.02,
            opexPerOperation= 0.01,
            economicLifetime=5,
            stockCommissioning={
                2005: pd.Series([10,5],index=perfectForesight_test_esM.locations),
                2012: pd.Series([10,5],index=perfectForesight_test_esM.locations),
                2015: pd.Series([0.5,0.25],index=perfectForesight_test_esM.locations),
            }
        )

def test_perfectForesight_stock(perfectForesight_test_esM):
    esM = perfectForesight_test_esM
    PVoperationRateMax = esM.getComponent("PV").operationRateMax

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            capacityMax=4e6,
            investPerCapacity=1e3,
            opexPerCapacity=1,
            interestRate=0.02,
            opexPerOperation=0.01,
            economicLifetime=10,
            stockCommissioning={
                2005: pd.Series([10,5],index=["ForesightLand","PerfectLand"]),
                2010: pd.Series([10,5], index=["ForesightLand","PerfectLand"]),
                2015: pd.Series([0.5,0.25],index=["ForesightLand","PerfectLand"]),
            }
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    
    # CHECKS 
    # check the objective value
    np.testing.assert_almost_equal(esM.pyM.Obj(), 11861.771783274202)
    
    # check some commissioning and decommissioning variables
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand","PV",-1)] == 0.25
    assert esM.pyM.decommis_srcSnk.get_values()[("PerfectLand","PV",1)] == 0.25
    assert esM.pyM.commis_srcSnk.get_values()[("ForesightLand","PV",-1)] ==0.5
    assert esM.pyM.decommis_srcSnk.get_values()[("ForesightLand","PV",1)] ==0.5
    assert esM.pyM.commis_srcSnk.get_values()[("ForesightLand","PV",-2)] ==10
    assert esM.pyM.decommis_srcSnk.get_values()[("ForesightLand","PV",0)] == 10
    assert esM.pyM.commis_srcSnk.get_values()[("ForesightLand","PV",0)] == 1.5
    assert esM.pyM.cap_srcSnk.get_values()[("ForesightLand","PV",0)]==2
        
    # check processedStockCommissioning
    assert list(esM.getComponent('PV').processedStockCommissioning.keys()) == [-1,-2]
    assert perfectForesight_test_esM.getComponent('PV').processedStockYears == [-2,-1]


    # check that parameters are correctly setup
    # a) parameters which need to include stock years as commissioning year dependent
    assert list(esM.getComponent("PV").processedInvestPerCapacity.keys()) == [-2,-1,0,1,2,3,4]
    assert list(esM.getComponent("PV").processedOpexPerCapacity.keys()) == [-2,-1,0,1,2,3,4]
    assert list(esM.getComponent("PV").processedOpexIfBuilt.keys()) == [-2,-1,0,1,2,3,4]
    assert list(esM.getComponent("PV").processedInvestIfBuilt.keys()) == [-2,-1,0,1,2,3,4]
    assert list(esM.getComponent("PV").QPbound.keys()) == [-2,-1,0,1,2,3,4]
    
    
    # b) parameters which do not need to include stock years 
    assert list(esM.getComponent("PV").processedOpexPerOperation.keys())== [0,1,2,3,4]
    assert list(esM.getComponent("PV").processedOperationRateMax.keys())== [0,1,2,3,4]
    
    # check the optimization summary
    srcSnk_optSum=esM.getOptimizationSummary("SourceSinkModel")
    assert srcSnk_optSum[2020].loc[("PV","decommissioning","[kW$_{el}$]"),"ForesightLand"]== 10
    assert srcSnk_optSum[2020].loc[("PV","capacity","[kW$_{el}$]"),"ForesightLand"] ==2
    assert srcSnk_optSum[2020].loc[("PV","commissioning","[kW$_{el}$]"),"ForesightLand"] ==1.5
    assert math.isnan(srcSnk_optSum[2020].loc[("EDemand","commissioning","[kW$_{el}$]"),"ForesightLand"])


if __name__ == "__main__":
    test_perfectForesight_stock(perfectForesight_test_esM())    