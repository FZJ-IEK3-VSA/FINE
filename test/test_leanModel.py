import sys
import os
import pandas as pd

from attr import dataclass 

import FINE as fn

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "Multi-regional_Energy_System_Workflow",
    )
)
from getData import getData


def addEmptyRegions(sel, transmissionComp, locations_list):
    """
    Set attributes to zeros if data for a region is missing. 
    """
    if transmissionComp == False:
        if type(sel) == pd.Series:
            if len(sel) != len(locations_list):
                for x in locations_list:
                    if x not in sel.index:
                        tst = pd.Series([0], index=[x])
                        sel = pd.concat([sel, tst], axis=0)
        elif type(sel) == pd.DataFrame:
            if len(sel.columns) != len(locations_list):
                for x in locations_list:
                    if x not in sel.columns:
                        sel[x] = 0
    return sel

def test_leanModel():
    # delete the capacity and operation rate max (time series) for one region (cluster_0)
    data = getData()

    # set up the energy system model instance
    locations = {
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "cluster_3",
        "cluster_4",
        "cluster_5",
        "cluster_6",
        "cluster_7",
    }

    # # Delete cluster_0 in Wind (onshore)
    data["Wind (onshore), operationRateMax"].drop('cluster_0', axis=1, inplace=True)
    data["Wind (onshore), capacityMax"].drop('cluster_0', inplace=True)

    data["Wind (onshore), operationRateMax"]= addEmptyRegions(data["Wind (onshore), operationRateMax"], False ,locations)
    data["Wind (onshore), capacityMax"]= addEmptyRegions(data["Wind (onshore), capacityMax"], False ,locations)


    # Delete cluster_0 in Pumped hydro storage
    data["Pumped hydro storage, capacityFix"].drop('cluster_0', inplace=True)
    data["Pumped hydro storage, capacityFix"]= addEmptyRegions(data["Pumped hydro storage, capacityFix"], False ,locations)
    
    # # Delete cluster_0 in Hydrogen demand
    data["Hydrogen demand, operationRateFix"].drop('cluster_0', axis=1, inplace=True)
    data["Hydrogen demand, operationRateFix"]= addEmptyRegions(data["Hydrogen demand, operationRateFix"], False ,locations)
    
    # # Delete cluster_0 in DC cables
    print(data["DC cables, losses"])
    # # data["DC cables, losses"].drop('cluster_0', inplace=True)
    # # losses=data["DC cables, losses"],
    # # distances=data["DC cables, distances"],
    # # capacityFix=data["DC cables, capacityFix"],


    commodityUnitDict = {
        "electricity": r"GW$_{el}$",
        "CO2": r"Mio. t$_{CO_2}$/h",
        "hydrogen": r"GW$_{H_{2},LHV}$",
    }
    commodities = {"electricity", "hydrogen", "CO2"}

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=8760,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    # onshore wind
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (onshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (onshore), operationRateMax"],
            capacityMax=data["Wind (onshore), capacityMax"],
            investPerCapacity=1.1,
            opexPerCapacity=1.1 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    # CO2 from environment
    CO2_reductionTarget = 1
    esM.add(
        fn.Source(
            esM=esM,
            name="CO2 from enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    # Electrolyzers
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electroylzers",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # Pumped hydro storage
    esM.add(
        fn.Storage(
            esM=esM,
            name="Pumped hydro storage",
            commodity="electricity",
            chargeEfficiency=0.88,
            dischargeEfficiency=0.88,
            hasCapacityVariable=True,
            selfDischarge=1 - (1 - 0.00375) ** (1 / (30 * 24)),
            chargeRate=0.16,
            dischargeRate=0.12,
            capacityFix=data["Pumped hydro storage, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.000153,
        )
    )

    # DC cables
    esM.add(
        fn.Transmission(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            losses=data["DC cables, losses"],
            distances=data["DC cables, distances"],
            hasCapacityVariable=True,
            capacityFix=data["DC cables, capacityFix"],
        )
    )

    # Hydrogen sinks
    FCEV_penetration = 0.5
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=data["Hydrogen demand, operationRateFix"]
            * FCEV_penetration,
        )
    )

    esM.cluster(numberOfTypicalPeriods=3)

    esM.optimize(timeSeriesAggregation=True, solver = 'glpk')



