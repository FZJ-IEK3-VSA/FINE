"""
    Aim: To check if passing "lean data" is possible i.e., pass spatially resolved data only for 
    eligible locations 
    Tests:
        #For each component class: 
            Give data only for subset of locations. FINE should then look for locationalEligibility. 
                # If it is None, raises error saying that locationalEligibility is mandatory, in this case
                # If it is provided, simply fills data at missing locations with 0. 
                    A check is later made against locationalEligibility. 
"""

import sys
import os
import pytest
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


@pytest.mark.parametrize(
    "locationalEligibility",
    [
        None,
        pd.Series(
            {
                "cluster_0": 1,
                "cluster_1": 1,
                "cluster_2": 1,
                "cluster_3": 1,
                "cluster_4": 1,
                "cluster_5": 1,
                "cluster_6": 1,
                "cluster_7": 1,
            }
        ),
    ],
)
def test_leanModel_with_wrong_locationalEligibility(esM_init, locationalEligibility):
    """
    Case 1: subset of locations provided but no locationalEligibility
    Case 2: subset of locations provided with locationalEligibility,
            but they don't match
    """
    data = getData()

    esM = esM_init
    # Wind (onshore)
    # Delete operationRateMax and capacityMax data corresponding to cluster_0
    data["Wind (onshore), operationRateMax"].drop("cluster_0", axis=1, inplace=True)
    data["Wind (onshore), capacityMax"].drop("cluster_0", inplace=True)

    with pytest.raises(ValueError) as e_info:
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
                locationalEligibility=locationalEligibility,
            )
        )


def test_leanModel_with_matching_locationalEligibility(esM_init):
    """
    Case: subset of locations provided with matching locationalEligibility
    """
    data = getData()

    esM = esM_init

    locationalEligibility = pd.Series(
        {
            "cluster_0": 1,
            "cluster_1": 1,
            "cluster_2": 1,
            "cluster_3": 1,
            "cluster_4": 1,
            "cluster_5": 1,
            "cluster_6": 1,
            "cluster_7": 1,
        }
    )

    # 1. Wind (onshore)
    # Delete operationRateMax and capacityMax data corresponding to cluster_0
    data["Wind (onshore), operationRateMax"].drop("cluster_0", axis=1, inplace=True)
    data["Wind (onshore), capacityMax"].drop("cluster_0", inplace=True)

    _locationalEligibility = locationalEligibility.copy()
    _locationalEligibility.update({"cluster_0": 0})

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
            locationalEligibility=_locationalEligibility,
        )
    )

    # 2. Electrolyzers
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

    # 3. Pumped hydro storage
    # Delete capacityFix data corresponding to cluster_7
    data["Pumped hydro storage, capacityFix"].drop("cluster_7", inplace=True)

    _locationalEligibility = locationalEligibility.copy()
    _locationalEligibility.update({"cluster_7": 0})

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
            locationalEligibility=_locationalEligibility,
        )
    )

    # 4. DC cables
    # NOTE: looks like for transmission components, this is already handled
    ## pass pd.Series instead of pd.DataFrame to test if it works fine
    losses = {}

    for loc in esM.locations:
        _dict = data["DC cables, losses"][loc].to_dict()
        losses.update({f"{loc}_{k}": v for (k, v) in _dict.items()})

    losses = pd.Series(losses)
    losses = losses[losses > 0]

    esM.add(
        fn.Transmission(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            losses=losses,
            distances=data["DC cables, distances"],
            hasCapacityVariable=True,
            capacityFix=data["DC cables, capacityFix"],
        )
    )

    # 5. Hydrogen sinks
    # Delete operationRateFix data corresponding to cluster_3
    data["Hydrogen demand, operationRateFix"].drop("cluster_3", axis=1, inplace=True)

    _locationalEligibility = locationalEligibility.copy()
    _locationalEligibility.update({"cluster_3": 0})

    FCEV_penetration = 0.5
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=data["Hydrogen demand, operationRateFix"]
            * FCEV_penetration,
            locationalEligibility=_locationalEligibility,
        )
    )

    esM.aggregateTemporally(
        numberOfTypicalPeriods=3,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    esM.optimize(timeSeriesAggregation=True, solver="glpk")
