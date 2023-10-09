import os
import time

import pandas as pd
import numpy as np

import FINE as fn


def test_watersupply():
    # read in original results
    results = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "_testInputFiles",
            "waterSupplySystem_totalTransmission.csv",
        ),
        index_col=[0, 1, 2],
        header=None,
    ).squeeze("columns")

    # get parameters
    locations = [
        "House 1",
        "House 2",
        "House 3",
        "House 4",
        "House 5",
        "House 6",
        "Node 1",
        "Node 2",
        "Node 3",
        "Node 4",
        "Water treatment",
        "Water tank",
    ]
    commodityUnitDict = {"clean water": "U", "river water": "U"}
    commodities = {"clean water", "river water"}

    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities=commodities,
        numberOfTimeSteps=8760,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e3 Euro",
        lengthUnit="m",
    )

    starttime = time.time()

    # Source
    riverFlow = pd.DataFrame(np.zeros((8760, 12)), columns=locations)
    np.random.seed(42)
    riverFlow.loc[:, "Water treatment"] = np.random.uniform(0, 4, (8760)) + 8 * np.sin(
        np.pi * np.arange(8760) / 8760
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="River",
            commodity="river water",
            hasCapacityVariable=False,
            operationRateMax=riverFlow,
            opexPerOperation=0.05,
        )
    )

    # Conversion
    eligibility = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], index=locations)
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Water treatment plant",
            physicalUnit="U",
            commodityConversionFactors={"river water": -1, "clean water": 1},
            hasCapacityVariable=True,
            locationalEligibility=eligibility,
            investPerCapacity=7,
            opexPerCapacity=0.02 * 7,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    # Storage
    eligibility = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], index=locations)
    esM.add(
        fn.Storage(
            esM=esM,
            name="Water tank",
            commodity="clean water",
            hasCapacityVariable=True,
            chargeRate=1 / 24,
            dischargeRate=1 / 24,
            locationalEligibility=eligibility,
            investPerCapacity=0.10,
            opexPerCapacity=0.02 * 0.1,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    # Transmission
    distances = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 38, 40, 0, 105, 0, 0, 0, 0],
            [0, 0, 38, 40, 0, 0, 105, 0, 100, 0, 0, 0],
            [38, 40, 0, 0, 0, 0, 0, 100, 0, 30, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 20, 50],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0],
        ]
    )

    distances = pd.DataFrame(distances, index=locations, columns=locations)

    # Old water pipes
    capacityFix = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        ]
    )

    capacityFix = pd.DataFrame(capacityFix, index=locations, columns=locations)

    isBuiltFix = capacityFix.copy()
    isBuiltFix[isBuiltFix > 0] = 1

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Old water pipes",
            commodity="clean water",
            losses=0.1e-2,
            distances=distances,
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=True,
            bigM=100,
            capacityFix=capacityFix,
            isBuiltFix=isBuiltFix,
        )
    )

    # New water pipes
    incidence = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )

    eligibility = pd.DataFrame(incidence, index=locations, columns=locations)

    esM.add(
        fn.Transmission(
            esM=esM,
            name="New water pipes",
            commodity="clean water",
            losses=0.05e-2,
            distances=distances,
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=True,
            bigM=100,
            locationalEligibility=eligibility,
            investPerCapacity=0.1,
            investIfBuilt=0.5,
            interestRate=0.08,
            economicLifetime=50,
        )
    )

    # Sink
    winterHours = np.append(range(8520, 8760), range(1920))
    springHours, summerHours, autumnHours = (
        np.arange(1920, 4128),
        np.arange(4128, 6384),
        np.arange(6384, 8520),
    )

    demand = pd.DataFrame(np.zeros((8760, 12)), columns=list(locations))
    np.random.seed(42)
    demand[locations[0:6]] = np.random.uniform(0, 1, (8760, 6))

    demand.loc[winterHours[(winterHours % 24 < 5) | (winterHours % 24 >= 23)]] = 0
    demand.loc[springHours[(springHours % 24 < 4)]] = 0
    demand.loc[summerHours[(summerHours % 24 < 5) | (summerHours % 24 >= 23)]] = 0
    demand.loc[autumnHours[(autumnHours % 24 < 6) | (autumnHours % 24 >= 23)]] = 0
    esM.add(
        fn.Sink(
            esM=esM,
            name="Water demand",
            commodity="clean water",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # # Optimize the system
    esM.aggregateTemporally(
        numberOfTypicalPeriods=7,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )
    esM.optimize(timeSeriesAggregation=True, solver="glpk")

    # # Selected results output
    esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)

    # ### Storage
    esM.getOptimizationSummary("StorageModel", outputLevel=2)

    # ### Transmission
    esM.getOptimizationSummary("TransmissionModel", outputLevel=2)
    esM.componentModelingDict["TransmissionModel"].operationVariablesOptimum.sum(axis=1)

    #
    testresults = esM.componentModelingDict[
        "TransmissionModel"
    ].operationVariablesOptimum.sum(axis=1)

    print("Optimization took " + str(time.time() - starttime))

    # test if here solved fits with original results
    np.testing.assert_array_almost_equal(testresults.values, results.values, decimal=2)


if __name__ == "__main__":
    test_watersupply()
