import fine as fn
import numpy as np
import pandas as pd


def create_material_storage_esm(stateOfChargeBoundary):
    esM = fn.EnergySystemModel(
        locations={"loc"},
        commodities={"electricity"},
        materials={"steel"},
        commodityUnitsDict={"electricity": "kW"},
        materialUnitsDict={"steel": "t/h"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=8760,
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=1,
        startYear=2020,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )
    esM.add(
        fn.MaterialStorage(
            esM=esM,
            name="steel storage",
            commodity="steel",
            stateOfChargeBoundary=stateOfChargeBoundary,
            investPerCapacity=1,
            interestRate=0.08,
            economicLifetime=10,
            technicalLifetime=10,
        )
    )
    esM.declareOptimizationProblem(timeSeriesAggregation=False)
    return esM


def test_material_storage_non_cyclic_inter_investment_period_connection():
    esM = create_material_storage_esm("interInvestmentPeriodNotCyclic")

    constraints = esM.pyM.ConstrInterInvestmentPeriodMaterialStock_matStor
    initial_stock = esM.pyM.ConstrInitialMaterialStock_matStor

    assert len(constraints) == 2
    assert ("loc", "steel storage", 0) in constraints
    assert ("loc", "steel storage", 1) in constraints
    assert ("loc", "steel storage", 2) not in constraints
    assert len(initial_stock) == 1
    assert ("loc", "steel storage", 0) in initial_stock


def test_material_storage_cyclic_inter_investment_period_connection():
    esM = create_material_storage_esm("interInvestmentPeriod")

    constraints = esM.pyM.ConstrInterInvestmentPeriodMaterialStock_matStor
    initial_stock = esM.pyM.ConstrInitialMaterialStock_matStor

    assert len(constraints) == 3
    assert ("loc", "steel storage", 2) in constraints
    assert len(initial_stock) == 0


def create_material_storage_operation_esm(
    stateOfChargeBoundary="interInvestmentPeriodNotCyclic",
):
    """Create a two-IP system with a known inter-IP material transfer."""
    esM = fn.EnergySystemModel(
        locations={"loc"},
        commodities={"electricity"},
        materials={"steel"},
        commodityUnitsDict={"electricity": "kW"},
        materialUnitsDict={"steel": "t/h"},
        numberOfTimeSteps=4,
        hoursPerTimeStep=1,
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=1,
        startYear=2020,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )
    columns = ["loc"]
    index = range(4)
    esM.add(
        fn.Source(
            esM=esM,
            name="steel supply",
            commodity="steel",
            hasCapacityVariable=False,
            operationRateFix={
                2020: pd.DataFrame(2.0, index=index, columns=columns),
                2021: pd.DataFrame(0.0, index=index, columns=columns),
            },
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="steel demand",
            commodity="steel",
            hasCapacityVariable=False,
            operationRateFix={
                year: pd.DataFrame(1.0, index=index, columns=columns)
                for year in esM.investmentPeriodNames
            },
        )
    )
    esM.add(
        fn.MaterialStorage(
            esM=esM,
            name="steel storage",
            commodity="steel",
            stateOfChargeBoundary=stateOfChargeBoundary,
            investPerCapacity=1,
            interestRate=0.08,
            economicLifetime=10,
            technicalLifetime=10,
        )
    )
    return esM


def get_material_storage_results(esM):
    """Return the relevant material-storage results for all IPs."""
    results = {}
    for year in esM.investmentPeriodNames:
        summary = esM.getOptimizationSummary(
            "MaterialStorageModel", outputLevel=0, ip=year
        )
        results[year] = {
            prop: float(
                summary.xs(prop, level="Property")
                .loc["steel storage", "loc"]
                .sum()
            )
            for prop in [
                "capacity",
                "operationCharge",
                "operationDischarge",
                "stateOfChargeStart",
                "stateOfChargeEnd",
            ]
        }
    return results


def test_material_storage_tsa_matches_full_time_series():
    full_esM = create_material_storage_operation_esm()
    full_esM.optimize(solver="glpk")
    full_results = get_material_storage_results(full_esM)

    np.testing.assert_allclose(
        list(full_results[2020].values()), [4, 4, 0, 0, 4]
    )
    np.testing.assert_allclose(
        list(full_results[2021].values()), [4, 0, 4, 4, 0]
    )

    tsa_esM = create_material_storage_operation_esm()
    tsa_esM.aggregateTemporally(
        numberOfTypicalPeriods=1,
        numberOfTimeStepsPerPeriod=2,
        clusterMethod="hierarchical",
        sortValues=False,
        rescaleClusterPeriods=False,
        representationMethod=None,
    )
    tsa_esM.optimize(timeSeriesAggregation=True, solver="glpk")

    assert tsa_esM.periodOccurrences[0][0] == 2
    assert tsa_esM.periodOccurrences[1][0] == 2
    np.testing.assert_allclose(tsa_esM.pyM.Obj(), full_esM.pyM.Obj())
    assert get_material_storage_results(tsa_esM) == full_results


def test_material_storage_tsa_with_segmentation_matches_full_time_series():
    full_esM = create_material_storage_operation_esm()
    full_esM.optimize(solver="glpk")

    tsa_esM = create_material_storage_operation_esm()
    tsa_esM.aggregateTemporally(
        numberOfTypicalPeriods=1,
        numberOfTimeStepsPerPeriod=4,
        segmentation=True,
        numberOfSegmentsPerPeriod=1,
        clusterMethod="hierarchical",
        sortValues=False,
        rescaleClusterPeriods=False,
        representationMethod=None,
    )
    tsa_esM.optimize(timeSeriesAggregation=True, solver="glpk")

    full_results = get_material_storage_results(full_esM)
    tsa_results = get_material_storage_results(tsa_esM)
    for year in full_esM.investmentPeriodNames:
        np.testing.assert_allclose(
            list(tsa_results[year].values()),
            list(full_results[year].values()),
        )
    np.testing.assert_allclose(tsa_esM.pyM.Obj(), full_esM.pyM.Obj())


def test_material_storage_cyclic_inter_ip_connection_with_tsa():
    esM = create_material_storage_operation_esm("interInvestmentPeriod")
    esM.aggregateTemporally(
        numberOfTypicalPeriods=1,
        numberOfTimeStepsPerPeriod=2,
        clusterMethod="hierarchical",
        sortValues=False,
        rescaleClusterPeriods=False,
        representationMethod=None,
    )
    esM.declareOptimizationProblem(timeSeriesAggregation=True)

    constraints = esM.pyM.ConstrInterInvestmentPeriodMaterialStock_matStor
    initial_stock = esM.pyM.ConstrInitialMaterialStock_matStor

    assert len(constraints) == 2
    assert ("loc", "steel storage", 0) in constraints
    assert ("loc", "steel storage", 1) in constraints
    assert len(initial_stock) == 0
