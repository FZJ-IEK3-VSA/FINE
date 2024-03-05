import pandas as pd

import fine as fn


def test_commissioningMinMaxFix(perfectForesight_test_esM):
    esM = perfectForesight_test_esM

    commissioningMax_PV_cheap = {2020: 0.5, 2025: 0, 2030: 1, 2035: 0, 2040: 0}
    esM.add(
        fn.Source(
            esM=esM,
            name="PV_cheap",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1e2,
            interestRate=0.02,
            commissioningMax=commissioningMax_PV_cheap,
        )
    )

    commissioningMin_PV_expensive = {2020: 0.5, 2025: 1, 2030: 0, 2035: 0, 2040: 0}
    esM.add(
        fn.Source(
            esM=esM,
            name="PV_expensive",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1e4,
            interestRate=0.02,
            commissioningMin=commissioningMin_PV_expensive,
        )
    )

    commissioningFix_H2_Source = {2020: 0.5, 2025: 1, 2030: 0, 2035: 0, 2040: 0}
    esM.add(
        fn.Source(
            esM=esM,
            name="H2_Source",
            commodity="hydrogen",
            hasCapacityVariable=True,
            investPerCapacity=1e4,
            interestRate=0.02,
            commissioningFix=commissioningFix_H2_Source,
        )
    )

    commissioningMin_Pipelines = pd.DataFrame(
        index=list(esM.locations), columns=list(esM.locations), data=[[0, 1], [1, 0]]
    )

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines",
            commodity="hydrogen",
            hasCapacityVariable=True,
            investPerCapacity=0.177,
            interestRate=0.08,
            economicLifetime=40,
            commissioningMin=commissioningMin_Pipelines,
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver="glpk")
    commissioning_Pipe = esM.getOptimizationSummary("TransmissionModel", ip=2020).loc[
        "Pipelines", "commissioning", "[kW$_{H_{2},LHV}$]", "ForesightLand"
    ]["PerfectLand"]
    assert commissioning_Pipe >= 1
    commissioning_PV_cheap = {
        ip: esM.getOptimizationSummary("SourceSinkModel", ip=ip).loc[
            "PV_cheap", "commissioning", "[kW$_{el}$]"
        ]["PerfectLand"]
        for ip in esM.investmentPeriodNames
    }
    commissioning_PV_expensive = {
        ip: esM.getOptimizationSummary("SourceSinkModel", ip=ip).loc[
            "PV_expensive", "commissioning", "[kW$_{el}$]"
        ]["PerfectLand"]
        for ip in esM.investmentPeriodNames
    }
    commissioning_H2_Source = {
        ip: esM.getOptimizationSummary("SourceSinkModel", ip=ip).loc[
            "H2_Source", "commissioning", "[kW$_{H_{2},LHV}$]"
        ]["PerfectLand"]
        for ip in esM.investmentPeriodNames
    }

    for ip in esM.investmentPeriodNames:
        assert commissioningMax_PV_cheap[ip] >= commissioning_PV_cheap[ip]
        assert commissioningMin_PV_expensive[ip] <= commissioning_PV_expensive[ip]
        assert commissioningFix_H2_Source[ip] == commissioning_H2_Source[ip]
