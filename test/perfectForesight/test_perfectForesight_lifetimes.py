import fine as fn
import numpy as np
import pandas as pd


def create_test_esM(techLifetime, economicLifetime, floorTechnicalLifetime):
    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
        },
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=2,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1e3,
            interestRate=0.02,
            economicLifetime=economicLifetime,
            technicalLifetime=techLifetime,
            floorTechnicalLifetime=floorTechnicalLifetime,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                index=[0, 1], columns=["PerfectLand"], data=4380
            ),
        )
    )
    return esM


def test_flooring_sameLifetimes():
    technicalLifetime = 12
    economicLifetime = 12
    floorTechnicalLifetime = True

    esM = create_test_esM(technicalLifetime, economicLifetime, floorTechnicalLifetime)
    esM.optimize()

    # objective value
    assert esM.pyM.Obj().round(0) == 1239

    # commissioning
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 0)] == 1
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 0
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 2)] == 1

    # optimization summary
    srcSnk_optSum_2020 = esM.getOptimizationSummary("SourceSinkModel", ip=2020)
    srcSnk_optSum_2025 = esM.getOptimizationSummary("SourceSinkModel", ip=2025)
    srcSnk_optSum_2030 = esM.getOptimizationSummary("SourceSinkModel", ip=2030)
    # invest costs
    _key = ("PV", "invest", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 1000
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 1000
    # scrapping bonus
    _key = ("PV", "revenueLifetimeShorteningResale", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key].round(0) == round(2 / 12 * 1000, 0)
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key].round(0) == round(2 / 12 * 1000, 0)
    # additional costs
    _key = ("PV", "investLifetimeExtension", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 0
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0


def test_ceiling_sameLifetimes():
    technicalLifetime = 12
    economicLifetime = 12
    floorTechnicalLifetime = False

    esM = create_test_esM(technicalLifetime, economicLifetime, floorTechnicalLifetime)
    esM.optimize()

    # objective value
    assert esM.pyM.Obj().round(0) == 1239

    # commissioning
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 0)] == 1
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 0
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 2)] == 0

    # optimization summary
    srcSnk_optSum_2020 = esM.getOptimizationSummary("SourceSinkModel", ip=2020)
    srcSnk_optSum_2025 = esM.getOptimizationSummary("SourceSinkModel", ip=2025)
    srcSnk_optSum_2030 = esM.getOptimizationSummary("SourceSinkModel", ip=2030)
    # invest costs
    _key = ("PV", "invest", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 1000
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0
    # scrapping bonus
    _key = ("PV", "revenueLifetimeShorteningResale", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 0
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0
    # additional costs
    _key = ("PV", "investLifetimeExtension", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key].round(0) == round(3 / 12 * 1000, 0)
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0


def test_flooring_sameInterval():
    technicalLifetime = 12
    economicLifetime = 11
    floorTechnicalLifetime = True

    esM = create_test_esM(technicalLifetime, economicLifetime, floorTechnicalLifetime)
    esM.optimize()

    # objective value
    assert esM.pyM.Obj().round(0) == 1339

    # commissioning
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 0)] == 1
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 0
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 2)] == 1

    # optimization summary
    srcSnk_optSum_2020 = esM.getOptimizationSummary("SourceSinkModel", ip=2020)
    srcSnk_optSum_2025 = esM.getOptimizationSummary("SourceSinkModel", ip=2025)
    srcSnk_optSum_2030 = esM.getOptimizationSummary("SourceSinkModel", ip=2030)
    # invest costs
    _key = ("PV", "invest", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 1000
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 1000
    # scrapping bonus
    _key = ("PV", "revenueLifetimeShorteningResale", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key].round(0) == round(1 / 11 * 1000, 0)
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key].round(0) == round(1 / 11 * 1000, 0)
    # additional costs
    _key = ("PV", "investLifetimeExtension", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 0
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0


def test_ceiling_sameInterval():
    technicalLifetime = 12
    economicLifetime = 11
    floorTechnicalLifetime = False

    esM = create_test_esM(technicalLifetime, economicLifetime, floorTechnicalLifetime)
    esM.optimize()

    # objective value
    assert esM.pyM.Obj().round(0) == 1257

    # commissioning
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 0)] == 1
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 0
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 2)] == 0

    # optimization summary
    srcSnk_optSum_2020 = esM.getOptimizationSummary("SourceSinkModel", ip=2020)
    srcSnk_optSum_2025 = esM.getOptimizationSummary("SourceSinkModel", ip=2025)
    srcSnk_optSum_2030 = esM.getOptimizationSummary("SourceSinkModel", ip=2030)
    # invest costs
    _key = ("PV", "invest", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 1000
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0
    # scrapping bonus
    _key = ("PV", "revenueLifetimeShorteningResale", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 0
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0
    # additional costs
    _key = ("PV", "investLifetimeExtension", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key].round(0) == round(3 / 11 * 1000, 0)
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0


def test_flooring_differentInterval():
    technicalLifetime = 12
    economicLifetime = 8
    floorTechnicalLifetime = True

    esM = create_test_esM(technicalLifetime, economicLifetime, floorTechnicalLifetime)
    esM.optimize()

    # objective value
    assert esM.pyM.Obj().round(0) == 1558

    # commissioning
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 0)] == 1
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 0
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 2)] == 1

    # optimization summary
    srcSnk_optSum_2020 = esM.getOptimizationSummary("SourceSinkModel", ip=2020)
    srcSnk_optSum_2025 = esM.getOptimizationSummary("SourceSinkModel", ip=2025)
    srcSnk_optSum_2030 = esM.getOptimizationSummary("SourceSinkModel", ip=2030)
    # invest costs
    _key = ("PV", "invest", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 1000
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 1000
    # scrapping bonus
    _key = ("PV", "revenueLifetimeShorteningResale", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 0
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0
    # additional costs
    _key = ("PV", "investLifetimeExtension", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 0
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0


def test_ceiling_differentInterval():
    technicalLifetime = 12
    economicLifetime = 8
    floorTechnicalLifetime = False

    esM = create_test_esM(technicalLifetime, economicLifetime, floorTechnicalLifetime)
    esM.optimize()

    # objective value
    assert esM.pyM.Obj().round(0) == 1337

    # commissioning
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 0)] == 1
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 0
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 2)] == 0

    # optimization summary
    srcSnk_optSum_2020 = esM.getOptimizationSummary("SourceSinkModel", ip=2020)
    srcSnk_optSum_2025 = esM.getOptimizationSummary("SourceSinkModel", ip=2025)
    srcSnk_optSum_2030 = esM.getOptimizationSummary("SourceSinkModel", ip=2030)
    # invest costs
    _key = ("PV", "invest", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 1000
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0
    # scrapping bonus
    _key = ("PV", "revenueLifetimeShorteningResale", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key] == 0
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0
    # additional costs
    _key = ("PV", "investLifetimeExtension", "[1 Euro]"), "PerfectLand"
    assert srcSnk_optSum_2020.loc[_key].round(0) == round(3 / 8 * 1000, 0)
    assert srcSnk_optSum_2025.loc[_key] == 0
    assert srcSnk_optSum_2030.loc[_key] == 0


def test_TAC_netPresentValueContributions():
    technicalLifetime = 12
    economicLifetime = 11
    floorTechnicalLifetime = False
    esM = create_test_esM(technicalLifetime, economicLifetime, floorTechnicalLifetime)
    esM.optimize()

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # the sum of all npv contributions in the optimization summary must equal
    # the objective value
    npv_sum_optSummary = 0
    for ip in esM.investmentPeriodNames:
        for mdl in esM.componentModelingDict.keys():
            optSum = esM.getOptimizationSummary(mdl, ip=ip)
            npv_sum_optSummary += optSum.loc[:, "NPVcontribution", :].sum().sum()

    np.testing.assert_almost_equal(esM.pyM.Obj(), npv_sum_optSummary)

    # the sum of discounted TAC must be equal to the NPV
    from fine.utils import annuityPresentValueFactor, discountFactor

    discounted_tac_sum = 0
    for ip in esM.investmentPeriodNames:
        srcSnk_optSummary = esM.getOptimizationSummary("SourceSinkModel", ip=ip)
        _tac = srcSnk_optSummary.loc[
            :, "TAC", :
        ].sum().sum() * annuityPresentValueFactor(
            esM, "PV", "PerfectLand", esM.investmentPeriodInterval
        )
        _ip = esM.investmentPeriodNames.index(ip)
        discounted_tac_sum += _tac * discountFactor(esM, _ip, "PV", "PerfectLand")

    np.testing.assert_almost_equal(esM.pyM.Obj(), discounted_tac_sum)
