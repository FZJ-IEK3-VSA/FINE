import FINE as fn
import pandas as pd
import numpy as np
import pytest


@pytest.mark.skip(reason="GPyOpt reached end of maintenance.")
def test_conversionPartLoad():
    # Set up energy system model instance
    locations = {"GlassProductionSite"}
    commodities = {"electricity", "heat", "hydrogen", "O2", "CO2", "rawMaterial"}
    commodityUnitDict = {
        "electricity": r"kW$_{el}$",
        "heat": r"kW$_{heat}$}",
        "hydrogen": r"kW$_{H2}$",
        "O2": r"kg$_{O_{2}}$/h",
        "CO2": r"kg$_{CO_{2}}$/h",
        "rawMaterial": r"kg$_{R}}$/h",
    }
    numberOfTimeSteps = 80
    hoursPerTimeStep = 0.25

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    ### Sources ###

    # Electricity from grid
    esM.add(
        fn.Source(
            esM=esM,
            name="ElectricityGrid",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=0.070,
        )
    )
    # Oxygen source from trailer
    esM.add(
        fn.Source(
            esM=esM,
            name="oxygenSource",
            commodity="O2",
            hasCapacityVariable=False,
            commodityCost=0.070,
        )
    )
    # Raw material for glass furnace
    esM.add(
        fn.Source(
            esM=esM,
            name="rawMaterialSource",
            commodity="rawMaterial",
            hasCapacityVariable=False,
            commodityCost=0.20,
        )
    )

    ### Conversion ###

    # PEM Electrolyzer
    # Get partLoadData from EC Campus Mainz
    Utilization = [
        0.0208023774145616,
        0.0222882615156017,
        0.0222882615156017,
        0.0222882615156017,
        0.025260029717682,
        0.025260029717682,
        0.0267459138187221,
        0.0282317979197622,
        0.0297176820208024,
        0.0312035661218424,
        0.0326894502228826,
        0.0341753343239227,
        0.0356612184249628,
        0.0371471025260029,
        0.038632986627043,
        0.0416047548291233,
        0.0430906389301634,
        0.0445765230312035,
        0.0475482912332838,
        0.0430906389301634,
        0.0490341753343239,
        0.050520059435364,
        0.0520059435364041,
        0.0549777117384844,
        0.0579494799405646,
        0.0609212481426448,
        0.0638930163447251,
        0.0683506686478455,
        0.0713224368499256,
        0.075780089153046,
        0.0787518573551263,
        0.0832095096582466,
        0.087667161961367,
        0.0921248142644873,
        0.0980683506686478,
        0.104011887072808,
        0.108469539375928,
        0.114413075780089,
        0.120356612184249,
        0.12778603268945,
        0.13521545319465,
        0.142644873699851,
        0.151560178306092,
        0.157503714710252,
        0.166419019316493,
        0.173848439821693,
        0.181277860326894,
        0.190193164933135,
        0.196136701337295,
        0.202080237741456,
        0.209509658246656,
        0.216939078751857,
        0.224368499257057,
        0.233283803863298,
        0.240713224368499,
        0.246656760772659,
        0.25408618127786,
        0.26151560178306,
        0.270430906389301,
        0.276374442793462,
        0.283803863298662,
        0.292719167904903,
        0.298662704309063,
        0.306092124814264,
        0.313521545319465,
        0.319465081723625,
        0.328380386329866,
        0.335809806835066,
        0.343239227340267,
        0.349182763744427,
        0.356612184249628,
        0.364041604754829,
        0.371471025260029,
        0.38038632986627,
        0.389301634472511,
        0.398216939078751,
        0.407132243684992,
        0.416047548291233,
        0.423476968796433,
        0.430906389301634,
        0.439821693907875,
        0.448736998514115,
        0.457652303120356,
        0.465081723625557,
        0.472511144130757,
        0.481426448736998,
        0.490341753343239,
        0.497771173848439,
        0.50668647845468,
        0.514115898959881,
        0.523031203566121,
        0.531946508172362,
        0.540861812778603,
        0.551263001485884,
        0.560178306092124,
        0.567607726597325,
        0.576523031203566,
        0.585438335809806,
        0.595839524517087,
        0.604754829123328,
        0.613670133729569,
        0.619613670133729,
        0.62852897473997,
        0.638930163447251,
        0.646359583952451,
        0.656760772659732,
        0.664190193164933,
        0.673105497771173,
        0.684992570579494,
        0.692421991084695,
        0.699851411589896,
        0.710252600297176,
        0.717682020802377,
        0.726597325408618,
        0.735512630014858,
        0.744427934621099,
        0.75334323922734,
        0.762258543833581,
        0.769687964338781,
        0.780089153046062,
        0.786032689450223,
        0.793462109955423,
        0.802377414561664,
        0.811292719167904,
        0.820208023774145,
        0.829123328380386,
        0.838038632986627,
        0.846953937592867,
        0.858841010401188,
        0.867756315007429,
        0.87667161961367,
        0.88707280832095,
        0.895988112927191,
        0.904903417533432,
        0.913818722139673,
        0.924219910846954,
        0.933135215453194,
        0.942050520059435,
        0.947994056463595,
        0.958395245170876,
        0.965824665676077,
        0.974739970282318,
        0.985141158989598,
        0.994056463595839,
    ]
    Efficiency = [
        0.03449362655834178,
        0.05655553999159559,
        0.04920156884717763,
        0.07616612971004356,
        0.09332539571368476,
        0.115387309146939,
        0.13254657515058135,
        0.1497058411542229,
        0.1693164308726712,
        0.1864756968763127,
        0.2036349628799551,
        0.2256968763132085,
        0.2404048186020449,
        0.25511276089088053,
        0.27472335060932884,
        0.2918826166129703,
        0.30904188261661275,
        0.33110379604986695,
        0.34581173833870255,
        0.3212985011906424,
        0.3629710043423449,
        0.38013027034598645,
        0.3923868889200161,
        0.4095461549236585,
        0.42425409721249496,
        0.4365107157865246,
        0.44876733436055427,
        0.45857262921977887,
        0.4683779240790026,
        0.4830858663678382,
        0.4953424849418686,
        0.5075991035158983,
        0.517404398375122,
        0.5247583695195407,
        0.5321123406639585,
        0.5419176355231823,
        0.5517229303824059,
        0.5639795489564365,
        0.5713335201008543,
        0.5811388149600779,
        0.5884927861044958,
        0.5933954335341085,
        0.5982980809637204,
        0.6007494046785262,
        0.6007494046785262,
        0.6007494046785262,
        0.5982980809637204,
        0.5982980809637204,
        0.5958467572489144,
        0.5958467572489144,
        0.5958467572489144,
        0.5933954335341085,
        0.5933954335341085,
        0.5933954335341085,
        0.5909441098193018,
        0.5909441098193018,
        0.5909441098193018,
        0.5884927861044958,
        0.5884927861044958,
        0.5884927861044958,
        0.5884927861044958,
        0.5884927861044958,
        0.5884927861044958,
        0.5860414623896899,
        0.5860414623896899,
        0.5860414623896899,
        0.583590138674884,
        0.583590138674884,
        0.583590138674884,
        0.5811388149600779,
        0.5811388149600779,
        0.5786874912452721,
        0.5762361675304661,
        0.5762361675304661,
        0.5737848438156602,
        0.5737848438156602,
        0.5713335201008543,
        0.5713335201008543,
        0.5688821963860483,
        0.5688821963860483,
        0.5664308726712425,
        0.5664308726712425,
        0.5639795489564365,
        0.5639795489564365,
        0.5615282252416306,
        0.5590769015268245,
        0.5590769015268245,
        0.5590769015268245,
        0.5566255778120178,
        0.5566255778120178,
        0.5541742540972119,
        0.5541742540972119,
        0.5517229303824059,
        0.5517229303824059,
        0.5517229303824059,
        0.5492716066676,
        0.5492716066676,
        0.5492716066676,
        0.5492716066676,
        0.5468202829527942,
        0.5443689592379882,
        0.5443689592379882,
        0.5443689592379882,
        0.5443689592379882,
        0.5419176355231823,
        0.5394663118083762,
        0.5394663118083762,
        0.5370149880935703,
        0.5370149880935703,
        0.5345636643787645,
        0.5345636643787645,
        0.5321123406639585,
        0.5321123406639585,
        0.5296610169491526,
        0.5296610169491526,
        0.5272096932343466,
        0.5272096932343466,
        0.5272096932343466,
        0.5247583695195399,
        0.5247583695195399,
        0.522307045804734,
        0.522307045804734,
        0.522307045804734,
        0.522307045804734,
        0.519855722089928,
        0.519855722089928,
        0.519855722089928,
        0.517404398375122,
        0.5149530746603161,
        0.5125017509455102,
        0.5125017509455102,
        0.5100504272307043,
        0.5100504272307043,
        0.5075991035158983,
        0.5051477798010924,
        0.5051477798010924,
        0.5051477798010924,
        0.5026964560862865,
        0.5026964560862865,
        0.5026964560862865,
        0.5026964560862865,
        0.5002451323714806,
        0.5002451323714806,
        0.49779380865667455,
    ]
    d = {"x": Utilization, "y": Efficiency}
    partLoadData = pd.DataFrame(d)
    partLoadData
    nSegments = 4
    esM.add(
        fn.ConversionPartLoad(
            esM=esM,
            name="PEMEC",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 1},
            commodityConversionFactorsPartLoad={
                "electricity": -1,
                "hydrogen": partLoadData,
            },
            nSegments=nSegments,
            hasCapacityVariable=True,
            bigM=99999,
            investPerCapacity=900,
            opexPerCapacity=900 * 0.01,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # Glass production - Hydrogen gas furnace
    capacityFix = 4985
    annualLoss = 0.03  # After one year a glass melting furnace needs 3% more energy to maintain process quality due to increasing thermal losses
    operationRateFix = pd.DataFrame(
        np.linspace(
            capacityFix * (1 - annualLoss / 2),
            capacityFix * (1 + annualLoss / 2),
            num=numberOfTimeSteps,
        ),
        columns=["GlassProductionSite"],
    )
    operationRateFix.mean()
    operationRateFix = operationRateFix / operationRateFix.mean()
    esM.add(
        fn.Conversion(
            esM=esM,
            name="hydrogenGasFurnace",
            physicalUnit=r"kW$_{H2}$",
            commodityConversionFactors={
                "hydrogen": -1,
                "electricity": -0.020,
                "O2": -0.137,  # stochiometric combustion: -0.238; lambda: -0.274
                "rawMaterial": -0.209,
                "heat": 0.070,
                "CO2": 0.020,
            },  # 'CO2':0.040 for H2 with german grid; 'CO2':0.021 for 100% renewable electricity
            hasCapacityVariable=True,
            capacityFix=capacityFix,
            operationRateFix=operationRateFix,
            investPerCapacity=1103.5,
            opexPerCapacity=652,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    ### Sinks ###

    # Heat output
    esM.add(
        fn.Sink(
            esM=esM, name="Heat output", commodity="heat", hasCapacityVariable=False
        )
    )
    # CO2 output
    esM.add(
        fn.Sink(esM=esM, name="CO2 output", commodity="CO2", hasCapacityVariable=False)
    )
    # O2 output
    # We include this sink to enable feasibility of the optimization problem in case the electrolyzer produces slightly more oxygen than required in the hydrogen furnace (round-off error < 1%) - we only that for stochiometric combustion -> for lambda > 1 that shouldn't be necessary
    esM.add(
        fn.Sink(esM=esM, name="O2 output", commodity="O2", hasCapacityVariable=False)
    )

    ### Optimization ###
    # Input parameters
    timeSeriesAggregation = False
    solver = "glpk"
    # Code
    esM.optimize(timeSeriesAggregation=timeSeriesAggregation, solver=solver)

    ### Test ###
    # Overall TAC
    srcSnkSummary = esM.getOptimizationSummary("SourceSinkModel", outputLevel=1)
    convSummary = esM.getOptimizationSummary("ConversionModel", outputLevel=1)
    convPartloadSummary = esM.getOptimizationSummary(
        "ConversionPartLoadModel", outputLevel=1
    )
    TAC = (
        srcSnkSummary.loc[
            ("ElectricityGrid", "TAC", "[1 Euro/a]"), "GlassProductionSite"
        ]
        + srcSnkSummary.loc[
            ("oxygenSource", "TAC", "[1 Euro/a]"), "GlassProductionSite"
        ]
        + srcSnkSummary.loc[
            ("rawMaterialSource", "TAC", "[1 Euro/a]"), "GlassProductionSite"
        ]
        + convSummary.loc[
            ("hydrogenGasFurnace", "TAC", "[1 Euro/a]"), "GlassProductionSite"
        ]
        + convPartloadSummary.loc[("PEMEC", "TAC", "[1 Euro/a]"), "GlassProductionSite"]
    )
    np.testing.assert_allclose(
        TAC, 14016197.2088, rtol=0.005
    )  # relative toerlance < 0.5%
    # Electricity TAC
    np.testing.assert_allclose(
        srcSnkSummary.loc[
            ("ElectricityGrid", "TAC", "[1 Euro/a]"), "GlassProductionSite"
        ],
        6.23855e06,
        rtol=0.01,
    )  # relative toerlance < 1%
    # PEMEC summary
    np.testing.assert_allclose(
        convPartloadSummary.loc[("PEMEC", "TAC", "[1 Euro/a]"), "GlassProductionSite"],
        1.46349e06,
        rtol=0.002,
    )  # relative toerlance < 0.2%
    np.testing.assert_allclose(
        convPartloadSummary.loc[
            ("PEMEC", "capacity", "[kW$_{el}$]"), "GlassProductionSite"
        ],
        10225.1729065,
        rtol=0.002,
    )  # relative toerlance < 0.2%
    np.testing.assert_allclose(
        convPartloadSummary.loc[
            ("PEMEC", "capexCap", "[1 Euro/a]"), "GlassProductionSite"
        ],
        1371467.06108539,
        rtol=0.002,
    )  # relative toerlance < 0.2%
    np.testing.assert_allclose(
        convPartloadSummary.loc[("PEMEC", "invest", "[1 Euro]"), "GlassProductionSite"],
        9202655.61585,
        rtol=0.002,
    )  # relative toerlance < 0.2%
    np.testing.assert_allclose(
        convPartloadSummary.loc[
            ("PEMEC", "opexCap", "[1 Euro/a]"), "GlassProductionSite"
        ],
        92026.5561585,
        rtol=0.002,
    )  # relative toerlance < 0.2%
    np.testing.assert_allclose(
        convPartloadSummary.loc[
            ("PEMEC", "operation", "[kW$_{el}$*h/a]"), "GlassProductionSite"
        ],
        8.82488e07,
        rtol=0.01,
    )  # relative toerlance < 1%
    # PEM operation results
    opVarOptPartLoad = esM.componentModelingDict[
        "ConversionPartLoadModel"
    ].operationVariablesOptimum.loc[("PEMEC", "GlassProductionSite")]
    opVarOptConstLoad = [
        2480.73776181,
        2481.6941601,
        2482.65055839,
        2483.60695668,
        2484.56335497,
        2485.51975325,
        2486.47615154,
        2487.43254983,
        2488.38894812,
        2489.34534641,
        2490.3017447,
        2491.25814299,
        2492.21454128,
        2493.17093957,
        2494.12733785,
        2495.08373614,
        2496.04013443,
        2496.99653272,
        2497.95293101,
        2498.9093293,
        2499.86572759,
        2500.82212588,
        2501.77852417,
        2502.73492245,
        2503.69132074,
        2504.64771903,
        2505.60411732,
        2506.56051561,
        2507.5169139,
        2508.47331219,
        2509.42971048,
        2510.38610877,
        2511.34250706,
        2512.29890534,
        2513.25530363,
        2514.21170192,
        2515.16810021,
        2516.1244985,
        2517.08089679,
        2518.03729508,
        2518.99369337,
        2519.95009166,
        2520.90648994,
        2521.86288823,
        2522.81928652,
        2523.77568481,
        2524.7320831,
        2525.68848139,
        2526.64487968,
        2527.60127797,
        2528.55767626,
        2529.51407455,
        2530.47047283,
        2531.42687112,
        2532.38326941,
        2533.3396677,
        2534.29606599,
        2535.25246428,
        2536.20886257,
        2537.16526086,
        2538.12165915,
        2539.07805743,
        2540.03445572,
        2540.99085401,
        2541.9472523,
        2542.90365059,
        2543.86004888,
        2544.81644717,
        2545.77284546,
        2546.72924375,
        2547.68564204,
        2548.64204032,
        2549.59843861,
        2550.5548369,
        2551.51123519,
        2552.46763348,
        2553.42403177,
        2554.38043006,
        2555.33682835,
        2556.29322664,
    ]
    np.testing.assert_allclose(opVarOptPartLoad, opVarOptConstLoad, rtol=0.01)


if __name__ == "__main__":
    test_conversionPartLoad()
