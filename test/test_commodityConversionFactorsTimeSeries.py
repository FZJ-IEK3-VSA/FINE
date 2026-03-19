import fine as fn
import pandas as pd
import numpy as np
import copy
import fine.IOManagement.xarrayIO as xrIO
from pandas.testing import assert_frame_equal


def create_simple_esm():
    """To observe the effects of variable conversion factors, we create a simple test
    esm. It consists of a source, a conversion and a sink. The sink has a fixed
    demand. The conversion rate of the electrolyzer changes in every period. We use
    a pandas.DataFrame for the electricity conversion factors and a pandas.Series for
    the hydrogen conversion factors to test the different inputs.
    """
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    locs = ["ElectrolyzerLocation"]
    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )
    # Source
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
        )
    )
    # Sink
    demand = pd.Series(np.array([1.0, 1.0, 1.0, 1.0])) * hoursPerTimeStep
    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    cfs = {}
    # Use Dataframe for conversion rate timeseries
    cfs["electricity"] = pd.DataFrame([np.array([-0.1, -1, -10, -100])], index=locs).T
    # Use Series for conversion rate timeseries
    cfs["hydrogen"] = pd.Series(np.array([0.7, 0.7, 0.7, 0.7]))
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_VarConvFac",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": cfs["electricity"],
                "hydrogen": cfs["hydrogen"],
            },
            hasCapacityVariable=True,
            investPerCapacity=1000,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            capacityMax=1000,
            economicLifetime=10,
            locationalEligibility=pd.Series([1], ["ElectrolyzerLocation"]),
        )
    )
    return esM


def test_variable_conversion_simple_no_tsa():
    """According to the changes in the conversion factors, the electricity demand
    will be different in every timestep.
    """
    esM = create_simple_esm()

    # optimize
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    df = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum

    # Assert the optimal operation
    # We are asserting up to a precision of one decimal to account for precision gaps
    # of the solver.
    np.testing.assert_almost_equal(
        df.xs(("Electricity market", "ElectrolyzerLocation"))[0], 312.8, decimal=1
    )
    np.testing.assert_almost_equal(
        df.xs(("Electricity market", "ElectrolyzerLocation"))[1], 3128.5, decimal=1
    )
    np.testing.assert_almost_equal(
        df.xs(("Electricity market", "ElectrolyzerLocation"))[2], 31285.7, decimal=1
    )
    np.testing.assert_almost_equal(
        df.xs(("Electricity market", "ElectrolyzerLocation"))[3], 312857.1, decimal=1
    )


def test_variable_conversion_simple_with_tsa():
    """Test if the conversion time series are temporally clustered properly.
    Temporal clustering with 2 typical periods leads to two distinguished demand values
    instead of 4 in the case without temporally clustered timeseries.
    """
    esM = create_simple_esm()

    # Temporal clustering
    esM.aggregateTemporally(
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=1,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )
    # Optimization
    esM.optimize(timeSeriesAggregation=True, solver="glpk")

    df = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum

    # Assert the optimal operation
    # We are asserting up to a precision of one decimal to account for precision gaps
    # of the solver.
    np.testing.assert_almost_equal(
        df.xs(("Electricity market", "ElectrolyzerLocation"))[0], 11575.7, decimal=1
    )
    np.testing.assert_almost_equal(
        df.xs(("Electricity market", "ElectrolyzerLocation"))[1], 11575.7, decimal=1
    )
    np.testing.assert_almost_equal(
        df.xs(("Electricity market", "ElectrolyzerLocation"))[2], 11575.7, decimal=1
    )
    np.testing.assert_almost_equal(
        df.xs(("Electricity market", "ElectrolyzerLocation"))[3], 312857.1, decimal=1
    )


def test_basecase(minimal_test_esM):
    """We test the minimal test system with constant conversion factor the get a reference.
    Optimal operation of the electrolyzer component is determined by the electricity price.
    """
    # Get the minimal test system from conftest
    esM = copy.deepcopy(minimal_test_esM)

    # Optimize without TSA
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # Get the optimal electrolyzer operation
    op_test = []
    for t in range(0, 4):
        op_test.append(
            esM.componentModelingDict["ConversionModel"]
            .operationVariablesOptimum.xs("Electrolyzers")
            .loc["ElectrolyzerLocation", t]
        )

    # Assert the optimal operation
    # We are asserting up to a precision of one decimal to account for precision gaps
    # of the solver.
    np.testing.assert_almost_equal(op_test[0], 18771428.5, decimal=1)
    np.testing.assert_almost_equal(op_test[1], 37542857.1, decimal=1)
    np.testing.assert_almost_equal(op_test[2], 0.0, decimal=1)
    np.testing.assert_almost_equal(op_test[3], 18771428.5, decimal=1)


def test_variable_conversion_factor_no_tsa(minimal_test_esM):
    """We add an additional electrolyzer component with variable conversion rates.
    It has a very high efficiency in time-step 1, where it is now choosen to operate
    in favour of the electrolyzer with constant efficiency.
    Efficiency in the last time-step is very low for the new electolyzer, therefore
    it is not operated in this time-step.
    """
    # Get the minimal test system from conftest
    esM = copy.deepcopy(minimal_test_esM)

    # Create time-variable conversion rates for the two locations as pandas.DataFrame.
    locs = ["ElectrolyzerLocation", "IndustryLocation"]
    cfs = {}
    cfs["electricity"] = pd.DataFrame(
        [np.array([-0.1, -1, -1, -10]), np.array([-0.1, -1, -1, -10])], index=locs
    ).T
    cfs["hydrogen"] = pd.DataFrame(
        [np.array([0.7, 0.7, 0.7, 0.7]), np.array([0.7, 0.7, 0.7, 0.7])], index=locs
    ).T

    # Add a new component with variable conversion rate to the EnergySystemModel.
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_VarConvFac",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": cfs["electricity"],
                "hydrogen": cfs["hydrogen"],
            },
            hasCapacityVariable=True,
            investPerCapacity=1000,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # Optimize the esM without TSA.
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # Get optimal electrolyzer operations
    op_test_const = []
    op_test_var = []
    for t in range(0, 4):
        op_test_const.append(
            esM.componentModelingDict["ConversionModel"]
            .operationVariablesOptimum.xs("Electrolyzers")
            .loc["ElectrolyzerLocation", t]
        )
        op_test_var.append(
            esM.componentModelingDict["ConversionModel"]
            .operationVariablesOptimum.xs("Electrolyzers_VarConvFac")
            .loc["ElectrolyzerLocation", t]
        )

    # Assert the optimal operation
    # We are asserting up to a precision of one decimal to account for precision gaps
    # of the solver.
    assertion_values_const = [0.0, 18771428.5, 0.0, 18771428.5]
    assertion_values_var = [18771428.5, 18771428.5, 0.0, 0.0]
    for t in range(0, 4):
        np.testing.assert_almost_equal(
            op_test_const[t], assertion_values_const[t], decimal=1
        )
        np.testing.assert_almost_equal(
            op_test_var[t], assertion_values_var[t], decimal=1
        )


def test_variable_conversion_factor_with_tsa(minimal_test_esM):
    """Same as `test_variable_conversion_factor_no_tsa` but with time series aggregation
    using 3 typical periods. Now the optimal solution is composed of only three different
    periods.
    """
    # Get the minimal test system from conftest
    esM = copy.deepcopy(minimal_test_esM)

    # Create time-variable conversion rates for the two locations as pandas.DataFrame.
    locs = ["ElectrolyzerLocation", "IndustryLocation"]
    cfs = {}
    cfs["electricity"] = pd.DataFrame(
        [np.array([-0.1, -1, -1, -10]), np.array([-0.1, -1, -1, -10])], index=locs
    ).T
    cfs["hydrogen"] = pd.DataFrame(
        [np.array([0.7, 0.7, 0.7, 0.7]), np.array([0.7, 0.7, 0.7, 0.7])], index=locs
    ).T

    # Add a new component with variable conversion rate to the EnergySystemModel.
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_VarConvFac",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": cfs["electricity"],
                "hydrogen": cfs["hydrogen"],
            },
            hasCapacityVariable=True,
            investPerCapacity=1000,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    esM.aggregateTemporally(
        numberOfTypicalPeriods=3,
        numberOfTimeStepsPerPeriod=1,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    esM.optimize(timeSeriesAggregation=True, solver="glpk")

    # Get optimal electrolyzer operations
    op_test_const = []
    op_test_var = []
    for t in range(0, 4):
        op_test_const.append(
            esM.componentModelingDict["ConversionModel"]
            .operationVariablesOptimum.xs("Electrolyzers")
            .loc["ElectrolyzerLocation", t]
        )
        op_test_var.append(
            esM.componentModelingDict["ConversionModel"]
            .operationVariablesOptimum.xs("Electrolyzers_VarConvFac")
            .loc["ElectrolyzerLocation", t]
        )

    # Assert the optimal operation
    # Asserting up to a precision of one decimal to account for precision gaps of the solver.
    assertion_values_const = [0.0, 9385714.2, 0.0, 9385714.2]
    assertion_values_var = [18771428.5, 18771428.5, 18771428.5, 0.0]
    for t in range(0, 4):
        np.testing.assert_almost_equal(
            op_test_const[t], assertion_values_const[t], decimal=1
        )
        np.testing.assert_almost_equal(
            op_test_var[t], assertion_values_var[t], decimal=1
        )


def test_variable_conversion_export_to_xarray():
    esM = create_simple_esm()

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_VarConvFac_Export",
            physicalUnit=r"kW$_{H_{2},LHV}$",
            commodityConversionFactors={
                0: {
                    "electricity": pd.DataFrame(
                        [np.array([-0.1, -1, -10, -100])],
                        index=["ElectrolyzerLocation"],
                    ).T,
                    "hydrogen": 1,
                }
            },
        )
    )

    esM_copy = copy.deepcopy(esM)
    xrds = xrIO.convertOptimizationInputToDatasets(esM_copy)
    input_ds = xrds["Input"]["Conversion"]["Electrolyzers_VarConvFac_Export"]

    # === Check exported electricity DataFrame ===
    expected_df = pd.DataFrame(
        [np.array([-0.1, -1, -10, -100])], index=["ElectrolyzerLocation"]
    ).T

    series = input_ds["ts_commodityConversionFactors.0.electricity"].to_pandas()
    actual_df = series.to_frame(name="ElectrolyzerLocation")

    # Normalize index/column names
    number_of_index_level_expected = expected_df.index.nlevels
    number_of_index_level_actual = actual_df.index.nlevels
    expected_df.index.set_names(
        names=[None] * number_of_index_level_expected, inplace=True
    )
    actual_df.index.set_names(names=[None] * number_of_index_level_actual, inplace=True)
    expected_df.columns.set_names(
        names=[None] * number_of_index_level_expected, inplace=True
    )
    actual_df.columns.set_names(
        names=[None] * number_of_index_level_actual, inplace=True
    )

    assert_frame_equal(
        actual_df.sort_index(), expected_df.sort_index(), check_dtype=False
    )

    # === Check exported hydrogen scalar ===
    hydrogen_val = input_ds["0d_commodityConversionFactors.0.hydrogen"].item()

    assert hydrogen_val == 1


def test_location_specific_timeseries_conversion_factors_dataframe():
    """Location-specific time series conversion factors via DataFrame (4 timesteps)."""
    # --- Build ESM ---
    numberOfTimeSteps = 4
    hoursPerTimeStep = 1

    esM = fn.EnergySystemModel(
        locations={"Loc1", "Loc2"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
        )
    )

    # --- Hydrogen demand  ---
    demand = pd.DataFrame(
        {
            "Loc1": [10.0, 10.0, 10.0, 10.0],
            "Loc2": [10.0, 10.0, 10.0, 10.0],
        },
        index=esM.totalTimeSteps,
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # --- Time- & location-specific conversion factors ---
    # Efficiency varies over time AND location
    ccf_h2 = pd.DataFrame(
        {
            "Loc1": [0.5, 0.6, 0.7, 0.8],
            "Loc2": [1.0, 0.9, 0.8, 0.7],
        },
        index=esM.totalTimeSteps,
        dtype="float64",
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_LocTsCcf",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": -1,
                "hydrogen": ccf_h2,
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # --- Check processed structure ---
    comp = esM.getComponent("Electrolyzers_LocTsCcf")
    full = comp.fullCommodityConversionFactors[0]["hydrogen"]

    assert isinstance(full, pd.DataFrame)

    # Check entries
    np.testing.assert_almost_equal(full.at[(0, 0), "Loc1"], 0.5)
    np.testing.assert_almost_equal(full.at[(0, 3), "Loc1"], 0.8)
    np.testing.assert_almost_equal(full.at[(0, 1), "Loc2"], 0.9)

    # --- Optimize ---
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    op = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs(
        "Electrolyzers_LocTsCcf"
    )

    # --- Expected electricity consumption ---
    # electricity = hydrogen_demand / efficiency

    expected_loc1 = [
        10 / 0.5,  # 20
        10 / 0.6,  # 16.6667
        10 / 0.7,  # 14.2857
        10 / 0.8,  # 12.5
    ]

    expected_loc2 = [
        10 / 1.0,  # 10
        10 / 0.9,  # 11.1111
        10 / 0.8,  # 12.5
        10 / 0.7,  # 14.2857
    ]

    for t in range(4):
        np.testing.assert_almost_equal(op.loc["Loc1", t], expected_loc1[t], decimal=5)
        np.testing.assert_almost_equal(op.loc["Loc2", t], expected_loc2[t], decimal=5)


def create_two_loc_esm_4ts_for_tsa():
    """Two locations, 4 timesteps, one source, one conversion, one sink with fixed demand."""
    numberOfTimeSteps = 4
    hoursPerTimeStep = 1

    esM = fn.EnergySystemModel(
        locations={"Loc1", "Loc2"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
        )
    )

    demand = pd.DataFrame(
        {"Loc1": [10.0, 10.0, 10.0, 10.0], "Loc2": [10.0, 10.0, 10.0, 10.0]},
        index=esM.totalTimeSteps,
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    return esM


def test_location_specific_timeseries_conversion_factors_with_tsa():
    """Location-specific time series conversion factors (DataFrame t x loc)
    must work with TSA (aggregation + optimize).
    We orient on existing TSA tests: aggregateTemporally + optimize(timeSeriesAggregation=True).
    """
    esM = create_two_loc_esm_4ts_for_tsa()

    # Location-specific + time-varying efficiency
    ccf_h2 = pd.DataFrame(
        {
            "Loc1": [0.5, 0.5, 1.0, 1.0],
            "Loc2": [0.6, 0.6, 0.9, 0.9],
        },
        index=esM.totalTimeSteps,
        dtype="float64",
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_LocTsCcf",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": -1,
                "hydrogen": ccf_h2,
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # --- TSA aggregation ---
    esM.aggregateTemporally(
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=1,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    # --- Optimize with TSA ---
    esM.optimize(timeSeriesAggregation=True, solver="glpk")

    # Get electricity market operation (this is total electricity supplied)
    df_src = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum

    # Electricity consumption should show 2 distinct levels (clustered),
    # replicated across timesteps depending on clustering / rescaling.
    # We assert the *set* of values rather than exact placement (robust to cluster ordering).
    elec = df_src.xs(("Electricity market", "Loc1")).values  # 4 entries
    unique_vals = np.unique(np.round(elec.astype(float), 6))

    # With 2 typical periods and our constructed factors, we expect only 2 unique electricity levels.
    assert len(unique_vals) == 2

    # Loc1: demand=10, eff either 0.5 => 20, or 1.0 => 10
    # TSA rescaling can replicate those levels; values should be near 10 and 20.
    assert np.isclose(unique_vals, 10.0, atol=1e-6).any()
    assert np.isclose(unique_vals, 20.0, atol=1e-6).any()


def test_location_specific_timeseries_conversion_factors_dataframe_pf():
    """Location- and time-specific conversion factors for multiple IPs (Perfect Foresight)."""
    numberOfTimeSteps = 4
    hoursPerTimeStep = 4
    numberOfInvestmentPeriods = 5
    investmentPeriodInterval = 5

    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation", "IndustryLocation"},
        commodities={"electricity", "hydrogen"},
        startYear=2025,
        numberOfInvestmentPeriods=numberOfInvestmentPeriods,
        investmentPeriodInterval=investmentPeriodInterval,
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
        balanceLimit=None,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
        )
    )

    demand = pd.DataFrame(
        {
            "ElectrolyzerLocation": [10.0, 10.0, 10.0, 10.0],
            "IndustryLocation": [10.0, 10.0, 10.0, 10.0],
        },
        index=esM.totalTimeSteps,
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # --- build per-IP DataFrames (time x location) ---
    df_2025 = pd.DataFrame(
        {
            "ElectrolyzerLocation": [0.5, 0.6, 0.7, 0.8],
            "IndustryLocation": [1.0, 0.9, 0.8, 0.7],
        },
        index=esM.totalTimeSteps,
        dtype="float64",
    )

    df_2030 = pd.DataFrame(
        {
            "ElectrolyzerLocation": [0.6, 0.7, 0.8, 0.9],
            "IndustryLocation": [0.9, 0.8, 0.7, 0.6],
        },
        index=esM.totalTimeSteps,
        dtype="float64",
    )

    df_2035 = pd.DataFrame(
        {
            "ElectrolyzerLocation": [0.7, 0.8, 0.9, 1.0],
            "IndustryLocation": [0.8, 0.7, 0.6, 0.5],
        },
        index=esM.totalTimeSteps,
        dtype="float64",
    )

    df_2040 = pd.DataFrame(
        {
            "ElectrolyzerLocation": [0.8, 0.9, 1.0, 1.1],
            "IndustryLocation": [0.7, 0.6, 0.5, 0.4],
        },
        index=esM.totalTimeSteps,
        dtype="float64",
    )

    df_2045 = pd.DataFrame(
        {
            "ElectrolyzerLocation": [0.9, 1.0, 1.1, 1.2],
            "IndustryLocation": [0.6, 0.5, 0.4, 0.3],
        },
        index=esM.totalTimeSteps,
        dtype="float64",
    )

    # --- correct ip-dependent commodityConversionFactors structure ---
    # top-level: investment period (YEAR) -> dict of commodity -> factor
    ccf_by_ip = {
        2025: {"electricity": -1, "hydrogen": df_2025},
        2030: {"electricity": -1, "hydrogen": df_2030},
        2035: {"electricity": -1, "hydrogen": df_2035},
        2040: {"electricity": -1, "hydrogen": df_2040},
        2045: {"electricity": -1, "hydrogen": df_2045},
    }

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_LocTsCcf",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors=ccf_by_ip,
            hasCapacityVariable=True,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    comp = esM.getComponent("Electrolyzers_LocTsCcf")
    full_ip0 = comp.fullCommodityConversionFactors[0]["hydrogen"]  # IP 0
    assert isinstance(full_ip0, pd.DataFrame)
    np.testing.assert_almost_equal(full_ip0.at[(0, 0), "ElectrolyzerLocation"], 0.5)
    np.testing.assert_almost_equal(full_ip0.at[(0, 3), "ElectrolyzerLocation"], 0.8)
    np.testing.assert_almost_equal(full_ip0.at[(0, 1), "IndustryLocation"], 0.9)

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    op_dict = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum

    # pick the first IP (should be 2025 in your setup)
    op_ip0 = op_dict[esM.investmentPeriodNames[0]].xs("Electrolyzers_LocTsCcf")
    # or explicitly:
    # op_ip0 = op_dict[2025].xs("Electrolyzers_LocTsCcf")

    expected_electrolyzer = [10 / 0.5, 10 / 0.6, 10 / 0.7, 10 / 0.8]
    expected_industry = [10 / 1.0, 10 / 0.9, 10 / 0.8, 10 / 0.7]

    for t in range(numberOfTimeSteps):
        np.testing.assert_almost_equal(
            op_ip0.loc["ElectrolyzerLocation", t], expected_electrolyzer[t], decimal=5
        )
        np.testing.assert_almost_equal(
            op_ip0.loc["IndustryLocation", t], expected_industry[t], decimal=5
        )

    # --- PF: operationVariablesOptimum is a dict keyed by investment period (years) ---
    op_dict = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum

    # Map each IP year to its expected efficiency dataframe
    eff_by_ip = {
        2025: df_2025,
        2030: df_2030,
        2035: df_2035,
        2040: df_2040,
        2045: df_2045,
    }

    for ip_year, eff_df in eff_by_ip.items():
        # Get operation result for this IP and this component
        op_ip = op_dict[ip_year].xs("Electrolyzers_LocTsCcf")

        # Check all locations and timesteps
        for loc in ["ElectrolyzerLocation", "IndustryLocation"]:
            for t in range(numberOfTimeSteps):
                eta = float(eff_df.loc[esM.totalTimeSteps[t], loc])
                expected = 10.0 / eta  # since demand is fixed at 10 each timestep

                np.testing.assert_almost_equal(
                    op_ip.loc[loc, t],
                    expected,
                    decimal=5,
                    err_msg=f"Mismatch in IP={ip_year}, loc={loc}, t={t}, eta={eta}",
                )


def test_location_specific_constant_conversion_factors_series():
    """Time-independent but location-dependent conversion factors via pd.Series."""
    esM = create_two_loc_esm_4ts_for_tsa()  # Loc1/Loc2, 4 timesteps, demand=10 each

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_LocCcf_Const",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": -1,
                "hydrogen": pd.Series({"Loc1": 0.5, "Loc2": 1.0}),
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # Check processed structure (should stay a location-indexed Series)
    comp = esM.getComponent("Electrolyzers_LocCcf_Const")
    processed = comp.processedCommodityConversionFactors[0]["hydrogen"]
    assert isinstance(processed, pd.Series)
    np.testing.assert_almost_equal(processed.loc["Loc1"], 0.5)
    np.testing.assert_almost_equal(processed.loc["Loc2"], 1.0)

    # Optimize (no TSA)
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    op = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs(
        "Electrolyzers_LocCcf_Const"
    )

    # Demand is 10 each timestep per location
    # operation = demand / efficiency
    for t in range(4):
        np.testing.assert_almost_equal(op.loc["Loc1", t], 10.0 / 0.5, decimal=6)  # 20
        np.testing.assert_almost_equal(op.loc["Loc2", t], 10.0 / 1.0, decimal=6)  # 10

