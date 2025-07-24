import fine as fn
import pandas as pd
import numpy as np
import copy


def create_simple_esm():
    """
    To observe the effects of variable conversion factors, we create a simple test
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
    cfs["electricity"] = pd.DataFrame([np.array([-1, -1, -1, -1])], index=locs).T
    # Use Series for conversion rate timeseries
    cfs["hydrogen"] = pd.Series(np.array([7, 0.7, 0.07, 0.007]))
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_VarConvFac",
            physicalUnit=r"kW$_{el}$",
            commodity="electricity",
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
    """
    According to the changes in the conversion factors, the electricity demand
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



def test_basecase(minimal_test_esM):
    """
    We test the minimal test system with constant conversion factor the get a reference.
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
    """
    We add an additional electrolyzer component with variable conversion rates.
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
        [np.array([-1, -1, -1, -1]), np.array([-1, -1, -1, -1])], index=locs
    ).T
    cfs["hydrogen"] = pd.DataFrame(
        [np.array([7.0, 0.7 , 0.7 , 0.07]), np.array([7, 0.7, 0.7, 0.07])], index=locs
    ).T

    # Add a new component with variable conversion rate to the EnergySystemModel.
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_VarConvFac",
            physicalUnit=r"kW$_{el}$",
            commodity="electricity",
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
    assertion_values_const = [0.0, 14600000.0, 0.0, 14600000.0]
    assertion_values_var = [4171428.6, 4171428.6, 0.0, 0.0]
    for t in range(0, 4):
        np.testing.assert_almost_equal(
            op_test_const[t], assertion_values_const[t], decimal=1
        )
        np.testing.assert_almost_equal(
            op_test_var[t], assertion_values_var[t], decimal=1
        )
