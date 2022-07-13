import pandas as pd
import FINE as fn
import numpy as np


def test_DSM(dsm_test_esM):
    """
    Given a one-node system with two generators, check whether the load and generation is shifted correctly in both
    directions with and without demand side management.
    """

    (
        esM_without,
        load_without_dsm,
        timestep_up,
        timestep_down,
        time_shift,
        cheap_capacity,
    ) = dsm_test_esM

    esM_without.add(
        fn.Sink(
            esM=esM_without,
            name="load",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=load_without_dsm,
        )
    )

    esM_without.optimize(timeSeriesAggregation=False, solver="glpk")  # without dsm

    generator_outputs = esM_without.componentModelingDict[
        "SourceSinkModel"
    ].operationVariablesOptimum

    # benchmark generation without dsm
    cheap_without_dsm = load_without_dsm.clip(0, cheap_capacity).copy()
    cheap_without_dsm.name = ('cheap', 'location')
    expensive_without_dsm = load_without_dsm - cheap_without_dsm
    expensive_without_dsm.name = ('expensive', 'location')

    # test without dsm
    print('generator_outputs')
    print(generator_outputs)
    pd.testing.assert_series_equal(generator_outputs.loc[('cheap', 'location')], cheap_without_dsm)
    pd.testing.assert_series_equal(generator_outputs.loc[('expensive', 'location')], expensive_without_dsm)

    # add DSM
    tFwd = 3
    tBwd = 3
    esM_with = dsm_test_esM[0]
    esM_with.removeComponent("load")
    shiftMax = 10

    esM_with.add(
        fn.DemandSideManagementBETA(
            esM=esM_with,
            name="flexible demand",
            commodity="electricity",
            hasCapacityVariable=False,
            tFwd=tFwd,
            tBwd=tBwd,
            operationRateFix=load_without_dsm,
            opexShift=1,
            shiftDownMax=shiftMax,
            shiftUpMax=shiftMax,
            socOffsetDown=-1,
            socOffsetUp=-1,
        )
    )

    esM_with.optimize(timeSeriesAggregation=False, solver="glpk")

    generator_outputs = esM_with.componentModelingDict[
        "SourceSinkModel"
    ].operationVariablesOptimum
    esM_load_with_DSM = esM_with.componentModelingDict[
        "DSMModel"
    ].operationVariablesOptimum

    # benchmark generation and load with dsm
    expensive_with_dsm = expensive_without_dsm.copy()
    expensive_with_dsm[timestep_up:timestep_up + time_shift] -= 10
    expensive_with_dsm[timestep_down - time_shift:timestep_down] -= 10
    expensive_with_dsm.name = ('expensive','location')

    cheap_with_dsm = cheap_without_dsm.copy()
    cheap_with_dsm[timestep_up - time_shift:timestep_up] += 10
    cheap_with_dsm[timestep_down:timestep_down + time_shift] += 10
    cheap_with_dsm.name = ('cheap', 'location')

    load_with_dsm = load_without_dsm.copy()
    load_with_dsm[timestep_up - time_shift:timestep_up] += 10
    load_with_dsm[timestep_up:timestep_up + time_shift] -= 10
    load_with_dsm[timestep_down - time_shift:timestep_down] -= 10
    load_with_dsm[timestep_down:timestep_down + time_shift] += 10
    load_with_dsm.name = ('flexible demand', 'location')

    # test with dsm
    pd.testing.assert_series_equal(generator_outputs.loc[('cheap',  'location')], cheap_with_dsm)
    pd.testing.assert_series_equal(generator_outputs.loc[('expensive',  'location')], expensive_with_dsm)
    pd.testing.assert_series_equal(esM_load_with_DSM.loc[('flexible demand',  'location')], load_with_dsm)

    esM_with.aggregateTemporally(numberOfTimeStepsPerPeriod=1, numberOfTypicalPeriods=25)    
    esM_with.optimize(timeSeriesAggregation=True, solver='gurobi', optimizationSpecs='LogToConsole=0')

    # benchmark generation and load with dsm
    expensive_with_dsm = expensive_without_dsm.copy()
    expensive_with_dsm[timestep_up:timestep_up + time_shift] -= 10
    expensive_with_dsm[timestep_down - time_shift:timestep_down] -= 10
    expensive_with_dsm.name = ('expensive', 'location')

    cheap_with_dsm = cheap_without_dsm.copy()
    cheap_with_dsm[timestep_up - time_shift:timestep_up] += 10
    cheap_with_dsm[timestep_down:timestep_down + time_shift] += 10
    cheap_with_dsm.name = ('cheap',  'location')

    load_with_dsm = load_without_dsm.copy()
    load_with_dsm[timestep_up - time_shift:timestep_up] += 10
    load_with_dsm[timestep_up:timestep_up + time_shift] -= 10
    load_with_dsm[timestep_down - time_shift:timestep_down] -= 10
    load_with_dsm[timestep_down:timestep_down + time_shift] += 10
    load_with_dsm.name = ('flexible demand',  'location')
    
    # test with dsm
    pd.testing.assert_series_equal(generator_outputs.loc[('cheap',  'location')], cheap_with_dsm)
    pd.testing.assert_series_equal(generator_outputs.loc[('expensive',  'location')], expensive_with_dsm)
    pd.testing.assert_series_equal(esM_load_with_DSM.loc[('flexible demand',  'location')], load_with_dsm)

    # test with dsm
    pd.testing.assert_series_equal(
        generator_outputs.loc[("cheap", "location")], cheap_with_dsm
    )
    pd.testing.assert_series_equal(
        generator_outputs.loc[("expensive", "location")], expensive_with_dsm
    )
    pd.testing.assert_series_equal(
        esM_load_with_DSM.loc[("flexible demand", "location")], load_with_dsm
    )

    esM_with.aggregateTemporally(
        numberOfTimeStepsPerPeriod=1, numberOfTypicalPeriods=25
    )
    esM_with.optimize(timeSeriesAggregation=True, solver="glpk")

    # benchmark generation and load with dsm
    expensive_with_dsm = expensive_without_dsm.copy()
    expensive_with_dsm[timestep_up:timestep_up + time_shift] -= 10
    expensive_with_dsm[timestep_down - time_shift:timestep_down] -= 10
    expensive_with_dsm.name = ('expensive', 'location')

    cheap_with_dsm = cheap_without_dsm.copy()
    cheap_with_dsm[timestep_up - time_shift:timestep_up] += 10
    cheap_with_dsm[timestep_down:timestep_down + time_shift] += 10
    cheap_with_dsm.name = ('cheap',  'location')

    load_with_dsm = load_without_dsm.copy()
    load_with_dsm[timestep_up - time_shift:timestep_up] += 10
    load_with_dsm[timestep_up:timestep_up + time_shift] -= 10
    load_with_dsm[timestep_down - time_shift:timestep_down] -= 10
    load_with_dsm[timestep_down:timestep_down + time_shift] += 10
    load_with_dsm.name = ('flexible demand',  'location')

    # test with dsm
    pd.testing.assert_series_equal(generator_outputs.loc[('cheap', 'location')], cheap_with_dsm)
    pd.testing.assert_series_equal(generator_outputs.loc[('expensive', 'location')], expensive_with_dsm)
    pd.testing.assert_series_equal(esM_load_with_DSM.loc[('flexible demand', 'location')], load_with_dsm)



def dsm_test_esM(scope="session"):
    """
    Generate a simple energy system model with one node, two fixed generators and one load time series
    for testing demand side management functionality.
    """
    # load without dsm
    now = pd.Timestamp.now().round("h")
    number_of_time_steps = 28
    # t_index = pd.date_range(now, now + pd.DateOffset(hours=number_of_timeSteps - 1), freq='h')
    t_index = range(number_of_time_steps)
    load_without_dsm = pd.Series([80.0] * number_of_time_steps, index=t_index)

    timestep_up = 10
    timestep_down = 20
    load_without_dsm[timestep_up:timestep_down] += 40.0

    time_shift = 3
    cheap_capacity = 100.0
    expensive_capacity = 20.0

    # set up energy model
    esM = fn.EnergySystemModel(
        locations={"location"},
        commodities={"electricity"},
        numberOfTimeSteps=number_of_time_steps,
        commodityUnitsDict={"electricity": r"MW$_{el}$"},
        hoursPerTimeStep=1,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="cheap",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(cheap_capacity, index=t_index),
            opexPerOperation=25,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="expensive",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(expensive_capacity, index=t_index),
            opexPerOperation=50,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="back-up",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(1000, index=t_index),
            opexPerOperation=1000,
        )
    )

    return esM, load_without_dsm, timestep_up, timestep_down, time_shift, cheap_capacity




if __name__ == "__main__":
    _test=dsm_test_esM()
    test_DSM(_test)