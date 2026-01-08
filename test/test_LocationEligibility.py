# function adjusted: utils.preprocess2dimData()

import fine as fn
import pandas as pd


def test_LocationEligibility():
    """Currently, the locational eligibility is automatically set if not given as input parameter.
    Transmission capacity min/max/fix or operation min/max/fix can be submitted as Dataframes or
    as Dataseries. These values are pre-processed in the preprocess2dimData function in Utils.
    During this pre-processing, if these values are in Dataframe format, only the locations which
    has a value > 0 are considered. But, if they are in Series format, all the values are considered
    including 0s. If locationEligibility is not set by the user in the beginning, it will be
    automatically set such that only the locations which has a value > 0 for capacity min/max/fix
    or operation min/max/fix are considered as eligible. Later, this locationEligibility is
    checked with the locations in the processed capacity min/max/fix or operation min/max/fix data
    and the process fails if they are not the same. This test make sure that whether capacity
    min/max/fix or operation min/max/fix are in Dataframe or Series format, the locationEligibility
    is set correctly.
    """
    esm = fn.EnergySystemModel(
        locations={"DE", "AT", "CH"},
        commodities={"energy"},
        commodityUnitsDict={"energy": "joule"},
    )

    operationRateMax = pd.DataFrame(index=range(8760))
    capacityMax = pd.Series()

    operationRateMax["DE_CH"] = 0
    operationRateMax["AT_CH"] = 0.5
    operationRateMax["AT_DE"] = 0.5
    operationRateMax["CH_DE"] = 0
    operationRateMax["CH_AT"] = 0.5
    operationRateMax["DE_AT"] = 0.5

    capacityMax["DE_CH"] = 0.5
    capacityMax["AT_CH"] = 0
    capacityMax["AT_DE"] = 0.5
    capacityMax["CH_DE"] = 0.5
    capacityMax["CH_AT"] = 0
    capacityMax["DE_AT"] = 0.5

    esm.add(
        fn.Transmission(
            esM=esm,
            name="transmission",
            commodity="energy",
            hasCapacityVariable=True,
            capacityFix=capacityMax,
            # operationRateMax=operationRateMax,
        )
    )


def test_TransmissionWithoutCapacityRestrictions():
    """Tests if Transmission components are initialized without error if
    just limited operation rate values are provided without specifiying capacity
    values in the case of location eligibility is not set at the begining.
    """
    esm = fn.EnergySystemModel(
        locations={"DE", "AT", "CH"},
        commodities={"energy"},
        commodityUnitsDict={"energy": "joule"},
    )

    operationRateMax = pd.DataFrame(index=range(8760))

    operationRateMax["DE_CH"] = 0.5
    operationRateMax["AT_CH"] = 0.5
    operationRateMax["AT_DE"] = 0.5
    operationRateMax["DE_AT"] = 0.5
    operationRateMax["CH_DE"] = 0.5
    operationRateMax["CH_AT"] = 0.5

    fn.Transmission(
        esM=esm,
        name="transmission",
        commodity="energy",
        operationRateMax=operationRateMax,
    )
