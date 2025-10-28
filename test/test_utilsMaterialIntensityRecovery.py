import pandas as pd
import pytest
import fine as fn
from fine import utils


def test_valid_material_intensity_inputs():
    esM = fn.EnergySystemModel(
        locations={"A"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW"},
        materials={"steel", "copper"},
        materialUnitsDict={"steel": "tons", "copper": "tons"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=8760,
        costUnit="1e6 Euro",
        lengthUnit="km",
    )

    # valid input
    comp = fn.Source(
        esM=esM,
        name="Wind (onshore)",
        commodity="electricity",
        hasCapacityVariable=True,
        materialIntensity={
            "A": {"steel": pd.Series({0: 2.0}), "copper": pd.Series({0: 1.0})}
        },
    )
    esM.add(comp)

    processedMaterialIntensity = utils.checkAndSetMaterialIntensity(
        esM, comp.materialIntensity, esM.locations, esM.investmentPeriods
    )

    assert processedMaterialIntensity["A"]["steel"].equals(pd.Series({0: 2.0}))
    assert processedMaterialIntensity["A"]["copper"].equals(pd.Series({0: 1.0}))

    # invalid location
    with pytest.raises(KeyError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialIntensity={
                    "B": {
                        "steel": pd.Series({0: 1.0}),
                        "copper": pd.Series({0: 1.0}),
                    }
                },
            )
        )

    # misaligned year
    with pytest.raises(ValueError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialIntensity={
                    "A": {
                        "steel": pd.Series({1: 1.0}),
                        "copper": pd.Series({0: 1.0}),
                    }
                },
            )
        )

    # missing year
    with pytest.raises(ValueError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialIntensity={
                    "A": {
                        "steel": pd.Series({2: 1.0}),
                        "copper": pd.Series({2: 1.0}),
                    }
                },
            )
        )

    # negative value
    with pytest.raises(ValueError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialIntensity={
                    "A": {
                        "steel": pd.Series({0: -1.0}),
                        "copper": pd.Series({0: 0.5}),
                    }
                },
            )
        )

    # no panas series
    with pytest.raises(TypeError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialIntensity={
                    "A": {
                        "steel": {0: 1.0},
                        "copper": {0: 0.5},
                    }
                },
            )
        )


def test_valid_material_recovery_inputs():
    esM = fn.EnergySystemModel(
        locations={"A"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW"},
        materials={"steel", "copper"},
        materialUnitsDict={"steel": "tons", "copper": "tons"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=8760,
        costUnit="1e6 Euro",
        lengthUnit="km",
    )

    # valid input
    comp = fn.Source(
        esM=esM,
        name="Wind (onshore)",
        commodity="electricity",
        hasCapacityVariable=True,
        materialCollection={
            "A": {"steel": pd.Series({0: 0.7}), "copper": pd.Series({0: 0.8})}
        },
    )
    esM.add(comp)

    processedMaterialCollection = utils.checkAndSetMaterialCollection(
        esM, comp.materialCollection, esM.locations, esM.investmentPeriods
    )

    assert processedMaterialCollection["A"]["steel"].equals(pd.Series({0: 0.7}))
    assert processedMaterialCollection["A"]["copper"].equals(pd.Series({0: 0.8}))

    # invalid location
    with pytest.raises(KeyError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialCollection={
                    "B": {
                        "steel": pd.Series({0: 0.7}),
                        "copper": pd.Series({0: 0.8}),
                    }
                },
            )
        )

    # misaligned year
    with pytest.raises(ValueError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialCollection={
                    "A": {
                        "steel": pd.Series({-1: 0.7}),
                        "copper": pd.Series({0: 0.8}),
                    }
                },
            )
        )

    # missing year
    with pytest.raises(ValueError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialCollection={
                    "A": {
                        "steel": pd.Series({2: 0.7}),
                        "copper": pd.Series({2: 0.8}),
                    }
                },
            )
        )

    # value not in range of 0 to 1
    with pytest.raises(ValueError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialCollection={
                    "A": {
                        "steel": pd.Series({0: 1.8}),
                        "copper": pd.Series({0: 0.5}),
                    }
                },
            )
        )

    # no panas series
    with pytest.raises(TypeError):
        esM.add(
            fn.Source(
                esM=esM,
                name="Wind (onshore)",
                commodity="electricity",
                hasCapacityVariable=True,
                materialCollection={
                    "A": {
                        "steel": {0: 1.0},
                        "copper": {0: 0.5},
                    }
                },
            )
        )
