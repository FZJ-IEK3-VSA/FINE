import FINE as fn
import pandas as pd


def test_linkedQuantityID(minimal_test_esM):
    """ """
    esM = minimal_test_esM

    # get components
    electrolyzer = esM.getComponent("Electrolyzers")
    market = esM.getComponent("Electricity market")

    # create dummy component
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Dummy",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "electricity": 1},
            hasCapacityVariable=True,
            capacityPerPlantUnit=1.0,
            opexPerCapacity=1.0,
            linkedQuantityID="test",
        )
    )

    # make electrolyzer sizing discrete
    electrolyzer.capacityPerPlantUnit = 1
    electrolyzer.linkedQuantityID = "test"
    electrolyzer.processedOpexPerCapacity[0] = pd.Series(1, index=esM.locations)

    # optimize
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    assert (
        esM.getOptimizationSummary("ConversionModel")
        .loc["Electrolyzers"]
        .loc["opexCap"]["ElectrolyzerLocation"]
        .values.astype(int)[0]
        == esM.getOptimizationSummary("ConversionModel")
        .loc["Dummy"]
        .loc["opexCap"]["ElectrolyzerLocation"]
        .values.astype(int)[0]
    )
