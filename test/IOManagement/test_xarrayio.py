from copy import deepcopy
import FINE.IOManagement.xarrayIO as xrIO


def test_esm_input_to_dataset_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)

    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esM)
    esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_datasets.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )
    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_datasets.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )

    assert (
        esM.getComponentAttribute("Pipelines", "investPerCapacity")[
            "ElectrolyzerLocation_IndustryLocation"
        ]
        == esm_from_datasets.getComponentAttribute("Pipelines", "investPerCapacity")[
            "ElectrolyzerLocation_IndustryLocation"
        ]
    )


def test_esm_output_to_dataset_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)
    esM.optimize()
    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esM)
    esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    assert (
        (
            esM.componentModelingDict["ConversionModel"]
            .getOptimalValues("operationVariablesOptimum")["values"]
            .loc["Electrolyzers"]
            .values.T
        )
        == (
            esm_datasets["Results"]["ConversionModel"]["Electrolyzers"][
                "operationVariablesOptimum"
            ].values
        )
    ).all()

    assert list(
        esM.getOptimizationSummary("SourceSinkModel", outputLevel=0)
        .loc["Electricity market"]
        .loc["TAC"]
        .values[0]
    ) == list(
        esm_datasets["Results"]["SourceSinkModel"]["Electricity market"].TAC.values
    )


def test_input_esm_to_netcdf_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)
    _ = xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath="test_esM.nc")
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM.nc")

    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_netcdf.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )
    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_netcdf.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )


def test_output_esm_to_netcdf_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)
    esM.optimize()
    _ = xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath="test_esM.nc")
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM.nc")
    print(
        esm_from_netcdf.componentModelingDict["ConversionModel"]
        .getOptimalValues("operationVariablesOptimum")["values"]
        .loc["Electrolyzers"]
        .values
    )
    assert (
        (
            esM.componentModelingDict["ConversionModel"]
            .getOptimalValues("operationVariablesOptimum")["values"]
            .loc["Electrolyzers"]
            .values
        )
        == (
            esm_from_netcdf.componentModelingDict["ConversionModel"]
            .getOptimalValues("operationVariablesOptimum")["values"]
            .loc["Electrolyzers"]
            .values
        )
    ).all()

    assert list(
        esM.getOptimizationSummary("SourceSinkModel", outputLevel=0)
        .loc["Electricity market"]
        .loc["TAC"]
        .values[0]
    ) == list(
        esm_from_netcdf.getOptimizationSummary("SourceSinkModel", outputLevel=0)
        .loc["Electricity market"]
        .loc["TAC"]
        .values[0]
    )



if __name__ == "__main__":
    import sys
    import os
    import FINE as fn
    import pandas as pd
    import numpy as np
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
   
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation", "IndustryLocation"},
        commodities={"electricity", "hydrogen"},
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
        lowerBound=False,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    ### Buy electricity at the electricity market
    costs = pd.DataFrame(
        [
            np.array(
                [
                    0.05,
                    0.0,
                    0.1,
                    0.051,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T
    revenues = pd.DataFrame(
        [
            np.array(
                [
                    0.0,
                    0.01,
                    0.0,
                    0.0,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T
    maxpurchase = (
        pd.DataFrame(
            [
                np.array(
                    [
                        1e6,
                        1e6,
                        1e6,
                        1e6,
                    ]
                ),
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ],
            index=["ElectrolyzerLocation", "IndustryLocation"],
        ).T
        * hoursPerTimeStep
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=maxpurchase,
            commodityCostTimeSeries=costs,
            commodityRevenueTimeSeries=revenues,
        )
    )  # eur/kWh

    ### Electrolyzers
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    ### Hydrogen filled somewhere
    esM.add(
        fn.Storage(
            esM=esM,
            name="Pressure tank",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,  # eur/kWh
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ### Hydrogen pipelines
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines",
            commodity="hydrogen",
            hasCapacityVariable=True,
            investPerCapacity=0.177,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    ### Industry site
    demand = (
        pd.DataFrame(
            [
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                np.array(
                    [
                        6e3,
                        6e3,
                        6e3,
                        6e3,
                    ]
                ),
            ],
            index=["ElectrolyzerLocation", "IndustryLocation"],
        ).T
        * hoursPerTimeStep
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

    test_input_esm_to_netcdf_and_back(esM)