import pandas as pd
import os
import xlrd


def getData():
    cwd = os.path.dirname(os.path.realpath(__file__))
    inputDataPath = os.path.join(cwd, "InputData")
    data = {}

    # Onshore data
    capacityMax = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'Wind', 'maxCapacityOnshore_GW_el.xlsx'),
                                index_col=0, squeeze=True)
    operationRateMax = pd.read_excel(
        os.path.join(inputDataPath, 'SpatialData', 'Wind', 'maxOperationRateOnshore_el.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'Wind (onshore), capacityMax': capacityMax})
    data.update({'Wind (onshore), operationRateMax': operationRateMax})

    # Offshore data
    capacityMax = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'Wind', 'maxCapacityOffshore_GW_el.xlsx'),
                                index_col=0, squeeze=True)
    operationRateMax = pd.read_excel(
        os.path.join(inputDataPath, 'SpatialData', 'Wind', 'maxOperationRateOffshore_el.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'Wind (offshore), capacityMax': capacityMax})
    data.update({'Wind (offshore), operationRateMax': operationRateMax})

    # PV data
    capacityMax = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'PV', 'maxCapacityPV_GW_el.xlsx'),
                                index_col=0, squeeze=True)
    operationRateMax = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'PV', 'maxOperationRatePV_el.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'PV, capacityMax': capacityMax})
    data.update({'PV, operationRateMax': operationRateMax})

    # Run of river data
    capacityFix = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'HydroPower', 'fixCapacityROR_GW_el.xlsx'),
                                index_col=0, squeeze=True)
    operationRateFix = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'HydroPower',
                                                  'fixOperationRateROR.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'Existing run-of-river plants, capacityFix': capacityFix})
    data.update({'Existing run-of-river plants, operationRateFix': operationRateFix})

    # Biogas data
    operationRateMax = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'Biogas',
                                                  'biogasPotential_GWh_biogas.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'Biogas, operationRateMax': operationRateMax})

    biogasCommodityCostTimeSeries = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'Biogas',
                                                  'biogasPriceTimeSeries_MrdEuro_GWh.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'Biogas, commodityCostTimeSeries': biogasCommodityCostTimeSeries})

    # Natural gas data
    naturalGasCommodityCostTimeSeries = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'NaturalGas',
                                                  'naturalGasPriceTimeSeries_MrdEuro_GWh.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'Natural Gas, commodityCostTimeSeries': naturalGasCommodityCostTimeSeries})

    # Natural gas plant data
    capacityMax = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'NaturalGasPlants',
                                             'existingCombinedCycleGasTurbinePlantsCapacity_GW_el.xlsx'),
                                index_col=0, squeeze=True)

    data.update({'Existing CCGT plants (methane), capacityMax': capacityMax})

    # Hydrogen salt cavern data
    capacityMax = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'GeologicalStorage',
                                             'existingSaltCavernsCapacity_GWh_methane.xlsx'),
                                index_col=0, squeeze=True) * 3 / 10

    data.update({'Salt caverns (hydrogen), capacityMax': capacityMax})

    # Methane salt cavern data
    capacityMax = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'GeologicalStorage',
                                             'existingSaltCavernsCapacity_GWh_methane.xlsx'),
                                index_col=0, squeeze=True)

    data.update({'Salt caverns (methane), capacityMax': capacityMax})

    # Pumped hydro storage data
    capacityFix = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'HydroPower',
                                             'fixCapacityPHS_storage_GWh_energyPHS.xlsx'),
                                index_col=0, squeeze=True)

    data.update({'Pumped hydro storage, capacityFix': capacityFix})

    # AC cables data
    capacityFix = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'ElectricGrid',
                                             'ACcableExistingCapacity_GW_el.xlsx'),
                                index_col=0, header=0)

    data.update({'AC cables, capacityFix': capacityFix})

    reactances = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'ElectricGrid',
                                            'ACcableReactance.xlsx'),
                                index_col=0, header=0)

    data.update({'AC cables, reactances': reactances})

    # DC cables data
    capacityFix = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'ElectricGrid',
                                             'DCcableExistingCapacity_GW_el.xlsx'),
                                index_col=0, header=0)
    distances = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'ElectricGrid',
                                           'DCcableLength_km.xlsx'),
                              index_col=0, header=0)
    losses = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'ElectricGrid',
                                        'DCcableLosses.xlsx'),
                           index_col=0, header=0)

    data.update({'DC cables, capacityFix': capacityFix})
    data.update({'DC cables, distances': distances})
    data.update({'DC cables, losses': losses})

    # Pipelines data
    eligibility = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'Pipelines',
                                             'pipelineIncidence.xlsx'), index_col=0, header=0)
    distances = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'Pipelines', 'pipelineLength.xlsx'),
                              index_col=0, header=0)

    data.update({'Pipelines, eligibility': eligibility})
    data.update({'Pipelines, distances': distances})

    # Electricity demand data
    operationRateFix = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'Demands',
                                                  'electricityDemand_GWh_el.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'Electricity demand, operationRateFix': operationRateFix})

    # Hydrogen demand data
    operationRateFix = pd.read_excel(os.path.join(inputDataPath, 'SpatialData', 'Demands',
                                                  'hydrogenDemand_GWh_hydrogen.xlsx'),
                                                  header = 0, index_col = 0)

    data.update({'Hydrogen demand, operationRateFix': operationRateFix})

    return data
