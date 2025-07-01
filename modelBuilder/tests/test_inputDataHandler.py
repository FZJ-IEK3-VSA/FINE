from modelBuilder.inputDataHandler.inputDataHandler import inputDataHandler
import geokit as gk
import numpy as np
import os
import pickle
import pytest
from .test_data import test_data_folder
import pandas as pd

from modelBuilder.singletons import InputDataInfo

InputDataInfo(weather_year=2018, base_year=2050,number_of_investment_periods=1,investment_period_interval=1)

@pytest.fixture
def ted():
    ted = {'esM': {'costUnit': '1e9 Euro', 'lengthUnit': 'km', 'numberOfTimeSteps': 8760, 'hoursPerTimeStep': 1}, 'sources': {'wind_onshore': {'investPerCapacity': {2020: 1.29, 2030: 1.13, 2040: 1.05, 2050: 1.0}, 'opexPerCapacity': 0.025, 'interestRate': 0.08, 'economicLifetime': 20, 'commodity': 'electricity', 'GlobEPName': 'Onshore', 'clearName': 'Wind Onshore', 'citation': 'FINE.NESTOR'}, 'wind_offshore': {'investPerCapacity': {2020: 1.016, 2030: 0.774, 2040: 0.582, 2050: 0.474}, 'opexPerCapacity': 0.021, 'interestRate': 0.04, 'economicLifetime': 20, 'commodity': 'electricity', 'GlobEPName': 'tbd', 'citation': 'FINE.NESTOR', 'roofTopPV': None}, 'ofpv_fix': {'investPerCapacity': {2020: 0.69, 2030: 0.45, 2040: 0.37, 2050: 0.32}, 'opexPerCapacity': 0.017, 'interestRate': 0.08, 'economicLifetime': 20, 'commodity': 'electricity', 'GlobEPName': 'OFPV', 'citation': 'FINE.NESTOR'}, 'hydro_SpecificType': {'investPerCapacity': {2020: 3.0, 2030: 3.0, 2040: 3.0, 2050: 3.0}, 'opexPerCapacity': 0.017, 'interestRate': 0.08, 'economicLifetime': 40, 'commodity': 'electricity', 'GlobEPName': 'tbd', 'citation': 'e.pena.sanchez@f-juelich.de'}, 'geothermal_EGS': {'investPerCapacity': {2020: '-1.000', 2030: '-1.000', 2040: '-1.000', 2050: '-1.000'}, 'opexPerCapacity': 0.02, 'interestRate': 0.08, 'economicLifetime': 20, 'commodity': 'electricity', 'clearName': 'Enhanced Geothermal', 'citation': 'd.franzmann'}}, 'conversion': {'liquefaction': {'physicalUnit': 'GW$_{LH_{2},LHV}$', 'commodityConversionFactors': {'hydrogen_liquid': 1, 'electricity': -0.2, 'hydrogen_gas': -1.02}, 'investPerCapacity': {2020: 1.5, 2030: 1.5, 2040: 1.5, 2050: 1.5}, 'opexPerCapacity': 0.08, 'interestRate': 0.08, 'economicLifetime': 20, 'citation': 'FINE.Infrastructure'}, 'electrolyzer_pem': {'physicalUnit': 'GW$_{H_{2},LHV}$', 'commodityConversionFactors': {'electricity': -1.429, 'hydrogen_gas': 1}, 'investPerCapacity': {2020: 0.8, 2030: 0.5, 2040: 0.4, 2050: 0.35}, 'opexPerCapacity': 0.03, 'opexPerOperation': 0, 'economicLifetime': 10, 'interestRate': 0.08, 'citation': 'FINE.NESTOR'}, 'regasification': {'commodityConversionFactors': {'hydrogen_liquid': -1, 'electricity': -0.02, 'hydrogen_gas': 1}, 'investPerCapacity': {2020: 0.024, 2030: 0.024, 2040: 0.024, 2050: 0.024}, 'opexPerCapacity': 0.03, 'opexPerOperation': 0, 'interestRate': 0.08, 'economicLifetime': 10, 'citation': 'FINE.Infrastructure'}, 'ccgt_hydrogenGas': {'commodityConversionFactors': {'electricity': 1, 'hydrogen_gas': -1.587301587}, 'investPerCapacity': {2020: 0.76, 2030: 0.76, 2040: 0.76, 2050: 0.76}, 'opexPerCapacity': 0.014, 'opexPerOperation': 0.002, 'interestRate': 0.08, 'economicLifetime': 20, 'citation': 'FINE.Nestor'}}, 'storage': {'battery_LIIon': {'chargeEfficiency': 0.99, 'dischargeEfficiency': 0.99, 'cyclicLifetime': 10000, 'chargeRate': 1, 'dischargeRate': 1, 'interestRate': 0.08, 'selfDischarge': 4.230356001899693e-05, 'investPerCapacity': {2020: 0.31149967, 2030: 0.175113328, 2040: 0.15322416199999997, 2050: 0.131334996}, 'opexPerCapacity': 0.025, 'opexPerOperation': 0.0001, 'economicLifetime': 15, 'commodity': 'electricity', 'citation': 'FINE.NESTOR'}, 'hydrogenGas_vessel': {'chargeRate': 0.08333333333333, 'dischargeRate': 0.08333333333333, 'stateOfChargeMin': 0.1, 'stateOfChargeMax': 1, 'investPerCapacity': {2020: 0.018, 2030: 0.018, 2040: 0.018, 2050: 0.018}, 'opexPerCapacity': 0.02, 'opexPerOperation': 0.0001, 'interestRate': 0.04, 'economicLifetime': 30, 'commodity': 'hydrogen_gas', 'citation': 'FINE.NESTOR'}, 'hydrogenLiquid _tank': {'selfDischarge': 1.25018e-05, 'investPerCapacity': {2020: 0.00075, 2030: 0.00075, 2040: 0.00075, 2050: 0.00075}, 'opexPerCapacity': 0.02, 'interestRate': 0.08, 'economicLifetime': 20, 'commodity': 'hydrogen_liquid', 'citation': 'Heuser, Philipp-Matthias; Stolten, Detlef "Weltweite Infrastruktur zur Wasserstoffbereitstellung auf Basis erneuerbarer Energien", RWTH AAchen, 2020'}}, 'transmission': {'electricity_grid_onshore': {'investPerCapacity': {2020: 0.00113, 2030: 0.00112, 2040: 0.00096, 2050: 0.0009}, 'losses': 1e-05, 'economicLifetime': 60, 'opexPerCapacity': 0.035, 'opexPerOperation': 0, 'interestRate': 0.08, 'commodity': 'electricity', 'citation': 'Lacal Arantegui R, Jaeger-Waldau A, Vellei M, Sigfusson B, Magagna D, Jakubcionis M, Perez Fortes M, Lazarou S, Giuntoli J, Weidner Ronnefeld E, De Marco G, Spisto A, Gutierrez Moles C, authors Carlsson J, editor. ETRI 2014 - Energy Technology Reference Indicator projections for 2010-2050. EUR 26950. Luxembourg (Luxembourg): Publications Office of the European Union; 2014. JRC92496'}, 'electricity_grid_offshore': {'investPerCapacity': {2020: 0.00113, 2030: 0.00112, 2040: 0.00096, 2050: 0.0009}, 'losses': 1e-05, 'economicLifetime': 60, 'opexPerCapacity': 0.035, 'opexPerOperation': 0, 'interestRate': 0.08, 'commodity': 'electricity', 'citation': 'Lacal Arantegui R, Jaeger-Waldau A, Vellei M, Sigfusson B, Magagna D, Jakubcionis M, Perez Fortes M, Lazarou S, Giuntoli J, Weidner Ronnefeld E, De Marco G, Spisto A, Gutierrez Moles C, authors Carlsson J, editor. ETRI 2014 - Energy Technology Reference Indicator projections for 2010-2050. EUR 26950. Luxembourg (Luxembourg): Publications Office of the European Union; 2014. JRC92496'}, 'hydrogenGas_pipeline_onshore': {'investPerCapacity': {2020: 0.000185, 2030: 0.000185, 2040: 0.000185, 2050: 0.000185}, 'losses': 1e-05, 'opexPerCapacity': 0.005405405405, 'opexPerOperation': 0.02, 'economicLifetime': 40, 'interestRate': 0.08, 'commodity': 'hydrogen_gas', 'citation': 'Çağlayan, Dilara Gülçin; Stolten, Detlef: "A robust design of a renewable European energy system encompassing a hydrogen infrastructure", RWTH Aachen University, 2020'}, 'hydrogenGas_pipeline_offshore': {'investPerCapacity': {2020: 0.000716, 2030: 0.000716, 2040: 0.000716, 2050: 0.000716}, 'losses': 1e-05, 'opexPerCapacity': 0.009, 'economicLifetime': 40, 'interestRate': 0.08, 'commodity': 'hydrogen_gas', 'citation': 'Max Stargardt based on European Hydrogen Backbone 2022'}, 'hydrogenLiquid_shipping': {'opexPerOperation': 1.24e-09, 'commodity': 'hydrogen_liquid'}}, 'demand': {'electricity_demand': {'commodity': 'electricity'}, 'HydrogenGas_demand': {'commodity': 'hydrogen_gas'}, 'HydrogenLiquid_demand': {'commodity': 'hydrogen_liquid'}}} 
    return ted

@pytest.fixture
def inputDataHandler_default_regions(ted) -> inputDataHandler:
    
    shapeFilePath = os.path.join(test_data_folder, "input_data/test_regions.shp")
    location_shape = gk.vector.extractFeatures(
        shapeFilePath,
        where= "GID_1 in ('BHR.3_1', 'BHR.4_1', 'BHR.5_1' )",
    )
    location_shape["dflt_type"] = ["default", "default", "default"]
    location_shape["locationID"] = location_shape["GID_1"]

    ted = ted

    model_base_folder = None

    ih = inputDataHandler(
            location_shape=location_shape,
            model_base_folder=model_base_folder,
            technoEconomicData=ted,
        )
    
    return ih

@pytest.fixture
def inputDataHandler_agg_regions(ted) -> inputDataHandler:
    
    shapeFilePath = os.path.join(test_data_folder, "input_data/test_regions.shp")
    location_shape = gk.vector.extractFeatures(
        shapeFilePath,
        where= "GID_1 in ('BHR.3_1', 'BHR.4_1', 'BHR.5_1' )",
    )
    location_shape["dflt_type"] = ["agg", "default", "default"]
    location_shape["locationID"] = location_shape["GID_1"]
    location_shape.loc[0, "locationID"] = "BHR.3_1__BHR.5_1"

    ted = ted
    model_base_folder = None

    ih = inputDataHandler(
            location_shape=location_shape,
            model_base_folder=model_base_folder,
            technoEconomicData=ted,
            default_regions_shp=shapeFilePath
        )
    
    return ih

@pytest.fixture
def inputDataHandler_custom_regions(ted) -> inputDataHandler:
    
    shapeFilePath = os.path.join(
        test_data_folder, 
        'input_data', 
        'test_regions.shp',
        )
    
    location_shape = gk.vector.extractFeatures(
        shapeFilePath,
        where= "GID_1 in ('BHR.3_1', 'BHR.4_1', 'BHR.5_1' )",
    )
    location_shape["dflt_type"] = ["custom", "default", "default"]
    location_shape["locationID"] = location_shape["GID_1"]
    location_shape["locationID"].iloc[0] = "somethingelse"

    ted = ted

    isDefaultRegions = False

    model_base_folder = None

    ih = inputDataHandler(
            location_shape=location_shape,
            model_base_folder=model_base_folder,
            technoEconomicData=ted,
            default_regions_shp=shapeFilePath,
        )
    
    return ih

def test_get_capacities_and_timeseries_from_nc4(inputDataHandler_default_regions, inputDataHandler_custom_regions, inputDataHandler_agg_regions):

    def compare_potential_dicts(dictA, dictB):
        """Compare potential dicts with different data type values."""
        assert sorted(dictA.keys()) == sorted(dictB.keys())
        equal = True
        for k, v in dictA.items():
            for k2, v2 in v.items():
                var = (np.isclose(dictB[k][k2].values, v2.values).all()).all()
                if not var:
                    print("difference in:",k,k2, ":\ndictA:",dictA[k][k2].sum(), "\ndictB:",dictB[k][k2].sum())
                equal=equal*var
        return equal

    # hard code check data that should be returned by below test methods
    true_data = pickle.load(open(os.path.join(
        test_data_folder, 
        'true_data', 
        'potentials',
        "true_data_potential_dict_CSP_SolarSalt_default_regions.pickle"
        ),"rb")) #real data

    # overwrite defaults with offline test data paths 
    #TODO when fully implemented, define a test_input_data.yaml with test data paths and load for initial InputDataInfo setup
    ts_base_path = os.path.join(
        test_data_folder,
        "input_data",
        "potentials",
        "CSP/CSPs4-V2/CSPs4-V2/<GID0>/100m/<WEATHERYEAR>/cluster_vars/cluster_vars__CSP__<GID1>__SG<SPATGROUP>__<WEATHERYEAR>__100res__CSPs4-V2__Dataset_SolarSalt_2050.nc4"
    )
    cap_base_path = os.path.join(
        test_data_folder,
        "input_data",
        "potentials",
        "CSP/CSPs4-V2/CSPs4-V2/<GID0>/100m/<WEATHERYEAR>/plant_vars/plant_vars__CSP__<GID1>__SG<SPATGROUP>__<WEATHERYEAR>__100res__CSPs4-V2.pickle",
    )
    InputDataInfo().set_info(
        tech="csp_solarsalt", 
        attrs=["ts_base_path", "cap_base_path"], 
        vals=[ts_base_path, cap_base_path],
        overwrite=True,    
    )

    # test single dimensional default potential with polygon placements, e.g. OFPV_fixed 
    default_potential_dict = inputDataHandler_default_regions.get_capacities_and_timeseries_from_nc4(
        technology='csp_solarsalt', 
        model_unit='GW',
        use_partial_polygon_capacities=True, 
        global_clusters=False,
        verbose=True,
        )

    # ensure that the returned data matches the check data
    assert compare_potential_dicts(default_potential_dict, true_data) #real data

    ## agg regions
    default_potential_dict_custom_regions = inputDataHandler_agg_regions.get_capacities_and_timeseries_from_nc4(
        technology='csp_solarsalt', 
        model_unit='GW',
        use_partial_polygon_capacities=False, 
        global_clusters=False,
        verbose=True,
        )

    # as the agg region only summs ub BHR3 and BHR5, just add them up :)
    for cluster in range(0,3): #real data
        for var in ["Csf_W", "Cplant_W", "Cstr_kWh", "LCOE_clstr", "DNInom_Wm2"]:
            assert np.allclose(
                default_potential_dict_custom_regions[cluster][var].values.astype(float)[1:], 
                true_data[cluster][var].values.astype(float)[1:]
            )
            assert np.allclose(
                default_potential_dict_custom_regions[cluster][var].values.astype(float)[0], 
                true_data[cluster][var].values.astype(float)[-1]
            )
        for var in ["ts_capacity_factor_sf", "ts_capacity_factor_heat_FP_sf", "ts_capacity_factor_plant"]:
            assert np.allclose(
                default_potential_dict_custom_regions[cluster][var].values.astype(float)[:,1:], 
                true_data[cluster][var].values.astype(float)[:,1:]
            )
            assert np.allclose(
                default_potential_dict_custom_regions[cluster][var].values.astype(float)[:,0], 
                true_data[cluster][var].values.astype(float)[:,-1]
            )

    #inputDataHandler_custom_regions.location_shape
    #compare with custom regions
    default_potential_dict_custom_regions = inputDataHandler_custom_regions.get_capacities_and_timeseries_from_nc4(
        technology='csp_solarsalt', 
        model_unit='GW',
        use_partial_polygon_capacities=False, 
        global_clusters=False,
        verbose=True,
        )

    # ensure that the returned data matches the check data
    for cluster in range(0,3): #real data
        for var in ["Csf_W", "Cplant_W", "Cstr_kWh", "LCOE_clstr", "DNInom_Wm2"]:
            assert np.allclose(
                default_potential_dict_custom_regions[cluster][var].values.astype(float), 
                true_data[cluster][var].reindex(["BHR.4_1", "BHR.5_1", "BHR.3_1"]).values.astype(float)
            )
        for var in ["ts_capacity_factor_sf", "ts_capacity_factor_heat_FP_sf", "ts_capacity_factor_plant"]:
            assert np.allclose(
                default_potential_dict_custom_regions[cluster][var].values.astype(float), 
                true_data[cluster][var].reindex(["BHR.4_1", "BHR.5_1", "BHR.3_1"],axis=1).values.astype(float)
            )

@pytest.mark.skip("Results for use_partial_polygon_capacities are reduced by area ratio. TODO: Check with Winkler")
def test_get_capacities_and_timeseries_from_nc4_use_partial_polygon_capacities(inputDataHandler_default_regions):
    
    # hard code check data that should be returned by below test methods
    true_data = pickle.load(open(os.path.join(
        test_data_folder, 
        'true_data', 
        'potentials',
        "true_data_potential_dict_CSP_SolarSalt_default_regions.pickle"
        ),"rb"))
    
    base_folder = os.path.join(test_data_folder, "input_data", "potentials")

    # now reload the exact same potential via the default loader path
    individual_potential_dict = inputDataHandler_default_regions.get_capacities_and_timeseries_from_nc4(
        technology='csp_solarsalt', 
        model_unit='GW',
        data_unit='W',
        base_folder=base_folder,
        scenario='CSPs4-V2',
        iteration_name='CSPs4-V2',
        resolution=100,
        potentials_technology_name='CSP',
        sub_dataset_name='Dataset_SolarSalt_2050',
        aggregation_dict={  'ts_capacity_factor_sf': 'Csf_W',
                            'ts_capacity_factor_heat_FP_sf': 'Csf_W',
                            'ts_capacity_factor_plant': 'Cplant_W',},
        additional_aggregation_vars=['Cstr_kWh'],
        daily_timeseries='ts_capacity_factor_plant',
        hourly_reference_timeseries='ts_capacity_factor_sf',
        use_partial_polygon_capacities=True, 
        region_col_name='region',
        N_clusters=3, 
        global_clusters=False,
        verbose=True,
        )

    # ensure that the returned data matches the check data
    for cluster in range(0,3):
        for var in ["Csf_W", "Cplant_W", "Cstr_kWh", "LCOE_clstr"]:
            assert np.allclose(
                individual_potential_dict[cluster][var].values, 
                true_data[cluster][var].reindex(["BHR.4_1", "BHR.5_1", "BHR.3_1"]).values
            )
        for var in ["ts_capacity_factor_sf", "ts_capacity_factor_heat_FP_sf", "ts_capacity_factor_plant"]:
            assert np.allclose(
                individual_potential_dict[cluster][var].values, 
                true_data[cluster][var].reindex(["BHR.4_1", "BHR.5_1", "BHR.3_1"],axis=1).values
            )


def test_load_constant_potentials(inputDataHandler_default_regions, inputDataHandler_agg_regions):

    #1) sql
    path = os.path.join(
        test_data_folder, 
        "input_data",
        "potentials",
        "constant_potentials",
        "dummy_geothermal.sqlite",
        )

    data = inputDataHandler_default_regions.load_constant_potentials(
        technology="geothermal_EGS",
        model_unit="GW",
        N_cluster=3,
        path=path,
    )
    assert len(data) == 3

    assert np.allclose(data[0]["capacityMax"], [0, 0, 0.0096])
    assert np.allclose(data[1]["capacityMax"], [0, 0, 0])
    assert np.allclose(data[2]["capacityMax"], [0, 0, 0.0048])

    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 0.0926])
    assert np.allclose(data[1]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1000000])
    assert np.allclose(data[2]["LCOE_EUR_per_kWh"], [1000000, 1000000, 0.1926])

    path = os.path.join(
        test_data_folder, 
        "input_data",
        "potentials",
        "constant_potentials",
        "dummy_geothermal.sqlite",
        )

    data = inputDataHandler_agg_regions.load_constant_potentials(
        technology="geothermal_EGS",
        model_unit="GW",
        N_cluster=3,
        path=path,
    )
    assert len(data) == 3

    assert np.allclose(data[0]["capacityMax"], [0.0096, 0, 0.0096])
    assert np.allclose(data[1]["capacityMax"], [0, 0, 0])
    assert np.allclose(data[2]["capacityMax"], [0.0048, 0, 0.0048])

    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [0.0926, 1000000, 0.0926])
    assert np.allclose(data[1]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1000000])
    assert np.allclose(data[2]["LCOE_EUR_per_kWh"], [0.1926, 1000000, 0.1926])

    #2) csv
    path = os.path.join(
        test_data_folder, 
        "input_data", 
        "potentials",
        "constant_potentials",
        "salt_dummy.csv",
        )

    data = inputDataHandler_default_regions.load_constant_potentials(
        technology="hydrogen_saltcavern",
        model_unit="GW*h",
        N_cluster=1,
        path=path,
    )
    assert np.allclose(data[0]["capacityMax"], [0, 0, 473.383977])
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1733.33333])

    #3) shp
    path = os.path.join(
        test_data_folder, 
        "input_data", 
        "potentials",
        "constant_potentials",
        "salt_dummy.shp",
        )

    data = inputDataHandler_default_regions.load_constant_potentials(
        technology="hydrogen_saltcavern",
        model_unit="GW*h",
        N_cluster=1,
        path=path,
    )
    assert np.allclose(data[0]["capacityMax"], [0, 0, 473.383977])
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1733.33333])

def test_load_demand_default_region(inputDataHandler_default_regions, inputDataHandler_agg_regions):
    
    technology = "electricity_demand"
    path_abs_demands = os.path.join(
        test_data_folder, 
        "input_data", 
        "demand_electricity",
        "<YEAR>",
        "absolute_electricity_demands_<YEAR>_GWh_gid1.csv"
    )
    path_ts = os.path.join(
        test_data_folder, 
        "input_data", 
        "demand_electricity",
        "All_Demand_UTC_2015_processed.csv"
    )
    year_demand = 2020

    ih = inputDataHandler_default_regions
    operationRateFix_GW = ih.load_demand(
        technology=technology,
        path_abs_demands=path_abs_demands,
        path_ts=path_ts,
        year_demand=year_demand,
    )
    assert isinstance(operationRateFix_GW, pd.DataFrame)
    assert (operationRateFix_GW.columns == ["BHR.3_1", "BHR.4_1", "BHR.5_1"]).all()
    assert (operationRateFix_GW.index == range(0, 8760)).all()
    assert np.allclose(
        operationRateFix_GW.sum(axis=0),
        demand_2020
    )
    assert np.allclose(
        operationRateFix_GW.std(axis=0),
        demand_2020_std
    )

    assert np.allclose(
        (operationRateFix_GW["BHR.4_1"]*100 / operationRateFix_GW["BHR.4_1"].sum()).iloc[:5],
        [0.003414, 0.003850, 0.004478, 0.005170, 0.005796],
        rtol= 1e-4
    )


def test_load_demand_customregions(inputDataHandler_custom_regions, inputDataHandler_agg_regions):
    technology = "electricity_demand"
    path_abs_demands = os.path.join(
        test_data_folder, 
        "input_data", 
        "demand_electricity",
        "<YEAR>",
        "absolute_electricity_demands_<YEAR>_GWh_gid1.csv"
    )
    path_ts = os.path.join(
        test_data_folder, 
        "input_data", 
        "demand_electricity",
        "All_Demand_UTC_2015_processed.csv"
    )
    year_demand = 2020

    ih = inputDataHandler_custom_regions
    operationRateFix_GW = ih.load_demand(
        technology=technology,
        path_abs_demands=path_abs_demands,
        path_ts=path_ts,
        year_demand=year_demand,
    )
    assert isinstance(operationRateFix_GW, pd.DataFrame)
    assert (operationRateFix_GW.columns == ["BHR.4_1", "BHR.5_1", "somethingelse"]).all()
    assert (operationRateFix_GW.index == range(0, 8760)).all()

    #shift data according to test data
    operationRateFix_GW_shifted = operationRateFix_GW.reindex(columns=["somethingelse", "BHR.4_1", "BHR.5_1"])

    assert np.allclose(
        operationRateFix_GW_shifted.sum(axis=0),
        demand_2020
    )
    assert np.allclose(
        operationRateFix_GW_shifted.std(axis=0),
        demand_2020_std
    )

    assert np.allclose(
        (operationRateFix_GW["BHR.4_1"]*100 / operationRateFix_GW["BHR.4_1"].sum()).iloc[:5],
        [0.003414, 0.003850, 0.004478, 0.005170, 0.005796],
        rtol= 1e-4
    )


    #as agg regions are not known to demand, only custom demands will be loaded based on the shape
    operationRateFix_GW = inputDataHandler_agg_regions.load_demand(
        technology=technology,
        path_abs_demands=path_abs_demands,
        path_ts=path_ts,
        year_demand=year_demand,
    )
    assert np.allclose(
        operationRateFix_GW.sum(axis=0),
        demand_2020
    )

@pytest.mark.parametrize("config_ih", [inputDataHandler_default_regions, inputDataHandler_custom_regions])
def test_load_VOLL(config_ih, request):
    
    #get fixture as ih
    ih = request.getfixturevalue(config_ih.__name__)

    path_VOLL = os.path.join(
        test_data_folder, 
        "input_data", 
        "VOLL",
        "VOLL_dummy.csv"
    )

    #test wrong inputs
    with pytest.raises(OSError):
        ih.load_VOLL(
            path_VOLL="doesnotexist",
            voll_to_BEUR_per_GWh_factor=1,
            time_steps=8760,
            voll_key= "VOLL_2020[EUR/MWh]",
            sectoral_disaggregation=True,
        )
    with pytest.raises(KeyError):
        ih.load_VOLL(
            path_VOLL=path_VOLL,
            voll_to_BEUR_per_GWh_factor=1,
            time_steps=8760,
            voll_key= "VOLL_-1[EUR/MWh]",
            sectoral_disaggregation=True,
        )

    #test expected inputs:
    time_steps = 8760
    voll_to_BEUR_per_GWh_factor = 1
    data, shares = ih.load_VOLL(
        voll_to_BEUR_per_GWh_factor=voll_to_BEUR_per_GWh_factor,
        path_VOLL=path_VOLL,
        time_steps=time_steps,
        voll_key= "VOLL_2020[EUR/MWh]",
        sectoral_disaggregation=False,
    )

    assert isinstance(data, dict)
    assert shares is None

    assert isinstance(data[0], pd.DataFrame)
    assert data[0].shape == (time_steps, len(ih.location_shape))
    assert set(data[0].columns) == set(ih.location_shape.locationID)
    assert list(data[0].index) == list(range(0,time_steps))

    assert np.allclose(
        data[0],
        1000*voll_to_BEUR_per_GWh_factor
    )

    #test expected inputs:
    time_steps = 8760
    voll_to_BEUR_per_GWh_factor = 1
    data, shares = ih.load_VOLL(
        voll_to_BEUR_per_GWh_factor=voll_to_BEUR_per_GWh_factor,
        path_VOLL=path_VOLL,
        time_steps=time_steps,
        voll_key= "VOLL_2050[EUR/MWh]",
        sectoral_disaggregation=False,
    )

    assert isinstance(data, dict)
    assert shares is None

    assert isinstance(data[0], pd.DataFrame)
    assert data[0].shape == (time_steps, len(ih.location_shape))
    assert set(data[0].columns) == set(ih.location_shape.locationID)
    assert list(data[0].index) == list(range(0,time_steps))

    assert np.allclose(
        data[0],
        2000*voll_to_BEUR_per_GWh_factor
    )

    #test expected inputs:
    time_steps = 8760
    voll_to_BEUR_per_GWh_factor = 10
    data, shares= ih.load_VOLL(
        voll_to_BEUR_per_GWh_factor=voll_to_BEUR_per_GWh_factor,
        path_VOLL=path_VOLL,
        time_steps=time_steps,
        voll_key= "VOLL_2050[EUR/MWh]",
        sectoral_disaggregation=False,
    )

    assert isinstance(data, dict)
    assert shares is None

    assert isinstance(data, dict)
    assert isinstance(data[0], pd.DataFrame)
    assert data[0].shape == (time_steps, len(ih.location_shape))
    assert set(data[0].columns) == set(ih.location_shape.locationID)
    assert list(data[0].index) == list(range(0,time_steps))

    assert np.allclose(
        data[0],
        2000*voll_to_BEUR_per_GWh_factor
    )

    #test expected inputs:
    time_steps = 1000
    voll_to_BEUR_per_GWh_factor = 10
    data, shares = ih.load_VOLL(
        voll_to_BEUR_per_GWh_factor=voll_to_BEUR_per_GWh_factor,
        path_VOLL=path_VOLL,
        time_steps=time_steps,
        voll_key= "VOLL_2050[EUR/MWh]",
        sectoral_disaggregation=False,
    )
    
    assert isinstance(data, dict)
    assert shares is None
    
    assert isinstance(data[0], pd.DataFrame)
    assert data[0].shape == (time_steps, len(ih.location_shape))
    assert set(data[0].columns) == set(ih.location_shape.locationID)
    assert list(data[0].index) == list(range(0,time_steps))

    assert np.allclose(
        data[0],
        2000*voll_to_BEUR_per_GWh_factor
    )

    #test sectoral disaggregation:
    time_steps = 1000
    voll_to_BEUR_per_GWh_factor = 10
    data, shares = ih.load_VOLL(
        voll_to_BEUR_per_GWh_factor=voll_to_BEUR_per_GWh_factor,
        path_VOLL=path_VOLL,
        time_steps=time_steps,
        voll_key= "VOLL_2050[EUR/MWh]",
        sectoral_disaggregation=True,
    )

    assert isinstance(data, dict)
    sectors = ['agriculture', 'industry', 'residential', 'services', 'transportation']
    assert sorted(list(data.keys())) == sectors
    for sector in sectors:
        assert isinstance(data[sector], pd.DataFrame)
        assert data[sector].shape == (time_steps, len(ih.location_shape))
        assert set(data[sector].columns) == set(ih.location_shape.locationID)
        assert list(data[sector].index) == list(range(0,time_steps))

    sector = 'industry'
    test_phi = 0.55
    test_chi = 2
    base_value = 2000
    assert np.allclose(
        data[sector],
        base_value*test_phi*test_chi*voll_to_BEUR_per_GWh_factor
    )
    sector = 'services'
    test_phi = 0.7
    test_chi = 2
    base_value = 2000
    assert np.allclose(
        data[sector],
        base_value*test_phi*test_chi*voll_to_BEUR_per_GWh_factor
    )

    assert isinstance(shares, pd.DataFrame)
    assert sorted(list(shares.columns)) == sorted([f"share_{c}" for c in sectors])
    assert set(shares.index) == set(ih.location_shape.locationID)
    for region in shares.index:
        print(region)
        assert np.allclose(
            shares.loc[region],
            list(sectoral_demand_shares.values())
        )


demand_2020 = np.array([2371.40478843, 3289.43683931, 7027.52176944])
demand_2020_std = np.array([0.10403347008985357, 0.14430751371690667, 0.30829720820937784])
sectoral_demand_shares = {
    "industry": 0.50059482,
    "residential": 0.28743556,
    "services": 0.20986487,
    "agriculture": 0.00210475,
    "transportation": 0,
}