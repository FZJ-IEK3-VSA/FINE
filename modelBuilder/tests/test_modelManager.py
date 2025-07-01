import modelBuilder
import os
import fine as fn
import geokit as gk
import numpy as np
import pandas as pd
import pytest
import shutil

from .test_data import test_data_folder
from .test_grid import share_onshore_matrix, region_distance_m
from .test_inputDataHandler import demand_2020, sectoral_demand_shares
from modelBuilder.inputDataHandler import preprocess_union_shape

from modelBuilder.singletons import UnitHandling


    #################
    ##  INIT MODEL ##
    #################

@pytest.fixture
def modelbuilder__init__():

    model_base_folder = os.path.join(test_data_folder, "test_output_data")
    os.makedirs(model_base_folder, exist_ok=True)

    location_shape_path = os.path.join(test_data_folder, "input_data", "test_regions.shp")
    location_shape = gk.vector.extractFeatures(location_shape_path)

    commodityUnitsDict = {
        "electricity": (r"GW$_{el}$", "GW"),
        "hydrogen_gas": (r"GW$_{H_{2},LHV}$", "GW"),
    }

    modelManager = modelBuilder.modelManager(
        location_shape=location_shape,
        locationID_column="GID_1",
        commodityUnitsDict=commodityUnitsDict,
        cost_year=2050,
        model_base_folder=model_base_folder, #Note: A new intermediates folder will be created in the same directory as your main git modelBuilder repository
        srs=4326,
        path_to_techno_economic_data_yaml=None, # Use default data#
        zero_threshold=0,
        default_regions_fp=location_shape_path,
    )

    yield modelManager

    shutil.rmtree(model_base_folder)

@pytest.fixture
def modelbuilder__init__agg_regions():

    model_base_folder = os.path.join(test_data_folder, "test_output_data")
    os.makedirs(model_base_folder, exist_ok=True)

    location_shape_path = os.path.join(test_data_folder, "input_data", "test_regions.shp")
    location_shape = gk.vector.extractFeatures(location_shape_path)

    location_shape_lean = preprocess_union_shape(
        location_shape=location_shape,
        max_regions=2,
        return_as_gk=True,
    )

    commodities = {"electricity", "hydrogen_gas"}
    commodityUnitsDict = {
        "electricity": (r"GW$_{el}$", "GW"),
        "hydrogen_gas": (r"GW$_{H_{2},LHV}$", "GW"),
    }

    modelManager = modelBuilder.modelManager(
        location_shape=location_shape_lean,
        locationID_column="GID_1",
        commodities=commodities,
        commodityUnitsDict=commodityUnitsDict,
        cost_year=2050,
        model_base_folder=model_base_folder, #Note: A new intermediates folder will be created in the same directory as your main git modelBuilder repository
        srs=4326,
        path_to_techo_economic_data_yaml=None, # Use default data
        complete_setup=False, # test only main mM instance first
        zero_threshold=0,
        default_regions_fp=location_shape_path,
    )

    yield modelManager

    shutil.rmtree(model_base_folder)

@pytest.mark.skip(reason="not implemented yet")
def test___init__(modelbuilder__init__):
    assert False

@pytest.mark.skip(reason="not implemented yet")
def test_prepare_location_shape_dataframe(modelbuilder__init__):
    assert False

@pytest.mark.skip(reason="not implemented yet")
def determine_if_locations_are_default_regions(modelbuilder__init__):
    assert False

@pytest.mark.skip(reason="not implemented yet")
def test_modelManager_setup():
    """
    Tests if modelBuilder can create an esM from a modelManager instance
    """
        

    testModel = modelBuilder.modelManager(
        location_shape="regionsPath",
        locationID_column="GID_1",
        commodityUnitsDict={'electricity':('GW', 'GW')},
        cost_year=2050,
        model_base_folder=None,
        srs=4326,
        path_to_techno_economic_data_yaml=None,
        complete_setup=False, # setups will be tested below
        default_regions_fp=location_shape_path,
    )
    testModel.modelSetup()

    assert isinstance(testModel.esM, fn.energySystemModel.EnergySystemModel)

@pytest.mark.skip(reason="not implemented yet")
def test_inputHandlerSetup():
    assert False

@pytest.mark.skip(reason="not implemented yet")
def test_technoEconomicData_setup():
    assert False

@pytest.mark.skip(reason="not implemented yet")
def test_defaultPotentials_setup():
    assert False


    ######################
    ##  ADDING TO MODEL ##
    ######################

@pytest.fixture
def modelbuilder_fully_inited(modelbuilder__init__):
    
    modelbuilder__init__.technoEconomicData_setup()
    modelbuilder__init__.inputHandlerSetup()
    modelbuilder__init__.modelSetup()

    yield modelbuilder__init__


@pytest.mark.skip(reason="not implemented yet")
def test_addPotentialWithTSGreenfield(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    base_folder=os.path.join(test_data_folder, "input_data", "potentials")

    # should fail for non existing technology name
    with pytest.raises(Exception):
        mB.addPotentialWithTSGreenfield(
            technology="does_not_exist"
        )

    #working function call
    mB.addPotentialWithTSGreenfield(
        technology="OFPV_fixed", 
        commodity="electricity",
        model_unit="GW",
        data_unit=None,
        base_folder=base_folder,
        scenario=None,
        iteration_name=None,
        resolution=100,
        potentials_technology_name=None,
        sub_dataset_name=None,
        aggregation_dict=None,
        additional_aggregation_vars=None,
        daily_timeseries=None,
        hourly_reference_timeseries=None,
        use_partial_polygon_capacities=True,
        region_col_name='region',
        N_clusters=3,
        global_clusters=False,
        verbose=True,
    ) 
    
    #make sure comps were added
    assert "OFPV_fixed__cluster_000" in mB.esM.componentNames.keys()
    assert "OFPV_fixed__cluster_001" in mB.esM.componentNames.keys()
    assert "OFPV_fixed__cluster_002" in mB.esM.componentNames.keys()
    assert mB.esM.componentNames["OFPV_fixed__cluster_000"] == 'SourceSinkModel'
    assert mB.esM.componentNames["OFPV_fixed__cluster_001"] == 'SourceSinkModel'
    assert mB.esM.componentNames["OFPV_fixed__cluster_002"] == 'SourceSinkModel'

    #make sure values were loaded poperly Exemplary, as this should also be checked in inputdatahandler
    assert np.allclose(
        mB.esM.getComponentAttribute("OFPV_fixed__cluster_000", "capacityMax").values,
        [0, 0, 0.0096] #TODO: add values here
    )
    assert np.isclose(mB.esM.getComponentAttribute("geothermal_EGS__cluster_000", "operationRateMax").sum(), "???") #TODO: add values here
    assert np.isclose(mB.esM.getComponentAttribute("geothermal_EGS__cluster_000", "operationRateMax").std(), "???") #TODO: add values here


@pytest.mark.skip(reason="not implemented yet")
def test_addPotentialWithTSGreenfield_withKwargs(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    base_folder=os.path.join(test_data_folder, "input_data", "potentials")

    # should fail since FINE notation would not expect investpercapacity in lower case
    FINE_kwargs_invalid={'investpercapacity':1200}

    with pytest.raises(Exception):
        mB.addPotentialWithTSGreenfield(
            technology="OFPV_fixed", 
            commodity="electricity",
            model_unit="GW",
            data_unit=None,
            base_folder=base_folder,
            scenario=None,
            iteration_name=None,
            resolution=100,
            potentials_technology_name=None,
            sub_dataset_name=None,
            aggregation_dict=None,
            additional_aggregation_vars=None,
            daily_timeseries=None,
            hourly_reference_timeseries=None,
            use_partial_polygon_capacities=True,
            region_col_name='region',
            N_clusters=3,
            global_clusters=False,
            verbose=True,
            **FINE_kwargs_invalid,
        )

    # should work but set the new test_value as opexPerCapacity
    test_value=20
    FINE_kwargs_valid={'opexPerCapacity':test_value}

    mB.addPotentialWithTSGreenfield(
        technology="OFPV_fixed", 
        commodity="electricity",
        model_unit="GW",
        data_unit=None,
        base_folder=base_folder,
        scenario=None,
        iteration_name=None,
        resolution=100,
        potentials_technology_name=None,
        sub_dataset_name=None,
        aggregation_dict=None,
        additional_aggregation_vars=None,
        daily_timeseries=None,
        hourly_reference_timeseries=None,
        use_partial_polygon_capacities=True,
        region_col_name='region',
        N_clusters=3,
        global_clusters=False,
        verbose=True,
        **FINE_kwargs_valid,
    ) 
    
    # make sure the parameter was set as component attribute in FINE esM
    assert mB.esM.getComponentAttribute("OFPV_fixed__cluster_000", "opexPerCapacity") == test_value

def test_add_commodities_and_units(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited
    # first test an additional commodity that already exists in model commodities
    test_commodity_unit_dict2 = {'electricity':('GW_therm', "GW")}
    with pytest.raises(Exception):
        mB._add_commodities_and_units(new_commodity_unit_dict=test_commodity_unit_dict2)
    # now add an actually new commodity
    test_commodity_unit_dict2 = {'new_commodity':('GW_therm', "GW")}
    mB._add_commodities_and_units(new_commodity_unit_dict=test_commodity_unit_dict2)
    # make sure all commodity names and units are in model commodity unit dict now
    assert all([v[0]==mB.esM.commodityUnitsDict[k] for k,v in test_commodity_unit_dict2.items()])
    assert all([k in UnitHandling().get_commodities() for k in test_commodity_unit_dict2.keys()])

def test_addPotentialWithTSGreenfield_Offshore(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    ts_base_path = os.path.join(
        test_data_folder,
        "input_data",
        "potentials",
        "Offshore/Base/v20230101/<GID0>/100m/<WEATHERYEAR>/cluster_vars/cluster_vars__Offshore__<GID1>__SG<SPATGROUP>__<WEATHERYEAR>__100res__Base.nc4",
    )
    cap_base_path = os.path.join(
        test_data_folder,
        "input_data",
        "potentials",
        "Offshore/Base/v20230101/<GID0>/100m/<WEATHERYEAR>/plant_vars/plant_vars__Offshore__<GID1>_off*__SG<SPATGROUP>__<WEATHERYEAR>__100res__Base.pickle",
    )

    #working function call
    mB.addPotentialWithTSGreenfield(
        technology="wind_offshore", 
        commodity="electricity",
        model_unit="GW",
        ts_base_path=ts_base_path,
        cap_base_path=cap_base_path,
    ) 
    
    # make sure comps were added
    assert "wind_offshore__0" in mB.esM.componentNames.keys()
    assert "wind_offshore__1" in mB.esM.componentNames.keys()
    assert "wind_offshore__2" in mB.esM.componentNames.keys()
    assert mB.esM.componentNames["wind_offshore__0"] == "SourceSinkModel"
    assert mB.esM.componentNames["wind_offshore__1"] == "SourceSinkModel"
    assert mB.esM.componentNames["wind_offshore__2"] == "SourceSinkModel"

    # make sure values were loaded poperly Exemplary, as this should also be checked in inputdatahandler
    assert np.allclose(
        mB.esM.getComponentAttribute("wind_offshore__0", "capacityMax").values, 
        [1.987576, 0.009584, 1.152691]
    ) # true data, checked against values in input .shp
    assert np.allclose(
        mB.esM.getComponentAttribute("wind_offshore__0", "operationRateMax").sum().values,
        [2476.320108, 2614.266389, 2393.492540],
    ) # true data, checked against values in input .nc4
    assert np.allclose(
        mB.esM.getComponentAttribute("wind_offshore__0", "investPerCapacity").values,
        [2.145966, 2.183046, 2.124681],
    ) # true data, average over all 3 clusters checked against input .shp

    # must fail with non-existing FINE parameter spelling
    with pytest.raises(Exception):
        mB.addPotentialWithTSGreenfield(
            technology="wind_offshore", 
            commodity="electricity",
            model_unit="GW",
            ts_base_path=ts_base_path,
            cap_base_path=cap_base_path,
            cluster_params={'invstPrCpcty': 'capx_EURkW'}, #invstPrCpcty is not a valid FINE.Source() parameter name
        )     

def test_addPotentialConstGreenfield(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    path_sql = os.path.join(test_data_folder, "input_data/potentials/constant_potentials/dummy_geothermal.sqlite")

    # should fail for non existing technology name
    with pytest.raises(Exception):
        mB.addPotentialConstGreenfield(
            technology="nonexisting_technology",
            N_cluster=None,
            path=None,
            LCOE_name=None,
            capacity_name=None,
            region_name_col=None,
            LCOE_to_EUR_per_kWh_factor=None,
            rounding=4,
            operationRateMax=1,
        )

    #working function call
    mB.addPotentialConstGreenfield(
        technology="geothermal_EGS",
        N_cluster=3,
        path=path_sql,
        LCOE_name=None,
        capacity_name=None,
        region_name_col=None,
        LCOE_to_EUR_per_kWh_factor=None,
        rounding=4,
        operationRateMax=1,
    )

    #make sure comps were added
    assert "geothermal_EGS__cluster_000" in mB.esM.componentNames.keys()
    assert "geothermal_EGS__cluster_001" in mB.esM.componentNames.keys()
    assert "geothermal_EGS__cluster_002" in mB.esM.componentNames.keys()
    assert mB.esM.componentNames["geothermal_EGS__cluster_000"] == 'SourceSinkModel'
    assert mB.esM.componentNames["geothermal_EGS__cluster_001"] == 'SourceSinkModel'
    assert mB.esM.componentNames["geothermal_EGS__cluster_002"] == 'SourceSinkModel'

    #make sure values were loaded poperly Exemplary, as this should also be checked in inputdatahandler
    assert np.allclose(
        mB.esM.getComponentAttribute("geothermal_EGS__cluster_000", "capacityMax").values,
        [0, 0, 0.0096]
    )
    assert (mB.esM.getComponentAttribute("geothermal_EGS__cluster_000", "operationRateMax")==1).all().all()

    assert np.allclose(
        mB.esM.getComponentAttribute("geothermal_EGS__cluster_000","investPerCapacity"),
        [8.04944096e+07, 8.04944096e+07, 7.45380000e+00],
        )
    assert np.allclose(
        mB.esM.getComponentAttribute("geothermal_EGS__cluster_001","investPerCapacity"),
        [8.04944096e+07, 8.04944096e+07, 8.04944096e+07],
        )
    assert np.allclose(
        mB.esM.getComponentAttribute("geothermal_EGS__cluster_002","investPerCapacity"),
        [8.04944096e+07, 8.04944096e+07, 1.55032000e+01],
        )


def test_addPotentialConstGreenfield_withKwargs(modelbuilder_fully_inited):

    mB = modelbuilder_fully_inited

    path_sql = os.path.join(test_data_folder, "input_data/potentials/constant_potentials/dummy_geothermal.sqlite")

    # should fail since FINE notation would not expect investpercapacity in lower case
    FINE_kwargs_invalid={'investpercapacity':1200}

    with pytest.raises(Exception):
        mB.addPotentialConstGreenfield(
            technology="geothermal_EGS",
            N_cluster=3,
            path=path_sql,
            LCOE_name=None,
            capacity_name=None,
            region_name_col=None,
            LCOE_to_EUR_per_kWh_factor=None,
            rounding=4,
            operationRateMax=1,
            **FINE_kwargs_invalid,
        )

    # should work but set the new test_value as opexPerCapacity
    test_value=20
    FINE_kwargs_valid={'opexPerCapacity':test_value}

    mB.addPotentialConstGreenfield(
        technology="geothermal_EGS",
        N_cluster=3,
        path=path_sql,
        LCOE_name=None,
        capacity_name=None,
        region_name_col=None,
        LCOE_to_EUR_per_kWh_factor=None,
        rounding=4,
        operationRateMax=1,
        **FINE_kwargs_valid,
    )

    # make sure the parameter was set as component attribute in FINE esM
    assert mB.esM.getComponentAttribute("geothermal_EGS__cluster_000", "opexPerCapacity") == test_value
    

@pytest.mark.skip(reason="not implemented yet")
def test_add_csp():
    assert False

def test_addPotentialWithTSBrownfield(modelbuilder_fully_inited):
    '''test succeds, if function raises error.
    Makes sure, that if function gets implemented, a test has to be written. 

    Parameters
    ----------
    modelbuilder__init__ : _type_
        _description_
    '''
    mb = modelbuilder_fully_inited
    
    with pytest.raises(NotImplementedError):
        mb.addPotentialWithTSBrownfield()

def test_addPotentialConstBrownfield(modelbuilder_fully_inited):
    '''test succeds, if function raises error.
    Makes sure, that if function gets implemented, a test has to be written. 

    Parameters
    ----------
    modelbuilder__init__ : _type_
        _description_
    '''
    mb = modelbuilder_fully_inited
    
    with pytest.raises(Exception):
        mb.addPotentialConstBrownfield()


def test_addGridBrownfield(modelbuilder_fully_inited):
    """test succeds, if function raises error.
    Makes sure, that if function gets implemented, a test has to be written. 

    Parameters
    ----------
    modelbuilder__init__ : _type_
        _description_
    """
    mB = modelbuilder_fully_inited

    # test failing for non existing name
    with pytest.raises(Exception):
        mB.addGridBrownfield(
            technology="does_not_exist", model_unit="GW",
        )

    path_grids = os.path.join(test_data_folder, "input_data/grids/brownfield_grid_dummy_BHR.shp")

    # test electricity_grid in detail
    technology = "testingparams_grid"
    mB.addGridBrownfield(technology=technology, model_unit="GW", data_unit="MW", path_grids=path_grids)

    technology = f"{technology}_brownfield"
    assert technology in mB.esM.componentNames.keys()
    assert mB.esM.componentNames[technology] == "TransmissionModel"

    # test vars
    # commodity
    assert mB.esM.getComponent(technology).commodity == "electricity"
    # hasCapacityVariable
    assert mB.esM.getComponent(technology).hasCapacityVariable  # == True
    # capcityFix
    print(mB.esM.getComponent(technology).capacityFix)
    assert np.allclose(mB.esM.getComponent(technology).capacityFix, pd.Series([0.4, 0.4, 0.5, 0.5]))
    # investPerCapacity
    assert np.allclose(mB.esM.getComponent(technology).investPerCapacity, 0)
    # opexPerOperation
    assert np.allclose(mB.esM.getComponent(technology).opexPerOperation, 0.0001)
    # opexPerCapacity
    assert np.allclose(mB.esM.getComponent(technology).opexPerCapacity, 0.01 * 1)
    # losses
    assert np.allclose(mB.esM.getComponent(technology).losses, 0.0001)
    # economicLifetime
    assert np.allclose(mB.esM.getComponent(technology).economicLifetime, 10)

    # check if shapefile with grids was created
    assert os.path.exists(os.path.join(mB.model_base_folder, "spatial_data", "transmission", "electricity_grid_brownfield.shp"))
        


def test_addGridGreenfield(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    #test failing for non existing name
    with pytest.raises(Exception):
        mB.addGridGreenfield(
            technology="does_not_exist",
            detour_factor=1.3
        )

    #test electricity_grid in detail
    technology="testingparams_grid"
    detour_factor=1.3

    mB.addGridGreenfield(
        technology=technology,
        detour_factor=detour_factor
    )

    assert technology in mB.esM.componentNames.keys()
    assert mB.esM.componentNames[technology] == 'TransmissionModel'

    assert (mB.esM.getComponent(technology).locationalEligibility \
            == pd.Series(
                [1,1,1,1],
                index = ['BHR.3_1_BHR.4_1', 'BHR.4_1_BHR.3_1', 'BHR.4_1_BHR.5_1', 'BHR.5_1_BHR.4_1']
            )
        ).all()
    
    assert np.allclose(
        mB.esM.getComponent(technology).distances,
        pd.Series(
            [21.0179584, 21.0179584, 25.808902, 25.808902],
            index = ['BHR.3_1_BHR.4_1', 'BHR.4_1_BHR.3_1', 'BHR.4_1_BHR.5_1', 'BHR.5_1_BHR.4_1']
        ) * detour_factor
    )
    

    region_distance_m_format_fine = np.array([
            (region_distance_m / 1E3 * detour_factor)[0,1],
            (region_distance_m / 1E3 * detour_factor)[1,0],
            (region_distance_m / 1E3 * detour_factor)[2,1],
            (region_distance_m / 1E3 * detour_factor)[2,1],
        ])
    share_onshore_matrix_format_fine = np.array([
            (share_onshore_matrix)[0,1],
            (share_onshore_matrix)[1,0],
            (share_onshore_matrix)[2,1],
            (share_onshore_matrix)[2,1],
        ])

    # test vars
    # commodity
    assert mB.esM.getComponent(technology).commodity == "electricity"
    # hasCapacityVariable
    assert mB.esM.getComponent(technology).hasCapacityVariable #== True
    # distances
    assert np.allclose(
        mB.esM.getComponent(technology).distances,
        region_distance_m_format_fine
    )
    # investPerCapacity
    assert np.allclose(
        mB.esM.getComponent(technology).investPerCapacity,
        1 + 1 * (1-share_onshore_matrix)
    )
    # opexPerOperation
    assert np.allclose(
        mB.esM.getComponent(technology).opexPerOperation,
        0.0001 + 0.0001 * (1-share_onshore_matrix_format_fine)
    )
    # opexPerCapacity
    assert np.allclose(
        mB.esM.getComponent(technology).opexPerCapacity,
        share_onshore_matrix * (0.01 * 1) + (1-share_onshore_matrix) * (0.02 * 2)
    )
    # losses
    assert np.allclose(
        mB.esM.getComponent(technology).losses,
        0.0001 + 0.0001 * (1-share_onshore_matrix_format_fine)
    )
    # economicLifetime
    #TODO: comment when fine bug fixed for variable economic lifetime
    assert np.allclose(mB.esM.getComponent(technology).economicLifetime, 10)
    #TODO: uncomment when fine bug fixed for variable economic lifetime
    # assert np.allclose(
    #     mB.esM.getComponent(technology).interestRate,
    #     10 + 10 * (1-share_onshore_matrix)
    # )

    # interestRate
    assert np.allclose(
        mB.esM.getComponent(technology).interestRate,
        0.01 + 0.01 * (1-share_onshore_matrix_format_fine)
    )

    
    #check "hydrogenGas_pipeline"
    mB.addGridGreenfield(
        technology="hydrogenGas_pipeline",
        detour_factor=1.3
    )
    assert "hydrogenGas_pipeline" in mB.esM.componentNames.keys()
    assert mB.esM.componentNames["hydrogenGas_pipeline"] == 'TransmissionModel'

    assert np.allclose( #this might fail, if techno economic params are changed. adapt test ;)
        mB.esM.getComponent("hydrogenGas_pipeline").investPerCapacity,
        np.array(
            [[0.000185  , 0.00038728, 0.000469  ],
            [0.00038728, 0.000185  , 0.000185  ],
            [0.000469  , 0.000185  , 0.000185  ]]
        )
    )
    

def test_addConversionUnlimitedGreenfield(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    #test for non existing technology
    with pytest.raises(Exception):
        mB.addConversionUnlimitedGreenfield(technology="does_not_exist")
    

    technology = "electrolyzer_pem_compressor"
    mB.addConversionUnlimitedGreenfield(technology=technology)

    #make sure comp is added
    assert technology in mB.esM.componentNames.keys()
    assert mB.esM.componentNames[technology] == 'ConversionModel'

    #make sure all vars are set:
    comp = mB.esM.getComponent(technology)

    assert comp.name == technology
    assert comp.physicalUnit == mB.ted["conversion"][technology]["physicalUnit"]
    assert comp.commodityConversionFactors == mB.ted["conversion"][technology]["commodityConversionFactors"]
    assert comp.hasCapacityVariable == True
    assert comp.opexPerOperation == mB.ted["conversion"][technology]["opexPerOperation"]
    assert comp.investPerCapacity == mB.ted["conversion"][technology]["investPerCapacity"][mB.cost_year]
    assert comp.opexPerCapacity == mB.ted["conversion"][technology]["opexFix"] * mB.ted["conversion"][technology]["investPerCapacity"][mB.cost_year]
    assert (comp.interestRate == mB.ted["conversion"][technology]["interestRate"]).all()
    assert (comp.economicLifetime == mB.ted["conversion"][technology]["economicLifetime"]).all()


def test_addConversionUnlimitedGreenfield_withKwargs(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    # should fail since FINE notation would not expect investpercapacity in lower case
    FINE_kwargs_invalid={'investpercapacity':1200}

    with pytest.raises(Exception):
        mB.addConversionUnlimitedGreenfield(technology="electrolyzer_pem_compressor", **FINE_kwargs_invalid)
    
    # should work but set the new test_value as opexPerCapacity
    test_value=20
    FINE_kwargs_valid={'opexPerCapacity':test_value}
    mB.addConversionUnlimitedGreenfield(technology="electrolyzer_pem_compressor", **FINE_kwargs_valid)

    # make sure the parameter was set as component attribute in FINE esM
    assert mB.esM.getComponentAttribute("electrolyzer_pem_compressor", "opexPerCapacity") == test_value
    

def test_addStorageUnlimitedGreenfield(modelbuilder_fully_inited):
    mB = modelbuilder_fully_inited
    with pytest.raises(Exception):
        mB.addStorageUnlimitedGreenfield(technology="does_not_exist")
    
    technology = "battery_LIIon"
    cost_factors = [1,2]

    for cost_factor in cost_factors:
        mB.addStorageUnlimitedGreenfield(
            technology=technology,
            cost_factor=cost_factor,
        )
        assert technology in mB.esM.componentNames.keys()
        assert mB.esM.componentNames[technology] == 'StorageModel'

        #make sure all vars are set:
        comp = mB.esM.getComponent(technology)

        capex = cost_factor * mB.ted["storage"][technology]["investPerCapacity"][mB.cost_year]

        assert comp.name == technology
        assert comp.commodity == mB.ted["storage"][technology]["commodity"]
        assert comp.hasCapacityVariable == True
        assert comp.chargeEfficiency == mB.ted["storage"][technology]["chargeEfficiency"]
        assert comp.dischargeEfficiency == mB.ted["storage"][technology]["dischargeEfficiency"]
        assert comp.cyclicLifetime == mB.ted["storage"][technology]["cyclicLifetime"]
        assert comp.selfDischarge == mB.ted["storage"][technology]["selfDischarge"]
        assert comp.chargeRate == mB.ted["storage"][technology]["chargeRate"]
        assert comp.dischargeRate == mB.ted["storage"][technology]["dischargeRate"]
        assert comp.doPreciseTsaModeling == False
        assert comp.investPerCapacity == capex
        assert comp.opexPerCapacity == capex * mB.ted["storage"][technology]["opexFix"]
        assert (comp.interestRate == mB.ted["storage"][technology]["interestRate"]).all()
        assert (comp.economicLifetime == mB.ted["storage"][technology]["economicLifetime"]).all()


def test_addStorageUnlimitedGreenfield_withKwargs(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    # should fail since FINE notation would not expect investpercapacity in lower case
    FINE_kwargs_invalid={'investpercapacity':1200}

    with pytest.raises(Exception):
        mB.addStorageUnlimitedGreenfield(technology="battery_LIIon", **FINE_kwargs_invalid)

    # should work but set the new test_value as opexPerCapacity
    test_value=20
    FINE_kwargs_valid={'opexPerCapacity':test_value}

    mB.addStorageUnlimitedGreenfield(technology='battery_LIIon', **FINE_kwargs_valid)
    
    # make sure the parameter was set as component attribute in FINE esM
    assert mB.esM.getComponentAttribute("battery_LIIon", "opexPerCapacity") == test_value


def test_addStorageLimitedGreenfield(modelbuilder_fully_inited):
    mB = modelbuilder_fully_inited
    with pytest.raises(Exception):
        mB.addStorageLimitedGreenfield(technology="does_not_exist")
    
    technology = "hydrogen_saltcavern"
    factors = [(1,1), (0.5, 2)]
    path_salt_cavern_data = os.path.join(
        test_data_folder, 
        "input_data", 
        "potentials",
        "constant_potentials",
        "salt_dummy.csv",
        )

    for cost_factor, capacity_factor in factors:
        mB.addStorageLimitedGreenfield(
            technology=technology,
            path=path_salt_cavern_data,
            LCOE_name=None,
            capacity_name=None,
            region_name_col=None,
            LCOE_to_EUR_per_kWh_factor=None,
            rounding=4,
            cost_factor=cost_factor,
            capacity_factor=capacity_factor,
        )
        #check if tech appears
        assert technology in mB.esM.componentNames.keys()
        assert mB.esM.componentNames[technology] == 'StorageModel'
        
        #check values
        assert np.allclose(
            mB.esM.getComponent(technology).capacityMax,
            pd.Series(
                [  0.   ,   0.   , 473.384],
                index=['BHR.3_1', 'BHR.4_1', 'BHR.5_1']
            ) * capacity_factor
        )
        
        capex = cost_factor * mB.ted["storage"][technology]["investPerCapacity"][mB.cost_year]
        assert np.allclose(
            mB.esM.getComponent(technology).investPerCapacity,
            capex
        )


def test_addStorageLimitedGreenfield_withKwargs(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    path_salt_cavern_data = os.path.join(
        test_data_folder, 
        "input_data", 
        "potentials",
        "constant_potentials",
        "salt_dummy.csv",
        )

    # should fail since FINE notation would not expect investpercapacity in lower case
    FINE_kwargs_invalid={'investpercapacity':1200}

    with pytest.raises(Exception):
        mB.addStorageLimitedGreenfield(
            technology='hydrogen_saltcavern',
            path=path_salt_cavern_data,
            LCOE_name=None,
            capacity_name=None,
            region_name_col=None,
            LCOE_to_EUR_per_kWh_factor=None,
            rounding=4,
            **FINE_kwargs_invalid,
            )

    # should work but set the new test_value as opexPerCapacity
    test_value=20
    FINE_kwargs_valid={'opexPerCapacity':test_value}

    mB.addStorageLimitedGreenfield(
        technology='hydrogen_saltcavern',
        path=path_salt_cavern_data,
        LCOE_name=None,
        capacity_name=None,
        region_name_col=None,
        LCOE_to_EUR_per_kWh_factor=None,
        rounding=4,
        **FINE_kwargs_valid,
    )

    # make sure the parameter was set as component attribute in FINE esM
    assert mB.esM.getComponentAttribute("hydrogen_saltcavern", "opexPerCapacity") == test_value


def test_addDemand_factor(modelbuilder_fully_inited):

    mB = modelbuilder_fully_inited

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

    technology="electricity_demand"
    factor = 1.9227
    mB.addDemand(
        technology=technology,
        year_demand=2020,
        factor = factor,
        path_abs_demands=path_abs_demands,
        path_ts=path_ts
    )
    assert technology in mB.esM.componentNames.keys()
    assert mB.esM.componentNames[technology] == 'SourceSinkModel'

    comp = mB.esM.getComponent(technology)
    assert comp.name == technology
    assert comp.commodity == "electricity"
    assert comp.hasCapacityVariable == False
    assert comp.operationRateFix is not None #values are checked within inputdatahandler 
    assert np.allclose(
        comp.operationRateFix.sum(axis=0),
        demand_2020 * factor
    )
    
def test_addDemand(modelbuilder_fully_inited):

    mB = modelbuilder_fully_inited

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
        "All_Demand_UTC_2015_processed.csv",
    )

    technology="electricity_demand"
    factor = 1
    mB.addDemand(
        technology=technology,
        year_demand=2020,
        factor = factor,
        path_abs_demands=path_abs_demands,
        path_ts=path_ts
    )
    assert technology in mB.esM.componentNames.keys()
    assert mB.esM.componentNames[technology] == 'SourceSinkModel'

    comp = mB.esM.getComponent(technology)
    assert comp.name == technology
    assert comp.commodity == "electricity"
    assert comp.hasCapacityVariable == False
    assert comp.operationRateFix is not None #values are checked within inputdatahandler 
    assert np.allclose(
        comp.operationRateFix.sum(axis=0),
        demand_2020
    )


    path_abs_demands = os.path.join(
        test_data_folder, 
        "input_data", 
        "demand_hydrogen",
        "<YEAR>",
        "hydrogen_demand_GWh_gid1.csv"
    )
    technology = "hydrogen_gas_demand"
    factor = 2
    mB.addDemand(
        technology=technology,
        year_demand=2050,
        factor = factor,
        path_abs_demands=path_abs_demands,
        path_ts=None
    )
    assert technology in mB.esM.componentNames.keys()
    assert mB.esM.componentNames[technology] == 'SourceSinkModel'
    comp = mB.esM.getComponent(technology)
    assert comp.name == technology
    assert comp.commodity == "hydrogen_gas"
    assert comp.hasCapacityVariable == False
    assert comp.operationRateFix is not None #values are checked within inputdatahandler 
    assert np.allclose(
        comp.operationRateFix.sum(axis=0),
        np.array([0, 0, 1677.532191]) * factor #real values

    )

def test_addWater(modelbuilder_fully_inited):
    '''test succeds, if function raises error.
    Makes sure, that if function gets implemented, a test has to be written. 

    Parameters
    ----------
    modelbuilder__init__ : _type_
        _description_
    '''
    mb = modelbuilder_fully_inited
    
    with pytest.raises(NotImplementedError):
        mb.addWater()

def test_addHydrogenGas(modelbuilder_fully_inited):
    '''test succeds, if function raises error.
    Makes sure, that if function gets implemented, a test has to be written. 

    Parameters
    ----------
    modelbuilder__init__ : _type_
        _description_
    '''
    mB = modelbuilder_fully_inited
    
    with pytest.raises(NotImplementedError):
        mB.addHydrogenGas()

def test_addLossOfLoad(modelbuilder_fully_inited):
    
    mB = modelbuilder_fully_inited

    path_VOLL = os.path.join(
        test_data_folder, 
        "input_data", 
        "VOLL",
        "VOLL_dummy.csv"
    )

    #cannot be inited without demand!
    with pytest.raises(KeyError):
        mB.addLossOfLoad(
            voll_to_BEUR_per_GWh_factor=None,
            path_VOLL=path_VOLL,
            voll_key="VOLL_2020[EUR/MWh]",
            sectoral_disaggregation=False,
            voll_factor=1,
            round=4
        )
    
    #add demand
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
        "All_Demand_UTC_2015_processed.csv",
    )

    technology="electricity_demand"
    factor = 1
    mB.addDemand(
        technology=technology,
        year_demand=2020,
        factor = factor,
        path_abs_demands=path_abs_demands,
        path_ts=path_ts
    )

    #add loss of load
    voll_factor = 2.1
    voll_to_BEUR_per_GWh_factor = 1.2
    mB.addLossOfLoad(
        voll_to_BEUR_per_GWh_factor=voll_to_BEUR_per_GWh_factor,
        path_VOLL=path_VOLL,
        voll_key="VOLL_2020[EUR/MWh]",
        sectoral_disaggregation=False,
        voll_factor=voll_factor,
        round=4
    )

    assert "Lull" in mB.esM.componentNames.keys()
    assert "electricity_demand" in mB.esM.componentNames.keys()
    assert mB.esM.componentNames["Lull"] == 'SourceSinkModel'

    comp = mB.esM.getComponent("Lull")
    assert comp.name == "Lull"
    assert comp.commodity == "electricity"
    assert comp.hasCapacityVariable == False
    assert comp.operationRateMax is not None #values are checked within inputdatahandler 
    assert np.allclose(
        comp.operationRateMax,
        mB.esM.getComponentAttribute("electricity_demand", "operationRateFix")
    )
    assert np.allclose(
        comp.commodityCostTimeSeries,
        1000 * voll_factor * voll_to_BEUR_per_GWh_factor
    )

    #add loss of load
    voll_factor = 2.1
    voll_to_BEUR_per_GWh_factor = 1.2
    mB.addLossOfLoad(
        voll_to_BEUR_per_GWh_factor=voll_to_BEUR_per_GWh_factor,
        path_VOLL=path_VOLL,
        voll_key="VOLL_2020[EUR/MWh]",
        sectoral_disaggregation=True,
        voll_factor=voll_factor,
        round=4
    )

    sectors = ['agriculture', 'industry', 'residential', 'services', 'transportation']
    for sector in sectors:
        name = f"Lull_{sector}"
        assert name in mB.esM.componentNames.keys()
        assert mB.esM.componentNames[name] == 'SourceSinkModel'

        comp = mB.esM.getComponent(name)
        assert comp.commodity == "electricity"
        assert comp.hasCapacityVariable == False
        assert comp.operationRateMax is not None #values are checked within inputdatahandler 
        assert np.allclose(
            comp.operationRateMax,
            mB.esM.getComponentAttribute("electricity_demand", "operationRateFix") * sectoral_demand_shares[sector]
        )

@pytest.mark.skip(reason="test not implemented yet")
def test_optimizeModel(modelbuilder_fully_inited):
    assert False

@pytest.mark.skip(reason="not implemented yet")
def test_saveToNC4(modelbuilder_fully_inited):
    assert False

def test_results_feasible(modelbuilder_fully_inited):
    
    mb = modelbuilder_fully_inited
    feasible = mb.results_feasible()
    assert not feasible

def test_has_results(modelbuilder_fully_inited):
    mb = modelbuilder_fully_inited
    has_results = mb.has_results()
    assert not has_results

def test__getCAPEXfromLCOE():

    ans = modelBuilder.modelManager._getCAPEXfromLCOE(
        LCOE_EUR_per_kWh=0.1,
        fixOPEX_CAPEX_per_a=0.02,
        varOPEX_notdefined=0,
        lifetime_a=20,
        WACC_1=0.08,
        meanCF= 1,
    )

    assert np.isclose(ans, 7.18903669010528)

    LCOE_EUR_per_kWh = pd.Series(
        [0.1, 0.2, 0.05],
        index = ['DEU.1_1','DEU.2_1', 'DEU.3_1']
    )
    ans_ser = modelBuilder.modelManager._getCAPEXfromLCOE(
        LCOE_EUR_per_kWh=LCOE_EUR_per_kWh,
        fixOPEX_CAPEX_per_a=0.02,
        varOPEX_notdefined=0,
        lifetime_a=20,
        WACC_1=0.08,
        meanCF= 1,
    )
    assert isinstance(ans_ser, pd.Series)
    assert list(ans_ser.index) == ['DEU.1_1','DEU.2_1', 'DEU.3_1']
    assert np.allclose(ans_ser.values, [7.18903669010528, 14.37807338021056, 3.59451834505264])    

def test__clip_close_to_zero():


    with pytest.raises(ValueError):
        result = modelBuilder.modelManager._clip_close_to_zero(
            array= 1,
            threshold=-1,
        )
    with pytest.raises(TypeError):
        result = modelBuilder.modelManager._clip_close_to_zero(
            array= "a",
            threshold=1,
        )
    
    #test scalars:
    threshold = 0.1
    tests = [
        (1, 1),
        (1.0, 1.0),
        (0.1, 0.1),
        (0.0999, 0),
    ]

    for input, output in tests:

        result = modelBuilder.modelManager._clip_close_to_zero(
            array= input,
            threshold=threshold,
        )
        assert np.isclose(result, output)

    #numpy
    result = modelBuilder.modelManager._clip_close_to_zero(
        array= np.array([0, 0.001, 0.01, 0.1, 1, 10]),
        threshold=0.1,
    )
    assert np.allclose(result, [0, 0, 0, 0.1, 1, 10])
    assert isinstance(result, np.ndarray)

    #pandas
    array = pd.DataFrame(
        data=[[0, 0.001, 0.01], [0.1, 1, 10]],
        index = ["a", "b"],
        columns = ["c", "d", "d"],
    )
    result_true_data = pd.DataFrame(
        data=[[0, 0, 0], [0.1, 1, 10]],
        index = ["a", "b"],
        columns = ["c", "d", "d"],
    )
    result = modelBuilder.modelManager._clip_close_to_zero(
        array= array,
        threshold=0.1,
    )
    assert np.allclose(result, result_true_data)
    assert isinstance(result, type(array))