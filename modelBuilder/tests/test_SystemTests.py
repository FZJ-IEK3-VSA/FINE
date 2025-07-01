#from modelBuilder.inputDataHandler.inputDataHandler import inputDataHandler
import modelBuilder
from modelBuilder.modelManager.modelManager import modelManager
from modelBuilder.outputDataHandler.outPutDataHandler import OutputHandler
import pandas as pd
import geokit as gk
import numpy as np
import os
import glob
import pytest
import shutil

test_data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
model_base_folder = os.path.join(test_data_folder, "output_data", "test_SystemTests")
os.makedirs(model_base_folder, exist_ok=True)

from .test_grid import share_onshore_matrix, region_distance_m
from modelBuilder.inputDataHandler import preprocess_union_shape


@pytest.fixture
def setup_model_builder():
    regionsPath = os.path.join(test_data_folder, "input_data", "test_regions.shp")

    shape = gk.vector.extractFeatures(regionsPath)

    commodities = {"electricity", "hydrogen_gas"}
    commodityUnitsDict = {
                    "electricity": (r"GW$_{el}$", "GW"),
                    "hydrogen_gas": (r"GW$_{H_{2},LHV}$", "GW"),
                }

    ### Init Model Manager only writes vars to self.xyz
    modelBuilder = modelManager(
        location_shape=shape,
        locationID_column="GID_1",
        commodityUnitsDict=commodityUnitsDict,
        cost_year=2050,
        model_base_folder=model_base_folder,
        srs=4326,
        path_to_techno_economic_data=None,
        zero_threshold=0,
        default_regions_fp=regionsPath,
    )
    yield modelBuilder

    #clean up

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
        commodityUnitsDict=commodityUnitsDict,
        cost_year=2050,
        model_base_folder=model_base_folder, #Note: A new intermediates folder will be created in the same directory as your main git modelBuilder repository
        srs=4326,
        path_to_techno_economic_data_yaml=None, # Use default data
        complete_setup=False,
        default_regions_fp=location_shape_path,
    )

    yield modelManager

    shutil.rmtree(model_base_folder)

def test_setup_with_agg_region(modelbuilder__init__agg_regions):

    regionsPath = os.path.join(test_data_folder, "input_data", "test_regions.shp")
    mb = modelbuilder__init__agg_regions
    ### Create instances of data loader and esm Object
    mb.technoEconomicData_setup()
    mb.inputHandlerSetup(default_regions_shp=regionsPath)
    mb.modelSetup()

    assert list(mb.ih.defaultregions_per_location_dict.keys()) == ['BHR.3_1', 'BHR.4_1__BHR.5_1']
    assert list(mb.ih.defaultregions_per_location_dict['BHR.3_1']) == ['BHR.3_1']
    assert list(mb.ih.defaultregions_per_location_dict['BHR.4_1__BHR.5_1']) == ['BHR.4_1', 'BHR.5_1']

    assert list(mb.location_shape.dflt_type) == ['default', 'agg']



def test_system_model(setup_model_builder):
    '''Test for the whole workflow from building model to extracting results!

    Parameters
    ----------
    setup_model_builder : modelManager
        see fixture
    '''

    mb = setup_model_builder
    ### Create instances of data loader and esm Object
    mb.technoEconomicData_setup()
    mb.inputHandlerSetup()
    mb.modelSetup()

    ### create model
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
    mb.addDemand(
        technology="electricity_demand",
        year_demand=2020,
        factor=1,
        path_abs_demands=path_abs_demands,
        path_ts=path_ts,
    )

    cap_base_path=os.path.join(
        test_data_folder, 
        "input_data", 
        "potentials",
        "OFPV_fixed/Base/v20230101/<GID0>/100m/<WEATHERYEAR>/plant_vars/plant_vars__OFPV_fixed__<GID1>__SG<SPATGROUP>__<WEATHERYEAR>__100res__Base.pickle"
    )
    ts_base_path=os.path.join(
        test_data_folder, 
        "input_data", 
        "potentials",
        "OFPV_fixed/Base/v20230101/<GID0>/100m/<WEATHERYEAR>/cluster_vars/cluster_vars__OFPV_fixed__<GID1>__SG<SPATGROUP>__<WEATHERYEAR>__100res__Base.nc4"
    )

    mb.addPotentialWithTSGreenfield(
        "ofpv_fixed",
        cap_base_path=cap_base_path,
        ts_base_path=ts_base_path,
        N_clusters=3,
        commodity="electricity",
        model_unit="GW",
    )

    mb.addGridGreenfield(
        technology="electricity_grid",
        detour_factor=1.3,
    )
    mb.addStorageUnlimitedGreenfield(technology='battery_LIIon')


    ### check set upped model
    assert sorted(list((mb.esM.componentNames.keys()))) \
        == ['battery_LIIon', 'electricity_demand','electricity_grid', 'ofpv_fixed__0', 'ofpv_fixed__1', 'ofpv_fixed__2']

    ### demand
    assert np.allclose(
        mb.esM.getComponentAttribute("electricity_demand", "operationRateFix").sum(axis=0).values,
        [2371.40478843, 3289.43683931, 7027.52176944]   #real values
    )
    assert np.allclose(
        mb.esM.getComponentAttribute("electricity_demand", "operationRateFix").std(axis=0).values,
        [0.10403347, 0.14430751, 0.30829721]   #real values
    )

    ### transmission
    assert sorted(list(mb.esM.getComponent("electricity_grid").locationalEligibility.index)) \
        == ['BHR.3_1_BHR.4_1', 'BHR.4_1_BHR.3_1', 'BHR.4_1_BHR.5_1', 'BHR.5_1_BHR.4_1']  #real values
    
    detour_factor = 1.3
    region_distance_m_format_fine = np.array([
            (region_distance_m / 1E3 * detour_factor)[0,1],
            (region_distance_m / 1E3 * detour_factor)[1,0],
            (region_distance_m / 1E3 * detour_factor)[2,1],   #real values
            (region_distance_m / 1E3 * detour_factor)[2,1],
        ])

    assert np.allclose(
        mb.esM.getComponent("electricity_grid").distances.values,
        region_distance_m_format_fine
    )
    assert np.allclose(
        mb.esM.getComponent("electricity_grid").investPerCapacity.values,
        np.array(
            [
                [0.00086   , 0.00117339, 0.0013    ],
                [0.00117339, 0.00086   , 0.00086   ],   #real values
                [0.0013    , 0.00086   , 0.00086   ]
            ]
        )
    ) 

    ### storages
    assert "battery_LIIon" in list((mb.esM.componentNames.keys()))

    ### potentials
    capacity = mb.esM.getComponent("ofpv_fixed__0").capacityMax.values \
        + mb.esM.getComponent("ofpv_fixed__1").capacityMax.values\
            + mb.esM.getComponent("ofpv_fixed__2").capacityMax.values
    assert np.allclose(
        capacity,
        np.array([4000, 1203000, 9861500])*1E-6 #real values from potentials! dont change unless there is a reason
    )

    ### run model
    kwargs_opt = {
        "solver" : "glpk",
        # "solver" : "gurobi",
    }
    mb.optimizeModel(
        numberOfTypicalPeriods=30,
        kwargs_opt=kwargs_opt,
        threads=1
    )

    print("mb.esM.objectiveValue", mb.esM.objectiveValue)
    assert np.isclose(
        mb.esM.objectiveValue,
        1.6405814888664576
        #1.665822189677932, # model output values, only plausible
    ) # old value befor transmission cost update: 1.637768878067714
    
    mb.saveToNC4()
   
    ### do the postprocessing
    postpro = OutputHandler(
        model_base_folder=model_base_folder,
        xr_dss=None,
        regions_shp=None,
        transmission_shp=None,
    )
    postpro.store_standard_evaluation()
    postpro.store_default_plots()

    assert os.path.isdir(os.path.join(model_base_folder, "ESM_summary"))
    assert len(glob.glob(os.path.join(model_base_folder, "ESM_summary", "standard_plots", "capacity*.png"))) > 0
    assert len(glob.glob(os.path.join(model_base_folder, "ESM_summary", "standard_plots", "FLH*.png"))) > 0
    assert len(glob.glob(os.path.join(model_base_folder, "ESM_summary", "standard_plots", "operation*.png"))) > 0

    assert os.path.isfile(os.path.join(model_base_folder, "ESM_summary", "summary.csv"))
    assert os.path.isfile(os.path.join(model_base_folder, "ESM_summary", "summary_regions.csv"))

    results = pd.read_csv(os.path.join(model_base_folder, "ESM_summary", "summary.csv"), index_col=[0])

    #results.to_csv("results.csv") #let this be here fore debugging the test.
    # Debugging at this part is hard, because glpk does not run in the debugger
    
    print("mb.results.capacity", results.capacity)
    assert np.allclose(
        results.capacity.values,
        [30.24298264,  0.        ,  2.33452888,  9.65666839], #only plausible values  # adapted, as losses were incorectly high by 100
        #[30.02014994,  0.        ,  7.78915497, 10.28100742], # model output values, only plausible
        rtol = 0.01
    )




    



