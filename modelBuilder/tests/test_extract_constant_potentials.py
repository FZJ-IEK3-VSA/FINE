from modelBuilder.inputDataHandler.potentials.extract_constant_potentials import extract_potentials_csv, extract_potentials_sql, extract_potentials_shp
import geokit as gk
import numpy as np
import os
import pytest

from .test_data import test_data_folder

defaultregions_per_location_dict_default = {
        "BHR.3_1": ["BHR.3_1"],
        "BHR.4_1": ["BHR.4_1"],
        "BHR.5_1": ["BHR.5_1"]
    }

defaultregions_per_location_dict_agg = {
    "BHR.3_1": ["BHR.3_1", "BHR.5_1"],
    "BHR.4_1": ["BHR.4_1"],
    "BHR.5_1": ["BHR.5_1"]
}

@pytest.fixture
def location_shape_default():
    shape_file_path = os.path.join(test_data_folder, "input_data/test_regions.shp")
    shape = gk.vector.extractFeatures(shape_file_path)
    shape.rename(columns={'GID_1': 'locationID'}, inplace=True)
    shape["dflt_type"] = ["default", "default", "default"]
    return shape

@pytest.fixture
def location_shape_agg():
    shape_file_path = os.path.join(test_data_folder, "input_data/test_regions.shp")
    shape = gk.vector.extractFeatures(shape_file_path)
    shape.rename(columns={'GID_1': 'locationID'}, inplace=True)
    shape["dflt_type"] = ["agg", "default", "default"]
    return shape

@pytest.fixture
def location_shape_custom():
    shape_file_path = os.path.join(test_data_folder, "input_data/test_regions.shp")
    shape = gk.vector.extractFeatures(shape_file_path)
    shape.rename(columns={'GID_1': 'locationID'}, inplace=True)
    shape["dflt_type"] = ["custom", "custom", "custom"]
    return shape

def test_extract_potentials_csv(location_shape_default, location_shape_agg, location_shape_custom):
    
    csv_file_path = os.path.join(test_data_folder, "input_data/potentials/constant_potentials/salt_dummy.csv")
   
    # default regions 1 clusters
    data = extract_potentials_csv(
        path=csv_file_path,
        LCOE_name = "depth_LCC",
        capacity_name = "H2capacity",
        region_name_col="GID_1",
        capacity_conversion_factor= 1E3,
        LCOE_to_EUR_per_kWh_factor = 2,
        N_cluster=1,
        location_shape=location_shape_default,
        defaultregions_per_location_dict=defaultregions_per_location_dict_default,
        rounding=4,
        verbose=True,
    )
    
    assert np.allclose(data[0]["capacityMax"], [0, 0, 473.383977])#real data
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1733.33333*2])#real data
    


    # default regions 2 clusters
    data = extract_potentials_csv(
        path=csv_file_path,
        LCOE_name = "depth_LCC",
        capacity_name = "H2capacity",
        region_name_col="GID_1",
        capacity_conversion_factor= 1E3,
        LCOE_to_EUR_per_kWh_factor = 2,
        N_cluster=2,
        location_shape=location_shape_default,
        defaultregions_per_location_dict=defaultregions_per_location_dict_default,
        rounding=4,
        verbose=True,
    )
    
    assert np.allclose(data[0]["capacityMax"], [0, 0, 315.5893])#real data
    assert np.allclose(data[1]["capacityMax"], [0, 0, 157.7947]) #real data
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1600*2]) #real data
    assert np.allclose(data[1]["LCOE_EUR_per_kWh"], [1000000, 1000000, 2000*2])#real data

    assert np.allclose(
        data[0]["capacityMax"] + data[1]["capacityMax"],
        [0, 0, 473.383977]
    )



    # agg regions
    data = extract_potentials_csv(
        path=csv_file_path,
        LCOE_name = "depth_LCC",
        capacity_name = "H2capacity",
        region_name_col="GID_1",
        capacity_conversion_factor= 1E3,
        LCOE_to_EUR_per_kWh_factor = 2,
        N_cluster=1,
        location_shape=location_shape_agg,
        defaultregions_per_location_dict=defaultregions_per_location_dict_agg,
        rounding=4,
        verbose=True,
    )

    assert np.allclose(data[0]["capacityMax"], [0, 0, 473.383977])#real data
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1733.33333*2])#real data

    # custom regions
    data = extract_potentials_csv(
        path=csv_file_path,
        LCOE_name = "depth_LCC",
        capacity_name = "H2capacity",
        region_name_col="GID_1",
        capacity_conversion_factor= 1E3,
        LCOE_to_EUR_per_kWh_factor = 2,
        N_cluster=1,
        location_shape=location_shape_custom,
        defaultregions_per_location_dict=defaultregions_per_location_dict_default,
        rounding=4,
        verbose=True,
    )

    assert np.allclose(data[0]["capacityMax"], [0, 0, 473.383977])#real data
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1733.33333*2])#real data


def test_extract_potentials_sql(location_shape_default, location_shape_agg, location_shape_custom):
    
    path_sql = os.path.join(test_data_folder, "input_data/potentials/constant_potentials/dummy_geothermal.sqlite")
    #Ncluster = 1
    Ncluster = 1
    data = extract_potentials_sql(
        path=path_sql, 
        LCOE_name = "LCOE_GR",
        capacity_name = "Pnet_GR_MW",
        region_name_col="gid1",
        capacity_conversion_factor=1E-3,
        LCOE_to_EUR_per_kWh_factor = 1,
        N_cluster=Ncluster,
        location_shape=location_shape_default,
        defaultregions_per_location_dict=defaultregions_per_location_dict_default,
        rounding=4,
        verbose=True,
    )
    assert len(data) == Ncluster
    assert np.allclose(data[0]["capacityMax"], [0, 0, 0.0143])#real data
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 0.126])#real data

    #agg
    #Ncluster = 3
    Ncluster = 3
    data = extract_potentials_sql(
        path=path_sql, 
        LCOE_name = "LCOE_GR",
        capacity_name = "Pnet_GR_MW",
        region_name_col="gid1",
        capacity_conversion_factor=1E-3,
        LCOE_to_EUR_per_kWh_factor = 1,
        N_cluster=Ncluster,
        location_shape=location_shape_agg,
        defaultregions_per_location_dict=defaultregions_per_location_dict_agg,
        rounding=4,
        verbose=True,
    )

    assert len(data) == Ncluster

    assert np.allclose(data[0]["capacityMax"], [0.0096, 0, 0.0096])#real data
    assert np.allclose(data[1]["capacityMax"], [0, 0, 0])#real data
    assert np.allclose(data[2]["capacityMax"], [0.0048, 0, 0.0048])#real data

    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [0.0926, 1000000, 0.0926])#real data
    assert np.allclose(data[1]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1000000])#real data
    assert np.allclose(data[2]["LCOE_EUR_per_kWh"], [0.1926, 1000000, 0.1926])#real data


    #custom
    #Ncluster = 3
    Ncluster = 3
    data = extract_potentials_sql(
        path=path_sql, 
        LCOE_name = "LCOE_GR",
        capacity_name = "Pnet_GR_MW",
        region_name_col="gid1",
        capacity_conversion_factor=1E-3,
        LCOE_to_EUR_per_kWh_factor = 1,
        N_cluster=Ncluster,
        location_shape=location_shape_custom,
        defaultregions_per_location_dict=defaultregions_per_location_dict_default,
        rounding=4,
        verbose=True,
    )

    assert len(data) == Ncluster

    assert np.allclose(data[0]["capacityMax"], [0, 0, 0.0096])#real data
    assert np.allclose(data[1]["capacityMax"], [0, 0, 0])#real data
    assert np.allclose(data[2]["capacityMax"], [0, 0, 0.0048])#real data

    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 0.0926])#real data
    assert np.allclose(data[1]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1000000])#real data
    assert np.allclose(data[2]["LCOE_EUR_per_kWh"], [1000000, 1000000, 0.1926])#real data



def test_extract_potentials_shape(location_shape_default, location_shape_agg, location_shape_custom):

    shp_file_path = os.path.join(test_data_folder, "input_data/potentials/constant_potentials/salt_dummy.shp")


    # default regions 1 clusters
    data = extract_potentials_shp(
        path=shp_file_path,
        LCOE_name = "depth_LCC",
        capacity_name = "H2capacity",
        region_name_col="GID_1",
        capacity_conversion_factor= 1E3,
        LCOE_to_EUR_per_kWh_factor = 2,
        N_cluster=1,
        location_shape=location_shape_default,
        defaultregions_per_location_dict={"BHR.3_1": ["BHR.3_1"], "BHR.4_1": ["BHR.4_1"], "BHR.5_1": ["BHR.5_1"]},
        rounding=4,
        verbose=True,
    )
    
    assert np.allclose(data[0]["capacityMax"], [0, 0, 473.383977])#real data
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1733.33333*2])#real data



    #agg
    # default regions 1 clusters
    data = extract_potentials_shp(
        path=shp_file_path,
        LCOE_name = "depth_LCC",
        capacity_name = "H2capacity",
        region_name_col="GID_1",
        capacity_conversion_factor= 1E3,
        LCOE_to_EUR_per_kWh_factor = 2,
        N_cluster=1,
        location_shape=location_shape_agg,
        defaultregions_per_location_dict=defaultregions_per_location_dict_agg,
        rounding=4,
        verbose=True,
    )
    
    assert np.allclose(data[0]["capacityMax"], [473.383977, 0, 473.383977])#real data
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1733.33333*2, 1000000, 1733.33333*2])#real data

    #custom
    # default regions 1 clusters
    data = extract_potentials_shp(
        path=shp_file_path,
        LCOE_name = "depth_LCC",
        capacity_name = "H2capacity",
        region_name_col="GID_1",
        capacity_conversion_factor= 1E3,
        LCOE_to_EUR_per_kWh_factor = 2,
        N_cluster=1,
        location_shape=location_shape_custom,
        defaultregions_per_location_dict=defaultregions_per_location_dict_default,
        rounding=4,
        verbose=True,
    )
    
    assert np.allclose(data[0]["capacityMax"], [0, 0, 473.383977])#real data
    assert np.allclose(data[0]["LCOE_EUR_per_kWh"], [1000000, 1000000, 1733.33333*2])#real data





