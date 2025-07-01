from modelBuilder.inputDataHandler.grid import spatialDefinition
import pytest
import numpy as np
import pandas as pd
import geokit as gk
import os
import shutil

from .test_data import test_data_folder


@pytest.fixture
def spatial_def(ModelPaths_default):

    #model_base_folder = os.path.join(test_data_folder, "test_output_data")
    #os.makedirs(model_base_folder, exist_ok=True)

    location_shape_path = os.path.join(test_data_folder, "input_data", "test_regions.shp")
    location_shape = gk.vector.extractFeatures(location_shape_path)
    location_shape["locationID"] = location_shape.region_id

    spatialDef = spatialDefinition(
        shape=location_shape,
        region_name_col="locationID",
        #path_datafolder=model_base_folder,
    )

    yield spatialDef

    #shutil.rmtree(model_base_folder)

def test__bufferFineRegions(spatial_def):
    buffered_locations = spatial_def._bufferFineRegions()
    assert np.allclose(
        buffered_locations.geom.apply(lambda x: x.GetArea()).values,
        [0.0069189 , 0.0248465 , 0.05463934]
    )
    assert (buffered_locations.drop(columns="geom") == spatial_def.shape.drop(columns="geom")).all().all()

def test__getNeighboursAndDistances(spatial_def):
    regions_fine_buffered = spatial_def._bufferFineRegions()
    eligibility_matrix, distance_matrix = spatial_def._getNeighboursAndDistances(regions_fine_buffered=regions_fine_buffered)
    
    #only BHR.4_1 and BHR.5_1 are connected
    assert np.allclose(
        eligibility_matrix,
        eligibility_matrix_touching
    )

    #match distances from QGIS:
    assert np.allclose(
        distance_matrix,
        region_distance_m
    )


def test_extractEligibilityMatrix_km(spatial_def):

    detour_factor = 1.3

    #check it twice, as storing and reading could potentially mess up things...
    for i in range(2):
        spatial_def.extractEligibilityMatrix_km(
            detour_factor=detour_factor
        )

        assert np.allclose(
            spatial_def.eligibility_matrix,
            eligibility_matrix_touching_and_remote_islands
        )

        assert np.allclose(
            spatial_def.distance_matrix_km,
            region_distance_m / 1E3 * detour_factor
        )

        assert np.allclose(
            spatial_def.share_onshore_matrix,
            share_onshore_matrix
        )  


def test_return_dict(spatial_def):

    tranmission_dict = spatial_def.return_dict(detour_factor=1.3)
    
    vars = [
        'locationalEligibility',
        'distances',
        'share_onshore',
        'shape_transmission',
    ]
    for var in vars:
        assert var in tranmission_dict.keys()

def test__connectSubgraphs(spatial_def):
    
    #setting up data
    distance_matrix_pd = pd.DataFrame(
        region_distance_m/1000*1.3,
        index=spatial_def._getFineRegionNames(),
        columns=spatial_def._getFineRegionNames(),
    )

    eligibility_matrix_touching_pd = pd.DataFrame(
        eligibility_matrix_touching,
        index=spatial_def._getFineRegionNames(),
        columns=spatial_def._getFineRegionNames(),
    )
    eligibility_matrix_touching_pd_before = eligibility_matrix_touching_pd.copy()


    #testing
    eligibility_matrix_conSubgraph_pd = spatial_def._connectSubgraphs(
        eligibility_matrix_touching_pd,
        distance_matrix_pd,
    )

    
    #check if eligibility_matrix_conSubgraph_pd has the right index:
    assert isinstance(eligibility_matrix_conSubgraph_pd, pd.DataFrame)
    assert (eligibility_matrix_conSubgraph_pd.index == eligibility_matrix_touching_pd_before.index).all()
    assert (eligibility_matrix_conSubgraph_pd.columns == eligibility_matrix_touching_pd_before.columns).all()
    #check values
    assert np.allclose(
        eligibility_matrix_conSubgraph_pd.values,
        eligibility_matrix_touching_and_remote_islands
    )

    #check if original not changed
    assert np.allclose(
        eligibility_matrix_touching_pd,
        eligibility_matrix_touching_pd_before
    )

@pytest.fixture
def spatial_def_with_transmission_shp(spatial_def):
    
    #setup params
    spatial_def.distance_matrix_km = pd.DataFrame( #needed for shape and index, columns
        np.zeros(
            (
                len(spatial_def._getFineRegionNames()),
                len(spatial_def._getFineRegionNames())
            ),
            dtype=float,
        ),
        index=spatial_def._getFineRegionNames(),
        columns=spatial_def._getFineRegionNames(),
    )

    spatial_def.shp_transmission = pd.DataFrame(
        np.array(
            [
                ["BHR.3_1", "BHR.4_1", 1],
                ["BHR.3_1", "BHR.5_1", 1],
                ["BHR.4_1", "BHR.5_1", 1]
            ]
        ),
        index = range(3),
        columns = ["bus_0", "bus_1", "len_km"],
    )

    #add geoms
    spatial_def.shp_transmission["geom"] = [
        gk.geom.line([(50.5501, 25.9306), (50.5783, 26.0653)], srs=gk.srs.loadSRS(4326)), # only land
        gk.geom.line([(50.5501, 25.9306), (50.7413,25.9431)], srs=gk.srs.loadSRS(4326)), #partially land
        gk.geom.line([(0,0), (-1, -1)], srs=gk.srs.loadSRS(4326)), #no land (should ne happen, but why not test)
    ]

    return spatial_def

def test__get_share_onshore_offshore(spatial_def_with_transmission_shp):

    #shorten name
    sp = spatial_def_with_transmission_shp
    
    #set before params for comaprison
    shape_before = sp.shp_transmission.copy()
    wkts = sp.shp_transmission.geom.apply(lambda geom: geom.ExportToWkt()) #do this seperate, as deepcopy does not work for gdal objects (only pointers to C objects)

    #run the function
    sp._get_share_onshore_offshore()
    
    #check shp_transmission
    assert "share_onsh" in sp.shp_transmission.columns
    assert np.allclose(
        sp.shp_transmission.share_onsh.values,
        [1., 0.33144344, 0.]
    )
    #check original values
    assert (shape_before.drop(columns=["geom"]) == sp.shp_transmission.drop(columns=["geom", "share_onsh"])).all().all()
    #check geoms via wkts
    assert (sp.shp_transmission.geom.apply(lambda geom: geom.ExportToWkt()) == wkts).all()

    #check share_onshore_matrix
    for _, trans_row in sp.shp_transmission.iterrows():
        assert np.isclose(
            sp.share_onshore_matrix.loc[trans_row.bus_0, trans_row.bus_1],
            trans_row.share_onsh
        )
        assert np.isclose(
            sp.share_onshore_matrix.loc[trans_row.bus_1, trans_row.bus_0],
            trans_row.share_onsh
        )

#### true data

region_distance_m = np.array(
    [
        [    0.        , 21017.95857126, 33913.62744142],
        [21017.95857126,     0.        , 25808.90257725],
        [33913.62744142, 25808.90257725,     0.        ]
    ]
)

eligibility_matrix_touching = np.array(
    [
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.]
    ]
)

eligibility_matrix_touching_and_remote_islands = np.array(
    [
        [0., 1., 0.],
        [1., 0., 1.],
        [0., 1., 0.]
    ]
)

share_onshore_matrix = np.array(
    [
        [1.        , 0.28776037, 0.        ],
        [0.28776037, 1.        , 1.        ],
        [0.        , 1.        , 1.        ]
    ]
)