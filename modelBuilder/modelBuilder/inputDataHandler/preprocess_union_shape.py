#%%
from distutils.log import warn
import modelBuilder
import fine as fn
import os
import geokit as gk
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from modelBuilder.data import default_path_information
import shutil
import datetime

#TODO: Restructure

def preprocess_union_shape(location_shape, max_regions, return_as_gk, auxillary_data_folder=None):
    '''_summary_

    Parameters
    ----------
    location_shape : pd.DataFrame
        geokit DataFrame from union shape
    max_regions : int
        max number of regions

    Returns
    -------
    _type_
        _description_
    '''

    verbose = False
    union_No = None  # default: None

    #add time with ns time step, so that parallel jobs do not access same files.
    if not auxillary_data_folder:
        time = datetime.datetime.now()
        random_str = (str(time).replace("-", "").replace(" ", "").replace(".", "").replace(":", "")) + str(np.random.randint(100000))
        auxillary_data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), f"aux_dat_{random_str}")

    if len(location_shape) <= max_regions:
        print("No need to aggregate regions")
        return location_shape

    #############################
    ## Step 1: Add information to location_shape
    ############################# 

    location_shape = _add_islands_information(
        location_shape=location_shape
    )

    location_shape = _add_areas(
        location_shape=location_shape,
        verbose=False
    )

    #############################
    ## Step 2: calcualte the needed aggregations per country as current_nonisland_regions_per_country
    ############################# 
    
    country_areas = location_shape.groupby("GID_0")["area_km2"].sum()

    # get number per region
    initial_regions_per_country = location_shape.groupby("GID_0")["GID_0"].count()
    initial_regions_per_country.name = "regNo_init"

    current_nonisland_regions_per_country = (
        location_shape.groupby("GID_0")["GID_0"].count() - location_shape.groupby("GID_0")["nat_isol"].sum()
    )

    minimal_max_regions = location_shape.groupby("GID_0")["nat_isol"].sum().sum() + len(location_shape.GID_0.unique())

    if max_regions < minimal_max_regions:
        warn(f"User input requests spatial aggregation of GID-1s to {max_regions}. Can only agg down to {minimal_max_regions} regions, so I will do so!")
        max_regions = minimal_max_regions

    current_nonisland_regions_per_country = _needed_agg_per_gid0(
        current_nonisland_regions_per_country=current_nonisland_regions_per_country,
        location_shape=location_shape,
        max_regions=max_regions,
        country_areas=country_areas,
        verbose=verbose
    )

    # regNo_redudec
    nat_islands_per_gid0 = location_shape.groupby("GID_0")["nat_isol"].sum()
    regNo_red = current_nonisland_regions_per_country + nat_islands_per_gid0
    regNo_red.name = "regNo_red"

    #concat
    all_national_shapes_df = pd.concat(
        [
            location_shape[["GID_0", "NAME_0", "Union_No"]].groupby("GID_0").min(), #names
            initial_regions_per_country,
            regNo_red
        ],
        axis=1
    )

    all_national_shapes_df["reduction"] = 1 - (
        all_national_shapes_df["regNo_red"] / all_national_shapes_df["regNo_init"]
    )
    all_national_shapes_df.reset_index(inplace=True)

    # Convert geokit geom to geopandas geometry
    crs = 4326
    location_shape["geometry"] = gpd.GeoSeries.from_wkt(location_shape["geom"].apply(lambda x: x.ExportToWkt()))
    location_shape_gpd = gpd.GeoDataFrame(location_shape, geometry="geometry", crs=crs)
    location_shape_gpd = location_shape_gpd.to_crs(crs)


    new_regions_per_country_union_x = all_national_shapes_df.copy()

    container_regions = []

    for country in new_regions_per_country_union_x.GID_0.unique():
        
        #############################
        ## Step x: ? country shape 
        #############################
        #TODO: we wont store data, keep them in the RAM!

        country_data = new_regions_per_country_union_x.loc[new_regions_per_country_union_x["GID_0"] == country]

        regions_gpd = location_shape_gpd[location_shape_gpd["GID_0"] == country]
        regions_gk = location_shape[location_shape["GID_0"] == country].drop(columns=["geometry"])

        regions_to_process_gpd = regions_gpd[regions_gpd["nat_isol"] == 0]
        regions_islands_gpd = regions_gpd[regions_gpd["nat_isol"] == 1]
        regions_islands_gk = regions_gk[regions_gk["nat_isol"] == 1]

        # regNo_red = red + islands
        target_regions_for_current_country = country_data["regNo_red"].unique()[0] - len(regions_islands_gpd)

        if verbose:
            print(
                f"#####\nperforming aggregation for {country}. From {len(regions_gpd)} regions, {len(regions_islands_gpd)} islands to {target_regions_for_current_country} regions"
            )
        
        if len(regions_to_process_gpd) > 0:
            regions_processed = _aggregate_regions(
                country=country,
                shapefile=regions_to_process_gpd,
                target_regions_for_current_country=target_regions_for_current_country,
                auxillary_data_folder=auxillary_data_folder,
                return_as_gk=return_as_gk,
            )
        else:
            regions_processed = pd.DataFrame()


        regions_processed = regions_processed.rename(columns={"space": "GID_1"})
        regions_processed["Union_No"] = country_data["Union_No"].iloc[0]
        regions_processed["GID_0"] = country_data["GID_0"].iloc[0]
        regions_processed["NAME_0"] = country_data["NAME_0"].iloc[0]

        regions_islands = regions_islands_gk if return_as_gk else regions_islands_gpd
        regions_per_gid0_final = pd.concat([regions_islands, regions_processed]).reset_index(drop=True) #gk or pandas df!

        container_regions.append(regions_per_gid0_final)
    
    #concat
    if return_as_gk:
        regions_final = pd.concat(container_regions, axis=0)
    else:
        regions_final = gpd.GeoDataFrame(pd.concat(container_regions, axis=0))

    # make nice
    for col in ["region_id", "island", "nat_isol", "area_km2"]:
        if col in regions_final.columns:
            regions_final = regions_final.drop(columns=[col])
    
    regions_final = regions_final.sort_values("GID_1").reset_index(drop=True)

    #Tests!
    assert len(regions_final) == max_regions 
    if return_as_gk:
        final_area = regions_final.geom.apply(lambda g: g.GetArea()).sum()
    else:
        final_area = regions_final.geometry.area.sum()

    if isinstance(location_shape, pd.DataFrame):
        reference_area = location_shape.geom.apply(lambda g: g.GetArea()).sum()
    else:
        reference_area = location_shape.geometry.area.sum()
    
    assert np.isclose(
        final_area,
        reference_area
    ), "Area changed during preprocessing of union shape. That is not good and could be considered a BUG :("

    return regions_final



def _aggregate_regions(
    country,
    shapefile,
    target_regions_for_current_country,
    auxillary_data_folder=None,
    return_as_gk=True,
):
    if not auxillary_data_folder:
        auxillary_data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "aux_data")
    os.makedirs(os.path.dirname(auxillary_data_folder), exist_ok=True)

    aggregatedResultsPath = os.path.join(auxillary_data_folder)

    aggregation_function_dict = {}
    aggregated_xr_filename = "aggregated_xr_ds.nc"

    shapefile["index"] = shapefile["GID_1"]

    commodities = {"electricity", "hydrogen_gas"}
    commodityUnitsDict = {
            "electricity": r"GW$_{el}$",
            "hydrogen_gas": r"GW$_{H_{2},LHV}$",
        }
    esM = fn.EnergySystemModel(
            locations=set(shapefile["GID_1"]),
            commodities=commodities,
            numberOfTimeSteps=8760,
            commodityUnitsDict=commodityUnitsDict,
            hoursPerTimeStep=1,
            costUnit="1e9 Euro",
            lengthUnit="km",
            verboseLogLevel=0,
        )

    aggregated_esM = esM.aggregateSpatially( #TODO: why is aggregated_esM not needed? --> resutls are stored to aggregatedResultsPath
            shapefile=shapefile,
            grouping_mode="distance_based",
            n_groups=int(target_regions_for_current_country),
            aggregatedResultsPath=aggregatedResultsPath,
            aggregation_function_dict=aggregation_function_dict,
            shp_name=f"{country}",
            aggregated_xr_filename=aggregated_xr_filename,
            solver="gurobi",
            crs=4326,
        )
    

    #load auxillary files
    load_path = os.path.join(
        aggregatedResultsPath,
        "aggregated_regions.shp"
    )

    if return_as_gk:
        processed_shape = gk.vector.extractFeatures(load_path)
    else:
        processed_shape = gpd.read_file(load_path)
    

    #delete folder
    shutil.rmtree(aggregatedResultsPath)

    return processed_shape

def _add_islands_information(location_shape):
    
    shape_file = gk.vector.createVector(location_shape)
    
    islands = []
    national_isolated = []
    for reg_geom, country in tqdm(zip(location_shape.geom, location_shape.GID_0), total=len(location_shape)):
        sub_df = gk.vector.extractFeatures(shape_file, geom=reg_geom) #TODO: can we reference to the loation shape?
        # append a boolean if the region is an island
        islands.append(len(sub_df) == 1)
        # check if other regions from the same country are bordering, if not append to national isolated regions list
        national_isolated.append(len(sub_df[sub_df.GID_0 == country]) == 1)

    location_shape["island"] = islands
    location_shape["nat_isol"] = national_isolated
    return location_shape

def _add_areas(location_shape, verbose=False):
    # CALCULATE AREAS

    # define an aux function that returns area in sq. kms for a geom in lat/lon
    def get_area(geom):
        srs = gk.srs.centeredLAEA(geom.Centroid().GetX(), geom.Centroid().GetY())
        return gk.geom.transform(geom, toSRS=srs).Area() / 1000000  # return area in km²

    # add an area column with values ins sq kms
    areas = []
    for g in tqdm(location_shape.geom):
        areas.append(get_area(g))
    location_shape["area_km2"] = areas

    if verbose:
        print(round(location_shape.island.sum() / len(location_shape) * 100, 2), "% of regions are islands.")
        print(round(location_shape.nat_isol.sum() / len(location_shape) * 100, 2), "% of regions are nationally isolated.")
    
    return location_shape

def _needed_agg_per_gid0(current_nonisland_regions_per_country, location_shape, max_regions, country_areas, verbose):
    while (
        current_nonisland_regions_per_country.sum() + location_shape.groupby("GID_0")["nat_isol"].sum().sum()
        > max_regions
    ):
        assert (
            len(current_nonisland_regions_per_country[current_nonisland_regions_per_country > 1]) > 0
        ), f"No more nation left with more than 2 non-island regions."  # TODO this COULD continue even if the remaining 2 mainland regions are on different mainlands!
        # identify the biggest problem i.e. country with the smallest average region size
        gid0_highest_region_nr = (
            (country_areas / current_nonisland_regions_per_country)
            .sort_values()[
                (country_areas / current_nonisland_regions_per_country)
                .sort_values()
                .index.isin(
                    list(current_nonisland_regions_per_country[current_nonisland_regions_per_country > 1].index)
                )
            ]
            .index[0]
        )
        # reduce the region amount for the gid0_highest_region_nr
        current_nonisland_regions_per_country[gid0_highest_region_nr] = (
            current_nonisland_regions_per_country[gid0_highest_region_nr] - 1
        )
        if verbose:
            print(gid0_highest_region_nr, "reduced")
    
    #iteration finished
    return current_nonisland_regions_per_country



if __name__ == "__main__":

    #load shape
    shapeFilePath = default_path_information["general_data"]["default_regions_shp"]
    shape = gk.vector.extractFeatures(shapeFilePath)
    shape = shape[shape["Union_No"] == 23]

    max_regions = 40

    #test geokit
    shape_gk = preprocess_union_shape(
        location_shape=shape,
        max_regions=max_regions,
        return_as_gk=True,
    )

    #Test format
    assert isinstance(shape_gk, pd.DataFrame)
    assert len(shape_gk) == max_regions
    assert shape_gk.columns == ["geom", "GID_0", "NAME_0", "GID_1", "NAME_1", "Union_No"]

    assert np.close(
        shape_gk.geom.apply(lambda g: g.GetArea()).sum(),
        shape.geom.apply(lambda g: g.GetArea()).sum()
    )

    #more fany test for values depending on test setup!


    #test geopandas
    shape_gpd = preprocess_union_shape(
        location_shape=shape,
        max_regions=max_regions,
        return_as_gk=False,
    )
    #Test format
    assert isinstance(shape_gpd, gpd.GeoDataFrame)
    assert len(shape_gpd) == max_regions
    assert shape_gpd.columns == ["geometry", "GID_0", "NAME_0", "GID_1", "NAME_1", "Union_No"]

    assert np.close(
        shape_gpd.geometry.area.sum(),
        shape.geom.apply(lambda g: g.GetArea()).sum()
    )
     

