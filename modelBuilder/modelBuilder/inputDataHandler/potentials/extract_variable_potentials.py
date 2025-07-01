# import standard packages
from copy import copy
import datetime
import glob
import numpy as np
import os
import pandas as pd
import random
import re
import time
import warnings
import yaml

# import third party packages
from ast import literal_eval
import geokit as gk
from geopandas import GeoDataFrame
import natsort
from natsort import natsort_keygen
from osgeo import ogr
from tqdm import tqdm
import xarray as xr

# import other modules
if not __name__ == '__main__':
    from ...data import data_folder
    from ... import utils
    from modelBuilder.singletons import InputDataInfo, UnitHandling

aggregation_function_mapper = {
    "mean": np.mean,
    "max": max,
    "sum": np.sum,
    "min": min,
}


class NC4Source():
    """Class for loading potential data from GlobEP nc4 and .shp/.pickle format."""
    def __init__(self, 
        technology, 
        model_unit,
        verbose=True,
        ):
        '''
        This class serves to load potentials data for a specific technology, scenario and weather name
        from disk when in standard .nc4 (timeseries) and .shp/.pickle (capacities) format as it is 
        produced by the Global Energy Potentials (GlobEP) package.

        Parameters
        ----------
        technology : str
            Technology for which potentials shall be loaded. Must be a key in default_potentials.yaml (data folder) 
            to allow loading of default potentials, else all variables below must be specified explicitly. 
        data_unit : str, optional
            The capacity unit used in the potentials data to be loaded, e.g. 'kW'. Will be extracted from default information if
            default potentials are used, else required. Defaults to None.
        verbose : bool, optional
            If True, additional progress statements will be printed, by default True.
        '''

        # define specific placements and time series base filepaths for the respective technology
        input_path_placements = InputDataInfo().update_and_get_path(
            tech=technology, 
            path_attr="cap_base_path", 
            add_spacer_mapper={}, 
            must_exist=False,
            no_more_spacers=False,
        )
        input_path_timeseries = InputDataInfo().update_and_get_path(
            tech=technology, 
            path_attr="ts_base_path", 
            add_spacer_mapper={}, 
            must_exist=False,
            no_more_spacers=False,
        )

        # issue a print statement detailing the paths that have been composed, helpful especially for non default-potentials
        if verbose: 
            print(f"The following base paths for placements and timeseries were defined: \n{input_path_placements}\n{input_path_timeseries}", flush=True)

        # the capacity_vars class attribute holds ALL variables together in the source that shall be aggregated
        self.aggregation_dict = InputDataInfo().get_info(tech=technology, attr="aggregation_dict")
        self.additional_aggregation_vars = InputDataInfo().get_info(tech=technology, attr="additional_aggregation_vars")
        capacity_vars = list(self.aggregation_dict.values())
        if not self.additional_aggregation_vars is None:
            capacity_vars += list(self.additional_aggregation_vars.keys())
        self.capacity_vars = natsort.natsorted(np.unique(capacity_vars))

        #make the aggregation mapper
        capacities_agg = list(self.aggregation_dict.values())
        agg_mapper_placements = {var: sum for var in capacities_agg}
        if self.additional_aggregation_vars:
            for additional_aggregation_var in self.additional_aggregation_vars:
                agg_str = self.additional_aggregation_vars[additional_aggregation_var]
                agg_func = aggregation_function_mapper[agg_str.lower()]
                agg_mapper_placements.update({additional_aggregation_var: agg_func})
        agg_mapper_placements.update({'locationID': max, 'LCOE_clstr': np.mean})
        self.agg_mapper_placements = agg_mapper_placements

        ts_vars = list(self.aggregation_dict.keys())
        agg_mapper_ts = {var: sum for var in ts_vars}
        agg_mapper_ts.update(self.agg_mapper_placements)
        self.agg_mapper_ts = agg_mapper_ts


        # load/save a list of all global default regions once here to access iteratively later
        with open(os.path.join(data_folder, "GID_1_split_region_codes.txt")) as f:
            all_gid1s_str = f.read()
        self.all_global_default_regions = all_gid1s_str[1:-2].split("','")

        # calculate a power unit conversion factor for the given model/potentials data combination
        self.capacity_conversion_factor = UnitHandling().get_unit_conversion_factor(
            input_unit=InputDataInfo().get_info(tech=technology, attr="data_unit"), 
            target_unit=model_unit
        )

        # save other key parameters
        self.technology = technology
        self.weather_year = InputDataInfo().weather_year
        self.negative_ts = InputDataInfo().get_info(tech=technology, attr="negative_ts")
        self.sub_dataset_name = InputDataInfo().get_info(tech=technology, attr="sub_dataset_name")
        self.input_path_placements = input_path_placements
        self.input_path_timeseries = input_path_timeseries
        self.daily_timeseries=InputDataInfo().get_info(tech=technology, attr="daily_timeseries")
        self.hourly_reference_timeseries=InputDataInfo().get_info(tech=technology, attr="hourly_reference_timeseries")
    
    def load_placements_per_default_region(self, gid1):
        """
        Extracts all placements for a given GID_1split region for the class-inherent technology and scenario etc.

        gid1 : str
            Region for which the potentials shall be extracted. Must be a GID_1split region (i.e. modelBuilder default region).

        Returns
        -------
        pd.DataFrame 
            A dataframe with all potentials plant locations in the given region.
        """
        # first ensure the gid1 code is indeed a default region
        assert isinstance(gid1, str), f"gid1 all_global_default_regions be str formatted."
        assert gid1 in self.all_global_default_regions, f"gid1 {gid1} is not a GID_1split default region!"

        # extract country alpha-3 code from region code
        gid0 = gid1[0:3]
        
        #get actual paths:
        input_path_placements_region = copy(self.input_path_placements)
        input_path_placements_region = input_path_placements_region.\
            replace("<GID0>", str(gid0)).replace("<GID1>", str(gid1)).replace('<SPATGROUP>', '*')
        # filter for available regional files
        paths_region = glob.glob(input_path_placements_region)
        #TODO: Instead of below, check that number of spatial groups is as expected- against what?
        if len(paths_region)==0:
            print(f"No placement files found for region {gid1} and the following base path: {input_path_placements_region}", flush=True)

        # iterate over all found files (i.e. one per eligibile spatial group in this region) to load data
        data_per_file = [] #container
        for path in paths_region:
            if os.path.splitext(path)[-1] == ".pickle":
                # if pickles shall be used, load in pandas
                data_file = pd.read_pickle(path)
            else:
                assert os.path.splitext(path)[-1] in ['.shp', '.geojson'], f"cap_base_path must have one of the following extensions: .pickle, .shp, .csv. Here: {path}"
                # if .shp shall be used instead of .pickle, load shapefile via geokit
                data_file = gk.vector.extractFeatures(path)
            #filter placements
            if not self.sub_dataset_name is None:
                data_file = data_file[data_file['dsetname'] == self.sub_dataset_name] #TODO make 'dsetname' dynamic. Again.
            # first process 'ERA5_cell' column if exists, was an old version of cell_No
            if 'ERA5_cell' in data_file.columns:
                assert not 'cell_No' in data_file.columns, \
                    f"Data has both 'ERA5_cell' and 'cell_No' columns. Spatial group can be defined only in one column ('cell_No' is the updated version)."
                data_file.rename(columns={"ERA5_cell": "cell_No"}, inplace=True)
            # append to list of dataframes to be concatenated later
            data_per_file.append(data_file)

        if len(paths_region) == 0:
            # since zero files were expected for this gid1 (#TODO implement above!), a dummy data file is created
            dummy = self._make_dummy_df()
            data_per_file.append(dummy)
        placements_per_region = pd.concat(data_per_file, axis=0)
        
        return placements_per_region
        
    
    def reduce_placement_polygons_to_location_shape(
        self, 
        placements_df, 
        location_shape, 
        capacity_attributes=None,
        drop_polygons_outside_location_shape=True,
        ):
        """
        This method reduces polygon placements to sub polygons that fall within a given 
        location shape boundary, clipping the placement polygon at the boundary of the 
        region/location shape. At the same time, a given capacity attribute is scaled
        linearly with the placement area proportion falling into the region/location.

        Parameters
        ----------
        placements_df : pd.DataFrame
            geokit shape file with geoms and location ID
        location_shape : osgeo.ogr.Geometry
            The geometry shape of the location/region.
        capacity_attributes : str or list of str
            Name of the capacity attribute which must contain int or float values.
            These values will be reduced in case that the placement polygon only
            partially overlaps with the location shape geometry. If None, default
            potential information must be available for the given technology, by 
            default None.
            NOTE: If a list of capacity attributes is passed or stored as a class
            attribute, all capacity attributes in the list will be scaled alike.
        drop_polygons_outside_location_shape : bool, optional
            If True, an additional vectorizing step will drop all polygons that do
            not touch the location shape at all, i.e. are completely outside the
            location/region. Can be set to False to save time if it can be ensured 
            that no polygons are outside the location or these do not matter. By 
            default True.
        Returns
        -------
        pd.DataFrame 
            The dataframe with the adapted 'geom' and capacity attribute columns
        """
        assert isinstance(placements_df, pd.DataFrame), "placements_df must be a pd.DataFrame"
        assert 'geom' in placements_df.columns, f"location_shape must contain geom column"
        assert isinstance(location_shape, ogr.Geometry), f"location_shape must be an ogr.Geometry (multi)polygon."

        if capacity_attributes is None:
            # ensure that we have a default potentials technology and hence default capacity_attributes
            assert not self.capacity_vars is None, f"If non-default technology is applied, capacity_attributes must be passed as input parameters."
            # if so, set default value as capacity_attributes
            capacity_attributes = self.capacity_vars
        elif isinstance(capacity_attributes, str):
            capacity_attributes=[capacity_attributes]
        # capacity_attributes=[attr[4:] for attr in capacity_attributes]
        assert isinstance(capacity_attributes, list), "capacity_attributes must be a str or list of str"
        assert all([attr in placements_df.columns for attr in capacity_attributes]), "All capacity_attributes must be column names in placements_df"

        # if we have extended area potentials like OFPV, we must clip them to the location shapes
        if "POLYGON" in placements_df.geom.iloc[0].GetGeometryName() and len(placements_df) > 0:
            srs=placements_df.geom.iloc[0].GetSpatialReference()
            assert all([g.GetSpatialReference().IsSame(srs) for g in placements_df.geom]), f"All geometries in placements_df['geom'] must have the same srs!"
            location_shape = gk.geom.transform(location_shape, toSRS=srs)

            if drop_polygons_outside_location_shape:
                # vectorize the plant dataframe and reload only these that touch the location shape
                placements_vec = gk.vector.createVector(placements_df)
                placements_df = gk.vector.extractFeatures(source=placements_vec, geom=location_shape, srs=srs)
                del placements_vec

            if len(placements_df)==0:
                # if we have no locations left after removing those outside the region shape, return the empty placement dataframe
                return placements_df

            # add an identifier and create a vector to only extract the boundaries
            placements_df["identifier"] = range(len(placements_df))
            placements_df["area_before"] = [g.Area() for g in placements_df.geom]
            # proceed only if dataframe is not empty
            if len(placements_df) > 0:
                placements_vec = gk.vector.createVector(placements_df)
                # extract only these polygons crossing the location shape edges (not the inner ones)
                placements_outer_df = gk.vector.extractFeatures(
                    source=placements_vec, geom=location_shape.Boundary(), srs=srs,
                )
                del placements_vec

                # proceed only if dataframe is not empty
                if len(placements_outer_df) > 0:
                    # clip the geoms
                    clipped_geoms = [location_shape.Intersection(g) for g in placements_outer_df.geom]
                    # replace these geoms in the intial reduced dataframe that overlap the location boundaries (with clipped geoms)
                    placements_df.loc[
                        placements_df.identifier.isin(list(placements_outer_df.identifier)), "geom"
                    ] = clipped_geoms
                # add another area column after clipping these geoms that overlapped the region boundary
                placements_df["area_clipped"] = [g.Area() for g in placements_df.geom]
                # now calculate the new, remaining capacity per polygon, based on an even capacity distribution per area and polygon
                for capacity_attribute in capacity_attributes:
                    placements_df[capacity_attribute] = (
                        placements_df[capacity_attribute] * placements_df["area_clipped"] / placements_df["area_before"]
                    )
        
        return placements_df


    def _make_dummy_df(self):
        """Creates a dummy placement dataframe for regions with zero potential."""
        # generate a list of column names that must be in the dummy dataframe
        dummy_columns = \
            list(self.aggregation_dict.keys()) \
            + self.capacity_vars \
            + ['locationID', self.region_col_name, 'cell_No', 'LCOE_clstr', 'ts_path'] 
        # return an empty dataframe with the expected column names
        return pd.DataFrame(columns=sorted(set(dummy_columns)))
    
    def load_placements(
        self, 
        location_shape, 
        defaultregions_per_location_dict, 
        use_partial_polygon_capacities=True, 
        region_col_name ="region",
        verbose=False
        ):
        '''
        Loading all placements for a given location_shape dataframe with osgeo.ogr.Geometry objects in 
        'geom' column (geokit dataframe).

        Parameters
        ----------
        location_shape : pd.DataFrame
            Dataframe with osgeo.ogr.Geometry 'geom' column and 'locationID' column for all model locations.
            Dataframe with osgeo.ogr.Geometry 'geom' column and 'locationID' column for all model locations.
        use_partial_polygon_capacities : bool, optional
            If True, the capacity attributes of each polygon-shaped plant will be reduced linearly with the 
            plant area percentage that actually overlaps with the respective model location. Set to False to 
            save time at the cost of possibly (partially) counting plant capacities twice that overlap the 
            location shape borders. Will only be applied to custom-shaped model locations and if possible, i.e. 
            when the geom column contains (multi)polygon geometries. By default True.
        region_col_name : str, optional
            Indicates the name of the (default) region column attribute in the placement shapefile/pickle to
            be loaded, by default 'region'.
        verbose : bool, optional
            If True, additional progress statements will be printed, by default False

        Returns
        -------
        -

        Sets
        ----
        self.cluster_agg : pd.DataFrame
            dataframe with all placements aggregated by lcoe-sg-clusters and their belonging ts file reference 
        '''
        # I/o handling:
        if isinstance(location_shape, GeoDataFrame):
            assert 'geom' in location_shape.columns, f"location_shape must have a 'geom' column with osgeo.ogr.Geometry polygons."
            location_shape = pd.DataFrame(location_shape.drop(columns=['geometry']))
        else:
            pass

        # extract model location IDs
        assert 'locationID' in location_shape.columns, f"location_shape dataframe must have a 'locationID' column with model location names."
        locationIDs = natsort.natsorted(location_shape.locationID)
        self.locationIDs = locationIDs
        assert 'locationID' in location_shape.columns, f"location_shape dataframe must have a 'locationID' column with model location names."
        locationIDs = natsort.natsorted(location_shape.locationID)
        self.locationIDs = locationIDs
        # self.additional_vars = additional_vars
        # self.dict_agg = dict_agg
        self.region_col_name = region_col_name
        
        cluster_agg = []

        #iterate regions
        for i_loc, locationID in enumerate(locationIDs):
            if verbose: print(f"Now extracting placements for location ID:", locationID, f"(location {i_loc+1}/{len(locationIDs)})", flush=True)
            
            dflt_type = location_shape[location_shape.locationID == locationID].iloc[0].dflt_type
            default_regions = defaultregions_per_location_dict[locationID]
        
            if dflt_type in ["default", "agg"]:
                placements_list = []
                for default_region in default_regions:
                    placements_per_default_location = self.load_placements_per_default_region(gid1=default_region)
                    if "wkb" in placements_per_default_location.columns: placements_per_default_location.drop(columns=['wkb'], inplace=True) 
                    if "geom" in placements_per_default_location.columns: placements_per_default_location.drop(columns=['geom'], inplace=True) 
                    placements_list.append(placements_per_default_location)
                placements_per_location = pd.concat(placements_list, axis=0)
                del placements_list, placements_per_default_location # save mem
            else:
                # geospatially extract all affected placements within the overlapping default regions

                # first check that necessary location shapes are passed and in right format
                assert isinstance(location_shape, pd.DataFrame) and 'geom' in location_shape.columns, f"location_shapes  must be passed as pd.DataFrame with a 'geom' column."
                assert all([isinstance(g, ogr.Geometry) for g in location_shape.geom]), f"All entries in location_shapes['geom'] column must be osgeo.ogr.Geometry objects."
                assert 'locationID' in location_shape.columns, f"location_shapes dataframe must have column 'locationID' with model location names."

                # extract placements per each affected default region #TODO replace this whole block by the new "extractAndClipFeatures()" method and process manually only when pickles/csv without 'geom' column are involved
                df_list = [] 
                for def_reg in defaultregions_per_location_dict[locationID]:
                    df_list.append(self.load_placements_per_default_region(gid1=def_reg))
                # then merge all default regions together
                df_list = [df for df in df_list if len(df)>0] #drop empty dfs that cannot be merged

                if len(df_list) > 0:
                    placements_per_affected_default_regions = pd.concat(df_list)

                    # now vectorise and extract only truly affected locations
                    # first create geometries from wkb if needed
                    if not 'geom' in placements_per_affected_default_regions.columns:
                        srid = placements_per_affected_default_regions['srs_epsg'].iloc[0]
                        srs = gk.srs.loadSRS(int(srid))
                        geoms = list(placements_per_affected_default_regions["wkb"].apply(lambda wkb: ogr.CreateGeometryFromWkb(wkb)))
                        geoms_w_srs = []
                        for g in geoms:
                            g.AssignSpatialReference(srs)
                            geoms_w_srs.append(g)
                        placements_per_affected_default_regions['geom'] = geoms_w_srs
                        del geoms, geoms_w_srs
                    else:
                        srs=placements_per_affected_default_regions.geom.iloc[0].GetSpatialReference()
                        assert all([srs.IsSame(g.GetSpatialReference()) for g in placements_per_affected_default_regions.geom]), f"The placements geom column contains geometries with different SRS!"
                    if 'wkb' in placements_per_affected_default_regions.columns:
                        # drop wkb to save space
                        placements_per_affected_default_regions.drop(columns=['wkb'], inplace=True)

                    # now extract only these plants that are within the location shape polygon of the given locationID
                    locationID_geom = location_shape[location_shape.locationID == locationID].geom
                    assert len(locationID_geom) == 1
                    locationID_geom = locationID_geom.iloc[0]

                    # if we have polygon-shaped plants and those overlapping the borders shall be clipped to areas within location shape, proceed here #TODO replace this whole block by using the new gk feature extractAndClipFeatures() above in placement extraction
                    if use_partial_polygon_capacities and 'POLYGON' in placements_per_affected_default_regions.geom.iloc[0].GetGeometryName():
                        # Note: This needs to be applied before centroid extraction below since geom column will be overwritten with centroid points
                        placements_per_location = self.reduce_placement_polygons_to_location_shape(
                            placements_df=placements_per_affected_default_regions, 
                            location_shape=locationID_geom, 
                            capacity_attributes=self.capacity_vars,
                            drop_polygons_outside_location_shape=True,
                            )
                    else: #TODO add elif to avoid double geom creation if we already have a "geom" column with POINT GeometryNames
                        # otherwise, we will extract the polygons by their centroids, this approach also works for points
                        if "lon" in placements_per_affected_default_regions.columns and "lat" in placements_per_affected_default_regions.columns:
                            # try to use lat/lon attributes if available
                            placements_per_affected_default_regions['geom'] = list(placements_per_affected_default_regions[["lon", "lat"]].apply(lambda x: gk.geom.point(x.lon, x.lat, srs=srs), axis=1))
                        else:
                            # if we do not have the centroid lon/lat yet, use geospatial centroids of polygons here to assign a unique model location
                            # the centroid extraction can also be applied to points, returns the same point
                            placements_per_affected_default_regions['geom'] = placements_per_affected_default_regions['geom'].apply(lambda x : x.Centroid())

                        # then extract only these plant locations that touch the model locationID considered here
                        # therefore use vectorizing of dataframe and extractFeatures(vector, geom)
                        vec = gk.vector.createVector(placements_per_affected_default_regions)
                        placements_per_location = gk.vector.extractFeatures(vec, geom = locationID_geom)
                        del vec, locationID_geom


                    # drop geom column to save memory
                    placements_per_location.drop(columns=['geom'], inplace=True) 

                else:
                    #create an empty dataframe for placements_per_location, this will trigger a proper dummy below
                    placements_per_location = pd.DataFrame()
            
            # create dummy
            if len(placements_per_location) == 0:
                #create dummy if empty
                placements_per_location = self._make_dummy_df()
 	       
            placements_per_location['locationID'] = locationID 

            for var in ['locationID', 'LCOE_clstr']: #TODO how to deal with 'cell_No' (not always needed?)
                assert var in placements_per_location.columns

            #make the time series identifier: (path lcoe_cluster)
            def make_identifier(gid1, cell_No, LCOE_clstr):
                gid0 = gid1[0:3]
                if not isinstance(cell_No, tuple):
                    cell_No = literal_eval (cell_No)
                str_cell_No =f'{str(int(cell_No[0])).zfill(4)}-{str(int(cell_No[1])).zfill(4)}'

                # if the sub_dataset_name is None, the __<SUBDATASETNAME> part in the basepath shall go completely
                # else we need to replace the spacer with the sub dataset name, preceded by two underscores
                SUBDATASETNAME_replacer = '' if self.sub_dataset_name is None else '__'+str(self.sub_dataset_name)
                path_to_ts = self.input_path_timeseries.replace(
                    "<GID0>", str(gid0)).replace(
                    "<GID1>", str(gid1)).replace( #TODO be consistent with caps and ts - once GID1split is set, once GID1split_off*
                    '<SPATGROUP>', str(str_cell_No)).replace(
                    '__<SUBDATASETNAME>', SUBDATASETNAME_replacer)
                
                if self.sub_dataset_name is None:
                    str_readable = f"{str(gid1)}_{str(str_cell_No)}_{str(LCOE_clstr)}"
                    return (path_to_ts, LCOE_clstr, str_readable)
                else: 
                    str_readable = f"{str(gid1)}_{str(str_cell_No)}_{str(LCOE_clstr)}_{str(self.sub_dataset_name)}"
                    return (path_to_ts, LCOE_clstr, self.sub_dataset_name, str_readable)      

            if len(placements_per_location) > 0:
                # generate a list of identification variables
                if '-' in list(placements_per_location.cell_No): #TODO remove, only for hydropower now
                    # as a preliminary fix, add dummy cell_No column #TODO remove when fixed properly
                    placements_per_location['cell_No']=[(0.0, 0.0)]*len(placements_per_location)
                id_vars = [self.region_col_name, 'cell_No', 'LCOE_clstr']
                # if not self.sub_dataset_name is None: id_vars.extend(list(self.sub_dataset_name)) #TODO remove since it is already in make_identifier self.sub_dataset_name?
                # generate a path to timeseries, return LCOE_cluster and an additional identifier string
                placements_per_location["ts_identifier"] = placements_per_location[id_vars].\
                    apply(lambda x: make_identifier(*x), axis=1)
            else: 
                placements_per_location["ts_identifier"] = []

            # aggregate by identifier
            cap_vars = self.capacity_vars.copy()
            # cap_vars = [attr[4:] for attr in cap_vars]
            cap_vars.append("ts_identifier")
            capacities_agg = placements_per_location.groupby("ts_identifier").agg(self.agg_mapper_placements).reset_index() 
            # unpack identifier and assign to individual columns
            capacities_agg["ts_path"] = capacities_agg["ts_identifier"].apply(lambda x: x[0])
            capacities_agg["LCOE_clstr"] = capacities_agg["ts_identifier"].apply(lambda x: x[1])
            if not self.sub_dataset_name is None:
                capacities_agg['dsetname'] = self.sub_dataset_name #TODO make dynamic again
            capacities_agg["ts_ID"] = capacities_agg["ts_identifier"].apply(lambda x: x[-1])
            # capacities_agg["region"] = capacities_agg["ts_identifier"].apply(lambda x: x[2].rsplit("_",2)[0])
            capacities_agg.drop(columns=["ts_identifier"], inplace=True)
            capacities_agg["locationID"] = locationID

            cluster_agg.append(capacities_agg)
        
        cluster_agg = pd.concat(cluster_agg, axis=0)
        cluster_agg.reset_index(drop=True, inplace=True)

        if len(cluster_agg) == 0:
            cluster_agg = self._make_dummy_df()
        
        for c in cluster_agg.columns:
            try:
                cluster_agg[c] = cluster_agg[c].astype(float)
            except ValueError:
                pass
        
        self.cluster_agg = cluster_agg

        # prepare for an aggregation completeness check later, therefore
        # extract an exemplary capacity and matching timeseries parameter name
        # and calculate and save total capacity before agg for this param
        check_ts_name = list(self.aggregation_dict.keys())[0]
        check_cap_name = self.aggregation_dict[check_ts_name]
        self.check_params = (check_cap_name, check_ts_name)
        self.check_capacity_beforeAgg = self.cluster_agg[check_cap_name].sum()

    def load_timeseries_from_nc4(self, norm_leap_year=True, threshold_negative_values=-0.001, verbose=False):
        '''
        Loads all time series from nc4 files on disc into self.cluster_agg.

        Parameters
        ----------
        verbose : bool, optional
            If True, additional progress statements will be printed, by default False
        norm_leap_year : bool, optional
            If True, ignores 29.02. in leap years, by default True
        threshold_negative_values : float, optional
            negative values above threshold will be corrected to 0, else error
        
        Sets
        ----
        self.cluster_agg : pd.DataFrame
            dataframe with all placements aggregated by lcoe-sg-clusters and their belonging ts file reference 
        '''

        #make sure self.cluster_agg is well indexed and has all ts columns to write in iteratively:
        self.cluster_agg.reset_index(drop=True, inplace=True)
        for ts_var_name in self.aggregation_dict.keys():
             self.cluster_agg[ts_var_name] = np.nan * np.ones_like(self.cluster_agg.locationID)

        _ts_paths = self.cluster_agg.ts_path.unique()
        # filter in case of asterisks
        ts_paths = list()
        for fp in _ts_paths:
            if '*' in fp:
                ts_paths.extend(glob.glob(fp))
            else:
                ts_paths.append(fp)
        time_index = ''
        
        tic = time.time()
        files_to_load = len(ts_paths)
        if len(self.cluster_agg)>0:
            assert files_to_load>0, f"Capacity data was loaded but no timeseries filepaths could be extracted."
        for iter_load, ts_path in enumerate(ts_paths):
            # issue progress prints if demanded
            if verbose and iter_load%(max(int(files_to_load/20),1))==0:
                print(datetime.datetime.now(), f"Loaded {str(round(iter_load/files_to_load*100))}% of time series files for {str(self.technology)}", flush=True)
            cluster_agg_per_sg = self.cluster_agg[self.cluster_agg.ts_path==ts_path]
            lcoe_clusters = list(cluster_agg_per_sg.LCOE_clstr)
            #load data 
            data_ts = xr.load_dataset(ts_path)
            
            for lcoe_cluster in lcoe_clusters:
                for ts_var_name in self.aggregation_dict.keys(): 
                    #get ts
                    ts_raw = data_ts.sel(LCOE_clstr=lcoe_cluster)[ts_var_name].to_pandas()
                    #filter leap year
                    if norm_leap_year:
                        ts_raw = ts_raw[(ts_raw.index.month != 2) | (ts_raw.index.day != 29)]
                    if isinstance(time_index, str):
                        time_index = ts_raw.index
                    # if not an explicitly negative ts, assert we do not have negative time series entries
                    if not ts_var_name in self.negative_ts:
                        #threshold=-0.001
                        assert all([not x<threshold_negative_values for x in ts_raw]), f"Time series {ts_var_name} for cluster {lcoe_cluster} has entries < {threshold_negative_values} (filepath: {ts_path})"
                        # correct minor rounding issues by setting minimal negatives to zeros
                        ts_raw = pd.Series([x if x>=0 else 0.0 for x in ts_raw], index=ts_raw.index)
                    
                    #set the value:
                    index_lcoe_sg_cluster = self.cluster_agg.loc[(self.cluster_agg.ts_path==ts_path) &(self.cluster_agg.LCOE_clstr==lcoe_cluster)].index
                    # if we have custom regions, index_lcoe_sg_cluster can be greater than one since one ts_path/ts_ID can belong to multiple regions
                    # create a list of the same values for each index, create a series with the correct index and assign it to cluster_agg
                    value_list = [ts_raw.values]*len(index_lcoe_sg_cluster)
                    self.cluster_agg.loc[index_lcoe_sg_cluster, ts_var_name] = pd.Series(value_list, index=index_lcoe_sg_cluster)


        toc = time.time()
        if verbose:
            print(f"Loading {files_to_load} time series took {toc-tic} s.")
        
        self.time_index = time_index
        
        # similar to before, we need to save the total energy generation before 
        # aggregation to later compare in an aggregation completeness check
        check_cap_name = self.check_params[0]
        check_ts_name = self.check_params[1]

        if len(self.cluster_agg) == 0:
            self.check_generation_beforeAgg = 0
        else:
            self.check_generation_beforeAgg = (self.cluster_agg[check_cap_name] * self.cluster_agg[check_ts_name]).sum().sum()
        
    def aggregate_timeseries(
        self, 
        N_clusters=None, 
        global_clusters=False
        ):
        '''
        Aggregates timeseries

        Parameters
        ----------
        N_clusters : int, optional
            The number of clusters after aggregation that shall be added to the energy system 
            model, either per region or across the whole model (see below). If None, no further
            aggregation will be done. By default None.
        global_clusters : bool, optional
            If True, the cluster definitions will be the same for all model locations, i.e. a 
            total of 'N_clusters' (see above) will be defined for the whole model. Per each
            model location, usually not all of these clusters will be available, so that only the
            available clusters will be added to each model location. If global_clusters is False,
            'N_clusters' are enforced per model location, filling missing clusters per region/
            location with zeros. Different regions will have different cost in the same cluster 
            numbers. By default False.
        '''
        
        if global_clusters:
            # extract all unqique clusters from the whole model, this is the No. of EXISTING clusters
            N_lcoe_clusters = len(np.unique(self.cluster_agg.LCOE_clstr))
            # if N_clusters is None, set to currently existing No. of clusters and nothing happens
            if N_clusters is None: N_clusters = N_lcoe_clusters
            if N_clusters > N_lcoe_clusters:
                print(f"NOTE: N_clusters ({N_clusters}) was chosen too big and is reduced to maximum number of clusters in model: {N_lcoe_clusters}", flush=True)
                N_clusters = N_lcoe_clusters
            N_lcoe_per_FINE_cluster = int(N_lcoe_clusters/N_clusters) #do floor by int(), is intended!
            #make the FINE_Cluster identifier;
            regular = np.arange(0, N_clusters).repeat(N_lcoe_per_FINE_cluster) #[0,0,..., 1,1,...,N_clusters-1,...,N_clusters-1]
            end = (N_clusters-1) * np.ones((N_lcoe_clusters-len(regular)))
            FINE_cluster = np.append(regular, end)
            map_lcoe_cluster_to_FINE_cluster = {k:v for k,v in zip(range(min_lcoe_cluster_total, max_lcoe_cluster_total+1), FINE_cluster)}
            self.cluster_agg["FINE_cluster"] = self.cluster_agg["LCOE_clstr"].apply(lambda x: map_lcoe_cluster_to_FINE_cluster[x])


        fine_clusters = []
        for locationID in self.locationIDs:
            # filter and sort by lcoe
            data_location = self.cluster_agg[self.cluster_agg.locationID == locationID].copy()
            data_location.sort_values('LCOE_clstr', ascending=True, inplace=True)

            if not global_clusters:
                # first copy and save original N_clusters since if None it would be overwritten for each locationID
                N_clusters_orig = copy(N_clusters)
                # define clustering based on lcoes for each region
                N_timeseries = len(data_location)
                if N_clusters is None:
                    N_clusters=N_timeseries
                if N_clusters == 0:
                    N_ts_per_cluster=0
                else:
                    N_ts_per_cluster = int(N_timeseries / N_clusters) #do floor by int(), is intended!
                if N_ts_per_cluster == 0:
                    # #not enough time series found, creating ones with zeroes
                    FINE_cluster = np.arange(0, N_timeseries)
                else:            
                    #make the FINE_Cluster identifier;
                    regular = np.arange(0, N_clusters).repeat(N_ts_per_cluster) #[0,0,..., 1,1,...,N_clusters-1,...,N_clusters-1]
                    end = (N_clusters-1) * np.ones((N_timeseries-len(regular)))
                    FINE_cluster = np.append(regular, end)
                data_location["FINE_cluster"] = FINE_cluster
            
            # generate absolute timeseries by multiplying capacity-factor timeseries with respective capacity values
            for ts_varname in self.aggregation_dict.keys():
                data_location[ts_varname] = data_location[ts_varname] * data_location[self.aggregation_dict[ts_varname]]
            
            # aggregate by FINE_cluster
            data_fine_agg_location = data_location.groupby("FINE_cluster").agg(self.agg_mapper_ts)
            
            # make ts relative again after having aggregated over the whole cluster
            for ts_varname in self.aggregation_dict.keys():
                data_fine_agg_location[ts_varname] = data_fine_agg_location[ts_varname] / data_fine_agg_location[self.aggregation_dict[ts_varname]]
            
            #add possibly missing empty clusters
            if len(data_fine_agg_location)==0 and N_clusters==0:
                # this would cause the component to not be added at all, enforce at least one zero-potential cluster
                N_clusters=1
            if len(data_fine_agg_location) < N_clusters:
                #make dummy ones
                missing_clusters = set(np.arange(0, N_clusters)) - set(data_fine_agg_location.index)
                for missing_cluster in missing_clusters:
                    data_fine_agg_location.loc[missing_cluster] = 0 #append zeroes
                    data_fine_agg_location.loc[missing_cluster, "locationID"] = locationID
                    for ts_name in self.aggregation_dict.keys():
                        if not data_fine_agg_location[ts_name].dtypes=='O':
                            data_fine_agg_location[ts_name] = data_fine_agg_location[ts_name].astype('O')
                        if ts_name in self.daily_timeseries:
                            data_fine_agg_location.at[missing_cluster, ts_name] = np.zeros(365)
                        else:
                            data_fine_agg_location.at[missing_cluster, ts_name] = np.zeros(8760)

            fine_clusters.append(data_fine_agg_location)
            
            # reset N_clusters to original value for next location ID iteration step
            N_clusters = copy(N_clusters_orig)
        
        fine_clusters = pd.concat(fine_clusters, axis=0)
        fine_clusters = fine_clusters.reset_index().set_index(["locationID", "FINE_cluster"])

        self.fine_clusters = fine_clusters

        assert self._check_aggregation_correctness(), f"Aggregation timeseries failed."

        
   
    def return_as_dict(
        self, 
        round_ts_digits=None, 
        ):
        '''Warp the aggregated data into a readable format for the model builder

        Parameters
        ----------
        round_ts_digits : int, optional
            rounding digits for time series, by default None.
        # factor_caps : int, optional
        #     unit scaling factor for capacity vars, by default 1

        Returns
        -------
        dict
            {Cluster: {var_name: variable}} for each cluster and variable stated within N_clusters and dict_agg
        '''
        model_builder_dict = {}
        clusters = list(self.fine_clusters.index.get_level_values(level=1).astype(int).unique())

        ts_names = list(self.aggregation_dict.keys())

        for cluster in clusters:
            cluster_values = self.fine_clusters.xs(cluster, axis=0, level=1)
            assert len(self.locationIDs) == len(cluster_values), "Some regions were lost."
            cluster_values = cluster_values.loc[natsort.natsorted(self.locationIDs)] #sort locations by natsorted           

            model_builder_dict[cluster]= {}
            for variable in cluster_values.columns:
                #if time series, warp to proper format
                if variable in ts_names:
                    df = cluster_values[[variable]].T.explode(column=self.locationIDs)
                    df.reset_index(drop=True, inplace=True)
                    if round_ts_digits is not None:
                        df = df.round(round_ts_digits)
                    model_builder_dict[cluster][variable] = df.astype(float)
                #if capacity, apply factor
                elif variable in self.capacity_vars:
                    model_builder_dict[cluster][variable] = cluster_values[variable] * self.capacity_conversion_factor
                else:
                #else: just pass
                    model_builder_dict[cluster][variable] = cluster_values[variable]

        return model_builder_dict  

    
    def _check_aggregation_correctness(self):
        '''
        Check if the data was correctly aggregated, determined by the total capacity
        and energy in the whole model must be similar to the loaded input values.

        Returns
        -------
        bool
            True if aggregation was sucesfully else False.
        '''
        # extract the capacity and timeseries variables to check exemplarily
        check_cap_name = self.check_params[0]
        check_ts_name = self.check_params[1]

        # calculate total capacity and energy generation AFTER aggregation
        if self.check_generation_beforeAgg == 0:
            check_generation_final = 0
        else:
            check_generation_final = (self.fine_clusters[check_cap_name] * self.fine_clusters[check_ts_name]).sum().sum()
        check_capacity_final = self.fine_clusters[check_cap_name].sum()
        
        # check if the values remained the same, compared to the saved values before aggregation
        capacity_stays_same = np.isclose(check_capacity_final, self.check_capacity_beforeAgg)
        energy_stays_same = np.isclose(check_generation_final, self.check_generation_beforeAgg)

        # return boolean indicating if check was passed
        return capacity_stays_same and energy_stays_same

    
    def get_potential_dict(
        self, 
        location_shape, 
        defaultregions_per_location_dict, 
        use_partial_polygon_capacities=True, 
        region_col_name ="region",
        N_clusters=None, 
        global_clusters=False,
        verbose=True,
        ):
        """
        Method to extract timeseries and capacity values from nc4 and shp/pickle 
        files, returning the loaded data in a dictionary format with clusters as 
        keys and value pairs ready to load into a FINE esM object.

        location_shape : pd.DataFrame
            A dataframe with all model locations and capacities of all placements to be 
            loaded into the model.
        defaultregions_per_location_dict : dict
            A dictionary with all model locations as key and lists of all overlapping 
            default regions (GID_1split) as values.
        use_partial_polygon_capacities : bool, optional
            If True, the capacity attributes of each polygon-shaped plant will be reduced linearly with the 
            plant area percentage that actually overlaps with the respective model location. Set to False to 
            save time at the cost of possibly (partially) counting plant capacities twice that overlap the 
            location shape borders. Will only be applied to custom-shaped model locations and if possible, i.e. 
            when the geom column contains (multi)polygon geometries. By default True.
        region_col_name : str, optional
            Indicates the name of the (default) region column attribute in the placement shapefile/pickle to
            be loaded, by default 'region'.
        N_clusters : int, optional
            The number of clusters after aggregation that shall be added to the energy system 
            model, either per region or across the whole model (see below). If None, no further
            aggregation will be done. By default None.
        global_clusters : bool, optional
            If True, the cluster definitions will be the same for all model locations, i.e. a 
            total of 'N_clusters' (see above) will be defined for the whole model. Per each
            model location, usually not all of these clusters will be available, so that only the
            available clusters will be added to each model location. If global_clusters is False,
            'N_clusters' are enforced per model location, filling missing clusters per region/
            location with zeros. Different regions will have different cost in the same cluster 
            numbers. By default False.
        verbose : bool, optional
            If True, additional progress statements will be printed, by default True.

        Returns: 
        -------
        dict
            {Cluster: {var_name: variable}} for each cluster and variable stated within N_clusters 
            and aggregation_dict.
        """
        # first extract placement dataframes for given model locations
        self.load_placements(
            location_shape = location_shape, 
            defaultregions_per_location_dict = defaultregions_per_location_dict, 
            use_partial_polygon_capacities=use_partial_polygon_capacities, 
            region_col_name =region_col_name,
            verbose=verbose,
        )

        # load the timeseries per cluster and spatial group from nc4 files
        if self.technology in ['wind_onshore', 'wind_offshore']:
            threshold_negative_values = -0.05 # we have minor negative cfs in the series due to an bug in reskit.wind 03.03.2023: d.franzmann
        else:
            threshold_negative_values = -0.001 #default
        self.load_timeseries_from_nc4(
            verbose=verbose,
            norm_leap_year=True,
            threshold_negative_values=threshold_negative_values
            )

        # now aggregate the timeseries based on capacity weighting
        self.aggregate_timeseries(
            N_clusters=N_clusters, 
            global_clusters=global_clusters,
        )

        return self.return_as_dict(
            round_ts_digits=4,
        )