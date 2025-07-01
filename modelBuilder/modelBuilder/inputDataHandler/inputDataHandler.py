# import standard packages
import copy
import glob
import numpy as np
import os
import pandas as pd
import re
import warnings
import yaml
import sys


# import third party packages
import geokit as gk
import natsort
from geopandas import GeoDataFrame

# import other modules
from ..data import data_folder
from .potentials.extract_variable_potentials import *
from .potentials import extract_constant_potentials
from .grid import spatialDefinition
from .grid import processExistingElectricityGrid
from .demands.extract_default_demands import get_electricity_data, get_hydrogen_demand_data, combine_to_abs_timeseries, scale_timeseries_to_locationsIDs
from modelBuilder.singletons import ModelPaths, ModelLocations, InputDataInfo

from os.path import dirname
test_data_folder = os.path.join(os.path.abspath(dirname(dirname(dirname(__file__)))), "tests", "test_data")
#puh this is ugly, but stack overflow mentioned: "Why doesn't python consider the current working directory to be a package? NO CLUE, but gosh it would be useful."

class inputDataHandler(object):  # (esMWorkflowManager):
    """
    #TODO
    """
    def __init__(
        self,
    ): 
        """
        #TODO
        """
        self.data_folder = data_folder

    ###################################################################################################################
    # Potentials loader methods
    ###################################################################################################################

    def get_capacities_and_timeseries_from_nc4(
        self,
        technology, 
        model_unit,
        use_partial_polygon_capacities=True, 
        N_clusters=3, 
        global_clusters=True,
        verbose=True,
        ):
        """
        Data loader method loading variable timeseries and regional potential values 
        from disk storage in GlobEP step 2 nc4 output data format, returning as dict ready
        to add to esM via modelManager potentials loader functions.

        Parameters
        ----------
        technology : str
            Technology for which potentials shall be loaded. Must be a 
            key in InputDataInfo to allow loading of default potentials, 
            else all variables below must be specified explicitly. 
        weather_year : (int, str)
            The weather year for which potentials shall be loaded. Use e.G.: '2018', 2018, ...
        model_unit : str
            The unit of the model for the commodity affected by this source, e.g. 'GW'. If the model unit is s.th.
            like 'GW_H2_LHV' please only use the physical unit description, in this case 'GW'.
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
            {cluster: {var_name: variable}} for each cluster and variable stated within N_clusters 
            and aggregation_dict, timeseries converted to hourly where required.
        """        
        def load_potentials_dict_from_nc4():    
            # first create NC4Source instance with basic information
            source = NC4Source(
                technology=technology, 
                model_unit=model_unit,
                verbose=verbose,
            )

            # then extract potentials dictionary
            potentials_dict = source.get_potential_dict(
                location_shape=ModelLocations().location_df,                # change issue 112
                defaultregions_per_location_dict=ModelLocations().get_default_regions_mapper(),  # change issue 112
                use_partial_polygon_capacities=use_partial_polygon_capacities, 
                region_col_name =InputDataInfo().get_info(tech=technology, attr="region_col_name"),
                N_clusters=N_clusters, 
                global_clusters=global_clusters,
                verbose=verbose,
            )

            # convert the daily timeseries to hourly timesteps if needed
            if not source.daily_timeseries == [None]: 
                for ts_daily, ts_ref in zip(source.daily_timeseries, source.hourly_reference_timeseries):
                    potentials_dict = self.convert_timesteps_from_daily_to_hourly(
                        potentials_dict=potentials_dict,
                        daily_to_hourly_timeseries= ts_daily,
                        hourly_reference_timeseries= ts_ref,
                        )
            
            # convert timeseries to be either negative or positive:
            for ts_name in source.aggregation_dict:
                for cluster in potentials_dict.keys():
                    ts = potentials_dict[cluster][ts_name]
                    ts_potitive = ts.values.sum() > 0
                    if ts_potitive:
                        potentials_dict[cluster][ts_name][ts<0] = 0
                    else:
                        potentials_dict[cluster][ts_name][ts>0] = 0

            return potentials_dict
    

        if not ModelPaths().intermediates_folder is None:
            # check if itermediates exist
            # load dictionary with default potential information from data folder
            #TODO: this is taken from NC4Source class. scenario, iteration_name should be defined way earlier, so this hack is not needed
            import copy # has to be done because someone did this somewhere in the code "from copy import copy"
            with open(InputDataInfo().path_to_input_data, 'r') as file:
                potentials_information = yaml.load(file, Loader=yaml.FullLoader)  
            scenario = potentials_information[technology.lower()]["scenario"]
            iteration_name = potentials_information[technology.lower()]["iteration_name"]
        
            def _get_intermediate_xarray():
                # convert to xarray
                potentials_dict = load_potentials_dict_from_nc4()
                _xrds = self.create_xarray_from_potential_dict(
                    potential_dict=copy.deepcopy(potentials_dict), 
                )
                return potentials_dict, _xrds

            def _save_intermediate_xarray(xrds, file_path):
                print(f"Saving intermediates to {file_path}")
                counter = 0
                while counter < 40:
                    try:
                        xrds.to_netcdf(file_path)
                        break
                    except:
                        counter += 1
                        print(f"Failed to save intermediates. Retrying {counter} of 40 times.")
                return xrds
                
            file_name = f"{technology}_{scenario}_{iteration_name}_clusters_{N_clusters}.nc4"
            full_file_path = os.path.join(ModelPaths().intermediates_folder, file_name)
            if os.path.isfile(full_file_path):  # checks via os if .nc4 file exist
                print(f"Loading intermediates from {full_file_path}",flush=True)
                xrds = xr.load_dataset(full_file_path)
                
                # check if all regions are present
                weather_year_exists = InputDataInfo().weather_year in xrds.weather_year
                all_locations_present = all([loc in xrds.locationID for loc in ModelLocations().locationIDs])
                if all_locations_present and weather_year_exists:
                    # check if there is data for all locations and the weather year
                    xrds_target = xrds.sel(locationID=list(ModelLocations().locationIDs) , weather_year=InputDataInfo().weather_year)
                    # check if "capacity" values are all nan
                    missing_data = bool(xrds_target.capacity.isnull().all())
                    
                    if not missing_data:            
                        print("OK: All data (regions and weather year) present in intermediates",flush=True)
                        potentials_dict = self.create_potential_dict_from_xarray(
                            xrds_w_cluster_w_year=xrds_target, 
                            )
                        return potentials_dict
                    else:
                            print("Not all data (regions and weather year) present in intermediates. Loading via NC4Source",flush=True)
                            potentials_dict, _xrds = _get_intermediate_xarray()
                            print(f"Saving intermediates to {full_file_path}",flush=True)
                            xrds = xr.merge([xrds, _xrds])
                            _save_intermediate_xarray(xrds, full_file_path)
                            return potentials_dict
                else:
                    print("Not all regions or weather year present in intermediates. Loading via NC4Source",flush=True)
                    potentials_dict, _xrds = _get_intermediate_xarray()
                    print(f"Saving intermediates to {full_file_path}",flush=True)
                    xrds = xr.merge([xrds, _xrds])
                    _save_intermediate_xarray(xrds, full_file_path)
                    return potentials_dict
            else: # load from NC4 source
                print(f"No intermediates found at {full_file_path}. Loading via NC4Source",flush=True)
                potentials_dict, _xrds  = _get_intermediate_xarray()
                # save to file
                print(f"Saving intermediates to {full_file_path}")
                _save_intermediate_xarray(_xrds, full_file_path)
                return potentials_dict
        else:
            return load_potentials_dict_from_nc4()        


    def create_xarray_from_potential_dict(self, potential_dict):
        """creates an xarray dataset from dictionary with capacities and timeseries 

        :param potential_dict: dictionary containing capacities and timeseries for all regions, weather years
        :type potential_dict: dict
        :param regions: list of model locations
        :type regions: list
        :param weather_year: weather year of the used time series
        :type weather_year: int
        :return: xarray in form of potential dict containing capacities and timeseries
        :rtype: xarray.core.dataset.Dataset
        """

        N_clusters_list = list(potential_dict.keys())
        time = np.arange(0, 8760)
        xrds_list_cluster = []
        for cluster in potential_dict.keys():
            _xrds = xr.Dataset(coords={"clusters": N_clusters_list, "locationID": list(ModelLocations().locationIDs), "time": time, "weather_year": InputDataInfo().weather_year})

            _cluster_data = potential_dict[cluster].copy()

            # add capacity
            if "capacity" in _cluster_data.keys():
                _cap = _cluster_data["capacity"].to_frame()
                _cap["clusters"] = cluster
                _cap["weather_year"] = InputDataInfo().weather_year
                _cap = _cap.set_index(["clusters", "weather_year"], append=True)
                _xrds_cap = _cap.to_xarray()["capacity"]
                _xrds["capacity"] = _xrds_cap
            
            # add ts_capacity_factor
            if "ts_capacity_factor" in _cluster_data.keys():
                _ts_cf = _cluster_data["ts_capacity_factor"]
                _ts_cf.index.name = "time"
                _ts_cf["clusters"] = cluster
                _ts_cf["weather_year"] = InputDataInfo().weather_year
                _ts_cf = _ts_cf.set_index(["clusters", "weather_year"], append=True)
                _ts_cf = _ts_cf.stack().to_xarray()
                _xrds["ts_capacity_factor"] = _ts_cf
            
            # add LCOE_clstr
            if "LCOE_clstr" in _cluster_data.keys():
                _lcoe_clstr = _cluster_data["LCOE_clstr"].to_frame()
                _lcoe_clstr["clusters"] = cluster
                _lcoe_clstr["weather_year"] = InputDataInfo().weather_year
                _lcoe_clstr = _lcoe_clstr.set_index(["clusters", "weather_year"], append=True)
                _xrds_lcoe_clstr = _lcoe_clstr.to_xarray()["LCOE_clstr"]
                _xrds["LCOE_clstr"] = _xrds_lcoe_clstr
            
            xrds_list_cluster.append(_xrds)
        # merge
        xrds_cluster = xr.merge(xrds_list_cluster)
        return xrds_cluster
    
    def create_potential_dict_from_xarray(self, xrds_w_cluster_w_year):
        """Gets capacities and time series from xarray in the formate of a potential_dict 
        similar to output of self.get_capacities_and_timeseries_from_nc4.

        Args:
            xrds_w_cluster_w_year (xarray.core.dataset.Dataset): xarray instance containing clustered time series and capacity data
            weather_year (int): Year of weather data

        Returns:
            dict: diconary with time series and capacity data 
        """
        # xrds_w_cluster = xrds_w_cluster_w_year.sel(weather_year=weather_year)
        if "weather_year" in xrds_w_cluster_w_year.keys():
            xrds_w_cluster_w_year = xrds_w_cluster_w_year.drop("weather_year")
        xrds_w_cluster = xrds_w_cluster_w_year
        potential_dict = {}
        for cluster in xrds_w_cluster.clusters.to_numpy():
            potential_dict[cluster] = {}
            if "capacity" in xrds_w_cluster.keys():
                cap = xrds_w_cluster.sel(clusters=cluster).drop("clusters")["capacity"].to_dataframe().squeeze()
                potential_dict[cluster]["capacity"] = cap
            if "ts_capacity_factor" in xrds_w_cluster.keys():
                ts_cf = xrds_w_cluster.sel(clusters=cluster).drop("clusters")["ts_capacity_factor"].to_dataframe().unstack().droplevel(0,axis=1)
                potential_dict[cluster]["ts_capacity_factor"] = ts_cf.fillna(0)

            if "LCOE_clstr" in xrds_w_cluster.keys():
                lcoe_clstr = xrds_w_cluster.sel(clusters=cluster).drop("clusters")["LCOE_clstr"].to_dataframe().squeeze()
                potential_dict[cluster]["LCOE_clstr"] = lcoe_clstr        

        return potential_dict

    def convert_timesteps_from_daily_to_hourly(self, potentials_dict, daily_to_hourly_timeseries, hourly_reference_timeseries): #OK
        """
        A method to convert daily timesteps in timeseries stored in a data dictionary
        to hourly output values.

        potentials_dict : dictionary
            A dictionary with clusters as keys, and a subdict of parameter name keys and 
            variable values. Must contain daily_to_hourly_timeseries and 
            hourly_reference_timeseries names as sub keys.
        daily_to_hourly_timeseries : str
            The timeseries name or list of timeseries names that need to be converted to 
            hourly timesteps
        hourly_reference_timeseries : str
            The reference time series that indicates the sub-daily distribution of the
            values to be generated.
        """
        #make the ts_daily from daily to hourly
        for cluster in potentials_dict.keys():
            #get ts
            ts_ref = potentials_dict[cluster][hourly_reference_timeseries]
            ts_daily = potentials_dict[cluster][daily_to_hourly_timeseries]

            # define default time series and lists of days for hourly and daily
            time_index_hours = pd.date_range(start='2001-01-01 00:00', end='2001-12-31 23:00', freq="60min")
            time_index_days = pd.date_range(start='2001-01-01 00:00', end='2001-12-31 23:00', freq="1d")
            days = np.unique(time_index_hours.dayofyear)

            # convert the time series
            ts_final = []
            for day in days:
                ts_day_ref = ts_ref[time_index_hours.dayofyear == day]
                new_mean = ts_daily[time_index_days.dayofyear==day].iloc[0]
                old_mean = ts_day_ref.mean(axis=0)
                ts_day_new = ts_day_ref * new_mean / old_mean
                ts_day_new = ts_day_new.fillna(0)

                assert np.allclose(ts_day_new.mean(axis=0), new_mean)

                ts_final.append(ts_day_new)
            ts_final = pd.concat(ts_final, axis=0)
            ts_final = ts_final.sort_index()

            potentials_dict[cluster][daily_to_hourly_timeseries] = ts_final

        return potentials_dict


    def load_constant_potentials(
        self,
        technology,
        N_cluster,
        model_unit,
        path=None,   
        LCOE_name=None,
        capacity_name=None,
        region_name_col=None,
        # capacity_to_GW_factor=None,
        LCOE_to_EUR_per_kWh_factor=None, 
        capacity_conversion_factor=None,
        rounding=4,
        _timeout=60,
        verbose=False,
        ):
        """wrapper for loading the EGS input data
        """
        if not path: path = InputDataInfo().get_info(tech=technology, attr="cap_base_path")
        if not LCOE_name: LCOE_name=InputDataInfo().get_info(tech=technology, attr="LCOE_name")
        if not capacity_name: capacity_name=InputDataInfo().get_info(tech=technology, attr="capacity_name")
        if not region_name_col: region_name_col=InputDataInfo().get_info(tech=technology, attr="region_name_col")
        if not capacity_conversion_factor : capacity_conversion_factor = UnitHandling().get_unit_conversion_factor(
            input_unit=InputDataInfo().get_info(tech=technology, attr="data_unit"), 
            target_unit=model_unit
        )
        if not LCOE_to_EUR_per_kWh_factor: LCOE_to_EUR_per_kWh_factor=InputDataInfo().get_info(tech=technology, attr="LCOE_to_EUR_per_kWh_factor") #default_potentials_information[technology.lower()]["LCOE_to_EUR_per_kWh_factor"]
        

        extension = os.path.splitext(os.path.basename(path))[1]

        if extension in [".sql", ".sqlite"]:

            potential_dict = extract_constant_potentials.extract_potentials_sql( #TODO David as discussed this should be generalized like get_potential_from_sql() or so to allow loading of power plant database etc.
                path=path,   
                LCOE_name=LCOE_name,
                capacity_name=capacity_name,
                region_name_col=region_name_col,
                capacity_conversion_factor=capacity_conversion_factor,
                LCOE_to_EUR_per_kWh_factor=LCOE_to_EUR_per_kWh_factor,
                N_cluster=N_cluster,
                location_shape=ModelLocations().location_df,  # change issue 112
                defaultregions_per_location_dict=ModelLocations().get_default_regions_mapper(), # change issue 112
                rounding=rounding,
                _timeout=_timeout,
                verbose=verbose,
            )
            return potential_dict
        
        elif extension in [".shp"]:
            
            potential_dict = extract_constant_potentials.extract_potentials_shp(
                path=path,
                LCOE_name=LCOE_name,
                capacity_name=capacity_name,
                region_name_col=region_name_col,
                capacity_conversion_factor=capacity_conversion_factor,
                LCOE_to_EUR_per_kWh_factor=LCOE_to_EUR_per_kWh_factor,
                N_cluster=N_cluster,
                location_shape=ModelLocations().location_df,  # change issue 112
                defaultregions_per_location_dict=ModelLocations().get_default_regions_mapper(), #chanfe issue 112
                rounding=rounding,
                verbose=verbose
                )
            return potential_dict

        elif extension in [".csv", ".xlsx"]:

            potential_dict = extract_constant_potentials.extract_potentials_csv(
                path=path,
                LCOE_name=LCOE_name,
                capacity_name=capacity_name,
                region_name_col=region_name_col,
                capacity_conversion_factor=capacity_conversion_factor,
                LCOE_to_EUR_per_kWh_factor=LCOE_to_EUR_per_kWh_factor,
                N_cluster=N_cluster,
                location_shape=ModelLocations().location_df,  # change issue 112
                defaultregions_per_location_dict=ModelLocations().get_default_regions_mapper(),
                rounding=rounding,
                verbose=verbose,
            )
            return potential_dict
        else:
            raise NotImplementedError(f"Cannot load filetype {extension}. Path was: {path}")
        

    ###################################################################################################################
    # Demands loader methods
    ###################################################################################################################


    def load_demand(self, technology, year_demand, path_abs_demands=None, path_ts=None):
        '''loads timeseries from given paths to FINE format

        Parameters
        ----------
        technology : str
            technology, must match the technologies in the technoeconomic data
            E.g.: "electricity", ""
        year_demand : int
            year
        path_abs_demands : str
            path to folder with demands:
        path_ts : str
            path to specific ts file ".csv". Can be None sometimes (e.g. hydrogen_gas, no ts for that)

        Returns
        -------
        final : pd.DataFrame
            abs timeseries per locationID. index: range(0, 8760), columns: locationIDs

        Raises
        ------
        OSError
        '''

        with open(os.path.join(self.data_folder, "default_demands.yaml")) as fp:
            default_demand_information = yaml.load(fp, Loader=yaml.FullLoader)
        
        if not path_abs_demands:
            assert technology in default_demand_information
            path_abs_demands = default_demand_information[technology]["path_abs_demands"]
            path_ts = default_demand_information[technology]["path_timeseries"]

            #if not os.path.isdir(path_abs_demands): raise OSError(f"No valid folder found for path_abs_demands: {path_abs_demands}")
            
        if (ModelLocations().location_df.dflt_type=="default").all():     # issue 112        #TODO: adapt tp agg regions, but not that neccessary for speed
            gid0s = list(ModelLocations().location_df["GID_0"].unique())  # issue 112
        else:               
            #find all occuring gid1s in defaultregions_per_location_dict
            gid0s = []
            for lists in ModelLocations().get_default_regions_mapper().values():  # issue 112
                gid0s.extend(lists)
            gid0s = list(set([v[0:3] for v in gid0s]))


        if "electricity" in technology:
            abs_demands, rel_demands = get_electricity_data(
                path_abs_demands=path_abs_demands, 
                path_ts=path_ts, 
                gid0s=gid0s,
            )
            abs_column_name = "total_el_demand"
            
        elif "hydrogen_gas" in technology:
            abs_demands, rel_demands = get_hydrogen_demand_data(
                path_abs_demands=path_abs_demands,
                year_demand=year_demand,
                gid0s=gid0s,
            )
            abs_column_name = "hydrogen_demand_gid1_GWh"
        else:
            raise OSError(f"technology {technology} unknown to demand loader.")

        
        abs_timeseries_per_gid1 = {
            ip: combine_to_abs_timeseries(abs_demands[ip], rel_demands, abs_column_name)
            for ip in InputDataInfo().investment_period_names
        }

        final = {
            ip: scale_timeseries_to_locationsIDs(abs_timeseries_per_gid1[ip])
            for ip in InputDataInfo().investment_period_names
        }
        return final
    

    ###################################################################################################################
    # Transmission loader methods
    ###################################################################################################################



    def load_transmission_vars(self, detour_factor):
        """Calculates the transmissional eligibility and distances for a given shape file."""
        # df from shapefile and drops geometry column
        shapeGEokit = ModelLocations().location_df.drop(columns=["geometry"])  # issue 112

        # load the class for processing
        spatialDef = spatialDefinition(
            shape=shapeGEokit,
            region_name_col=ModelLocations().locationID_attr,  # issue 112
            path_datafolder=ModelPaths().base_folder,
        )
        # return vars in dict style
        transmission_vars = spatialDef.return_dict(detour_factor)
        return transmission_vars

    def load_existing_electricity_grid(self, technology_name, model_unit, data_unit, path_grids): # issue 112
        """Loads and clusters the electricity grid data for a given shape file."""

        transmission_vars = processExistingElectricityGrid(
            self, technology_name, model_unit, data_unit, path_grids
        )
        return transmission_vars

    ###################################################################################################################
    # Lull loader methods
    ###################################################################################################################
    # TODO: @David, please check whether this code style is sufficient for general mb use 

    def load_VOLL(self, voll_to_BEUR_per_GWh_factor, voll_key, path_VOLL=None, time_steps=8760, sectoral_disaggregation=True):      
            '''load VoLL for each region

            Parameters
            ----------
            voll_to_BEUR_per_GWh_factor : float
                factor, to obtain data in the right format

            voll_key : str
                Name of the to use VoLL column in the data

            path_VOLL : str
                Path to gid0 level data for VoLL

            time_steps : int
                Model nr of time steps

            sectoral_disaggregation : boolean
                True: add several VoLL for different sectors
                False: add one average VoLL

            Returns
            -------
            pd.DataFrame
                VoLL in BEUR/GWh
                for each region and timestep
            '''

            #set defaults

            with open(os.path.join(self.data_folder, "default_demands.yaml")) as fp:
                default_voll = yaml.load(fp, Loader=yaml.FullLoader)["VOLL"]

            if not path_VOLL: path_VOLL = default_voll["path_VOLL"]
            if not voll_to_BEUR_per_GWh_factor: voll_to_BEUR_per_GWh_factor = float(default_voll["voll_to_BEUR_per_GWh_factor"])
            if not path_VOLL: path_VOLL = default_voll["path_VOLL"]
            if not voll_key: voll_key = default_voll["voll_key"]

            if not os.path.isfile(path_VOLL):
                raise OSError(f"File not found: {path_VOLL}")
            
            #load all data
            voll_data = pd.read_csv(path_VOLL, index_col=[0])
            if not voll_key in voll_data.columns:
                    raise KeyError(f"{voll_key} not in data at: {path_VOLL}")

            if sectoral_disaggregation:
                sectors = [c.split("_")[1] for c in voll_data.columns if ("phi" in c)] #select all colmns which are not chi or voll
                phi = voll_data[[f"phi_{s}" for s in sectors]]
                chi = voll_data["chi_i"]
                #share per gid1
                shares_gid0 = voll_data[[f"share_{sec}" for sec in sectors]]
                locations= ModelLocations().location_df[[ModelLocations().locationID_attr, "GID_0"]].copy().set_index(ModelLocations().locationID_attr) # issue 112
                shares = locations.GID_0.apply(lambda x: shares_gid0.loc[x])
                del shares_gid0, locations
            else:
                sectors = [0]
                phi = 1
                chi = 1
                shares = None
            
            voll_dict = {}

            for sector in sectors:

                #calculate the VoLL:
                if sectoral_disaggregation:
                    country_voll_gid0_EUR_per_Wh = voll_data[voll_key] * voll_to_BEUR_per_GWh_factor * phi[f"phi_{sector}"] * chi
                else: 
                    country_voll_gid0_EUR_per_Wh = voll_data[voll_key] * voll_to_BEUR_per_GWh_factor
                #load the 

                #matching data to FINE:
                # 1) per gid1:
                VOLL_EUR_per_Wh = ModelLocations().location_df[[ModelLocations().locationID_attr, "GID_0"]].copy().set_index(ModelLocations().locationID_attr) # issue 112
                VOLL_EUR_per_Wh["VOLL_EUR_per_Wh"] = VOLL_EUR_per_Wh.GID_0.apply(lambda x: country_voll_gid0_EUR_per_Wh[x])
                VOLL_EUR_per_Wh = VOLL_EUR_per_Wh["VOLL_EUR_per_Wh"]
                #2) per time step
                VOLL_EUR_per_Wh = pd.DataFrame(
                    data=np.tile(VOLL_EUR_per_Wh.values, (time_steps ,1)).T,
                    index=VOLL_EUR_per_Wh.index,
                    columns=range(time_steps ),
                )
                VOLL_EUR_per_Wh = VOLL_EUR_per_Wh.loc[ModelLocations().locationIDs] # issue 112

                voll_dict[sector] = VOLL_EUR_per_Wh.T

            return voll_dict, shares
            