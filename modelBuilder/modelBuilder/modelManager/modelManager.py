from distutils.log import warn
import numbers
from tokenize import Number
import fine as fn
import yaml
import sys
import os
import fine.IOManagement.xarrayIO as xrIO
import natsort
import osgeo
import geokit as gk
import geopandas as gpd
import shapely
import pandas as pd
import numpy as np

# import other modules
from modelBuilder.inputDataHandler.inputDataHandler import inputDataHandler
from modelBuilder.inputDataHandler.potentials.extract_constant_potentials import get_stockCommissioning_dict
from ..data import data_folder
from .. import utils
from modelBuilder.singletons import ModelTechnoEconomicData,InputDataInfo,ModelLocations,ModelPaths,UnitHandling
from modelBuilder.inputDataHandler.potentials.extract_variable_potentials import NC4Source
from modelBuilder.utils import get_raw_regions

class modelManager(object):

    def __init__( #TODO possibly rename input args as they are named in singletons? #ISSUE112
        self,
        location_shape,
        locationID_column,
        commodityUnitsDict,
        cost_year,
        number_of_investment_periods=1,
        investment_period_interval=1,
        model_base_folder='default',
        srs=4326,
        path_to_techno_economic_data=None,
        path_to_custom_input_data=None,
        path_to_unit_conversions=None,
        weather_year=2018,
        zero_threshold=0.01,
        default_paths_fp=None,
        intermediates_folder=None,
    ):
        self.zero_threshold = zero_threshold
        self.number_of_investment_periods = number_of_investment_periods
        self.investment_period_interval = investment_period_interval

        # INITIALIZE SINGLETONS
        
        # initialize the ModelPaths singleton
        ModelPaths.reset()
        ModelPaths(
            base_folder=model_base_folder, 
            techno_economic_data_fp = path_to_techno_economic_data,
            default_paths_fp = default_paths_fp,
            intermediates_folder=intermediates_folder,
        )

        # initialize the ModelLocations singleton
        ModelLocations.reset()              
        ModelLocations( 
            location_df = location_shape,
            locationID_attr = locationID_column,
            srs = srs
        )
      
        # get default region info dict and dflt type needs to be initialized    
        ModelLocations().get_default_regions_info()

        # initialize the InputDataInfo singleton
        InputDataInfo.reset()
        InputDataInfo(
            weather_year = weather_year,
            base_year=cost_year,
            number_of_investment_periods=number_of_investment_periods,
            investment_period_interval=investment_period_interval,
            path_to_custom_input_data = path_to_custom_input_data,
        )

        # initialize UnitHandling singleton
        UnitHandling.reset()
        UnitHandling(
            commodity_units_dict=commodityUnitsDict, 
            unit_conversions_yaml=path_to_unit_conversions,
        )
        ModelTechnoEconomicData.reset()
        ModelTechnoEconomicData()
        
        self.modelSetup()
        self.ih = inputDataHandler()

    #################
    ##  INIT MODEL ##
    #################

    def completeSetup(self):
        """Convenience wrapper to complete the modelManager setup via initialization of model and necessary attributes."""
        # load techno-economic data, for completeness only, would be loaded in modelSetup and iH setup otherwise
        self.technoEconomicData_setup()
        # initialize esM
        self.modelSetup()
        # generate inputHandler instance with ted
        self.inputHandlerSetup()

    def modelSetup(self, verbose_log_level=0):
        """Initializes the FINE esM model as self.esM

        Parameters:
        ----------
        verboseLogLevel: int
            number_of_investment_periods
            investment_period_interval
            Verbose log level for fine model

        """

        #if not hasattr(self, "ted"):
        #    self.technoEconomicData_setup()

        self.esM = fn.EnergySystemModel(
            locations=ModelLocations().locationIDs,                                                 
            commodities=UnitHandling().get_commodities(as_set=True),
            numberOfTimeSteps= ModelTechnoEconomicData().esm_params["esM"]["numberOfTimeSteps"],
            commodityUnitsDict=UnitHandling().get_esM_commodityUnitsDict(),
            hoursPerTimeStep=ModelTechnoEconomicData().esm_params["esM"]["hoursPerTimeStep"],
            costUnit=ModelTechnoEconomicData().esm_params["esM"]["costUnit"],
            startYear=InputDataInfo().base_year,
            numberOfInvestmentPeriods=self.number_of_investment_periods,
            investmentPeriodInterval=self.investment_period_interval,
            lengthUnit= ModelTechnoEconomicData().esm_params["esM"]["lengthUnit"],
            verboseLogLevel=verbose_log_level,
        )
    
    #################
    ## AUXILIARIES ##
    #################

    def _add_commodities_and_units(
        self,
        new_commodity_unit_dict,
        ):
        """
        Adds one or several new commodities in case that it is needed after initialization of the 
        modelManager instance.
        
        new_commodity_unit_dict : dict
            Dictionary with one or more additional commodity names and unit pairs (each string-formatted) 
            in the following format: {commodity_name : (commodity_unit_esM, commodity_unit_multiple_of_SI)}
        """
        # first make sure commodity unit dicts are the same
        UnitHandling().compare_esM_commodityUnitsDict(esM=self.esM)
        # now check format and add new_commodity_unit_dict to UnitHandling singleton commodity units dict
        UnitHandling().add_commodity_units_dict(new_commodity_units_dict=new_commodity_unit_dict)
        # last add to esM.commodityUnitsDict
        for k,v in new_commodity_unit_dict.items():
            if k in list(self.esM.commodities):
                assert self.esM.commodityUnitsDict[k]==v[0], f"{k} is already in model commodities but unit is {self.esM.commodityUnitsDict[k]} (instead of {v[0]})!"
            else:
                commodities_in_model = self.esM.commodities
                QTY_commodities_in_model = len(commodities_in_model)
                # add the commodity by updating the set
                commodities_in_model.update({k})
                # ensure that this worked
                assert len(commodities_in_model) == QTY_commodities_in_model+1
                # write new commodity set back into esM
                self.esM.commodities = commodities_in_model
                # lastly add the corresponding unit to units dict
                self.esM.commodityUnitsDict.update({k: v[0]})

    def _process_add_function_params(self, technology, args, ignore_args=None):
        """
        This auxiliary function generates or resets the custom technology
        parameter layer of the InputDataInfo singleton based on the input
        arguments of a function.

        args : dict
            Input arguments (parameter and values) of the function.
        ignore_args : str, list, optional
            If given, must be a str formatted argument name that shall 
            not be added/updated in custom InputDataInfo layer for this 
            technology. Can also be a list of multiple args. By default 
            None.
        """
        ignore_always = ["technology", "self"]
        if not isinstance(args, dict):
            raise TypeError(f"args must be a dict type.")
        if ignore_args is not None:
            if isinstance(ignore_args, str):
                ignore_args = [ignore_args]
            if not isinstance(ignore_args, list):
                raise TypeError(f"If not None, ignore_args must be a str formatted argument to ignore or a list thereof.")
            assert all([isinstance(_arg, str) for _arg in ignore_args]),\
                f"All args in ignore_args list must be str formatted function args to be ignored."
            # add arguments that must be ignored by definition
            ignore_args.extend(ignore_always)
        else:
            # set to base ignore args
            ignore_args = ignore_always

        # check if we already have data to base upon
        if InputDataInfo().has_tech(tech=technology):
            # we have data that we can copy as a base input
            InputDataInfo().reset_custom_layer(tech=technology)
        else:
            # we must generate a custom layer
            InputDataInfo().define_custom_layer(tech=technology)

        # update custom layer with all given information
        for param, value in args.items():
            if param in ignore_args or param[:2]=='__' or callable(getattr(self, param)):
                # skip arguments to be ignored, and also class methods and all system args '__xyz__' style
                continue
            else:
                # update/add the param in custom input data layer
                InputDataInfo().set_info(tech=technology, attrs=param, vals=value)

    #################
    ## BUILD MODEL ##
    #################

    # Sources:
    def addPotentialWithTSGreenfield(
        self, 
        technology, 
        model_unit,
        data_unit=None,
        cap_base_path=None,
        ts_base_path=None,
        sub_dataset_name=None,
        aggregation_dict=None,
        additional_aggregation_vars=None,
        cluster_params=None,
        daily_timeseries=None,
        hourly_reference_timeseries=None,
        use_partial_polygon_capacities=True,
        region_col_name=None,
        N_clusters=3,
        global_clusters=False,
        verbose=True,
        # expected_FINE_args = [] # minimum list with expected fine args for source  TODO
        **FINE_kwargs,
        ):
        """
        A loader method for variable potentials (with maximum capacity factor timeseries)
        and greenfield approach, i.e. zero installation is enforced but maximum installation
        allowed up to a certain capacity limit per model location. Potential data to be 
        loaded must follow the GlobEP step 2 output formatting. Inluding unit conversion.

        Note for offshore potentials: If added for customized regions, a shape file containing 
        the considered offshore regions is needed. 

        Parameters
        ----------
        technology : str
            Technology for which potentials shall be loaded. Must be a key in default_potentials.yaml (data folder) 
            to allow loading of default potentials, else all variables below must be specified explicitly. 
        commodity : str,
            The commodity for the added technology source.
        model_unit : str
            The unit of the model for the commodity affected by this source, e.g. 'GW'. If the model unit is s.th.
            like 'GW_H2_LHV' please only use the physical unit description, in this case 'GW'.
        data_unit : str, optional
            The capacity unit used in the potentials data to be loaded, e.g. 'kW'. Will be extracted from default information if
            default potentials are used, else required. Defaults to None.
        sub_dataset_name : str, optional
            If nc4 files contain several dataset layers, the dataset name can be specified here. If None given, a single dataset
            layer nc4 file is assumed unless sub dataset name is loaded from default potentials information, defaults to None.
            NOTE: <COSTYEAR> sub strings will be replaced by the respective model cost year.
        aggregation_dict : dict, optional
            A dictionary with timeseries names as keys and capacity attributes as values. The capacity values will be used
            as weighting factors for the respective timeseries key. Will be replaced by default information if default potentials
            are applied, else required. By default None.
        additional_aggregation_vars : list, optional
            A list of additional (e.g. capacity) variables that will be accumulated over the respective model locations. If None, only 
            the capacity attributes given as values in aggregation_dict will be accumulated, defaults to None.
        cluster_params : dict, optional
            FINE.source() parameters that shall be extracted per cluster from input data. The key must be a valid FINE.source() parameter,
            the value the corresponding attribute name in the input shp/pickle files, e.g. {'investPerCapacity':'capex'}. Will overwrite 
            default values from techno-economic dict. If None, cluster_params will be taken from default_potentials, set to empty dict {} 
            to ignore completely, by default None.
        daily_timeseries : (str, list), optional
            The parameter name of the daily timeseries that shall be coverted to hourly in the process, by default None. 
        hourly_reference_timeseries: (str, list), optional
            The parameter of the hourly time series that will serve as a reference for the sub-daily distribution when above
            daily_timeseries is converted to hourly. Required only when daily_timeseries is defined. If the daily_timeseries is
            defined as a list, hourly_reference_timeseries must be a list, too, with the same length, so that the first entry 
            will be used as reference for conversion of the first entry in daily_timeseries and so forth. If needed, list those
            references multiple times that are needed as a reference for more than one daily timeseries. By default None.
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
            total of 'N_clusters' (see above) will be defined for the whole model. If False,
            'N_clusters' are enforced per model location, filling missing clusters per region/
            location with zeros. Different regions will have different cost in the same cluster 
            numbers. By default False.
        verbose : bool, optional
            If True, additional progress statements will be printed, by default True.
        **FINE_args
            Keyword arguments will be passed on to FINE.Source(). 
        """
        
        technology=technology.lower()

        # update InputDataInfo in case of custom info
        InputDataInfo().update_tech_info(
            tech=technology, 
            update_data=locals(), 
            ignore_args=["use_partial_polygon_capacities", "N_clusters", "global_clusters"])

        # first assert these inputs that are not asserted in sub methods
        assert ModelTechnoEconomicData().get_data(component=technology,attribute="commodity") in self.esM.commodities, f"Selected commodity must be a model commodity, i.e. select from: {', '.join(self.esM.commodities)}"
        assert technology.lower() in set(ModelTechnoEconomicData().data.index.get_level_values("component")), f"Technology '{technology}' must be in source technologies defined in techno-economic data (case-insensitive): {', '.join(self.ted['sources'].keys())}"

        # extract cluster params from default potential information if available
        if cluster_params is None:
            try:
                cluster_params = InputDataInfo().get_info(tech=technology, attr='cluster_params')
            except KeyError:
                pass
        
        # then extract and preprocess all capacity and timeseries data
        potential_dict = self.ih.get_capacities_and_timeseries_from_nc4(
            technology=technology, 
            model_unit=model_unit,
            use_partial_polygon_capacities=use_partial_polygon_capacities,
            N_clusters=N_clusters, 
            global_clusters=global_clusters,
            verbose=verbose,
        )

        # if not defined, set the standard parameter names for ts and cap in GlobEP potentials
        if aggregation_dict is None and InputDataInfo().has_tech(tech=technology):
            # we have no aggregation dict, but we have a default potential, load first if needed
            # Note: This should be the case if None aggregation_dict since else non-default technology
            # would have failed earlier in self.ih.get_capacities_and_timeseries_from_nc4()
            aggregation_dict = InputDataInfo().get_info(tech=technology, attr='aggregation_dict')
        else:
            # an individual aggregation dict was passed
            assert isinstance(aggregation_dict, dict), f"The aggregation dict must be a dictionary with timeseries parameter names as keys and related capacity factor (e.g. for weighting) as values."
            assert len(aggregation_dict.keys())==1, f"For more than one timeseries/capacity per technology, complex loaders are required, see below."

        if verbose:
            if len(potential_dict.keys())==0:
                print(f"No potential clusters found for {technology} in any region, will not be added to model.")
            elif N_clusters is None:
                print(f"{len(potential_dict.keys())} potential clusters found for {technology} model-wide, will be added to model.")
            else:
                print(f"{len(potential_dict.keys())} potential clusters defined for {technology} per region (including empty clusters), will be added to model.")

        # iterate over the clusters to add them to the esM as separate components
        for cluster in potential_dict.keys():
            # define default FINE arguments
            FINE_args={
                'esM':self.esM,
                'name':f"{technology}__{str(cluster)}",
                'commodity':ModelTechnoEconomicData().get_data(component=technology, attribute='commodity'),
                'hasCapacityVariable':True,
                'operationRateMax': self._clip_close_to_zero(potential_dict[cluster][list(aggregation_dict.keys())[0]], self.zero_threshold),
                'capacityMax':potential_dict[cluster][list(aggregation_dict.values())[0]],
            }
            # checks all possible arguments for fine
            considerable_FINE_args = self._get_considerable_FINE_args("source")
            # get all args available in ted csv and set considered fine args list
            ted_att_list = ModelTechnoEconomicData().data[technology].keys().unique()
            consider_FINE_args = [arg for arg in considerable_FINE_args if arg in ted_att_list]
            print(f"List of considered FINE_args for {technology}, as available in ted:{list(FINE_args.keys())+consider_FINE_args}.") 
            # add all args to FINE_args dict, with get_data   
            for arg in consider_FINE_args:
                FINE_args[arg] = ModelTechnoEconomicData().get_data(component=technology,attribute=arg)

            # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
            FINE_args = {**FINE_args, **FINE_kwargs}

            # if given, replace FINE parameters with cluster specific values from potential_dict extracted from input data
            if not cluster_params is None:
                #update FINE args with input data
                FINE_args = {**FINE_args, **dict(zip(list(cluster_params.keys()), [potential_dict[cluster][v] for v in cluster_params.values()]))}
                # adapt units if extracted from cluster_params 
                for k,v in self.param_units.items():
                    if k in list(cluster_params.keys()):
                        if not "wind_offshore" in technology.lower():
                            raise NotImplementedError("Temporary remedy implemented for wind_offshore only. Will be resolved with modelManager-wide unit handling.")
                        try:
                            FINE_args[k]=v*FINE_args[k] #TODO the param_units values and handling will be adapted when clean unit handling is introduced
                        except:
                            pass

            # add the component for the respecitve cluster
            self.esM.add(fn.Source(**FINE_args,floorTechnicalLifetime=False))   # floorTechnicalLifetime=False --> TODO: Just for debugging, remove later

    def addStockWithTSBrownfield(
        self, 
        technology, 
        model_unit,
        aggregation_dict=None,
        cluster_params=None,
        use_partial_polygon_capacities=True,
        N_clusters=3,
        global_clusters=False,
        verbose=True,
        national_commissioning_data_path="/storage_cluster/shared_data/2023_gears/FineUnion/potentials/global_plant_database/IRENA_national_power_plant_capacities_ren.csv",
        stock=True,
        **FINE_kwargs,
        ):
        """
        A loader method for stock potentials XXXX. Potential data to be 
        loaded must follow the GlobEP step 2 output formatting. Inluding unit conversion.

        Note for offshore potentials: If added for customized regions, a shape file containing 
        the considered offshore regions is needed. 

        Parameters
        ----------
        technology : str
            Technology for which potentials shall be loaded. Must be a key in default_potentials.yaml (data folder) 
            to allow loading of default potentials, else all variables below must be specified explicitly. 
        model_unit : str
            The unit of the model for the commodity affected by this source, e.g. 'GW'. If the model unit is s.th.
            like 'GW_H2_LHV' please only use the physical unit description, in this case 'GW'.
        aggregation_dict : dict, optional
            A dictionary with timeseries names as keys and capacity attributes as values. The capacity values will be used
            as weighting factors for the respective timeseries key. Will be replaced by default information if default potentials
            are applied, else required. By default None.
        cluster_params : dict, optional
            FINE.source() parameters that shall be extracted per cluster from input data. The key must be a valid FINE.source() parameter,
            the value the corresponding attribute name in the input shp/pickle files, e.g. {'investPerCapacity':'capex'}. Will overwrite 
            default values from techno-economic dict. If None, cluster_params will be taken from default_potentials, set to empty dict {} 
            to ignore completely, by default None.
        use_partial_polygon_capacities : bool, optional
            If True, the capacity attributes of each polygon-shaped plant will be reduced linearly with the 
            plant area percentage that actually overlaps with the respective model location. Set to False to 
            save time at the cost of possibly (partially) counting plant capacities twice that overlap the 
            location shape borders. Will only be applied to custom-shaped model locations and if possible, i.e. 
            when the geom column contains (multi)polygon geometries. By default True.
        N_clusters : int, optional
            The number of clusters after aggregation that shall be added to the energy system 
            model, either per region or across the whole model (see below). If None, no further
            aggregation will be done. By default None.
        global_clusters : bool, optional
            If True, the cluster definitions will be the same for all model locations, i.e. a 
            total of 'N_clusters' (see above) will be defined for the whole model. If False,
            'N_clusters' are enforced per model location, filling missing clusters per region/
            location with zeros. Different regions will have different cost in the same cluster 
            numbers. By default False.
        verbose : bool, optional
            If True, additional progress statements will be printed, by default True.
        national_commissioning_data_path: str
            Used to get gid0_commissioning_df and country_cap df, containing gid0 as index and commissioning capacities and stock capacities as values. 
            A filpath to national commissioning data. Preprocessing for new filepath containing other data would need to be adjusted within in function.
            country cap: pd.series, index=GID_0,data=capacities as float
            gid0_commissioning_df: pd.DataFrame, index=GID_0,columns=commissioning_years,data=commissioning capacities as float
        stock: bool
            should always be True, as this function is made for loading stocks
        **FINE_args
            Keyword arguments will be passed on to FINE.Source(). 
        """
        
                
        def get_national_caps(
                technology=technology,
                model_unit=model_unit,
                verbose=verbose,
                use_partial_polygon_capacities=use_partial_polygon_capacities,
        ):
            # load national capacities for scaling factor
            source = NC4Source(
                technology=technology,
                model_unit=model_unit,
                verbose=verbose,
            )
            
            _country_df = ModelLocations().get_country_df()
            _country_df = _country_df[_country_df['GID_0'].isin(ModelLocations().get_main_country()['main_gid0'].unique())]
            _country_df.loc[_country_df["shore_type"]=="onshore","locationID"]=_country_df.loc[_country_df["shore_type"]=="onshore","GID_0"]
            _country_df = _country_df.loc[~_country_df["shore_type"].isin(["offshore"])]
            _country_df["dflt_type"]='custom'
            

            _defaultregions_per_location_dict = {}
            for gid0 in ModelLocations().get_main_country()['main_gid0'].unique():
                _defaultregions_per_location_dict[gid0] = get_raw_regions(shore_type='ONSHORE',country_list=[gid0])
                #_defaultregions_per_location_dict[f'{gid0}_off'] = get_raw_regions(shore_type='OFFSHORE',country_list=[gid0])

            source.load_placements(
                location_shape = _country_df,
                defaultregions_per_location_dict = _defaultregions_per_location_dict,
                use_partial_polygon_capacities=use_partial_polygon_capacities,
                region_col_name =InputDataInfo().get_info(tech=technology, attr="region_col_name"),
                verbose=verbose,
            )  
            
            placements_df = source.cluster_agg
            return placements_df.groupby("locationID")[placements_df.select_dtypes(include="number").columns].sum() / 1e6
        
        def scale_model_caps(
            model_caps,
            country_caps,
            model_caps_GID0,
            ):          
        
            model_caps_reg = model_caps.sum(axis=1).to_frame("capacity")
            model_caps_reg["GID_0"] = model_caps_reg.index.map(ModelLocations().get_main_country()['main_gid0'])
            
            scaling_factor_dict = {}
            for country, cap in model_caps_GID0.iterrows():
                scaling_factor = country_caps[country]/cap["capacity"]
                print(f"Official capacity for {country}: {country_caps[country]}")
                print(f"Model capacity for {country}: {cap['capacity']}")
                print(f"Scaling factor for {country}: {scaling_factor}")
                model_regions_in_country_sel = model_caps_reg[model_caps_reg["GID_0"]==country].index
                _scaling_factor_dict = {region: scaling_factor for region in model_regions_in_country_sel}
                scaling_factor_dict.update(_scaling_factor_dict)

            model_caps_scaled = model_caps.copy().mul(pd.Series(scaling_factor_dict),axis=0)
            return model_caps_scaled.to_dict(orient="series")

        
        def get_component_country_commissioning(gid0_commissioning_df, model_start_year, investment_period_interval):

            historic_min_year = int(gid0_commissioning_df.columns.min())
            # calcualte how many investment_periods are in the past
            investment_periods_in_past = (model_start_year - 1 - historic_min_year) // investment_period_interval
            investment_periods_in_past

            # calculate years of all past investment periods from model_start_year backwards
            years_of_past_investment_periods = [model_start_year - (investment_period_interval * (i+1)) for i in range(investment_periods_in_past)]
            years_of_past_investment_periods

            # get cumsum from historic_min_year to first year in years_of_past_investment_periods
            gid0_commissioning_df.columns = gid0_commissioning_df.columns.astype(int)
            comm_df = []
            for i, _year in enumerate(sorted(years_of_past_investment_periods)):
                real_start_year = _year
                
                if i == 0:
                    start_year = historic_min_year
                    end_year = _year + investment_period_interval -1
                    _df = gid0_commissioning_df.loc[:,slice(start_year,end_year)].sum(axis=1)
                    _df.name = real_start_year
                else:
                    start_year = _year
                    end_year = _year + investment_period_interval -1
                    _df = gid0_commissioning_df.loc[:,slice(start_year,end_year)].sum(axis=1)
                    _df.name = real_start_year
                comm_df.append(_df)
                print(i,start_year, end_year)
                print(f"Model Comissioning year: {real_start_year}")
                print(f"*"*4)
            country_commissioning_df = pd.concat(comm_df, axis=1)
            return country_commissioning_df


        def build_commissioning_dict_per_cluster(potential_dict, country_commissioning_df):
            model_caps = pd.concat([x["capacity"] for x in potential_dict.values()],axis=1,keys=potential_dict.keys())

            model_caps["GID_0"] = model_caps.index.map(ModelLocations().get_main_country()['main_gid0'])
            comm_full_list = []
            for _country, model_caps_cntr in model_caps.groupby("GID_0"):
                model_caps_cntr = model_caps_cntr.drop(columns="GID_0")
                
                # avoid division by 0
                if model_caps_cntr.sum().sum() == 0 or np.isnan(model_caps_cntr.sum().sum()):
                    share_per_cluster = model_caps_cntr.sum(axis=0)
                else:    
                    share_per_cluster = model_caps_cntr.sum(axis=0).div(model_caps_cntr.sum().sum())
                
                comp_comm_reg_dict = {}
                for cluster, cluster_caps in model_caps_cntr.groupby(level=0, axis=1):
                    if len(cluster_caps) > 1:
                        cluster_caps = cluster_caps.squeeze()
                    else:
                        cluster_caps = cluster_caps.squeeze(axis=1)
                    
                    # avoid division by 0
                    if cluster_caps.sum() == 0 or np.isnan(model_caps_cntr.sum().sum()):
                        cluster_caps_weights=cluster_caps.fillna(0)
                    else:
                        cluster_caps_weights = cluster_caps.div(cluster_caps.sum())
                    
                    cluster_share = share_per_cluster[cluster]
                    comm_df_cntr = country_commissioning_df.loc[_country]*cluster_share

                    cluster_caps_weights.name = _country
                    comm_df_reg = comm_df_cntr.to_frame().dot(cluster_caps_weights.to_frame().T).T
                    comp_comm_reg_dict[cluster] = comm_df_reg
                comm_full_list.append(comp_comm_reg_dict)

            comm_per_cluster = {}
            for i in range(0, N_clusters):
                if comm_full_list:
                    comm_dict = pd.concat([x[i] for x in comm_full_list], axis=0).to_dict(orient="series")
                else:
                    comm_dict = None
                comm_per_cluster[i] = comm_dict
            return comm_per_cluster

        # get base info: technology and economic lifetime
        technology=technology.lower()
        # get economicLifetime
        economicLifetime = ModelTechnoEconomicData().get_data(component=technology,attribute='economicLifetime')[0]

        # preprocess national stock capacity commissioning and capacity scaling-data
        
        if "irena" in national_commissioning_data_path.lower():
            # tech mapper
            tech_mapper ={
                'wind_onshore_stock':'wind_onshore',
                'wind_offshore_stock':'wind_offshore',
                'ofpv_hsat_stock':'solar_pv',
                'ofpv_fixed_stock':'solar_pv',
            }

            # get commissioning df from overhanded national_commissioning_data_path
            _gid0_commissioning_df_all = pd.read_csv(national_commissioning_data_path).drop(columns=["Unnamed: 0"]).set_index(["GID_0","plant_type"])
            _gid0_commissioning_df = _gid0_commissioning_df_all.xs(tech_mapper[technology], level="plant_type")
            # only keep commissioning till base_year
            gid0_commissioning_df = _gid0_commissioning_df.loc[:, _gid0_commissioning_df.columns.astype(int) <= InputDataInfo().base_year]
            # get country cap
            country_cap = gid0_commissioning_df.sum(axis=1)
        else:
            raise ValueError(f"No preprocessing implemented for national_commissioning_data_path:{national_commissioning_data_path} until now.")


        # update InputDataInfo in case of custom info
        InputDataInfo().update_tech_info(
            tech=technology, 
            update_data=locals(), 
            ignore_args=["use_partial_polygon_capacities", "N_clusters", "global_clusters"])

        # first assert these inputs that are not asserted in sub methods
        assert ModelTechnoEconomicData().get_data(component=technology,attribute="commodity") in self.esM.commodities, f"Selected commodity must be a model commodity, i.e. select from: {', '.join(self.esM.commodities)}"
        assert technology.lower() in set(ModelTechnoEconomicData().data.index.get_level_values("component")), f"Technology '{technology}' must be in source technologies defined in techno-economic data (case-insensitive): {', '.join(self.ted['sources'].keys())}"

        # extract cluster params from default potential information if available
        if cluster_params is None:
            try:
                cluster_params = InputDataInfo().get_info(tech=technology, attr='cluster_params')
            except KeyError:
                pass
        
        # then extract and preprocess all capacity and timeseries data
        potential_dict = self.ih.get_capacities_and_timeseries_from_nc4(
            technology=technology, 
            model_unit=model_unit,
            use_partial_polygon_capacities=use_partial_polygon_capacities,
            N_clusters=N_clusters, 
            global_clusters=global_clusters,
            verbose=verbose,
        )

        # if not defined, set the standard parameter names for ts and cap in GlobEP potentials
        if aggregation_dict is None and InputDataInfo().has_tech(tech=technology):
            # we have no aggregation dict, but we have a default potential, load first if needed
            # Note: This should be the case if None aggregation_dict since else non-default technology
            # would have failed earlier in self.ih.get_capacities_and_timeseries_from_nc4()
            aggregation_dict = InputDataInfo().get_info(tech=technology, attr='aggregation_dict')
        else:
            # an individual aggregation dict was passed
            assert isinstance(aggregation_dict, dict), f"The aggregation dict must be a dictionary with timeseries parameter names as keys and related capacity factor (e.g. for weighting) as values."
            assert len(aggregation_dict.keys())==1, f"For more than one timeseries/capacity per technology, complex loaders are required, see below."

        if verbose:
            if len(potential_dict.keys())==0:
                print(f"No potential clusters found for {technology} in any region, will not be added to model.")
            elif N_clusters is None:
                print(f"{len(potential_dict.keys())} potential clusters found for {technology} model-wide, will be added to model.")
            else:
                print(f"{len(potential_dict.keys())} potential clusters defined for {technology} per region (including empty clusters), will be added to model.")

        # get model caps per GID0
        model_caps_GID0 = get_national_caps()

        # Scale model capacities to official capacities
        model_caps = pd.concat([x["capacity"] for x in potential_dict.values()],axis=1,keys=potential_dict.keys())
        model_caps_reg = model_caps.sum(axis=1).to_frame("capacity")
        model_caps_reg["GID_0"] = model_caps_reg.index.map(ModelLocations().get_main_country()['main_gid0'])

        loaded_gid0_capacity_share = model_caps_reg.groupby("GID_0")["capacity"].sum()/model_caps_GID0["capacity"]

        if country_cap is not None:
            model_caps_scaled = scale_model_caps(
                model_caps = model_caps,
                country_caps = country_cap,
                model_caps_GID0=model_caps_GID0,
            )

            for cluster in potential_dict.keys():
                potential_dict[cluster]["capacity"] = model_caps_scaled[cluster]
        
        if gid0_commissioning_df is not None:
            country_commissioning_df = get_component_country_commissioning(gid0_commissioning_df, InputDataInfo().base_year, InputDataInfo().investment_period_interval)
            comissioning_per_cluster = build_commissioning_dict_per_cluster(potential_dict, country_commissioning_df.mul(loaded_gid0_capacity_share,axis=0))

        # iterate over the clusters to add them to the esM as separate components
        for cluster in potential_dict.keys():
            # define default FINE arguments
            FINE_args={
                'esM':self.esM,
                'name':f"{technology}__{str(cluster)}",
                'commodity':ModelTechnoEconomicData().get_data(component=technology, attribute='commodity'),
                'hasCapacityVariable':True,
                'operationRateMax': self._clip_close_to_zero(potential_dict[cluster][list(aggregation_dict.keys())[0]], self.zero_threshold),
               # 'capacityFix':potential_dict[cluster][list(aggregation_dict.values())[0]],
                'stockCommissioning': comissioning_per_cluster[cluster] if gid0_commissioning_df is not None else None,
            }
            # checks all possible arguments for fine
            considerable_FINE_args = self._get_considerable_FINE_args("source")
            # get all args available in ted csv and set considered fine args list
            ted_att_list = ModelTechnoEconomicData().data[technology].keys().unique()
            consider_FINE_args = [arg for arg in considerable_FINE_args if arg in ted_att_list]
            print(f"List of considered FINE_args for {technology}, as available in ted:{list(FINE_args.keys())+consider_FINE_args}.") 
            # add all args to FINE_args dict, with get_data   
            for arg in consider_FINE_args:
                FINE_args[arg] = ModelTechnoEconomicData().get_data(component=technology,attribute=arg,stock=stock,economicLifetime=economicLifetime)

            # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
            FINE_args = {**FINE_args, **FINE_kwargs}

            # if given, replace FINE parameters with cluster specific values from potential_dict extracted from input data
            if not cluster_params is None:
                #update FINE args with input data
                FINE_args = {**FINE_args, **dict(zip(list(cluster_params.keys()), [potential_dict[cluster][v] for v in cluster_params.values()]))}
                # adapt units if extracted from cluster_params 
                for k,v in self.param_units.items():
                    if k in list(cluster_params.keys()):
                        if not "wind_offshore" in technology.lower():
                            raise NotImplementedError("Temporary remedy implemented for wind_offshore only. Will be resolved with modelManager-wide unit handling.")
                        try:
                            FINE_args[k]=v*FINE_args[k] #TODO the param_units values and handling will be adapted when clean unit handling is introduced
                        except:
                            pass

            # add the component for the respective cluster
            self.esM.add(fn.Source(**FINE_args,floorTechnicalLifetime=False))   # floorTechnicalLifetime=False --> TODO: Just for debugging, remove later
        self.potential_dict = potential_dict


    def addPotentialCsvWithTSGreenfield(
            self,
            technology,
            cap_fp,
            cap_ts_fp,
            model_unit,
            **FINE_kwargs,
    ):
        '''
        Adds potentials based on csv sheets. Can be helpful for potential variations or generally new input potentials. 
        '''
        technology=technology.lower()

        ts_cap_factor_df = pd.read_csv(cap_ts_fp, delimiter=";",decimal=",")
        cap_df           = pd.read_csv(cap_fp, delimiter=";",index_col="Region",decimal=",")

        
        # delete unnecessary column in ts_cap_factor_df
        ts_cap_factor_df = ts_cap_factor_df.drop(ts_cap_factor_df.columns[0],axis=1)
        # cap_df to series
        cap_df           = cap_df.drop("Unnamed: 0", axis=1)
        # from hourly df to annual capacities series
        cap_srs           = cap_df["Power_Potential_Available_GW"]

        # get time series as dataframes, named as cluster in dict. Example: ts_cap_factor_df_dict["q90"] = ts_cap_factor_df_q90
        ts_cap_factor_df_dict, clusters = self._get_ts_dfs_from_csv(ts_cap_factor_df=ts_cap_factor_df)       

        # iterate over the clusters to add them to the esM as separate components
        sum = 0
        for cluster in clusters:

            # set share of capacities, that quantile has
            if cluster == "q90":
                cl_cap_share = 0.2
                sum = sum+cl_cap_share
            else:
                cl_cap_share = 0.4
                sum = sum+cl_cap_share
            assert sum <= 1, f"Sum of cluster shares of capacities is higher than 1 for {technology}!" 
            
            # define default FINE arguments
            FINE_args={
                'esM':self.esM,
                'name':f"{technology}__{str(cluster)}",
                'commodity':ModelTechnoEconomicData().get_data(component=technology, attribute='commodity'),
                'hasCapacityVariable':True,
                'operationRateMax': ts_cap_factor_df_dict[f'ts_cap_factor_df_{cluster}'],
                'capacityMax':cap_srs*cl_cap_share,
            }
            # checks all possible arguments for fine
            considerable_FINE_args = self._get_considerable_FINE_args("source")
            # get all args available in ted csv and set considered fine args list
            ted_att_list = ModelTechnoEconomicData().data[technology].keys().unique()
            consider_FINE_args = [arg for arg in considerable_FINE_args if arg in ted_att_list]
            print(f"List of considered FINE_args for {technology}, as available in ted:{list(FINE_args.keys())+consider_FINE_args}.") 
            # add all args to FINE_args dict, with get_data   
            for arg in consider_FINE_args:
                FINE_args[arg] = ModelTechnoEconomicData().get_data(component=technology,attribute=arg)

            # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
            FINE_args = {**FINE_args, **FINE_kwargs}
            # add the component for the respecitve cluster
            self.esM.add(fn.Source(**FINE_args,floorTechnicalLifetime=False))   # floorTechnicalLifetime=False --> TODO: Just for debugging, remove later
        
    def _get_ts_dfs_from_csv(self,ts_cap_factor_df):
        clusters = ["q90","q60","q20"]

        # Dictionary to store the new dataframes
        dataframes_dict = {}

        # Loop through each unique entry and create a new dataframe
        for cluster in clusters:
            columns = [col for col in ts_cap_factor_df.columns if cluster in col]
            new_df  = ts_cap_factor_df[columns]
            new_df = new_df.reset_index(drop=True)
            new_df = new_df.replace(',', '.', regex=True)
            new_df = new_df.astype(float)

            # clean column name
            new_df = self._clean_column_names(new_df) 
            
            # Dynamically create variables with appropriate names
            var_name = f'ts_cap_factor_df_{cluster}'
            globals()[var_name] = new_df
            dataframes_dict[var_name] = new_df
        return dataframes_dict, clusters
        
    # delete everything after the second point within the column names of a df
    def _clean_column_names(self,df):
        def remove_after_second_dot(col_name):
            parts = col_name.split('_')
            return '_'.join(parts[:2]) if len(parts) > 2 else col_name
        df.columns = [remove_after_second_dot(col) for col in df.columns]
        return df

    def addPotentialConst(
            self,
            technology,
            model_unit="GW",
            db_stock_path="",
            db_unit="GW",
            db_name='power-plant-matching-tool',
            **FINE_kwargs,
    ):
        '''
        Adds constant potentials to the esM (e.g. coal, gas or nuclear power plants). If plant database is passed it extracts the stock capacities per modelregion.

        technology: str
            technology type of to be loaded technology. eg: "EGS"
        model_unit: str
            unit of the model (e.g. GW, MW, ...)
        db_path: str
            if available overhands a file path to a database with min. longitutde, latitude and a capacity of plants.
        db_unit: str
            overhands the unit of the database
        db_name: str
            depending on the name of the database the preprocessing of the database is implemented 
        '''

        technology = technology.lower()
        
        # set fuel type
        if "gas" in technology:
            fuel = 'gas'
        elif "coal" in technology:
            fuel = 'coal'
        elif "brown" in technology or "lignite" in technology:
            fuel = 'lignite'
        elif "nuclear" in technology:
            fuel = 'nuclear' 
        else:
            raise ValueError(f"fuel for {technology} is not implemented yet.")

        # considering stock, if stock in data base:
        if db_stock_path =="":
            stock = False
        else:
            # economic lifetime later getting stock investment periods in get_data funtion
            economicLifetime = ModelTechnoEconomicData().get_data(component=technology,attribute='economicLifetime')[0]
            _stockCommissioning = get_stockCommissioning_dict(db_stock_path=db_stock_path,db_name=db_name,technology=technology,model_unit=model_unit,db_unit=db_unit,fuel=fuel)
            if _stockCommissioning:
                stock = True
            # if stock commissioning dict empty, no plants in data base:
            else:
                print(f'No stock: Given database {db_stock_path} does not contain placements of {technology}',flush=True)
                stock = False
        
        # define default FINE arguments
        FINE_args={
            'esM':self.esM,
            'name':f"{technology}",
            'commodity':ModelTechnoEconomicData().get_data(component=technology, attribute='commodity',stock=stock,economicLifetime=economicLifetime),
            'hasCapacityVariable':True,
        }
        
        # add stockCommissioning
        if stock:
                FINE_args['stockCommissioning'] = _stockCommissioning
        
        # checks all possible arguments for fine in ted
        considerable_FINE_args = self._get_considerable_FINE_args("source")
        # get all args available in ted csv and set considered fine args list
        ted_att_list = ModelTechnoEconomicData().data[technology].keys().unique()
        consider_FINE_args = [arg for arg in considerable_FINE_args if arg in ted_att_list]
        print(f"List of considered FINE_args for {technology}, as available in ted:{list(FINE_args.keys())+consider_FINE_args}.") 
        # add all args to FINE_args dict, with get_data   
        for arg in consider_FINE_args:
            FINE_args[arg] = ModelTechnoEconomicData().get_data(component=technology,attribute=arg,stock=stock,economicLifetime=economicLifetime)
        
        # overwrite operationRateMin, with operation rate min data frame, as fine demands
        if 'operationRateMin' in FINE_args.keys():
            FINE_args['operationRateMin'] = pd.DataFrame(
                ModelTechnoEconomicData().get_data(component=technology,attribute='operationRateMin'),
                index=list(range(0,self.esM.numberOfTimeSteps,1)),
                columns=ModelLocations().locationIDs
                )

        # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
        FINE_args = {**FINE_args, **FINE_kwargs}
        
        # add FINE_args to esM
        self.esM.add(fn.Source(**FINE_args,floorTechnicalLifetime=False))   # floorTechnicalLifetime=False --> TODO: Just for debugging, remove later 
    
    
    def addPotentialConstGreenfield(
        self,
        technology=None,
        N_cluster=None,
        path=None,
        LCOE_name=None,
        capacity_name=None,
        region_name_col=None,
        # capacity_to_GW_factor=None
        LCOE_to_EUR_per_kWh_factor=None,
        rounding=4,
        operationRateMax=1,
        CAPEX_attribute=None,
        LCOE_attribute_for_CAPEX='from_default',
        **FINE_kwargs,
    ):
        """Add a potential source with a constant time series to the self.esM object.

        Parameters
        ----------
        technology : str
            technology type of to be loaded technology. eg: "EGS"
        operationRateMax : int, optional
            constant operation rate value, by default 1
        Ncluster : int, optional
            number of different cost cluster to be added, by default 1
        rounding : int, optional
            rounding if inputs, by default 4
        CAPEX_attribute : str, optional
            If given, specific CAPEX per plant (in cost per capacity unit) will be expected in 
            .shp/.pickle file attribute name. Cannot co-exist with LCOE_attribute_for_CAPEX. 
            By default None.
        LCOE_attribute_for_CAPEX : str, optional
            If given, CAPEX per cluster will be extracted from average LCOE per cluster, assuming 
            unit EUR/kWh. If 'from_default', capex will be extractd from default_potentials.yaml.
            Cannot co-exist with CAPEX_attribute. Set to None to ignore. By default 'from_default'.
        **FINE_kwargs 
            Will be passed on to FINE.Source().

        Raises
        ------
        ValueError
            technology is not properly defined
        """
        technology=technology.lower()
        assert not (CAPEX_attribute is not None and LCOE_attribute_for_CAPEX is not None), f"LCOE_attribute_for_CAPEX and CAPEX_attribute cannot both be given."
        
        # define base FINE args - note: name, capacityMax and investPerCapacity/opexPerCapacity will be defined per cluster
        FINE_args={
            'esM':self.esM,
            'commodity':ModelTechnoEconomicData().get_data(component=technology, attribute='commodity'),
            'hasCapacityVariable':True,
            'operationRateMax':round(operationRateMax, rounding),
            'interestRate':ModelTechnoEconomicData().get_data(component=technology, attribute='interestRate'),
            'economicLifetime':ModelTechnoEconomicData().get_data(component=technology,attribute='economicLifetime'),
            'opexPerOperation':ModelTechnoEconomicData().get_data(component=technology,attribute='opexPerOperation'),
        }
        # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
        FINE_args = {**FINE_args, **FINE_kwargs}
    
        #TODO process CAPEX_attribute param
        if technology.lower() == "geothermal_egs":
            potential_dict = self.ih.load_constant_potentials(
                technology=technology,
                N_cluster=N_cluster, # storages do nat have flhs, so this does not make sense for greenfield storages
                path=path,   
                LCOE_name=LCOE_name,
                capacity_name=capacity_name,
                region_name_col=region_name_col,
                model_unit=UnitHandling().get_model_unit_as_multiple_of_SI_unit(commodity=FINE_args['commodity']),
                LCOE_to_EUR_per_kWh_factor=LCOE_to_EUR_per_kWh_factor, 
                rounding=rounding,
                _timeout=60,
                verbose=False,
            )
        elif technology == "OTHER":
            raise ValueError(f"Now workflow defined. Be creative!")
        else:
            raise ValueError(f"Technology {technology} not implemented.")
        
        # if LCOE_attribute_for_CAPEX is 'from_default', check if by default one is defined in default potential params
        if LCOE_attribute_for_CAPEX == 'from_default':
            try:
                # try to edefault potential param for technology, will fail if technology all lower case is not a key
                InputDataInfo().has_tech(technology, fail_on_false=True)
                LCOE_attribute_for_CAPEX = InputDataInfo().get_info(tech=technology, attr="LCOE_attribute_for_CAPEX")
            except:
                # if no default value can be extracted, set to None
                LCOE_attribute_for_CAPEX = None
        # get CAPEX based on average LCOE from cluster if required
        if not LCOE_attribute_for_CAPEX is None:
            assert LCOE_attribute_for_CAPEX in potential_dict[0].keys()
            investPerCapacity_dict = {}
            for cluster in potential_dict.keys():

                investPerCapacity_dict[cluster] = self._getCAPEXfromLCOE(
                    LCOE_EUR_per_kWh=potential_dict[cluster][LCOE_attribute_for_CAPEX],
                    fixOPEX_CAPEX_per_a=ModelTechnoEconomicData().get_data(component=technology,attribute='opexPerCapacity'),
                    varOPEX_notdefined=0,
                    lifetime_a=ModelTechnoEconomicData().get_data(component=technology,attribute="economicLifetime"),
                    WACC_1=ModelTechnoEconomicData().get_data(component=technology,attribute="interestRate"),
                    meanCF=operationRateMax,
                    self=self,
                )
        else:
            pass

        # manipulate operationRateMax to be a proper pd.DataFrame
        operationRateMaxInput = operationRateMax
        # get index and values
        regions = list(potential_dict[0]["capacityMax"].index)
        operationRateMaxValues = operationRateMaxInput * np.ones((ModelTechnoEconomicData().esm_params["esM"]["numberOfTimeSteps"], len(regions)))
        # create pd.DaraFrame
        operationRateMax = pd.DataFrame(
            operationRateMaxValues,
            index=range(ModelTechnoEconomicData().esm_params["esM"]["numberOfTimeSteps"]),
            columns=regions,
        )
        # double check pd.DaraFrame
        assert np.allclose(operationRateMax, operationRateMaxInput)  # check that nothing stupid happens
        FINE_args['operationRateMax'] = operationRateMax

        # Write the data into FINE
        for cluster in potential_dict.keys():

            if not LCOE_attribute_for_CAPEX is None:
                investPerCapacity = investPerCapacity_dict[cluster]
            else:
                investPerCapacity = ModelTechnoEconomicData().get_data(component=technology,attribute="investPerCapacity")

            # update FINE kwargs with cluster-specific values
            FINE_args_cluster = {
                **FINE_args, 
                **{
                    'name':f"{technology}__cluster_{str(cluster).zfill(3)}",
                    'capacityMax':round(potential_dict[cluster]["capacityMax"], rounding),
                    'investPerCapacity':round(investPerCapacity, rounding),
                    'opexPerCapacity': {
                        key: df.mul(investPerCapacity).round(rounding) 
                        for key, df in ModelTechnoEconomicData().get_data(component=technology, attribute="opexPerCapacity").items()
                    }
                    #'opexPerCapacity':round(ModelTechnoEconomicData().get_data(component=technology,attribute="opexPerCapacity") * investPerCapacity, rounding),
                }
            }
            # overwrite once more with function kwargs to ensure that they are used also for investPerCapacity etc
            FINE_args_cluster = {**FINE_args_cluster, **FINE_kwargs}

            self.esM.add(fn.Source(**FINE_args_cluster))
            del FINE_args_cluster
        pass
    # add commodity purchase
    def addCommodityPurchase(
            self,
            technology,
            hasCapacityVariable = False,
            factor=1,
            model_unit=None,
            **FINE_kwargs,
    ):
        ''' 
        Function for adding used commodities, mostly materials, that only have commodityCosts 

        Input:
            technology: str 
                name of technology

        Output:
            Commodity cost added to FINE as cost for a sink
        '''
        technology=technology.lower()
        print(f"add {technology}",flush=True)

        # define default FINE args
        FINE_args = {
            'esM':self.esM,
            'name': technology,
            'commodity':ModelTechnoEconomicData().get_data(component=technology,attribute="commodity"),
            'hasCapacityVariable':hasCapacityVariable,
            'commodityCost':self._multiply(data_to_multiply=ModelTechnoEconomicData().get_data(component=technology,attribute="commodityCost"),factor=factor),
        }
        # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
        FINE_args = {**FINE_args, **FINE_kwargs}
        self.esM.add(fn.Source(**FINE_args,floorTechnicalLifetime=False))   # floorTechnicalLifetime=False --> TODO: Just for debugging, remove later))        
        
    def add_csp(self, representation="heat", N_clusters=1, model_unit="GW", global_clusters=False):
        """Add CSP the david way"""
        print("add CSP")
        
        assert representation in ["elec", "heat"]

    
        if representation == "heat":

            #loadvars
            potential_dict_Heliosol = self.ih.get_capacities_and_timeseries_from_nc4(
                technology="csp_heliosol",
                model_unit=model_unit,
                N_clusters=N_clusters, 
                global_clusters=global_clusters,
                use_partial_polygon_capacities=False,
            )

            potential_dict_SolarSalt = self.ih.get_capacities_and_timeseries_from_nc4(
                technology="csp_solarsalt",
                model_unit=model_unit,
                N_clusters=N_clusters, 
                global_clusters=global_clusters,
                use_partial_polygon_capacities=False,
            )


            #addcomponents
            HTFs = ['he', 'ss']
            potential_dicts = [potential_dict_Heliosol, potential_dict_SolarSalt]

            for potential_dict, HTF in zip(potential_dicts, HTFs):
                
                #add commodities
                self._add_commodities_and_units(new_commodity_unit_dict={f"heat_csp_{HTF}": ("GW$_{heat}$", "GW")}) #TODO remove hardcode

                for cluster in list(potential_dict.keys()):

                    # 1) Source
                    solar_field_name = f"csp_{HTF}_sf"

                    #correct base invPer CapacityBy DNI reference values!
                    dni_reference = potential_dict[cluster]["DNInom_Wm2"]
                    # there is a known bug, which return all results with a scaling factor, but not all need one.
                    # this is a quick fix. to solve this. To make sure nothing unkown happens, assert that the resutls are in the right range!
                    dni_reference = dni_reference *1E9
                    assert np.logical_or(
                        np.logical_and(
                            dni_reference>100,
                            dni_reference<2000
                        )
                        , np.isclose(dni_reference, 0)
                    ).all()
                    dni_reference[np.isclose(potential_dict[cluster]['Csf_W'],0)] = 830

                    investPerCapacity = self.ted["sources"][solar_field_name]["investPerCapacity"][self.cost_year] * 830 / dni_reference

                    self.esM.add(
                        fn.Source(
                            esM=self.esM,
                            name=f"{solar_field_name}__cluster_{str(int(cluster))}",
                            commodity=self.ted["sources"][solar_field_name]["commodity"],
                            hasCapacityVariable=True,
                            operationRateMax=self._clip_close_to_zero(potential_dict[cluster]['ts_capacity_factor_sf'].reset_index(drop=True), self.zero_threshold),
                            capacityMax=potential_dict[cluster]['Csf_W'], # already GW by 'unitConversionFactor'
                            investPerCapacity=investPerCapacity,
                            opexPerCapacity=self.ted["sources"][solar_field_name]["opexFix"] * investPerCapacity,
                            interestRate=self.ted["sources"][solar_field_name]["interestRate"],
                            economicLifetime=self.ted["sources"][solar_field_name]["economicLifetime"],
                            sharedPotentialID=f"PV_cluster_{str(cluster)}",
                            #linkedQuantityID=f'CSP_{HTF}_SF_capacity_Cluster{str(cluster)}',
                        )
                    )

                    if False:
                        #solar field heat demand
                        self.esM.add(
                            fn.Sink(
                                esM=self.esM,
                                name=f"CSP_{HTF}_SF_demand_Cluster_{str(cluster)}",
                                commodity=f"heat_csp_{HTF}",
                                hasCapacityVariable=True,
                                linkedQuantityID=f'CSP_{HTF}_SF_capacity_Cluster{str(cluster)}',
                                operationRateFix=data_as_dict[cluster]['ts_capacity_factor_heat_FP_sf'].reset_index(drop=True) * (-1),
                            )
                        )

                # 2) heat storage

                storage_name = f"csp_{HTF}_storage"
                self.esM.add(
                    fn.Storage(
                        esM=self.esM,
                        name=storage_name,
                        commodity=self.ted["storage"][storage_name]["commodity"],
                        hasCapacityVariable=True,
                        chargeEfficiency=self.ted["storage"][storage_name]["chargeEfficiency"],
                        dischargeEfficiency=self.ted["storage"][storage_name]["dischargeEfficiency"],
                        cyclicLifetime=self.ted["storage"][storage_name]["cyclicLifetime"],
                        selfDischarge=self.ted["storage"][storage_name]["selfDischarge"],
                        chargeRate=self.ted["storage"][storage_name]["chargeRate"],
                        dischargeRate=self.ted["storage"][storage_name]["dischargeRate"],
                        investPerCapacity=self.ted["storage"][storage_name]["investPerCapacity"][self.cost_year], #22.5EUR/kWh = 22.5MEUR/GWh = 0.0225BEUR/GWh
                        opexPerCapacity=self.ted["storage"][storage_name]["opexFix"] * self.ted["storage"][storage_name]["investPerCapacity"][self.cost_year],
                        opexPerChargeOperation=self.ted["storage"][storage_name]["opexPerChargeOperation"],
                        opexPerDischargeOperation=self.ted["storage"][storage_name]["opexPerDischargeOperation"],
                        interestRate=self.ted["storage"][storage_name]["interestRate"],
                        economicLifetime=self.ted["storage"][storage_name]["economicLifetime"],
                    )
                )

                # 3) power plant
                plant_name = f"csp_{HTF}_powerplant"
                self.esM.add(
                    fn.Conversion(
                        esM=self.esM,
                        name=plant_name,
                        physicalUnit=self.ted["conversion"][plant_name]["physicalUnit"],
                        commodityConversionFactors=self.ted["conversion"][plant_name]["commodityConversionFactors"],
                        hasCapacityVariable=True,
                        investPerCapacity=self.ted["conversion"][plant_name]["investPerCapacity"][self.cost_year], #883 EUR/kW = 883MEUR/GW = 0.883 BEUR/GW
                        opexPerCapacity=self.ted["conversion"][plant_name]["opexFix"] * self.ted["conversion"][plant_name]["investPerCapacity"][self.cost_year],
                        opexPerOperation=self.ted["conversion"][plant_name]["opexPerOperation"],
                        interestRate=self.ted["conversion"][plant_name]["interestRate"],
                        economicLifetime=self.ted["conversion"][plant_name]["economicLifetime"],
                    )
                )
                
                if False:
                    #electric heating
                    self.esM.add(
                        fn.Conversion(
                            esM=self.esM,
                            name=f"CSP_{HTF}_elec_heating",
                            physicalUnit=r'GW$_{th}$',
                            commodityConversionFactors={"electricity": -1, f"heat_csp_{HTF}": 1},
                            hasCapacityVariable=True,
                            investPerCapacity=0.01,
                            opexPerCapacity=0,
                            interestRate=0.08,
                            economicLifetime=25,
                        )
                    )
        
        elif representation == "elec":
            raise NotImplementedError("Techno economics need to be adapted")
            CapexFromCluster=True
            potential_dict = self.ih.get_potential_data_CSP(
                weather_year=weather_year,
                N_clusters=N_clusters,
                Lea_szenario=Lea_szenario,
                datasetname=None
            )

            #add commodities
            self._add_commodities_and_units(new_commodity_unit_dict={f"elec_csp": ("GW$_{el}$", "GW")}) #TODO remove hard code

            #get costs
            if CapexFromCluster:
                assert "LCOE_clstr" in potential_dict[0].keys()
                # get CAPEX based on average LCOE from cluster
                investPerCapacity_dict = {}
                for cluster in potential_dict.keys():

                    investPerCapacity_dict[cluster] = self._getCAPEXfromLCOE(
                        LCOE_EUR_per_kWh=potential_dict[cluster]["LCOE_clstr"],
                        fixOPEX_CAPEX_per_a=0.02,
                        varOPEX_notdefined=0,
                        lifetime_a=25,
                        WACC_1=0.08,
                        meanCF=potential_dict[cluster]['ts_capacity_factor_plant'].mean(),
                        self=self,
                    )
            else:
                raise NotImplementedError

            for cluster in potential_dict.keys():
                #get capex based on lcoe
                eta_plant = 0.4
                tes = (potential_dict[cluster]['Cstr_kWh'] / (potential_dict[cluster]['Cplant_W'] / eta_plant) * 1000).mean()
                #sm = potential_dict[cluster]['Csf_W'] / (potential_dict[cluster]['Cplant_W'] / eta_plant)

                self.esM.add(
                    fn.Source(
                        esM=self.esM,
                        name=f"CSP_elec_Cluster_{str(cluster)}",
                        commodity=f"elec_csp",
                        hasCapacityVariable=True,
                        operationRateMax=potential_dict[cluster]['ts_capacity_factor_plant'],
                        capacityMax=potential_dict[cluster]['Cplant_W'], # already GW by 'unitConversionFactor'
                        investPerCapacity=investPerCapacity_dict[cluster],
                        opexPerCapacity=investPerCapacity_dict[cluster] * 0.02,
                        interestRate=0.08,
                        economicLifetime=25,
                        capacityPerPlantUnit=1,
                        linkedQuantityID=f'CSP_SF_capacity_Cluster{str(cluster)}',
                    )
                )

                self.esM.add(
                    fn.Storage(
                        esM=self.esM,
                        name=f"CSP_storage_{str(cluster)}",
                        commodity=f"elec_csp",
                        hasCapacityVariable=True,
                        chargeEfficiency=0.99,
                        cyclicLifetime=10000,
                        dischargeEfficiency=0.99,
                        selfDischarge=0.01 / 24,
                        chargeRate=1/tes,
                        dischargeRate=1/tes,
                        doPreciseTsaModeling=False,
                        investPerCapacity=0, #22.5EUR/kWh = 22.5MEUR/GWh = 0.0225BEUR/GWh
                        opexPerCapacity=0,
                        interestRate=0.08,
                        economicLifetime=25,
                        capacityPerPlantUnit=tes,
                        linkedQuantityID=f'CSP_SF_capacity_Cluster{str(cluster)}'
                    )
                )

            self.esM.add(
                fn.Conversion(
                    esM=self.esM,
                    name="CSP_powerplant",
                    physicalUnit=r"GW$_{el}$", # remove hard code
                    commodityConversionFactors={"elec_csp": -1, "electricity": 1},
                    hasCapacityVariable=True,
                    capacityPerPlantUnit=1,
                    linkedQuantityID=f'CSP_SF_capacity_Cluster{str(cluster)}'

                )
            )
        else:
            raise ValueError(representation)
            

    def addPotentialConstBrownfield(self, technology="ExistingCoal", Ncluster=None):
        raise NotImplementedError("Not implemented. Have fun doing so ;)") # TODO
        print(f"Adding: {technology}")
        pass

    # Transmission:
    # Transmission:
    def addGridBrownfield(self, technology, model_unit, data_unit=None, path_grids=None):
        technology=technology.lower()
        print(f"Adding: {technology}_brownfield")

        # assert f"{technology}_onshore" in self.ted["transmission"].keys(), f"Technology {technology}_onshore not found in ted"

        print(technology)
        # load transmission data
        locationID_column = self.locationID_column
        transmissionVars = self.ih.load_existing_electricity_grid(
            technology, model_unit, locationID_column, data_unit, path_grids
        )
        if transmissionVars == None:
            print("No transmission data found in regions. Skipping this technology.")
            pass

        commodity = self.ted["transmission"][f"{technology}_onshore"]["commodity"]
        opexPerOperation = self.ted["transmission"][f"{technology}_onshore"]["opexPerOperation"]
        economicLifetime = self.ted["transmission"][f"{technology}_onshore"]["economicLifetime"]
        opexPerCapacity = (
            self.ted["transmission"][f"{technology}_onshore"]["opexFix"]
            * self.ted["transmission"][f"{technology}_onshore"]["investPerCapacity"][self.cost_year]
        )
        losses = self.ted["transmission"][f"{technology}_onshore"]["losses"]
        interestRate = self.ted["transmission"][f"{technology}_onshore"]["interestRate"]

        self.esM.add(
            fn.Transmission(
                esM=self.esM,
                name=f"{technology}_brownfield",
                commodity=commodity,
                hasCapacityVariable=True,
                distances=transmissionVars["distances"],
                locationalEligibility=transmissionVars["locationalEligibility"],
                capacityFix=transmissionVars["capacityFix"],
                investPerCapacity=0,
                opexPerOperation=opexPerOperation,
                opexPerCapacity=opexPerCapacity,
                losses=losses,
                economicLifetime=economicLifetime,
                interestRate=interestRate,
            ),
        )

    def addGridGreenfield(self, technology="electricity_grid", detour_factor=1.4):
        technology=technology.lower()
        if len(self.esM.locations) < 2:
            print(f"Cannot add tranmission {technology} if there are less than 2 regions. Skipped.")
            return

        print(f"Adding {technology}")
        assert technology in ['electricity_grid','hydrogengas_pipeline', 'testingparams_grid']

        # load transmission data
        transmissionVars = self.ih.load_transmission_vars(detour_factor=detour_factor)

        # Calculate Onshore/Offshore dependend costs
        share_onshore = transmissionVars['share_onshore']
        share_offshore = (1- transmissionVars['share_onshore'].replace(0,np.nan)).fillna(1)
        
        # prepare dicts
        investPerCapacityOnshore = {}
        investPerCapacityOffshore = {}
        investPerCapacity = {}
        opexPerCapacityOnshore = {}
        opexPerCapacityOffshore = {}
        opexPerCapacity = {}
        opexPerOperation = {}
        losses = {}
        interestRate = {}
        economicLifetime = {}

        for ip in list(map(int,InputDataInfo().investment_period_names)):    
            # investpercapacity
            investPerCapacityOnshore[ip] = share_onshore * ModelTechnoEconomicData().get_data(component=f"{technology}_onshore",attribute="investPerCapacity")[ip].mean() # TODO: currently average values of all regions. Re-write, with approach for regional investment costs df, from get_data regional costs dict.
            investPerCapacityOffshore[ip] = share_offshore * ModelTechnoEconomicData().get_data(component=f"{technology}_offshore",attribute="investPerCapacity")[ip].mean()
            investPerCapacity[ip] = investPerCapacityOnshore[ip] + investPerCapacityOffshore[ip]
            # opexpercapacity
            opexPerCapacityOnshore[ip] = share_onshore * ModelTechnoEconomicData().get_data(component=f"{technology}_onshore",attribute="opexPerCapacity")[ip].mean()
            opexPerCapacityOffshore[ip] = share_offshore * ModelTechnoEconomicData().get_data(component=f"{technology}_offshore",attribute="opexPerCapacity")[ip].mean()
            opexPerCapacity[ip] = opexPerCapacityOnshore[ip] + opexPerCapacityOffshore[ip]
            # opexperoperation
            opexPerOperation[ip] = share_onshore * ModelTechnoEconomicData().get_data(component=f"{technology}_onshore",attribute="opexPerOperation")[ip].mean() \
                + share_offshore * ModelTechnoEconomicData().get_data(component=f"{technology}_offshore",attribute="opexPerOperation")[ip].mean()
        
        # currently not yearly differentiated as fine does not allow:
        losses = share_onshore * ModelTechnoEconomicData().get_data(component=f"{technology}_onshore",attribute="losses") \
            + share_offshore * ModelTechnoEconomicData().get_data(component=f"{technology}_offshore",attribute="losses") 
        # #economicLifetime
        economicLifetime = share_onshore * ModelTechnoEconomicData().get_data(component=f"{technology}_onshore",attribute="economicLifetime").mean() \
            + share_offshore * ModelTechnoEconomicData().get_data(component=f"{technology}_offshore",attribute="economicLifetime").mean()
        #interestRate -> 
        interestRate = share_onshore * ModelTechnoEconomicData().get_data(component=f"{technology}_onshore",attribute="interestRate").mean() \
            + share_offshore * ModelTechnoEconomicData().get_data(component=f"{technology}_offshore",attribute="interestRate").mean()
        # commodity
        assert ModelTechnoEconomicData().get_data(component=f"{technology}_onshore",attribute="commodity") == ModelTechnoEconomicData().get_data(component=f"{technology}_offshore",attribute="commodity")
        commodity = ModelTechnoEconomicData().get_data(component=f"{technology}_onshore",attribute="commodity")
        
        # add Transmission
        self.esM.add(
            fn.Transmission(
                esM=self.esM,
                name=technology,
                commodity=commodity,
                hasCapacityVariable=True,
                distances=transmissionVars["distances"],
                locationalEligibility=transmissionVars["locationalEligibility"],
                investPerCapacity=investPerCapacity,
                opexPerOperation=opexPerOperation,
                opexPerCapacity=opexPerCapacity,
                losses=losses,
                economicLifetime=economicLifetime,
                interestRate=interestRate,
            ),
        )

    # Conversion:
    def addConversionUnlimitedGreenfield(self, technology,regionalized_commodityConversionFactors=False, **FINE_kwargs):
        """
        Add a conversion option to self.esM object.

        Parameters
        ----------
        technology : str
            technology type of to be loaded technology. eg: "SaltCaverns"
        **FINE_kwargs 
            Will be passed on to FINE.Source().
        """
        technology=technology.lower()
        print(f"Adding: {technology}")

        # define default FINE arguments
        if regionalized_commodityConversionFactors:
            for single_location in list(ModelLocations().locationIDs):
                locationalEligibility = pd.Series(0,index=ModelLocations().locationIDs)
                locationalEligibility[single_location] = 1
                FINE_args={
                    'esM':self.esM,
                    'name':f'{technology}__{single_location}',
                    'physicalUnit':ModelTechnoEconomicData().get_data(component=technology, attribute="physicalUnit",single_location=single_location),
                    'commodityConversionFactors':ModelTechnoEconomicData().get_data(component=technology, attribute="commodityConversionFactors",single_location=single_location),
                    'hasCapacityVariable':True,
                    'locationalEligibility':locationalEligibility,
                }    
                # checks all possible arguments for fine
                considerable_FINE_args = self._get_considerable_FINE_args("conversion")
                # get all args available in ted csv and set considered fine args list
                ted_att_list = ModelTechnoEconomicData().data[technology].keys().unique()
                consider_FINE_args = [arg for arg in considerable_FINE_args if arg in ted_att_list]
                print(f"List of considered FINE_args for {technology}, as available in ted:{list(FINE_args.keys())+consider_FINE_args}.") 
                # add all args to FINE_args dict, with get_data
                for arg in consider_FINE_args:
                    FINE_args[arg] = ModelTechnoEconomicData().get_data(component=technology,attribute=arg,single_location=single_location)
            
                # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
                FINE_args = {**FINE_args, **FINE_kwargs}
                self.esM.add(fn.Conversion(**FINE_args,floorTechnicalLifetime=False)) # TODO: remove floorTechnicalLifetime=False as soon as data correct (lifetime can not be shorter then interval)
        else:
            # no regionalized commodityConversionFactors
            FINE_args={
                'esM':self.esM,
                'name':f'{technology}',
                'physicalUnit':ModelTechnoEconomicData().get_data(component=technology, attribute="physicalUnit"),
                'commodityConversionFactors':ModelTechnoEconomicData().get_data(component=technology, attribute="commodityConversionFactors"),
                'hasCapacityVariable':True,
            }
            # checks all possible arguments for fine
            considerable_FINE_args = self._get_considerable_FINE_args("conversion")
            # get all args available in ted csv and set considered fine args list
            ted_att_list = ModelTechnoEconomicData().data[technology].keys().unique()
            consider_FINE_args = [arg for arg in considerable_FINE_args if arg in ted_att_list]
            print(f"List of considered FINE_args for {technology}, as available in ted:{list(FINE_args.keys())+consider_FINE_args}.") 
            # add all args to FINE_args dict, with get_data   
            for arg in consider_FINE_args:
                FINE_args[arg] = ModelTechnoEconomicData().get_data(component=technology,attribute=arg)            

            # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
            FINE_args = {**FINE_args, **FINE_kwargs}
            self.esM.add(fn.Conversion(**FINE_args,floorTechnicalLifetime=False)) # TODO: remove floorTechnicalLifetime=False as soon as data correct (lifetime can not be shorter then interval)


    # Storage:
    def addStorageUnlimitedGreenfield(self, technology, cost_factor=1, **FINE_kwargs):
        """
        Add a storage option without storage limitations to self.esM object.

        Parameters
        ----------
        technology : str
            technology type of to be loaded technology. eg: "SaltCaverns"
        cost_factor : float, optional
            Increases CAPEX and OPEX for the given technology linearly,
            defaults to 1.
        **FINE_kwargs 
            Will be passed on to FINE.Source().
        """
        
        technology=technology.lower()
        
        # extract the default parameters for the component
        FINE_args={
            'esM':self.esM,
            'name':technology,
            'commodity':ModelTechnoEconomicData().get_data(component=technology, attribute='commodity'),
            'hasCapacityVariable':True,
        }
        # checks all possible arguments for fine
        considerable_FINE_args = self._get_considerable_FINE_args("storage")
        # get all args available in ted csv and set considered fine args list
        ted_att_list = ModelTechnoEconomicData().data[technology].keys().unique()
        consider_FINE_args = [arg for arg in considerable_FINE_args if arg in ted_att_list]
        print(f"List of considered FINE_args for {technology}, as available in ted:{list(FINE_args.keys())+consider_FINE_args}.") 
        # add all args to FINE_args dict, with get_data    
        for arg in consider_FINE_args:
            FINE_args[arg] = ModelTechnoEconomicData().get_data(component=technology,attribute=arg)

        # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
        FINE_args = {**FINE_args, **FINE_kwargs}

        self.esM.add(fn.Storage(**FINE_args,floorTechnicalLifetime=False))   # floorTechnicalLifetime=False --> TODO: Just for debugging, remove later

    def addStorageLimitedGreenfield(
        self,
        technology,
        path=None,
        LCOE_name=None,
        capacity_name=None,
        region_name_col=None,
        LCOE_to_EUR_per_kWh_factor=None,
        rounding=4,
        cost_factor=1,
        capacity_factor=1,
        **FINE_kwargs
    ):
        """
        Add a storage option with regional storage limitations to self.esM object.

        Parameters
        ----------
        technology : str
            technology type of to be loaded technology. eg: "SaltCaverns"

        #TODO @d.franzmann
        
        cost_factor : float, optional
            Increases CAPEX and OPEX for the given technology linearly,
            defaults to 1.
        **FINE_kwargs 
            Will be passed on to FINE.Source().
        """
        technology=technology.lower()
        
        print(f"Adding: {technology}")

        # extract the default parameters for the component
        FINE_args={
            'esM':self.esM,
            'name':f'{technology}',
            'commodity':ModelTechnoEconomicData().get_data(component=technology, attribute='commodity'),
            'hasCapacityVariable':True,
            'chargeEfficiency':ModelTechnoEconomicData().get_data(component=technology, attribute='chargeEfficiency'),
            'dischargeEfficiency':ModelTechnoEconomicData().get_data(component=technology, attribute='dischargeEfficiency'),
            'cyclicLifetime':ModelTechnoEconomicData().get_data(component=technology, attribute='cyclicLifetime'),
            'selfDischarge':ModelTechnoEconomicData().get_data(component=technology, attribute='selfDischarge'),
            'chargeRate':ModelTechnoEconomicData().get_data(component=technology, attribute='chargeRate'),
            'dischargeRate':ModelTechnoEconomicData().get_data(component=technology, attribute='dischargeRate'),
            'doPreciseTsaModeling':False,
            'opexPerChargeOperation':ModelTechnoEconomicData().get_data(component=technology, attribute='opexPerChargeOperation'),
            'opexPerDischargeOperation':ModelTechnoEconomicData().get_data(component=technology, attribute='opexPerDischargeOperation'),
            'interestRate':ModelTechnoEconomicData().get_data(component=technology, attribute='interestRate'),
            'economicLifetime':ModelTechnoEconomicData().get_data(component=technology, attribute='economicLifetime'),
        }
        # concatenate the default dict with the **kwargs, overwriting defaults by kwargs
        FINE_args = {**FINE_args, **FINE_kwargs}

        capacity_max = self.ih.load_constant_potentials(
            technology=technology,
            N_cluster=1, # storages do nat have flhs, so this does not make sense for greenfield storages
            path=path,   
            LCOE_name=LCOE_name,
            capacity_name=capacity_name,
            region_name_col=region_name_col,
            model_unit=UnitHandling().get_model_unit_as_multiple_of_SI_unit(commodity=FINE_args['commodity'])+'*h', #storage: x h
            LCOE_to_EUR_per_kWh_factor=LCOE_to_EUR_per_kWh_factor, 
            rounding=rounding,
            _timeout=60,
            verbose=False,
        )[0]["capacityMax"] * capacity_factor
    
        stateOfChargeMin = 0
        if  ModelTechnoEconomicData().get_data(component=technology, attribute='stateOfChargeMin'): stateOfChargeMin =ModelTechnoEconomicData().get_data(component=technology, attribute='stateOfChargeMin')
        stateOfChargeMax = 1
        if ModelTechnoEconomicData().get_data(component=technology, attribute='stateOfChargeMax'): stateOfChargeMax = ModelTechnoEconomicData().get_data(component=technology, attribute='stateOfChargeMax')
        
        # update with extracted data
        FINE_args = {
            **FINE_args, 
            **{
                'capacityMax':capacity_max,
                'investPerCapacity':self._multiply(ModelTechnoEconomicData().get_data(component=technology, attribute='investPerCapacity'),cost_factor),
                'opexPerCapacity':ModelTechnoEconomicData().get_data(component=technology, attribute='opexPerCapacity'),
                'stateOfChargeMin':stateOfChargeMin,
                'stateOfChargeMax':stateOfChargeMax,
            }
        }
        # overwrite with function kwargs once more to ensure their usage
        FINE_args = {**FINE_args, **FINE_kwargs}

        self.esM.add(fn.Storage(**FINE_args))
                    

    def AddStorageBrownfield(self, technology):
        print(f"Adding: {technology}")  # TODO
        pass

    # Demand:
    def addDemand(self, technology="electricity_demand", year_demand=2050, factor = 1, path_abs_demands=None, path_ts=None): # TODO: Add fine args as in addPotentialGreenfieldWithTimeSeries
        technology=technology.lower()
        print(f"Adding: {technology}")

        operationRateFix_GW = self.ih.load_demand(
            technology=technology,
            year_demand=year_demand,
            path_abs_demands=path_abs_demands,
            path_ts=path_ts
        )  # TODO: update function

        self.esM.add(
            fn.Sink(
                esM=self.esM,
                name=technology,
                commodity=ModelTechnoEconomicData().get_data(component=technology,attribute="commodity"),
                hasCapacityVariable=False,
                operationRateFix=self._clip_close_to_zero(self._multiply(operationRateFix_GW,factor), self.zero_threshold),
                floorTechnicalLifetime=False,   # floorTechnicalLifetime=False --> TODO: Just for debugging, remove later
            ),
        )
    

    def addArtificalDemand(self, technology="electricity_demand", year_demand=2050, factor = 1, demandVal=0,path_abs_demands=None, path_ts=None):
        """
        this function adds a demand to the ESM. Therefore, external demand files is parsed. The possibility to artifically extend the specific demand in certain location is 
        possible.

        Args:
            technology (str): specifies he demand to dd on. Important for further processing, electricity_demand or hydrogen_gas_demand 
            year_demand (int): Demand year specififes the demand to be loaded. Important to check that exogenous demand is available
            factor (int): Multiples the given demands 
            addingDem (bool, optional): True, if you want to manipulate the demand 
            increasLoc (_type_, optional): Specifiy the Region, where you want to manipualte the demand 
            demandVal (int, optional): Annual additonal demand  Defaults to 0.
            path_abs_demands (_type_, optional): Path to external demands, if None, default path s will be used via Input-Data-handler
            path_ts (_type_, optional): Path to external demands ts, if None, default path s will be used via Input-Data-handler
        """
        
        assert isinstance(year_demand,int), 'Given Year must be an integer'
        assert isinstance(demandVal, (int,float)), 'Given Demand Value must be either integer or float'
        assert isinstance(technology, str), 'Given technology must be a string'
        #assert isinstance(increasLoc, (str, None)), 'Given Location must be a string'





        print(f"Adding: {technology}")

        operationRateFix_GW = self.ih.load_demand(
            technology=technology,
            year_demand=year_demand,
            path_abs_demands=path_abs_demands,
            path_ts=path_ts
        )  # TODO: update function

        #Artificially increasing the demand to create Export and Import potential curves 

        a=operationRateFix_GW
        df=a[year_demand]
        for k in range(0,len(df)):
            df.loc[k]=(demandVal/len(df))
        operationRateFix_GW[2050]=df    
        del(a,df)




        self.esM.add(
            fn.Sink(
                esM=self.esM,
                name=technology,
                commodity=ModelTechnoEconomicData().get_data(component=technology,attribute="commodity"),
                hasCapacityVariable=False,
                operationRateFix=self._clip_close_to_zero(self._multiply(operationRateFix_GW,factor), self.zero_threshold),
            )
        )
    



    # Specific Demand:
    def addDemandSpec(self, technology):
        '''
        addDemandSpec adds demand of one commodity unit per year. 
        '''
        technology=technology.lower()
        print(f"Adding: {technology}")

        # Build DataFrame with 8760 columns for each hour and 1/8760 in each cell
        data = {i: [1] for i in range(8760)}
        operation_rate_df = pd.DataFrame.from_dict(data, columns=ModelLocations().locationIDs, orient="index")

        self.esM.add(
            fn.Sink(
                esM=self.esM,
                name=technology,
                commodity=ModelTechnoEconomicData().get_data(component=technology,attribute="commodity"),
                hasCapacityVariable=False,
                operationRateFix={ip:operation_rate_df for ip in InputDataInfo().investment_period_names},
                floorTechnicalLifetime=False,   # floorTechnicalLifetime=False --> TODO: Just for debugging, remove later
            ),
        )

    # Bloc functions as Convinient Wrapper:
    def addWater(self):
        print(f"Adding: AddWater")
        raise NotImplementedError("Not implemented. Have fun doing so ;)") # TODO
        pass

    def addHydrogenGas(self):
        print(f"Adding: AddHydrogenGas")
        raise NotImplementedError("Not implemented. Have fun doing so ;)") # TODO
        pass
    
    def addLossOfLoad(self, voll_to_BEUR_per_GWh_factor=None, path_VOLL=None, voll_key = None, sectoral_disaggregation=True, voll_factor=1, round=4):
        '''adding a source, which represents the non-covered demand at the costs od VoLL 

        Parameters
        ----------
        voll_to_BEUR_per_GWh_factor : float, optional
            unit conversion factor to model units, by default None
        path_VOLL : str, optional
            path to VoLL file
        voll_key : str, optional
            key for selecting VoLL column from data
        sectoral_disaggregation : boolean
            True: get several VoLL for different sectors
            False: get one average VoLL
        voll_factor : int, optional
            factor on VoLL for variation, by default 1
        round : int, optional
            rounding of VoLL, by default 4

        Raises
        ------
        KeyError
            _description_
        '''
        print('Adding Loss of load. Only use for lull analysis, otherwise energy balance not fulfilled!')

        demand_technology = "electricity_demand"

        dict_VOLL_EUR_per_Wh, shares = self.ih.load_VOLL(
            voll_to_BEUR_per_GWh_factor=voll_to_BEUR_per_GWh_factor,
            voll_key=voll_key,
            path_VOLL=path_VOLL,
            time_steps=self.esM.numberOfTimeSteps,
            sectoral_disaggregation=sectoral_disaggregation,
        )
        
        onesectoral = len(dict_VOLL_EUR_per_Wh) == 1

        for sector in dict_VOLL_EUR_per_Wh:
            
            #adjust all inputs for source
            # VOLL_EUR_per_Wh
            VOLL_EUR_per_Wh = dict_VOLL_EUR_per_Wh[sector]
            VOLL_EUR_per_Wh = VOLL_EUR_per_Wh*voll_factor
            VOLL_EUR_per_Wh = VOLL_EUR_per_Wh.round(round)

            #Save VOLL data to output file
            try:
                path_VOLL_csv = os.path.join(self.model_base_folder, f'VOLL_{sector}.csv')
                VOLL_EUR_per_Wh.to_csv(path_VOLL_csv)
            except:
                pass
            if not demand_technology in self.esM.componentNames:
                raise KeyError(f"{demand_technology} not found in esM object. Please add before!")

            elec_demand_GW = self.esM.getComponentAttribute(demand_technology, "operationRateFix")

            # name
            if onesectoral:
                name = 'Lull'
            else:
                name = f'Lull_{sector}'

            # operationRateMax
            if onesectoral:
                operationRateMax = self._clip_close_to_zero(elec_demand_GW, self.zero_threshold)
            else:
                values = elec_demand_GW * shares[f"share_{sector}"]
                operationRateMax = self._clip_close_to_zero(values, self.zero_threshold)
            
            self.esM.add(
                fn.Source(
                    esM=self.esM, 
                    name=name,
                    commodity='electricity', 
                    hasCapacityVariable=False,
                    commodityCostTimeSeries=VOLL_EUR_per_Wh,
                    #commodityCost=0.006242, #Western European Average VOLL for uniform distribution
                    operationRateMax=operationRateMax, #cannot have more loss of load than load!
                ),
            )
            
            print('Added Loss Of Load.')


    #################
    ## SOLVE MODEL ##
    #################

    def optimizeModel(
        self,
        threads,
        timeSeriesAggregation=True,
        numberOfTypicalPeriods=7,
        #roundOutput=None,
        optimizationSpecs="",
        kwargs_tsam={},
        kwargs_opt={}
    ):
        """
        Parameters:
        -----------
        timeSeriesAggregation : bool; optional
            Determines whether time series aggregation is applied in the optimization

        numberOfTypicalPeriods : int; optional
            Specifies the number of typical periods
            *This should be defined when timeSeriesAggregation is True.

        optimizationSpecs : str; optional
            It includes the specifications of the optimization solver (Gurobi)

        """

        # if roundOutput is not None:
        #     self.roundOutput = roundOutput

        # temporal aggregation
        if timeSeriesAggregation:
            self.esM.aggregateTemporally(
                numberOfTypicalPeriods=numberOfTypicalPeriods,
                **kwargs_tsam
            )

        # initialize the optimization
        print("Optimize", flush=True)
        self.esM.optimize(
            timeSeriesAggregation=timeSeriesAggregation,
            optimizationSpecs=optimizationSpecs,
            logFileName= os.path.join(ModelPaths().base_folder, "esM_gurobi_log.txt"),
            threads=threads,
            **kwargs_opt
        )
        print("Optimization done!", flush=True)

    def saveToNC4(self, pathNC4=None):
        """[summary]

        Parameters
        ----------
        pathNC4 : [type]
            [description]
        """
        # check wether results are fasible or not
        if self.results_feasible():

            if pathNC4 is None:
                savepathNC4 = os.path.join(ModelPaths().base_folder, "esM_optimized.nc4")
            else:
                savepathNC4 = pathNC4

            os.makedirs(os.path.dirname(os.path.abspath(savepathNC4)), exist_ok=True)

            print("Save to nc4:", savepathNC4, flush=True)
            xrIO.writeEnergySystemModelToNetCDF(
                self.esM,
                outputFilePath=savepathNC4,
                overwriteExisting=True,
            )
        else:
            print("Infeasible, not saved.")

    def results_feasible(self):
        
        if self.has_results():
            feasible = not any(
                self.esM.getOptimizationSummary("SourceSinkModel", ip=ip) is None
                for ip in InputDataInfo().investment_period_names
            )
        else:
            #has no results
            feasible = False
        
        return feasible
    
    def has_results(self):
        try:
            #has results, but is not feasible
            for ip in InputDataInfo().investment_period_names:
                _ = self.esM.getOptimizationSummary("SourceSinkModel",ip=ip)
            has_results = True
        except KeyError:
            has_results = False
        
        return has_results
    
    #################
    ## aux functions ##
    #################

    def _get_considerable_FINE_args(self, fineclass): 
            if fineclass == "conversion":
                return ["name",
                        "physicalUnit",
                        "commodityConversionFactors",
                        "hasCapacityVariable",
                        "capacityVariableDomain",
                        "capacityPerPlantUnit",
                        "linkedConversionCapacityID",
                        "hasIsBuiltBinaryVariable",
                        "bigM",
                        "operationRateMin",
                        "operationRateMax",
                        "operationRateFix",
                        "tsaWeight",
                        "locationalEligibility",
                        "capacityMin",
                        "capacityMax",
                        "partLoadMin",
                        "sharedPotentialID",
                        "linkedQuantityID",
                        "capacityFix",
                        "commissioningMin",
                        "commissioningMax",
                        "commissioningFix",
                        "isBuiltFix",
                        "investPerCapacity",
                        "investIfBuilt",
                        "opexPerOperation",
                        "opexPerCapacity",
                        "opexIfBuilt",
                        "QPcostScale",
                        "interestRate",
                        "economicLifetime",
                        "technicalLifetime",
                        "yearlyFullLoadHoursMin",
                        "yearlyFullLoadHoursMax",
                        "stockCommissioning",
                        "floorTechnicalLifetime",
                    ]
            if fineclass == "source" or fineclass == "sink":
                return ["name",
                        "commodity",
                        "hasCapacityVariable",
                        "capacityVariableDomain",
                        "capacityPerPlantUnit",
                        "hasIsBuiltBinaryVariable",
                        "bigM",
                        "operationRateMin",
                        "operationRateMax",
                        "operationRateFix",
                        "tsaWeight",
                        "commodityLimitID",
                        "yearlyLimit",
                        "locationalEligibility",
                        "capacityMin",
                        "capacityMax",
                        "partLoadMin",
                        "sharedPotentialID",
                        "linkedQuantityID",
                        "capacityFix",
                        "commissioningMin",
                        "commissioningMax",
                        "commissioningFix",
                        "isBuiltFix",
                        "investPerCapacity",
                        "investIfBuilt",
                        "opexPerOperation",
                        "commodityCost",
                        "commodityRevenue",
                        "commodityCostTimeSeries",
                        "commodityRevenueTimeSeries",
                        "opexPerCapacity",
                        "opexIfBuilt",
                        "QPcostScale",
                        "interestRate",
                        "economicLifetime",
                        "technicalLifetime",
                        "yearlyFullLoadHoursMin",
                        "yearlyFullLoadHoursMax",
                        "balanceLimitID",
                        "pathwayBalanceLimitID",
                        "stockCommissioning",
                        "floorTechnicalLifetime",
                    ]
            if fineclass =="storage":
                return ["name",
                        "commodity",
                        "chargeRate",
                        "dischargeRate",
                        "chargeEfficiency",
                        "dischargeEfficiency",
                        "selfDischarge",
                        "cyclicLifetime",
                        "stateOfChargeMin",
                        "stateOfChargeMax",
                        "hasCapacityVariable",
                        "capacityVariableDomain",
                        "capacityPerPlantUnit",
                        "hasIsBuiltBinaryVariable",
                        "bigM",
                        "doPreciseTsaModeling",
                        "chargeOpRateMax",
                        "chargeOpRateFix",
                        "chargeTsaWeight",
                        "dischargeOpRateMax",
                        "dischargeOpRateFix",
                        "dischargeTsaWeight",
                        "isPeriodicalStorage",
                        "locationalEligibility",
                        "capacityMin",
                        "capacityMax",
                        "partLoadMin",
                        "sharedPotentialID",
                        "linkedQuantityID",
                        "capacityFix",
                        "commissioningMin",
                        "commissioningMax",
                        "commissioningFix",
                        "isBuiltFix",
                        "investPerCapacity",
                        "investIfBuilt",
                        "opexPerChargeOperation",
                        "opexPerDischargeOperation",
                        "opexPerCapacity",
                        "opexIfBuilt",
                        "interestRate",
                        "economicLifetime",
                        "technicalLifetime",
                        "floorTechnicalLifetime",
                        "socOffsetDown",
                        "socOffsetUp",
                        "stockCommissioning",
                    ]
            if fineclass=="transmission":
                return["name",
                       "commodity",
                        "losses",
                        "distances",
                        "hasCapacityVariable",
                        "capacityVariableDomain",
                        "capacityPerPlantUnit",
                        "hasIsBuiltBinaryVariable",
                        "bigM",
                        "operationRateMax",
                        "operationRateFix",
                        "tsaWeight",
                        "locationalEligibility",
                        "capacityMin",
                        "capacityMax",
                        "partLoadMin",
                        "sharedPotentialID",
                        "linkedQuantityID",
                        "capacityFix",
                        "commissioningMin",
                        "commissioningMax",
                        "commissioningFix",
                        "isBuiltFix",
                        "investPerCapacity",
                        "investIfBuilt",
                        "opexPerOperation",
                        "opexPerCapacity",
                        "opexIfBuilt",
                        "QPcostScale",
                        "interestRate",
                        "economicLifetime",
                        "technicalLifetime",
                        "floorTechnicalLifetime",
                        "balanceLimitID",
                        "pathwayBalanceLimitID",
                        "stockCommissioning",
                    ]


    #################
    ## utility functions ##
    #################
    @staticmethod
    def _getCAPEXfromLCOE(
        LCOE_EUR_per_kWh, fixOPEX_CAPEX_per_a, varOPEX_notdefined, lifetime_a, WACC_1, meanCF, self=None
    ):
        LCOE_EUR_per_kWh_first_single_value = LCOE_EUR_per_kWh[list(ModelLocations().locationIDs)[0]]
        fixOPEX_CAPEX_per_a_first_single_value = fixOPEX_CAPEX_per_a[InputDataInfo().investment_period_names[0]][list(ModelLocations().locationIDs)[0]]
        lifetime_a_first_single_value = lifetime_a[list(ModelLocations().locationIDs)[0]]
        WACC_1_first_single_value = WACC_1[list(ModelLocations().locationIDs)[0]]
        
        assert isinstance(fixOPEX_CAPEX_per_a_first_single_value, (int, float, np.integer, np.floating))
        assert isinstance(lifetime_a_first_single_value, (int, float, np.integer, np.floating))
        assert isinstance(WACC_1_first_single_value, (int, float, np.integer, np.floating))
        assert WACC_1_first_single_value < 1, "WACC in [1], not [%]"
        assert fixOPEX_CAPEX_per_a_first_single_value < 1, "fixOPEX_CAPEX_per_a in [1*Capex per year], not [%*Capex per year]"
        assert varOPEX_notdefined == 0, "Ah not implemented yet, sorry!"
        if not self is None:
            assert ModelTechnoEconomicData().esm_params["esM"]["costUnit"] == "1e9 Euro", "Cost units do not align"
        else:
            warn("Could no check cost units. Output will be in BEUR_per_GW as default")

        Capacity_GW = 1  # this does not impact the results, as it will be reduced later, its just here to have a straightforward way of programming the equations

        annuity = ((1 + WACC_1_first_single_value) ** lifetime_a_first_single_value * WACC_1_first_single_value) / ((1 + WACC_1_first_single_value) ** lifetime_a_first_single_value - 1)
        Power_GWh_per_a = Capacity_GW * meanCF * 8760
        TOTEX_MEUR_per_a = LCOE_EUR_per_kWh_first_single_value * Power_GWh_per_a
        varOPEX_MEUR_per_a = 0  # Power_GWh_per_a * varOPEX_notdefined #to be iplemented
        CAPEX_fixOPEX_MEUR_per_a = TOTEX_MEUR_per_a - varOPEX_MEUR_per_a
        CAPEX_MEUR = CAPEX_fixOPEX_MEUR_per_a / (annuity + fixOPEX_CAPEX_per_a_first_single_value)
        CAPEX_BEUR_per_GW = CAPEX_MEUR / Capacity_GW / 1e3  # Capacity_GW gets reduced from the equations her

        return CAPEX_BEUR_per_GW        # TODO: Check if right unit reaches CAPEX_BEUR_per_GW

    @staticmethod
    def _multiply(data_to_multiply,factor):
        '''
        multiplies different input formats by a factor.

        data_to_multiply: dict, pd.DataFrame, float
        factor: float, int

        Returns: same data type as argument, multiplied  
        '''
        if isinstance(data_to_multiply,dict):
            multiplied_dict = {}
            for key, val in data_to_multiply.items():
                if isinstance(val,pd.Series):
                    multiplied_dict[key] = val.astype(float) * factor
                else:    
                    multiplied_dict[key] = val * factor
            return multiplied_dict
        else:
            return data_to_multiply*factor

    @staticmethod
    def _clip_close_to_zero(array, threshold=0.01):
        '''Replace all values close to zero with zero.
        abs(array) < threshold --> 0 

        Parameters
        ----------
        array : DataFrame, ndarray, Number
            data, dict
        threshold : float, optional
            threshold for cliping, by default 0.01

        Returns
        -------
        DataFrame, ndarray, Number, dict
            data with clipped zeros

        Raises
        ------
        TypeError
            invalid data type for array
        '''
        if not isinstance(threshold, numbers.Number):
            raise TypeError
        if not threshold >= 0:
            raise ValueError 

        if isinstance(array, pd.DataFrame) or isinstance(array, np.ndarray):
            array[np.abs(array)<threshold] = 0
        elif isinstance(array, numbers.Number):
            array = 0 if np.abs(array)<threshold else array
        elif isinstance(array, dict):
            for year, df in array.items():
                if isinstance(df, pd.DataFrame):
                    df[np.abs(df)<threshold] = 0
                    array[year]=df
        else:
            raise TypeError("Invalid data type for array")
        
        return array