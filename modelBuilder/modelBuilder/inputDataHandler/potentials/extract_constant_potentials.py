import sqlite3
import numpy as np
import pandas as pd
import geokit as gk
from natsort import natsorted
import glob
import os
import numbers
from osgeo import ogr
from modelBuilder.data import default_path_information
from modelBuilder.singletons import ModelTechnoEconomicData,InputDataInfo,ModelLocations,ModelPaths

def get_stockCommissioning_dict(
        technology,
        db_stock_path,
        db_name,
        model_unit,
        db_unit,
        fuel,
    ):
            '''
            Compiles the stockCommissioning dict from a database containing the stock locations -lat,lon - and a capacity. 

            technology: str
                technology name, containing the fuel
            db_stock_path: str
                file path to a database with min. longitutde, latitude and a capacity of plants
            model_unit: str
                unit of the model (e.g. GW, MW, ...)
            db_unit: str
                overhands the unit of the database
            db_name: str
                depending on the name of the database the preprocessing of the database is implemented

            '''

            print("Compiling stockCommissioning dict",flush=True)

            #############################################################
            ## Start: Helping functions of get_stockCommissioning_dict ##
            #############################################################
            def _replace_datein(row,fuel_mapper,base_year,economicLifetime,investment_period_interval):
                '''
                Replaces commissioning name data in databases that are so old that the plants would no longer 
                be considered by fine due to their lifetime. Based on the assumption that the database may not 
                have taken general overhauls into account, the commissioning data is adjusted so that the plants 
                are still running 1 to 2 model periods after the base year. 
                '''
                if row['Fueltype'] == fuel_mapper['nuclear']:
                    return max(row['DateIn'], base_year - economicLifetime + 2*investment_period_interval)
                else:
                    return max(row['DateIn'], base_year - economicLifetime + investment_period_interval)

            # Function to aggregate years in reverse periods
            def _aggregate_by_period(df, start_year, range_size):
                # Get the years in descending order
                years = sorted([col for col in df.columns if isinstance(col, int)], reverse=True)
                periods = []                
                # Create periods, going backwards
                for start in range(start_year-1, start_year-ModelTechnoEconomicData().get_data(component=technology,attribute='economicLifetime')[0]+range_size, -range_size):
                    end = start - range_size + 1
                    periods.append((start, end))
                
                # Initialize dictionary to store aggregated results
                aggregated_dict = {}

                # Iterate through the periods and aggregate values
                for period in periods:
                    start, end = period
                    period_columns = [year for year in range(end, start + 1) if year in years]
                    
                    # Sum the values over the selected columns for each row
                    aggregated_data = df[period_columns].sum(axis=1)
                    
                    # Store the aggregated results as pandas.Series
                    _aggregated_dict = pd.Series(aggregated_data, index=df.index)
                    # Ensure all locationIDs are present, if missing fill with 0
                    aggregated_dict[end] = _aggregated_dict.reindex(ModelLocations().locationIDs, fill_value=0)
                return aggregated_dict
            
            def _set_conversion_factor(model_unit,db_unit):
                    if 'kw' in db_unit.lower():
                        if model_unit.lower()=='gw':
                            conversion_factor=1e-6
                        elif model_unit.lower()=='mw':
                            conversion_factor=1e-3    
                        elif model_unit.lower()=='kW':
                            conversion_factor=1
                        else:
                            raise ValueError(f"No conversion factor implemented for model unit: {model_unit}")
                    elif 'mw' in db_unit.lower():
                        if model_unit.lower()=='gw':
                            conversion_factor=1e-3
                        elif model_unit.lower()=='mw':
                            conversion_factor=1   
                        elif model_unit.lower()=='kW':
                            conversion_factor=1e3
                        else:
                            raise ValueError(f"No conversion factor implemented for model unit: {model_unit}")
                    elif 'gw' in db_unit.lower():
                        if model_unit.lower()=='gw':
                            conversion_factor=1
                        elif model_unit.lower()=='mw':
                            conversion_factor=1e3   
                        elif model_unit.lower()=='kW':
                            conversion_factor=1e6
                        else:
                            raise ValueError(f"No conversion factor implemented for model unit: {model_unit}")                    
                    else:
                        raise ValueError(f"No conversion factor implemented for unit of db: {db_unit}")
                    
                    return conversion_factor

            ## END: Helping functions of get_stockCommissioning_dict ##

            # load and preprocess placements df
            placements_df = pd.read_csv(db_stock_path)
            # preprocessing of data base and setting column and fuel name dicts
            if "matching-tool" in db_name:
                print(f"Preprocess given database data base {db_stock_path}",flush=True)
                # column name mapper
                column_name_mapper = {
                    'Capacity':'capacity',
                    'Technology':'technology',
                    'Fueltype':'fuel',
                    'DateIn':'dateIn',
                    'lon':'lon',
                    'lat':'lat',
                }
                # fuel name mapper
                fuel_mapper = {
                    'gas':'Natural Gas',
                    'coal':'Hard Coal',
                    'nuclear':'Nuclear',
                    'lignite':'Lignite',
                }

                placements_df['DateIn'] = placements_df['DateIn'].round(0).astype(int)
                for i in ['DateIn','DateOut']:
                    placements_df[i] = placements_df[i].replace([np.inf, -np.inf], np.nan)  # replace Inf-values
                    if i == 'DateOut':
                        placements_df[i] = placements_df[i].fillna(2100)  # replace nan values
                    if i == 'DateIn':
                        placements_df[i] = placements_df[i].fillna(InputDataInfo().base_year-10)  # replace nan values
                    placements_df[i] = placements_df[i].round(0).astype(int)    

                # filter depending on cost_year
                placements_df = placements_df[placements_df['DateOut']>InputDataInfo().base_year]
                placements_df = placements_df[placements_df['DateIn']<=InputDataInfo().base_year]

                # replace to old commissioning years
                base_year   = InputDataInfo().base_year
                economicLifetime    = ModelTechnoEconomicData().get_data(component=technology,attribute='economicLifetime')[0]
                investment_period_interval       = InputDataInfo().investment_period_interval
                placements_df['DateIn'] = placements_df.apply(lambda row: _replace_datein(row, fuel_mapper,base_year,economicLifetime,investment_period_interval), axis=1) #apply(_replace_datein(row,fuel_mapper), axis=1)
                placements_df = placements_df[list(column_name_mapper.keys())]

                # rename columns of placements_df
                for c in placements_df.columns:    
                    placements_df.rename(columns={c:column_name_mapper[c]},inplace=True)
                                
                # Replace NaN values in the 'capacity' column with 0
                placements_df['capacity'] = placements_df['capacity'].fillna(0)
                # Convert all values in the 'capacity' column to float
                placements_df['capacity'] = placements_df['capacity'].astype(float)
                placements_df['capacity'] = placements_df['capacity'] * _set_conversion_factor(model_unit=model_unit,db_unit=db_unit)
            else:
                raise ValueError("So far, no database preprocessing has been implemented other than for the power-plant-matching-tool. Feel free to add your own database preprocessing")
            
            # add geom to df
            placements_df['geom'] = placements_df[['lon', 'lat']].apply(lambda x: gk.geom.point(x[0], x[1], srs=4326), axis=1)
            # store intermediate plant database shape in your base folder
            placements_shp_path = fr"{ModelPaths().base_folder}/intermediate_plant_locations_shape.shp"
            # get path wo ending, for deleting the intermediate files later with delete_intermediates()
            placements_shp_path_wo_ending = fr"{ModelPaths().base_folder}/intermediate_plant_locations_shape*"
            

            # store shape file and load as vector
            gk.vector.createVector(geoms=placements_df,output=placements_shp_path)
            placements_vec = gk.vector.loadVector(placements_shp_path)

            print("load capacities per locationID",flush=True)

            # function to delete intermediate placement shape
            def delete_intermediates(filepath_wo_ending):
                for filename in glob.glob(filepath_wo_ending):
                   os.remove(filename)

            # load placements per locationID
            placements_dfs = []
            for loc_geom,loc_id in zip(ModelLocations().location_df.geom, ModelLocations().location_df.locationID):
                _df = gk.vector.extractFeatures(placements_vec,geom=loc_geom)
                _df["locationID"] = loc_id
                _grouped_df = _df.groupby(['fuel', 'dateIn','locationID'])['capacity'].sum().reset_index()
                _pivot_df = _grouped_df.pivot_table(index='locationID', columns=['fuel', 'dateIn'], values='capacity', aggfunc='sum')
                if fuel_mapper[fuel] in _pivot_df.columns.get_level_values('fuel'):
                    _fuel_df = _pivot_df[fuel_mapper[fuel]]
                    _fuel_df.insert(0,"locationID",loc_id)
                    _fuel_df = _fuel_df.reset_index(drop=True)
                    #_fuel_df = _fuel_df.set_index('locationID')
                    placements_dfs.append(_fuel_df)
            if not placements_dfs:
                aggregated_dict = {}
                delete_intermediates(placements_shp_path_wo_ending)
                return aggregated_dict
            else:
                placements_filtered_df = pd.concat(placements_dfs).reset_index(drop=True)
                placements_filtered_df.set_index('locationID', inplace=True)
                placements_filtered_df = placements_filtered_df.fillna(0)

                aggregated_dict = _aggregate_by_period(placements_filtered_df,start_year=InputDataInfo().base_year, range_size=InputDataInfo().investment_period_interval)
                delete_intermediates(placements_shp_path_wo_ending)
                return aggregated_dict

def extract_potentials_sql(
    path,
    LCOE_name,
    capacity_name,
    region_name_col,
    capacity_conversion_factor,
    LCOE_to_EUR_per_kWh_factor,
    N_cluster,
    location_shape,
    defaultregions_per_location_dict,
    rounding=4,
    _timeout = 60,
    verbose=False):
    '''loading capacity potentials from shape file

    Parameters
    ----------
    pathSQL : str
        _description_
    path to shaoe file. use "*" for variability
    LCOE_name : str
        name of LCOE var
    capacity_name : str
        name of capacity var
    region_name_col : str
        name of column with region identifier
    capacity_conversion_factor : float
        factor of source capacity data to model unit
    LCOE_to_EUR_per_kWh_factor : float
        factor of source LCOE data to EUR/kWh
    N_cluster : _type_
        _description_
    location_shape : gk.DataFrame
        shape from inputDataHandler.location_shape
    defaultregions_per_location_dict : dict
        from inputDataHandler.defaultregions_per_location_dict
    rounding : int, optional
        rounding of digits, by default 4
    _timeout : int, optional
        timneout for the sql connection, by default 60
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    e
        _description_
    e
        _description_
    '''
    
    assert os.path.isfile(path)

    potential_data_dict = {}

    regions = list(location_shape.locationID)

    for region in regions:
        if verbose:
            print(region)

        potential_data_dict[region] = {}
        #fetch the placements
        dflt_type = location_shape[location_shape.locationID == region].iloc[0].dflt_type
        default_regions = defaultregions_per_location_dict[region]
        if dflt_type in ["default", "agg"]:
            if verbose:
                print('Loading default/agg region')
            #do default region stuff
            placements_list = []
            for default_region in default_regions:
                conn = sqlite3.connect(path, timeout=_timeout)
                try:
                    placements_default_region = pd.read_sql_query(f"SELECT {LCOE_name}, {capacity_name} FROM placements WHERE {region_name_col} = '{default_region}'", conn)
                    conn.close()
                except Exception as e:
                    conn.close()
                    raise e
                placements_list.append(placements_default_region)
            
            placements_region = pd.concat(placements_list, axis=0).reset_index(drop=True)
            del placements_list
        else:
            #do geospatial stuff
            if verbose:
                print('Loading geospatially')
            
            #print("This is a little devy here and will fail.")
            #1) Get bounding box
            region_shape = location_shape[location_shape.locationID == region].geom.iloc[0]
            bounds = region_shape.GetEnvelope() 
            xMin, xMax, yMin, yMax = bounds
            #2) Get placemnts in bounding box
            conn = sqlite3.connect(path, timeout=_timeout)
            try:
                placements_bounding_box = pd.read_sql_query(
                    f"""SELECT {LCOE_name}, {capacity_name}, lat, lon
                    FROM placements
                    WHERE lon >= {str(xMin)} and lon <= {str(xMax)} and lat >= {str(yMin)} and lat <= {str(yMax)}
                    """,
                    conn
                )
                conn.close()
            except Exception as e:
                conn.close()
                raise e
            
            #3) Select placements within geom
            srs = gk.srs.loadSRS(4326)
            #make geoms
            if len(placements_bounding_box) > 0:
                placements_bounding_box['geom'] = placements_bounding_box[['lon', 'lat']].apply(lambda x: gk.geom.point(x[0], x[1], srs=srs), axis=1)
            else:
                placements_bounding_box['geom'] = []
            #filter geoms
            geoms = placements_bounding_box['geom']
            isinRegion=[]
            for geom in geoms:
                isinRegion.append(geom.Within(region_shape))
            placements_bounding_box['isinRegion'] = isinRegion

            placements_region = placements_bounding_box[placements_bounding_box.isinRegion]
        
        #do the aggregation:
        if verbose:
            print('Aggregating')
        potential_data_dict = _add_placement_cluster_potential_data_dict(
            potential_data_dict=potential_data_dict,
            placements_region=placements_region,
            region=region,
            N_cluster=N_cluster,
            LCOE_name=LCOE_name,
            capacity_name=capacity_name,
            capacity_conversion_factor=capacity_conversion_factor,
            LCOE_to_EUR_per_kWh_factor=LCOE_to_EUR_per_kWh_factor,
            rounding=rounding,
        )

    #sort data
    if verbose:
        print('\nRestruct data for output.')
    potential_data_dict_final = _adapt_dict_to_FINE(
        potential_data_dict,
        N_cluster,
        regions
    )

    return potential_data_dict_final


def extract_potentials_shp(
    path,
    LCOE_name,
    capacity_name,
    region_name_col,
    capacity_conversion_factor,
    LCOE_to_EUR_per_kWh_factor,
    N_cluster,
    location_shape,
    defaultregions_per_location_dict,
    rounding=4,
    verbose=False):
    '''load potentials from shape file

    Parameters
    ----------
    path_shape : str
        path to shaoe file. use "*" for variability
    LCOE_name : str
        name of LCOE var
    capacity_name : str
        name of capacity var
    capacity_conversion_factor : float
        factor of source capacity data to model unit
    LCOE_to_EUR_per_kWh_factor : float
        factor of source LCOE data to EUR/kWh
    N_cluster : _type_
        _description_
    location_shape : gk.DataFrame
        shape from inputDataHandler.location_shape
    defaultregions_per_location_dict : dict
        from inputDataHandler.defaultregions_per_location_dict
    rounding : int, optional
        rounding of digits, by default 4
    verbose : bool, optional
        _description_, by default False
    '''
    potential_data_dict = {}
    regions = list(location_shape.locationID)

    for region in regions:
        if verbose:
            print(region)
    
        potential_data_dict[region] = {}
        #fetch the placements
        dflt_type = location_shape[location_shape.locationID == region].iloc[0].dflt_type
        default_regions = defaultregions_per_location_dict[region]
        if dflt_type in ["default", "agg"]:
            if verbose:
                print('Loading default/agg region')
            #do default region stuff
            placements_list = []
            for default_region in default_regions:
                placements_default_region = _load_placements_per_default_region_from_shape(path, default_region, region_name_col=region_name_col)
                placements_list.append(placements_default_region)

            placements_region = pd.concat(placements_list, axis=0).reset_index(drop=True)
            del placements_list
        
        else:
        #do geospatial stuff
            #1) Get affected regions
            df_list = [] 
            for def_reg in defaultregions_per_location_dict[region]:
                df_list.append(_load_placements_per_default_region_from_shape(path_shape=path, default_region=def_reg, region_name_col=region_name_col))
            placements_per_affected_default_regions = pd.concat(df_list)

            #2) extract placements within region
            if len(placements_per_affected_default_regions) == 0:
                placements_region = pd.DataFrame()
            else:
                region_shape = location_shape[location_shape.locationID == region].geom.iloc[0]
                vec = gk.vector.createVector(placements_per_affected_default_regions)
                placements_region = gk.vector.extractFeatures(vec, geom = region_shape) #actual filtering
                del vec, region_shape
        
        #all placemnts loaded and filtered, now aggregating:
        if verbose:
            print('Aggregating')
        potential_data_dict = _add_placement_cluster_potential_data_dict(
            potential_data_dict=potential_data_dict,
            placements_region=placements_region,
            region=region,
            N_cluster=N_cluster,
            LCOE_name=LCOE_name,
            capacity_name=capacity_name,
            capacity_conversion_factor=capacity_conversion_factor,
            LCOE_to_EUR_per_kWh_factor=LCOE_to_EUR_per_kWh_factor,
            rounding=rounding,
        )

    #sort data
    if verbose:
        print('\nRestruct data for output.')
    potential_data_dict_final = _adapt_dict_to_FINE(
        potential_data_dict,
        N_cluster,
        regions
    )

    return potential_data_dict_final


def extract_potentials_csv(
    path,
    LCOE_name,
    capacity_name,
    region_name_col,
    capacity_conversion_factor,
    LCOE_to_EUR_per_kWh_factor,
    N_cluster,
    location_shape,
    defaultregions_per_location_dict,
    rounding=4,
    verbose=False):
    '''load potentials from shape file

    Parameters
    ----------
    path_csv : str
        path to csv file. use "*" for variability
    LCOE_name : str
        name of LCOE var
    capacity_name : str
        name of capacity var
    region_name_col : str
        name of column with region name
    capacity_conversion_factor : float
        factor of source capacity data to model unit
    LCOE_to_EUR_per_kWh_factor : float
        factor of source LCOE data to EUR/kWh
    N_cluster : _type_
        _description_
    location_shape : gk.DataFrame
        shape from inputDataHandler.location_shape
    defaultregions_per_location_dict : dict
        from inputDataHandler.defaultregions_per_location_dict
    rounding : int, optional
        rounding of digits, by default 4
    verbose : bool, optional
        _description_, by default False
    '''
    potential_data_dict = {}
    regions = list(location_shape.locationID)

    single_file =  os.path.isfile(path)

    if single_file:
        placements_all = _load_placements_per_default_region_from_excel(
            path
        )


    for region in regions:
        if verbose:
            print(region)
    
        potential_data_dict[region] = {}
        #fetch the placements
        dflt_type = location_shape[location_shape.locationID == region].iloc[0].dflt_type
        default_regions = defaultregions_per_location_dict[region]
        if dflt_type in ["default", "agg"]:
            if verbose:
                print('Loading default/agg region')
            #do default region stuff
            placements_list = []
            for default_region in default_regions:
                if single_file:
                    placements_default_region = placements_all[placements_all[region_name_col]==region]
                else:
                    placements_default_region = _load_placements_per_default_region_from_excel(
                        path,
                        default_region
                    )
                placements_list.append(placements_default_region)
            placements_region = pd.concat(placements_list, axis=0).reset_index(drop=True)
            del placements_list
        
        else:
        #do geospatial stuff
            #1) Get affected regions
            df_list = [] 
            for def_reg in defaultregions_per_location_dict[region]:
                if single_file:
                    placements_def_region = placements_all[placements_all[region_name_col]==def_reg]
                else:
                    placements_def_region = _load_placements_per_default_region_from_excel(
                        path,
                        def_reg
                    )
                df_list.append(placements_def_region)
            placements_per_affected_default_regions = pd.concat(df_list)

            #2) extract placements within region
            if len(placements_per_affected_default_regions) == 0:
                placements_region = placements_per_affected_default_regions
            else:
                region_shape = location_shape[location_shape.locationID == region].geom.iloc[0]
                vec = gk.vector.createVector(placements_per_affected_default_regions)
                placements_region = gk.vector.extractFeatures(vec, geom = region_shape) #actual filtering
                del vec, region_shape
        
        #all placemnts loaded and filtered, now aggregating:
        if verbose:
            print('Aggregating')
        potential_data_dict = _add_placement_cluster_potential_data_dict(
            potential_data_dict=potential_data_dict,
            placements_region=placements_region,
            region=region,
            N_cluster=N_cluster,
            LCOE_name=LCOE_name,
            capacity_name=capacity_name,
            capacity_conversion_factor=capacity_conversion_factor,
            LCOE_to_EUR_per_kWh_factor=LCOE_to_EUR_per_kWh_factor,
            rounding=rounding,
        )

    #sort data
    if verbose:
        print('\nRestruct data for output.')
    potential_data_dict_final = _adapt_dict_to_FINE(
        potential_data_dict,
        N_cluster,
        regions
    )

    return potential_data_dict_final

def _load_placements_per_default_region_from_excel(path_excel, default_region=None):

    extension = os.path.splitext(os.path.basename(path_excel))[1]
    if extension == ".csv":
        loader = pd.read_csv
    elif extension == ".xlsx":
        loader = pd.read_excel
    else:
        raise OSError(f"No class found for loading {extension}, pls only load .csv and .xlsx: path_excel")

    #single file found! load once
    if os.path.isfile(path_excel):
        placements = loader(path_excel, index_col=[0])
        if default_region:
            if len(default_region) == 3:
                #GID0
                placements_region = placements[placements.GID_0 == default_region]
            else:
                placements_region = placements[placements.GID_0 == default_region]
        else:
            placements_region = placements

    #load with GID= identifier
    elif "<GID0>" in path_excel:
        path = path_excel.replace("<GID0>", default_region)
        assert os.path.isfile(path)
        placements_region = loader(path, index_col=[0])

    #load with '*' identifier and glob
    elif "*" in path_excel:
        paths = sorted(list(glob.glob(path_excel)))
        assert len(paths) > 0, f"No valid files found for loading: {path_excel}"
        container_placements = []
        for path in paths:
            container_placements.append(
                loader(path, index_col=[0])
            )
        placements = pd.concat(container_placements, axis=0)

        if default_region:
            if len(default_region) == 3:
                #GID0
                placements_region = placements[placements.GID_0 == default_region]
            else:
                placements_region = placements[placements.GID_0 == default_region]
        else:
            placements_region = placements

    #nothing found
    else:
        raise ValueError(f"No valid files found for loading: {path_excel}")
    
    #make polys
    if "geom" in placements_region.columns:
        placements_region.geom = placements_region.geom.apply(lambda wkt: ogr.CreateGeometryFromWkt(wkt))
        for i in range(len(placements_region)):
            placements_region.geom.iloc[i].AssignSpatialReference(gk.srs.loadSRS(4326))
        pass
    elif "lat" in placements_region.columns and "lon" in placements_region.columns:
        placements_region.geom = placements_region["lon", "lat"].apply(lambda lonlat: gk.geom.point(lonlat.lon, lonlat.lat, srs=4326), axis=1)

    return placements_region


def _load_placements_per_default_region_from_shape(path_shape, default_region, region_name_col):

    if os.path.isfile(path_shape):
        # one file found
        placements_region = gk.vector.extractFeatures(
            source=path_shape,
            where=f"{region_name_col} = '{default_region}'",
        )
        #not a file
    elif f"<{region_name_col}>" in path_shape:
        path_shape = path_shape.replace(f"<{region_name_col}>", str(default_region))
        placements_region = gk.vector.extractFeatures(
                source=path_shape,
            )
    elif "*" in path_shape:
        #check with glob
        paths = sorted(list(glob.glob(path_shape)))
        assert len(paths) > 0, f"No valid files found for loading: {path_shape}"
        container_placements = []
        #loop files:
        for file in paths:
            if len(default_region) == 3:
                #GID0
                placements_iter = gk.vector.extractFeatures(
                    source=file,
                    where=f"GID_0 = '{default_region}'",
                )
            else:
                #GID1
                placements_iter = gk.vector.extractFeatures(
                    source=file,
                    where=f"GID_1 = '{default_region}'",
                )
            container_placements.append(placements_iter)
        #concat:
        placements_region = pd.concat(container_placements, axis=0)
    else:
        raise ValueError(f"No valid files found for loading: {path_shape}")
    
    return placements_region


def _add_placement_cluster_potential_data_dict(
    potential_data_dict,
    placements_region,
    region,
    N_cluster,
    LCOE_name,
    capacity_name,
    rounding,
    capacity_conversion_factor,
    LCOE_to_EUR_per_kWh_factor,
    ):
    '''aggregate placements_region to potential_data_dict

    Parameters
    ----------
    potential_data_dict : _type_
        _description_
    placements_region : dict[dict][list]
        potential data from loaders above
    region : str
        resion to add
    N_cluster : int
        number of clusters
    LCOE_name : str
        name of LCOE var
    capacity_name : str
        name of capacity var
    rounding : int
        number of digits to round
    capacity_conversion_factor : float
        factor of source capacity data to model unit
    LCOE_to_EUR_per_kWh_factor : float
        factor of source LCOE data to EUR/kWh

    Returns
    -------
    _type_
        _description_
    '''

    assert isinstance(capacity_conversion_factor, numbers.Number)
    assert isinstance(LCOE_to_EUR_per_kWh_factor, numbers.Number)
    
    _dummy_LCOE = 1000000 #dummy values, should not matter as there is no capacity

    if len(placements_region) == 0:
        #no placements found. aggregation is easy
        for cluster in range(N_cluster):
            potential_data_dict[region][cluster] = {}
            potential_data_dict[region][cluster]['capacityMax'] = 0
            potential_data_dict[region][cluster]['LCOE_EUR_per_kWh'] =  _dummy_LCOE
    else: 
        placements_region.sort_values(by=LCOE_name, ascending=True, inplace=True)
        
        #get the max and min LCOE to span the results
        max_LCOE = placements_region[LCOE_name].max()
        min_LCOE = placements_region[LCOE_name].min()
        
        if min_LCOE == max_LCOE:
            #only one placement found. manipulate maxLCOE so that the normal aggregation will still work:
            max_LCOE = min_LCOE + 1

        #get the LCOE limits for each LCOE Cluster
        delta = (max_LCOE-min_LCOE)/N_cluster
        clusters_lower_LCOE = np.arange(min_LCOE, max_LCOE+delta/2, (max_LCOE-min_LCOE)/N_cluster)
        clusters_lower_LCOE[0] = clusters_lower_LCOE[0]-1
        clusters_lower_LCOE[-1] = clusters_lower_LCOE[-1]+1

        #filter and agg for each LCOE cluster
        for cluster in range(N_cluster):
            
            #filter by LCOE
            min_LCOE_cluster = clusters_lower_LCOE[cluster]
            max_LCOE_cluster = clusters_lower_LCOE[cluster+1]
            placements_cluster = placements_region[(placements_region[LCOE_name] > min_LCOE_cluster) & (placements_region[LCOE_name] <= max_LCOE_cluster)]

            #aggregate
            potential_MW = placements_cluster[capacity_name].sum()
            potential_GW = potential_MW * capacity_conversion_factor
            if potential_GW == 0:
                LCOE = _dummy_LCOE
            else:
                LCOE = placements_cluster[LCOE_name].mean() * LCOE_to_EUR_per_kWh_factor

            #write data
            potential_data_dict[region][cluster] = {}
            potential_data_dict[region][cluster]['capacityMax'] = round(potential_GW, rounding)
            potential_data_dict[region][cluster]['LCOE_EUR_per_kWh'] = round(LCOE, rounding)

    return potential_data_dict

def _adapt_dict_to_FINE(potential_data_dict, N_cluster, regions):
    '''rearrange potential_data_dict to convention: [cluster][vars]

    Parameters
    ----------
    potential_data_dict : dict[dict][list]
        potential data from loaders above
    N_cluster : int
        number of clusters
    regions : list
        list of regions

    Returns
    -------
    dict[dict][series]
        data for FINE:
        dict[cluster]['capacityMax','LCOE_EUR_per_kWh'] --> pd.Series(index=regions)
    '''
    potential_data_dict_final = {}
    #write for each cluster
    for cluster in range(N_cluster):

        potential_data_dict_final[cluster] = {}
        #write for each variable
        for var in ['capacityMax','LCOE_EUR_per_kWh']:
            #collect all vars into dict
            d = {}
            for region in natsorted(regions):
                d[region] = potential_data_dict[region][cluster][var]
            #merge al data into a FINE ready series
            potential_data_dict_final[cluster][var] = pd.Series(d)

    return potential_data_dict_final

if __name__ == "__main__":

    #test!
    shapeFilePath = default_path_information["general_data"]["default_regions_shp"]
    
    
    shape = gk.vector.extractFeatures(shapeFilePath, where="GID_1 in ('BHR.3_1', 'BHR.4_1', 'BHR.5_1')") #TODO
    #shape = gk.vector.extractFeatures(shapeFilePath, where="GID_1 in ('DEU.13_1')") #TODO

    # #select AUSNZL = 23
    # shape = shape.iloc[:10]
    shape.rename(columns={'GID_1': 'locationID'}, inplace=True)
    
    Ncluster = 3

    # #test non default regions:
    # data = extract_potentials_sql(
    #     pathSQL="/storage_cluster/internal/data/d-franzmann/03_Diss/03_Geothermal/results/03_Merged/POL-medV2_Doublette_adapted_Wellcosts/allPlacements_POL-medV2_Doublette_adapted_Wellcosts.sqlite", 
    #     LCOE_name = "LCOE_GR",
    #     capacity_name = "Pnet_GR_MW",
    #     capacity_conversion_factor=1E-3,
    #     LCOE_to_EUR_per_kWh_factor = 1,
    #     N_cluster=Ncluster,
    #     location_shape=shape,
    #     rounding=4,
    #     verbose=True,
    # )


    