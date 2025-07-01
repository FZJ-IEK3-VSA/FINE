# import standard packages
import numpy as np
import os
import pandas as pd
import glob

# import third party packages
import natsort
# import other modules
from modelBuilder.singletons import *


def get_hydrogen_demand_data(path_abs_demands, year_demand, gid0s):
    '''_summary_

    Parameters
    ----------
    path_abs_demands : str
        path to folder with demands:
    year_demand : int
        year
    gid0s : list 
        list of all gid0s

    Returns
    -------
    abs_demands : pd.DataFrame
        index: GID_1, columns: ["GID_0", "hydrogen_demand_gid1_GWh"]. Values in GWh
    rel_demands : pd.DataFrame
        index: time_stamp_UTC, columns: GID_0s
    '''
    abs_demands = { 
        ip: _get_abs_hydorgen_demand_per_gid1(
            path_abs_demands=path_abs_demands, 
            year_demand=year_demand, 
            gid0s=gid0s
            )
        for ip in InputDataInfo().investment_period_names
                   
    }

    #rel demands dummy:
    time_stamps = pd.date_range("2015-01-01 00:00:00+00:00", "2015-12-31 23:00:00+00:00", freq="1h") # i think this deas not matter tbh. but lets be consistent
    shape = (int(len(time_stamps)/int(ModelTechnoEconomicData().esm_params["esM"]["hoursPerTimeStep"])), len(gid0s))
    rel_demands = pd.DataFrame(
        np.ones(shape=shape) * 1/len(time_stamps)*int(ModelTechnoEconomicData().esm_params["esM"]["hoursPerTimeStep"]),
        index=time_stamps[0:shape[0]],
        columns=gid0s,
    )

    return abs_demands, rel_demands

#other implementation
def get_electricity_data(path_abs_demands, path_ts, gid0s):
    '''returns the abs demand and rel demand for electricity

    Parameters
    ----------
    path_abs_demands : str
        path to folder with demands:
    path_ts : str
        path to specific ts file ".csv"
    year_demand : intOSError
        year
    gid0s : list 
        list of all gid0s

    Returns
    -------
    abs_demands : pd.DataFrame
        index: GID_1, columns: ["GID_0", "total_el_demand"]. Values in GWh
    rel_demands : pd.DataFrame
        index: time_stamp_UTC, columns: GID_0s

    Raises
    ------
    OSError
    '''
    if not os.path.isfile(path_ts): raise OSError(f"No valid file found for path_ts: {path_ts}")
    
    #load values
    abs_demands = {
        ip: _get_abs_electricity_demand_per_gid1(
            path_abs_demands=path_abs_demands, 
            investment_period=ip, 
            gid0s=gid0s
            )
        for ip in InputDataInfo().investment_period_names
    }
    rel_demands = _get_rel_electricity_demand_timeseries_per_gid0(path_ts=path_ts)

    return abs_demands, rel_demands


def combine_to_abs_timeseries(abs_demands, rel_demands, abs_column_name):
    '''combine rel timeseries and absolute timeseries to abs timeseires per gid1

    Parameters
    ----------
    abs_demands : abs_demands : pd.DataFrame
        index: GID_1, columns: ["GID_0", "total_el_demand"]. Values in GWh
    rel_demands : pd.DataFrame
        index: time_stamp_UTC, columns: GID_0s
    abs_column_name : str
        name of the demand column

    Returns
    -------
    abs_timeseries_per_gid1 : pd.DataFrame
        abs timeseries. index: range(0, 8760), columns: all GID1s
    '''
    #make absolute demand time series per gid1:
    abs_timeseries_per_gid1 = {}
    for gid1 in list(abs_demands.index.unique()):
        gid0 = gid1[0:3]
        rel_demand = rel_demands[gid0]
        assert np.isclose(sum(rel_demand), 1)
        abs_demand = abs_demands.loc[gid1][abs_column_name]
        #assert len(abs_demand) == 1

        abs_ts = abs_demand*rel_demand
        abs_timeseries_per_gid1[gid1] = abs_ts
    abs_timeseries_per_gid1 = pd.DataFrame(abs_timeseries_per_gid1)

    return abs_timeseries_per_gid1

def scale_timeseries_to_locationsIDs(abs_timeseries_per_gid1): #Funktion nochmal schreiben
    '''scale the abs_timeseries_per_gid1 to locationIDs by area ("coverage_per_default_region")

    Parameters
    ----------
    abs_timeseries_per_gid1 : pd.DataFrame
        abs timeseries for Fine. index: range(0, 8760), columns: locationIDs
    coverage_per_default_region : pd.DataFrame
        coverage_per_default_region from inputdatahandeler
    locationIDs : iterateble
        locationIDs as a list / series

    Returns
    -------
        final : pd.DataFrame
            abs timeseries per locationID. index: range(0, 8760), columns: locationIDs
    '''
    
    abs_timeseries_per_location = pd.DataFrame()

    for loc in ModelLocations().locationIDs:
        for i,(def_reg,share) in enumerate(ModelLocations().get_default_region_overlap_shares_with_location()[loc].items()):
            if i ==0:
                _ts=abs_timeseries_per_gid1[def_reg]*share
            else:
                _ts=_ts+abs_timeseries_per_gid1[def_reg]*share
        abs_timeseries_per_location[loc]=_ts
    return abs_timeseries_per_location[natsort.natsorted(ModelLocations().locationIDs)].reset_index(drop=True)

    # #scale timeseries to custom regions by area
    
    # for i, row in coverage_per_default_region.iterrows():
    #     # rel demand:
    #     if isinstance(row.GID_1, str):
    #         #is default, just get the gid1
    #         abs_timeseries_at_location = abs_timeseries_per_gid1[row.GID_1]
    #     else:
    #         #need to be compiled from several gid1s
    #         abs_timeseries_at_location = ""
    #         for gid1 in row.GID_1:
    #             abs_timeseries_iter_weighted = abs_timeseries_per_gid1[gid1] * row.GID_1[gid1]
    #             #set time series
    #             if isinstance(abs_timeseries_at_location, str):
    #                 abs_timeseries_at_location = abs_timeseries_iter_weighted
    #             else:
    #                 abs_timeseries_at_location += abs_timeseries_iter_weighted
            
    #     #remember values
    #     abs_timeseries_per_location[row.name] = abs_timeseries_at_location
        
    
    # final = pd.DataFrame(abs_timeseries_per_location)

    # final = final[natsort.natsorted(locationIDs)]
    # final.reset_index(drop=True, inplace=True)

def _get_abs_electricity_demand_per_gid1(path_abs_demands, investment_period, gid1s=None, gid0s=None):
    '''load all abs demands an return it as an pd.DataFrame

    Parameters
    ----------
    path_abs_demands : str
        path to folder with demands:
    year_demand : int
        year
    gid1s : list, optional
        list of gid1s to return, by default None
    gid0s : list, optional
        list of gid0s to return, by default None

    Returns
    -------
    pd.DataFrame
        index: GID_0s, columns: ["GID_1", "total_el_demand"]

    Raises
    ------
    OSError
    '''

    file = path_abs_demands.replace("<YEAR>", str(investment_period))
    abs_demands = pd.read_csv(file)[["GID_0", "GID_1", "total_el_demand"]]
    abs_demands.set_index("GID_1", inplace=True)
    #do selection if wanted (for compatibility with other code)
    if gid0s:
        abs_demands=abs_demands[abs_demands.GID_0.isin(gid0s)]
    if gid1s:
        abs_demands=abs_demands.loc[gid1s]
    return abs_demands

def _get_rel_electricity_demand_timeseries_per_gid0(path_ts):
    """ load all time series and return it as an pd.DataFrame """
    load_curves_path = path_ts
    time_series_raw = pd.read_csv(load_curves_path, index_col=[0])
    time_series_normal = time_series_raw / time_series_raw.sum(axis=0)
    return time_series_normal

def _get_abs_hydorgen_demand_per_gid1(path_abs_demands, year_demand, gid1s=None, gid0s=None):
    '''load all abs demands an return it as an pd.DataFrame

    Parameters
    ----------
    path_abs_demands : str
        path to folder with demands:
    year_demand : int
        year
    gid1s : list, optional
        list of gid1s to return, by default None
    gid0s : list, optional
        list of gid0s to return, by default None

    Returns
    -------
    pd.DataFrame
        index: GID_0s, columns: ["GID_1", "hydrogen_demand_gid1_GWh"]

    Raises
    ------
    OSError
    '''
    file = path_abs_demands.replace("<YEAR>", str(year_demand))
    all_files = pd.read_csv(file)[["GID_1", "hydrogen_demand_gid1_GWh"]]
    all_files["GID_0"] = all_files.GID_1.apply(lambda g: g[0:3])
    all_files.set_index("GID_1", inplace=True)
    #do selection if wanted (for compatibility with other code)
    if gid0s:
        all_files=all_files[all_files.GID_0.isin(gid0s)]
    if gid1s:
        all_files=all_files.loc[gid1s]
    return all_files