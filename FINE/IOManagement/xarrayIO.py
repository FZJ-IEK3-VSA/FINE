import time
from pathlib import Path
from typing import Dict

import pandas as pd
import xarray as xr
from netCDF4 import Dataset

import FINE.utils as utils
from FINE.IOManagement import dictIO, utilsIO

def saveNetcdfFile(xarray_dataset, file_name='esM_instance.nc4'):
    """Saves the given xarray dataset as a netcdf file. 
    
    :param xarray_dataset: The dataset holding all data required to set up an esM instance. 
    :type xarray_dataset: xr.Dataset

    :param file_name: output file name (can include full path)
        |br| * the default value is 'esM_instance.nc4'
    :type file_name: string
    """

    #STEP 1. Convertion of datatypes 
    #NOTE: data types such as sets, dicts, bool, pandas df/series and Nonetype
    # are not serializable. Therefore, they are converted to lists/strings while saving 
    # (Applied only to xarray_dataset.attrs where esM init info is stored)
    
    _xarray_dataset = xarray_dataset.copy() #Copying to avoid errors due to change of size during iteration

    for attr_name, attr_value in _xarray_dataset.attrs.items():
        
        #if the attribute is set, convert into sorted list 
        if isinstance(attr_value, set):
            xarray_dataset.attrs[attr_name] = sorted(xarray_dataset.attrs[attr_name])

        #if the attribute is dict, convert into a "flattened" list 
        elif isinstance(attr_value, dict):
            xarray_dataset.attrs[attr_name] = \
                list(f"{k} : {v}" for (k,v) in xarray_dataset.attrs[attr_name].items())

        #if the attribute is pandas series, add a new attribute corresponding 
        # to each row.  
        elif isinstance(attr_value, pd.Series):
            for idx, value in attr_value.items():
                xarray_dataset.attrs.update({f"{attr_name}.{idx}" : value})

            # Delete the original attribute  
            del xarray_dataset.attrs[attr_name] 

        #if the attribute is pandas df, add a new attribute corresponding 
        # to each row by converting the column into a numpy array.   
        elif isinstance(attr_value, pd.DataFrame):
            _df = attr_value
            _df = _df.reindex(sorted(_df.columns), axis=1)
            for idx, row in _df.iterrows():
                xarray_dataset.attrs.update({f"{attr_name}.{idx}" : row.to_numpy()})

            # Delete the original attribute  
            del xarray_dataset.attrs[attr_name]

        #if the attribute is bool, add a corresponding string  
        elif isinstance(attr_value, bool):
            xarray_dataset.attrs[attr_name] = "True" if attr_value == True else "False"
        
        #if the attribute is None, add a corresponding string  
        elif attr_value == None:
            xarray_dataset.attrs[attr_name] = "None"

    #STEP 2. Saving to a netcdf file     
    xarray_dataset.to_netcdf(file_name)



def convertEsmInstanceToXarrayDataset(esM, save=False, file_name='esM_instance.nc4', groups=False):  
    """Takes esM instance and converts it into an xarray dataset. Optionally, the 
    dataset can be saved as a netcdf file.
    
    :param esM: EnergySystemModel instance in which the optimized model is held
    :type esM: EnergySystemModel instance

    :param save: indicates if the created xarray dataset should be saved
        |br| * the default value is False
    :type save: boolean

    :param file_name: output file name (can include full path)
        |br| * the default value is 'esM_instance.nc4'
    :type file_name: string
 
    :return: xr_ds - esM instance data in xarray dataset format 
    """
    
    #STEP 1. Get the esm and component dicts 
    esm_dict, component_dict = dictIO.exportToDict(esM)

    #STEP 2. Get the iteration dicts 
    df_iteration_dict, series_iteration_dict, constants_iteration_dict = \
        utilsIO.generateIterationDicts(component_dict)
    
    #STEP 3. Initiate xarray dataset 
    xr_dss= dict.fromkeys(component_dict.keys())
    for classname in component_dict:
            xr_dss[classname] = {
                component: xr.Dataset()
                for component in component_dict[classname]
            }
    xr_ds = xr.Dataset()

    #STEP 4. Add all df variables to xr_ds
    xr_ds, xr_dss = utilsIO.addDFVariablesToXarray(xr_ds, xr_dss, component_dict, df_iteration_dict) 

    #STEP 5. Add all series variables to xr_ds
    locations = sorted(esm_dict['locations'])
    xr_ds, xr_dss = utilsIO.addSeriesVariablesToXarray(xr_ds, xr_dss, component_dict, series_iteration_dict, locations)

    #STEP 6. Add all constant value variables to xr_ds
    xr_ds, xr_dss = utilsIO.addConstantsToXarray(xr_ds, xr_dss, component_dict, constants_iteration_dict) 

    #STEP 7. Add the data present in esm_dict as xarray attributes 
    # (These attributes contain esM init info). 
    xr_ds.attrs = esm_dict

    #STEP 8. Save to netCDF file 
    if save:
        saveNetcdfFile(xr_ds, file_name)

        if groups:

            # Create netCDF file, remove existant
            grouped_file_path = f"grouped_{file_name}"
            if Path(grouped_file_path).is_file():
                Path(grouped_file_path).unlink()
            rootgrp = Dataset(grouped_file_path, "w", format="NETCDF4")
            rootgrp.close()

            for model, comps  in xr_dss.items():
                for component in comps.keys():
                    xr_dss[model][component].to_netcdf(
                        path=f"grouped_{file_name}",
                        # Datasets per component will be reflectes as groups in the NetCDF file.
                        group=f"Parameters/{model}/{component}",
                        # Use mode='a' to append datasets to existing file. Variables will be overwritten.
                        mode="a",
                        # Use zlib variable compression to reduce filesize with little performance loss
                        # for our use-case. Complevel 9 for best compression.
                        encoding={
                            var: {"zlib": True, "complevel": 9}
                            for var in list(xr_dss[model][component].data_vars)
                        },
                    )
    xr_dss = {"Parameters": xr_dss}
        
    return xr_ds, xr_dss


def convertXarrayDatasetToEsmInstance(xarray_dataset):
    """Takes as its input xarray dataset or path to a netcdf file, converts it into an esM instance. 
    
    :param xarray_dataset: The dataset holding all data required to set up an esM instance. 
        Can be a read-in xarray dataset. Alternatively, full path to a netcdf file is also acceptable. 
    :type xarray_dataset: xr.Dataset or string 
 
    :return: esM - EnergySystemModel instance in which the optimization model is held
    """

    #STEP 1. Read in the netcdf file
    if isinstance(xarray_dataset, str): 
        xarray_dataset = xr.open_dataset(xarray_dataset)
    
    elif not isinstance(xarray_dataset, xr.Dataset):
        raise TypeError("xarray_dataset must either be a path to a netcdf file or xarray dataset")
    
    #STEP 2. Convert data types of attributes if necessary 
    xarray_dataset = utilsIO.processXarrayAttributes(xarray_dataset)
        
    #STEP 3. Create esm_dict 
    esm_dict = xarray_dataset.attrs

    #STEP 4. Iterate through each component-variable pair, depending on the variable's prefix 
    # restructure the data and add it to component_dict 
    component_dict = utilsIO.PowerDict()

    for component in xarray_dataset.component.values:
        comp_xr = xarray_dataset.sel(component=component)
        
        for variable, comp_var_xr in comp_xr.data_vars.items():

            if not pd.isnull(comp_var_xr.values).all(): # Skip if all are NAs 
                
                #STEP 4 (i). Set regional time series (region, time)
                if variable[:3]== 'ts_':
                    component_dict = \
                        utilsIO.addTimeSeriesVariableToDict(component_dict, comp_var_xr, component, variable)
            
                #STEP 4 (ii). Set 2d data (region, region)
                elif variable[:3]== '2d_':
                    component_dict = \
                        utilsIO.add2dVariableToDict(component_dict, comp_var_xr, component, variable)
                    
                #STEP 4 (iii). Set 1d data (region)
                elif variable[:3]== '1d_':
                    component_dict = \
                        utilsIO.add1dVariableToDict(component_dict, comp_var_xr, component, variable)

                #STEP 4 (iv). Set 0d data 
                elif variable[:3]== '0d_':
                    component_dict = \
                        utilsIO.add0dVariableToDict(component_dict, comp_var_xr, component, variable)


    #STEP 5. Create esm from esm_dict and component_dict
    esM = dictIO.importFromDict(esm_dict, component_dict)

    return esM 




def writeOptimizationOutputToNetCDF(
    esM,
    outputFileName="esM_results.nc4",
    overwrite_existing=False,
    optSumOutputLevel=2,
    optValOutputLevel=1,
) -> Dict[str, Dict[str, xr.Dataset]]:
    """
    Write optimization output to netCDF file.

    :param esM: EnergySystemModel instance in which the optimized model is hold
    :type esM: EnergySystemModel instance

    :param outputFileName: Name of the netCDF output file 
        |br| * the default value is 'scenarioOutput'
    :type outputFileName: string

    :param overwrite_existing: Overwrite existing netCDF file 
        |br| * the default value is 'scenarioOutput'
    :type outputFileName: string

    :param optSumOutputLevel: Output level of the optimization summary (see EnergySystemModel). Either an integer
        (0,1,2) which holds for all model classes or a dictionary with model class names as keys and an integer
        (0,1,2) for each key (e.g. {'StorageModel':1,'SourceSinkModel':1,...}
        |br| * the default value is 2
    :type optSumOutputLevel: int (0,1,2) or dict

    :param optValOutputLevel: Output level of the optimal values. Either an integer (0,1) which holds for all
        model classes or a dictionary with model class names as keys and an integer (0,1) for each key
        (e.g. {'StorageModel':1,'SourceSinkModel':1,...}
        - 0: all values are kept.
        - 1: Lines containing only zeroes are dropped.
        |br| * the default value is 1
    :type optValOutputLevel: int (0,1) or dict

    :return: Nested dictionary containing an xr.Dataset with all result values for each component.
    :rtype: Dict[str, Dict[str, xr.Dataset]]
    """

    file_path = outputFileName

    # Remove output file if already existant.
    if Path(file_path).is_file():
        if overwrite_existing:
            Path(file_path).unlink()
            rootgrp = Dataset(file_path, "w", format="NETCDF4")
            rootgrp.close()

    utils.output("\nWriting output to netCDF... ", esM.verbose, 0)
    _t = time.time()

    # Create the netCDF file and the xr.Dataset dict for all components
    xr_dss = dict.fromkeys(esM.componentModelingDict.keys())
    for model_dict in esM.componentModelingDict.keys():
        xr_dss[model_dict] = {
            key: xr.Dataset()
            for key in esM.componentModelingDict[model_dict].componentsDict.keys()
        }

    # Write output from esM.getOptimizationSummary to datasets
    for name in esM.componentModelingDict.keys():
        utils.output("\tProcessing " + name + " ...", esM.verbose, 0)
        oL = optSumOutputLevel
        oL_ = oL[name] if type(oL) == dict else oL
        optSum = esM.getOptimizationSummary(name, outputLevel=oL_)
        if esM.componentModelingDict[name].dimension == "1dim":
            for component in optSum.index.get_level_values(0).unique():
                for variable in (
                    optSum.loc[component].index.get_level_values(0).unique()
                ):
                    df = optSum.loc[(component, variable)]
                    df = df.iloc[-1]
                    df.name = variable
                    df.index.rename("space", inplace=True)
                    df = pd.to_numeric(df)
                    xr_da = df.to_xarray()
                    xr_dss[name][component] = xr.merge([xr_dss[name][component], xr_da])
        elif esM.componentModelingDict[name].dimension == "2dim":
            for component in optSum.index.get_level_values(0).unique():
                for variable in (
                    optSum.loc[component].index.get_level_values(0).unique()
                ):
                    df = optSum.loc[(component, variable)]
                    if len(df.index.get_level_values(0).unique()) > 1:
                        idx = df.index.get_level_values(0).unique()[-1]
                        df = df.xs(idx, level=0)
                    else:
                        df.index = df.index.droplevel(0)
                    # df = df.iloc[-1]
                    df = df.stack()
                    # df.name = (name, component, variable)
                    df.name = variable
                    df.index.rename(["space", "space_2"], inplace=True)
                    df = pd.to_numeric(df)
                    xr_da = df.to_xarray()
                    xr_dss[name][component] = xr.merge([xr_dss[name][component], xr_da])

        # Write output from esM.esM.componentModelingDict[name].getOptimalValues() to datasets
        data = esM.componentModelingDict[name].getOptimalValues()
        oL = optValOutputLevel
        oL_ = oL[name] if type(oL) == dict else oL
        dataTD1dim, indexTD1dim, dataTD2dim, indexTD2dim = [], [], [], []
        dataTI, indexTI = [], []
        for key, d in data.items():
            if d["values"] is None:
                continue
            if d["timeDependent"]:
                if d["dimension"] == "1dim":
                    dataTD1dim.append(d["values"]), indexTD1dim.append(key)
                elif d["dimension"] == "2dim":
                    dataTD2dim.append(d["values"]), indexTD2dim.append(key)
            else:
                dataTI.append(d["values"]), indexTI.append(key)
        # One dimensional time dependent data
        if dataTD1dim:
            names = ["Variable", "Component", "Location"]
            dfTD1dim = pd.concat(dataTD1dim, keys=indexTD1dim, names=names)
            dfTD1dim = dfTD1dim.loc[
                ((dfTD1dim != 0) & (~dfTD1dim.isnull())).any(axis=1)
            ]
            for variable in dfTD1dim.index.get_level_values(0).unique():
                for component in dfTD1dim.index.get_level_values(1).unique():
                    df = dfTD1dim.loc[(variable, component)].T.stack()
                    # df.name = (name, component, variable)
                    df.name = variable
                    df.index.rename(["time", "space"], inplace=True)
                    xr_da = df.to_xarray()
                    xr_dss[name][component] = xr.merge([xr_dss[name][component], xr_da])
        # Two dimensional time dependent data
        if dataTD2dim:
            names = ["Variable", "Component", "LocationIn", "LocationOut"]
            dfTD2dim = pd.concat(dataTD2dim, keys=indexTD2dim, names=names)
            dfTD2dim = dfTD2dim.loc[
                ((dfTD2dim != 0) & (~dfTD2dim.isnull())).any(axis=1)
            ]
            for variable in dfTD2dim.index.get_level_values(0).unique():
                for component in dfTD2dim.index.get_level_values(1).unique():
                    df = dfTD2dim.loc[(variable, component)].stack()
                    # df.name = (name, component, variable)
                    df.name = variable
                    df.index.rename(["space", "space_2", "time"], inplace=True)
                    df.index = df.index.reorder_levels([2, 0, 1])
                    xr_da = df.to_xarray()
                    xr_dss[name][component] = xr.merge([xr_dss[name][component], xr_da])
        # Time independent data
        if dataTI:
            # One dimensional
            if esM.componentModelingDict[name].dimension == "1dim":
                names = ["Variable type", "Component"]
                dfTI = pd.concat(dataTI, keys=indexTI, names=names)
                dfTI = dfTI.loc[((dfTI != 0) & (~dfTI.isnull())).any(axis=1)]
                for variable in dfTI.index.get_level_values(0).unique():
                    for component in dfTI.index.get_level_values(1).unique():
                        df = dfTI.loc[(variable, component)].T
                        # df.name = (name, component, variable)
                        df.name = variable
                        df.index.rename("space", inplace=True)
                        xr_da = df.to_xarray()
                        xr_dss[name][component] = xr.merge(
                            [xr_dss[name][component], xr_da]
                        )
            # Two dimensional
            elif esM.componentModelingDict[name].dimension == "2dim":
                names = ["Variable type", "Component", "Location"]
                dfTI = pd.concat(dataTI, keys=indexTI, names=names)
                dfTI = dfTI.loc[((dfTI != 0) & (~dfTI.isnull())).any(axis=1)]
                for variable in dfTI.index.get_level_values(0).unique():
                    for component in dfTI.index.get_level_values(1).unique():
                        df = dfTI.loc[(variable, component)].T.stack()
                        # df.name = (name, component, variable)
                        df.name = variable
                        df.index.rename(["space", "space_2"], inplace=True)
                        xr_da = df.to_xarray()
                        xr_dss[name][component] = xr.merge(
                            [xr_dss[name][component], xr_da]
                        )

    utils.output("\tSaving file...", esM.verbose, 0)

    # Write to netCDF
    for name in esM.componentModelingDict.keys():
        # Cast space coordinats to str. If this is not done then dtype will be object.
        for component in esM.componentModelingDict[name].componentsDict.keys():
            xr_dss[name][component].coords["space"] = (
                xr_dss[name][component].coords["space"].astype(str)
            )
            if esM.componentModelingDict[name].dimension == "2dim":
                xr_dss[name][component].coords["space_2"] = (
                    xr_dss[name][component].coords["space_2"].astype(str)
                )
            xr_dss[name][component].to_netcdf(
                path=file_path,
                # Datasets per component will be reflectes as groups in the NetCDF file.
                group=f"Results/{name}/{component}",
                # Use mode='a' to append datasets to existing file. Variables will be overwritten.
                mode="a",
                # Use zlib variable compression to reduce filesize with little performance loss
                # for our use-case. Complevel 9 for best compression.
                encoding={
                    var: {"zlib": True, "complevel": 9}
                    for var in list(xr_dss[name][component].data_vars)
                },
            )
    utils.output("Done. (%.4f" % (time.time() - _t) + " sec)", esM.verbose, 0)

    xr_dss_results = {"Results": xr_dss}

    return xr_dss_results


def readOptimizationOutputFromNetCDF(
    inputFileName="esM_results.nc4",
) -> Dict[str, Dict[str, xr.Dataset]]:
    """Read optimization results from grouped netCDF file to dictionary of xr.Datasets.

    :param inputFileName: Path to input netCDF file, defaults to "esM_results.nc4"
    :type inputFileName: str, optional

    :return: Nested dictionary containing an xr.Dataset with all result values for each component.
    :rtype: Dict[str, Dict[str, xr.Dataset]]
    """

    rootgrp = Dataset(inputFileName, "r", format="NETCDF4")
    xr_dss = {parameter_result_key: 
                 {model_key: 
                    {comp_key: 
                        xr.open_dataset(inputFileName, group=f"{parameter_result_key}/{model_key}/{comp_key}")
                    for comp_key in rootgrp[parameter_result_key][model_key].groups}
                for model_key in rootgrp[parameter_result_key].groups} 
            for parameter_result_key in rootgrp.groups}

    return xr_dss
