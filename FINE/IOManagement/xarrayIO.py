import numpy as np
import pandas as pd
import xarray as xr

from FINE.IOManagement import dictIO, utilsIO

def saveNetcdfFile(xarray_dataset, file_name='esM_instance.nc4'):
    """Saves the given xarray dataset as a netcdf file. 
    
    :param xarray_dataset: The dataset holding all data required to set up an esM instance. 
    :type xarray_dataset: xr.Dataset

    :param file_name: output file name (can include full path)
        |br| * the default value is 'esM_instance.nc4'
    :type file_name: string
    """

    #NOTE: data types such as sets, dicts, bool, pandas df/series and Nonetype
    #  are not serializable. Therefore, they are converted to lists/strings while saving

    xarray_dataset.attrs['locations'] = sorted(xarray_dataset.attrs['locations'])
    xarray_dataset.attrs['commodities'] = sorted(xarray_dataset.attrs['commodities'])

    xarray_dataset.attrs['commodityUnitsDict'] = \
        list(f"{k} : {v}" for (k,v) in xarray_dataset.attrs['commodityUnitsDict'].items())

    if isinstance(xarray_dataset.attrs['balanceLimit'], pd.Series):
        for idx, value in xarray_dataset.attrs['balanceLimit'].items():
            xarray_dataset.attrs.update({f"balanceLimit_{idx}" : value})
    
    elif isinstance(xarray_dataset.attrs['balanceLimit'], pd.DataFrame):
        df = xarray_dataset.attrs['balanceLimit']
        df = df.reindex(sorted(df.columns), axis=1)
        for idx, row in xarray_dataset.attrs['balanceLimit'].iterrows():
            xarray_dataset.attrs.update({f"balanceLimit_{idx}" : row.to_numpy()})
    
    del xarray_dataset.attrs['balanceLimit']  

    xarray_dataset.attrs['lowerBound'] = "True" if xarray_dataset.attrs['lowerBound'] == True else "False"
    xarray_dataset.to_netcdf(file_name)



def convertEsmInstanceToXarrayDataset(esM, save=False, file_name='esM_instance.nc4'):  
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
    xr_ds = xr.Dataset()

    #STEP 4. Add all df variables to xr_ds
    xr_ds = utilsIO.addDFVariablesToXarray(xr_ds, component_dict, df_iteration_dict) 

    #STEP 5. Add all series variables to xr_ds
    xr_ds = utilsIO.addSeriesVariablesToXarray(xr_ds, component_dict, series_iteration_dict, esm_dict['locations'])

    #STEP 6. Add all constant value variables to xr_ds
    xr_ds = utilsIO.addConstantsToXarray(xr_ds, component_dict, constants_iteration_dict) 

    #STEP 7. Add the data present in esm_dict as xarray attributes (these are dimensionless data). 
    xr_ds.attrs = esm_dict

    #STEP 8. Save to netCDF file 
    if save:
        saveNetcdfFile(xr_ds, file_name)
        
    return xr_ds



def convertXarrayDatasetToEsmInstance(xarray_dataset):
    """Takes as its input xarray dataset or path to a netcdf file, converts it into an esM instance. 
    
    :param xarray_dataset: The dataset holding all data required to set up an esM instance. 
        Can be a read-in xarray dataset. Alternatively, full path to a netcdf file is also acceptable. 
    :type xarray_dataset: xr.Dataset or string 
 
    :return: esM - EnergySystemModel instance in which the optimized model is held
    """

    #STEP 1. Read in the netcdf file
    if isinstance(xarray_dataset, str): 
        xarray_dataset = xr.open_dataset(xarray_dataset)
    
    elif not isinstance(xarray_dataset, xr.Dataset):
        raise TypeError("xarray_dataset must either be a path to a netcdf file or xarray dataset")
    
    #STEP 2. Convert data types of attributes if necessary 
    #NOTE: data types such as sets, dicts, bool, pandas df/series and Nonetype
    #  are not serializable. Therefore, they are converted to lists/strings while saving.
    # They need to be converted back to right formats 

    ## locations 
    if isinstance(xarray_dataset.attrs['locations'], list):
        xarray_dataset.attrs['locations'] = set(xarray_dataset.attrs['locations'])

    ## commodities 
    if isinstance(xarray_dataset.attrs['commodities'], list):
        xarray_dataset.attrs['commodities'] = set(xarray_dataset.attrs['commodities'])

    ## commoditiesUnitsDict 
    if isinstance(xarray_dataset.attrs['commodityUnitsDict'], list):
        new_commodityUnitsDict = {}

        for item in xarray_dataset.attrs['commodityUnitsDict']:
            [commodity, unit] = item.split(' : ')
            new_commodityUnitsDict.update({commodity : unit})

        xarray_dataset.attrs['commodityUnitsDict'] = new_commodityUnitsDict

    ## balanceLimit
    try:
        xarray_dataset.attrs['balanceLimit']

    except KeyError:
    
        balanceLimit_dict = {}
        keys_to_delete = []

        for key in xarray_dataset.attrs.keys():
            if key[:12] == 'balanceLimit':
                [bl, idx] = key.split('_') 
                balanceLimit_dict.update({idx : xarray_dataset.attrs.get(key)}) 

                keys_to_delete.append(key)

        # cleaning up the many keys belonging to balanceLimit
        for key in keys_to_delete:
            xarray_dataset.attrs.pop(key)

        if balanceLimit_dict == {}:
            xarray_dataset.attrs.update({'balanceLimit' : None})

        elif all([isinstance(value, np.ndarray) for value in list(balanceLimit_dict.values())]):   
            data = np.stack(balanceLimit_dict.values())
            columns = sorted(xarray_dataset.attrs['locations'])
            index = balanceLimit_dict.keys()

            df = pd.DataFrame(data, columns=columns, index=index)
            
            xarray_dataset.attrs.update({'balanceLimit' : df})

        else: 
            series = pd.Series(balanceLimit_dict)
            xarray_dataset.attrs.update({'balanceLimit' : series})
        
        
    ## lowerBound
    if isinstance(xarray_dataset.attrs['lowerBound'], str):
        xarray_dataset.attrs['lowerBound'] = True if xarray_dataset.attrs['lowerBound'] == "True" else False

    #NOTE: ints are converted to numpy ints while saving, but these should strictly be ints. 
    if not isinstance(xarray_dataset.attrs['numberOfTimeSteps'], int):
        xarray_dataset.attrs['numberOfTimeSteps'] = int(xarray_dataset.attrs['numberOfTimeSteps'])
    
    if not isinstance(xarray_dataset.attrs['hoursPerTimeStep'], int):
        xarray_dataset.attrs['hoursPerTimeStep'] = int(xarray_dataset.attrs['hoursPerTimeStep'])
        

    #STEP 3. Create esm_dict 
    esm_dict = xarray_dataset.attrs

    #STEP 4. Iterate through each component-variable pair, depending on the variable's prefix 
    # restructure the data and add it to component_dict 
    component_dict = utilsIO.PowerDict()

    for component in xarray_dataset.component.values:
        comp_xr = xarray_dataset.sel(component=component)
        
        for variable, comp_var_xr in comp_xr.data_vars.items():

            if not pd.isnull(comp_var_xr.values).all():
                
                #STEP 4 (i). Set all regional time series (region, time)
                if variable[:3]== 'ts_':
            
                    df = comp_var_xr.drop("component").to_dataframe().unstack(level=1)
                    if len(df.columns) > 1:
                        df.columns = df.columns.droplevel(0)
                    
                    [class_name, comp_name] = component.split(', ')

                    if '.' in variable:
                        [var_name, nested_var_name] = variable.split('.')
                        component_dict[class_name][comp_name][var_name[3:]][nested_var_name] = df.sort_index()
                        #NOTE: Thanks to utils.PowerDict(), the nested dictionaries need not be created before adding the data. 

                    else:
                        component_dict[class_name][comp_name][variable[3:]] = df.sort_index()

                #STEP 4 (ii). Set all 2d data (region, region)
                elif variable[:3]== '2d_':
                    
                    series = comp_var_xr.drop("component").to_dataframe().stack(level=0)
                    series.index = series.index.droplevel(level=2).map('_'.join)

                    #NOTE: In FINE, a check is made to make sure that locationalEligibility indices matches indices of other 
                    # attributes. Removing 0 values ensures the match. If all are 0s, empty series is fed in, leading to error. 
                    # Therefore, if series is empty, the variable is not added. 
                    series = series[series>0]  

                    if not len(series.index) == 0:

                        [class_name, comp_name] = component.split(', ')

                        if '.' in variable:
                            [var_name, nested_var_name] = variable.split('.')
                            component_dict[class_name][comp_name][var_name[3:]][nested_var_name] = series.sort_index()
                        else:
                            component_dict[class_name][comp_name][variable[3:]] = series.sort_index()

                #STEP 4 (iii). Set all 1d data (region)
                elif variable[:3]== '1d_':

                    series = comp_var_xr.drop("component").to_dataframe().unstack(level=0)
                    series.index = series.index.droplevel(level=0)
                    
                    [class_name, comp_name] = component.split(', ')

                    if '.' in variable:
                        [var_name, nested_var_name] = variable.split('.')
                        component_dict[class_name][comp_name][var_name[3:]][nested_var_name] = series.sort_index()
                    else:
                        component_dict[class_name][comp_name][variable[3:]] = series.sort_index()

                #STEP 4 (iv). Set all 0d data 
                elif variable[:3]== '0d_':

                    var_value = comp_var_xr.values
                    if not var_value == '': #NOTE: when saving to netcdf, the nans in string arrays are converted 
                                            # to empty string (''). These need to be skipped. 
                    
                        [class_name, comp_name] = component.split(', ')

                        if '.' in variable:
                            [var_name, nested_var_name] = variable.split('.')
                            component_dict[class_name][comp_name][var_name[3:]][nested_var_name] = var_value.item()
                        else:
                            component_dict[class_name][comp_name][variable[3:]] = var_value.item()

    #STEP 5. Create esm from esm_dict and component_dict
    esM = dictIO.importFromDict(esm_dict, component_dict)

    return esM 
