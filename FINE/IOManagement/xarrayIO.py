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

    #STEP 1. Convertion of datatypes 
    #NOTE: data types such as sets, dicts, bool, pandas df/series and Nonetype
    # are not serializable. Therefore, they are converted to lists/strings while saving
    
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
    locations = sorted(esm_dict['locations'])
    xr_ds = utilsIO.addSeriesVariablesToXarray(xr_ds, component_dict, series_iteration_dict, locations)

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
    xarray_dataset = utilsIO.processXarrayAttributes(xarray_dataset)
        
    #STEP 3. Create esm_dict 
    esm_dict = xarray_dataset.attrs

    #STEP 4. Iterate through each component-variable pair, depending on the variable's prefix 
    # restructure the data and add it to component_dict 
    component_dict = utilsIO.PowerDict()

    for component in xarray_dataset.component.values:
        comp_xr = xarray_dataset.sel(component=component)
        
        for variable, comp_var_xr in comp_xr.data_vars.items():

            if not pd.isnull(comp_var_xr.values).all():
                
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
