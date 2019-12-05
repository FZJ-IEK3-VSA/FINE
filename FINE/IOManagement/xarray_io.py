import xarray as xr
import FINE.IOManagement.dictIO as dictio
import FINE as fn
import numpy as np
import pandas as pd

def create_component_ds(esM):
    """Reads a dictionary of dataframes created from an esM instance into an xarray dataset"""

    esm_dict, comp_dict = dictio.exportToDict(esM)

    locations = list(esm_dict['locations'])

    n_timesteps = esm_dict['numberOfTimeSteps']

    ds = xr.Dataset({"time": np.arange(n_timesteps), "location": locations})

    for classname in comp_dict:
        # get class
        class_ = getattr(fn, classname)

        for comp in comp_dict[classname]:            
            comp_dict[classname][comp]

            for key, value in comp_dict[classname][comp].items():
                if isinstance(value, pd.DataFrame):
                    ds[comp] = (('time', 'location'), value.values)
                elif isinstance(value, pd.DataFrame):
                    ds[comp] = (('location'), value.values)

    return ds
