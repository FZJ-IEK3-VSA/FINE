import xarray as xr
import FINE.IOManagement.dictIO as dictio
import FINE as fn
import numpy as np
import pandas as pd

def create_component_ds(esM):
    """Reads a dictionary of dataframes created from an esM instance into an xarray dataset"""

    esm_dict, comp_dict = dictio.exportToDict(esM)

    locations = list(esm_dict['locations'])
    locations.sort()

    n_timesteps = esm_dict['numberOfTimeSteps']

    time = np.arange(n_timesteps)

    ds = xr.Dataset({"time": time, "location": locations})

    for classname in comp_dict:
        # get class
        class_ = getattr(fn, classname)

        for comp in comp_dict[classname]:            
            comp_dict[classname][comp]

            for key, value in comp_dict[classname][comp].items():
                if isinstance(value, pd.DataFrame):
                    # import pdb; pdb.set_trace()
                    ds[comp] = (('time', 'location'), value.loc[time, locations].values)
                    # TODO: replace this with da = da.read_from_dataframe and then ds[comp] = da

                elif isinstance(value, pd.DataFrame):
                    # import pdb; pdb.set_trace() # TODO: test this
                    ds[comp] = (('location'), value.loc[locations].values)
    # import pdb; pdb.set_trace()

    return ds
