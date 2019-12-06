import xarray as xr
import FINE.IOManagement.dictIO as dictio
from FINE import utils
import FINE as fn
import numpy as np
import pandas as pd

def create_component_ds(esM):
    """Reads a dictionary of dataframes created from an esM instance into an xarray dataset"""

    esm_dict, component_dict = dictio.exportToDict(esM)

    locations = list(esm_dict['locations'])
    locations.sort() # TODO: should not be necessary any more

    ds = xr.Dataset()

    for classname in component_dict:

        for component in component_dict[classname]:            
            component_dict[classname][component]

            for description, data in component_dict[classname][component].items():

                # description_tuple = (classname, component, description)
                description_tuple = f"{classname}, {component}, {description}"

                if isinstance(data, pd.DataFrame):
                    multi_index_dataframe = data.stack()
                    multi_index_dataframe.index.set_names("location", level=2, inplace=True)

                    ds[description_tuple] = multi_index_dataframe.to_xarray()

                elif isinstance(data, pd.Series):

                    if classname == 'Transmission':
                        # TODO: which one of transmission's components are 2d and which 1d or dimensionless

                        df = utils.transform1dSeriesto2dDataFrame(data, locations)
                        multi_index_dataframe = df.stack()
                        multi_index_dataframe.index.set_names(["location", "location_2"], inplace=True)

                        ds[description_tuple] = multi_index_dataframe.to_xarray()

                    else:
                        ds[description_tuple] = data.rename_axis("location").to_xarray()

    return ds
