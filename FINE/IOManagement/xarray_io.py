import xarray as xr
import FINE.IOManagement.dictIO as dictio
import FINE as fn

def dict_of_df_to_xarray(esM):
    """Reads a dictionary of dataframes created from an esM instance into an xarray dataset"""

    esm_dict, comp_dict = dictio.exportToDict(esM)

    for classname in comp_dict:
        # get class
        class_ = getattr(fn, classname)

        for comp in comp_dict[classname]:            
            import pdb; pdb.set_trace()
            comp_dict[classname][comp]

