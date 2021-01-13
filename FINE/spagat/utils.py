"""Helper functions that serve different modules of SPAGAT.

"""
import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, List

def plt_savefig(fig=None, save_name : str = "test", path : str = None,  bbox_inches=None): 

    if fig is None: fig = plt.gcf()

    if path is not None:
        save_name = os.path.join(path, save_name)
        print(f" Your figure is here -> {save_name}")

    plt.savefig(f"{save_name}.png", format="png", bbox_inches="tight", dpi=200)


def timer(func):
    def f(*args, **kwargs):
        before = time.perf_counter()  # maybe exchange with time.process_time()
        rv = func(*args, **kwargs)
        after = time.perf_counter()
        print(
            "elapsed time for {.__name__}: {:.2f} minutes".format(
                func, (after - before) / 60
            )
        )
        return rv

    return f


def create_dir(directory):
    """Creates a new directory, if it doesn't exist yet."""
    if not os.path.exists(directory):

        os.makedirs(directory)


def create_gdf(df, geometries, crs=3035, file_path=None, files_name = 'sds_regions'):
    
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=f'epsg:{crs}') 

    if file_path is not None:  
        gdf['geometry'].to_file(file_path, layer = f'{files_name}') 

    return gdf
