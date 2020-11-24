"""Helper functions that serve different modules of SPAGAT.

"""

import geopandas as gpd
import os

import time

import matplotlib.pyplot as plt

from typing import Dict, List


def plt_savefig(save_name : str = None, path : str = None, fig=None, bbox_inches=None):
    if fig is None:
        fig = plt.gcf()

    if path is not None:
        save_name = os.path.join(path, save_name)
        print(f"{save_name}")

    if save_name is None:
        plt.savefig("test.png", format="png")
    else:
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
    
    gdf = gpd.GeoDataFrame(df, geometry=geometries) 
    gdf.crs = {"init": f"epsg:{crs}"}  # TODO: check whether this works correctly
                                       #NOTE: gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=f"EPSG:{crs}") format is more recent

    if file_path is not None:  
        gdf["geometry"].to_file(file_path, layer = files_name) #TODO: check if it is required anywhere to save files 
                                                                # with different extensions or .shp is enough

    return gdf
