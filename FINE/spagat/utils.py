import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, List

def plt_savefig(fig=None, save_name="test", path=None): 
    """Save a figure in .png format. 

    Parameters
    ----------
    fig : Figure, optional (default=None) 
        If None, it get the current figure and saves it 
    save_name : str, optional (default="test") 
        Name of the figure
    path : str, optional (default=None) 
        The path to which to save the figure. 
        If default None, it is saved in the current working directory. 
    """
    if fig is None: fig = plt.gcf()

    if path is not None:
        save_name = os.path.join(path, save_name)

    plt.savefig(f"{save_name}.png", format="png", bbox_inches="tight", dpi=200)


def timer(func):
    """Wrapper around a function to track the time taken by 
    the function. 

    Parameters
    ----------
    func : Function 

    Notes
    -----
    Usage : as a decorator before a function -> @spu.timer
    """
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
    """Creates a new directory, if it doesn't exist yet. 

    Parameters
    ----------
    directory : str
        Format - "<path_to_new_directory>/<directory_name>"  
    """
    if not os.path.exists(directory): os.makedirs(directory)


def create_gdf(df, geometries, crs=3035, file_path=None, files_name = 'sds_regions'):
    """Creates a geodataframe.   

    Parameters
    ----------
    df : pd.dataframe 
        The dataframe which would, among other things, have the region ids/names 
    geometries : List[Geometries]
        List of geometries/shapes 
    crs : int, optional (default=3035)
        The coordinate reference system in which to create the geodataframe 
    file_path : str, optional (default=None) 
        If default None, files (called shapefiles) are not saved 
    files_name : str, optional (default='sds_regions') 
        The name of the saved files 
        yes it is plural! -> many files with same name but 
        different extensions '.cpg', '.dbf', '.prj', '.shp', '.shx'
    
    Returns
    -------
    gdf : gpd.GeoDataFrame
        A geodataframe that is created 
    """
    
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=f'epsg:{crs}') 

    if file_path is not None:  
        gdf['geometry'].to_file(file_path, layer = f'{files_name}') 

    return gdf
