import os
import time

import numpy as np 
import pandas as pd 
import geopandas as gpd
import xarray as xr 
import matplotlib.pyplot as plt
from shapely.geometry import LineString

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
        before = time.perf_counter()  
        rv = func(*args, **kwargs)
        after = time.perf_counter()
        print(
            "elapsed time for {.__name__}: {:.2f} minutes".format(
                func, (after - before) / 60
            )
        )
        return rv

    return f


def create_dir(directory): #TODO: check if this actually is used somewhere
    """Creates a new directory, if it doesn't exist yet. 

    Parameters
    ----------
    directory : str
        Format - "<path_to_new_directory>/<directory_name>"  
    """
    if not os.path.exists(directory): os.makedirs(directory)


def create_gdf(df, geometries, crs=3035, file_path=None, files_name = 'xr_regions'):
    """Creates a geodataframe.   

    Parameters
    ----------
    df : pd.DataFrame 
        The dataframe which would, among other things, have the region ids/names 
    geometries : List[Geometries]
        List of geometries/shapes 
    crs : int, optional (default=3035)
        The coordinate reference system in which to create the geodataframe 
    file_path : str, optional (default=None) 
        If default None, files (called shapefiles) are not saved 
    files_name : str, optional (default='xr_regions') 
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
        gdf.reset_index(drop=True, inplace=True) #NOTE: pandas different versions behave differently here!
        gdf.to_file(file_path, layer = f'{files_name}') 

    return gdf

def add_objects_to_xarray(xarray_dataset,
                        description,
                        dimension_list,
                        object_list):
        """Adds a list of objects to the given `xarray_dataset`.
        
        Parameters
        ----------
        xarray_dataset : xr.Dataset
            the xarray dataset to which the objects need to be added 
        description : str
            description of the objects
        dimension_list : List[str]
            list of all xarray_dataset dimensions the objects live in
        object_list : List[object]
            list of objects that will be added to the xarray_dataset

        Returns
        -------
        xarray_dataset : xr.Dataset 
            `xarray_dataset` with added objects     
        """

        xarray_dataset[description] = (dimension_list, pd.Series(object_list).values)  

        return xarray_dataset

def add_space_coords_to_xarray(xarray_dataset, space_coords):
        """Adds space coordinates to the given `xarray_dataset`.
        
        Parameters
        ----------
        xarray_dataset : xr.Dataset
            the xarray dataset to which the space coordinates need to be added 
        space_coords : List[object]
            coordinates of the space, for example region names

        Returns
        -------
        xarray_dataset : xr.Dataset 
            `xarray_dataset` with added space coordinates 
        """
        xarray_dataset.coords['space'] = space_coords
        xarray_dataset.coords['space_2'] = space_coords

        return xarray_dataset

def add_region_centroids_to_xarray(xarray_dataset):
    """Calculates centroid of each region and 
    adds this to the `xarray_dataset`. 

    Parameters
    ----------
    xarray_dataset : xr.Dataset
        the xarray dataset to which the centroid info needs to be added 
    
    Returns
    -------
    xarray_dataset : xr.Dataset 
        `xarray_dataset` with added centroids  
    """
    gpd_centroids = pd.Series(
        [geom.centroid for geom in xarray_dataset.gpd_geometries.values]
    )
    xarray_dataset['gpd_centroids'] = ('space', gpd_centroids.values)

    return xarray_dataset


def add_centroid_distances_to_xarray(xarray_dataset):
    """Calculates distance between centroids and add this to `xarray_dataset`

    Parameters
    ----------
    xarray_dataset : xr.Dataset
        the xarray dataset to which the centroid distance info needs to be added 
    
    Returns
    -------
    xarray_dataset : xr.Dataset 
        `xarray_dataset` with added centroid distances   
    """
    data_out_dummy = np.zeros(
        (len(xarray_dataset['space']), len(xarray_dataset['space']))
    )

    space_coords = xarray_dataset['space'].values

    xr_data_array_out = xr.DataArray(
        data_out_dummy, coords=[space_coords, space_coords], dims=['space', 'space_2']
    )

    for region_id_1 in xarray_dataset['space']:
        for region_id_2 in xarray_dataset['space']:
            centroid_1 = xarray_dataset.sel(space=region_id_1).gpd_centroids.item(0)
            centroid_2 = xarray_dataset.sel(space=region_id_2).gpd_centroids.item(0)
            xr_data_array_out.loc[dict(space=region_id_1, space_2=region_id_2)] = (
                centroid_1.distance(centroid_2) / 1e3
            )  # distances in km

    xarray_dataset['centroid_distances'] = (
                                            ['space', 'space_2'],     
                                            xr_data_array_out.values,
                                        )
    
    return xarray_dataset


def save_shapefile_from_xarray(xarray_dataset, 
                    save_path, 
                    shp_name = 'aggregated_regions', 
                    crs : int = 3035) :
        """Extracts regions and their geometries from `xarray_dataset` 
        and saves to a shapefile.
        
        Parameters
        ----------
        xarray_dataset : xr.Dataset
            the xarray dataset from which regions and their geometries 
            are to be obtained 
        save_path : str
            path to folder in which to save the shapefile
        shp_name : str, optional (default='aggregated_regions')
            name to be given to the saved files 
        crs
            coordinate reference system (crs) in which to save the shapefiles
        """

        df = xarray_dataset.space.to_dataframe()
        geometries = xarray_dataset.gpd_geometries.values

        create_gdf(df=df, 
                    geometries=geometries, 
                    crs=crs, 
                    file_path=save_path,
                    files_name = shp_name)


def create_grid_shapefile(xarray_dataset,
                        variable_description,
                        component_description,
                        file_path, 
                        files_name="AC_lines"):
    """Creates a geodataframe which indicates whether two regions are connected for the 
    given variable-component pair. 

    Parameters
    ----------
    xarray_dataset : xr.Dataset
        The xarray dataset holding the esM's info 
    variable_description :  str
        Variable in `xarray_dataset` that should be considered 
    component_description :  str
        Component in `xarray_dataset` that should be considered
    file_path : str
        The path to which to save the geodataframe
    files_name : str, optional (default="AC_lines")
        The name of the saved geodataframe
    
    """
    
    xarray_dataset = add_region_centroids_to_xarray(xarray_dataset)

    buses_0 = []
    buses_1 = []
    geoms = []

    eligibility_xr_array = xarray_dataset[variable_description].sel(component=component_description)
    
    for region_id_1 in xarray_dataset["space"].values:
        for region_id_2 in xarray_dataset["space_2"].values:
            if eligibility_xr_array.sel(space=region_id_1, space_2=region_id_2).values: 
                buses_0.append(region_id_1)
                buses_1.append(region_id_2)

                point_1 = xarray_dataset.gpd_centroids.sel(space=region_id_1).item(0)
                point_2 = xarray_dataset.gpd_centroids.sel(space=region_id_2).item(0)
                line = LineString([(point_1.x, point_1.y), (point_2.x, point_2.y)])

                geoms.append(line)

    df = pd.DataFrame(
        {
            "bus0": buses_0,
            "bus1": buses_1,
        }
    )

    create_gdf(df, geoms, crs=3035, file_path=file_path, files_name=files_name)


