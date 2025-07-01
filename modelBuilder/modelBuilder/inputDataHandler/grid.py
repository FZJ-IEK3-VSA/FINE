import geokit as gk
import pandas as pd
import numpy as np
import os
import networkx as nx
from modelBuilder.data import default_path_information
from modelBuilder.data import default_grids_information
from modelBuilder import utils

from modelBuilder.singletons import UnitHandling, ModelLocations, ModelPaths

class spatialDefinition:
    '''[reads in shape file of region]
    '''
    def __init__(self, shape, region_name_col, path_datafolder=None) -> None:
        '''Class to handel all shape file relevant operations for the fine region:

        Parameters
        ----------
        gid0 : [str]
            [description]
        shape : [str], optional
            [description], by default None
        
        Functions
        ---------
        getFineRegions :
            loads all regions for FINE for a given shape file and adds the export location to the shape file
        extractEligibilityMatrix_km :
            returns distance matrix between regions
        buildPipelineShape :
            saves the pipeline shape to a given location
        '''
        #check inputs
        # assert isinstance(gid0, str)

        # if path_country_shape == None:
        #     raise ValueError()
        # assert isinstance(path_country_shape, str)
        # assert os.path.isfile(path_country_shape)

        # if path_datafolder == None:
        #     raise ValueError()
        # assert isinstance(path_datafolder, str)
        # if not os.path.isdir(path_datafolder):
        #     os.makedirs(path_datafolder)
        
        # self.gid0 = gid0
        # self.path_country_shape = path_country_shape

        #folders
        if path_datafolder is not None:
            self.path_distancematrix = os.path.join(ModelPaths().base_folder, 'auxillary_files', f'distance_matrix.csv')
            self.path_eligibilitymatrix = os.path.join(ModelPaths().base_folder, 'auxillary_files', f'eligibility_matrix.csv')
            self.path_share_onshore_matrix = os.path.join(ModelPaths().base_folder, 'auxillary_files', f'share_onshore_matrix.csv')
            self.path_regions = os.path.join(ModelPaths().base_folder, 'spatial_data', f'regions.shp')
            self.path_transmission = os.path.join(ModelPaths().base_folder, 'spatial_data', f'transmissions.shp')
            os.makedirs(os.path.dirname(self.path_distancematrix), exist_ok=True)
            os.makedirs(os.path.dirname(self.path_regions), exist_ok=True)
            self.readorssave=True
        else:
            self.readorssave = False

        #objects
        assert isinstance(shape, pd.DataFrame), 'Can only handle geokit shapes.'
        shape = shape.reset_index()
        self.shape = shape
        self.region_name_col = region_name_col

    #not used any more
    # def saveFineRegions(self):
    #     '''loads all regions for FINE for a given shape file path

    #     Returns
    #     -------
    #     [pd.DataFrame (As shape type)]
    #         shape file of the region
    #     '''
    #     #save regions
    #     if self.readorssave:
    #         os.makedirs(os.path.dirname(self.path_regions), exist_ok=True)
    #         print(f'Saving region shape to:')
    #         print(self.path_regions)
    #         gk.vector.createVector(
    #             geoms = self.shape,
    #             output = self.path_regions,
    #             srs = gk.srs.loadSRS(4326),
    #             overwrite = True
    #         )

    #     return self.shape

    def _getFineRegionNames(self):
        '''get the region shape names as a list

        Returns
        -------
        [list]
            [fine region names]
        '''
                
        self.regions_fine_names = self.shape[self.region_name_col].to_list()

        return self.regions_fine_names

    def _bufferFineRegions(self, additionalBuffer=0.01):
        '''returns a shape file od the fine region with an additional buffer to each region 

        Returns
        -------
        [type]
            [description]
        '''       

        # new shape with Buffered regions
        FineRegionsBuffered = self.shape.copy()

        # dont know why, but approach downwards is terribly slow.. so use for loop instead
        # FineRegionsBuffered['geom'] = FineRegionsBuffered['geom'].apply(lambda g: g.Buffer(_additionalBuffer))

        container = []
        for i in range(FineRegionsBuffered.shape[0]):
            # iteration counter: print(f'{i} of {FineRegionsBuffered.shape[0]}')
            simpl = 0.01
            container.append(
                FineRegionsBuffered.geom.iloc[i]
                .Simplify(simpl)
                .Buffer(additionalBuffer, 15)
            )
        FineRegionsBuffered.geom = container

        return FineRegionsBuffered

    def _getNeighboursAndDistances(self, regions_fine_buffered):
        
        equal_disance_srs = gk.srs.loadSRS(4088)

        n_regions = self.shape.shape[0]
        distance_matrix = np.zeros(shape=(n_regions, n_regions))
        eligibility_matrix = np.zeros(shape=(n_regions, n_regions))

        for i in range(n_regions):
            for j in range(n_regions):
                # check if touches
                touches = regions_fine_buffered.geom.loc[i].Intersects(\
                    regions_fine_buffered.geom.loc[j])
                if i == j:
                    touches = False

                # if i != n_regions-1 and j != n_regions-1: 
                #     touches = self.regions_fine.geom.loc[i].Touches(self.regions_fine.geom.loc[j])
                # else:
                #     touches = self.regions_fine.geom.loc[i].Overlaps(self.regions_fine.geom.loc[j])
                
                #write eligibility
                if touches:
                    eligibility_matrix[i,j] = 1
                else:
                    eligibility_matrix[i,j] = 0

                #calculate distance
                point1 = getCentroid180Long(self.shape.geom.loc[i])
                point1.AssignSpatialReference(gk.srs.EPSG4326)
                
                
                point2 = getCentroid180Long(self.shape.geom.loc[j])
                point2.AssignSpatialReference(gk.srs.EPSG4326)

                #180° longitude
                lon1 = point1.GetX()
                lon2 = point2.GetX()
                if lon1 * lon2 < 0 and abs(lon1) > 90 and abs(lon2) > 90:
                    #shift them by 180 degs
                    lon1_new = lon1 + 180
                    if lon1_new > 180: lon1_new = lon1_new - 360
                    lon2_new = lon2 + 180
                    if lon2_new > 180: lon2_new = lon2_new - 360 

                    point1 = gk.geom.point(lon1_new, point1.GetY(), srs=4326)
                    point2 = gk.geom.point(lon2_new, point2.GetY(), srs=4326)


                point1 = gk.geom.transform(point1, equal_disance_srs)
                point2 = gk.geom.transform(point2, equal_disance_srs)
                
                #write distance
                distance_matrix[i,j] = point1.Distance(point2)

        assert (distance_matrix == 0).sum() == n_regions

        return (eligibility_matrix, distance_matrix)
    
    def _connectSubgraphs(self, eligibility_matrix, distance_matrix):
        '''connect non connected independant systems

        Parameters
        ----------
        eligibility_matrix : pd.DataFrame
            _description_
        distance_matrix : pd.DataFrame
            _description_

        Retruns
        -------
        self.eligibility_matrix : pd.DataFrame
            eligibility_matrix with updated eligibilites

        Raises
        ------
        RuntimeError
            bad constructed graphs
        '''
        
        assert isinstance(eligibility_matrix, pd.DataFrame)
        assert isinstance(distance_matrix, pd.DataFrame)

        #make sure indexes are the same, otherwise numpy represenation is wrong and results are not  one-to-one
        assert (eligibility_matrix.columns == distance_matrix.columns).all()
        assert (eligibility_matrix.index == distance_matrix.index).all()

        eligibility_matrix_np = eligibility_matrix.values.copy()
        distance_matrix_np = distance_matrix.values.copy()


        iter = 0
        max_iters = 100

        while iter < max_iters:
            iter += 1

            #find non connected subgraphs:
            G = nx.from_numpy_array(eligibility_matrix_np)
            subgraphs = [c for c in nx.connected_components(G)]
            num_subgraphs = len(subgraphs)

            to_connect = []
            if num_subgraphs == 1:
                # Heureka, the objective goal is fulfilled
                # exit function here! (or RuntimeError)
                eligibility_matrix_pd = pd.DataFrame(
                    eligibility_matrix_np,
                    index = eligibility_matrix.index,
                    columns = eligibility_matrix.columns,
                )
                return eligibility_matrix_pd
            
            for current_subgraph in range(num_subgraphs):
                
                #get distances from regions from currens subgraph to all regions other subgraphs
                regions_current_subgraph = list(subgraphs[current_subgraph]) #regions from currens subgraph
                regions_other_subgraph = list(set(range(len(eligibility_matrix_np))) - subgraphs[current_subgraph]) #regions from other subgraphs
                distances = distance_matrix_np[regions_current_subgraph,:][:, regions_other_subgraph]

                #get regions to connect from distances.min
                dist_min = distances.min()
                region_min = np.where(distances == dist_min)
                current_region_argmin = regions_current_subgraph[region_min[0][0]]
                comp_region_argmin = regions_other_subgraph[region_min[1][0]]
                #store them to to_connect list
                to_connect.append((current_region_argmin, comp_region_argmin))
            
            #connect subgraphs fom to_connect
            for connect in to_connect:
                reg1 = connect[0]
                reg2 = connect[1]

                eligibility_matrix_np[reg1, reg2] = 1
                eligibility_matrix_np[reg2, reg1] = 1

        #while run out of iterations without returning final condition num_subgraphs == 1
        raise RuntimeError(f"Grid connection iteration failed after {max_iters} iterations. Not all regions could be connected.")

    def extractEligibilityMatrix_km(self, detour_factor):
        '''determines the distance matrix, and eligibility matrix,
        allowing only neighbouring connections

        Returns
        -------
        np.nparray (2D)
            symetric matrix with distance locations
        '''

        print(f'Extract Transmission matrix for.')
        
        n_regions = self.shape.shape[0]
        
        #try unsing previos calculation file
        if self.readorssave:
            if False:#os.path.isfile(self.path_distancematrix) and os.path.isfile(self.path_eligibilitymatrix) and os.path.isfile(self.path_distancematrix):
                distance_matrix = pd.read_csv(self.path_distancematrix, index_col=[0], header=[0])
                eligibility_matrix = pd.read_csv(self.path_eligibilitymatrix, index_col=[0], header=[0])
                if distance_matrix.shape == (n_regions, n_regions) and eligibility_matrix.shape == (n_regions, n_regions):
                    print('Usinig previous distance matrix!')
                    print(f'Path: {self.path_distancematrix}')
                    self.distance_matrix_km = distance_matrix
                    self.eligibility_matrix = eligibility_matrix
                    return self
        
        # No file found. Calcualte new file
    
        # generate bufferd shape
        regions_fine_buffered = self._bufferFineRegions()

        #check touching neighbors
        eligibility_matrix, distance_matrix = self._getNeighboursAndDistances(regions_fine_buffered=regions_fine_buffered)
        
        #m to km and detour faactor
        distance_matrix_km = distance_matrix / 1E3 * detour_factor

        del distance_matrix

        self.distance_matrix_km = pd.DataFrame(
            distance_matrix_km,
            index=self._getFineRegionNames(),
            columns=self._getFineRegionNames(),
        )

        self.eligibility_matrix = pd.DataFrame(
            eligibility_matrix,
            index=self._getFineRegionNames(),
            columns=self._getFineRegionNames(),
        )
        del eligibility_matrix
        
        #check for regions without access and connect them
        if (self.eligibility_matrix.sum(axis=0) == 0).any():
            remote_gid1s = list(self.eligibility_matrix.index[(self.eligibility_matrix.sum(axis=0) == 0)])
            for remote_gid1  in remote_gid1s:
                #find nearest
                df_drop = self.distance_matrix_km[remote_gid1].drop(remote_gid1)
                nearest = df_drop.index[df_drop.argmin()]
                #set connection to nearest:
                self.eligibility_matrix[nearest][remote_gid1] = 1
                self.eligibility_matrix[remote_gid1][nearest] = 1


        self.eligibility_matrix = self._connectSubgraphs(self.eligibility_matrix, self.distance_matrix_km)

        #get share of on to offshore
        self._buildPipelineShape() #we should now have a shape: self.shp_transmission
        self._get_share_onshore_offshore()
        self._save_pipeline_shape()
            
        #dump to csv
        if self.readorssave:
            os.makedirs(os.path.dirname(self.path_distancematrix), exist_ok=True)
            self.distance_matrix_km.to_csv(self.path_distancematrix)
            self.eligibility_matrix.to_csv(self.path_eligibilitymatrix)
            self.share_onshore_matrix.to_csv(self.path_share_onshore_matrix)
        return self

    def _buildPipelineShape(self):
        
        print(f'Building pipeline shape.')
        if not hasattr(self, 'eligibility_matrix'):
            _ = self.extractEligibilityMatrix_km()
        
        n_regions = self.shape.shape[0]
        srs = gk.srs.EPSG4326

        centroids = self.shape.copy()

        #GetCentroids
        def getCentroids(geom):
            point = getCentroid180Long(geom)
            point.AssignSpatialReference(srs)
            return point#gk.geom.transform(point, equal_disance_srs)

        centroids['centroids'] = centroids['geom'].apply(getCentroids)

        shape = pd.DataFrame()

        #loop all connections
        for i in range(n_regions):
            for j in range(i,n_regions):
                if i==j:
                    continue
                if self.eligibility_matrix.iloc[i,j] == 1:
                    geom = gk.geom.line(
                        [
                            (centroids.centroids.iloc[i].GetX(), centroids.centroids.iloc[i].GetY()),
                            (centroids.centroids.iloc[j].GetX(), centroids.centroids.iloc[j].GetY())
                        ],
                        srs=srs
                        )
                    bus_0 = centroids[self.region_name_col].iloc[i]
                    bus_1 = centroids[self.region_name_col].iloc[j]
                    s_norm = self.distance_matrix_km.iloc[i,j]
                    columns = ['geom', 'bus_0', 'bus_1', 'len_km']
                    values = [[geom, bus_0, bus_1, s_norm]]
                    temp = pd.DataFrame(values, columns=columns)
                    shape = pd.concat([shape, temp], ignore_index=True)

        self.shp_transmission = shape

    def _save_pipeline_shape(self):
        srs = gk.srs.EPSG4326
        if self.readorssave:
            print(f'Saving pipeline shape to:')
            print(self.path_transmission)
            gk.vector.createVector(
                geoms = self.shp_transmission,
                output = self.path_transmission,
                srs = srs,
                overwrite = True
            )

    def _get_share_onshore_offshore(self):
        '''calculates the share of the transmissions over the region shape file
        '''
        #make one poly with all regions
        self.share_onshore_matrix = pd.DataFrame(
            np.eye(len(self.distance_matrix_km)),
            index=self.distance_matrix_km.index,
            columns=self.distance_matrix_km.columns,
            )
            
        regions = list(self.shape.geom)
        regions_all = regions[0]
        for region in regions:
            regions_all = regions_all.Union(region)

        for i, row_line in self.shp_transmission.iterrows():
            index_i = row_line.name
            line = row_line.geom
            line_onshore = regions_all.Intersection(line)

            share_onshore = line_onshore.Length() / line.Length()
            assert 0 <= share_onshore and share_onshore <=1
            
            #write into shape
            self.shp_transmission.loc[index_i,"share_onsh"] = share_onshore
            self.share_onshore_matrix.loc[row_line.bus_0, row_line.bus_1] = share_onshore
            self.share_onshore_matrix.loc[row_line.bus_1, row_line.bus_0] = share_onshore
        
    def return_dict(self, detour_factor):
        # self.saveFineRegions() # TODO: Delete, as done in singleton _save_regions, and available here with self.path_regions 

        self.extractEligibilityMatrix_km(detour_factor)

        transmission_vars = dict()
        transmission_vars['locationalEligibility'] = self.eligibility_matrix
        transmission_vars['distances'] = self.distance_matrix_km
        transmission_vars['share_onshore'] = self.share_onshore_matrix
        transmission_vars['shape_transmission'] = self.shp_transmission
        if self.readorssave:
            transmission_vars['shape_path_transmisions'] = self.path_transmission
        else:
            transmission_vars['shape_path_transmisions'] = None
        transmission_vars['shape_regions'] = self.shape
        if self.readorssave:
            transmission_vars['shape_path_regions'] = self.path_regions
        else:
            transmission_vars['shape_path_regions'] = None

        return transmission_vars

def getCentroid180Long(geometry):
    '''calculates the centroid for a shape with respect to the +-180 longitude.
    Output SRS is in original SRS. Change with gk.geom.transform if needed

    Parameters
    ----------
    geometry : Gdal Polygon / Multipolygon
        [description]

    Returns
    -------
    Centorid
        Gdal Point in 'geometry' srs 
    '''
    srs = geometry.GetSpatialReference()

    #transform to srs
    geometry = gk.geom.transform(geoms=geometry, toSRS=gk.srs.loadSRS(4326))
    env = geometry.GetEnvelope()
    if not (max(env)>179.9 and min(env)<-179.9):
        #no +-180 longitude problems:
        return geometry.Centroid()
    else:
        #we have a +- 180deg problem
        box_coords_east =[(0,-90), (180,-90), (180,90), (0,90)]
        box_east = gk.geom.polygon(box_coords_east)
        box_coords_west =[(0,-90), (-180,-90), (-180,90), (0,90)]
        box_west = gk.geom.polygon(box_coords_west)

        g_splitted = [
            geometry.Intersection(box_east),
            geometry.Intersection(box_west)
        ]
        g_Centroids = [g.Centroid() for g in g_splitted]
        g_Areas = [g.Area() for g in g_splitted]                        

        #correct +-180 degree:
        g_Centroids_new = []
        for g_c in g_Centroids:
            if g_c.GetX()<0:
                Xnew = g_c.GetX() + 360
                Y = g_c.GetY()
                c_new = gk.geom.point(Xnew,Y)
                g_Centroids_new.append(c_new)
            else:
                g_Centroids_new.append(g_c)
        
        #calc new averages
        x_sum = 0
        y_sum = 0
        for i in range(len(g_Centroids)):
            x_sum += g_Areas[i] * g_Centroids_new[i].GetX()
            y_sum += g_Areas[i] * g_Centroids_new[i].GetY()
        x_c = x_sum / sum(g_Areas)
        y_c = y_sum / sum(g_Areas)
        if x_c > 180:
            x_c = x_c-360
        
        centroid = gk.geom.point(x_c, y_c, srs=gk.srs.loadSRS(4326))
        #retransform to original srs
        centroid = gk.geom.transform(geoms=centroid, toSRS=srs)

        return centroid


# @profile
def _clusterGrid(grid_gdf, regions_gdf):
    import geopandas as gpd

    from distutils.version import StrictVersion

    gpd_version = StrictVersion(gpd.__version__)
    from shapely.geometry import Point

    # has to be converted to epsg 4326
    df = grid_gdf.to_crs(4326)
    bus_regions = regions_gdf.to_crs(4326)
    bus_regions = bus_regions.set_index("GID_1")

    df["point0"] = df.geometry.apply(lambda x: Point(x.coords[0]))
    df["point1"] = df.geometry.apply(lambda x: Point(x.coords[-1]))

    #%%
    from distutils.version import StrictVersion

    gpd_version = StrictVersion(gpd.__version__)

    def haversine_pts(a, b):
        """
        Determines crow-flies distance between points in a and b

        ie. distance[i] = crow-fly-distance between a[i] and b[i]

        Parameters
        ----------
        a, b : N x 2 - array of dtype float
            Geographical coordinates in longitude, latitude ordering

        Returns
        -------
        c : N - array of dtype float
            Distance in km

        See also
        --------
        haversine : Matrix of distances between all pairs in a and b
        """

        lon0, lat0 = np.deg2rad(np.asarray(a, dtype=float)).T
        lon1, lat1 = np.deg2rad(np.asarray(b, dtype=float)).T

        c = np.sin((lat1 - lat0) / 2.0) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin((lon0 - lon1) / 2.0) ** 2
        return 6371.000 * 2 * np.arctan2(np.sqrt(c), np.sqrt(1 - c))

    def build_clustered_gas_network(df, bus_regions, length_factor=1.25):
        df = df.reset_index(drop=True)
        for i in [0, 1]:

            gdf = gpd.GeoDataFrame(geometry=df[f"point{i}"], crs="EPSG:4326")

            kws = dict(op="within") if gpd_version < "0.10" else dict(predicate="within")
            bus_mapping = gpd.sjoin(gdf, bus_regions, how="left", **kws).index_right
            bus_mapping = bus_mapping.groupby(bus_mapping.index).first()

            df[f"bus{i}"] = bus_mapping

            df[f"point{i}"] = df[f"bus{i}"].map(bus_regions.to_crs(3857).centroid.to_crs(4326))

        # drop pipes where not both buses are inside regions
        df = df.loc[~df.bus0.isna() & ~df.bus1.isna()]

        # drop pipes within the same region
        df = df.loc[df.bus1 != df.bus0]

        # recalculate lengths as center to center * length factor
        df["length"] = df.apply(lambda p: length_factor * haversine_pts([p.point0.x, p.point0.y], [p.point1.x, p.point1.y]), axis=1)

        # tidy and create new numbered index
        df.drop(["point0", "point1"], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)

        def sortNames(a, b):
            o = f"{a}.{b}" if a < b else f"{b}.{a}"
            return o

        df["bus"] = df.apply(lambda row: sortNames(row.bus0, row.bus1), axis=1)
        df = df.set_index("bus")

        return df

    def reindex_pipes(df):

        df.sort_index(axis=1, inplace=True)

    def aggregate_parallel_pipes(df):

        strategies = {
            "bus0": "first",
            "bus1": "first",
            # "cables": "sum",
            # "num_parall": "sum",
            # "v_nom": "sum",
            "s_nom": "sum",
            # "s_max_pu": "max",
            "length": "mean",
            # "line_id": " ".join,
        }
        return df.groupby(df.index).agg(strategies)

    # collects grids that cross a border
    clustered_network = build_clustered_gas_network(df, bus_regions)

    # reindex
    reindex_pipes(clustered_network)

    # aggregate parallel grids
    gas_network_final = aggregate_parallel_pipes(clustered_network)

    return gas_network_final


def processExistingElectricityGrid(self, technology_name, model_unit, data_unit=None, path_grids=None): # issue 112
    """
    processes the existing electricity grid for the region shape file
    technology_name: name of the technology
    path_grids: path to the shape file of the electricity grid
    """

    if path_grids is None:
        path_grids = default_grids_information[technology_name]["base_folder"]
    if data_unit is None:
        data_unit = default_grids_information[technology_name]["capacity_unit"]

    # calculate a power unit conversion factor for the given model/potentials data combination
    capacity_conversion_factor = UnitHandling().get_unit_conversion_factor(input_unit=data_unit, target_unit=model_unit)

    import geopandas as gpd

    # load region shapefile with geopandas
    regionShapeFilePath = os.path.join(self.model_base_folder, "spatial_data", "regions.shp")
    regions = gpd.read_file(regionShapeFilePath).to_crs(3857)
    regions_orig = regions.copy()
    # regions_orig["geometry"] = regions_orig.geometry.buffer(10000) # Not sure why this was used

    # simplyfy geometry to increase performance (cant do that for small regions!!!)
    if (regions.geometry.area / 1e6).sum() > 10000:  # check if larger than 10000km2
        regions.geometry = regions.geometry.simplify(500)
    mask_df = gpd.GeoDataFrame(regions.geometry).reset_index(drop=True)
    # check for invalid geometries
    if mask_df.is_valid.all() == False:
        from shapely.validation import make_valid
        mask_df.geometry = mask_df.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)

    # load grid shapefile with geopandas
    grids = gpd.read_file(path_grids).to_crs(3857)
    grids["id"] = grids.reset_index().index # assign unique id to each grid
    # TODO: speed up by removing lines with zero capacity

    # clip grids first to bounding box and then to polygon (increases performance)
    min_bounds = regions.geometry.bounds.min()
    max_bounds = regions.geometry.bounds.max()

    # Create Offshore Region
    from shapely.geometry import Polygon, MultiPolygon, LineString
    union_poly = regions.unary_union
    # bounding box of the union
    bbox =  union_poly.bounds
    bbox_poly = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])
    # find the difference between the union polygon and the bounding box polygon
    neg_union_poly = union_poly.symmetric_difference(bbox_poly)
    _offshore_reg = gpd.GeoSeries(neg_union_poly)
    offshore_reg = gpd.GeoDataFrame(_offshore_reg, columns=['geometry'],crs=regions.crs)
    # find lines with offshore intersection
    offshore_lines_id = gpd.overlay(grids, offshore_reg, how='intersection')
    offshore_lines = grids[grids.id.isin(offshore_lines_id.id)]
    # check which of these lines have endpoints in at least one onshore region
    offshore_lines = offshore_lines.sjoin(regions, how='left', op='intersects').dropna(subset=["GID_0"]).drop("index_right",axis=1).drop_duplicates(subset="id")
    # convert all multilinestrings to linestrings
    # !!!!!! WARNING: If offshore lines are multilinestrings, they will be split into multiple lines and thus not considered in the clustering !!!
    offshore_lines = offshore_lines.explode(index_parts=False)
    # Offshore lines get converted to single line from endpoint to endpoint
    offshore_lines["geometry"] = offshore_lines.geometry.apply(lambda x: LineString([x.coords[0], x.coords[-1]]))
    # remove these lines from the grid (since they are treated seperately)
    grids = grids[~grids.id.isin(offshore_lines.id)]

    grids_clipped = gpd.clip(grids, [min_bounds.minx, min_bounds.miny, max_bounds.maxx, max_bounds.maxy])
    grids_clipped = gpd.clip(grids_clipped, mask_df)

    if grids_clipped.empty:
        return None

    #### Process grids
    ##########################

    # explode multilinestrings to linestrings
    grids_clipped_explode = grids_clipped.explode(index_parts=False)

    # split linestrings into segments
    from shapely.geometry import LineString, LinearRing

    def segments(curve):
        return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))

    grids_clipped_explode["geom"] = grids_clipped_explode["geometry"].apply(lambda x: [x for x in segments(x)])
    gec_gpd_ls_seg = grids_clipped_explode.explode("geom").drop("geometry", axis=1).rename(columns={"geom": "geometry"}).reset_index(drop=True)
    gec_gpd_ls_seg = gpd.GeoDataFrame(gec_gpd_ls_seg, geometry="geometry", crs=grids_clipped_explode.crs)
    # add back offshore lines
    gec_gpd_ls_seg = pd.concat([gec_gpd_ls_seg,offshore_lines[gec_gpd_ls_seg.columns]])

    aggregated_grid = _clusterGrid(gec_gpd_ls_seg, regions_orig) # use original regions without simplify

    # Build transmission dataframe
    def add_missing_regions_2d(locations, data_df):
        missing_2d = pd.DataFrame(0, index=list(locations.difference(data_df.index)), columns=list(locations.difference(data_df.columns)))
        missing_2d = missing_2d.sort_index()
        missing_2d = missing_2d.reindex(sorted(missing_2d.columns), axis=1)

        data_df = data_df.join(missing_2d, how="outer")
        data_df = data_df.fillna(0)
        return data_df

    # create capacity matrix
    cap_data = aggregated_grid[["bus0", "bus1", "s_nom"]].set_index(["bus0", "bus1"])
    cap_matrix = cap_data.unstack().fillna(0).droplevel(0, axis=1)
    # make symmetrtic
    cap_matrix = cap_matrix.add(cap_matrix.T, fill_value=0)
    # add missing regions
    cap_matrix_all_regions = add_missing_regions_2d((ModelLocations().locationIDs), cap_matrix) # issue 112
    # apply unit conversion
    cap_matrix_all_regions = cap_matrix_all_regions * capacity_conversion_factor

    # calculate locational eligibility
    locationalEligibility = cap_matrix_all_regions.copy()
    locationalEligibility[locationalEligibility > 0] = 1

    # calculate distances
    distances = aggregated_grid[["bus0", "bus1", "length"]].set_index(["bus0", "bus1"])
    distances = distances.unstack().fillna(0).droplevel(0, axis=1)
    # make symmetrtic
    distances = distances.add(cap_matrix.T, fill_value=0)
    # add missing regions 
    distances = add_missing_regions_2d((ModelLocations().locationIDs), distances) # issue 112

    # create transmission vars dict
    transmissionVars = {
        "locationalEligibility": locationalEligibility,
        "capacityFix": cap_matrix_all_regions,
        "distances": distances,
    }

    # save to file
    def create_simple_transmission_gdf(model_regions, data, variable_name="capacity"):
        """
        data: 2d array with transmission capacities/eligibility etc.
        """
        import numpy as np

        _data = data.unstack().replace(0, np.nan).dropna()

        from shapely.geometry import LineString

        reg0_list = []
        reg1_list = []
        geom_list = []
        data_list = []

        for idx, row in _data.items():
            # print(idx, row)
            reg0_idx = idx[0]
            reg1_idx = idx[1]
            reg0_geom = model_regions.loc[reg0_idx, "geometry"]
            reg1_geom = model_regions.loc[reg1_idx, "geometry"]
            line = LineString([reg0_geom.centroid, reg1_geom.centroid])
            reg0_list.append(reg0_idx)
            reg1_list.append(reg1_idx)
            geom_list.append(line)
            data_list.append(row)

        transmission_gdf = gpd.GeoDataFrame(
            {"reg0": reg0_list, "reg1": reg1_list, "geometry": geom_list, f"{variable_name}": data_list}, crs=model_regions.crs
        )

        return transmission_gdf

    cap_matrix_full_gdf = create_simple_transmission_gdf(regions.set_index(ModelLocations().locationID_attr), cap_matrix_all_regions)   # issue 112
    if not os.path.exists(os.path.join(self.model_base_folder, "spatial_data", "transmission")):
        os.makedirs(os.path.join(self.model_base_folder, "spatial_data", "transmission"))
    cap_matrix_full_gdf.to_file(os.path.join(self.model_base_folder, "spatial_data", "transmission", "electricity_grid_brownfield.shp"))

    return transmissionVars

if __name__ == '__main__':
    

    do_test1 = False

    if do_test1:
        #Test with ARG
        shape = default_path_information["general_data"]["default_regions_shp"]
        shape = gk.vector.extractFeatures(shape)

        sp = spatialDefinition(
            shape=shape,
            region_name_col='GID_1',
        )
        sp.extractEligibilityMatrix_km()
        sp.buildPipelineShape()

        transmission_vars = sp.return_dict()

        assert transmission_vars['locationalEligibility'].sum().sum() == 92
        assert np.isclose(transmission_vars['distances'].sum().sum(), 662109.5556177341)
        assert transmission_vars['distances'].shape == (24,24)


    #Argentina
    shape_fine_unions = r"/storage_cluster/internal/data/gears/FineUnion/regions/gadm36_0_inc_regions_v2.1.shp"
    shape = default_path_information["general_data"]["default_regions_shp"]

    shapes = [
        gk.vector.extractFeatures(shape.replace('<GID0>', 'AUS')),
        gk.vector.extractFeatures(shape.replace('<GID0>', 'NZL'))
        ]

    shape = pd.concat(shapes).reset_index(drop=True)
    gk.vector.createVector(shape, 'regions_shape.shp')
    

    sp = spatialDefinition(
        shape=shape,
        region_name_col='GID_1',
    )
    sp.extractEligibilityMatrix_km(detour_factor=1.3)

    assert sp.eligibility_matrix.sum().sum() == 202 #26.02.2023, d.franzmann: cannot say what changed, but I checked the shapes manually and they seem fine. 
    
    pass