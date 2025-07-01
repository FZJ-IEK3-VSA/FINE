import geopandas as gpd
from .modelManager import modelManager


class esMWorkflowManager(object):
    """
    The modelInstance class creates an object that store the geographic scope and the spatial resolution
    of the generated model, as well as all generated input data.
    """

    def __init__(self, shapefile, column, crs="epsg:3857"):
        """
        Constructor for creating an spatialScope class instance.

        **Required arguments:**
        :param shapefile: Shapefile of regions of interest.
        :type model_scenario: path to shapefile

        **Default arguments:**
        :param crs: Defines the coordination system
            |br| * default value is 'epsg:3857'
        :type crs: string
        """
        # Set attributes
        self.region_shape = gpd.read_file(shapefile).to_crs({"init": crs})

        # Create the esM instance
        self.get_location_names(column)
        self.get_input_data()
        self.get_model()

    def get_location_names(self, column):
        self.locations = list(self.region_shape[column])

    def get_input_data(self):
        self.data = {}

    def get_model(self):

        # create a modelManager instance
        mm = modelManager(locations=self.locations, data=self.data)

        # create a model setup with all initial default input
        self.esM = modelManager.model_setup(mm)

        # mm.add_sources(self, esM=self.esM)
        # mm.add_sink(self, esM=self.esM)
