import geokit as gk
import modelBuilder
from modelBuilder.inputDataHandler import preprocess_union_shape

shapeFilePath = "/storage_cluster/internal/data/gears/FineUnion/regions/gadm36_GID1(0)_EastWest_and_largeRegion_split_epsg4326_withUnion.shp"

shape = gk.vector.extractFeatures(
    shapeFilePath,
    where=f"GID_0 in ('DEU')"
)

shape = preprocess_union_shape(
    location_shape=shape,
    max_regions=12,
    return_as_gk=True,
)

commodityUnitsDict = {
                "electricity": r"GW$_{el}$",
                "hydrogen_gas": r"GW$_{H_{2},LHV}$",
            }

model_base_folder = "."

### Init Model Manager only writes vars to self.xyz
mb = modelBuilder.modelManager(
    location_shape=shape,
    locationID_column="GID_1",
    commodityUnitsDict=commodityUnitsDict,
    cost_year=2050,
    model_base_folder=model_base_folder,
    srs=4326,
    path_to_techno_economic_data_yaml=None,
    weather_year=2018,
)

mb.completeSetup()

mb.addPotentialConstGreenfield(
    technology="geothermal_EGS",
    N_cluster=3,
)
pass