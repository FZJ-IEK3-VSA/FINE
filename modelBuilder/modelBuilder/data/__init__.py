import os
import yaml
data_folder = os.path.dirname(__file__)

with open(os.path.join(data_folder, "default_demands.yaml")) as fp:
    default_demand_information = yaml.load(fp, Loader=yaml.FullLoader)

with open(os.path.join(data_folder, "default_paths.yaml")) as fp:
    default_path_information = yaml.load(fp, Loader=yaml.FullLoader)

with open(os.path.join(data_folder, "default_potentials.yaml")) as fp:
    default_potentials_information = yaml.load(fp, Loader=yaml.FullLoader)

with open(os.path.join(data_folder, "default_grids.yaml")) as fp:
    default_grids_information = yaml.load(fp, Loader=yaml.FullLoader)

