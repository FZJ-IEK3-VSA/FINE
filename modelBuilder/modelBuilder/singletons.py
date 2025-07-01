# import basic packages
import datetime
import inspect
import json
import numpy as np
import os
import pandas as pd
from typing import Any, Dict
from threading import Lock
import time

# import third-party packages
import geokit as gk
import geopandas as gpd
import natsort
import osgeo
import shapely
import yaml

# import other modules
from modelBuilder.data import data_folder
from modelBuilder import utils

# TODO replace this by an import of the agg funcs mapper when merged, see new brownfield branch
aggregation_function_mapper = {
    "mean": np.mean,
    "max": max,
    "sum": np.sum,
    "min": min,
}

class SingletonMeta(type):
    """
    A class for a thread-safe implementation of Singleton:
    https://refactoring.guru/design-patterns/singleton/python/example#example-1
    """
    _instances: Dict[Any, Any] = {}

    _lock: Lock = Lock()
    # We now have a lock object that will be used to synchronize threads during first access to the Singleton.

    def __call__(cls, *args, **kwargs):
        """Possible changes to the value of the `__init__` argument do not affect the returned instance."""
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
    
    def exists(cls):
        """Check if an instance has been created already."""
        return cls in cls._instances
    
    def get_attribute(cls, attr_name):
        if cls.exists():
            cls._lock.release()
            attr = getattr(cls(), attr_name)
            cls._lock.acquire()
            return attr
        else:
            raise NameError(f"{cls.__name__} singleton does not exist.")

    def reset(cls):
        """Resets initialization of an existing singleton."""
        if cls in cls._instances:
            cls._instances.pop(cls)

#%% UnitConversion
    
class UnitHandling(metaclass=SingletonMeta):
    """
    This singleton holds information about all units in model and input 
    data and provides functionalities to convert or add commodities and 
    units to the commodity units dict.
    """
    def __init__(self, commodity_units_dict={}, unit_conversions_yaml=None):
        """
        commodity_units_dict : dict
            Starting point for commodity unit dict, containing the units 
            for all model units. Formatted with commodity as key, unit 
            tuple as value (esM unit and multiple of SI unit). Can also 
            be an empty dict. By default {}.
        """
        # load unit conversions
        self._load_unit_conversions_dict(unit_conversions_yaml=unit_conversions_yaml)

        # check and set commodity unit dict
        self.commodity_units_dict = dict()
        self.check_new_commodity_units_dict(commodity_units_dict)
        self.commodity_units_dict = commodity_units_dict


    def _load_unit_conversions_dict(self, unit_conversions_yaml):
        """
        Auxiliary method to load and add the unit conversions yaml as an 
        attribute.

        unit_conversions_yaml : str
            Filepath to yaml with unit conversions like in data folder.
        """
        if unit_conversions_yaml is None:
            unit_conversions_yaml = os.path.join(data_folder, "unit_conversions.yaml")
        elif not isinstance(unit_conversions_yaml, str):
            raise TypeError(f"unit_conversions_yaml must be str formatted: {unit_conversions_yaml}")
        elif not os.path.isfile(unit_conversions_yaml):
            raise FileNotFoundError(f"unit_conversions_yaml not found: {unit_conversions_yaml}")
        elif not os.path.splitext(unit_conversions_yaml)[-1] in ['.yml', '.yaml']:
            raise TypeError(f"unit_conversions_yaml must be of type .yml or .yaml: {unit_conversions_yaml}")
        
        # load default paths as a dictionary attribute
        with open(unit_conversions_yaml) as fp:
            unit_conversions_dict = yaml.load(fp, Loader=yaml.FullLoader)
        self.unit_conversions_yaml = unit_conversions_yaml
        
        # make all units upper case
        unit_conversions_dict_upper = dict()
        for _type, _conversions in unit_conversions_dict.items():
            unit_conversions_dict_upper[_type]=dict()
            assert isinstance(_conversions, dict), f"Each unit type must contain a nested dict with unit sub keys and conversion factor values."
            for _unit, _factor in _conversions.items():
                assert isinstance(_factor, (int, float)), f"values of unit conversion dict must be int or float conversion factors relative to SI unit."
                unit_conversions_dict_upper[_type][_unit.upper()]=_factor
        
        # check the unit conversion dict for uniqueness
        all_units = [u for us in unit_conversions_dict.values() for u in us]
        all_units_upper = [u for us in unit_conversions_dict_upper.values() for u in us]
        assert len(all_units)==len(set(all_units)),\
            f"unit_conversion dict contains duplicate units (case-insensitive)."
        
        # set as attributes
        self.unit_conversions_dict = unit_conversions_dict_upper
        self.all_units = all_units
        self.all_units_upper = all_units_upper


    def check_new_commodity_units_dict(self, new_commodity_units_dict):
        """
        Asserts correct format of any commodity units dict.

        commodity_units_dict : dict
            commodity as key, unit tuple as value (esM unit and multiple 
            of SI unit)
        """
        assert len(set(new_commodity_units_dict.keys()))==len(new_commodity_units_dict.keys()), f"new commodity names must be unique!"
        for k,v in new_commodity_units_dict.items():
            assert isinstance(k, str), f"All commodity_units_dict keys must be str formatted. Here: {k}"
            assert isinstance(v, (list,tuple)), "All commodity_units_dict values must be tuples or lists of esM unit and multiple of SI unit (e.g.: ('GW$_{el}$', 'GW'))"+f" Here: {v}"
            for _u in v: 
                assert isinstance(_u, str), f"All units in commodity_units_dict value tuples must be str formatted. Here: {_u}"
            assert v[1].upper() in self.all_units_upper, \
                f"multiple of SI unit '{v[1]}' for commodity '{k}' is not a known SI unit in unit conversion dict. Known units (case-insensitive): {', '.join(self.all_units)}"
            # check if we already have data
            if k in self.commodity_units_dict.keys():
                # if so make sure units are the same
                assert v == self.commodity_units_dict[k], \
                    f"new_commodity_units_dict commodity '{k}' is already a key in existing commodity_units_dict, but new units '{v}' are not the same as existing '{self.commodity_units_dict[k]}'"


    def add_commodity_units_dict(self, new_commodity_units_dict):
        """
        Adds a new_commodity_units_dict in case that data is not yet 
        in existing commodity_units_dict.

        new_commodity_units_dict : dict
            Dictinary with commodity name keys and unit tuples as 
            values (esM unit, multiple of SI unit) like example: 
            ("GW$_{el}", "GW")
        """
        # check input
        self.check_new_commodity_units_dict(new_commodity_units_dict=new_commodity_units_dict)
        for k, v in new_commodity_units_dict.items():
            if k in self.commodity_units_dict.keys():
                # already exists
                continue
            else:
                self.commodity_units_dict[k] = v
    
    def compare_esM_commodityUnitsDict(self, esM):
        """
        Compares if the commodityUnitsDict of a given esM object aligns 
        with the internal commodity_units_dict.
        """
        # extract commodityUnitsDict
        try:
            esm_commodityUnitsDict = getattr(esM, "commodityUnitsDict")
        except:
            raise AttributeError(f"esM does not have commodityUnitsDict attribute. Check esM input.")
        assert isinstance(esm_commodityUnitsDict, dict) # make sure
        # first check if all data in esm_commodityUnitsDict is in internal dict
        for k, v in esm_commodityUnitsDict.items():
            if not k in self.commodity_units_dict.keys():
                raise KeyError(f"commodity '{k}' from esM.commodityUnitsDict is not in UnitHandling commodity units dict. Existing keys: {', '.join(self.commodity_units_dict.keys())}")
            if not v == self.commodity_units_dict[k][0]:
                raise ValueError(f"'{k}' commodity unit in esm.commodity_units_dict is '{v}', but '{self.commodity_units_dict[k][0]}' in internal UnitHandling commodity_units_dict.")
        # now check the inverse
        for k, v in self.commodity_units_dict.items():
            if not k in esm_commodityUnitsDict.keys():
                raise KeyError(f"commodity '{k}' from UnitHandling commodity units dict is not in esM.commodityUnitsDict. Existing keys: {', '.join(self.commodity_units_dict.keys())}")
            if not v[0] == esm_commodityUnitsDict[k]:
                raise ValueError(f"'{k}' commodity unit in internal UnitHandling commodity_units_dict is '{v[0]}', but '{self.commodity_units_dict[k]}' in esm.commodity_units_dict.")
        
    
    def get_model_unit_as_multiple_of_SI_unit(self, commodity):
        """
        Returns the unit for a given commodity as multiple of an SI unit.
        
        commodity : str
            commodity name for which unit shall be returned.
        """
        if not commodity in self.commodity_units_dict.keys():
            raise KeyError(f"commodity '{commodity}' is not an existing commodity in commodity_units_dict. Select from: {', '.join(self.commodity_units_dict.keys())}")
        _unit = self.commodity_units_dict[commodity][1]
        assert _unit.upper() in self.all_units_upper # make sure 
        return _unit


    def get_esM_commodityUnitsDict(self):
        """
        Returns the commodity units dict in a format that is applicable 
        as ETHOS.FINE model setup 'commodityUnitsDict' parameter.
        """
        return {k:v[0] for k,v in self.commodity_units_dict.items()}


    def get_commodities(self, as_set=False):
        """
        Returns all commodities that are currently in the commodity units
        dict.

        as_set : bool, optional
            If True, the commodities will be returned as set, else as a 
            list. By default False.
        """
        if as_set:
            return set(self.commodity_units_dict.keys())
        else:
            return list(self.commodity_units_dict.keys())
    
            # if k in list(self.esM.commodities):
                
            #     assert self.esM.commodityUnitsDict[k]==v, f"{k} is already in model commodities but unit is {self.esM.commodityUnitsDict[k]} (instead of {v})!"

    

    def get_unit_conversion_factor(self, input_unit:str, target_unit:str) -> float:
        """
        Returns unit conversion factor = input_unit/target_unit.

        Args:
            input_unit (str): String formatted SI input unit
            target_unit (str): String formatted SI output unit, must be of 
                the same unit type as(convertible into) input_unit.

        Returns:
            float: Conversion factor = input/target.
        """
        # make sure we have information on the specific in- and output units
        assert input_unit.upper() in self.all_units_upper, f"input_unit {input_unit} is not available as a unit in: {self.unit_conversions_yaml}. Select from the following conversion units (not case-sensitive): {', '.join(self.all_units)}."
        assert target_unit.upper() in self.all_units_upper, f"target_unit {target_unit} is not available as a unit in: {self.unit_conversions_yaml}. Select from the following conversion units (not case-sensitive): {', '.join(self.all_units)}."

        # now extract the unit type key of the model unit
        unittype = [_type for _type,_conversions in self.unit_conversions_dict.items() if target_unit.upper() in _conversions.keys()][0]

        # extract the multiplication factor for each of the units
        input_factor=self.unit_conversions_dict[unittype][input_unit.upper()]
        output_factor=self.unit_conversions_dict[unittype][target_unit.upper()]
        
        return input_factor/output_factor


#%%
    
# INPUT DATA INFO
 
class InputDataInfo(metaclass=SingletonMeta):
    """
    This singleton holds information on all default or custom input data,
    such as data- and scenario-specific paths, attributes, clustering 
    requirements etc.

    NOTE: This singleton has 2 main layers. 'Original' data is the data 
    that was loaded from the yaml definition ('raw' data is a sub version
    hereof where case-sensitivity has not yet been considered). 'Custom' 
    data is data that was added to the singleton in a later stage, e.g. 
    by a technology adding function of the modelManager. It can be reset 
    to the 'original' data anytime. 'Original' data should be immutable. 
    """
    # define a list of allowed attributes to set under each technology key
    allowed_attrs=[
        "ts_base_path", "cap_base_path", "data_unit", "aggregation_dict", 
        "additional_aggregation_vars", "sub_dataset_name", "negative_ts",
        "daily_timeseries", "hourly_reference_timeseries"
    ]

    def __init__(self, weather_year, base_year, number_of_investment_periods, investment_period_interval, path_to_custom_input_data=None):
        """
        path_to_custom_input_data : str
            The path to a yaml containing all input data information for
            all technologies, with technologies as keys.
        """
        assert isinstance(weather_year, int) and weather_year>1980, f"weather_year must be an integer > 1980"
        self.weather_year = weather_year #TODO rather save weather year and other model params in a separate singleton? not really InputDataInfo
        assert isinstance(base_year, int) and base_year>1980, f"base_year must be an integer > 1980"
        self.base_year = base_year #TODO rather save base year and other model params in a separate singleton? not really InputDataInfo
        self.investment_period_interval=investment_period_interval
        finalyear = self.base_year + number_of_investment_periods * investment_period_interval
        self.investment_period_names = list(range(self.base_year, finalyear, investment_period_interval))

        if path_to_custom_input_data is None:
            # use the deafault input data from this repository data folder
            self.path_to_input_data = os.path.join(
                data_folder, 
                'default_potentials.yaml'
            )
        else:
            # check and use the path_to_custom_input_data
            if not os.path.isfile(path_to_custom_input_data):
                raise FileNotFoundError(f"'path_to_custom_input_data' must be an existing file: {path_to_custom_input_data}")
            if not os.path.splitext(path_to_custom_input_data)[-1] in ['.yml', '.yaml']:
                raise TypeError(f"path_to_custom_input_data must point to a .yml or .yaml file: {path_to_custom_input_data}")
            self.path_to_input_data = path_to_custom_input_data

        # load the input data into as 'original_data' attribute - this will never be altered!
        self.original_data = yaml.load(
                open(self.path_to_input_data), 
                Loader=yaml.FullLoader
            )
        # add attribute of all original-spelling techs as list
        self.info_techs_originalcase = list(self.original_data.keys())
        # now create a duplicate of the data that can be altered and will be used for data loading
        # rename all dict keys with lower case strings to make access easier
        self.data = dict()
        for tech in self.original_data.keys():
            self.data[tech.lower()] = self.original_data[tech]
        # add attribute of all lower case tech names as list
        self.info_techs_lowercase = list(self.data.keys())

        # check formatting and completeness
        self._check_inputs()

    def _check_inputs(self):
        """This auxiliary method checks the formatting and completeness 
        of all input data info"""
        for tech, tech_data in self.data.items():
            if not isinstance(tech_data, dict):
                raise TypeError(f"value of '{tech}' in InputDataInfo must be a sub dictionary of attribute keys and values.")
        # check attrs for all techs
        self._check_mandatory_attributes()

    def get_info(self, tech, attr):
        """
        This getter allows to extract a specific input data information 
        attribute for a given technology.

        technology : str
            The technology name, case-insensitive.
        attribute : str
            The information attribute to extract.
        """
        # check tech and attribute
        self.has_attr(tech=tech, attr=attr, fail_on_false=True)
        # return value
        return self.data[tech.lower()][attr]
    
    def update_and_get_path(self, tech, path_attr, add_spacer_mapper={}, must_exist=False, no_more_spacers=False):
        """
        This method is a convenience wrapper to get path info and replace
        all potential spacers defined by upper case <SPACER_NAMES>.
        """
        # predefine mapper and combine with custom data
        base_spacer_mapper = {
            "<WEATHERYEAR>" : self.weather_year,
            "<BASEYEAR>" : self.base_year,
        }
        _spacer_mapper = {**base_spacer_mapper, **add_spacer_mapper}

        fp = self.get_info(tech=tech, attr=path_attr)
        for _spacer, _value in _spacer_mapper.items():
            fp = fp.replace(_spacer, str(_value))
        if no_more_spacers and any ([x in fp for x in ['<', '>']]):
            raise ValueError(f"After replacement of all known spacers ({','.join(_spacer_mapper.keys())}), still '<' or '>' remain: {fp}")
        if must_exist and not os.path.exists(fp):
            raise FileNotFoundError(f"must_exist is True but filepath does not exist: {fp}")
        return fp

    def set_info(self, tech, attrs, vals, overwrite=False, verbose=True):
        """This method allows to set (or overwrite) attributes for a new 
        or an existing technology."""
        # check inputs
        if isinstance(attrs, str): 
            # if only one attr is given, make lists of both attr name and value(s)
            attrs = [attrs]
            vals = [vals]
        if not len(vals)==len(attrs):
            raise ValueError(f"If attrs or vals are given as iterables, the length must match.")
        # check if overwriting is ok
        if not self.has_tech(tech=tech):
            print(datetime.datetime.now(), f"Technology '{tech}' will be added to InputDataInfo singleton.", flush=True)
        if not overwrite:
            for attr in attrs:
                if self.has_attr(tech=tech, attr=attr, fail_on_false=False):
                    raise KeyError(f"attribute '{attr}' for technology '{tech}' already exists as a key in InputDataInfo and overwrite is False.")
        # now write data
        for attr, val in zip(attrs, vals):
            if not self.has_tech(tech=tech):
                self.data[tech]=dict()
            self.data[tech][attr] = val
        if verbose: print(datetime.datetime.now(), f"The following InputDataInfo attributes for technology '{tech}' were written : "+", ".join([f"{a}={v}" for a,v in zip(attrs, vals)]), flush=True)

    def has_tech(self, tech, fail_on_false=False):
        """Checks if a technology is contained in InoutDataInfo."""
        assert isinstance(tech, str), f"technology must be str formatted."
        if tech.lower() in self.info_techs_lowercase:
            return True
        elif fail_on_false:
            raise ValueError(f"technology '{tech}' (case-insensitive) is not in InputDataInfo keys. Available technologies: {', '.join(self.info_techs_originalcase)}. Check technology name or input file: {self.path_to_input_data}")
        else:
            return False
    
    def has_attr(self, tech, attr, fail_on_false=True):
        """Checks if a tech exists and has a given attribute."""
        # first check if tech exists
        self.has_tech(tech=tech, fail_on_false=True)
        # then check if attribute exists
        if attr in self.data[tech.lower()].keys():
            return True
        elif fail_on_false:
            raise AttributeError(f"attribute '{attr}' is not an attribute of '{tech}' in InputDataInfo. Check inputs or select from: {', '.join(self.data[tech].keys())}")
        else:
            return False
        
    def update_tech_info(self, tech, update_data, ignore_args=None):
        """
        Updates the technology inside the InputData instance (or sets a 
        technology if new) #TODO
        """
        if not isinstance(update_data, dict):
            raise TypeError(f"update_data must be dict type.")
        if ignore_args is not None and (isinstance(ignore_args, str) or not hasattr(ignore_args, '__iter__')):
            ignore_args = [ignore_args]
        # always ignore some arguments
        ignore_args.extend(['self', 'modelUnit', 'technology', 'FINE_kwargs', 'verbose'])

        # make sure we do not duplicate a tech due to case sensitivity
        if (
            (tech.upper() in [t.upper() for t in self.data.keys()])
            and (not tech in self.data.keys())
            ):
            _existing_tech = [t for t in self.data.keys() if t.upper()==tech.upper()][0]
            raise KeyError(f"Technology '{tech}' exists already as '{_existing_tech}'.")
        # add tech if not exists
        if not tech in self.data.keys():
            self.data[tech]=dict()
        # add or update self.data
        for attr, val in update_data.items():
            if attr in ignore_args:
                continue
            if val is None:
                continue
            #TODO add data integrity checks here!
            self.data[tech][attr] = val


    def _check_mandatory_attributes(self):
        """
        This method checks attributes for all InputDataInfo technologies 
        for formal correctness.
        """
        # define some helpers
        def _multiple_type_check(vals, types, fail=False):
            """Compares the type of one or multiple values against a given type or list of types (incl. None)"""
            if isinstance(vals, str) or isinstance(vals, dict) or not hasattr(vals, '__iter__'):
                vals = [vals]
            # deal with None
            _None = False
            _othertypes = False
            _typetuple = False
            if types is None:
                _None = True
            # if (NOT a type) AND (NOT a str) AND (is iterable) AND (contains None):
            elif (not inspect.isclass(types)) and (not isinstance(types, str)) and hasattr(types, '__iter__') and None in types:
                _None = True
                types = [x for x in types if not x is None]
                if len(types)==1:
                    types=types[0]
            # deal with other types
            if not inspect.isclass(types):
                assert isinstance(types, list), f"types must be a type class or a list of type classes. Here: {types} (type: {type(types)})."
                if len(types)>0:
                    types = tuple(types)
                    _typetuple = True
                    _othertypes = True
            else:
                _othertypes = True
            # check types
            for val in vals:
                if _None and val is None:
                    continue
                elif _othertypes and isinstance(val, types):
                    continue
                elif fail:
                    raise TypeError(f"val '{val}' (type {type(val)}) must be {'None or ' if _None else ''}any of the following types: {', '.join(types) if _typetuple else types}{' + None' if _None else ''}.")
                else:
                    print(f"'{val}' is of type: {type(val)}. Expected types: {', '.join(types) if _typetuple else types}{' + None' if _None else ''}")
                    return False
            return True
        
        def _check_instance(tech, attr, insts, must_exist=True):
            if must_exist:
                self.has_attr(tech=tech, attr=attr, fail_on_false=True)
            elif not self.has_attr(tech=tech, attr=attr, fail_on_false=False):
                return
            # check type
            if not _multiple_type_check(vals=self.get_info(tech=tech, attr=attr), types=insts):
                raise TypeError(f"Attribute '{attr}' of technology '{tech}' in InputDataInfo must be of any of the following types: {insts}")
        
        def _check_list_entries(tech, attr, entry_types):
            vals = self.get_info(tech=tech, attr=attr)
            if not isinstance(vals, list):
                raise TypeError(f"InputDataInfo attribute '{attr}' of technology '{tech}' is expected to be of list type. Here: {type(vals)}")
            if not _multiple_type_check(vals=self.get_info(tech=tech, attr=attr), types=entry_types):
                raise TypeError(f"Not all list entries of '{attr}' of technology '{tech}' in InputDataInfo are of the following types: {entry_types}")

        def _check_existence(tech, attr, _type="file"):
            self.has_attr(tech=tech, attr=attr, fail_on_false=True)
            assert _type in ['folder', 'file', 'both'], f"_type must be either 'file', 'folder' or 'both'."
            if _type.lower()=="folder" and not os.path.isdir(self.get_info(tech=tech, attr=attr)):
                raise FileNotFoundError(f"Attibute '{attr}' for technology '{tech}' is expected to be an existing folder: {self.get_info(tech=tech, attr=attr)}")
            if _type.lower()=="file" and not os.path.isfile(self.get_info(tech=tech, attr=attr)):
                raise FileNotFoundError(f"Attibute '{attr}' for technology '{tech}' is expected to be an existing file: {self.get_info(tech=tech, attr=attr)}")
            if _type.lower()=="both" and not os.path.exists(self.get_info(tech=tech, attr=attr)):
                raise FileNotFoundError(f"Attibute '{attr}' for technology '{tech}' is expected to exist: {self.get_info(tech=tech, attr=attr)}")

        def _make_list(tech, attr, empty_for_None=True):
            val = self.get_info(tech=tech, attr=attr)
            # cover the already-a-list-case
            if isinstance(val, list):
                # no need for action
                return
            # treat the None case separately
            if empty_for_None and val is None:
                new_val =  []
            elif isinstance(val, str):
                new_val = [val]
            elif hasattr(val, '__iter__'):
                # we have an iterable but not a str
                new_val = list(val)
            else:
                # must then be a scalar value or None if not empty_for_None
                new_val = [val]
            # set the new value
            self.set_info(tech=tech, attrs=attr, vals=new_val, overwrite=True, verbose=False)
        
        # process all technologies and attributes
        for tech in self.info_techs_lowercase:
            # check the scenario/iteration paths
            # TODO adapt this such that demands, conversions and storages, without or with timeseries, capacity or balance limits can be considered
            _check_instance(tech=tech, attr="ts_base_path", insts=[str, None])
            _check_instance(tech=tech, attr="cap_base_path", insts=[str, None])
            # _check_instance(tech=tech, attr="potentials_technology_name", insts=str) #TODO remove
            # _check_instance(tech=tech, attr="scenario", insts=str) #TODO remove
            # _check_instance(tech=tech, attr="iteration_name", insts=str) #TODO remove
            _check_instance(tech=tech, attr="data_unit", insts=str) #TODO change to 'data_unit' once other data than volatile potentials were added

            # check combined iteration results path
            if self.has_attr(tech=tech, attr="base_folder", fail_on_false=False):
                assert not self.has_attr(tech=tech, attr="base_path", fail_on_false=False), \
                    f"Technology '{tech}' in InputDataInfo may not have both base_folder (for GlobEP data) and base_path attribute."
                # we have a GlobEP result structure at hand, check combined path
                iteration_folder = os.path.join(
                    self.get_info(tech=tech, attr="base_folder"), 
                    self.get_info(tech=tech, attr="potentials_technology_name"), 
                    self.get_info(tech=tech, attr="scenario"), 
                    self.get_info(tech=tech, attr="iteration_name"),
                )
                assert os.path.isdir(iteration_folder), f"<base_folder>/<potentials_technology_name>/<scenario>/<iteration_name> must be an existing directory: {iteration_folder}"

            # check aggregation info
            _check_instance(tech=tech, attr="aggregation_dict", insts=[dict, None])
            additional_aggregation_vars = self.get_info(tech=tech, attr="additional_aggregation_vars")
            if not additional_aggregation_vars is None:
                for agg_funtion in additional_aggregation_vars.values():
                    assert agg_funtion in aggregation_function_mapper.keys(),\
                        f"additional_aggregation_vars function '{agg_funtion}' for technology '{tech}' is not a known aggregation function name. Check, expand or select from: {', '.join(aggregation_function_mapper.keys())}"
            
            _check_instance(tech=tech, attr="sub_dataset_name", insts=[str, None])
            _check_instance(tech=tech, attr="negative_ts", insts=[str, None])
            _check_instance(tech=tech, attr="daily_timeseries", insts=[str, list, None])
            _check_instance(tech=tech, attr="hourly_reference_timeseries", insts=[str, list, None])
            
            # PREPROCESS ATTRIBUTES

            # negative_ts
            _make_list(tech=tech, attr="negative_ts", empty_for_None=True)
            _check_list_entries(tech=tech, attr="negative_ts", entry_types=[str])
            assert all([isinstance(ts, str) for ts in self.get_info(tech=tech, attr="negative_ts")]),\
                f"If negative_ts for '{tech}' in InputDataInfo is given as a list, it must contain only str type formatted ts name entries."
            
            # sub_dataset_name
            if self.get_info(tech=tech, attr="sub_dataset_name") is None: 
                print(f"Note that sub_dataset_name for technology '{tech}' is None, will lead to errors if files with sub datasets are loaded.", flush=True) #TODO remove when sub dataset info is moved to metadata
            # elif '<COSTYEAR>' in self.get_info(tech=tech, attr="sub_dataset_name"): #TODO cost year is a scalar now with investment periods. Integrate replace() into getter?
            #     assert not cost_year is None, f"sub_dataset_name string contains <COSTYEAR> sub string but cost_year parameter is None. Pass cost_year as an integer"
            #     sub_dataset_name = sub_dataset_name.replace('<COSTYEAR>', str(cost_year))

            # daily_timeseries
            _make_list(tech=tech, attr="daily_timeseries", empty_for_None=False)
            _check_list_entries(tech=tech, attr="daily_timeseries", entry_types=[str, None])
            _make_list(tech=tech, attr="hourly_reference_timeseries", empty_for_None=False)
            _check_list_entries(tech=tech, attr="hourly_reference_timeseries", entry_types=[str, None])
            if self.get_info(tech=tech, attr="daily_timeseries") == [None]:
                assert self.get_info(tech=tech, attr="hourly_reference_timeseries") == [None], \
                    f"hourly_reference_timeseries has no effect if no daily_timeseries was given. Check inputs."
            assert len(self.get_info(tech=tech, attr="hourly_reference_timeseries"))==len(self.get_info(tech=tech, attr="daily_timeseries")),\
                f"The number of timeseries names in daily_timeseries and hourly_reference_timeseries for technology '{tech}' in InputDataInfo must be the same."

    # FUTURE IDEA:
    # add all input data information for ALL technologies/component to this file, not only volatile potentials
    # if new custom attribute values are given, overwrite self.data (but never self.original_data, this allows fallback for sensitivities applicable to core value)
    # introduce 3 categories: timeseries, capacity, balance
    # all of the categories get one data path and one metadata path (or None if irrelevant)
    # # ONE actual data path is defined by using predefined <SPACERS> - no more special cases for GlobEP structures etc
    # metadata contains info on data units, attribute names etc and depend on file type of data (can also contain info for multiple attributes)
    # standard terms are defined for specific terms:
        # 'capacity' describes the maximum unit/time, 
        # 'annual_balance' describes the aggregated value per year, 
        # 'timeseries_abs' describes the absolute values timeseries, 
        # 'timeseries_cf' describes the relative timeseries 0-1.0 etc.)
        # 'unit_cost' describes any unit cost like LCOE etc
        # 'specific_capex' 
    # all those attributes CAN but do not NEED to be contained in a dataset, but at least one is mandatory
    # every key must then have a sub key for 'data_unit'
    # aggregation info is then contained in InputDataInfo per technology, will be applied via standardized metadata terms (see above)
    # a fallback or error should be implemented if one data element is not available, e.g. "unit_cost" shall be weighted by "capacity" and 'capacity' is not in metadata

#%%

# MODEL PATHS
    
class ModelPaths(metaclass=SingletonMeta):
    """
    This singleton holds information on paths used in the model.
    """
    def __init__(self, base_folder, techno_economic_data_fp, preprocessed_folder=None, default_paths_fp=None,intermediates_folder:str|None=None):
        # set data folder as attribute
        self.data_folder = data_folder
        # first load default paths 
        self._load_default_paths(default_paths_fp=default_paths_fp)
        # now set all default paths args that are defined in default paths
        self._set_all_default_path_attrs()
        
        # process and set base_folder
        base_folder = self._preprocess_base_folder(base_folder)
        self._check_and_set_folderpath(attr_name="base_folder", folder=base_folder)

        # process and set techno-economic data filepath
        techno_economic_data_fp = self._preprocess_techno_economic_data_filepath(techno_economic_data_fp)
        self._check_and_set_filepath(attr_name="techno_economic_data_fp", file=techno_economic_data_fp)

        # set folderpath for preprocessed data, first default folder in repo data and then custom folder
        self._check_and_set_folderpath(attr_name="preprocessed_folder_default", folder=None) # will fall back on default path
        self._check_and_set_folderpath(attr_name="preprocessed_folder_custom", folder=preprocessed_folder)

        # set intermediate folder
        #intermediates_folder = self._intermediates_folder(intermediates_folder)
        intermediates_folder = self._intermediates_folder(intermediates_folder)
        self._check_and_set_folderpath(attr_name="intermediates_folder", folder=intermediates_folder)

    def _set_all_default_path_attrs(self):
        """
        Iterates over all attrs in default paths yaml and sets them as 
        attributes in the form of 'self.{mainkey}_{subkey}'.
        """
        assert hasattr(self, "default_paths"), \
            f"ModelPaths()._load_default_paths() must be executed before ModelPaths()._set_all_default_path_attrs"
        known_data = ["countries", "default_regions", "eezs"]
        known_attrs = ["filepath", "attribute"]
        filepath_attrs=["filepath"]
        #TODO assert that required combinations of data type and attribute are given (e.g. 'default_regions' + 'filepath' etc.)
        for _key, _nesteddict in self.default_paths.items():
            _attrs = list()
            assert _key in known_data, \
                f"key '{_key}' of default_paths_fp is not in known data: {', '.join(known_data)}"
            for _attr, _val in _nesteddict.items():
                assert _attr in known_attrs, \
                    f"Attribute '{_attr}' for key '{_key}' of default_paths_fp is not in known attributes: {', '.join(known_attrs)}"
                if isinstance(_val, str):
                    # may be a filepatha and contain spacers
                    _val = _val.replace("DATAFOLDER", data_folder)
                if _attr in filepath_attrs and not os.path.isfile(_val):
                    # must be an existing file
                    raise FileNotFoundError(f"{_attr}' for '{_key}' data is not an existing file: {_val}")
                self.set_attribute(attr_name=f"{_key}_{_attr}", attr_value=_val)
            print(f"Attributes loaded from default paths for '{_key}': {', '.join(_attrs)}")


    def set_attribute(self, attr_name, attr_value):
        """This method allows to set attributes forcefully after initialization, e.g. for deviating test paths."""
        if not isinstance(attr_name, str):
            raise TypeError("attr_name must be str type.")
        setattr(self, attr_name, attr_value)


    def _check_and_set_folderpath(self, attr_name, folder):
        """Checks existence of a folder, replaces by default and inserts datafolder path if needed/possible and sets as new singleton attribute."""
        folder = self._process_path_argument(attr_name=attr_name, value=folder)
        # raise error if folder is not None but not exists
        if not (folder is None or os.path.isdir(folder)):
            if os.path.isdir(os.path.dirname(folder)):
                print(f"Given base folder: {folder} does not exist, while {os.path.dirname(folder)} exists. Folder is created!",flush=True)
                os.makedirs(folder)
            else:
                raise FileNotFoundError(f"{attr_name} folder does not exist: {folder}")
        setattr(self, attr_name, folder)
    
    
    def _check_and_set_filepath(self, attr_name, file):
        """Checks existence of a file, replaces by default and inserts datafolder path if needed/possible and sets as new singleton attribute."""
        file = self._process_path_argument(attr_name=attr_name, value=file)
        # raise error if folder is not None but not exists
        if not (file is None or os.path.isfile(file)):
            raise FileNotFoundError(f"{attr_name} file does not exist: {file}")
        setattr(self, attr_name, file)
    

    def _preprocess_base_folder(self, base_folder):
        """Generates a default base folder path parallel to git repo if base folder is not given."""
        # model base folder
        if base_folder == 'default':
            # if the default location is selected, e.g. for testing, the data will be saved in 
            # a folder at the same level as the git repository, so that the data will not be
            # pushed or confuse the main repo
            base_folder = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                'model_intermediate_data'
            )
            # create if the generic base folder does not exist
            if not os.path.isdir(base_folder):
                # only the last folder level can possibly not exist, all others are repo paths
                os.mkdir(base_folder)
        return base_folder
    

    def _preprocess_techno_economic_data_filepath(self, techno_economic_data_fp):
        """Generates a default path to techno-economic data yaml if not given."""
        if techno_economic_data_fp is None:
            # generate filepath to default file in repo datafolder
            techno_economic_data_fp = os.path.join(
                data_folder,
                "technoeconomic_params.csv")
        return techno_economic_data_fp
    
    def _intermediates_folder(self, intermediates_folder):
        '''Gets intermediates folder or generates it, if not existent'''
        if not intermediates_folder is None:
            assert isinstance(intermediates_folder, str), "Intermediates folder needs to be a str!"
            if not os.path.exists(intermediates_folder):
                print(f"Intermediates folder {intermediates_folder} does not exist, will be created.")
                os.makedirs(intermediates_folder)
        return intermediates_folder
            
    def _process_path_argument(self, attr_name, value):
        """
        Replaces None inputs by default paths if available. If paths contain "DATAFOLDER" substring, 
        it will be replaced by the actual data_folder path in modelBuilder repo.
        """
        if value is None:
            try:
                value = self.default_paths[attr_name]["shapefile"]
            except:
                return None
        return value.replace("DATAFOLDER", data_folder)
    

    def _process_attr_argument(self, attr_name, value):
        """
        Gives back attribute of default_general_paths 
        """
        if value is None:
            try:
                value = self.default_paths[attr_name]["attribute"]
            except:
                return None
        return value

    def _load_default_paths(self,default_paths_fp):
        """Loads default paths as attribute."""
        if not default_paths_fp:
            default_paths_fp = os.path.join(self.data_folder, "default_general_paths.yml")
        if not os.path.isfile(default_paths_fp):
            raise FileNotFoundError(f"default_paths_fp does not exist: {default_paths_fp}")
        if not os.path.splitext(default_paths_fp)[-1] in [".yml",".yaml"]:
            raise TypeError(f"default_paths_fp extension must be .yaml, or .yml. Here {default_paths_fp}")
        with open(default_paths_fp) as fp:
            self.default_paths = yaml.load(fp, Loader=yaml.FullLoader) #TODO remove general data here and in yaml when Singleton introduced everywhere
    

#%%

# MODEL LOCATIONS

class ModelLocations(metaclass=SingletonMeta):
    """
    This singleton holds all information on the model locations including geometry data,
    and also contains data for the related default regions and countries.

    """

    def __init__(self, location_df, locationID_attr, srs,check_default_location_geometries=False):

        # first process the location dataframe with location IDs and geometries
        self.location_df = self._prepare_location_dataframe(                        # ModelLocations().location_df was self.location_shape before #TODO delete comment when all self.location_shape replaced
            shapefile=location_df, srs=srs, locationID_attr=locationID_attr,
        )
        self.locationIDs = set(natsort.natsorted(list(self.location_df[locationID_attr]))) # ModelLocations().locationIDs was self.locations before #TODO delete comment when all self.locations replaced
        self.locationID_attr = locationID_attr # ModelLocations().locationID_attr was self.locationID_column before #TODO delete comment when all self.locationID_column replaced
        
        # if base folder is given in ModelPaths, save location_df to file
        self._save_regions()
        # load a mapper with default countries and regions
        self._load_default_regions()

        print("ModelLocations singleton instance created.")

    ###########################
    # REGION-SPECIFIC METHODS #
    ###########################
        
    def get_default_regions_mapper(self)->dict:
        """Returns a dict with model locations as keys and lists of overlapping default regions as values."""
        # first get default regions info
        default_regions_info = self.get_default_regions_info()
        # return only location names as keys and list of default regions as values
        return {loc:list(default_regions_info[loc].keys()) for loc in default_regions_info.keys()}
 

    def get_normalized_location_overlap_shares_with_default_region(self)->dict:
        """Returns a dict with model locations as keys and nested dicts with default regions as sub keys and share of location area covered by the respective default region (always normalized to 100%) as value."""
        # first get default regions info
        default_regions_info = self.get_default_regions_info()
        # return only location names as keys and nested dicts of default regions and their region shares 
        return {loc:{def_reg:default_regions_info[loc][def_reg]["overlap_share_of_location_area_normalized"] for def_reg in default_regions_info[loc].keys()} for loc in default_regions_info.keys()}


    def get_default_region_overlap_shares_with_location(self)->dict:
        """Returns a dict with model locations as keys and nested dicts with default regions as sub keys and share of default region area covered by the location as value."""
        # first get default regions info
        default_regions_info = self.get_default_regions_info()
        # return only location names as keys and nested dicts of default regions and their overlap shares with location 
        return {loc:{def_reg:default_regions_info[loc][def_reg]["overlap_share_of_default_region_area"] for def_reg in default_regions_info[loc].keys()} for loc in default_regions_info.keys()}


    def get_default_regions_info(self):
        """
        Returns a dict with model locations as keys, overlapping default regions as sub keys and a nested 
        dict with keys and values for overlap share of location area and default region area
        """
        # prepare the dict if not existent
        if not hasattr(self, "default_regions_info"):
            self.get_agg_regions_by_geometry(upper_threshold=0.999, lower_threshold=0.001)
        return self.default_regions_info
    
    ############################
    # COUNTRY-SPECIFIC METHODS #
    ############################

    def get_country_df(self):
        """Returns the country dataframe with country geometries."""
        if not hasattr(self, "country_df"):
            self._load_country_df()
        return self.country_df


    def get_country_mapper(self):   
        """Returns a mapper with all model locationIDs as keys and lists of overlapping countries as values."""
        if not hasattr(self, "country_mapper"):
            self._define_country_mapper()
        return self.country_mapper


    def get_main_country(self, loc=None):
        '''
        returns , or a single gid0 is only for one loc
        loc: str
            region that is part of locationIDs
        
        returns:
            - main_countries_df containing locationID column and main_gid0 column
            - or a single gid0 if loc is given
        '''

        def _getter(loc):
            if loc=="world":
                return "world"
            _country_df = self.get_country_df()
            geom = self.location_df.loc[self.location_df["locationID"]==loc,"geom"].iloc[0]
            country_overlap_normalized_df   = pd.DataFrame()
            country_overlap_normalized_df['GID_0']      = _country_df["GID_0"]
            country_overlap_normalized_df['overlap']    = _country_df.geom.apply(lambda x:x.Intersection(geom).Area()/geom.Area())
            country_overlap_normalized_df.reset_index(inplace=True,drop=True)
            max_value_row = country_overlap_normalized_df.loc[country_overlap_normalized_df['overlap'].idxmax()]
            return max_value_row['GID_0']
        if loc:
            return _getter(loc)
        else:
            main_countries_df = pd.DataFrame(index=self.location_df.locationID,columns=['main_gid0'])
            for loc in self.locationIDs:
                main_countries_df.loc[loc,'main_gid0']= _getter(loc)
            return main_countries_df

    
    def get_overlapping_countries_df_per_location(self, locationID):
        """Returns a subset of the country dataframe with those countries that overlap the given model location geometry."""
        _countries = self.get_overlapping_countryIDs(locationID)
        return self.country_df[self.country_df[ModelPaths().countries_attribute].isin(_countries)]

    def get_overlapping_countryIDs_per_location(self, locationID):
        """Returns a list of all overlapping countries for a given model location."""
        # make sure model location ID is valid
        if not locationID in self.locationIDs:
            raise ValueError(f"locationID '{locationID}' is not in model locationIDs. Select from: {', '.join(self.locationIDs)}")
        return self.get_country_mapper()[locationID]

    #####################
    # REGION PROCESSING #
    #####################
    
    def _prepare_location_dataframe(self, shapefile, srs, locationID_attr , _max_str_length = 245):
        """Processes input shapefile path or dataframe and returns geospatial dataframe in a format as expected for location_df attribute."""
        if isinstance(shapefile, pd.core.frame.DataFrame):
            # a dataframe is passed, make sure inputs is OK, convert to given srs and process to expected format
            assert (
                "geom" in shapefile.columns
            ), f"If a pd.Dataframe is passed as shapefile, it must be a geospatial dataframe and contain a 'geom' column with geospatial data."
            assert (
                not "geometry" in shapefile.columns
            ), f"If dataframe is not passed as gpd.DataFrame, it must not have a geometry column (will be needed later)."
            assert all(
                [isinstance(g, osgeo.ogr.Geometry) for g in shapefile.geom]
            ), f"At least one entry in 'geom' column of shapefile is not of type osgeo.ogr.Geometry."
            # create a copy and preprocess class attribute
            location_df = shapefile.copy()
            location_df["geom"] = location_df["geom"].apply(lambda x: gk.geom.transform(x, toSRS=srs))
            location_df["locationID"] = location_df[locationID_attr]
        elif isinstance(shapefile, gpd.GeoDataFrame):
            assert all(
                [
                    isinstance(g, (shapely.geometry.polygon.Polygon, shapely.geometry.multipolygon.MultiPolygon))
                    for g in shapefile.geom
                ]
            ), f"At least one entry in 'geometry' column of shapefile is not of type shapely.geometry.polygon.Polygon or shapely.geometry.multipolygon.MultiPolygon."
            assert not shapefile.crs is None, f"gpd.GeoDataFrame must have a crs assigned!"
            # create a copy and preprocess class attribute
            location_df = shapefile.copy()
            location_df = location_df.to_crs(srs)
            location_df = location_df.rename(columns={locationID_attr: "locationID"}, inplace=True)
        elif isinstance(shapefile, str):
            # if a shp path is passed, load dataframe via geopandas first (osgeo geometries will be added later)
            assert os.path.isfile(
                shapefile
            ), f"shapefile parameter was given as str, filepath is assumed but file does not exist: {shapefile}"
            location_df = (
                gpd.read_file(shapefile)
                .to_crs(srs)[[locationID_attr, "geometry"]]
                .rename(columns={locationID_attr: "locationID"})
            )
        else:
            raise OSError(
                f"'shapefile' parameter must be given as str filepath, as pd.core.frame.DataFrame with osgeo.ogr.Geometry geom column or as gpd.GeoDataFrame"
            )

        # ensure that both osgeo geometries (for modelBuilder processing) and shapely geometries (for FINE output processing) are in the dataframe, else add
        if not "geom" in location_df.columns:
            srs = osgeo.osr.SpatialReference()
            srs.ImportFromEPSG(srs)  # TODO this was 3857 but why? changed to 4326 as standard default
            location_df["geom"] = location_df.geometry.apply(lambda x: gk.geom.convertWKT(x.wkt, srs=srs))
        if not "geometry" in location_df.columns:
            location_df["geometry"] = gpd.GeoSeries.from_wkt(
                location_df["geom"].apply(lambda x: x.ExportToWkt())
            )
            location_df = gpd.GeoDataFrame(location_df, geometry="geometry", crs=srs)
            location_df = location_df.to_crs(srs)

        #make better names        
        # make sure regions are cut of properly (FINE has a max length (for agg and for saving to nc4!)):
        # also there is some information loss, but the information were lost already, so at this point whatever
        # but otherwise it can happen that some regions potentials are not loaded properly..
        # they will then be loaded as custom region, but better than triggering wrong loading!
        long_names = location_df["locationID"].str.len() > _max_str_length
        location_df["locationID"][long_names] = \
            location_df["locationID"][long_names].apply(lambda x: x[:_max_str_length] + "cutoff")

        return location_df
    

    def get_agg_regions_by_geometry(self, upper_threshold=0.999, lower_threshold=0.001):
        """
        Assesses the dflt_type status of the model locations via a geometrical
        match of the location geom with default regions. Yields (a) the 
        'dflt_type' column of self.location_df, (b) self.default_region_info_dict
        with the default regions that are contained in the default/agg 
        region including their overlap shares.
        NOTE: Assumes default type from 'dflt_share' column if self.location_df 
        contains this attribute.
        
        upper_threshold : float, optional
            The area share that must overlap with the default region to 
            consider the default region as fully part of the location 
            geometry. By default 0.999.
        lower_threshold : float, optional
            The share of the default region overlapped by the model location 
            that is negligible, i.e. under the default region will not be 
            considered under this area share. By default 0.01
        """
        assert hasattr(self, "location_df"), f"location_df must be loaded as attribute to assign dflt_type attribute to model locations"
        assert 0<upper_threshold<=1, f"upper_threshold must be 0 < upper_threshold<= 1.0"
        assert 0<=lower_threshold<1, f"lower_threshold must be 0 <= lower_threshold < 1.0"

        # initialize the default_regions_info_dict - this dict contains only fully contained def regs
        self.default_regions_info = dict()
        # initialize the ABC - this dict contains ALL default regions that are fully or only partly overlapping with the model loc
        self.ABC_dict = dict()

        # default type could have been preprocessed
        if 'dflt_type' in self.location_df.columns:
            assert all([x in ['agg', 'default', 'custom'] for x in self.location_df.dflt_type]),\
            f"self.location_df.dflt_type must only contain 'agg', 'default' or 'custom'."
            # get default region codes for later comparison
            all_default_regioncodes = utils.get_raw_regions(shore_type=None, country_list=[])

        # iterate over all model locations and extract dflt_type
        for i, (locgeom, locname) in enumerate(zip(self.location_df.geom, self.location_df.locationID)):
            # we can save time if we know the dflt_type already to be a "default" location
            if 'dflt_type' in self.location_df.columns and self.location_df.dflt_type.iloc[i]=="default": 
                # make sure it's actually a default region, at least via string name
                assert locname in all_default_regioncodes, \
                    f"locationID '{locname}' is flagged as default region but locationID str does not match any default region code."
                # set all shares to 1.0 for default regions
                self.default_regions_info.setdefault(locname, {}).setdefault(locname, {}).setdefault("overlap_share_of_default_region_area", 1.0)
                self.default_regions_info.setdefault(locname, {}).setdefault(locname, {}).setdefault("overlap_share_of_location_area", 1.0)
                self.default_regions_info.setdefault(locname, {}).setdefault(locname, {}).setdefault("overlap_share_of_location_area_normalized", 1.0)
                continue

            # get Area of location
            _locarea = locgeom.Area()
            # extract all overlapping def regions
            _possdefregs_df = gk.vector.extractFeatures(
                ModelPaths().default_regions_filepath, 
                geom=locgeom, 
                spatialPredicate="Overlaps", 
                srs=locgeom.GetSpatialReference()
            )
            assert ModelPaths().default_regions_attribute in _possdefregs_df,\
                f"default_regions_attribute '{ModelPaths().default_regions_attribute}' is not an attribute of default_regions_shp: {ModelPaths().default_regions_shp}"
            _possdefregs_df.rename(columns={ModelPaths().default_regions_attribute : "locationID"}, inplace=True)

            if len(_possdefregs_df)==0:
                # we have no overlap with any def reg, set empty dict for def regs as spacer
                self.default_regions_info[locname] = dict()
                continue


            # add up overlapped area shares of location area
            _cumshares = 0
            for defgeom, defregcode in zip(_possdefregs_df.geom, _possdefregs_df.locationID):           
                assert defgeom.GetSpatialReference().IsSame(locgeom.GetSpatialReference()) # make sure
                _inters = defgeom.Intersection(locgeom)
                _defarea = defgeom.Area()
                _intersarea = _inters.Area()
                if _intersarea/_defarea<lower_threshold:
                    # the overlap is only a geospatial mismatch, skip this default region
                    continue

                # set the share of the default geometry which is overlapped by the respective location geom
                self.default_regions_info.setdefault(locname, {}).setdefault(defregcode, {}).setdefault("overlap_share_of_default_region_area", _intersarea/_defarea)
                # set the share of the location area which is overlapped by the respective default geometry
                self.default_regions_info.setdefault(locname, {}).setdefault(defregcode, {}).setdefault("overlap_share_of_location_area", _intersarea/_locarea)            
                # add the overlapped location share to collector
                _cumshares += _intersarea/_locarea
            
            if len(self.default_regions_info)==0:
                # no potential default regions (may have had overlaps but below area overlap threshold)
                # set dummy dict for no default regs and continue with next loc
                self.default_regions_info[locname] = dict()
                continue


            # iterate over def regs again to normalize the overlapped location share now that all def regs have been processed 
            # NOTE: The shares must not add up to 1.0 when e.g. offshore areas are contained, but when e.g. values per area are weighed, the total area must cumulate to 100%
            for defregcode in self.default_regions_info[locname].keys():
                self.default_regions_info.setdefault(locname, {}).setdefault(defregcode, {}).setdefault("overlap_share_of_location_area_normalized", self.default_regions_info[locname][defregcode]["overlap_share_of_location_area"]/_cumshares)

        # now get the agg type if required
        if not 'dflt_type' in self.location_df.columns:
            # initialize default type collector
            dflt_types = []
            for locname in self.location_df.locationID:
                if len(self.default_regions_info[locname]) == 0:
                    # we have no default region overlap -> custom region
                    dflt_types.append("custom")
                else:
                    # get the shares 
                    _defregs = list(self.default_regions_info[locname].keys())
                    _shares = [self.default_regions_info[locname][_r]["overlap_share_of_default_region_area"] for _r in _defregs]
                    if all([_shr > upper_threshold for _shr in _shares]):
                        # we only have fully covered regions
                        if len(_defregs)==1:
                            # only one default region -> default location
                            dflt_types.append("default")
                        else:
                            # multiple default regions -> agg location
                            dflt_types.append("agg")
                    else:
                        # we have one or more default regions that overlap only partially!
                        dflt_types.append("custom")

            # add agg status as attr to location_df
            self.location_df["dflt_type"] = dflt_types

        return

    def _save_regions(self):
        """Save the preprocessed location_df to file if base folder is given."""
        try:
            ModelPaths._lock.release()
            base_folder = ModelPaths().base_folder
            ModelPaths._lock.acquire()
        except:
            # ModelPaths is not initialzed, no base folder available
            return
        # if model_base_folder is set, save regions
        if base_folder is not None:
            output = os.path.join(base_folder, "spatial_data", "regions.shp")
            os.makedirs(os.path.dirname(output), exist_ok=True)
            if isinstance(self.location_df, gpd.GeoDataFrame):
                gk.vector.createVector(
                    pd.DataFrame(self.location_df).drop(columns=['geometry']),
                    output=output,
                )
            else:
                gk.vector.createVector(
                    self.location_df,
                    output=output,
                )
        return
    
    def _merge_location_geoms(self):
        """Generates a merged geometry of all location geoms and sets as attribute."""
        for i, g in enumerate(self.location_df.geom):
            if i == 0:
                _merged = g
            else:
                _merged = _merged.Union(g)
        self.merged_location_geom = _merged
    

    def _extract_defaultregions_per_location(self):
        """
        This auxiliary function extracts the GID_1split region names
        from the location shape dataframe and returns them as a
        dictionary of the relevant GID_1split codes per each location name.
        """
        # # if we have only default regions, save time and return a mapper with each model location as key and single value respectively # TODO:delete block
        # if (self.location_df.dflt_type=="default").all():
        #     return dict(zip(self.location_df.locationID, [[x] for x in self.location_df.locationID]))

        # else load the vector from disk once to save time later on
        all_default_regions_vec = gk.vector.loadVector(ModelPaths().default_regions_filepath) 
        # initialize a dictionary to hold the default region names per location
        default_region_info_dict = {}
        # iterate over locations and extract overlapping gid1splits
        def _check_default_loc_name(loc_name):
            _start=time.time()
            try:
                temp = gk.vector.extractFeatures(all_default_regions_vec, where=f"{ModelPaths().default_regions_attribute}='{loc_name}'")
                assert len(temp)==1
            except:
                raise ValueError(f"Excactly one default region expected for {ModelPaths().default_regions_attribute} = '{loc_name}' in {ModelPaths().default_regions_filepath}!")
            print(f"checking default region time took{time.time()-_start}!")

        for loc_name, loc_geom in zip(self.location_df["locationID"], self.location_df.geom):
            
            if self.location_df[self.location_df["locationID"]==loc_name].dflt_type.iloc[0] == "custom":
                # filter only for truly overlapping default regions
                overlapping_default_regions_df = gk.vector.extractFeatures(
                    source=all_default_regions_vec, 
                    geom=loc_geom, 
                    srs=loc_geom.GetSpatialReference(), 
                    spatialPredicate="Overlaps",
                )

                assert (
                    len(overlapping_default_regions_df) > 0
                ), f"Not a single default region could be found for location {loc_name}!"
                # add location share that overlaps with th respected default region 
                overlapping_default_regions_df["overlap_area"]= overlapping_default_regions_df.geom.apply(lambda x:x.Intersection(loc_geom).Area())
                location_overlap_shares=(overlapping_default_regions_df.overlap_area/loc_geom.Area()).to_list()
                # add share of default region covered by location shape 
                default_region_overlap_shares=overlapping_default_regions_df.overlap_area/overlapping_default_regions_df.geom.apply(lambda x:x.Area()).to_list()

                overlapping_default_regions = overlapping_default_regions_df[ModelPaths().default_regions_attribute].to_list()

            #we have an aggregated region
            elif self.location_df[self.location_df["locationID"]==loc_name].dflt_type.iloc[0] == "agg":
                overlapping_default_regions = loc_name.split("__")      # David fragen, wenn agg regions zu lang über identifier gehen, hinten schneiden und damit hantieren, damit die nicht einfach fehlen
                overlapping_default_regions_df = gk.vector.extractFeatures(
                    source=all_default_regions_vec, 
                    srs=loc_geom.GetSpatialReference(),
                    where=f"{ModelPaths().default_region_attribute} in ('"+"','".join(overlapping_default_regions)+"')", 
                )
                assert all([reg in overlapping_default_regions_df[ModelPaths().default_regions_attribute].to_list() for reg in overlapping_default_regions]),\
                    f"Not all default regions ({overlapping_default_regions}) could be extracted from {ModelPaths().default_regions_attribute} column in {ModelPaths().default_regions_filepath}!"
                # 
                overlapping_default_regions_df["overlap_area"]= overlapping_default_regions_df.geom.apply(lambda x:x.Intersection(loc_geom).Area())
                location_overlap_shares=overlapping_default_regions_df.overlap_area/loc_geom.Area().to_list()
                default_region_overlap_shares = [1.0]*len(overlapping_default_regions)

            #we have a default region
            elif self.location_df[self.location_df["locationID"]==loc_name].dflt_type.iloc[0] == "default":
                _check_default_loc_name(loc_name)
                overlapping_default_regions = [loc_name]
                location_overlap_shares=[1.0]
                default_region_overlap_shares=[1.0]
            else:
                raise ValueError("dflt.type must be in 'agg','default' and 'custom'!")

            # normalize the covered location shape to always 100% 
            location_overlap_shares_normalized=list(np.array(location_overlap_shares)/sum(location_overlap_shares))
            
            # add to dictionary with location name as key
            for i,def_reg in enumerate(overlapping_default_regions):
                default_region_info_dict.setdefault(loc_name,{}).setdefault(def_reg,{}).setdefault("overlap_share_of_location_area_normalized",location_overlap_shares_normalized[i])
                default_region_info_dict.setdefault(loc_name,{}).setdefault(def_reg,{}).setdefault("overlap_share_of_default_region_area",default_region_overlap_shares[i])

        # return a dictionary with location names as keys and all affected GID1splits as values
        return default_region_info_dict


    def _load_default_regions(self, shore_type=None, country_list=[])->list:
        """_summary_

        Args:
            shore_type (str, optional): 
            'Onshore' or 'Offshore'. If None passed, both will be returned. Defaults to None:str.

        Returns:
            list: List with str default region codes for the input selection.
        """
        if not shore_type is None:
            assert isinstance(shore_type, str), f"shore_type must be a str."
            assert shore_type.upper() in ['ONSHORE','OFFSHORE'], f"shore_type must be either 'Onshore' or 'Offshore' (case-insensitive)."
            shore_type_list=[shore_type.upper()]
        else:
            shore_type_list=['ONSHORE','OFFSHORE']
        # load the raw default region code dict
        with open(os.path.join(data_folder, 'default_regions.json')) as json_file:
            all_default_regioncodes_dict = json.load(json_file)
        # check or set country list
        available_countries=[]
        for shore in shore_type_list:
            available_countries+=all_default_regioncodes_dict[shore].keys()
        if country_list == []:
            country_list=available_countries
        else:
            assert isinstance(country_list, list), f"country_list must be a list."
            assert all([country in available_countries for country in country_list]), f"All countries in country_list must be in: {', '.join(available_countries)}"
        # extract the raw gid1split codes for the given params
        all_default_regioncodes=[]
        for k1 in all_default_regioncodes_dict.keys():
            # skip if shore type is not desired
            if not k1.upper() in shore_type_list:
                continue
            for k2 in all_default_regioncodes_dict[k1].keys():
                # skip if country is not desired
                if not k2.upper() in country_list:
                    continue
                # else add all region codes to all_default_regioncodes
                all_default_regioncodes+=all_default_regioncodes_dict[k1][k2]

        return all_default_regioncodes
    
    ######################
    # COUNTRY PROCESSING #
    ######################

    def _define_country_mapper(self):
        """Creates country_mapper attribute as a dict with all model loction IDs as keys and lists of overlapping country IDs as values."""
        if not hasattr(self, "country_df"):
            self._load_country_df()
        # generate a vector with the model countries
        country_vec = gk.vector.createVector(self.country_df)
        country_mapper = dict()
        for locationID, location_geom in zip(self.location_df.locationID, self.location_df.geom):
            # extract all touching country geoms #TODO implement fallback for offshore regions - via EEZ?
            _overlapping = gk.vector.extractFeatures(
                country_vec,
                geom = location_geom,
                spatialPredicate="Overlaps",
            )
            if len(_overlapping) == 0:
                raise LookupError(f"No overlapping country geometries found for location '{locationID}'.")
            # set list of countries as value of countrymapper
            _countries = _overlapping[ModelPaths().countries_attribute].to_list()
            country_mapper[locationID] = _countries
        self.country_mapper=country_mapper


    def _load_country_df(self):
        """"Loads country shapefile as an attribute df."""
        # try to extract valid country shapefile path from ModelPaths singleton
        if not ModelPaths.exists():
            raise ModuleNotFoundError(f"ModelPaths singleton instance must be initialized to load country df.")
        
        fps = dict()
        for fp in ["countries_filepath", "eezs_filepath"]:
            try:
                fps[fp] = getattr(ModelPaths(), fp)
            except:
                AttributeError(f"'{fp}' attribute could not be extracted from ModelPaths() singleton.")
                
            if fps[fp] is None:
                raise ValueError(f"ModelPaths().{fp} must not be None to load country df.")
            if not isinstance(fps[fp], str) and fps[fp][-4:]==".shp":
                raise TypeError(f"{fp} must be a filepath to a .shp file.")
            if not os.path.isfile(fps[fp]):
                raise FileNotFoundError(f"{fp} must be an existing shapefile: {fps[fp]}")
        
        for attr in ["countries_attribute", "eezs_attribute"]:
            # make sure we have a ModelPaths().countries_attribute/eezs_attribute
            if getattr(ModelPaths(), attr) is None:
                raise ValueError(f"No ModelPaths().{attr} given. Required to load country df.")
            
        def _merge_subnational_geoms(_df, _gid0_attr):
            """Merges all subnational geometries for every unique GID_0 value and returns a dataframe with GID_0 codes and geoms."""
            assert _gid0_attr in _df.columns, \
                f"_gid0_attr '{_gid0_attr}' is not in _df. Available columns: {', '.join(_df.columns)}"
            
            def _merge_geoms(ser):
                """merges all geoms in an iterable"""
                for i, g in enumerate(ser):
                    if i==0:
                        _merged = g
                        _srs = g.GetSpatialReference()
                    else:
                        # make sure all srs in series are the same
                        assert g.GetSpatialReference().IsSame(_srs), f"srs mismatch in aggregation series"
                        _merged = _merged.Union(g)
                return _merged
            
            agg_funcs={
                "geom" : _merge_geoms, 
                _gid0_attr : "first", # get the first (unique) GID_0 code
            }
            return _df[list(agg_funcs.keys())].groupby(_gid0_attr).agg(agg_funcs)
        
        # load the country df geospatially based on (merged) location geoms
        if not hasattr(self, "merged_location_geom"):
            self._merge_location_geoms()
        country_onshore_df = gk.vector.extractFeatures(
                source = fps["countries_filepath"],
                geom = self.merged_location_geom,
                spatialPredicate="Overlaps",
            )
        country_onshore_df["shore_type"]="onshore"
        # make sure ModelPaths().countries_attribute is valid
        if len(country_onshore_df)>0 and not ModelPaths().countries_attribute in country_onshore_df.columns:
            raise AttributeError(f"ModelPaths().countries_attribute '{ModelPaths().countries_attribute}' is not an attribute of the onshore country shapefile: {fps['countries_filepath']}. Select from: {', '.join(country_onshore_df.columns)}")
        # onshore countries must be unique
        if len(country_onshore_df)>0 and not len(set(country_onshore_df[ModelPaths().countries_attribute]))==len(country_onshore_df[ModelPaths().countries_attribute]):
            # country values are not unique within the geometries overlapping with the model locations, merge all subnational geometries
            country_onshore_df = _merge_subnational_geoms(country_onshore_df, _gid0_attr=ModelPaths().countries_attribute)


        country_offshore_df = gk.vector.extractFeatures(
                source = fps["eezs_filepath"],
                geom = self.merged_location_geom,
                spatialPredicate="Overlaps",
            )
        country_offshore_df["shore_type"]="offshore"
        # make sure ModelPaths().eezs_attribute is valid
        if len(country_offshore_df)>0 and not ModelPaths().eezs_attribute in country_offshore_df.columns:
            raise AttributeError(f"ModelPaths().eezs_attribute '{ModelPaths().eezs_attribute}' is not an attribute of the offshore country shapefile: {fps['eezs_filepath']}. Select from: {', '.join(country_offshore_df.columns)}")
        # offshore countries must be unique
        if len(country_offshore_df)>0 and not len(set(country_offshore_df[ModelPaths().eezs_attribute]))==len(country_offshore_df[ModelPaths().eezs_attribute]):
            # country values are not unique within the geometries overlapping with the model locations, merge all subnational geometries
            country_offshore_df = _merge_subnational_geoms(country_offshore_df, _gid0_attr=ModelPaths().eezs_attribute)
        # rename the relevant offshore columns to onshore "countries" version
        if len(country_offshore_df) > 0:
            country_offshore_df.rename(columns={ModelPaths().eezs_attribute : ModelPaths().countries_attribute}, inplace=True)
        # combine onshore and offshore countries
        country_df = pd.concat([country_onshore_df, country_offshore_df])
        
        # TODO discuss if we should add a check that all location geoms are actually covered
        
        # make sure we have actually data
        if len(country_df) == 0:
            raise ValueError(f"No countries extracted from countries_filepath attribute '{ModelPaths().countries_attribute}' in file: {fps['countries_filepath']} nor from attribute {ModelPaths().eezs_attribute} in file: {fps['eezs_filepath']}.")

        # set as attribute
        self.country_df = country_df

# Techno-economic data of model

class ModelTechnoEconomicData(metaclass=SingletonMeta):
    """
    This singleton holds techno economic data for all model processes.
    """

    def __init__(self):
        ted_path = ModelPaths.get_attribute('techno_economic_data_fp')
        data_format = os.path.splitext(ted_path)[-1]
        if data_format in [".yaml", ".yml"]:
            self.data = self._load_ted_yaml(ted_path)
        elif data_format == ".csv":
            self.ted_data_raw = self._load_ted_csv(ted_path)
            self.data = self._process_ted(self.ted_data_raw)
        else:
            raise NotImplementedError(
                f"Import of techno economic data from {data_format}"
                " files is currently not supported."
                )
        self.data.sort_index(inplace=True)
        self.esm_params = self._load_esm_data(ted_path)
    

    def has_data(
        self,
        component=None,
        attribute=None,
        region=None,
        year=None,
        ) -> bool:
        """
        Returns a boolean if the required combination of technology, 
        attribute and region exists.

        Parameter:
        component|str
        attribute|str
        region|str
        year|int, str

        Return: 
        boolean
        """
        args=[component, attribute, region, year]
        assert not all([x is None for x in args]),\
            f"Component, attribute, region,year must not all be None."
        for i, arg in enumerate(args):
            if arg is None:
                continue
            try:
                if i==0:
                    assert arg.lower() in self.data.index.get_level_values(i)
                else:
                    assert arg in self.data.index.get_level_values(i)
            except:
                return False
        return True


    def get_data(
        self,
        component, 
        attribute,
        single_location=None,
        stock=False,
        economicLifetime="",
        )->dict|pd.Series|str|float|int:
        '''
        Get data gives back the fine arguments as a dict from the techno-economic data sheet, for every attribute. Returned format depends on fine compability for attribute.
        
        Parameter:
        component|str
            Technology that we add to esm, i.e. wind_onshore
        attribute|str
            Desired fine attribute, i.e. investmentPerCapacity
            

        Return: 
        dict, series_with regions as key, series_with years as key, single value
        '''
        component = component.lower()
                        
        # set investment period names with stock or without stock investment periods
        investment_period_names = self._get_investment_period_names(attribute=attribute,stock=stock, economicLifetime=economicLifetime)

        # check if ted data 
        if len(self.data[component,attribute,:,:]) == 0:
            return np.nan

        if single_location:
            # set return_format
            return_format = self._set_return_format_conv(attribute)
            if return_format == "dict":
                return {ip:self._iterate_available_data(component, attribute, region=single_location, ip=ip) for ip in investment_period_names}
            else:
                return self._iterate_available_data(component, attribute, region=single_location,ip=InputDataInfo().base_year)

        # set return_format
        return_format = self._set_return_format(attribute)
        # create dict, series, or value
        if return_format == "dict":
            dictionary= {
                ip: pd.Series({
                    region: self._iterate_available_data(component, attribute, region, ip) 
                    for region in ModelLocations().locationIDs
                })
            for ip in investment_period_names
            }
            return dictionary
        # TODO: Adjust workflow for regions: substitute "constant" and "world" by first unique level
        elif return_format == "series_years":
            try:
                series={
                    ip: self._iterate_available_data(component, attribute, "world", ip)
                    for ip in investment_period_names    
                }
                return series
            except: # if commodityConversionFactors for non regionalized values
                print("singletons, line 1361, commodityConversionFactors given back from USA examplary for whole world", flush=True)
                series={
                    ip: self._iterate_available_data(component, attribute, "USA", ip)  # TODO: works because all countires have same values, but is dirty and has to be changed to a random country included in ted, or value should be world value in ted
                    for ip in investment_period_names    
                }
                return series
        elif return_format == "series_regions":
            series = {
                region: self._iterate_available_data(component, attribute, region, "constant")
                for region in ModelLocations().locationIDs    
            }
            return pd.Series(series)
        elif return_format == "value" or return_format== "commodityConversionFactors":
            value= self._iterate_available_data(component=component, attribute=attribute, region="world", ip="constant")
            return value
        else:
            raise ValueError("Error: When calling the get_data function, the return_format parameter can only be 'dict', 'series_years', 'series_regions', 'value', or 'commodityConversionFactors'.")
            
    ########################################################
    # AUXILIARY functions of ModelTechnoEconomicData class #
    ########################################################
    
    def _get_investment_period_names(self,attribute:str,stock:bool,economicLifetime:str)->list:
            '''
            returns investment period (ip) names list with, or without stock ips, depending if stock = True
            '''
            if stock and (attribute == 'opexPerCapacity' or attribute == 'investPerCapacity'):
                investment_period_names = InputDataInfo().investment_period_names.copy()
                interval = InputDataInfo().investment_period_interval
                base_year = InputDataInfo().base_year
                # extend by stock investement period name list
                for start in range(base_year, base_year-economicLifetime+interval, -interval):
                    end = start - interval
                    investment_period_names.append(end)
            else:
                investment_period_names = InputDataInfo().investment_period_names.copy()
            return investment_period_names

    def _set_return_format(self, attribute)->str:
        return_formats = ["dict","series_years","series_regions","value"]
        return_formats_spec = {
            'interestRate':return_formats[2],
            'economicLifetime':return_formats[2],
            'commodity':return_formats[3],
            'operationRateMin':return_formats[3],
            #storage
            'chargeEfficiency':return_formats[3],
            'dischargeEfficiency':return_formats[3],
            'cyclicLifetime':return_formats[3],
            'selfDischarge':return_formats[3],
            'chargeRate':return_formats[3],
            'dischargeRate':return_formats[3],
            'stateOfChargeMin':return_formats[3],
            'stateOfChargeMax':return_formats[3],
            'losses':return_formats[3],
            'commodityConversionFactors':return_formats[1], 
            'physicalUnit':return_formats[3], 
        }

        if attribute in return_formats_spec.keys():
            return return_formats_spec[attribute]  
        else:
            return return_formats[0]  


    def _set_return_format_conv(self, attribute)->str:
        return_formats = ["dict","value"]
        return_formats_spec = {
            'interestRate':return_formats[1],
            'economicLifetime':return_formats[1],
            'physicalUnit':return_formats[1],
        }

        if attribute in return_formats_spec.keys():
            return return_formats_spec[attribute]  
        else:
            return return_formats[0]


    def _load_esm_data(self, ted_path)->dict:
        esm_params_path = os.path.join(data_folder,"esm_params.yaml") # '/'.join(ted_path.split('/')[:-1] + ['esm_params.yaml'])
        esm_params = yaml.load(open(os.path.abspath(esm_params_path)), Loader=yaml.FullLoader)
        return esm_params
         
    def _load_ted_yaml(self, ted_path):
        ted_dict = yaml.load(open(os.path.abspath(ted_path)), Loader=yaml.FullLoader)
        miDict = {}
        for modelClass in ted_dict.values():
            for comp, attrs in modelClass.items():
                for attr, ips in attrs.items():
                    if attr == "commodityConversionFactors":
                        if isinstance(list(ips.keys())[0], int):
                            for ip, commods in ips.items():
                                for commod, value in commods.items():
                                    miDict[(comp, attr, None, ip, None, commod)] = value
                        else:
                            for commod, value in ips.items():
                                miDict[(comp, attr, None, None, None, commod)] = value
                    elif isinstance(ips, dict):
                        for ip, value in ips.items():
                            miDict[(comp, attr, None, ip, None, None)] = value
                    else:
                        value = ips
                        miDict[(comp, attr, None, None, None, None)] = value
                                
                
        ted_data = pd.Series(miDict)
        ted_data.index.names = ['component', 'attribute', 'region', 'investment_period', 'unit', 'conversion_commodity']
        return ted_data
    

    def _load_ted_csv(self, ted_path)->pd.DataFrame:
        ''' loads and returns raw ted from csv file as pd.DataFrame() '''
        def _read_csv(delimiter): 
            return pd.read_csv(
                ted_path,
                delimiter=delimiter,
                header=0, 
                usecols=["component", "attribute", "region", "investment_period", "unit", "conversion_commodity", "values"],
            )
        # Assure readability of csv as different delimiters are possible
        try:
            return _read_csv(delimiter=';')
        except:
            return _read_csv(delimiter=',')
    

    def _process_ted(self,ted_data_raw)->pd.Series: # TODO: delete dobble lines, as they are causing an error later
        ted_data = ted_data_raw.drop_duplicates()
        ted_data = self._convert_units(ted_data)
        ted_data = self._test_ted(ted_data)
        ted_data = self._addopexPerCapacity(ted_data)
        ted_data = self._commodityConversionDict(ted_data)
        ted_data_processed = ted_data
        return ted_data_processed
    
    def _addopexPerCapacity(self, ted_data):
        # df with all opexFix containing lines and df with all investPerCapacity containing lines
        ted_data_investPerCap = ted_data[ted_data["attribute"]=="investPerCapacity"]
        ted_data_opexFix = ted_data[ted_data["attribute"]=="opexFix"]
        ted_data_opexFix.reset_index(inplace=True, drop=True) 
        ted_data_investPerCap.reset_index(inplace=True, drop=True) 

        # test: opexFix should not be more regionalized then investment costs
        # for i in range(len(ted_data_opexFix)):
        #     component_opexFix = ted_data_opexFix.loc[i,"component"]
        #     region_opexFix =    ted_data_opexFix.loc[i,"region"]
        #     if not (region_opexFix == "world" or region_opexFix == "example"):
        #         for j in range(len(ted_data_investPerCap)):
        #                 component = True if ted_data_investPerCap.loc[j, "component"] ==component_opexFix else False
        #                 attribute = True if ted_data_investPerCap.loc[j, "attribute"] =="investPerCapacity" else False
        #                 region =    True if ted_data_investPerCap.loc[j, "region"]    ==region_opexFix else False
        #                 if component is True and attribute is True and region is True:
        #                     break
        #         if component is True and attribute is True and region is True:
        #             continue
        #         else:
        #             raise ValueError(f"opexFix entries in techno-economic data csv for {component_opexFix} can not be more regionalized that investmentPerCapacity!")
        
        # _start = time.time()
        # print("Start at", _start)
        # # first get all techs that actually do have an opexFix attribute
        # opex_fix_techs = sorted(set(ted_data[ted_data.attribute=="opexFix"].component))
        # assert list(ted_data.columns)==["component", "attribute", "region", "investment_period",  "conversion_commodity","values"]
        # add_rows=list()
        # for tech in opex_fix_techs:
        #     # get only the data for the current tech and make sure it has invest data
        #     _tmp = ted_data[ted_data.component==tech]
        #     assert 'investPerCapacity' in _tmp.attribute.to_list(), \
        #         f"Component '{tech}' does have 'opexFix' but not 'investPerCapacity' attribute."
        #     # make sure all opexFix region+year combinations do not exceed investPerCapacity detail level
        #     # LOGIC: every opexFix combination must be in investPerCapacity combination, or its single region and time values combined with general or example
        #     _invest_combs = list(zip(_tmp[_tmp.attribute=="investPerCapacity"].region, _tmp[_tmp.attribute=="investPerCapacity"].investment_period))
        #     # NOTE: It is not possible to match locationIDs at this level as we do not know in which country they are!
        #     _possible_combs = list()
        #     for _c in _invest_combs:
        #         _possible_combs.extend([_c, ("global", _c[1]), ("example", _c[1]), (_c[0], "constant"), (_c[0], "example")]) # TODO: Add continents when available
        #     _possible_combs.extend([("global", "constant"), ("global", "example"), ("example", "constant"), ("example", "example")])
        #     for _opex_comb in list(zip(_tmp[_tmp.attribute=="opexFix"].region, _tmp[_tmp.attribute=="opexFix"].investment_period)):
        #         assert _opex_comb in _possible_combs, \
        #             f"opexFix must not be higher resolved spatially or temporally than investPerCapacity. Here: component='{tech}', region={row.region}, investment_period={row.investment_period}." # TODO: Make opex with higher res than cap possible

        #     # iterate over all opexFix values and convert to opexPerCapacity
        #     for i, row in _tmp[_tmp.attribute=="investPerCapacity"].iterrows():
        #         # get the matching investment comb
        #         def _get_opex_region_year_combination(invest_region, invest_year):
        #             _actual_comb = (invest_region, invest_year)
        #             # prioritize year over region #TODO add priority of region over year when year is constant or example
        #             _fallback_combs = [_actual_comb, ("global", _actual_comb[1]), ("example", _actual_comb[1]), (_actual_comb[0], "constant"), (_actual_comb[0], "example")] # TODO: Add continents when available
        #             # iterate over the optional combinations, starting from the actual 
        #             for _invest_comb in _fallback_combs:
        #                 for _opex_comb in list(zip(_tmp[_tmp.attribute=="opexFix"].region, _tmp[_tmp.attribute=="opexFix"].investment_period)):
        #                     if _opex_comb == _invest_comb:
        #                         return _opex_comb[0], _opex_comb[1]
        #             raise ValueError(f"investPerCapacity region={row.region} + investment_period={row.investment_period} has no opexFix region+investment_period match for component '{tech}'.")
        #         _reg, _time = _get_opex_region_year_combination(row.region, row.investment_period)
        #         # calculate opex per capacity
        #         _opexPerCapacity = float(row["values"]) * float(_tmp[(_tmp.attribute=="opexFix")&(_tmp.region==_reg)&(_tmp.investment_period==_time)]["values"].iloc[0])
        #         # prepare additional dataframe row
        #         add_rows.append(pd.DataFrame(index=ted_data.columns, data=np.array([row.component, "opexPerCapacity", row.region, row.investment_period, None, _opexPerCapacity])).T)

        # # add new data to ted dataframe 
        # ted_data = pd.concat([ted_data]+add_rows, axis=0)
        # # remove the opexFix rows
        # ted_data = ted_data[ted_data.attribute!="opexFix"].reset_index(drop=True)
        # print("opexPerCapacity took:", time.time()-_start)
        # return ted_data

        # # iterate over all investPerCap lines in ted_data to add opexPerCapacity for every investPerCapacity
        # for i in range(len(ted_data_investPerCap)):
        #     # create intermediate df --> will contain new line with opexPerCapacity for every investPerCapacity entry 
        #     df = pd.DataFrame(columns=list(list(ted_data_investPerCap.columns)))
        #     # iterate over columns of ted_data: component, attribute, region, investment period, values
        #     for column in list(ted_data_investPerCap.columns):
        #         if column == "values":
        #             # filter until regions and investment periods
        #             ted_data_filter = ted_data[ted_data["component"]==ted_data_investPerCap.loc[i,"component"]]
        #             ted_data_filter_attr = ted_data_filter[ted_data_filter["attribute"]=="opexFix"]
        #             ted_data_filter_attr_reg = ted_data_filter_attr[ted_data_filter_attr["region"]==ted_data_investPerCap.loc[i,"region"]]
        #             ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
        #             # investment period not available
        #             if ted_data_filter_fin.empty:
        #                 ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]=="constant"]
        #             if ted_data_filter_fin.empty:
        #                 ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]=="example"]
        #             # region not available, try world and investment period
        #             if ted_data_filter_fin.empty:
        #                 ted_data_filter_attr_world = ted_data_filter_attr[ted_data_filter_attr["region"]=="world"]
        #                 ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
        #                 # try if const available
        #                 if ted_data_filter_fin.empty:
        #                     ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]=="constant"]
        #                 # try if example available
        #                 if ted_data_filter_fin.empty:
        #                     ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]=="example"]
        #             # world not available, try example and investment period
        #             if ted_data_filter_fin.empty:
        #                 ted_data_filter_attr_example = ted_data_filter_attr[ted_data_filter_attr["region"]=="example"]
        #                 ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
        #                 # try if const available
        #                 if ted_data_filter_fin.empty:
        #                     ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]=="constant"]
        #                 # try if example available
        #                 if ted_data_filter_fin.empty:
        #                     ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]=="example"]  
        #             # set opexFix value
        #             opex_fix = ted_data_filter_fin[column].iloc[0]
        #             df.loc[0,column] = float(ted_data_investPerCap.loc[i,column]) * float(opex_fix)
        #         elif column == "attribute":
        #             df.loc[0,column] = "opexPerCapacity"
        #         else:
        #             df.loc[0,column] = ted_data_investPerCap.loc[i,column]
        #     # for loop finished, df line with operationPerCapacity loaded and can be added to ted_data 
        #     ted_data = pd.concat([ted_data,df])
        # # opexfix is now replaced by opexPerCapacity and can be removed from ted data
        # ted_data = ted_data[ted_data["attribute"]!="opexFix"]
        # return ted_data

        # iterate over all investPerCap lines in ted_data to add opexPerCapacity for every investPerCapacity
        _start = time.time()
        print("Start at", _start)
        for i in range(len(ted_data_investPerCap)):
            # create intermediate df --> will contain new line with opexPerCapacity for every investPerCapacity entry 
            df = pd.DataFrame(columns=list(list(ted_data_investPerCap.columns)))
            # iterate over columns of ted_data: component, attribute, region, investment period, values
            for column in list(ted_data_investPerCap.columns):
                if column == "values":
                    # filter until regions and investment periods
                    ted_data_filter = ted_data[ted_data["component"]==ted_data_investPerCap.loc[i,"component"]]
                    ted_data_filter_attr = ted_data_filter[ted_data_filter["attribute"]=="opexFix"]
                    ted_data_filter_attr_reg = ted_data_filter_attr[ted_data_filter_attr["region"]==ted_data_investPerCap.loc[i,"region"]]
                    ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
                    # investment period not available
                    if ted_data_filter_fin.empty:
                        ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]=="constant"]
                    if ted_data_filter_fin.empty:
                        ted_data_filter_fin = ted_data_filter_attr_reg[ted_data_filter_attr_reg["investment_period"]=="example"]
                    # region not available, try world and investment period
                    if ted_data_filter_fin.empty:
                        ted_data_filter_attr_world = ted_data_filter_attr[ted_data_filter_attr["region"]=="world"]
                        ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
                        # try if const available
                        if ted_data_filter_fin.empty:
                            ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]=="constant"]
                        # try if example available
                        if ted_data_filter_fin.empty:
                            ted_data_filter_fin = ted_data_filter_attr_world[ted_data_filter_attr_world["investment_period"]=="example"]
                    # world not available, try example and investment period
                    if ted_data_filter_fin.empty:
                        ted_data_filter_attr_example = ted_data_filter_attr[ted_data_filter_attr["region"]=="example"]
                        ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]==ted_data_investPerCap.loc[i,"investment_period"]]
                        # try if const available
                        if ted_data_filter_fin.empty:
                            ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]=="constant"]
                        # try if example available
                        if ted_data_filter_fin.empty:
                            ted_data_filter_fin = ted_data_filter_attr_example[ted_data_filter_attr_example["investment_period"]=="example"]  
                    # set opexFix value
                    opex_fix = ted_data_filter_fin[column].iloc[0]
                    df.loc[0,column] = float(ted_data_investPerCap.loc[i,column]) * float(opex_fix)
                elif column == "attribute":
                    df.loc[0,column] = "opexPerCapacity"
                else:
                    df.loc[0,column] = ted_data_investPerCap.loc[i,column]
            # for loop finished, df line with operationPerCapacity loaded and can be added to ted_data 
            ted_data = pd.concat([ted_data,df])
        # opexfix is now replaced by opexPerCapacity and can be removed from ted data
        ted_data = ted_data[ted_data["attribute"]!="opexFix"]
        print("opexPerCapacity took:", time.time()-_start)
        return ted_data

    def _commodityConversionDict(self, ted_data):
        '''
        1. Combines commodity conversions per component to commodity dicts --> deletes conversion_commodity column --> deletes obsolete rows
        2. Converts dataframe to multiindex series

        Input: pd.Dataframe
        Output: pd.MultiIndexSeries
        '''
        ted_data_commodityConversionFactors = ted_data[ted_data["attribute"]=="commodityConversionFactors"]
        ted_data = ted_data[ted_data["attribute"]!="commodityConversionFactors"]
        ted_data = ted_data.drop("conversion_commodity",axis=1)
        # set index
        ted_data.set_index(["component", "attribute", "region", "investment_period"], inplace=True)
        # turn multiindex df into multiindex series --> only possible with one column
        ted_data = ted_data.squeeze()
        # add commodityConversionFactors dict component, region and investment specific by using for loop
        list_conv_components = list(ted_data_commodityConversionFactors["component"].unique())
        for component in list_conv_components:
            list_conv_components_regions = list(ted_data_commodityConversionFactors.loc[ted_data_commodityConversionFactors["component"]==component,"region"].unique())
            for region in list_conv_components_regions:
                ted_data_commodityConversionFactors_filtered = ted_data_commodityConversionFactors.loc[ted_data_commodityConversionFactors["component"]==component]
                ted_data_commodityConversionFactors_filtered = ted_data_commodityConversionFactors_filtered.loc[ted_data_commodityConversionFactors_filtered["region"]==region]
                list_conv_components_investment_periods = list(ted_data_commodityConversionFactors_filtered["investment_period"])
                for investment_period in list_conv_components_investment_periods:
                    ted_data_commodityConversionFactors_filtered = ted_data_commodityConversionFactors_filtered.loc[ted_data_commodityConversionFactors_filtered["investment_period"]==investment_period]
                    df_component_conv_factors = ted_data_commodityConversionFactors[ted_data_commodityConversionFactors["component"]==component]
                    # build converionfactor dict
                    conversion_dict = df_component_conv_factors.set_index('conversion_commodity')['values'].to_dict()
                    # from str to float in dict
                    conversion_dict = {key: float(value) for key, value in conversion_dict.items()}
                    ted_data[component,"commodityConversionFactors",region,investment_period] = conversion_dict
        
        return ted_data
    
    def _test_ted(self,ted_data):
        '''
        1. Tests if necessary regional and investmentperiod entries in ted_data, blancs result in error 
        2. lowers entries of component column
        '''
        # check: ivestment periods are int, float values, or 'constant' --> first by for loop make numbers to int
        for i in range(len(ted_data)): 
            if not (ted_data.iloc[i,3]=="constant" or ted_data.iloc[i,3]=="example"):
                ted_data.iloc[i,3] = int(ted_data.iloc[i,3])
        assert ted_data["investment_period"].apply(self._check_value).all(), "Not all values in techno-economic data CSV investment period column are int, float, 'constant', or 'example'."
        # check for empty regions cells
        assert ted_data["region"].notna().all(), "Empty cells in region column, check your input csv, the region column should contain: locationIDs, countries, continents, world, or example!"
        # check for empty investment_period cells
        assert ted_data["investment_period"].notna().all(),"Empty cells in investment_period column, check your input csv, the investment_period column should contain an int, constant, or example!"
        # asure all component letters in lower case
        ted_data["component"] = ted_data["component"].apply(lambda x: x.lower() if isinstance(x, str) else x)
        return ted_data 

    def _check_value(self,x):
        return isinstance(x, int) or isinstance(x, float) or x == "constant" or x == "example"
    
    def _convert_units(self,ted_data_raw):
        '''
        Future function converting units to modelunits and afterwards deleting unit column
        '''
        # check for empty unit cells
        assert ted_data_raw["unit"].notna().all(),"Empty cells in unit column, check your input csv, the unit column should contain a unit!"
        #drop unit column after asuring, that correct units in df
        ted_data_units_converted = ted_data_raw.drop("unit",axis=1)
        return ted_data_units_converted

    def _iterate_available_data(self, component, attribute, region, ip)-> dict|float|int|str:
        '''
        Returns value with highest available resolution in techo-economic data:
            Region value order: LocationID --> country of LocationID --> world --> example
            Timeperiod value order: Investment period --> constant --> example  
        '''
        try: # try region
            try: # try investment period
                return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute][region][ip])
            except: # try investment period
                try: # try investment period
                    return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute][region]["constant"])
                except: # try investment period
                    return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute][region]["example"])
        except: # try region
            try: # try region
                main_gid0 = ModelLocations().get_main_country(region)
                try: # try investment period
                    return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute][main_gid0][ip])
                except: # try investment period
                    try: # try investment period
                        return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute][main_gid0]["constant"])
                    except: # try investment period
                        return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute][main_gid0]["example"])
            except: # try region
                try: # try region
                    try: # try investment period
                        return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute]["world"][ip])
                    except: # try investment period
                        try: # try investment period
                            return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute]["world"]["constant"])
                        except: # try investment period
                            return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute]["world"]["example"])
                except:
                    try: # try investment period
                        return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute]["example"][ip])
                    except: # try investment period
                        try: # try investment period
                            return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute]["example"]["constant"])
                        except: # try investment period
                            return self._asure_datatype(attribute=attribute,fn_attr=self.data[component][attribute]["example"]["example"])

    
    def _asure_datatype(self,attribute,fn_attr):
        '''
        just makes sure, that the datatype is correct

        Returns: attribute as correct data type
        '''
        if attribute == "investPerCapacity": 
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a float. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "opexPerCapacity":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a float. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "opexPerOperation":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a float. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "interestRate":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a float. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "economicLifetime":
            fn_attr = int(fn_attr)
            assert isinstance(fn_attr,int), f"{attribute} is not convertable to a int. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "commodity":
            fn_attr = str(fn_attr)
            assert isinstance(fn_attr,str), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "physicalUnit":
            fn_attr = str(fn_attr)
            assert isinstance(fn_attr,str), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "chargeEfficiency":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "dischargeEfficiency":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "cyclicLifetime":
            fn_attr = int(fn_attr)
            assert isinstance(fn_attr,int), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "selfDischarge":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "chargeRate":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "dischargeRate":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "stateOfChargeMin":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "stateOfChargeMax":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "opexPerChargeOperation":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "opexPerDischargeOperation":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        if attribute == "losses":
            fn_attr = float(fn_attr)
            assert isinstance(fn_attr,float), f"{attribute} is not convertable to a str. One possible error could be, that more than one line with the same content exists in ted_csv."
        return fn_attr

