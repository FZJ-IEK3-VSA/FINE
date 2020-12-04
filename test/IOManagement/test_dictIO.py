import pytest
import os
import sys

import numpy as np
import xarray as xr
import pandas as pd 
import json

import FINE as fn
from FINE.IOManagement import dictIO

sys.path.append(os.path.join(os.path.dirname(__file__),'..','examples','Multi-regional_Energy_System_Workflow'))
from getData import getData

def test_export_to_dict_minimal(minimal_test_esM):
    #EXPECTED 
    test_locations={'ElectrolyzerLocation', 'IndustryLocation'} 
    test_commodities={'electricity', 'hydrogen'}
    test_commodityUnitsDict={'electricity': r'kW$_{el}$', 'hydrogen': r'kW$_{H_{2},LHV}$'}
    n_TimeSteps= 4
    n_hours= 2190 
    costUnit='1 Euro' 
    lengthUnit='km'
    verboseLogLevel=2

    expected_esm_dict = dict(zip(('locations','commodities',
                              'commodityUnitsDict',
                              'numberOfTimeSteps', 'hoursPerTimeStep',
                              'costUnit', 'lengthUnit',
                             'verboseLogLevel'), 
                            (test_locations,test_commodities, 
                            test_commodityUnitsDict,
                            n_TimeSteps, n_hours,
                            costUnit, lengthUnit,
                            verboseLogLevel)))

    expected_comp_subvalue = pd.Series([500, 500], index=list(test_locations))
    
    #FUNCTION CALL 
    output_esm_dict, output_comp_dict = dictIO.exportToDict(minimal_test_esM)

    #ASSERTION
    assert output_esm_dict == expected_esm_dict
    assert expected_comp_subvalue.eq(output_comp_dict.get('Conversion').get('Electrolyzers').get('investPerCapacity')).all()
   
    #TODO: Check why values like operationRateMax, and operationRateFix (basically time series) are None in the comp_dict
    # assert for Source and Sink values also.


def test_export_to_dict_multinode(multi_node_test_esM_init):
    data = getData()
    #EXPECTED 
    test_locations = {'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7'}
    test_commodities = {'electricity', 'hydrogen', 'methane', 'biogas', 'CO2'}
    test_commodityUnitsDict = {'electricity': r'GW$_{el}$', 'methane': r'GW$_{CH_{4},LHV}$', 'biogas': r'GW$_{biogas,LHV}$',
                        'CO2': r'Mio. t$_{CO_2}$/h', 'hydrogen': r'GW$_{H_{2},LHV}$'}
    n_TimeSteps = 8760
    n_hours = 1
    costUnit='1e9 Euro'
    lengthUnit='km'
    verboseLogLevel=0

    expected_esm_dict = dict(zip(('locations','commodities',
                            'commodityUnitsDict',
                            'numberOfTimeSteps', 'hoursPerTimeStep',
                            'costUnit', 'lengthUnit',
                            'verboseLogLevel'), 
                            (test_locations,test_commodities, 
                            test_commodityUnitsDict,
                            n_TimeSteps, n_hours,
                            costUnit, lengthUnit,
                            verboseLogLevel)))
    
    expected_comp_subvalue = data['Wind (onshore), capacityMax']

    #FUNCTION CALL
    output_esm_dict, output_comp_dict = dictIO.exportToDict(multi_node_test_esM_init)

    #ASSERTION
    assert output_esm_dict == expected_esm_dict
    assert expected_comp_subvalue.eq(output_comp_dict.get('Source').get('Wind (onshore)').get('capacityMax')).all()

# @pytest.mark.parametrize("test_esM_fixture", "expected_locations", "expected_commodityUnitsDict", 
#                               [( 'minimal_test_esM', 
#                                  {'ElectrolyzerLocation', 'IndustryLocation'}, 
#                                  {'electricity': r'kW$_{el}$', 'hydrogen': r'kW$_{H_{2},LHV}$'} 
#                                  ),
#                                ( 'multi_node_test_esM_init', 
#                                  {'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 
#                                   'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7'},
#                                  {'electricity': r'GW$_{el}$', 'methane': r'GW$_{CH_{4},LHV}$', 
#                                  'biogas': r'GW$_{biogas,LHV}$','CO2': r'Mio. t$_{CO_2}$/h', 
#                                  'hydrogen': r'GW$_{H_{2},LHV}$'}
#                                  )
#                               ]
#                             )
#TODO: check how to pass fixtures and other parameters at the same time. if it's possible use the above
@pytest.mark.parametrize("test_esM_fixture", ['minimal_test_esM', 'multi_node_test_esM_init'])
# def test_import_from_dict_minimal(test_esM_fixture, expected_locations, expected_commodityUnitsDict, request):
def test_import_from_dict_minimal(test_esM_fixture, request):
    test_esM = request.getfixturevalue(test_esM_fixture)
    esm_dict, comp_dict = dictIO.exportToDict(test_esM)

    # FUNCTION CALL 
    output_esM = dictIO.importFromDict(esm_dict, comp_dict)

    if test_esM_fixture == 'minimal_test_esM':
        expected_locations = {'ElectrolyzerLocation', 'IndustryLocation'}
        expected_commodityUnitsDict = {'electricity': r'kW$_{el}$', 'hydrogen': r'kW$_{H_{2},LHV}$'}
    else:
        expected_locations = {'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3',
                              'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7'}
        expected_commodityUnitsDict = {'electricity': r'GW$_{el}$', 'methane': r'GW$_{CH_{4},LHV}$', 
                                       'biogas': r'GW$_{biogas,LHV}$', 'CO2': r'Mio. t$_{CO_2}$/h',
                                       'hydrogen': r'GW$_{H_{2},LHV}$'}
        
    #ASSERTION
    assert output_esM.locations == expected_locations
    assert output_esM.commodityUnitsDict == expected_commodityUnitsDict
    #TODO: assert some values from comp_dict also. You'll need mates' OutputManager to access values from esM instance 




    

