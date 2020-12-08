import pytest
import os
import sys

import numpy as np
import xarray as xr
import pandas as pd 
import json

import FINE as fn
sys.path.append(os.path.join(os.path.dirname(__file__),'..','examples','Multi-regional_Energy_System_Workflow'))
from getData import getData


def test_allDFs_present_in_esM_instance(minimal_test_esM):
    #EXPECTED (obtained from minimal_test_esM fixture)
    hoursPerTimeStep = 2190

    costs = pd.DataFrame([np.array([ 0.05, 0., 0.1, 0.051,]),np.array([0., 0., 0., 0.,])],
                            index = ['ElectrolyzerLocation', 'IndustryLocation']).T
    revenues = pd.DataFrame([np.array([ 0., 0.01, 0., 0.,]),np.array([0., 0., 0., 0.,])],
                            index = ['ElectrolyzerLocation', 'IndustryLocation']).T
    maxpurchase = pd.DataFrame([np.array([1e6, 1e6, 1e6, 1e6,]),np.array([0., 0., 0., 0.,])],
                            index = ['ElectrolyzerLocation', 'IndustryLocation']).T * hoursPerTimeStep

    demand = pd.DataFrame([np.array([0., 0., 0., 0.,]), np.array([6e3, 6e3, 6e3, 6e3,]),],
                    index = ['ElectrolyzerLocation', 'IndustryLocation']).T * hoursPerTimeStep

    #OUTPUT 
    operationRateMax = minimal_test_esM.getComponentAttribute('Electricity market', 'operationRateMax')
    commodityCostTimeSeries = minimal_test_esM.getComponentAttribute('Electricity market', 'commodityCostTimeSeries')
    commodityRevenueTimeSeries = minimal_test_esM.getComponentAttribute('Electricity market', 'commodityRevenueTimeSeries')
    operationRateFix = minimal_test_esM.getComponentAttribute('Industry site', 'operationRateFix')

    operationRateMax.reset_index(drop=True, inplace=True)
    commodityCostTimeSeries.reset_index(drop=True, inplace=True)
    commodityRevenueTimeSeries.reset_index(drop=True, inplace=True)
    operationRateFix.reset_index(drop=True, inplace=True)

    #ASSERTION
    assert operationRateMax.equals(maxpurchase)
    assert commodityCostTimeSeries.equals(costs)
    assert commodityRevenueTimeSeries.equals(revenues)
    assert operationRateFix.equals(demand)

    #TODO: maybe move this to FINE tests and assert for other data like pd.series 


def test_export_to_dict_minimal(minimal_test_esM):
    #EXPECTED 
    expected_esm_dict = dict(zip(('locations',
                              'commodities',
                              'commodityUnitsDict',
                              'numberOfTimeSteps', 
                              'hoursPerTimeStep',
                              'costUnit', 
                              'lengthUnit',
                              'verboseLogLevel'), 
                            (minimal_test_esM.locations,
                             minimal_test_esM.commodities, 
                             minimal_test_esM.commodityUnitsDict,
                             minimal_test_esM.numberOfTimeSteps, 
                             minimal_test_esM.hoursPerTimeStep,
                             minimal_test_esM.costUnit, 
                             minimal_test_esM.lengthUnit,
                             minimal_test_esM.verboseLogLevel)))
    
    expected_Electrolyzers_investPerCapacity = minimal_test_esM.getComponentAttribute('Electrolyzers', 'investPerCapacity')
    expected_Electricitymarket_operationRateMax = minimal_test_esM.getComponentAttribute('Electricity market', 'operationRateMax')
    expected_Industrysite_operationRateFix = minimal_test_esM.getComponentAttribute('Industry site', 'operationRateFix')
    
    #FUNCTION CALL 
    output_esm_dict, output_comp_dict = fn.dictIO.exportToDict(minimal_test_esM)

    output_Conversion_investPerCapacity = output_comp_dict.get('Conversion').get('Electrolyzers').get('investPerCapacity')
    output_Source_operationRateMax = output_comp_dict.get('Source').get('Electricity market').get('operationRateMax')
    output_Sink_operationRateFix = output_comp_dict.get('Sink').get('Industry site').get('operationRateFix')

    #ASSERTION
    assert output_esm_dict == expected_esm_dict
    assert expected_Electrolyzers_investPerCapacity.equals(output_Conversion_investPerCapacity)
    assert expected_Electricitymarket_operationRateMax.equals(output_Source_operationRateMax)
    assert expected_Industrysite_operationRateFix.equals(output_Sink_operationRateFix)


def test_export_to_dict_multinode(multi_node_test_esM_init):
    #EXPECTED 
    expected_esm_dict = dict(zip(('locations',
                              'commodities',
                              'commodityUnitsDict',
                              'numberOfTimeSteps', 
                              'hoursPerTimeStep',
                              'costUnit', 
                              'lengthUnit',
                              'verboseLogLevel'), 
                            (multi_node_test_esM_init.locations,
                             multi_node_test_esM_init.commodities, 
                             multi_node_test_esM_init.commodityUnitsDict,
                             multi_node_test_esM_init.numberOfTimeSteps, 
                             multi_node_test_esM_init.hoursPerTimeStep,
                             multi_node_test_esM_init.costUnit, 
                             multi_node_test_esM_init.lengthUnit,
                             multi_node_test_esM_init.verboseLogLevel)))

    expected_Windonshore_operationRateMax = multi_node_test_esM_init.getComponentAttribute('Wind (onshore)', 'operationRateMax')
    expected_CCGTplantsmethane_investPerCapacity = multi_node_test_esM_init.getComponentAttribute('CCGT plants (methane)', 'investPerCapacity')
    expected_Saltcavernshydrogen_capacityMax = multi_node_test_esM_init.getComponentAttribute('Salt caverns (hydrogen)', 'capacityMax')
    expected_ACcables_reactances = multi_node_test_esM_init.getComponentAttribute('AC cables', 'reactances')
    expected_Hydrogendemand_operationRateFix = multi_node_test_esM_init.getComponentAttribute('Hydrogen demand', 'operationRateFix')

    #FUNCTION CALL
    output_esm_dict, output_comp_dict = fn.dictIO.exportToDict(multi_node_test_esM_init)

    output_Windonshore_operationRateMax = output_comp_dict.get('Source').get('Wind (onshore)').get('operationRateMax')
    output_CCGTplantsmethane_investPerCapacity = output_comp_dict.get('Conversion').get('CCGT plants (methane)').get('investPerCapacity')
    output_Saltcavernshydrogen_capacityMax = output_comp_dict.get('Storage').get('Salt caverns (hydrogen)').get('capacityMax')
    output_ACcables_reactances = output_comp_dict.get('LinearOptimalPowerFlow').get('AC cables').get('reactances')
    output_Hydrogendemand_operationRateFix = output_comp_dict.get('Sink').get('Hydrogen demand').get('operationRateFix')

    #ASSERTION
    assert output_esm_dict == expected_esm_dict
    assert expected_Windonshore_operationRateMax.equals(output_Windonshore_operationRateMax)
    assert expected_CCGTplantsmethane_investPerCapacity.equals(output_CCGTplantsmethane_investPerCapacity)
    assert expected_Saltcavernshydrogen_capacityMax.equals(output_Saltcavernshydrogen_capacityMax)
    assert expected_ACcables_reactances.equals(output_ACcables_reactances)
    assert expected_Hydrogendemand_operationRateFix.equals(output_Hydrogendemand_operationRateFix)



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

# def test_import_from_dict_minimal(test_esM_fixture, expected_locations, expected_commodityUnitsDict, request):
#TODO: check how to pass fixtures and other parameters at the same time. if it's possible use the above


@pytest.mark.parametrize("test_esM_fixture", ['minimal_test_esM', 'multi_node_test_esM_init'])
def test_import_from_dict_minimal(test_esM_fixture, request):

    test_esM = request.getfixturevalue(test_esM_fixture)

    # FUNCTION CALL 
    ## get dicts 
    esm_dict, comp_dict = fn.dictIO.exportToDict(test_esM)
    ## call the function on dicts 
    output_esM = fn.dictIO.importFromDict(esm_dict, comp_dict)
    
    #EXPECTED (AND OUTPUT)
    expected_locations = test_esM.locations
    expected_commodityUnitsDict = test_esM.commodityUnitsDict

    if test_esM_fixture == 'minimal_test_esM':
        ## expected 
        expected_df = test_esM.getComponentAttribute('Electricity market', 'operationRateMax') 
        expected_series = test_esM.getComponentAttribute('Electrolyzers', 'investPerCapacity') 
        ## output 
        output_df = output_esM.getComponentAttribute('Electricity market', 'operationRateMax')
        output_df.reset_index(level=0, drop=True, inplace=True) #TODO: check why output_esM's dfs 
                                                                # have multiindex (Period, TimeStep)
                                                                # whereas test_esM's just 1 (TimeStep)

        output_series = output_esM.getComponentAttribute('Electrolyzers', 'investPerCapacity') 

    else:
        ## expected 
        expected_df = test_esM.getComponentAttribute('Hydrogen demand', 'operationRateFix') 
        expected_series = test_esM.getComponentAttribute('AC cables', 'reactances') 
        ## output
        output_df = output_esM.getComponentAttribute('Hydrogen demand', 'operationRateFix') 
        output_df.reset_index(level=0, drop=True, inplace=True)

        output_series = output_esM.getComponentAttribute('AC cables', 'reactances')


    #ASSERTION
    assert output_esM.locations == expected_locations
    assert output_esM.commodityUnitsDict == expected_commodityUnitsDict

    assert output_df.equals(expected_df)
    assert output_series.equals(expected_series)

    






    

