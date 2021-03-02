import FINE as fn
import pandas as pd
import numpy as np

def test_initializeTransmission():
    '''
    Tests if Transmission components are initialized without error if
    just required parameters are given.
    '''
    # Define general parameters for esM-instance
    locations = ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4']
    commodityUnitDict = {'commodity1': 'commodity_unit'}
    commodities = {'commodity1'}

    # Initialize esM-instance
    esM = fn.EnergySystemModel(locations=set(locations), commodities=commodities, numberOfTimeSteps=4,
                               commodityUnitsDict=commodityUnitDict,
                               hoursPerTimeStep=1, costUnit='cost_unit', lengthUnit='length_unit')

    # Initialize Transmission
    esM.add(fn.Transmission(esM=esM, 
                            name='Transmission_1', 
                            commodity='commodity1', 
                            hasCapacityVariable=True))

def test_initializeTransmission_withDataFrame():
    '''
    Tests if Transmission components are initialized without error if
    additional parameters are given as DataFrame.
    '''
    # Define general parameters for esM-instance
    locations = ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4']
    commodityUnitDict = {'commodity1': 'commodity_unit'}
    commodities = {'commodity1'}

    # Initialize esM-instance
    esM = fn.EnergySystemModel(locations=set(locations), commodities=commodities, numberOfTimeSteps=4,
                               commodityUnitsDict=commodityUnitDict,
                               hoursPerTimeStep=1, costUnit='cost_unit', lengthUnit='length_unit')

    # # create DataFrame for eligibility and capacityMin
    elig_data = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1], [1,1,1,0]])

    elig_df = pd.DataFrame(elig_data, index=locations, columns=locations)

    capMin_df = elig_df*2
    capMax_df = elig_df*3

    opexPerOp_df = elig_df*0.02
    opexPerOp_df.loc['cluster_1','cluster_2'] = 0.03

    # Initialize Transmission
    esM.add(fn.Transmission(esM=esM, 
                            name='Transmission_1', 
                            commodity='commodity1', 
                            hasCapacityVariable=True,
                            locationalEligibility=elig_df,
                            capacityMax=capMax_df,
                            capacityMin=capMin_df,
                            opexPerOperation=opexPerOp_df)) 




    