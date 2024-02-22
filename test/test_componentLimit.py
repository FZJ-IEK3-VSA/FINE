import fine as fn
import pandas as pd
import numpy as np
import pytest


def test_componentLimitConstraint(multi_node_test_esM_init):
    locations = multi_node_test_esM_init.locations
    
    # 1) Define componentLimit
    
    # We are constructing 3 componentLimits
    # 1. LIM1: Capacity limit of 20 for Wind (offshore) in locations cluster_0, cluster_1, cluster_2
    # 2. LIM2: Operation limit of 10000 for Wind (onshore) in locations cluster_5, cluster_6, cluster_7
    # 3. LIM3: Fixed capacity limit of 10 for Wind (offshore) and Wind (onshore) combined in locations cluster_0, cluster_1
    
    _componentLimit = pd.DataFrame(
        columns=["value","bound","type"],
        index=["LIM1","LIM2","LIM3"], 
        data=[[20,"upper","capacity"],[10000,"upper","operation"],[10,"fixed","capacity"]]
    )

    _componentLimitEligibility = pd.DataFrame(
        columns=["LIM1","LIM2","LIM3"],
        index=sorted(list(locations)),
        data=[[1, 0, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
    )
    
    componentLimit = {}
    componentLimitEligibility = {}
    componentLimit[0] = _componentLimit
    componentLimitEligibility[0] = _componentLimitEligibility
    
    multi_node_test_esM_init.updateComponent(
        componentName="Wind (offshore)",
        updateAttrs={
            "componentLimitID" : ["LIM1","LIM3"]
        },
    )
    multi_node_test_esM_init.updateComponent(
        componentName="Wind (onshore)",
        updateAttrs={
            "componentLimitID" : ["LIM2","LIM3"]
        },
    )
    
    # Add componentLimit to esM manually. Not recommended for normal use.
    multi_node_test_esM_init.processedComponentLimit = componentLimit
    multi_node_test_esM_init.processedComponentLimitEligibility = componentLimitEligibility
    
    # 2) Optimize esM       
    multi_node_test_esM_init.aggregateTemporally(
        numberOfTypicalPeriods=3,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    multi_node_test_esM_init.optimize(timeSeriesAggregation=True, solver="glpk")
    
    _data = multi_node_test_esM_init.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum
    elig = _componentLimitEligibility[_componentLimitEligibility==1]["LIM1"].dropna().index
    value = _data.loc["Wind (offshore)", elig].sum()
    
    assert np.greater_equal(20,np.round(value,0))
    
    _data = multi_node_test_esM_init.componentModelingDict["SourceSinkModel"].operationVariablesOptimum
    elig = _componentLimitEligibility[_componentLimitEligibility==1]["LIM2"].dropna().index
    value = _data.loc[("Wind (onshore)", elig),:].sum().sum()
    assert np.greater_equal(10000,np.round(value,0))
    
    
    _data = multi_node_test_esM_init.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum
    elig = _componentLimitEligibility[_componentLimitEligibility==1]["LIM3"].dropna().index
    techs = ["Wind (offshore)","Wind (onshore)"]
    value = _data.loc[techs, elig].sum().sum()
    
    assert np.isclose(10,value)