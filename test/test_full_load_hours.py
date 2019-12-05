import pandas as pd

def test_fullloadhours_above(minimal_test_esM):
   '''
   Get the minimal test system, and check if the fulllload hours of electrolyzer are above 4000.
   '''
   esM = minimal_test_esM

   esM.optimize(timeSeriesAggregation=False, solver = 'glpk')

   # get cumulative operation   
   operationSum = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs('Electrolyzers').sum().sum()

   # get capacity
   capacitySum = esM.componentModelingDict["ConversionModel"].capacityVariablesOptimum.xs('Electrolyzers').sum()

   # calculate fullloadhours
   fullloadhours = operationSum/capacitySum
   
   assert fullloadhours > 4000.


def test_fullloadhours_max(minimal_test_esM):
   '''
   Get the minimal test system, and check if the fulllload hour limitation works
   '''
   
   # modify full load hour limit
   esM = minimal_test_esM

   # get the electolyzer 
   electrolyzer = esM.getComponent('Electrolyzers')

   # set fullloadhour limit
   electrolyzer.yearlyFullLoadHoursMax = pd.Series(3000., index = esM.locations)

   # optimize
   esM.optimize(timeSeriesAggregation=False, solver = 'glpk')

   # get cumulative operation   
   operationSum = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs('Electrolyzers').sum().sum()

   # get capacity
   capacitySum = esM.componentModelingDict["ConversionModel"].capacityVariablesOptimum.xs('Electrolyzers').sum()

   # calculate fullloadhours
   fullloadhours = operationSum/capacitySum
   
   assert fullloadhours < 3000.01


def test_fullloadhours_min(minimal_test_esM):
   '''
   Get the minimal test system, and check if the fulllload hour limitation works
   '''
   
   # modify full load hour limit
   esM = minimal_test_esM

   # get the electolyzer 
   electrolyzer = esM.getComponent('Electrolyzers')

   # set fullloadhour limit
   electrolyzer.yearlyFullLoadHoursMin = pd.Series(5000., index = esM.locations)

   # optimize
   esM.optimize(timeSeriesAggregation=False, solver = 'glpk')

   # get cumulative operation   
   operationSum = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs('Electrolyzers').sum().sum()

   # get capacity
   capacitySum = esM.componentModelingDict["ConversionModel"].capacityVariablesOptimum.xs('Electrolyzers').sum()

   # calculate fullloadhours
   fullloadhours = operationSum/capacitySum
   
   assert fullloadhours > 4999.99