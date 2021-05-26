# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from IPython.display import Image
Image("images\strukturExample.png");

# %% [markdown]
# # Workflow for the EnergyLand energy system
#
# In this application of the FINE framework, a 1-node energy system is modeled and optimized.
#
# The workflow is structures as follows:
# 1. Required packages are imported and the input data path is set
# 2. An energy system model instance is created
# 3. Commodity sources are added to the energy system model
# 4. Commodity conversion components are added to the energy system model
# 5. Commodity storages are added to the energy system model
# 6. Commodity sinks are added to the energy system model
# 7. The energy system model is optimized
# 8. Selected optimization results are presented

# %% [markdown]
# ![Structure of EnergyLand](images\strukturExample.png)

# %% [markdown]
# # 1. Import packages
#
# The FINE framework is imported which provides the required classes and functions for modeling the energy system.
# The working directory and the underlying excelfile which provides some of the input data is imported.

# %%
import FINE as fn
from getData import getData
import pandas as pd
import os
cwd = os.getcwd()
data = getData()

# %% [markdown]
# # 2. Set up energy system model instance
#
# The structure of the energy system model is given by the considered locations, in this case we consider only one location (EnergyLand), commodities, the number of time steps as well as the hours per time step.
#
# The commodities are specified by a unit, which can be given as an energy or mass unit per hour.

# %%
locations = {'EnergyLand'}
commodityUnitDict = {'electricity': r'GW$_{el}$', 'hydrogen': r'GW$_{H_{2},LHV}$', 
                     'nGas':r'GW$_{CH4}$', 'coal':r'GW$_{coal}$', 'PHeat': r'GW$_{Pheat}$',
                     'LTHeat':r'GW$_{LTHeat}$', 'CO2':r'kt$_{CO_{2}}$/h', 'pTransport': 'Mio pkm/h',
                     'fTransport': 'Mio tkm/h', 'crudeOil': r'GW$_{Oil}$', 'wood': r'GW$_{wood}$', 
                     'biowaste': r'GW$_{biowaste}$', 'bioslurry': r'GW$_{bioslurry}$',
                     'diesel': r'GW$_{diesel}$', 'biogas': r'GW$_{CH4}$', 'nGasImp': r'GW$_{CH4}$'}
commodities = {'electricity', 'hydrogen', 'nGas', 'coal', 'PHeat', 'LTHeat', 'CO2', 'pTransport', 
               'fTransport', 'crudeOil', 'wood', 'biowaste', 'bioslurry', 'diesel', 'nGasImp', 'biogas'}
numberOfTimeSteps=8760
hoursPerTimeStep=1

# %%
esM = fn.EnergySystemModel(locations=locations, commodities=commodities, numberOfTimeSteps=8760,
                           commodityUnitsDict=commodityUnitDict,
                           hoursPerTimeStep=1, costUnit='1e6 Euro', lengthUnit='km', verboseLogLevel=0)

# %% [markdown]
# # 3. Sources
#
# Source components transfer a commodity from outside the system boundary of EnergyLand into the system.

# %% [markdown]
# ## 3.1 Electricity sources

# %% [markdown]
# ### Wind turbines

# %% [markdown]
# #### Onshore Wind Turbines

# %%
esM.add(fn.Source(esM=esM, name='Wind_Onshore', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=data['Wind_onshore, operationRateMax'],
                  capacityMax=data['Wind_onshore, capacityMax'],
                  investPerCapacity=1250, opexPerCapacity=1250*0.02, interestRate=0.08,
                  economicLifetime=20))

# %% [markdown]
# #### Offshore Wind Turbines

# %%
esM.add(fn.Source(esM=esM, name='Wind_Offshore', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=data['Wind_offshore, operationRateMax'],
                  capacityMax=data['Wind_offshore, capacityMax'],
                  investPerCapacity=2530, opexPerCapacity=2530*0.045, interestRate=0.08,
                  economicLifetime=20))

# %% [markdown]
# ### Photovoltaic

# %%
esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=data['PV, operationRateMax'],
                  capacityMax=data['PV, capacityMax'],
                  investPerCapacity=800, opexPerCapacity=800*0.019, interestRate=0.08,
                  economicLifetime=20))

# %% [markdown]
# ### Electricity import

# %%
esM.add(fn.Source(esM=esM, name='el_Import', commodity='electricity', hasCapacityVariable=False,
                  operationRateMax=data['el_Import, operationRateMax']))

# %% [markdown]
# ## 3.2 Hydrogen source

# %%
esM.add(fn.Source(esM=esM, name='H2_Import', commodity='hydrogen', hasCapacityVariable=False,
                  operationRateMax=data['H2_Import, operationRateMax'],
                  commodityCost=0.132))

# %% [markdown]
# ## 3.3 Coal source

# %%
esM.add(fn.Source(esM=esM, name='CoalSource', commodity='coal', hasCapacityVariable=False,
                  commodityCost=0.021))

# %% [markdown]
# ## 3.4 Crude Oil source

# %%
esM.add(fn.Source(esM=esM, name='CrudeOilSource', commodity='crudeOil', hasCapacityVariable=False,
                  commodityCost=0.036))

# %% [markdown]
# ## 3.5 Natural gas source

# %%
esM.add(fn.Source(esM=esM, name='nGasSource', commodity='nGasImp', hasCapacityVariable=False,
                  commodityCost=0.0256))

# %% [markdown]
# ## 3.6 Biomass sources

# %% [markdown]
# #### Wood Source

# %%
esM.add(fn.Source(esM=esM, name='WoodSource', commodity='wood', hasCapacityVariable=True,
                  capacityMax=data['wood_source, capacityMax'],
                  commodityCost=0.028))

# %% [markdown]
# #### Biowaste Source

# %%
esM.add(fn.Source(esM=esM, name='BiowasteSource', commodity='biowaste', hasCapacityVariable=True,
                  capacityMax=data['biowaste_source, capacityMax'],
                  commodityCost=0.07))

# %% [markdown]
# #### Bioslurry Source

# %%
esM.add(fn.Source(esM=esM, name='BioslurrySource', commodity='bioslurry', hasCapacityVariable=True,
                  capacityMax=data['bioslurry_source, capacityMax'],
                  commodityCost=0.07))

# %% [markdown]
# # 4. Conversion components
#
# These are the components which can transfer one commodity into another one.

# %% [markdown]
# ## 4.1 Biomas to biogas

# %% [markdown]
# ### Bioslurry to Biogas

# %%
esM.add(fn.Conversion(esM=esM, name='bioslurry-biogas', physicalUnit= r'GW$_{CH4}$',
                      commodityConversionFactors={'bioslurry':-1, 'biogas':1},
                      hasCapacityVariable=False))

# %% [markdown]
# ### Biowaste to Biogas

# %%
esM.add(fn.Conversion(esM=esM, name='biowaste-biogas', physicalUnit= r'GW$_{CH4}$',
                      commodityConversionFactors={'biowaste':-1, 'biogas':1},
                      hasCapacityVariable=False))

# %% [markdown]
# ## 4.2 Methane Slip (Virtual conversion)

# %%
methaneSlip=0.1
esM.add(fn.Conversion(esM=esM, name='CH4Slip', physicalUnit= r'GW$_{CH4}$',
                      commodityConversionFactors={'nGasImp':-1, 'nGas':1, 'CO2':methaneSlip*2.014},
                      hasCapacityVariable=False))

# %% [markdown]
# ## 4.3 Biogas to Methane (Virtual conversion)

# %%
esM.add(fn.Conversion(esM=esM, name='biogas-nGas', physicalUnit= r'GW$_{CH4}$',
                      commodityConversionFactors={'biogas':-1, 'nGas':1, 'CO2':-0.2014},
                      hasCapacityVariable=True, opexPerOperation=0.0003,
                      investPerCapacity=343, opexPerCapacity=343*0.025, interestRate=0.08,
                      economicLifetime=15))

# %% [markdown]
# ## 4.4 Transport

# %% [markdown]
# ### Batterie Electric Vehicle

# %% [markdown]
# #### BEV Car

# %%
esM.add(fn.Conversion(esM=esM, name='BEV_PCar', physicalUnit= r'Mio pkm/h',
                      commodityConversionFactors={'electricity':-1/7.676226, 'pTransport':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=15694, opexPerCapacity=15694*0.009, interestRate=0.08,
                      economicLifetime=12))

# %% [markdown]
# #### BEV Truck

# %%
esM.add(fn.Conversion(esM=esM, name='BEV_Truck', physicalUnit= r'Mio tkm/h',
                      commodityConversionFactors={'electricity':-1/11.401, 'fTransport':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=4304, opexPerCapacity=4304*0.009, interestRate=0.08,
                      economicLifetime=15))

# %% [markdown]
# ### Fuel Cell Electric Vehicle

# %% [markdown]
# #### FCEV Car

# %%
esM.add(fn.Conversion(esM=esM, name='FCEV_PCar', physicalUnit= r'Mio pkm/h',
                      commodityConversionFactors={'hydrogen':-1/4.7472, 'pTransport':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=15694, opexPerCapacity=15694*0.009, interestRate=0.08,
                      economicLifetime=12))

# %% [markdown]
# #### FCEV Truck

# %%
esM.add(fn.Conversion(esM=esM, name='FCEV_Truck', physicalUnit= r'Mio tkm/h',
                      commodityConversionFactors={'hydrogen':-1/8.251, 'fTransport':1},
                      hasCapacityVariable=True,
                      investPerCapacity=4283, opexPerCapacity=4283*0.009, interestRate=0.08,
                      economicLifetime=15))

# %% [markdown]
# ### Fossil Vehicles

# %% [markdown]
# #### Fossil Car

# %%
esM.add(fn.Conversion(esM=esM, name='FossilCar', physicalUnit= r'Mio pkm/h',
                      commodityConversionFactors={'diesel':-1/3.1308, 'pTransport':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=15694, opexPerCapacity=15694*0.016, interestRate=0.08,
                      economicLifetime=12))

# %% [markdown]
# #### Fossil Truck

# %%
esM.add(fn.Conversion(esM=esM, name='FossilTruck', physicalUnit= r'Mio tkm/h',
                      commodityConversionFactors={'diesel':-1/7.938, 'fTransport':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=3342, opexPerCapacity=3342*0.016, interestRate=0.08,
                      economicLifetime=15))

# %% [markdown]
# ## 4.5 Diesel Refinery

# %%
esM.add(fn.Conversion(esM=esM, name='DieselRef', physicalUnit= r'GW$_{diesel}$',
                      commodityConversionFactors={'crudeOil':-1/0.364, 'diesel':1, 'CO2':0.725},
                      hasCapacityVariable=True, 
                      investPerCapacity=1/0.364, opexPerCapacity=(1/0.364)*0.001, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# ## 4.6 Power Plants

# %% [markdown]
# ### Combined Cycle Gas Turbine

# %% [markdown]
# #### Natural Gas CCGT

# %%
esM.add(fn.Conversion(esM=esM, name='CCGT plants (NGas)', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'nGas':-1/0.65, 'CO2':0.31},
                      hasCapacityVariable=True,investPerCapacity=850, 
                      opexPerCapacity=850*0.03, opexPerOperation=0.002, interestRate=0.08,
                      economicLifetime=30))

# %% [markdown]
# #### H2 CCGT

# %%
esM.add(fn.Conversion(esM=esM, name='CCGT plants (hydrogen)', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'hydrogen':-1/0.6},
                      hasCapacityVariable=True, investPerCapacity=760, 
                      opexPerCapacity=760*0.014, opexPerOperation=0.002, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# ### Fuel cell

# %%
esM.add(fn.Conversion(esM=esM, name='LS-SOFC', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'hydrogen':-1/0.7, 'LTHeat':0.25/0.7},
                      hasCapacityVariable=True, investPerCapacity=1210, 
                      opexPerCapacity=1210*0.008, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# ### Coal power plant

# %%
esM.add(fn.Conversion(esM=esM, name='CoalPP', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'coal':-1/0.5, 'CO2':0.674},
                      hasCapacityVariable=True, opexPerOperation=0.0015,
                      investPerCapacity=1450, opexPerCapacity=1450*0.026, interestRate=0.08,
                      economicLifetime=40))

# %% [markdown]
# ### Combined Heat and Power Plants

# %% [markdown]
# #### Coal CHP

# %%
esM.add(fn.Conversion(esM=esM, name='CoalCHP', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'LTHeat':0.51/0.38, 'coal':-1/0.38, 'CO2':0.886},
                      hasCapacityVariable=True, opexPerOperation=0.0051,
                      investPerCapacity=1847, opexPerCapacity=1847*0.027, interestRate=0.08,
                      economicLifetime=35))

# %% [markdown]
# #### Wood CHP

# %%
esM.add(fn.Conversion(esM=esM, name='WoodCHP', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'LTHeat':0.826/0.291, 'wood':-1/0.291},
                      hasCapacityVariable=True, opexPerOperation=0.0038,
                      investPerCapacity=3000, opexPerCapacity=3000*0.029, interestRate=0.08,
                      economicLifetime=25))

# %% [markdown]
# #### Natural Gas CHP

# %%
esM.add(fn.Conversion(esM=esM, name='nGasCHP', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'LTHeat':0.5/0.35, 'nGas':-1/0.35, 'CO2':0.575},
                      hasCapacityVariable=True, opexPerOperation=0.0015,
                      investPerCapacity=666, opexPerCapacity=666*0.041, interestRate=0.08,
                      economicLifetime=30))

# %% [markdown]
# #### Biogas CHP

# %%
esM.add(fn.Conversion(esM=esM, name='BioGasCHP', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'LTHeat':1, 'biogas':-1/0.47},
                      hasCapacityVariable=True, opexPerOperation=0.008,
                      investPerCapacity=850, opexPerCapacity=850*0.01, interestRate=0.08,
                      economicLifetime=25))

# %% [markdown]
# #### H2 CHP

# %%
esM.add(fn.Conversion(esM=esM, name='H2CHP', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'LTHeat':0.41/0.49, 'hydrogen':-1/0.49},
                      hasCapacityVariable=True, opexPerOperation=0.0006,
                      investPerCapacity=715, opexPerCapacity=715*0.001, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# ## 4.7 Thermal power plants

# %% [markdown]
# #### Oil Boiler

# %%
esM.add(fn.Conversion(esM=esM, name='oilBoiler', physicalUnit=r'GW$_{LTHeat}$',
                      commodityConversionFactors={'crudeOil':-1/0.96, 'LTHeat':1, 'CO2':0.275},
                      hasCapacityVariable=True, 
                      investPerCapacity=330, opexPerCapacity=330*0.041, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# #### Gas Boiler

# %%
esM.add(fn.Conversion(esM=esM, name='gasBoiler', physicalUnit=r'GW$_{LTHeat}$',
                      commodityConversionFactors={'nGas':-1/0.96, 'LTHeat':1, 'CO2':0.21},
                      hasCapacityVariable=True, 
                      investPerCapacity=330, opexPerCapacity=330*0.012, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# #### H2 Boiler

# %%
esM.add(fn.Conversion(esM=esM, name='H2Boiler', physicalUnit=r'GW$_{LTHeat}$',
                      commodityConversionFactors={'hydrogen':-1/0.98, 'LTHeat':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=655, opexPerCapacity=655*0.01, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# ### Heat pump

# %%
esM.add(fn.Conversion(esM=esM, name='Heatpump', physicalUnit=r'GW$_{LTHeat}$',
                      commodityConversionFactors={'electricity':-1/0.45, 'LTHeat':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=725, opexPerCapacity=725*0.02, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# ### Heating rod

# %%
esM.add(fn.Conversion(esM=esM, name='Heating rod', physicalUnit=r'GW$_{LTHeat}$',
                      commodityConversionFactors={'electricity':-1/0.99, 'LTHeat':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=800, opexPerCapacity=800*0.0125, interestRate=0.08,
                      economicLifetime=30))

# %% [markdown]
# ### Electrode boiler

# %%
esM.add(fn.Conversion(esM=esM, name='electrode boiler', physicalUnit=r'GW$_{Pheat}$',
                      commodityConversionFactors={'electricity':-1/0.99, 'PHeat':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=140, opexPerCapacity=140*0.02, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# ### Stove

# %%
esM.add(fn.Conversion(esM=esM, name='woood Stove', physicalUnit=r'GW$_{LTHeat}$',
                      commodityConversionFactors={'wood':-1/0.75, 'LTHeat':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=775, opexPerCapacity=775*0.06, interestRate=0.08,
                      economicLifetime=20))

# %% [markdown]
# ## 4.8 Electrolyzer

# %%
esM.add(fn.Conversion(esM=esM, name='electroylzer', physicalUnit=r'GW$_{H_{2},LHV}$',
                      commodityConversionFactors={'electricity':-1/0.7, 'hydrogen':1},
                      hasCapacityVariable=True, 
                      investPerCapacity=500, opexPerCapacity=500*0.03, interestRate=0.08,
                      economicLifetime=10))

# %% [markdown]
# # 5. Storages
#
# The storages which can be used by the EnergyLand model are constructed.

# %% [markdown]
# ## Lithium ion batteries

# %%
esM.add(fn.Storage(esM=esM, name='Li-ion batteries', commodity='electricity',
                   hasCapacityVariable=True, chargeEfficiency=0.99,
                   dischargeEfficiency=0.99, selfDischarge=0.004,
                   doPreciseTsaModeling=False,investPerCapacity=120, 
                   opexPerCapacity=120*0.014, opexPerChargeOperation=0.0001, 
                   interestRate=0.08, economicLifetime=10))

# %% [markdown]
# ## Hydrogen filled salt caverns

# %%
esM.add(fn.Storage(esM=esM, name='H2Storage', commodity='hydrogen',
                   hasCapacityVariable=True, chargeEfficiency=0.98,
                   dischargeEfficiency=0.998, doPreciseTsaModeling=False,
                   investPerCapacity=362, opexPerCapacity=362*0.02, opexPerChargeOperation=0.0001, 
                   interestRate=0.08, economicLifetime=40))

# %% [markdown]
# ## Heat storage

# %%
esM.add(fn.Storage(esM=esM, name='LTHeatstorage', commodity='LTHeat',
                   hasCapacityVariable=True, chargeEfficiency=0.95,
                   dischargeEfficiency=0.95, selfDischarge=0.0003,
                   chargeRate=1, dischargeRate=1, doPreciseTsaModeling=False,
                   investPerCapacity=147, opexPerCapacity=147*0.01, opexPerChargeOperation=0.0001,
                   interestRate=0.08, economicLifetime=20))

# %% [markdown]
# # 6. Sinks
#
# Electricity, heat and transport demand are set in the following components.

# %% [markdown]
# ## Electricity demand

# %%
eDemand=516
esM.add(fn.Sink(esM=esM, name='Electricity demand', commodity='electricity',
                hasCapacityVariable=False, operationRateFix=data['Electricity demand, operationRateFix']*eDemand))

# %% [markdown]
# ## Passenger Transportation demand

# %%
pTdemand=867
esM.add(fn.Sink(esM=esM, name='pT_demand', commodity='pTransport',
                hasCapacityVariable=False, operationRateFix=data['T_demand, operationRateFix']*pTdemand))

# %% [markdown]
# ## Freight Transportation demand

# %%
fTdemand=945.5
esM.add(fn.Sink(esM=esM, name='fT_demand', commodity='fTransport',
                hasCapacityVariable=False, operationRateFix=data['T_demand, operationRateFix']*fTdemand))

# %% [markdown]
# ## Heat demand

# %% [markdown]
# ### Process heat demand

# %%
pHeatDemand=423.75
esM.add(fn.Sink(esM=esM, name='PHeat_demand', commodity='PHeat',
                hasCapacityVariable=False, operationRateFix=data['pHeat_demand, operationRateFix']*pHeatDemand))

# %% [markdown]
# ### Low temperature residential heat demand

# %%
LTHeatDemand=560.8
esM.add(fn.Sink(esM=esM, name='LTHeat_demand', commodity='LTHeat',
                hasCapacityVariable=False, operationRateFix=data['LtHeat_demand, operationRateFix']*LTHeatDemand))

# %% [markdown]
# ## Environment
#
# The CO2 limit is set in this component.

# %%
CO2limit=210000
esM.add(fn.Sink(esM=esM, name='environment', commodity='CO2', commodityLimitID='CO2_cap',
                hasCapacityVariable=False, yearlyLimit=CO2limit))

# %% [markdown]
# # 7. Optimization of EnergyLand

# %%
esM.cluster(numberOfTypicalPeriods=48)

# %% tags=["nbval-skip"]
esM.optimize(timeSeriesAggregation=True, solver='gurobi')

# %% [markdown]
# # 8. Results

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("ConversionModel", outputLevel=2)

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("StorageModel", outputLevel=2)
