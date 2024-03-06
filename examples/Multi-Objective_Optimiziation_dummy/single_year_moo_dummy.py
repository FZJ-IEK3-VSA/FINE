# %%
import FINE as fn  # Provides objects and functions to model an energy system
import pandas as pd  # Used to manage data in tables
import shapely as shp  # Used to generate geometric objects
import numpy as np  # Used to generate random input data

# Input parameters
locations = {"region1"}
commodityUnitDict = {
    "electricity": r"GW$_{el}$",
    "naturalGas": r"GW$_{CH_{4},LHV}$",
    "CO2": r"Mio. t$_{CO_2}$/h",
}
commodities = {"electricity", "naturalGas", "CO2"}
numberOfTimeSteps, hoursPerTimeStep = 8760, 1
costUnit, lengthUnit = "1e6 Euro", "km"

# Code
esM = fn.EnergySystemModel(
    locations=locations,
    commodities=commodities,
    numberOfTimeSteps=numberOfTimeSteps,
    commodityUnitsDict=commodityUnitDict,
    hoursPerTimeStep=hoursPerTimeStep,
    costUnit=costUnit,
    lengthUnit=lengthUnit,
    verboseLogLevel=0,
)

#####################
###### SOURCES ######
#####################

# Input parameters
name, commodity = "Natural gas import", "naturalGas"
hasCapacityVariable = False
commodityCost = 0.03

# Code
esM.add(
    fn.Source(
        esM=esM,
        name=name,
        commodity=commodity,
        hasCapacityVariable=hasCapacityVariable,
        commodityCost=commodityCost,
    )
)

#########################
###### CONVERSIONS ######
#########################

# Input parameters
name, physicalUnit = "Gas power plants", r"GW$_{el}$"
commodityConversionFactors = {
    "electricity": 1,
    "naturalGas": -1 / 0.63,
    "CO2": 201 * 1e-6 / 0.63,
}
hasCapacityVariable = True
investPerCapacity, opexPerCapacity = 650, 650 * 0.03
interestRate, economicLifetime = 0.08, 30

# Code
esM.add(
    fn.Conversion(
        esM=esM,
        name=name,
        physicalUnit=physicalUnit,
        commodityConversionFactors=commodityConversionFactors,
        hasCapacityVariable=hasCapacityVariable,
        investPerCapacity=investPerCapacity,
        opexPerCapacity=opexPerCapacity,
        interestRate=interestRate,
        economicLifetime=economicLifetime,
        materialDemandPerCapacity={"copper": 10},
    )
)

esM.declareOptimizationProblem()


# %%
