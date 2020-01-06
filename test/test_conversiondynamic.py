#!/usr/bin/env python
# coding: utf-8

# # Workflow for a multi-regional energy system
#
import FINE as fn

import pytest


def test_ConversionFancy_needs_capacity():

    esM = fn.EnergySystemModel(locations={'example_region1', }, 
                               commodities={'electricity',},
                               commodityUnitsDict={'electricity': r'GW$_{el}$'},
                               hoursPerTimeStep=1, costUnit='1e9 Euro', lengthUnit='km', verboseLogLevel=0)

    with pytest.raises(ValueError, match=r".*hasCapacityVariable.*"):

        fn.ConversionFancy(esM=esM, name='restricted', physicalUnit=r'GW$_{el}$',
                            commodityConversionFactors={'electricity':1, 'methane':-1/0.625},
                            partLoadMin = 0.3, bigM= 100, rampDownMax=0.5,
                            investPerCapacity=0.5, opexPerCapacity=0.021, opexPerOperation =1, interestRate=0.08,
                            economicLifetime=33, hasCapacityVariable = False)


if __name__ == "__main__":
    test_ConversionFancy_needs_capacity()