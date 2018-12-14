import FINE as fn
import FINE.utils as utils
import matplotlib.pyplot as plt
import pandas as pd
import ast
import inspect
import time
import warnings
import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    warnings.warn('The netCDF4 python package could not be imported.')


def createOutputAsNetCDF(esM, output='output.nc', initialTime='2050-01-01 00:30:00', freq='H', saveLocationWKT=False, locationSource=None):
    """
    Function which creates a netCDF file including operation, capacity and cost variables in the energy system

    :param esM: EnergySystemModel instance in which the optimized model is hold
    :type esM: EnergySystemModel instance

    :param output: path for output file
    :type output: string

    :param initialTime: initial date and time of operation in the energy system
    :type initialTime: string

    :param freq: frequency of the time steps considered in the energy system
    :type freq: string ('H', '5H' , 'D', 'M')

    :returns: None
    :rtype: depends on the components in the energy system
    """

    # initiate netCDF file
    ds = nc.Dataset(output, mode='w')
    # create the dimensions for locations and time at top level
    ds.createDimension('locations', size=len(esM.locations))
    ds.createDimension('time', size=len(esM.totalTimeSteps))

    # Create the variable for time stamps
    var = ds.createVariable('time', 'u4', dimensions=('time',))
    var.description = 'Time stamps to be used in the operation time series'
    var.units = 'Minutes since 1900-01-01 00:00:00'
    datetimes = pd.date_range(initialTime, periods=len(esM.totalTimeSteps), freq=freq)
    var[:]= nc.date2num( datetimes.to_pydatetime(), var.units )
    #to create the time stamps back...
    #pd.to_datetime( nc.num2date( var[:], var.units ) )

    # Create the group for operation time series
    ds.createGroup('/operationTimeSeries/')

    # Create the group for each modeling class
    for compMod in esM.componentModelingDict.keys():
        tsD = ds.createGroup('/operationTimeSeries/{}'.format(compMod[:-8]))
        if compMod == 'StorageModel': df = esM.componentModelingDict[compMod].stateOfChargeOperationVariablesOptimum
        else: df = esM.componentModelingDict[compMod].operationVariablesOptimum

        # Create dimension 
        opColDim = '{}_col'.format(compMod[:-8])
        tsD.createDimension(opColDim, size=len(df.index.get_level_values(0)))

        var = tsD.createVariable('{}_operation'.format(compMod[:-8]), 'f', dimensions=('time', opColDim,), zlib=True)
        var.description = 'Big table containing operation data for all technologies and regions under {} class'.format(compMod)
        var[:]= df.T.values

        for levelNo in range(len(df.index.levels)):
            var = tsD.createVariable('{}_op_col_{}'.format(compMod[:-8],levelNo), str, dimensions=(opColDim,))
            var.description = 'Level_{} index to be used in multiIndex'.format(levelNo)
            for i in range(len(df.index.get_level_values(levelNo))): var[i]= df.index.get_level_values(levelNo)[i]

    # Create the group for capacity variables
    ds.createGroup('/capacityVariables/')
    for compMod in esM.componentModelingDict.keys():
        cvD = ds.createGroup('/capacityVariables/{}'.format(compMod[:-8]))
        df = esM.componentModelingDict[compMod].capacityVariablesOptimum

        dil = pd.Series()
        if compMod =='TransmissionModel':
            for loc1 in df.columns: 
                for tech in df.index.get_level_values(0):
                    for loc2 in df.loc[tech].index:
                        capVar = df.loc[tech].loc[loc2][loc1]
                        if not np.isnan(capVar): dil['{}:{}:{}'.format(loc1,loc2,tech)]=capVar   
        else:
            for loc1 in df.columns: 
                for tech in df.index.get_level_values(0):
                    capVar = df.loc[tech][loc1]
                    if not np.isnan(capVar): dil['{}:{}'.format(loc1,tech)]=capVar
        dil.sort_index(inplace=True)    

        dimName = '{}_capVar_ix'.format(compMod[:-8])
        cvD.createDimension(dimName, size=len(dil.index))

        var = cvD.createVariable('{}_capVar'.format(compMod[:-8]), 'f', dimensions=(dimName,), zlib=True)
        var.description = 'Capacity variables as a result of optimization'
        var[:]= dil.values

        var = cvD.createVariable('{}_capVar_col'.format(compMod[:-8]), str, dimensions=(dimName,))
        var.description = 'List of names to be used in capacity variable columns'
        for i in range(len(dil.index)): var[i]= dil.index[i]

    # Create the group for cost components and binary variables
    ds.createGroup('/costComponents/')
    for compMod in esM.componentModelingDict.keys():
        ccD = ds.createGroup('/costComponents/{}'.format(compMod[:-8]))   

        data = esM.componentModelingDict[compMod].optSummary
        s = data.index.get_level_values(1) == 'TAC'
        df = data.loc[s].sum(level=0)
        df.sort_index(inplace=True) 

        TACcolDim ='{}_TAC_col'.format(compMod[:-8])
        TACindexDim= '{}_TAC_ix'.format(compMod[:-8])
        ccD.createDimension(TACcolDim, size=len(df.columns))
        ccD.createDimension(TACindexDim, size=len(df.index))

        costCompindexDim ='{}_costComp_ix'.format(compMod[:-8])
        costCompcolDim= '{}_costComp_col'.format(compMod[:-8])
        ccD.createDimension(costCompindexDim, size=len(data.index))
        ccD.createDimension(costCompcolDim, size=len(data.columns))

        var = ccD.createVariable('{}_TAC'.format(compMod[:-8]), 'f', dimensions=(TACindexDim, TACcolDim,), zlib=True)
        var.description = 'Table illustrating the contribution of each technology and region to the TAC of system'
        var.unit = '{}/a'.format(esM.costUnit)
        var[:]= df.values

        var = ccD .createVariable('{}_TAC_col'.format(compMod[:-8]), str, dimensions=(TACcolDim,))
        var.description = 'Column for TAC table'
        for i in range(len(df.columns)): var[i]= df.columns[i]

        var = ccD .createVariable('{}_TAC_ix'.format(compMod[:-8]), str, dimensions=(TACindexDim,))
        var.description = 'Index for TAC table'
        for i in range(len(df.index)): var[i]= df.index[i]

        for levelNo in range(len(data.index.levels)):
            var = ccD.createVariable('{}_costComp_ix_{}'.format(compMod[:-8],levelNo), str, dimensions=(costCompindexDim,))
            var.description = 'Level_{} index to be used in multiIndex'.format(levelNo)
            var.colName = data.index.names[levelNo]
            for i in range(len(data.index.get_level_values(levelNo))): var[i]= data.index.get_level_values(levelNo)[i]

        var = ccD .createVariable('{}_costComp_col'.format(compMod[:-8]), str, dimensions=(costCompcolDim,))
        var.description = 'Column names for cost component table'
        for i in range(len(data.columns)): var[i]= data.columns[i]

        var = ccD.createVariable('{}_costComp'.format(compMod[:-8]), 'f', dimensions=(costCompindexDim, costCompcolDim,), zlib=True)
        var.description = 'Main table with cost break down of each technology in each region'
        var[:]= data.values

    ds.close()

    return

