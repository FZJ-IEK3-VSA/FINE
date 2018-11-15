import FINE as fn
import FINE.utils as utils
import matplotlib.pyplot as plt
import pandas as pd
import ast
import inspect
import time
import warnings

try:
    import geopandas as gpd
except ImportError:
    warnings.warn('The GeoPandas python package could not be imported.')


def readOptimizationOutputFromExcel(fileName='scenarioResults.xlsx'):
    """
    :param fileName: excel file name oder path (including .xlsx ending)
        |br| * the default value is 'scenarioResults.xlsx'
    :type fileName: string

    :return:
    """



    file = pd.ExcelFile(fileName)

