import FINE as fn
import numpy as np
import pandas as pd

def test_perfectForesight_mini(perfectForesight_test_esM):
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    print(perfectForesight_test_esM)