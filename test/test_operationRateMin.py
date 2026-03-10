# %%
import numpy as np
import pandas as pd
import pytest

from fine.utils import ImplementedSolvers


def test_operationRateMin(minimal_test_esM):
    esM = minimal_test_esM
    numberOfTimeSteps = esM.numberOfTimeSteps

    min_load_factor = 0.8
    operationRateMin = pd.DataFrame(
        np.ones((numberOfTimeSteps, 2)) * min_load_factor,
        columns=["ElectrolyzerLocation", "IndustryLocation"],
    )

    with pytest.warns(
        UserWarning, match="Component identifier Electrolyzers already exists"
    ):
        minimal_test_esM.updateComponent(
            componentName="Electrolyzers",
            updateAttrs={"operationRateMin": operationRateMin},
        )

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_OPEN_SOURCE_SOLVER.value,
    )

    ts = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.loc[
        "Electrolyzers"
    ]
    cap = esM.componentModelingDict["ConversionModel"].capacityVariablesOptimum.loc[
        "Electrolyzers"
    ]

    cf_ts = ts.div(cap, axis=0) / esM.hoursPerTimeStep

    assert (cf_ts.min() >= min_load_factor).all()
