# function adjusted: utils.preprocess2dimData()

import fine as fn
import pandas as pd

esm = fn.EnergySystemModel(
    locations={"DE", "AT", "CH"},
    commodities={"energy"},
    commodityUnitsDict={"energy": "joule"})

operationRateMax = pd.DataFrame(index=range(8760))
capacityMax = pd.Series()

operationRateMax["DE_CH"] = 0
operationRateMax["AT_CH"] = 0.5
operationRateMax["AT_DE"] = 0.5
operationRateMax["CH_DE"] = 0
operationRateMax["CH_AT"] = 0.5
operationRateMax["DE_AT"] = 0.5

capacityMax["DE_CH"] = 0.5
capacityMax["AT_CH"] = 0
capacityMax["AT_DE"] = 0.5
capacityMax["CH_DE"] = 0.5
capacityMax["CH_AT"] = 0
capacityMax["DE_AT"] = 0.5

esm.add(
    fn.Transmission(
        esM=esm,
        name="transmission",
        commodity="energy",
        hasCapacityVariable = True,
        capacityFix=capacityMax,
        # operationRateMax=operationRateMax
        )
)
