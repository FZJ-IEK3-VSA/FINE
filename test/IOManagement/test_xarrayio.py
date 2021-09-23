from copy import deepcopy
import FINE.IOManagement.xarrayIO as xrIO


def test_esm_to_dataset_and_back(minimal_test_esM):

    esM = deepcopy(minimal_test_esM)
    esM.optimize()

    esm_datasets = xrIO.esm_to_datasets(esM)
    esm_from_datasets = xrIO.datasets_to_esm(esm_datasets)

    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_datasets["Input"]["Sink"]["Industry site"]["ts_operationRateFix"].loc[
            :, "IndustryLocation"
        ]
    )
    assert list(
        esM.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    ) == list(
        esm_from_datasets.getComponentAttribute("Industry site", "operationRateFix")[
            "IndustryLocation"
        ]
    )
