import inspect

from fine import ComponentModel, plotPieChart


def test_function_defaults_are_not_mutable():
    parameters = {
        ComponentModel.declareBinOpVarSet: ["binaryOperationParameter"],
        ComponentModel.getEconomicsDesign: ["QPfactorNames", "QPdivisorNames"],
        ComponentModel.getLocEconomicsDesign: ["QPfactorNames", "QPdivisorNames"],
        plotPieChart: ["color_list"],
    }

    for function, parameter_names in parameters.items():
        signature = inspect.signature(function)
        for parameter_name in parameter_names:
            assert signature.parameters[parameter_name].default is None
