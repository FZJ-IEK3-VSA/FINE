import inspect

import FINE as fn
from FINE.IOManagement import utilsIO


def exportToDict(esM, useProcessedValues=False):
    """
    Writes the input arguments of EnergySysteModel and its Components input to a dictionary.

    :param esM: EnergySystemModel instance in which the optimization model is held
    :type esM: EnergySystemModel instance

    :return: esmDict, compDict - dicts containing input arguments of
            EnergySysteModel and its Components input, respectively
    """

    # Get all input properties of the esM
    inputkwargs = inspect.getfullargspec(fn.EnergySystemModel.__init__)

    esmDict = {}
    # Loop over all props
    for arg in inputkwargs.args:
        if arg != "self":
            esmDict[arg] = getattr(esM, arg)

    compDict = utilsIO.PowerDict()
    # Loop over all component models
    for componentModel in esM.componentModelingDict.values():
        # Loop over all components belonging to the model
        for componentname in componentModel.componentsDict:
            # Get class name of component
            classname = type(componentModel.componentsDict[componentname]).__name__

            # Get class
            class_ = getattr(fn, classname)
            # Get input arguments of the class
            inputkwargs = inspect.getfullargspec(class_.__init__)
            prop_list = inputkwargs.args

            # Get component data
            component = componentModel.componentsDict[componentname]

            # replace raw values with processed values if useProcessedValues is True
            if useProcessedValues:
                prop_list_full_set = component.__dict__.keys()

                _prop_list = prop_list.copy()
                for prop in _prop_list:
                    # for the transmission investment parameters, the preprocessed version
                    # (without multiplication with the distance and 0.5) must be used
                    if (
                        f"preprocessed{prop[:1].upper()}{prop[1:]}"
                        in prop_list_full_set
                    ):
                        prop_list.remove(prop)
                        prop_list.append(
                            (prop, f"preprocessed{prop[:1].upper()}{prop[1:]}")
                        )

                    # for time series, the full version must be used
                    elif f"full{prop[:1].upper()}{prop[1:]}" in prop_list_full_set:
                        prop_list.remove(prop)
                        prop_list.append((prop, f"full{prop[:1].upper()}{prop[1:]}"))

                    # for all other components, the processed version is used, if it exists
                    elif f"processed{prop[:1].upper()}{prop[1:]}" in prop_list_full_set:
                        prop_list.remove(prop)
                        prop_list.append(
                            (prop, f"processed{prop[:1].upper()}{prop[1:]}")
                        )
                    else:
                        prop_list.remove(prop)
                        prop_list.append((prop, prop))

                # Loop over all input props
                for prop in prop_list:
                    if (prop[0] != "self") and (prop[0] != "esM"):
                        # NOTE: thanks to utilsIO.PowerDict(), the nested dictionaries need
                        # not be created before adding the data.
                        compDict[classname][componentname][prop[0]] = getattr(
                            component, prop[1]
                        )
            else:
                # Loop over all input props
                for prop in prop_list:
                    if (prop != "self") and (prop != "esM"):
                        compDict[classname][componentname][prop] = getattr(
                            component, prop
                        )

    return esmDict, compDict


def importFromDict(esmDict, compDict):
    """
    Converts the dictionaries created by the exportToDict function to an EnergySystemModel.

    :param esMDict: dictionary created from exportToDict contains all esM information
    :type dict: dictionary instance

    :param compDict: dictionary create from exportToDict containing all component information
    :type dict: dictionary instance

    :return: esM - EnergySystemModel instance in which the optimized model is held
    """
    esM = fn.EnergySystemModel(**esmDict)

    # add components
    for classname in compDict:
        # get class
        class_ = getattr(fn, classname)

        for comp in compDict[classname]:
            esM.add(class_(esM, **compDict[classname][comp]))

    return esM
