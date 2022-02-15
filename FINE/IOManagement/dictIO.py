import inspect

import FINE as fn
from FINE.IOManagement import utilsIO


def exportToDict(esM):
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
        if not arg is "self":
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

            # Get component data
            component = componentModel.componentsDict[componentname]

            # Loop over all input props
            for prop in inputkwargs.args:
                if (prop is not "self") and (prop is not "esM"):
                    # NOTE: thanks to utilsIO.PowerDict(), the nested dictionaries need
                    # not be created before adding the data.
                    compDict[classname][componentname][prop] = getattr(component, prop)

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
