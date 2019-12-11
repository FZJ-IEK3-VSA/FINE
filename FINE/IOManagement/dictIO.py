import FINE as fn
import FINE.utils as utils

import inspect


def exportToDict(esM):
    """
    Writes the input arguments of EnergySysteModel and its Components input to a dictionary.

    :param esM: EnergySystemModel instance in which the optimized model is hold
    :type esM: EnergySystemModel instance
    """

    # Get all input properties of the esM
    inputkwargs = inspect.getfullargspec(fn.EnergySystemModel.__init__)

    esmDict = {}        
    # Loop over all props
    for arg in inputkwargs.args:
        if not arg is 'self':
            esmDict[arg] = getattr(esM,arg)

    compDict = utils.PowerDict()
    # Loop over all components
    for modelname in esM.componentModelingDict.keys():

        # Get all component models
        componentModel = esM.componentModelingDict[modelname]

        # Loop over all components belonging to the model
        for componentname in componentModel.componentsDict:
            
            # Get class name of component
            classname = type(componentModel.componentsDict[componentname]).__name__
            if not classname in compDict:
                compDict[classname] = utils.PowerDict()

            compDict[classname][componentname] = utils.PowerDict()
            component = componentModel.componentsDict[componentname]
            
            # Get class
            class_ = getattr(fn, classname)

            # Get input arguments of the class
            inputkwargs = inspect.getfullargspec(class_.__init__)

            # Loop over all input props
            for prop in inputkwargs.args:
                if (prop is not 'self') and (prop is not 'esM'):
                    compDict[classname][componentname][prop] = getattr(component,prop)

    return esmDict, compDict


def importFromDict(esmDict, compDict):
    """
    Writes the dictionaries created by the exportToDict function to an EnergySystemModel.

    :param esM: dictionary created from exportToDict contains all esM information
    :type esmDict: dictionary instance

    :param esM: dictionary create from exportToDict containing all component information
    :type esmDict: dictionary instance
    """
    
    # Init an new energy system model
    esM = fn.EnergySystemModel(**esmDict)

    # add components
    for classname in compDict:
        # get class
        class_ = getattr(fn, classname)

        for comp in compDict[classname]:            
            esM.add(class_(esM, **compDict[classname][comp]))

    return esM



