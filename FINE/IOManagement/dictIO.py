import FINE as fn
import FINE.utils as utils

import inspect


def exportToDict(esM):
    """
    Writes an optimization input to a dictionary.

    :param esM: EnergySystemModel instance in which the optimized model is hold
    :type esM: EnergySystemModel instance
    """

    # get all input properties of the esM
    inputkwargs = inspect.getfullargspec(fn.EnergySystemModel.__init__)

    esmDict = {}        
    # loop over all props
    for arg in inputkwargs.args:
        if not arg is 'self':
            esmDict[arg] = getattr(esM,arg)

    compDict = utils.PowerDict()
    # loop over all components
    for modelname in esM.componentModelingDict.keys():

        # get all component models
        componentModel = esM.componentModelingDict[modelname]

        # loop over all components belonging to the model
        for componentname in componentModel.componentsDict:
            
            # get class of component
            classname = type(componentModel.componentsDict[componentname]).__name__
            if not classname in compDict:
                compDict[classname] = utils.PowerDict()

            compDict[classname][componentname] = utils.PowerDict()
            component = componentModel.componentsDict[componentname]
            
            # get class
            class_ = getattr(fn, classname)

            # get input arguments of the class
            inputkwargs = inspect.getfullargspec(class_.__init__)

            # loop over all input props
            for prop in inputkwargs.args:
                if (prop is not 'self') and (prop is not 'esM'):
                    compDict[classname][componentname][prop] = getattr(component,prop)

    return esmDict, compDict


def importFromDict(esmDict, compDict, esM=None):
    """
    Writes the dictionaries to an EnergySystemModel.

    :param esMDict: dictionary created from exportToDict contains all esM information
    :type dict: dictionary instance

    :param compDict: dictionary create from exportToDict containing all component information
    :type dict: dictionary instance
    """

    if esM is None:
        esM = fn.EnergySystemModel(**esmDict)

    # add components
    for classname in compDict:
        # get class
        class_ = getattr(fn, classname)

        for comp in compDict[classname]:            
            esM.add(class_(esM, **compDict[classname][comp]))

    return esM



