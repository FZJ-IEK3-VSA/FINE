import FINE as fn
import FINE.utils as utils

import inspect
import json

def exportToDict(esM):
    """
    Writes the input arguments of EnergySysteModel and its Components input to a dictionary.

    :param esM: EnergySystemModel instance in which the optimized model is held
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

def importFromDict(esmDict, compDict, esM=None):
    """
    Writes the dictionaries created by the exportToDict function to an EnergySystemModel.

    :param esMDict: dictionary created from exportToDict contains all esM information
    :type dict: dictionary instance

    :param compDict: dictionary create from exportToDict containing all component information
    :type dict: dictionary instance
    """

    if esM is None:
        # TODO: drop LinearOptimalPowerFlow ?
        esM = fn.EnergySystemModel(**esmDict)

    # add components
    for classname in compDict:
        # get class
        class_ = getattr(fn, classname)

        for comp in compDict[classname]:
            # if classname != 'LinearOptimalPowerFlow':
            esM.add(class_(esM, **compDict[classname][comp]))

    return esM

#NOTE: Writing the dicts to a json file throws error because there are some "set" type objects within these dicts which 
# are not json serializable. Only other way is to use pickle or cPickle package and save .txt files. However, this is 
# also throwing error at the moment
#TODO: check if it is absolutely necessary to save these dicts, if yes, try again with the above mentioned packages
# otherwise, just delete the functions!

# def exportToJSON(esM, filename):
#     esmDict, compDict = exportToDict(esM)

#     entireEsmDict = {'esmDict': esmDict, 'compDict': compDict}

#     with open(filename, 'w') as fp:
#         json.dump(entireEsmDict, fp)

# def importFromJSON(filename):

#     with open(filename, 'r') as fp:
#         entireEsmDict = json.load(fp)

#     esM = importFromDict(entireEsmDict['esmDict'], entireEsmDict['compDict'])

#     return esM