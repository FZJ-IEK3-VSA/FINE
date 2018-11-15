def readOptimizationOutputFromExcel(fileName='scenarioResults.xlsx'):
    """
    Reads optimization output from excel file

    :param fileName: excel file name or path (including .xlsx ending)
        |br| * the default value ist 'scenarioResults.xlsx'
    :type fileName: string

    :return:
    """

    utils.output('\nReading output from Excel ... ', esM.verbose, 0)

# Excel einlesen via Pandas -> Dataframe
    file = pd.ExcelFile(fileName)  # class 'pandas.io.excel.ExcelFile' -> all sheets included
    file = pd.read_excel(fileName, sheet_name=None) # class 'collections.OrderedDict' -> all sheets
    file = pd.read_excel(fileName)# class 'pandas.core.frame.DataFrame' -> one(!) sheet (first sheet of excel)


# Einlesen der zugehörigen esM erforderlich -> Input + Output müssen zusammenpassen!
# Ist es möglich, nur Output einzulesen und weiterzuverarbeiten? bzw. ist es sinnvoll, nicht auf die Inputparameter
# zurückzugreifen?


# Aus Funktion optimize (siehe energySystemModel.py):
#
for key, mdl in self.componentModelingDict.items():
    __t = time.time()
    mdl.setOptimalValues(self, self.pyM)
    outputString = ('for {:' + w + '}').format(key + ' ...') + "(%.4f" % (time.time() - __t) + "sec)"
    utils.output(outputString, self.verbose, 0)



# Funktion getOptimizationSummary (siehe energySystemModel.py):
    # Anzeigen der Optimierungszusammenfassung

    def getOptimizationSummary(self, modelingClass, outputLevel=0):
        """
        Function which returns the optimization summary (design variables, aggregated operation variables,
        objective contributions) of a modeling class

        :param modelingClass: name of the modeling class from which the optimization summary should be obtained
        :type modelingClass: string

        :param outputLevel: states the level of detail of the output summary:
            - 0: full optimization summary is returned
            - 1: full optimization summary is returned but rows in which all values are NaN (not a number) are dropped
            - 2: full optimization summary is returned but rows in which all values are NaN or 0 are dropped
            |br| * the default value is 0
        :type outputLevel: integer (0, 1 or 2)

        :returns: the attribute specified by the attributeName of the component with the name componentName
        :rtype: depends on the specified attribute
        """
        if outputLevel == 0:
            return self.componentModelingDict[modelingClass].optSummary
        elif outputLevel == 1:
            return self.componentModelingDict[modelingClass].optSummary.dropna(how='all')
        else:
            if outputLevel != 2 and self.verbose < 2:
                warnings.warn('Invalid input. An outputLevel parameter of 2 is assumed.')
            df = self.componentModelingDict[modelingClass].optSummary.dropna(how='all')
            return df.loc[((df != 0) & (~df.isnull())).any(axis=1)]


        ## Plotfunktionen
        # plotOperation -> componentModelingDict -> getOptimalValues
        data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(variableName)

        # plotOperationColorMap -> componentModelingDict -> getOptimalValues
        data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(variableName)

        # plotTransmission -> componentModelingDict -> getOptimalValues + getComponentAttribute!!
        data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(variableName)
        unit = esM.getComponentAttribute(compName, 'commodityUnit')

        # plotLocationalColorMap -> componentModelingDict -> getOptimalValues
        data = esM.componentModelingDict[esM.componentNames[compName]].getOptimalValues(variableName)


