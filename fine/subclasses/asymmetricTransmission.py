from fine.transmission import Transmission, TransmissionModel


class AsymmetricTransmission(Transmission):
    """An AsymmetricTransmission component shows the behavior of a Transmission component but allows
    direction-dependent capacities, i.e. the capacity from location_1 to location_2 can differ from the
    capacity from location_2 to location_1. The AsymmetricTransmission class inherits from the Transmission
    class.

    Differences to the Transmission component:

    * The capacities cap_(loc1,loc2) and cap_(loc2,loc1) are independent variables (no symmetry constraint).
    * The operation in each direction is limited by the capacity of that direction only
      (instead of the sum of both directions being limited by the common capacity).
    * Capacity-related input data (e.g. capacityMax, capacityFix, capacityMin) given as a DataFrame is
      interpreted per direction: ``data[loc1][loc2]`` refers to the connection from loc1 to loc2.
      Connections may also be eligible in one direction only.
    * Capacity-related costs (investPerCapacity, investIfBuilt, opexPerCapacity, opexIfBuilt) are, as in
      the Transmission component, multiplied by 0.5 * distance per direction. For equal capacities in both
      directions this yields exactly the costs of a Transmission component.
    """

    def __init__(
        self,
        esM,
        name,
        commodity,
        losses=0,
        distances=None,
        hasCapacityVariable=True,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=1,
        hasIsBuiltBinaryVariable=False,
        bigM=None,
        operationRateMax=None,
        operationRateFix=None,
        tsaWeight=1,
        locationalEligibility=None,
        capacityMin=None,
        capacityMax=None,
        partLoadMin=None,
        sharedPotentialID=None,
        linkedQuantityID=None,
        capacityFix=None,
        commissioningMin=None,
        commissioningMax=None,
        commissioningFix=None,
        isBuiltFix=None,
        investPerCapacity=0,
        investIfBuilt=0,
        opexPerOperation=0,
        opexPerCapacity=0,
        opexIfBuilt=0,
        QPcostScale=0,
        interestRate=0.08,
        economicLifetime=10,
        technicalLifetime=None,
        floorTechnicalLifetime=True,
        balanceLimitID=None,
        pathwayBalanceLimitID=None,
        stockCommissioning=None,
        pwlcfParameters=None,
    ):
        """Create an AsymmetricTransmission class instance.
        The input arguments are identical to the ones of the Transmission class. Capacity-related
        two-dimensional input data is interpreted per direction (see class docstring).
        """
        Transmission.__init__(
            self,
            esM,
            name,
            commodity,
            losses=losses,
            distances=distances,
            hasCapacityVariable=hasCapacityVariable,
            capacityVariableDomain=capacityVariableDomain,
            capacityPerPlantUnit=capacityPerPlantUnit,
            hasIsBuiltBinaryVariable=hasIsBuiltBinaryVariable,
            bigM=bigM,
            operationRateMax=operationRateMax,
            operationRateFix=operationRateFix,
            tsaWeight=tsaWeight,
            locationalEligibility=locationalEligibility,
            capacityMin=capacityMin,
            capacityMax=capacityMax,
            partLoadMin=partLoadMin,
            sharedPotentialID=sharedPotentialID,
            linkedQuantityID=linkedQuantityID,
            capacityFix=capacityFix,
            commissioningMin=commissioningMin,
            commissioningMax=commissioningMax,
            commissioningFix=commissioningFix,
            isBuiltFix=isBuiltFix,
            investPerCapacity=investPerCapacity,
            investIfBuilt=investIfBuilt,
            opexPerOperation=opexPerOperation,
            opexPerCapacity=opexPerCapacity,
            opexIfBuilt=opexIfBuilt,
            QPcostScale=QPcostScale,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
            technicalLifetime=technicalLifetime,
            floorTechnicalLifetime=floorTechnicalLifetime,
            balanceLimitID=balanceLimitID,
            pathwayBalanceLimitID=pathwayBalanceLimitID,
            stockCommissioning=stockCommissioning,
            pwlcfParameters=pwlcfParameters,
        )

        self.modelingClass = AsymmetricTransmissionModel


class AsymmetricTransmissionModel(TransmissionModel):
    """An AsymmetricTransmissionModel class instance will be instantly created if an AsymmetricTransmission
    class instance is initialized. It is used for the declaration of the sets, variables and constraints which
    are valid for the AsymmetricTransmission class instance. The AsymmetricTransmissionModel class inherits
    from the TransmissionModel class.
    """

    def __init__(self):
        super().__init__()
        self.abbrvName = "asymTrans"
        self.dimension = "2dim"

    ####################################################################################################################
    #                                          Declare component constraints                                           #
    ####################################################################################################################

    def symmetricalCapacity(self, pyM):
        """Overwrite the symmetry constraint of the Transmission component: the capacities of the two
        directions of a connection are independent of each other, so no constraint is declared.

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """

    def operationMode1_2dim(self, pyM, esM, constrName, constrSetName, opVarName):
        r"""Declare the constraint that the operation [commodityUnit*hour] in one direction is limited by the
        installed capacity [commodityUnit] of that direction multiplied by the hours per time step.

        .. math::

            op^{comp,op}_{(loc_1,loc_2),ip,p,t} \\leq \\tau^{hours} \\cdot \\text{cap}^{comp}_{(loc_1,loc_2),ip}

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        :param esM: EnergySystemModel instance representing the energy system in which the component should be modeled.
        :type esM: esM - EnergySystemModel class instance
        """
        self.operationMode1(pyM, esM, constrName, constrSetName, opVarName)