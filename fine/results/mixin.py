"""Post-processing of a solved component model.

Split out of :class:`fine.component.ComponentModel` so that the class keeps the model
formulation (sets, variables, constraints) as its subject. Everything here runs after the
solver: it reads the solved pyomo variables once, derives the economics from them and
assembles the optimization summary and the result accessors as views of those frames.

The methods are unchanged and still run as methods of the modeling class, because every
one of the seven modeling classes overrides one of the hooks
(``_extractSubclassRawResults``, ``_deriveSubclassEconomics``,
``_buildSubclassOptimizationSummary``, ...).
"""

import math
import warnings

import numpy as np
import pandas as pd

from fine import utils
from fine.enums import CostType, Dimension, VarType


class ComponentResultsMixin:
    """Result extraction, optimization summary and result accessors of a component model.

    Mixed into :class:`fine.component.ComponentModel`; not meant to be used on its own.
    The result state it owns is initialized in :meth:`__init__`.

    Attributes are populated by the result pipeline that
    :meth:`fine.energySystemModel.EnergySystemModel.optimize` runs:
    :meth:`extractRawResults`, :meth:`deriveEconomics` and
    :meth:`buildOptimizationSummary`.
    """

    def __init__(self):
        """Initialize the result state; empty until the model is solved."""
        super().__init__()
        self._capacityVariablesOptimum = {}
        self._commissioningVariablesOptimum = {}
        self._decommissioningVariablesOptimum = {}
        self._isBuiltVariablesOptimum = {}
        self._optSummary = {}
        # Raw results dict (single source of truth for the summary and the staged
        # raw-results export accessors), filled by extractRawResults/deriveEconomics during
        # optimize(). The current xarrayIO exporter continues to read the public summary and
        # getOptimalValues until its separate refactor lands. Empty until the model is solved.
        self._rawResults = {}
        self._rawResults1dim = {}
        # Additional summary rows contributed by expansion modules that run after
        # the result pipeline (currently the piecewise linear cost function, see
        # registerExtraSummaryRows). Keyed by investment period name.
        self._extraSummaryRows = {}

    def extractRawResults(self, esM, pyM):
        """Extract the raw solved variable values into a results dictionary.

        The function is called after a successful optimization. It reads the solved pyomo
        design variables (capacity, commissioning, decommissioning, isBuilt) once per
        investment period and formats them with :func:`utils.formatOptimizationOutput`. It
        deliberately performs *no* economics calculations, big-M warnings or summary
        assembly. Subclass specific variables (e.g. operation, charge/discharge, state of
        charge) are added through the overridable :meth:`_extractSubclassRawResults` hook.

        The existing ``self._*VariablesOptimum`` attributes are populated from the extracted
        values so that the public getter behavior is preserved.

        **Required arguments**

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel

        The results are stored on the component as ``self._rawResults`` (native dimension)
        and ``self._rawResults1dim`` (1-dim companion used by the economics/summary phase).
        Nothing is returned.
        """
        compDict, abbrvName = self.componentsDict, self.abbrvName
        capVar = getattr(esM.pyM, "cap_" + abbrvName)
        binVar = getattr(esM.pyM, "commisBin_" + abbrvName)
        commisVar = getattr(esM.pyM, "commis_" + abbrvName)
        decommisVar = getattr(esM.pyM, "decommis_" + abbrvName)

        # mapping of dict key -> (pyomo variable, attribute populated from it)
        designVars = [
            ("capacity", capVar, self._capacityVariablesOptimum),
            ("commissioning", commisVar, self._commissioningVariablesOptimum),
            ("decommissioning", decommisVar, self._decommissioningVariablesOptimum),
            ("isBuilt", binVar, self._isBuiltVariablesOptimum),
        ]

        rawResults = {}
        # 1-dim companion frames (used by the economics and summary phases)
        self._rawResults1dim = {}
        # drop rows contributed by expansion modules during a previous optimization
        self._extraSummaryRows = {}
        for ip in esM.investmentPeriods:
            ipName = esM.investmentPeriodNames[ip]
            rawResults[ipName] = {}
            self._rawResults1dim[ipName] = {}
            for varName, pyomoVar, optimumAttr in designVars:
                values = pyomoVar.get_values()
                optVal = utils.formatOptimizationOutput(
                    values, VarType.DESIGN, Dimension.ONE, ip
                )
                optVal_ = utils.formatOptimizationOutput(
                    values, VarType.DESIGN, self.dimension, ip, compDict=compDict
                )
                # NOTE (aliasing invariant): the ``self._*VariablesOptimum`` attribute and the
                # raw results dict deliberately share the *same* frame object (avoids doubling
                # the memory of the optima). ``self._rawResults`` is the single source of truth,
                # so these frames must be treated as immutable after optimization: never mutate
                # a ``*VariablesOptimum`` frame in place, or the summary/export read from
                # ``_rawResults`` would silently see the mutation. Readers copy before returning
                # (see :meth:`getResultOptimalValues` / :meth:`getResultSummaryDict`).
                optimumAttr[ipName] = optVal_
                rawResults[ipName][varName] = optVal_
                self._rawResults1dim[ipName][varName] = optVal

        self._rawResults = rawResults
        # let subclasses add their specific variables (operation, charge/discharge, ...)
        self._extractSubclassRawResults(esM, pyM, rawResults)

    def _extractSubclassRawResults(self, esM, pyM, rawResults):
        """Extract subclass specific raw variables (overridable hook).

        Subclasses add their variables (e.g. ``operation``) to ``rawResults`` in place and
        populate their own ``self._*VariablesOptimum`` attributes. The base implementation
        does nothing.

        :param esM: EnergySystemModel instance.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel.
        :type pyM: pyomo ConcreteModel

        :param rawResults: nested dictionary built by :meth:`extractRawResults`, keyed by
            investment period name then variable name. Mutated in place.
        :type rawResults: dict
        """
        pass

    def deriveEconomics(self, esM, pyM):
        """Derive the economic (cost) results and write them into ``self._rawResults``.

        Called after :meth:`extractRawResults` populated ``self._rawResults`` with the solved
        raw variables. It computes the cost rows that are shown in the optimization summary
        (``invest``, ``capexCap``, ``opexCap``, ``capexIfBuilt``, ``opexIfBuilt``,
        ``investLifetimeExtension``, ``revenueLifetimeShorteningResale``, ``TAC`` and
        ``NPVcontribution``) from the component parameters and the raw 1-dim variable frames
        (:attr:`_rawResults1dim`). The derived frames are stored into the *same*
        ``self._rawResults`` dictionary under keys that match the summary property rows, so
        that the later summary assembly is a mechanical write. Subclass specific cost
        contributions (e.g. a storage's charge/discharge opex) are added through the
        overridable :meth:`_deriveSubclassEconomics` hook. Nothing is returned; the results
        dictionary is mutated in place.

        **Required arguments**

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel which stores the mathematical formulation of the model.
        :type pyM: pyomo ConcreteModel
        """
        compDict = self.componentsDict

        # Get the design dependent cost contributions for all components. Each call
        # returns a dict keyed by investment period.
        resultsNPV_cx = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedInvestPerCapacity"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commis",
            divisorName="CCF",
            QPdivisorNames=["QPbound", "CCF"],
            getOptValue=True,
            getOptValueCostType=CostType.NPV,
        )

        resultsTAC_cx = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedInvestPerCapacity"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commis",
            divisorName="CCF",
            QPdivisorNames=["QPbound", "CCF"],
            getOptValue=True,
            getOptValueCostType=CostType.TAC,
        )

        resultsNPV_ox = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedOpexPerCapacity"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commis",
            QPdivisorNames=["QPbound"],
            getOptValue=True,
            getOptValueCostType=CostType.NPV,
        )

        resultsTAC_ox = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexPerCapacity", "QPcostDev"],
            QPfactorNames=["processedQPcostScale", "processedOpexPerCapacity"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commis",
            QPdivisorNames=["QPbound"],
            getOptValue=True,
            getOptValueCostType=CostType.TAC,
        )

        # Get NPV contribution for investmentIfBuilt
        resultsNPV_cx_bin = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestIfBuilt"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commisBin",
            divisorName="CCF",
            getOptValue=True,
            getOptValueCostType=CostType.NPV,
        )

        # Calculate the annualized investment costs cx (CAPEX) if built
        resultsTAC_cx_bin = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedInvestIfBuilt"],
            lifetimeAttr="ipEconomicLifetime",
            varName="commisBin",
            divisorName="CCF",
            getOptValue=True,
            getOptValueCostType=CostType.TAC,
        )

        # Get NPV cost contribution for the annualized operational costs if built ox (OPEX)
        resultsNPV_ox_bin = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexIfBuilt"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commisBin",
            getOptValue=True,
            getOptValueCostType=CostType.NPV,
        )

        # Calculate the annualized operational costs if built ox (OPEX)
        resultTAC_ox_bin = self.getEconomicsDesign(
            pyM,
            esM,
            factorNames=["processedOpexIfBuilt"],
            lifetimeAttr="ipTechnicalLifetime",
            varName="commisBin",
            getOptValue=True,
            getOptValueCostType=CostType.TAC,
        )

        for ip in esM.investmentPeriods:
            ipName = esM.investmentPeriodNames[ip]
            # write the economic frames into the same results dict
            results_ip = self._rawResults[ipName]

            # Read raw solved design variables (1-dim) extracted by extractRawResults
            commisOptVal = self._rawResults1dim[ipName]["commissioning"]
            capOptVal = self._rawResults1dim[ipName]["capacity"]
            binCapOptVal = self._rawResults1dim[ipName]["isBuilt"]

            if capOptVal is not None:
                # Calculate the investment costs i (proportional to commissioning)
                i = commisOptVal.apply(
                    lambda commis: (
                        commis
                        * compDict[commis.name].processedInvestPerCapacity[ip]
                        * compDict[commis.name].QPcostDev[ip]
                        + (
                            compDict[commis.name].processedInvestPerCapacity[ip]
                            * compDict[commis.name].processedQPcostScale[ip]
                            / (compDict[commis.name].QPbound[ip])
                            * commis
                            * commis
                        )
                    ),
                    axis=1,
                )
                results_ip["invest"] = i

                # Annualized investment (CAPEX) and operational (OPEX) costs due to
                # capacity expansion.
                results_ip["capexCap"] = resultsTAC_cx[ip]
                results_ip["opexCap"] = resultsTAC_ox[ip]

                # add additional costs for lifetime extension or scrapping bonus if lifetime
                # is floored or ceiled to next interval
                investLifetimeExtension = pd.DataFrame(
                    0.0, index=i.index, columns=i.columns
                )
                revenueLifetimeShorteningResale = pd.DataFrame(
                    0.0, index=i.index, columns=i.columns
                )
                for component in i.index:
                    for loc in i.columns:
                        # only relevant if there is any invest
                        if np.isnan(i.loc[component, loc]):
                            val_investLifetimeExtension = 0
                            val_revenueLifetimeShorteningResale = 0
                        else:
                            techLifetime = compDict[component].technicalLifetime[loc]
                            econLifetime = compDict[component].economicLifetime[loc]
                            sameInterval = math.floor(
                                compDict[component].ipTechnicalLifetime[loc]
                            ) == math.floor(compDict[component].ipEconomicLifetime[loc])

                            # investLifetimeExtension
                            if (
                                esM.numberOfInvestmentPeriods > 1
                                and (techLifetime % esM.investmentPeriodInterval != 0)
                                and not compDict[component].floorTechnicalLifetime
                            ):
                                intervalPart = 1 - (
                                    compDict[component].ipTechnicalLifetime[loc] % 1
                                )
                                val_investLifetimeExtension = (
                                    i.loc[component, loc]
                                    * intervalPart
                                    / compDict[component].ipEconomicLifetime[loc]
                                )
                            else:
                                val_investLifetimeExtension = 0

                            # revenueLifetimeShorteningResale
                            if (
                                esM.numberOfInvestmentPeriods > 1
                                and econLifetime % esM.investmentPeriodInterval != 0
                                and compDict[component].floorTechnicalLifetime
                                and sameInterval
                            ):
                                intervalPart = (
                                    compDict[component].ipEconomicLifetime[loc] % 1
                                )
                                val_revenueLifetimeShorteningResale = (
                                    i.loc[component, loc]
                                    * intervalPart
                                    / compDict[component].ipEconomicLifetime[loc]
                                )
                            else:
                                val_revenueLifetimeShorteningResale = 0

                        investLifetimeExtension.loc[component, loc] = (
                            val_investLifetimeExtension
                        )
                        revenueLifetimeShorteningResale.loc[component, loc] = (
                            val_revenueLifetimeShorteningResale
                        )

                results_ip["investLifetimeExtension"] = investLifetimeExtension
                results_ip["revenueLifetimeShorteningResale"] = (
                    revenueLifetimeShorteningResale
                )

            if binCapOptVal is not None:
                # Calculate the investment costs i (fix value if component is built)
                i_bin = binCapOptVal.apply(
                    lambda dec: dec * compDict[dec.name].processedInvestIfBuilt[ip],
                    axis=1,
                )
                # invest is the sum of capacity expansion and isBuilt contributions
                if "invest" in results_ip:
                    results_ip["invest"] = results_ip["invest"].add(i_bin, fill_value=0)
                else:
                    results_ip["invest"] = i_bin

                # Annualized investment (CAPEX) and operational (OPEX) costs if built
                results_ip["capexIfBuilt"] = resultsTAC_cx_bin[ip]
                results_ip["opexIfBuilt"] = resultTAC_ox_bin[ip]

            # Summarize all annualized contributions to the total annual cost. Mirrors the
            # former summary groupby (note: opexIfBuilt is intentionally not part of the TAC,
            # matching the previous behavior).
            tacParts = [
                results_ip[key]
                for key in ("capexCap", "opexCap", "capexIfBuilt")
                if key in results_ip
            ]
            if tacParts:
                results_ip["TAC"] = pd.concat(tacParts).groupby(level=0).sum()

            # Net present value contribution
            npv = pd.DataFrame()
            if capOptVal is not None:
                npv = npv.add(resultsNPV_cx[ip], fill_value=0)
                npv = npv.add(resultsNPV_ox[ip], fill_value=0)
            if binCapOptVal is not None:
                npv = npv.add(resultsNPV_cx_bin[ip], fill_value=0)
                npv = npv.add(resultsNPV_ox_bin[ip], fill_value=0)
            results_ip["NPVcontribution"] = npv

        # let subclasses add their specific cost contributions (e.g. charge/discharge opex)
        self._deriveSubclassEconomics(esM, pyM, self._rawResults)

        # Label the derived economic frames so the raw results dict is self-describing,
        # consistent with the solved-variable frames named in utils.formatOptimizationOutput.
        # All derived economics are flat ``component x location`` (2-dim connections are the
        # 1-dim pseudo-location convention) frames; the raw design/operation frames are already
        # named and either multi-level (2-dim, operation) or time-columned, so this sweep only
        # touches the flat component-indexed cost frames and needs no per-cost-term upkeep.
        for ipResults in self._rawResults.values():
            for frame in ipResults.values():
                if (
                    isinstance(frame, pd.DataFrame)
                    and frame.index.nlevels == 1
                    and frame.columns.nlevels == 1
                    and frame.index.name != "component"
                ):
                    frame.index = frame.index.set_names("component")
                    frame.columns = frame.columns.set_names("location")

    def _deriveSubclassEconomics(self, esM, pyM, rawResults):
        """Add subclass specific economic (cost) contributions (overridable hook).

        Subclasses add their cost frames (e.g. a storage's ``opexCharge``/``opexDischarge``)
        to ``rawResults`` in place and, where applicable, augment the aggregated ``TAC`` and
        ``NPVcontribution`` frames. The base implementation does nothing.

        :param esM: EnergySystemModel instance.
        :type esM: EnergySystemModel instance

        :param pyM: pyomo ConcreteModel.
        :type pyM: pyomo ConcreteModel

        :param rawResults: nested dictionary built by :meth:`extractRawResults` and extended
            by :meth:`deriveEconomics`, keyed by investment period name then variable name.
            Mutated in place.
        :type rawResults: dict
        """
        pass

    def _summaryIndexColumns(self, esM):
        """Column indices of the optimization summary, derived from the model's dimension.

        1-dim models are summarized per location, 2-dim models per connection between two
        locations (``"locIn_locOut"`` keys, see :meth:`_connectionLocationMap`).

        :param esM: EnergySystemModel instance (provides ``locations``).
        :type esM: EnergySystemModel instance

        :return: the column indices of the summary.
        :rtype: iterable of strings
        """
        if self.dimension == Dimension.TWO:
            return self._connectionLocationMap(esM).keys()
        return esM.locations

    def buildOptimizationSummary(self, esM):
        """Assemble the optimization summary and store it in ``self._optSummary``.

        Third and last phase of the result pipeline run by
        :meth:`fine.energySystemModel.EnergySystemModel.optimize`, after
        :meth:`extractRawResults` and :meth:`deriveEconomics` have populated
        ``self._rawResults``. It reads those frames only - no extraction, no economics.

        The summary is built in two steps: the design/economics rows shared by every
        component (:meth:`_buildOptimizationSummary`), followed by the component specific
        operation rows contributed by :meth:`_buildSubclassOptimizationSummary`. The
        summary units and column indices come from :meth:`_summaryPlantUnit` and
        :meth:`_summaryIndexColumns`, so a modeling class declares them once.

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance
        """
        plantUnit, unitApp = self._summaryPlantUnit()
        optSummaryBasic = self._buildOptimizationSummary(
            esM, self._summaryIndexColumns(esM), plantUnit, unitApp
        )
        self._optSummary = self._buildSubclassOptimizationSummary(esM, optSummaryBasic)

    def _convertOptimalValueNames(self, esM):
        """Rename the internal ``_*VariablesOptimum``/``_optSummary`` attributes to their
        public names (e.g. ``_capacityVariablesOptimum`` -> ``capacityVariablesOptimum``).

        For a perfect-foresight run the per-investment-period dict is kept as is; for a
        single-year optimization it is unwrapped to the one dataframe it contains, so that
        models built before the multi-investment-period support was added keep working.

        Called by :meth:`fine.energySystemModel.EnergySystemModel.optimize` once per
        modeling class, after :meth:`buildOptimizationSummary` has set ``self._optSummary``.
        It is driven from there rather than from the modeling classes themselves so that a
        class overriding a phase of the result pipeline cannot omit it.

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance
        """
        optimalValueParameters = [
            "_optSummary",
            "_stateOfChargeOperationVariablesOptimum",
            "_chargeOperationVariablesOptimum",
            "_dischargeOperationVariablesOptimum",
            "_phaseAngleVariablesOptimum",
            "_operationVariablesOptimum",
            "_discretizationPointVariablesOptimum",
            "_discretizationSegmentConVariablesOptimum",
            "_discretizationSegmentBinVariablesOptimum",
            "_capacityVariablesOptimum",
            "_isBuiltVariablesOptimum",
            "_commissioningVariablesOptimum",
            "_decommissioningVariablesOptimum",
        ]

        for key in optimalValueParameters:
            if key not in self.__dict__:
                continue
            # strip only the leading underscore; key.replace("_", "") would also drop
            # underscores inside a name
            publicName = key[1:]
            if esM.numberOfInvestmentPeriods == 1:
                setattr(
                    self,
                    publicName,
                    getattr(self, key)[esM.investmentPeriodNames[0]],
                )
            else:
                setattr(self, publicName, getattr(self, key))

    def _connectionLocationMap(self, esM):
        """Build (and cache) the ``"locIn_locOut" -> (locIn, locOut)`` map for 2-dim connection splitting.

        Built once per (component-model, location set) and rebuilt only if ``esM.locations``
        changes, avoiding the O(locations^2) rebuild on every summary/export call.

        :param esM: EnergySystemModel instance (provides ``locations``).
        :rtype: dict
        """
        cache = getattr(self, "_connLocMapCache", None)
        if cache is None or cache[0] != esM.locations:
            mapC = {
                l1 + "_" + l2: (l1, l2) for l1 in esM.locations for l2 in esM.locations
            }
            cache = (set(esM.locations), mapC)
            self._connLocMapCache = cache
        return cache[1]

    def _economicSummaryUnits(self, esM):
        """Property -> unit string for the derived economic summary rows.

        Single source for these units; consumed by both the optimization summary
        (:meth:`_buildOptimizationSummary`) and the export (:meth:`getResultSummaryDict`).

        :param esM: EnergySystemModel instance (provides ``costUnit``).
        :return: ordered ``{property: unitString}`` mapping.
        :rtype: dict
        """
        perA = "[" + esM.costUnit + "/a]"
        cost = "[" + esM.costUnit + "]"
        return {
            "capexCap": perA,
            "capexIfBuilt": perA,
            "opexCap": perA,
            "opexIfBuilt": perA,
            "TAC": perA,
            "NPVcontribution": cost,
            "invest": cost,
            "investLifetimeExtension": cost,
            "revenueLifetimeShorteningResale": cost,
        }

    def _buildOptimizationSummary(self, esM, indexColumns, plantUnit, unitApp=""):
        r"""Assemble the optimization summary as a view of ``self._rawResults``.

        Called by :meth:`buildOptimizationSummary` after :meth:`extractRawResults` and
        :meth:`deriveEconomics` have populated ``self._rawResults`` (raw solved variables and
        derived economic frames) and ``self._rawResults1dim`` (1-dim companions). This method
        performs no extraction or economics; it only writes the already computed frames into
        the ``(Component, Property, Unit) x Locations`` summary DataFrame, keyed per
        investment period. It also emits the big-M proximity warning for capacities close to
        the chosen big-M value.

        **Required arguments**

        :param esM: EnergySystemModel instance representing the energy system in which the
            components are modeled.
        :type esM: EnergySystemModel instance

        :param indexColumns: set of strings with the column indices of the summary (locations
            or connections between locations).
        :type indexColumns: set

        :param plantUnit: attribute of the component that describes the unit of the plants
            (e.g. "commodityUnit" or "physicalUnit").
        :type plantUnit: string

        **Default arguments**

        :param unitApp: string appended to the capacity unit in the summary (e.g. '\\*h' for
            storage). |br| * the default value is ''.
        :type unitApp: string

        :return: summary of the optimized values, keyed by investment period name.
        :rtype: dict
        """
        compDict = self.componentsDict

        # Single source of truth for the summary's (Property -> Unit) rows. The design rows
        # (capacity/commissioning/decommissioning) carry a per-component plant unit resolved
        # below and are marked with ``None``; every other unit is fixed. The economic units
        # are shared with the export via :meth:`_economicSummaryUnits`. The same mapping
        # drives both the MultiIndex and the economic-frame write loop further down.
        summaryUnits = {
            "capacity": None,
            "commissioning": None,
            "decommissioning": None,
            "isBuilt": "[-]",
            **self._economicSummaryUnits(esM),
        }
        # Design rows are written explicitly below (from the 1-dim frames, with their own
        # conditionals); the remaining rows are the economic frames derived by deriveEconomics.
        designProps = ("capacity", "commissioning", "decommissioning", "isBuilt")

        def resolveUnit(compName, prop):
            # ``None`` marks a capacity-like row whose unit is the component's plant unit.
            unit = summaryUnits[prop]
            if unit is None:
                unit = "[" + getattr(compDict[compName], plantUnit) + unitApp + "]"
            return unit

        mIndex = pd.MultiIndex.from_tuples(
            [
                (compName, prop, resolveUnit(compName, prop))
                for compName in compDict.keys()
                for prop in summaryUnits
            ],
            names=["Component", "Property", "Unit"],
        )

        optSummary = {}
        for ip in esM.investmentPeriods:
            ipName = esM.investmentPeriodNames[ip]
            optSummary_ip = pd.DataFrame(
                index=mIndex, columns=sorted(indexColumns)
            ).sort_index()
            # raw + derived economic frames produced by extractRawResults / deriveEconomics
            results_ip = self._rawResults[ipName]

            # Read raw solved design variables (1-dim) extracted by extractRawResults
            capOptVal = self._rawResults1dim[ipName]["capacity"]
            commisOptVal = self._rawResults1dim[ipName]["commissioning"]
            decommisOptVal = self._rawResults1dim[ipName]["decommissioning"]
            binCapOptVal = self._rawResults1dim[ipName]["isBuilt"]

            if capOptVal is not None:
                # Check if the installed capacities are close to a bigM val
                # ue for components with design decision variables but
                # ignores cases where bigM was substituted by capacityMax parameter (see bigM constraint
                for compName, comp in compDict.items():
                    if (
                        comp.hasIsBuiltBinaryVariable
                        and (comp.processedCapacityMax is None)
                        and capOptVal.loc[compName].max() >= comp.bigM * 0.9
                        and esM.verboseLogLevel < 2
                    ):
                        warnings.warn(
                            "the capacity of component "
                            + compName
                            + " is in one or more locations close "
                            + "or equal to the chosen Big M. Consider rerunning the simulation with a higher"
                            + " Big M."
                        )

                # Fill the optimization summary with the optimal capacities.
                optSummary_ip.loc[
                    [
                        (
                            ix,
                            "capacity",
                            "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                        )
                        for ix in capOptVal.index
                    ],
                    capOptVal.columns,
                ] = capOptVal.values

            # Fill the optimization summary with the isBuilt decisions.
            if binCapOptVal is not None:
                optSummary_ip.loc[
                    [(ix, "isBuilt", "[-]") for ix in binCapOptVal.index],
                    binCapOptVal.columns,
                ] = binCapOptVal.values

            # Get and set optimal values for commissioning and decommissioning
            # not applicable for singleyear optimization, hence dropped from summary
            # either decommissioning or capacity exists
            # (years can have decommissioning, leading to no left capacity)
            if decommisOptVal is not None or capOptVal is not None:
                # commissioning
                optSummary_ip.loc[
                    [
                        (
                            ix,
                            "commissioning",
                            "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                        )
                        for ix in commisOptVal.index
                    ],
                    commisOptVal.columns,
                ] = commisOptVal.values
                # decommissioning
                optSummary_ip.loc[
                    [
                        (
                            ix,
                            "decommissioning",
                            "[" + getattr(compDict[ix], plantUnit) + unitApp + "]",
                        )
                        for ix in decommisOptVal.index
                    ],
                    decommisOptVal.columns,
                ] = decommisOptVal.values

            # Fill the optimization summary with the derived economic frames (invest,
            # capexCap, opexCap, capexIfBuilt, opexIfBuilt, lifetime corrections, TAC and
            # NPVcontribution) computed by deriveEconomics.
            # The lifetime correction rows are written cell-wise to keep their numpy scalar
            # dtype (as in the former inline implementation).
            perCellProps = (
                "investLifetimeExtension",
                "revenueLifetimeShorteningResale",
            )
            for prop, unit in summaryUnits.items():
                if prop in designProps or prop not in results_ip:
                    continue
                frame = results_ip[prop]
                if frame.empty:
                    continue
                if prop in perCellProps:
                    for component in frame.index:
                        for loc in frame.columns:
                            optSummary_ip.loc[(component, prop, unit), loc] = frame.loc[
                                component, loc
                            ]
                else:
                    optSummary_ip.loc[
                        [(ix, prop, unit) for ix in frame.index],
                        frame.columns,
                    ] = frame.values

            optSummary[ipName] = optSummary_ip

        return optSummary

    def getOptimalValues(self, name="all", ip=0):
        """Return optimal values of the components.

        :param name: name of the variables of which the optimal values should be returned:

            * 'capacityVariablesOptimum',
            * 'isBuiltVariablesOptimum',
            * 'operationVariablesOptimum',
            * 'commissioningVariablesOptimum'
            * 'decommissioningVariablesOptimum'
            * 'all' or another input: all variables are returned.

        :type name: string

        The returned frames carry named axes: the index is ``component`` and ``location``
        (1-dim) or ``component``, ``locationIn`` and ``locationOut`` (2-dim), and for
        operation variables the columns are ``time``. Code that relies on the axes being
        unnamed (e.g. ``reset_index()`` producing ``level_0``/``level_1`` columns) has to
        be adapted.

        :param ip: investment period of transformation path analysis.
            |br| * the default value is 0
        :type ip: int

        :returns: a dictionary with the optimal values of the components
        :rtype: dict
        """
        timeDependentMapping = {
            "capacityVariablesOptimum": False,
            "isBuiltVariablesOptimum": False,
            "operationVariablesOptimum": True,
            "commissioningVariablesOptimum": False,
            "decommissioningVariablesOptimum": False,
        }

        if name in timeDependentMapping:
            return {
                "values": getattr(self, f"_{name}")[ip],
                "timeDependent": timeDependentMapping[name],
                "dimension": self.dimension,
            }
        return {
            valName: {
                "values": getattr(self, f"_{valName}")[ip],
                "timeDependent": timeDependentMapping[valName],
                "dimension": self.dimension,
            }
            for valName in timeDependentMapping
        }

    # ------------------------------------------------------------------
    # Results-dict accessors staged for the separate xarray/netCDF export refactor.
    # They read directly from the raw results dict (``self._rawResults`` /
    # ``self._rawResults1dim``), while the current xarrayIO exporter still reads the public
    # optimization summary and getOptimalValues. Once adopted by that refactor, these
    # accessors avoid reparsing the summary DataFrame.
    # ------------------------------------------------------------------

    def _requireRawResults(self, ip):
        """Return ``self._rawResults[ip]``, with an explanatory error if it is not available.

        The raw results dict is populated by :meth:`extractRawResults` /
        :meth:`deriveEconomics` during ``esM.optimize()``. It is deliberately *not*
        reconstructed when an EnergySystemModel is read back from a netCDF file
        (:func:`fine.IOManagement.xarrayIO.readNetCDFtoEnergySystemModel` restores the
        optimization summary and the ``*VariablesOptimum`` attributes only), so results loaded
        from a file cannot be exported again without re-optimizing.

        :param ip: investment period name (key into ``self._rawResults``).
        :type ip: string

        :return: the raw results of that investment period.
        :rtype: dict
        """
        if not self._rawResults:
            raise RuntimeError(
                f"No raw optimization results available for '{type(self).__name__}'. They are "
                "created by esM.optimize() and are not restored when an EnergySystemModel is "
                "read from a netCDF file - re-optimize the model before exporting its results."
            )
        if ip not in self._rawResults:
            raise KeyError(
                f"No raw optimization results for investment period '{ip}' in "
                f"'{type(self).__name__}'. Available: {sorted(self._rawResults)}."
            )
        return self._rawResults[ip]

    def _exportOptimumVarMap(self):
        """Map raw result keys to the optimum variable names used by the export.

        Each entry carries an explicit ``dimension`` so a variable can be shaped
        independently of the component's own dimension (e.g. the LOPF phase angle is
        1-dim on a 2-dim component). Subclasses extend this list with their own
        variables (see :meth:`fine.storage.StorageModel._exportOptimumVarMap`).

        :return: list of ``(rawResultsKey, optimumVariableName, timeDependent, dimension)``
            tuples in the same order/meaning as :meth:`getOptimalValues`.
        :rtype: list
        """
        d = self.dimension
        return [
            ("capacity", "capacityVariablesOptimum", False, d),
            ("isBuilt", "isBuiltVariablesOptimum", False, d),
            ("operation", "operationVariablesOptimum", True, d),
            ("commissioning", "commissioningVariablesOptimum", False, d),
            ("decommissioning", "decommissioningVariablesOptimum", False, d),
        ]

    def getResultOptimalValues(self, ip):
        """Return the design/operation optima for the export, read from ``self._rawResults``.

        Per component, each optimum variable is shaped into a ``Series`` ready for
        ``.to_xarray()`` (same structure / dimension names the export produced before). The
        values are the exact same frame objects the optimum attributes hold (both are populated
        from the same object in :meth:`extractRawResults`), so the export output is unchanged.
        These optima carry no unit, so the entries pair the series with ``None``.

        :param ip: investment period name (key into ``self._rawResults``).
        :type ip: string

        :return: ``{componentName: {optimumVariableName: (values, None)}}``.
        :rtype: dict
        """
        results_ip = self._requireRawResults(ip)
        out = {compName: {} for compName in self.componentsDict}
        for rawKey, optName, timeDependent, dimension in self._exportOptimumVarMap():
            frame = results_ip.get(rawKey)
            if frame is None:
                continue
            for compName in self.componentsDict:
                if compName not in frame.index.get_level_values(0):
                    continue
                series = self._shapeOptimumResult(
                    frame.loc[compName], optName, timeDependent, dimension
                )
                out[compName][optName] = (series, None)
        return out

    def _shapeOptimumResult(self, sub, name, timeDependent, dimension):
        """Shape a single component's optimum frame into a ``to_xarray``-ready ``Series``.

        Reproduces the per-case index handling the export applied to ``getOptimalValues`` output:
        time-dependent rows gain a ``time`` dimension; 2-dim rows are split into
        ``(locationIn, locationOut)`` (the time-independent 2-dim case keeps the historical
        transpose). The shaping uses the variable's own ``dimension`` (from
        :meth:`_exportOptimumVarMap`), which may differ from ``self.dimension`` (e.g. the LOPF
        phase angle is 1-dim on a 2-dim component).

        Variables that carry an extra index level beyond ``location`` (e.g. the part-load
        discretization point/segment variables, indexed by ``(discretizationIndex, location)``
        per component) keep that level so each variable exports under its own name instead of
        colliding on an anonymous stacked column. The extra level names are propagated from the
        frame (labelled in :func:`utils.formatOptimizationOutput`) rather than re-derived here.

        :param sub: the component slice ``frame.loc[component]``.
        :param name: variable name (becomes the data variable name).
        :param timeDependent: whether the variable carries a ``time`` dimension.
        :param dimension: ``"1dim"`` or ``"2dim"`` shaping to apply to this variable.

        :rtype: pandas.Series
        """
        if timeDependent and dimension == Dimension.ONE:
            subT = sub.T
            if subT.columns.nlevels == 1:
                series = subT.stack()
                series.index = series.index.rename(["time", "location"])
            else:
                # extra index levels (e.g. discretizationIndex) sit before location; stack
                # every column level so nothing is lost or collides on export, keeping the
                # level names set by formatOptimizationOutput.
                series = subT.stack(list(range(subT.columns.nlevels)))
                extraNames = list(sub.index.names[:-1])
                series.index = series.index.rename(["time", *extraNames, "location"])
        elif timeDependent and dimension == Dimension.TWO:
            series = sub.stack()
            series.index = series.index.rename(["locationIn", "locationOut", "time"])
            series = series.reorder_levels(["time", "locationIn", "locationOut"])
        elif not timeDependent and dimension == Dimension.ONE:
            series = sub.rename_axis("location")
        else:  # time-independent 2-dim
            series = sub.T.stack()
            series.index = series.index.rename(["locationIn", "locationOut"])
        series = series.copy()
        series.name = name
        return series

    def _summaryPlantUnit(self):
        """(plant unit attribute name, capacity unit suffix) for the design summary rows.

        :rtype: tuple(str, str)
        """
        return "commodityUnit", ""

    def _buildSubclassOptimizationSummary(self, esM, optSummaryBasic):
        """Add the subclass specific summary rows to the basic summary (overridable hook).

        Called by :meth:`buildOptimizationSummary` with the design/economics summary that
        every component shares. Subclasses prepend their operation rows (see e.g.
        :meth:`fine.conversion.ConversionModel._buildSubclassOptimizationSummary`) by reading
        the frames already present in ``self._rawResults``; they perform no extraction and no
        economics. The base implementation has no operation rows and passes the summary
        through unchanged.

        :param esM: EnergySystemModel instance.
        :type esM: EnergySystemModel instance

        :param optSummaryBasic: basic summary, keyed by investment period name.
        :type optSummaryBasic: dict

        :return: full optimization summary, keyed by investment period name.
        :rtype: dict
        """
        return optSummaryBasic

    def _subclassSummaryFrames(self, esM, ip):
        """Operation summary rows (per subclass) derived from ``self._rawResults``.

        These frames are the single source of the aggregated operation rows: they feed the
        optimization summary (written into the summary skeleton by
        :meth:`_writeOperationSummaryRows`) and the staged raw-results export accessor
        (:meth:`getResultSummaryDict`).

        :return: ordered list of ``(property, frame, unitFn)`` where ``frame`` is indexed by
            component with locations (1dim) / connections (2dim) as columns, and ``unitFn`` maps
            a component name to its unit string. The base class has no operation rows.
        :rtype: list
        """
        return []

    def _writeOperationSummaryRows(self, optSummary, esM, ipName):
        """Write the subclass operation rows (:meth:`_subclassSummaryFrames`) into the summary.

        Shared by every subclass' ``_buildSubclassOptimizationSummary`` so the operation
        aggregation is computed once (in :meth:`_subclassSummaryFrames`) and reused by the
        summary and staged raw-results export accessor.

        :param optSummary: summary skeleton with a ``(Component, Property, Unit) x columns``
            MultiIndex; filled in place.
        :param esM: EnergySystemModel instance.
        :param ipName: investment period name (key into ``self._rawResults``).

        :return: the ``{property: frame}`` mapping (so callers can reuse the aggregated frames,
            e.g. for the storage charge/discharge warning).
        :rtype: dict
        """
        framesByProp = {}
        for prop, frame, unit in self._subclassSummaryFrames(esM, ipName):
            framesByProp[prop] = frame
            if frame is None or frame.empty:
                continue
            optSummary.loc[
                [
                    (ix, prop, unit(ix) if callable(unit) else unit)
                    for ix in frame.index
                ],
                frame.columns,
            ] = frame.values
        return framesByProp

    def getResultSummaryDict(self, esM, ip):
        """Assemble the time-independent summary results for the export from ``self._rawResults``.

        Reproduces, per component, the variable -> (values, unit) entries the export previously
        obtained by re-parsing ``getOptimizationSummary``. Design rows come from
        ``self._rawResults1dim`` and the derived economic rows from ``self._rawResults``; the
        subclass operation rows are added through :meth:`_subclassSummaryFrames`. For 1-dim
        components a value is a per-location ``Series`` (NaN-filled where absent, matching the
        summary's fixed columns); for 2-dim components it is a ``Series`` indexed by
        ``(locationIn, locationOut)`` with absent/NaN connections dropped (matching the summary's
        ``stack``), or ``None`` when the whole property is absent.

        :param esM: EnergySystemModel instance.
        :type esM: EnergySystemModel instance

        :param ip: investment period name (key into ``self._rawResults``).
        :type ip: string

        :return: ``{componentName: {property: (values, unit)}}``.
        :rtype: dict
        """
        compDict = self.componentsDict
        results_ip = self._requireRawResults(ip)
        results1dim_ip = self._rawResults1dim[ip]
        plantUnit, unitApp = self._summaryPlantUnit()

        def plantUnitFn(compName, suffix):
            return "[" + getattr(compDict[compName], plantUnit) + suffix + "]"

        # (property, frame, unit) in the order the summary lists them. ``unit`` is either a
        # fixed string or a callable ``comp -> unit`` for the per-component plant units. Design
        # rows use the 1-dim companion frames; economic rows use the derived frames. The
        # economic units are shared with the summary via :meth:`_economicSummaryUnits`.
        designRows = [
            (
                "capacity",
                results1dim_ip.get("capacity"),
                lambda c: plantUnitFn(c, unitApp),
            ),
            (
                "commissioning",
                results1dim_ip.get("commissioning"),
                lambda c: plantUnitFn(c, unitApp),
            ),
            (
                "decommissioning",
                results1dim_ip.get("decommissioning"),
                lambda c: plantUnitFn(c, unitApp),
            ),
            ("isBuilt", results1dim_ip.get("isBuilt"), "[-]"),
        ]
        econRows = [
            (prop, results_ip.get(prop), unit)
            for prop, unit in self._economicSummaryUnits(esM).items()
        ]
        rows = designRows + econRows + self._subclassSummaryFrames(esM, ip)

        mapC = self._connectionLocationMap(esM)
        out = {compName: {} for compName in compDict}
        for compName in compDict:
            for prop, frame, unit in rows:
                values = self._extractComponentResult(frame, compName, esM, mapC)
                if values is None:
                    continue
                series = self._nameResultSeries(pd.to_numeric(values), prop)
                out[compName][prop] = (
                    series,
                    unit(compName) if callable(unit) else unit,
                )

        # Rows registered by expansion modules exist only for the components they apply to
        # (unlike the fixed property set above, which is NaN-filled for 1-dim components), so
        # they are only emitted where the component is actually present in the frame.
        for prop, frame, unit in self._extraSummaryRows.get(ip, []):
            for compName in compDict:
                if frame is None or compName not in frame.index:
                    continue
                values = self._extractComponentResult(frame, compName, esM, mapC)
                if values is None:
                    continue
                series = self._nameResultSeries(pd.to_numeric(values), prop)
                out[compName][prop] = (
                    series,
                    unit(compName) if callable(unit) else unit,
                )
        return out

    def registerExtraSummaryRows(self, ip, rows):
        """Publish result frames produced *after* the result pipeline into the results dict.

        Expansion modules that run once the component models are done (currently
        :class:`fine.expansionModules.piecewiseLinearCostFunction.PiecewiseLinearCostFunctionModel`,
        called from ``EnergySystemModel.optimize``) contribute additional result rows. They are
        stored in ``self._rawResults[ip]`` so the dict stays the single source of truth, and
        registered in ``self._extraSummaryRows`` so :meth:`getResultSummaryDict` exports them -
        the caller builds its optimization summary rows from the very same frames.

        :param ip: investment period name (key into ``self._rawResults``).
        :type ip: string

        :param rows: ``(property, frame, unit)`` triples - the same shape as
            :meth:`_subclassSummaryFrames` - where ``frame`` is indexed by component with one
            column per location and ``unit`` is a unit string or a ``component -> unit`` callable.
        :type rows: list

        :return: nothing; the frames are stored on the modeling class.
        """
        if not self._rawResults or ip not in self._rawResults or not rows:
            return
        if self.dimension != Dimension.ONE:
            # The frames are indexed by location, while a 2-dim model's result frames are
            # indexed by connection - there is no meaningful mapping, so nothing is published.
            warnings.warn(
                f"Extra result rows for the 2-dimensional modeling class "
                f"'{type(self).__name__}' cannot be added to the result export.",
                UserWarning,
            )
            return

        for prop, frame, _ in rows:
            self._rawResults[ip][prop] = frame
        self._extraSummaryRows.setdefault(ip, []).extend(rows)

    def _nameResultSeries(self, series, name):
        """Set the variable name and dimension-specific index names for the export.

        :param series: per-component result series (1dim: index = locations; 2dim: index =
            ``(locationIn, locationOut)`` tuples).
        :param name: variable name (becomes the data variable name after ``to_xarray``).

        :return: the same series, ready for ``.to_xarray()``.
        :rtype: pandas.Series
        """
        series = series.copy()
        series.name = name
        if self.dimension == Dimension.ONE:
            series.index = series.index.rename("location")
        else:
            series.index = series.index.rename(["locationIn", "locationOut"])
        return series

    def _extractComponentResult(self, frame, compName, esM, mapC):
        """Shape a class-level result frame into the per-component export values.

        :param frame: frame indexed by component, columns are locations (1dim) or connections
            (2dim); may be ``None``.
        :param compName: component name to extract.
        :param mapC: mapping ``"locIn_locOut" -> (locIn, locOut)`` for the 2-dim split.

        :return: per-location ``Series`` (1dim, NaN-filled), ``(locationIn, locationOut)``
            ``Series`` (2dim, NaN dropped) or ``None`` to skip the variable.
        :rtype: pandas.Series or None
        """
        if self.dimension == Dimension.ONE:
            locations = sorted(esM.locations)
            if frame is None or compName not in frame.index:
                return pd.Series(np.nan, index=locations)
            return frame.loc[compName].reindex(locations)
        # 2dim: split connection columns into (locationIn, locationOut), dropping NaN
        if frame is None or compName not in frame.index:
            return None
        row = frame.loc[compName]
        index, values = [], []
        for connection, value in row.items():
            if pd.isna(value):
                continue
            index.append(mapC[connection])
            values.append(value)
        if not index:
            return None
        return pd.Series(values, index=pd.MultiIndex.from_tuples(index))
