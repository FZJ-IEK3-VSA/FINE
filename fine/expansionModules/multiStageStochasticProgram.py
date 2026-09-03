"""Multi-stage stochastic programming (MSSP) for FINE energy system models via mpi-sppy.

The module wraps FINE: a deterministic ``EnergySystemModel`` is
built with the normal FINE API, one independent copy of it is created per scenario with the
uncertain parameter values substituted, and the resulting pyomo models are handed to
mpi-sppy, which enforces non-anticipativity and solves the program either as an extensive
form (EF) or by progressive hedging (PH).

Uncertainty is described by a **scenario tree** given as a dictionary of nodes. Each node
carries its parent, its conditional probability given that parent, and the realization of
the uncertain parameters at that node. The leaves of the tree are the scenarios; a
scenario's probability is the product of the conditional probabilities along its path.
Because the data lives at the nodes, two scenarios that share a node necessarily share that
node's data, so non-anticipativity cannot be violated by inconsistent inputs.

Example::

    tree = {
        "ROOT": {
            "probability": 1.0,
            "values": {"Demand": {"operationRateFix": demand2020}},
        },
        "ROOT_0": {
            "parent": "ROOT", "probability": 0.7,
            "values": {"Demand": {"operationRateFix": demandHigh2025}},
        },
        "ROOT_1": {
            "parent": "ROOT", "probability": 0.3,
            "values": {"Demand": {"operationRateFix": demandLow2025}},
        },
        "ROOT_0_0": {"parent": "ROOT_0", "probability": 0.6, "values": {...}},
        "ROOT_0_1": {"parent": "ROOT_0", "probability": 0.4, "values": {...}},
        "ROOT_1_0": {"parent": "ROOT_1", "probability": 0.5, "values": {...}},
        "ROOT_1_1": {"parent": "ROOT_1", "probability": 0.5, "values": {...}},
    }

    results = optimizeMultiStageStochastic(esM, tree, solver="gurobi")
    results.objectiveValue
    results.scenarioModels["ROOT_0_0"].getOptimizationSummary("SourceSinkModel", ip=2020)

.. note::
    Only parameters that FINE holds per investment period can be made uncertain. Which
    ones those are is worked out from the components themselves at runtime, so it follows
    FINE rather than being listed here. This automatically excludes the parameters that
    describe what a component *is* (``hasCapacityVariable``, ``locationalEligibility``,
    ``commodity``, ``distances``, ...) as well as data that FINE stores for the whole
    horizon at once (``interestRate``, ``economicLifetime``, ...), which cannot differ
    between scenarios without making scenarios that still share a node disagree about the
    periods before they branch. On top of that the scenarios are compared against each
    other before solving, so a difference that does slip through and changes the model's
    design variables is reported rather than silently mismatched.

.. note::
    mpi-sppy is an optional dependency of FINE. Install it with
    ``pip install mpi-sppy``. Running in parallel additionally requires a working MPI
    runtime and ``mpi4py``; without those, mpi-sppy falls back to serial execution and this
    module works unchanged.

.. note::
    Per-node stage costs are reported as zero (see ``_attachScenarioTree``). This does not
    affect the optimal solution or the enforcement of non-anticipativity. mpi-sppy
    optimizes each scenario's own objective, but it does mean mpi-sppy's stage-cost and
    bound diagnostics are not meaningful. FINE builds its objective as a single aggregated
    expression per component modeling class, so a per-investment-period slice of it is not
    available without reimplementing FINE's cost accounting.
"""

import copy
import inspect
import logging
import warnings

import pandas as pd
import pyomo.environ as pyomo
from pyomo import opt

logger = logging.getLogger(__name__)

ROOT_NODE_NAME = "ROOT"

#: Prefixes under which FINE stores the normalized, per-investment-period form of a
#: constructor argument. Used to work out at runtime which parameters may be uncertain,
#: instead of maintaining a hard-coded list that would drift as FINE changes.
_PER_INVESTMENT_PERIOD_PREFIXES = ("processed", "full")

#: Prefixes of the design variables whose index sets have to agree across scenarios.
#: Operation variables are deliberately excluded: with time series aggregation enabled the
#: scenarios are clustered individually and legitimately end up with different typical
#: periods, which does not affect non-anticipativity because the investment decisions are
#: not time dependent.
_DESIGN_VARIABLE_PREFIXES = ("commis_", "commisBin_", "cap_", "decommis_")


def _requireMpisppy():
    """Import mpi-sppy on demand and return the pieces this module needs.

    :return: tuple of (ExtensiveForm, PH, ScenarioNode)
    :rtype: tuple
    """
    try:
        from mpisppy.opt.ef import ExtensiveForm  # noqa: PLC0415
        from mpisppy.opt.ph import PH  # noqa: PLC0415
        from mpisppy.scenario_tree import ScenarioNode  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Multi-stage stochastic optimization requires the optional dependency "
            "'mpi-sppy', which could not be imported. Install it with "
            "'pip install mpi-sppy'."
        ) from exc
    return ExtensiveForm, PH, ScenarioNode


class ScenarioTree:
    """A validated multi-stage scenario tree.

    The tree is built from a dictionary of nodes. Every node except the root declares its
    ``parent``; every node declares the conditional ``probability`` of reaching it given
    its parent, and optionally the ``values`` of the uncertain parameters realized at it.

    :param nodes: dictionary mapping node name to a node specification. A node
        specification is a dict with the keys

        * ``parent``: name of the parent node. Omitted (or None) for the root node.
        * ``probability``: conditional probability of this node given its parent. The
          root node's probability is 1.0. The conditional probabilities of all children
          of a node must sum to 1.
        * ``values``: optional, ``{componentName: {parameterName: value}}`` giving the
          realization of the uncertain parameters at this node. The value is either a
          single value, which is used for every investment period of this node's stage,
          or a dict keyed by investment period name for finer control.

    :type nodes: dict
    """

    def __init__(self, nodes):
        if not isinstance(nodes, dict) or not nodes:
            raise TypeError(
                "The scenario tree must be a non-empty dictionary of nodes."
            )

        self.nodes = {}
        for name, spec in nodes.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"Scenario tree node names must be strings, got {name!r}."
                )
            if not isinstance(spec, dict):
                raise TypeError(
                    f"The specification of scenario tree node '{name}' must be a dict."
                )
            unknownKeys = set(spec) - {"parent", "probability", "values"}
            if unknownKeys:
                raise ValueError(
                    f"Unknown keys {sorted(unknownKeys)} in scenario tree node '{name}'. "
                    "Allowed keys are 'parent', 'probability' and 'values'."
                )
            self.nodes[name] = {
                "parent": spec.get("parent"),
                "probability": spec.get("probability", 1.0),
                "values": spec.get("values", {}) or {},
            }

        self._validateStructure()
        self._validateProbabilities()

        self.stageOfNode = self._deriveStages()
        self.stageCount = max(self.stageOfNode.values())
        self._validateUniformDepth()

        self.leafNames = sorted(
            name for name in self.nodes if self.stageOfNode[name] == self.stageCount
        )
        self.nonLeafNames = sorted(
            (name for name in self.nodes if self.stageOfNode[name] < self.stageCount),
            key=lambda name: (self.stageOfNode[name], name),
        )
        self._deriveMpisppyNames()

    def _deriveMpisppyNames(self):
        """Translate the node names into the naming scheme mpi-sppy requires.

        mpi-sppy reconstructs the shape of the tree from the node names alone: the root
        has to be called ``ROOT`` and the children of a node ``X`` have to be called
        ``X_0``, ``X_1``, ... consecutively, because a node counts as a leaf exactly when
        ``X_0`` is not among the node names. Node names are therefore translated rather
        than taken over, which leaves the user free to name nodes meaningfully.

        Sets ``mpisppyNodeName`` (the translation), ``mpisppyNodeNames`` (all translated
        names, leaves included, in the depth-first order mpi-sppy assumes) and
        ``scenarioNames`` (the leaf names in that same depth-first order, which is the
        order in which mpi-sppy assigns scenarios to leaves).
        """
        self.mpisppyNodeName = {}
        self.mpisppyNodeNames = []
        self.scenarioNames = []

        def walk(nodeName, mpisppyName):
            self.mpisppyNodeName[nodeName] = mpisppyName
            self.mpisppyNodeNames.append(mpisppyName)
            children = self.childrenOfNode[nodeName]
            if not children:
                self.scenarioNames.append(nodeName)
            for index, child in enumerate(children):
                walk(child, f"{mpisppyName}_{index}")

        walk(self.rootName, ROOT_NODE_NAME)

    def _validateStructure(self):
        """Check that the nodes form a single tree rooted at exactly one node."""
        roots = [name for name, spec in self.nodes.items() if spec["parent"] is None]
        if len(roots) != 1:
            raise ValueError(
                "The scenario tree must have exactly one root node (a node without a "
                f"'parent' entry), but found {len(roots)}: {sorted(roots)}."
            )
        self.rootName = roots[0]

        for name, spec in self.nodes.items():
            parent = spec["parent"]
            if parent is not None and parent not in self.nodes:
                raise ValueError(
                    f"Scenario tree node '{name}' declares parent '{parent}', which is "
                    "not a node of the tree."
                )

        self.childrenOfNode = {name: [] for name in self.nodes}
        for name, spec in self.nodes.items():
            if spec["parent"] is not None:
                self.childrenOfNode[spec["parent"]].append(name)
        for children in self.childrenOfNode.values():
            children.sort()

        # Walking up from every node must terminate at the root, i.e. no cycles.
        for name in self.nodes:
            seen = set()
            current = name
            while current is not None:
                if current in seen:
                    raise ValueError(
                        f"The scenario tree contains a cycle involving node '{current}'."
                    )
                seen.add(current)
                current = self.nodes[current]["parent"]

    def _validateProbabilities(self):
        """Check that conditional probabilities are valid and that siblings sum to 1."""
        for name, spec in self.nodes.items():
            probability = spec["probability"]
            if not isinstance(probability, (int, float)) or isinstance(
                probability, bool
            ):
                raise TypeError(
                    f"The probability of scenario tree node '{name}' must be a number, "
                    f"got {probability!r}."
                )
            if not 0 <= probability <= 1:
                raise ValueError(
                    f"The conditional probability of scenario tree node '{name}' must "
                    f"lie between 0 and 1, got {probability}."
                )

        if abs(self.nodes[self.rootName]["probability"] - 1.0) > 1e-9:
            raise ValueError(
                f"The root node '{self.rootName}' must have probability 1.0, got "
                f"{self.nodes[self.rootName]['probability']}."
            )

        for name, children in self.childrenOfNode.items():
            if not children:
                continue
            total = sum(self.nodes[child]["probability"] for child in children)
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"The conditional probabilities of the children of scenario tree "
                    f"node '{name}' must sum to 1, got {total}. Children: "
                    f"{ {child: self.nodes[child]['probability'] for child in children} }."
                )

    def _deriveStages(self):
        """Return the 1-based stage of every node (the root node is stage 1)."""
        stageOfNode = {}

        def stageOf(name):
            if name not in stageOfNode:
                parent = self.nodes[name]["parent"]
                stageOfNode[name] = 1 if parent is None else stageOf(parent) + 1
            return stageOfNode[name]

        for name in self.nodes:
            stageOf(name)
        return stageOfNode

    def _validateUniformDepth(self):
        """Check that all leaves sit at the same stage.

        mpi-sppy expects every scenario to pass through the same number of stages, so
        trees with leaves at different depths are rejected rather than silently producing
        scenarios with inconsistent node lists.
        """
        shallowLeaves = sorted(
            name
            for name in self.nodes
            if not self.childrenOfNode[name]
            and self.stageOfNode[name] < self.stageCount
        )
        if shallowLeaves:
            raise ValueError(
                "All leaves of the scenario tree must be at the same stage "
                f"({self.stageCount}), but these leaves end earlier: {shallowLeaves}. "
                "Extend the tree so that every scenario spans all stages."
            )

    def pathTo(self, nodeName):
        """Return the list of node names from the root down to ``nodeName``, inclusive.

        :param nodeName: name of the node
        :type nodeName: string

        :return: node names ordered from the root to the given node
        :rtype: list of strings
        """
        path = []
        current = nodeName
        while current is not None:
            path.append(current)
            current = self.nodes[current]["parent"]
        return list(reversed(path))

    def probabilityOf(self, nodeName):
        """Return the unconditional probability of reaching a node.

        :param nodeName: name of the node
        :type nodeName: string

        :return: product of the conditional probabilities along the path to the node
        :rtype: float
        """
        probability = 1.0
        for name in self.pathTo(nodeName):
            probability *= self.nodes[name]["probability"]
        return probability

    def describe(self):
        """Return a human readable summary of the tree.

        :return: multi-line description of nodes, stages, probabilities and scenarios
        :rtype: string
        """
        lines = [
            f"Scenario tree with {self.stageCount} stages, "
            f"{len(self.nodes)} nodes and {len(self.leafNames)} scenarios:"
        ]
        for name in sorted(self.nodes, key=lambda n: (self.stageOfNode[n], n)):
            spec = self.nodes[name]
            parent = spec["parent"] if spec["parent"] is not None else "-"
            uncertain = sorted(
                f"{comp}.{param}"
                for comp, params in spec["values"].items()
                for param in params
            )
            lines.append(
                f"  stage {self.stageOfNode[name]:>2}  {name:<20} parent={parent:<20} "
                f"p={spec['probability']:.4f}  values={uncertain}"
            )
        lines.append("Scenarios (leaves) and their probabilities:")
        for leaf in self.leafNames:
            lines.append(f"  {leaf:<20} p={self.probabilityOf(leaf):.6f}")
        return "\n".join(lines)


def _defaultStageInvestmentPeriods(esM, stageCount):
    """Map each tree stage onto one investment period.

    :param esM: energy system model
    :type esM: EnergySystemModel instance

    :param stageCount: number of stages of the scenario tree
    :type stageCount: int

    :return: list with one list of investment period indices per stage
    :rtype: list of lists of int
    """
    if stageCount != esM.numberOfInvestmentPeriods:
        raise ValueError(
            f"The scenario tree has {stageCount} stages but the energy system model has "
            f"{esM.numberOfInvestmentPeriods} investment periods. Either match them, or "
            "pass 'stageInvestmentPeriods' to state explicitly which investment periods "
            "belong to which stage."
        )
    return [[ip] for ip in esM.investmentPeriods]


def _validateStageInvestmentPeriods(esM, stageInvestmentPeriods, stageCount):
    """Check an explicit stage-to-investment-period mapping.

    Stages may cover different numbers of investment periods, but together they must cover
    every investment period of the model exactly once and in ascending order.

    :param esM: energy system model
    :type esM: EnergySystemModel instance

    :param stageInvestmentPeriods: list with one list of investment period indices per stage
    :type stageInvestmentPeriods: list of lists of int

    :param stageCount: number of stages of the scenario tree
    :type stageCount: int

    :return: the validated mapping, as lists of int
    :rtype: list of lists of int
    """
    if len(stageInvestmentPeriods) != stageCount:
        raise ValueError(
            f"'stageInvestmentPeriods' describes {len(stageInvestmentPeriods)} stages "
            f"but the scenario tree has {stageCount} stages."
        )

    validated = []
    seen = []
    for stage, stagePeriods in enumerate(stageInvestmentPeriods):
        periods = (
            [stagePeriods] if isinstance(stagePeriods, int) else list(stagePeriods)
        )
        if not periods:
            raise ValueError(
                f"Stage {stage + 1} of 'stageInvestmentPeriods' is empty. Every stage "
                "must cover at least one investment period."
            )
        for ip in periods:
            if ip not in esM.investmentPeriods:
                raise ValueError(
                    f"Stage {stage + 1} of 'stageInvestmentPeriods' refers to investment "
                    f"period {ip}, which is not one of the model's investment periods "
                    f"{esM.investmentPeriods}."
                )
        validated.append(periods)
        seen.extend(periods)

    if sorted(seen) != list(esM.investmentPeriods):
        raise ValueError(
            "'stageInvestmentPeriods' must cover every investment period of the model "
            f"exactly once. Expected {list(esM.investmentPeriods)}, got {sorted(seen)}."
        )
    if seen != sorted(seen):
        raise ValueError(
            "The investment periods in 'stageInvestmentPeriods' must be in ascending "
            f"order across stages, got {seen}."
        )
    return validated


def _perInvestmentPeriodParameters(esM, component):
    """Return the constructor arguments that FINE stores per investment period.

    Determined at runtime from the component itself rather than from a hard-coded list, so
    that it follows FINE as parameters are added, renamed or removed: a parameter counts as
    a per-investment-period parameter exactly when the component holds a normalized
    counterpart of it (``processedX`` or ``fullX``) that is a dictionary keyed by
    investment period.

    :param esM: energy system model
    :type esM: EnergySystemModel instance

    :param component: the component to inspect
    :type component: Component instance

    :return: names of the constructor arguments that may vary by investment period
    :rtype: set of strings
    """
    investmentPeriods = set(esM.investmentPeriods)
    perInvestmentPeriod = set()
    for parameterName in inspect.signature(type(component)).parameters:
        if parameterName in ("self", "esM"):
            continue
        capitalized = parameterName[:1].upper() + parameterName[1:]
        for prefix in _PER_INVESTMENT_PERIOD_PREFIXES:
            value = getattr(component, f"{prefix}{capitalized}", None)
            if isinstance(value, dict) and set(value) >= investmentPeriods:
                perInvestmentPeriod.add(parameterName)
                break
    return perInvestmentPeriod


def _validateTreeValues(esM, tree, stageInvestmentPeriods):
    """Check that the values in the scenario tree refer to real, uncertain-able parameters.

    Only parameters that FINE stores per investment period may be uncertain, which is
    determined from the component itself by :func:`_perInvestmentPeriodParameters`. This
    rules out the parameters that describe the structure of a component
    (``hasCapacityVariable``, ``locationalEligibility``, ``commodity``, ``distances``, ...)
    without needing to enumerate them, and it also rules out data that FINE holds for the
    whole horizon at once, such as ``interestRate`` or ``economicLifetime``: a single value
    covering every investment period cannot differ between scenarios without making
    scenarios that still share a node disagree about the periods before they branch.

    Also checks completeness: if a component parameter is uncertain, every stage must
    supply a value for it, so that the assembled per-investment-period dictionary covers
    all investment periods of the model. FINE requires per-investment-period parameters to
    be given for every investment period.

    :param esM: energy system model
    :type esM: EnergySystemModel instance

    :param tree: the scenario tree
    :type tree: ScenarioTree instance

    :param stageInvestmentPeriods: stage to investment period mapping
    :type stageInvestmentPeriods: list of lists of int
    """
    uncertainParameters = set()
    for nodeName, spec in tree.nodes.items():
        for compName, parameters in spec["values"].items():
            if compName not in esM.componentNames:
                raise ValueError(
                    f"Scenario tree node '{nodeName}' sets values for component "
                    f"'{compName}', which is not part of the energy system model. "
                    f"Known components: {sorted(esM.componentNames)}."
                )
            if not isinstance(parameters, dict):
                raise TypeError(
                    f"The values of component '{compName}' in scenario tree node "
                    f"'{nodeName}' must be a dict of parameter name to value."
                )
            component = esM.getComponent(compName)
            componentClass = type(component)
            classParameters = set(inspect.signature(componentClass).parameters)
            allowedParameters = _perInvestmentPeriodParameters(esM, component)
            for parameterName in parameters:
                if parameterName not in classParameters:
                    raise ValueError(
                        f"'{parameterName}' (set in scenario tree node '{nodeName}') is "
                        f"not a constructor argument of {componentClass.__name__}, the "
                        f"class of component '{compName}'."
                    )
                if parameterName not in allowedParameters:
                    raise ValueError(
                        f"'{parameterName}' of component '{compName}' (set in scenario "
                        f"tree node '{nodeName}') cannot be uncertain: FINE does not hold "
                        "it per investment period, so it describes either the structure of "
                        "the component or a property of the whole time horizon, and a "
                        "single value cannot differ between scenarios that still share a "
                        "node. Parameters that may be uncertain for this component are: "
                        f"{sorted(allowedParameters)}."
                    )
                uncertainParameters.add((compName, parameterName))

    # Completeness: every uncertain parameter must be set at every node, so that each
    # scenario ends up with a value for every investment period.
    for compName, parameterName in sorted(uncertainParameters):
        for nodeName in sorted(tree.nodes):
            if parameterName not in tree.nodes[nodeName]["values"].get(compName, {}):
                raise ValueError(
                    f"Parameter '{parameterName}' of component '{compName}' is uncertain "
                    f"(it is set in at least one node of the scenario tree) but node "
                    f"'{nodeName}' does not provide a value for it. Every node must "
                    "provide a value for every uncertain parameter, so that each scenario "
                    "has a value for every investment period."
                )


def _assembleScenarioValues(esM, tree, leafName, stageInvestmentPeriods):
    """Collect the parameter values a scenario sees, walking the tree from root to leaf.

    Each node on the path contributes the values for the investment periods of its own
    stage. The result is keyed by investment period *name* (the calendar year), which is
    the form FINE expects for per-investment-period parameters.

    :param esM: energy system model
    :type esM: EnergySystemModel instance

    :param tree: the scenario tree
    :type tree: ScenarioTree instance

    :param leafName: name of the leaf node identifying the scenario
    :type leafName: string

    :param stageInvestmentPeriods: stage to investment period mapping
    :type stageInvestmentPeriods: list of lists of int

    :return: ``{componentName: {parameterName: {investmentPeriodName: value}}}``
    :rtype: dict
    """
    assembled = {}
    for stageIndex, nodeName in enumerate(tree.pathTo(leafName)):
        periods = stageInvestmentPeriods[stageIndex]
        for compName, parameters in tree.nodes[nodeName]["values"].items():
            for parameterName, value in parameters.items():
                target = assembled.setdefault(compName, {}).setdefault(
                    parameterName, {}
                )
                for ip in periods:
                    ipName = esM.investmentPeriodNames[ip]
                    if isinstance(value, dict) and ipName in value:
                        target[ipName] = value[ipName]
                    elif isinstance(value, dict):
                        raise ValueError(
                            f"Node '{nodeName}' provides '{parameterName}' of component "
                            f"'{compName}' as a dict, but it has no entry for investment "
                            f"period '{ipName}', which belongs to this node's stage. "
                            f"Expected entries for {[esM.investmentPeriodNames[p] for p in periods]}."
                        )
                    else:
                        target[ipName] = value
    return assembled


def _componentParameterSnapshot(component):
    """Return the component's constructor arguments as currently stored on it.

    :param component: the component to inspect
    :type component: Component instance

    :return: mapping of constructor argument name to its stored value
    :rtype: dict
    """
    parameters = inspect.signature(type(component)).parameters
    return {
        name: getattr(component, name)
        for name in parameters
        if name not in ("self", "esM") and hasattr(component, name)
    }


def _valuesEqual(left, right):  # noqa: PLR0911
    """Compare two parameter values, handling pandas objects and None.

    :return: True if the two values are considered equal
    :rtype: bool
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (pd.DataFrame, pd.Series)) or isinstance(
        right, (pd.DataFrame, pd.Series)
    ):
        if type(left) is not type(right):
            return False
        return left.equals(right)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_valuesEqual(left[key], right[key]) for key in left)
    try:
        return bool(left == right)
    except ValueError:
        # Ambiguous truth value of an array-like comparison.
        return False


def _applyScenarioValues(esM, scenarioValues):
    """Replace the uncertain parameter values of a scenario's energy system model.

    Uses :meth:`EnergySystemModel.updateComponent`, which rebuilds the affected component
    through its constructor. All of FINE's input checks therefore run again and all
    derived quantities (``processed*`` attributes, capital charge factors, lifetimes in
    investment period units) are recomputed for the new values.

    :param esM: the scenario's own energy system model, modified in place
    :type esM: EnergySystemModel instance

    :param scenarioValues: ``{componentName: {parameterName: value}}`` for this scenario
    :type scenarioValues: dict
    """
    for compName in sorted(scenarioValues):
        parameters = scenarioValues[compName]
        before = _componentParameterSnapshot(esM.getComponent(compName))

        with warnings.catch_warnings():
            # updateComponent re-adds the component under the same name, which FINE
            # reports as an overwrite. That is exactly what is intended here.
            warnings.filterwarnings(
                "ignore", message=".*Data will be overwritten.*", category=UserWarning
            )
            esM.updateComponent(compName, dict(parameters))

        after = _componentParameterSnapshot(esM.getComponent(compName))

        # updateComponent rebuilds the component from the constructor arguments it can
        # find as attributes. A constructor argument that a subclass does not expose is
        # silently replaced by its default (this was a real defect for
        # LinearOptimalPowerFlow, fixed in commit 3a8b92be). Fail loudly if that happens
        # rather than silently optimizing a different model than the user described.
        lost = sorted(
            name
            for name in set(before) | set(after)
            if name not in parameters
            and not _valuesEqual(before.get(name), after.get(name))
        )
        if lost:
            raise RuntimeError(
                f"Updating component '{compName}' unintentionally changed the "
                f"parameters {lost}. This happens when a constructor argument of "
                f"{type(esM.getComponent(compName)).__name__} is not stored under the "
                "same name on the component, so 'EnergySystemModel.updateComponent' "
                "cannot carry it over. Please report this as a FINE bug."
            )


def buildScenarioEnergySystemModel(
    baseModel,
    tree,
    leafName,
    stageInvestmentPeriods,
    timeSeriesAggregation=False,
    temporalAggregationSpecs=None,
):
    """Create the energy system model of a single scenario.

    The base model is copied, the scenario's uncertain parameter values are substituted,
    and only afterwards is the pyomo model declared. Substituting values before declaring
    is the only correct order: FINE does not use ``pyomo.Param`` anywhere, it writes the
    parameter values directly into the constraint and objective expressions while
    declaring, so a model that has already been declared can no longer be re-parameterized.
    Building in this order also means every scenario passes through FINE's regular input
    checks.

    :param baseModel: the deterministic energy system model to derive the scenario from
    :type baseModel: EnergySystemModel instance

    :param tree: the scenario tree
    :type tree: ScenarioTree instance

    :param leafName: name of the leaf node identifying the scenario
    :type leafName: string

    :param stageInvestmentPeriods: stage to investment period mapping
    :type stageInvestmentPeriods: list of lists of int

    **Default arguments:**

    :param timeSeriesAggregation: whether the scenario should be temporally aggregated
        before the optimization problem is declared. The aggregation runs per scenario and
        after the scenario values have been substituted, because adding or updating a
        component invalidates any existing clustering.
        |br| * the default value is False
    :type timeSeriesAggregation: boolean

    :param temporalAggregationSpecs: keyword arguments passed on to
        ``EnergySystemModel.aggregateTemporally`` when ``timeSeriesAggregation`` is True
        |br| * the default value is None
    :type temporalAggregationSpecs: dict or None

    :return: the scenario's energy system model, with its pyomo model declared
    :rtype: EnergySystemModel instance
    """
    # Do not drag an already declared pyomo model into the copy. The attribute is restored
    # on the original immediately afterwards.
    originalPyM = baseModel.pyM
    baseModel.pyM = None
    try:
        scenarioModel = copy.deepcopy(baseModel)
    finally:
        baseModel.pyM = originalPyM

    scenarioValues = _assembleScenarioValues(
        scenarioModel, tree, leafName, stageInvestmentPeriods
    )
    _applyScenarioValues(scenarioModel, scenarioValues)

    if timeSeriesAggregation:
        scenarioModel.aggregateTemporally(**(temporalAggregationSpecs or {}))
    else:
        # EnergySystemModel.optimize sets this before declaring the problem, and the
        # result processing relies on it. This module declares the problem itself, so it
        # has to do the same.
        scenarioModel.segmentation = False

    scenarioModel.declareOptimizationProblem(
        timeSeriesAggregation=timeSeriesAggregation
    )
    return scenarioModel


def _nonAnticipativeVariables(
    esM, investmentPeriods, includeIsBuiltBinaryVariables=True
):
    """Collect the investment decision variables of the given investment periods.

    The commissioning variables ``commis_<abbrvName>[loc, comp, ip]`` are the decisions
    that have to be identical across scenarios that cannot yet be told apart. Where a
    component uses a binary is-built decision, ``commisBin_<abbrvName>`` is an independent
    decision as well and is included too. The decommissioning variables are *not*
    included: FINE ties them to the commissioning variables by an equality constraint, so
    they inherit non-anticipativity automatically.

    The variables are returned in a deterministic order (modeling class, then component,
    then location, then investment period) because mpi-sppy matches them positionally
    across scenarios.

    :param esM: the scenario's energy system model with a declared pyomo model
    :type esM: EnergySystemModel instance

    :param investmentPeriods: investment periods belonging to the node
    :type investmentPeriods: iterable of int

    :param includeIsBuiltBinaryVariables: whether binary is-built decisions are treated as
        non-anticipative as well
        |br| * the default value is True
    :type includeIsBuiltBinaryVariables: boolean

    :return: the pyomo variable data objects of this node
    :rtype: list
    """
    pyM = esM.pyM
    periods = set(investmentPeriods)
    variables = []

    variableNames = ["commis_"]
    if includeIsBuiltBinaryVariables:
        variableNames.append("commisBin_")

    for modelingClassName in sorted(esM.componentModelingDict):
        abbrvName = esM.componentModelingDict[modelingClassName].abbrvName
        for prefix in variableNames:
            variable = getattr(pyM, prefix + abbrvName, None)
            if variable is None:
                continue
            for key in sorted(variable):
                # Negative investment periods hold the historical stock, which is fixed
                # input data rather than a decision.
                if key[2] in periods:
                    variables.append(variable[key])
    return variables


def _checkSolverStatus(solverResults):
    """Raise a readable error if the extensive form was not solved to optimality.

    Without this, an infeasible scenario only surfaces much later as a pyomo error about
    uninitialized variables, which says nothing about what went wrong.

    :param solverResults: the results object returned by the solver
    :type solverResults: pyomo SolverResults instance or None

    :raises RuntimeError: if the solver did not report an optimal solution
    """
    if solverResults is None or not hasattr(solverResults, "solver"):
        return
    terminationCondition = solverResults.solver.termination_condition
    if terminationCondition == opt.TerminationCondition.optimal:
        return
    if terminationCondition in (
        opt.TerminationCondition.infeasible,
        opt.TerminationCondition.infeasibleOrUnbounded,
        opt.TerminationCondition.unbounded,
    ):
        raise RuntimeError(
            f"The extensive form is {terminationCondition}. Every scenario has to be "
            "solvable on its own, so check whether the parameter values of one of the "
            "scenarios make the system infeasible, for instance a scenario whose demand "
            "cannot be met, or whose capacity limits are too tight."
        )
    warnings.warn(
        "The extensive form was not solved to optimality "
        f"(termination condition: {terminationCondition}). The results are those of the "
        "best solution found so far."
    )


def _designVariableSignature(esM):
    """Describe the design variable index sets of a scenario.

    mpi-sppy matches the non-anticipative variables of two scenarios by their position in
    the node's variable list, so the scenarios have to agree on which design variables
    exist and in which order they are collected. This signature captures exactly that, and
    is compared across scenarios by :func:`_verifyStructuralConsistency`.

    :param esM: a scenario's energy system model with a declared pyomo model
    :type esM: EnergySystemModel instance

    :return: for every modeling class and design variable, the ordered index keys
    :rtype: tuple
    """
    signature = []
    for modelingClassName in sorted(esM.componentModelingDict):
        abbrvName = esM.componentModelingDict[modelingClassName].abbrvName
        for prefix in _DESIGN_VARIABLE_PREFIXES:
            variable = getattr(esM.pyM, prefix + abbrvName, None)
            if variable is None:
                continue
            signature.append((modelingClassName, prefix, tuple(sorted(variable))))
    return tuple(signature)


def _verifyStructuralConsistency(scenarioModels):
    """Check that all scenarios describe the same system, differing only in data.

    This is what actually guarantees that mpi-sppy's positional matching of the
    non-anticipative variables is meaningful. It compares the scenarios themselves rather
    than reasoning about which input parameters might have changed the structure, so it
    also covers parameters that :data:`STRUCTURAL_PARAMETERS` does not list, including
    ones added to FINE after this module was written.

    :param scenarioModels: mapping of scenario name to the scenario's energy system model
    :type scenarioModels: dict

    :raises ValueError: if two scenarios differ in their design variable index sets
    """
    reference = None
    referenceName = None
    for scenarioName in sorted(scenarioModels):
        signature = _designVariableSignature(scenarioModels[scenarioName])
        if reference is None:
            reference, referenceName = signature, scenarioName
            continue
        if signature == reference:
            continue

        # Report the first difference concretely rather than dumping both signatures.
        referenceEntries = {(entry[0], entry[1]): entry[2] for entry in reference}
        scenarioEntries = {(entry[0], entry[1]): entry[2] for entry in signature}
        details = []
        for key in sorted(set(referenceEntries) | set(scenarioEntries)):
            expected = set(referenceEntries.get(key, ()))
            actual = set(scenarioEntries.get(key, ()))
            missing = sorted(expected - actual)[:3]
            added = sorted(actual - expected)[:3]
            if missing or added:
                details.append(
                    f"{key[1]}{key[0]}: missing {missing}, unexpected {added}"
                )
        raise ValueError(
            f"Scenarios '{referenceName}' and '{scenarioName}' do not describe the same "
            "system: their design variables differ. Scenarios may only differ in the "
            "values of their parameters, because mpi-sppy matches the non-anticipative "
            "variables of the scenarios by position. Differences found: "
            + "; ".join(details)
        )


def _attachScenarioTree(
    esM,
    tree,
    leafName,
    stageInvestmentPeriods,
    includeIsBuiltBinaryVariables=True,
):
    """Attach the mpi-sppy scenario tree information to a scenario's pyomo model.

    One :class:`mpisppy.scenario_tree.ScenarioNode` is created for every non-leaf node on
    the scenario's path. mpi-sppy does not use explicit leaf nodes; the decisions of the
    last stage stay private to the scenario.

    The per-node cost expression is set to zero. mpi-sppy uses it for stage-cost
    bookkeeping and bound reporting only -- what it actually optimizes is each scenario's
    own objective -- so the optimal solution and the non-anticipativity constraints are
    unaffected, but stage-wise cost diagnostics are not meaningful. FINE aggregates its
    objective into a single expression per component modeling class, so a per-stage slice
    of it is not available without reimplementing FINE's cost accounting.

    :param esM: the scenario's energy system model with a declared pyomo model
    :type esM: EnergySystemModel instance

    :param tree: the scenario tree
    :type tree: ScenarioTree instance

    :param leafName: name of the leaf node identifying the scenario
    :type leafName: string

    :param stageInvestmentPeriods: stage to investment period mapping
    :type stageInvestmentPeriods: list of lists of int

    :param includeIsBuiltBinaryVariables: whether binary is-built decisions are treated as
        non-anticipative as well
        |br| * the default value is True
    :type includeIsBuiltBinaryVariables: boolean
    """
    _, _, ScenarioNode = _requireMpisppy()

    pyM = esM.pyM
    path = tree.pathTo(leafName)
    nodeList = []

    for stageIndex, nodeName in enumerate(path[:-1]):
        variables = _nonAnticipativeVariables(
            esM,
            stageInvestmentPeriods[stageIndex],
            includeIsBuiltBinaryVariables=includeIsBuiltBinaryVariables,
        )

        costExpressionName = f"_msspStageCost_{tree.mpisppyNodeName[nodeName]}"
        if not hasattr(pyM, costExpressionName):
            setattr(pyM, costExpressionName, pyomo.Expression(expr=0.0))

        parentName = tree.nodes[nodeName]["parent"]
        nodeList.append(
            ScenarioNode(
                name=tree.mpisppyNodeName[nodeName],
                cond_prob=tree.nodes[nodeName]["probability"],
                stage=stageIndex + 1,
                cost_expression=getattr(pyM, costExpressionName),
                nonant_list=variables,
                scen_model=pyM,
                parent_name=(
                    None if parentName is None else tree.mpisppyNodeName[parentName]
                ),
            )
        )

    pyM._mpisppy_node_list = nodeList
    pyM._mpisppy_probability = tree.probabilityOf(leafName)


def scenario_creator(scenario_name, **kwargs):
    """Build one scenario, in the form mpi-sppy expects.

    This is the callback handed to mpi-sppy. It is a module level function so that it can
    be pickled and sent to worker processes when running under MPI.

    :param scenario_name: name of the leaf node identifying the scenario
    :type scenario_name: string

    **Keyword arguments** (passed through by ``scenario_creator_kwargs``):

    :param baseModel: the deterministic energy system model
    :type baseModel: EnergySystemModel instance

    :param tree: the scenario tree
    :type tree: ScenarioTree instance

    :param stageInvestmentPeriods: stage to investment period mapping
    :type stageInvestmentPeriods: list of lists of int

    :param includeIsBuiltBinaryVariables: whether binary is-built decisions are
        non-anticipative as well
    :type includeIsBuiltBinaryVariables: boolean

    :param timeSeriesAggregation: whether to temporally aggregate each scenario
    :type timeSeriesAggregation: boolean

    :param temporalAggregationSpecs: keyword arguments for ``aggregateTemporally``
    :type temporalAggregationSpecs: dict or None

    :return: the scenario's pyomo model, carrying the scenario tree information and a
        reference to its energy system model in the attribute ``_fineEsM``
    :rtype: pyomo ConcreteModel
    """
    baseModel = kwargs["baseModel"]
    tree = kwargs["tree"]
    stageInvestmentPeriods = kwargs["stageInvestmentPeriods"]
    includeIsBuiltBinaryVariables = kwargs.get("includeIsBuiltBinaryVariables", True)
    timeSeriesAggregation = kwargs.get("timeSeriesAggregation", False)
    temporalAggregationSpecs = kwargs.get("temporalAggregationSpecs")

    logger.debug("Building scenario %s", scenario_name)

    esM = buildScenarioEnergySystemModel(
        baseModel,
        tree,
        scenario_name,
        stageInvestmentPeriods,
        timeSeriesAggregation=timeSeriesAggregation,
        temporalAggregationSpecs=temporalAggregationSpecs,
    )
    _attachScenarioTree(
        esM,
        tree,
        scenario_name,
        stageInvestmentPeriods,
        includeIsBuiltBinaryVariables=includeIsBuiltBinaryVariables,
    )

    # Keep the energy system model alive next to its pyomo model so that FINE's regular
    # result processing can run on it once mpi-sppy has populated the variable values.
    esM.pyM._fineEsM = esM
    return esM.pyM


def _processScenarioResults(esM):
    """Run FINE's regular result processing on a solved scenario.

    Mirrors the post-processing loop of ``EnergySystemModel.optimize``, which only reads
    the values of the already solved pyomo variables and is therefore independent of how
    the model was solved.

    :param esM: the scenario's energy system model, with values loaded into its pyomo model
    :type esM: EnergySystemModel instance
    """
    for mdl in esM.componentModelingDict.values():
        if not isinstance(mdl._capacityVariablesOptimum, dict):
            mdl._capacityVariablesOptimum = {}
        mdl.extractRawResults(esM, esM.pyM)
        mdl.deriveEconomics(esM, esM.pyM)
        mdl.buildOptimizationSummary(esM)
        mdl._convertOptimalValueNames(esM)
    esM.objectiveValue = esM.pyM.Obj()


class StochasticOptimizationResults:
    """Container for the outcome of a multi-stage stochastic optimization.

    :param tree: the scenario tree that was solved
    :type tree: ScenarioTree instance

    :param solverObject: the mpi-sppy object that carried out the solve, i.e. an
        ``ExtensiveForm`` or a ``PH`` instance. Use it to reach mpi-sppy's own diagnostics.
    :type solverObject: object

    :param scenarioModels: mapping of scenario name to the scenario's energy system model,
        for the scenarios held by this process. Each of them carries FINE's regular
        results, so ``getOptimizationSummary`` and the plotting functions work as usual.
    :type scenarioModels: dict

    :param objectiveValue: the optimal value of the stochastic program, i.e. the expected
        net present value over all scenarios
    :type objectiveValue: float or None
    """

    def __init__(self, tree, solverObject, scenarioModels, objectiveValue):
        self.tree = tree
        self.solverObject = solverObject
        self.scenarioModels = scenarioModels
        self.objectiveValue = objectiveValue

    def firstStageDecisions(self):
        """Return the first-stage commissioning decisions, which are equal in all scenarios.

        :return: mapping of ``(modelingClass, componentName, location, investmentPeriod)``
            to the commissioned capacity
        :rtype: dict
        """
        if not self.scenarioModels:
            return {}
        esM = next(iter(self.scenarioModels.values()))
        decisions = {}
        for modelingClassName in sorted(esM.componentModelingDict):
            mdl = esM.componentModelingDict[modelingClassName]
            variable = getattr(esM.pyM, "commis_" + mdl.abbrvName, None)
            if variable is None:
                continue
            for key in sorted(variable):
                loc, compName, ip = key
                if ip == esM.investmentPeriods[0]:
                    decisions[(modelingClassName, compName, loc, ip)] = variable[
                        key
                    ].value
        return decisions


def optimizeMultiStageStochastic(
    baseModel,
    tree,
    stageInvestmentPeriods=None,
    method="ef",
    solver="gurobi",
    solverOptions=None,
    mpisppyOptions=None,
    includeIsBuiltBinaryVariables=True,
    timeSeriesAggregation=False,
    temporalAggregationSpecs=None,
    processResults=True,
):
    """Solve an energy system model as a multi-stage stochastic program.

    The deterministic model given as ``baseModel`` describes the structure of the system
    and holds the values of all parameters that are certain. The scenario ``tree``
    describes the uncertain ones: which parameters vary, what values they take at each
    node, and with which probability. One independent copy of the base model is built per
    scenario with those values substituted, and mpi-sppy enforces that decisions which are
    taken before the scenarios can be told apart are identical across them.

    **Required arguments:**

    :param baseModel: the deterministic energy system model. It is not modified. Its own
        parameter values for the uncertain parameters are irrelevant: every scenario
        overwrites them, and the scenario tree is validated to make sure of that.
    :type baseModel: EnergySystemModel instance

    :param tree: the scenario tree, either as a dictionary of nodes (see
        :class:`ScenarioTree`) or as an already built ``ScenarioTree``
    :type tree: dict or ScenarioTree instance

    **Default arguments:**

    :param stageInvestmentPeriods: which investment periods belong to which stage, as one
        list of investment period indices per stage, for instance ``[[0, 1], [2], [3, 4]]``.
        Stages may cover different numbers of investment periods, but together they have to
        cover every investment period exactly once and in ascending order. If not given,
        every stage corresponds to exactly one investment period, which requires the tree
        to have as many stages as the model has investment periods.
        |br| * the default value is None
    :type stageInvestmentPeriods: list of lists of int, or None

    :param method: how to solve the program.

        (a) 'ef': build the extensive form, i.e. one large deterministic equivalent, and
            solve it directly. Exact, and the right choice for small trees.
        (b) 'ph': progressive hedging, which decomposes the program into one problem per
            scenario and iterates. Approximate up to the convergence threshold, but it
            parallelizes across scenarios and scales to trees for which the extensive form
            is too large.

        |br| * the default value is 'ef'
    :type method: string

    :param solver: the solver used for the scenario subproblems or for the extensive form
        |br| * the default value is 'gurobi'
    :type solver: string

    :param solverOptions: solver options, passed on to mpi-sppy
        |br| * the default value is None
    :type solverOptions: dict or None

    :param mpisppyOptions: additional options handed to the mpi-sppy solver object. For
        progressive hedging the defaults are ``PHIterLimit=50``, ``defaultPHrho=1.0`` and
        ``convthresh=1e-4``; anything given here overrides them.
        |br| * the default value is None
    :type mpisppyOptions: dict or None

    :param includeIsBuiltBinaryVariables: whether the binary is-built decisions of
        components with ``hasIsBuiltBinaryVariable=True`` are non-anticipative as well.
        Leave this at True unless you deliberately want those decisions to differ between
        scenarios that cannot yet be told apart.
        |br| * the default value is True
    :type includeIsBuiltBinaryVariables: boolean

    :param timeSeriesAggregation: whether each scenario should be temporally aggregated
        before its optimization problem is declared. The aggregation necessarily happens
        per scenario and after the scenario values are substituted, so scenarios with
        different time series get different typical periods. This does not affect
        non-anticipativity, which only concerns the investment decisions, but it does mean
        the scenarios use slightly different operational approximations.
        |br| * the default value is False
    :type timeSeriesAggregation: boolean

    :param temporalAggregationSpecs: keyword arguments for
        ``EnergySystemModel.aggregateTemporally``, used when ``timeSeriesAggregation`` is
        True
        |br| * the default value is None
    :type temporalAggregationSpecs: dict or None

    :param processResults: whether FINE's regular result processing should run on every
        scenario after the solve, so that ``getOptimizationSummary`` and the plotting
        functions work on the scenario models
        |br| * the default value is True
    :type processResults: boolean

    :return: the results of the stochastic optimization
    :rtype: StochasticOptimizationResults instance
    """
    ExtensiveForm, PH, _ = _requireMpisppy()

    if baseModel.stochasticModel:
        raise ValueError(
            "The base model must not be built with 'stochasticModel=True'. That option "
            "selects FINE's own single-stage stochastic formulation, which reinterprets "
            "the investment period axis as a scenario axis. Multi-stage stochastic "
            "programming needs the investment periods to be actual investment periods; "
            "the scenarios are handled by this module instead."
        )

    if not isinstance(tree, ScenarioTree):
        tree = ScenarioTree(tree)

    if stageInvestmentPeriods is None:
        stageInvestmentPeriods = _defaultStageInvestmentPeriods(
            baseModel, tree.stageCount
        )
    else:
        stageInvestmentPeriods = _validateStageInvestmentPeriods(
            baseModel, stageInvestmentPeriods, tree.stageCount
        )

    _validateTreeValues(baseModel, tree, stageInvestmentPeriods)

    # mpi-sppy assigns scenarios to the leaves of the tree by position, in depth-first
    # order, so the order of this list is significant.
    scenarioNames = list(tree.scenarioNames)
    logger.info(
        "Solving a %d-stage stochastic program with %d scenarios using %s.",
        tree.stageCount,
        len(scenarioNames),
        method.upper(),
    )

    scenarioCreatorKwargs = {
        "baseModel": baseModel,
        "tree": tree,
        "stageInvestmentPeriods": stageInvestmentPeriods,
        "includeIsBuiltBinaryVariables": includeIsBuiltBinaryVariables,
        "timeSeriesAggregation": timeSeriesAggregation,
        "temporalAggregationSpecs": temporalAggregationSpecs,
    }

    options = {"solver": solver, "solver_name": solver}
    if solverOptions is not None:
        options["solver_options"] = dict(solverOptions)

    methodKey = method.lower()
    if methodKey == "ef":
        options.update(mpisppyOptions or {})
        solverObject = ExtensiveForm(
            options,
            scenarioNames,
            scenario_creator,
            all_nodenames=tree.mpisppyNodeNames,
            scenario_creator_kwargs=scenarioCreatorKwargs,
        )
    elif methodKey == "ph":
        phDefaults = {
            "PHIterLimit": 50,
            "defaultPHrho": 1.0,
            "convthresh": 1e-4,
            "verbose": False,
            "display_progress": False,
            "display_timing": False,
            "iter0_solver_options": dict(solverOptions or {}),
            "iterk_solver_options": dict(solverOptions or {}),
        }
        phDefaults.update(options)
        phDefaults.update(mpisppyOptions or {})
        solverObject = PH(
            phDefaults,
            scenarioNames,
            scenario_creator,
            all_nodenames=tree.mpisppyNodeNames,
            scenario_creator_kwargs=scenarioCreatorKwargs,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'ef' for the extensive form or 'ph' for "
            "progressive hedging."
        )

    # The scenarios have been built by now. Check that they really are the same system
    # before solving, so that a structural difference is reported as such instead of
    # surfacing as a silently mismatched non-anticipativity constraint. Under MPI this
    # compares the scenarios held by this process.
    _verifyStructuralConsistency(
        {name: pyM._fineEsM for name, pyM in solverObject.local_scenarios.items()}
    )

    if methodKey == "ef":
        solverResults = solverObject.solve_extensive_form()
        _checkSolverStatus(solverResults)
    else:
        solverObject.ph_main()
        solverResults = None

    try:
        objectiveValue = (
            pyomo.value(solverObject.ef.EF_Obj)
            if methodKey == "ef"
            else solverObject.Eobjective()
        )
    except ValueError as exc:
        raise RuntimeError(
            "The stochastic program was not solved to a usable solution, so no objective "
            "value is available. The most common cause is that one of the scenarios is "
            "infeasible on its own: check that every scenario's parameter values still "
            "admit a feasible system, for instance that demand can be met in the "
            "scenarios with the highest demand or the tightest capacity limits."
        ) from exc

    scenarioModels = {}
    for scenarioName, scenarioPyM in solverObject.local_scenarios.items():
        scenarioEsM = scenarioPyM._fineEsM
        if processResults:
            _processScenarioResults(scenarioEsM)
        scenarioModels[scenarioName] = scenarioEsM

    return StochasticOptimizationResults(
        tree, solverObject, scenarioModels, objectiveValue
    )
