"""Infeasibility pre-checks which run before the optimization problem is declared.

Every check evaluates a necessary condition for feasibility and resolves unknown
or unbounded quantities in favour of feasibility. A reported problem therefore
proves that the model is infeasible, while passing all checks is no guarantee.
The checks use the full time series and are evaluated separately for each
investment period; the coupling between investment periods (capacities installed
over their lifetime, decommissioning of the stock) is not considered. A pooled
commodity can be moved freely between the locations of a pool, like over
transmission links with unlimited capacity; outside of its pools FINE does not
balance it, so it is treated as unlimited there.

checkCommodityReachability
    Starting from the sources, a commodity becomes available in all locations
    connected by transmission, and a conversion makes its outputs available
    where all of its inputs (one input of a flexible conversion) are available.
    Every sink with a demand has to be reachable. The check is structural and
    detects missing components, missing links and supply time series which are
    zero.
checkBalanceLimits
    The range of the summed contribution of the components of each
    balanceLimitID has to contain the limit. Operation is only enforced by
    operationRateFix (with a capacity variable also by a guaranteed capacity),
    transmission is bounded by its capacity and all other components are
    unbounded. Not considered: pathwayBalanceLimit, minimum operation rates.
checkJointInputDemandAggregated
    Over the whole time horizon, the demand of all sinks is propagated backwards
    through the conversions and has to be covered by the sources. This detects
    shortages which occur because several conversions compete for the same
    input. The input demand is only propagated through commodities with a
    single producer; otherwise only the combined capacity of the producers is
    checked. Storage and transmission capacities within a connected group of
    locations are not considered.
checkTimeStepBalance
    In every time step, the maximum flow from the supply (sources, storage
    discharge, conversion output) to the demand over the transmission network
    has to cover the total demand. This detects local shortages as well as
    bottlenecks on paths across intermediate locations. Not considered: the
    state of charge of storages and the competition of conversions for inputs.
checkJointInputDemandPerTimeStep
    Like checkJointInputDemandAggregated for every time step, with the discharge
    bound of the storage components added to the supply.
"""

from collections import defaultdict
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.flow import edmonds_karp

from fine.enums import Dimension
from fine.utils import output

_TOL = 1e-6
_MAX_ITERATION = 50


# ----------------------------------------------------------------------------
# Access to the component data
# ----------------------------------------------------------------------------


def _ipValue(data, ip):
    """Return the value of an attribute which may be keyed by investment period."""
    return data.get(ip) if isinstance(data, dict) else data


def _timeSeries(comp, name, ip):
    """Return the full (not aggregated) time series of a component, or None."""
    series = _ipValue(getattr(comp, "full" + name), ip)
    return series if isinstance(series, pd.DataFrame) else None


def _eligible(comp):
    """Return the eligible locations (or edge keys) of a component."""
    return [
        key for key, value in comp.processedLocationalEligibility.items() if value > 0
    ]


def _upperCapacity(comp, ip):
    """Return the capacity bound per location or edge key, missing ones are unbounded."""
    for attribute in (comp.processedCapacityFix, comp.processedCapacityMax):
        bound = _ipValue(attribute, ip)
        if bound is not None:
            return bound.astype(float).fillna(np.inf).to_dict()
    return {}


def _toSeries(value, locations):
    """Convert None, a scalar or a Series to a float Series over the locations."""
    if value is None:
        return pd.Series(0.0, index=locations, dtype=float)
    if isinstance(value, pd.Series):
        return value.reindex(locations).astype(float).fillna(0.0)
    return pd.Series(float(value), index=locations)


def _guaranteedCapacity(comp, ip):
    """Return the capacity which has to be installed at each eligible location.

    It is given by capacityFix, otherwise capacityMin, and in the first
    investment period at least by the stock. With an isBuilt binary variable it
    only counts where isBuiltFix is 1.
    """
    locations = sorted(_eligible(comp))
    bound = _ipValue(comp.processedCapacityFix, ip)
    if bound is None:
        bound = _ipValue(comp.processedCapacityMin, ip)
    guaranteed = _toSeries(bound, locations)
    if comp.hasIsBuiltBinaryVariable:
        guaranteed = guaranteed * (_toSeries(comp.isBuiltFix, locations) >= 1)
    if ip == 0 and isinstance(comp.stockCapacityStartYear, pd.Series):
        guaranteed = np.maximum(
            guaranteed, _toSeries(comp.stockCapacityStartYear, locations)
        )
    return guaranteed.replace(np.inf, 0.0)


def _guaranteedDemand(snk, ip, hoursPerTimeStep):
    """Return the demand of a sink which has to be met in any feasible solution.

    With a capacity variable, operationRateFix is a relative profile which is
    scaled by the guaranteed capacity.
    """
    rate = _timeSeries(snk, "OperationRateFix", ip)
    if rate is None or not snk.hasCapacityVariable:
        return rate
    capacity = _guaranteedCapacity(snk, ip).reindex(rate.columns).fillna(0.0)
    return rate.mul(capacity * hoursPerTimeStep, axis=1)


def _edgeValue(data, edgeKey):
    """Return a scalar or edge-indexed transmission attribute at an edge."""
    if isinstance(data, pd.Series):
        value = data.get(edgeKey, np.nan)
        return 0.0 if pd.isna(value) else float(value)
    return 0.0 if data is None else float(data)


def _reduceFactor(value):
    """Reduce a location- or time-dependent conversion factor to a number:
    inputs to their smallest, outputs to their largest magnitude.
    """
    if not isinstance(value, (pd.Series, pd.DataFrame)):
        return float(value)
    values = np.asarray(value, dtype=float).flatten()
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0
    largest = float(max(values, key=abs))
    return float(min(values, key=abs)) if largest < 0 else largest


def _conversionFactors(comp, ip):
    """Return {commodity: factor} of a conversion and whether it is flexible.

    The factors of all vintages which can operate in the investment period are
    merged. The commodity groups of a flexible conversion are flattened, of its
    inputs only one is required.
    """
    collected, isFlexible = defaultdict(list), False
    for key, factors in comp.preprocessedCommodityConversionFactors.items():
        if key != ip and not (isinstance(key, tuple) and key[1] == ip):
            continue
        for commodity, value in factors.items():
            isFlexible |= isinstance(value, dict)
            group = value if isinstance(value, dict) else {commodity: value}
            for groupCommodity, groupValue in group.items():
                collected[groupCommodity].append(_reduceFactor(groupValue))
    reduced = {
        c: _reduceFactor(pd.Series(v, dtype=float)) for c, v in collected.items()
    }
    return reduced, isFlexible


def _safeProduct(a, b):
    """Multiply two numbers, treating 0 * inf as 0."""
    return 0.0 if a == 0 or b == 0 else float(a) * float(b)


# ----------------------------------------------------------------------------
# Model data of one investment period
# ----------------------------------------------------------------------------


class _PeriodData:
    """Model data of one investment period, prepared once and shared by the checks.

    supply: (commodity, location, supply per time step) of the sources, an
        array or a number valid for all time steps
    storage: (commodity, location, discharge bound per time step)
    demand: (sink, commodity, location, guaranteed demand per time step)
    conversions: factors, inputs, outputs, capacity and eligible locations
    links: {commodity: {frozenset of two locations: maximum flow per time step}}
    """

    def __init__(self, esM, ip):
        self.esM, self.ip = esM, ip
        self.numberOfTimeSteps = esM.numberOfTimeSteps
        self.hoursPerTimeStep = esM.hoursPerTimeStep
        self.components = [
            comp
            for model in esM.componentModelingDict.values()
            for comp in model.componentsDict.values()
        ]
        self.sources, self.supply, self.storage = [], [], []
        self.demand, self.conversions = [], []
        for comp in self.components:
            sign = getattr(comp, "sign", None)
            if sign == 1:
                self._addSource(comp)
            elif sign == -1:
                self._addSink(comp)
            if hasattr(comp, "commodityConversionFactors"):
                self._addConversion(comp)
            if hasattr(comp, "chargeEfficiency"):
                self._addStorage(comp)
        # FINE does not balance a pooled commodity at the locations outside of
        # its pools, so it is unlimited there
        for commodity, pools in esM.pooledCommodities.items():
            pooled = {loc for pool in pools.values() for loc in pool}
            unbalanced = sorted(esM.locations - pooled)
            if unbalanced:
                pseudoSource = SimpleNamespace(commodity=commodity)
                self.sources.append((pseudoSource, None, unbalanced))
                self.supply += [(commodity, loc, np.inf) for loc in unbalanced]
        self.links = self._getTransmissionLinks()
        self._islands = {}
        for commodity, links in self.links.items():
            graph = nx.Graph()
            graph.add_nodes_from(esM.locations)
            graph.add_edges_from(tuple(locations) for locations in links)
            for group in nx.connected_components(graph):
                self._islands.update(
                    {(commodity, loc): frozenset(group) for loc in group}
                )
        self._producers = {}

    def key(self, commodity, loc):
        """Return (commodity, group of locations connected to loc by transmission
        or a pool).
        """
        return commodity, self._islands.get((commodity, loc), frozenset((loc,)))

    def _addSource(self, comp):
        rate = _timeSeries(comp, "OperationRateFix", self.ip)
        if rate is None:
            rate = _timeSeries(comp, "OperationRateMax", self.ip)
        capacity, eligible = _upperCapacity(comp, self.ip), _eligible(comp)
        self.sources.append((comp, rate, eligible))
        for loc in eligible:
            hasRate = rate is not None and loc in rate.columns
            if comp.hasCapacityVariable:
                bound = capacity.get(loc, np.inf) * self.hoursPerTimeStep
                if hasRate:
                    values = rate[loc].to_numpy(dtype=float)
                    with np.errstate(invalid="ignore"):  # 0 * inf is 0
                        supply = np.where(values == 0, 0.0, bound * values)
                else:
                    supply = bound
            else:
                supply = rate[loc].to_numpy(dtype=float) if hasRate else np.inf
            self.supply.append((comp.commodity, loc, supply))

    def _addSink(self, comp):
        demand = _guaranteedDemand(comp, self.ip, self.hoursPerTimeStep)
        if demand is not None:
            for loc in demand.columns:
                values = demand[loc].to_numpy(dtype=float)
                self.demand.append((comp, comp.commodity, loc, values))

    def _addConversion(self, comp):
        factors, isFlexible = _conversionFactors(comp, self.ip)
        self.conversions.append(
            SimpleNamespace(
                factors=factors,
                isFlexible=isFlexible,
                # the inputs of a flexible conversion cannot be assigned to
                # a single commodity, so they do not restrict its operation
                inputs={}
                if isFlexible
                else {c: abs(f) for c, f in factors.items() if f < 0},
                outputs={c: f for c, f in factors.items() if f > 0},
                capacity=_upperCapacity(comp, self.ip),
                eligible=_eligible(comp),
            )
        )

    def _addStorage(self, comp):
        """Add the discharge bound; whether the storage was charged is not checked."""
        capacity = _upperCapacity(comp, self.ip)
        for loc in _eligible(comp):
            bound = (
                capacity.get(loc, np.inf)
                * float(comp.dischargeRate or 1)
                * float(comp.dischargeEfficiency or 1)
                * self.hoursPerTimeStep
            )
            self.storage.append((comp.commodity, loc, bound))

    def _getTransmissionLinks(self):
        """Both directions of a symmetric eligibility describe the same link,
        parallel components add up. The flow is reduced by the losses.
        """
        perComponent = {}
        for comp in self.components:
            if getattr(comp, "dimension", None) != Dimension.TWO or not hasattr(
                comp, "commodity"
            ):
                continue
            capacity = _upperCapacity(comp, self.ip)
            for edge in _eligible(comp):
                loss = _edgeValue(comp.losses, edge) * _edgeValue(comp.distances, edge)
                flow = (
                    capacity.get(edge, np.inf)
                    * self.hoursPerTimeStep
                    * max(0.0, 1.0 - loss)
                )
                key = (comp.commodity, comp.name, frozenset(comp._mapC[edge]))
                perComponent[key] = max(perComponent.get(key, 0.0), flow)
        links = defaultdict(dict)
        for (commodity, _, locations), flow in perComponent.items():
            links[commodity][locations] = links[commodity].get(locations, 0.0) + flow
        # A pooled commodity is balanced over each pool as a whole, i.e. it can
        # be moved freely between the locations of a pool.
        for commodity, pools in self.esM.pooledCommodities.items():
            for pool in pools.values():
                members = sorted(pool)
                for loc in members[1:]:
                    links[commodity][frozenset((members[0], loc))] = np.inf
        return dict(links)

    def producers(self, commodity, island):
        """Return the producers of a commodity in an island, their output
        capacity per time step and, for a unique producer, where its input
        demand and its other outputs go.
        """
        key = (commodity, island)
        if key not in self._producers:
            producers = [
                c
                for c in self.conversions
                if commodity in c.outputs and any(loc in island for loc in c.eligible)
            ]
            capacity = sum(
                c.capacity.get(loc, np.inf)
                * self.hoursPerTimeStep
                * c.outputs[commodity]
                for c in producers
                for loc in c.eligible
                if loc in island
            )
            targets = None
            if len(producers) == 1:
                (producer,) = producers
                locations = [loc for loc in producer.eligible if loc in island]

                def targetsOf(factors):
                    result = []
                    for otherCommodity, factor in factors.items():
                        keys = {self.key(otherCommodity, loc) for loc in locations}
                        if len(keys) == 1:  # only commodities in one connected group
                            result.append((keys.pop(), factor))
                    return result

                otherOutputs = {
                    c: f for c, f in producer.outputs.items() if c != commodity
                }
                targets = (
                    producer.outputs[commodity],
                    targetsOf(producer.inputs),
                    targetsOf(otherOutputs),
                )
            self._producers[key] = (producers, capacity, targets)
        return self._producers[key]


# ----------------------------------------------------------------------------
# Checks for one investment period, see the module docstring
# ----------------------------------------------------------------------------


def _checkCommodityReachability(data):
    available = set()
    for comp, rate, eligible in data.sources:
        for loc in eligible:
            if rate is None or loc not in rate.columns or rate[loc].sum() > 0:
                available.add(data.key(comp.commodity, loc))

    changed = True
    while changed:
        changed = False
        for conv in data.conversions:
            inputs = [c for c, f in conv.factors.items() if f < 0]
            outputs = [c for c, f in conv.factors.items() if f > 0]
            for loc in conv.eligible:
                found = [data.key(c, loc) in available for c in inputs]
                if (any(found) or not inputs) if conv.isFlexible else all(found):
                    new = {data.key(c, loc) for c in outputs} - available
                    if new:
                        available |= new
                        changed = True

    return [
        f"The sink '{snk.name}' has a demand for the commodity '{commodity}' in "
        f"location '{loc}', but the commodity can neither be produced nor imported "
        "there."
        for snk, commodity, loc, values in data.demand
        if values.sum() > 0 and data.key(commodity, loc) not in available
    ]


def _propagateDeficits(data, label, numberOfSteps, supply, required, problems):
    """Propagate the deficits of a balance backwards through the conversions.

    The keys of required are processed in their order; a unique producer
    passes its input demand on and adds its other outputs to the supply.
    """
    for _ in range(_MAX_ITERATION):
        changed = False
        for key, requiredValue in list(required.items()):
            deficit = requiredValue - supply[key]
            if deficit <= _TOL:
                continue
            commodity, island = key
            producers, capacity, targets = data.producers(commodity, island)
            if not producers:
                problems.append(
                    f"{label}: the joint demand of {requiredValue:.4g} for the "
                    f"commodity '{commodity}' in the locations {sorted(island)} "
                    f"exceeds the supply of {supply[key]:.4g} and no component "
                    "produces this commodity."
                )
            else:
                if capacity * numberOfSteps + _TOL < deficit:
                    problems.append(
                        f"{label}: the deficit of {deficit:.4g} for the commodity "
                        f"'{commodity}' in the locations {sorted(island)} exceeds the "
                        f"combined conversion capacity of {capacity * numberOfSteps:.4g}."
                    )
                if targets:
                    outputFactor, inputTargets, outputTargets = targets
                    operation = deficit / outputFactor
                    for targetKey, factor in inputTargets:
                        required[targetKey] += operation * factor
                    for targetKey, factor in outputTargets:
                        supply[targetKey] += operation * factor
            supply[key] = requiredValue
            changed = True
        if not changed:
            break


def _checkJointInputDemandAggregated(data):
    supply, required = defaultdict(float), defaultdict(float)
    for commodity, loc, values in data.supply:
        total = (
            values.sum()
            if isinstance(values, np.ndarray)
            else values * data.numberOfTimeSteps
        )
        supply[data.key(commodity, loc)] += total
    for _, commodity, loc, values in data.demand:
        if values.sum() > 0:
            required[data.key(commodity, loc)] += values.sum()
    problems = []
    _propagateDeficits(
        data, "Total time horizon", data.numberOfTimeSteps, supply, required, problems
    )
    return problems


def _checkJointInputDemandPerTimeStep(data):
    zeros = np.zeros(data.numberOfTimeSteps)
    supplyArrays, requiredArrays, entries = {}, {}, []
    for commodity, loc, values in data.supply + data.storage:
        key = data.key(commodity, loc)
        supplyArrays[key] = supplyArrays.get(key, zeros) + values
    for _, commodity, loc, values in data.demand:
        key = data.key(commodity, loc)
        requiredArrays[key] = requiredArrays.get(key, zeros) + np.where(
            values > 0, values, 0.0
        )
        entries.append((key, values.tolist()))

    hasDeficit = np.zeros(data.numberOfTimeSteps, dtype=bool)
    with np.errstate(invalid="ignore"):
        for key, values in requiredArrays.items():
            hasDeficit |= (values > 0) & (values - supplyArrays.get(key, zeros) > _TOL)

    # Only time steps with an initial deficit can change during the propagation
    supplyLists = [(key, values.tolist()) for key, values in supplyArrays.items()]
    requiredLists = {key: values.tolist() for key, values in requiredArrays.items()}
    problems = []
    for t in np.flatnonzero(hasDeficit).tolist():
        supply = defaultdict(float, {key: values[t] for key, values in supplyLists})
        required = defaultdict(float)
        for key, values in entries:  # ordered by the first sink with a demand
            if values[t] > 0 and key not in required:
                required[key] = requiredLists[key][t]
        _propagateDeficits(data, f"Time step {t}", 1, supply, required, problems)
    return problems


def _isClose(a, b):
    """Elementwise equivalent of math.isclose with abs_tol=_TOL."""
    with np.errstate(invalid="ignore"):
        bound = np.maximum(1e-9 * np.maximum(np.abs(a), np.abs(b)), _TOL)
        return (a == b) | (np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= bound))


def _maxFlow(supply, demand, members, links):
    """Return the maximum flow from the supply to the demand of an island per
    time step. Identical time steps are solved once; infinite capacities are
    capped by a value which exceeds every demand. Edmonds-Karp is the fastest
    networkx algorithm for these small graphs.
    """
    bigCapacity = max(float(demand.sum(axis=1).max()), 1.0) * 1e6
    source, sink = ("_supply_",), ("_demand_",)
    graph = nx.DiGraph()
    for loc in members:
        graph.add_edge(source, loc, capacity=0.0)
        graph.add_edge(loc, sink, capacity=0.0)
    for locations, flow in links.items():
        if flow > 0 and locations <= set(members):
            loc1, loc2 = locations
            graph.add_edge(loc1, loc2, capacity=min(flow, bigCapacity))
            graph.add_edge(loc2, loc1, capacity=min(flow, bigCapacity))
    rows, inverse = np.unique(np.hstack([supply, demand]), axis=0, return_inverse=True)
    flows = np.empty(len(rows))
    for r, row in enumerate(rows):
        for i, loc in enumerate(members):
            graph[source][loc]["capacity"] = min(float(row[i]), bigCapacity)
            graph[loc][sink]["capacity"] = float(row[len(members) + i])
        flows[r] = nx.maximum_flow_value(graph, source, sink, flow_func=edmonds_karp)
    return flows[inverse.reshape(-1)]


def _maxTransportableFlow(data, commodity, localSupply, locationDemand):
    """Return the maximum transportable flow and the total demand per time step.

    The locally served demand is a lower bound of the maximum flow of an island
    and exact where every location covers its own demand. The exact maximum flow
    is only computed where the lower bounds do not cover the total demand.
    """
    zeros = np.zeros(data.numberOfTimeSteps)
    totalDemand = sum(
        (d for (c, _), d in locationDemand.items() if c == commodity), zeros
    )
    islands = []
    for island in sorted(
        {data.key(commodity, loc)[1] for loc in data.esM.locations}, key=sorted
    ):
        members = sorted(island)
        demand = np.column_stack(
            [locationDemand.get((commodity, loc), zeros) for loc in members]
        )
        if (demand > 0).any():
            supply = np.column_stack(
                [localSupply.get((commodity, loc), zeros) for loc in members]
            )
            islands.append(
                (members, supply, demand, np.minimum(supply, demand).sum(axis=1))
            )

    couldBeShort = sum((flow for *_, flow in islands), zeros) + _TOL < totalDemand
    for members, supply, demand, flow in islands:
        steps = np.flatnonzero(couldBeShort & (supply < demand).any(axis=1))
        if len(members) > 1 and len(steps):
            flow[steps] = _maxFlow(
                supply[steps], demand[steps], members, data.links.get(commodity, {})
            )
    return sum((flow for *_, flow in islands), zeros), totalDemand


def _checkTimeStepBalance(data):
    numberOfTimeSteps = data.numberOfTimeSteps
    zeros = np.zeros(numberOfTimeSteps)

    localSupply, storageSupply = {}, defaultdict(float)
    for commodity, loc, values in data.supply:
        localSupply[(commodity, loc)] = (
            localSupply.get((commodity, loc), zeros) + values
        )
    for commodity, loc, value in data.storage:
        storageSupply[(commodity, loc)] += value
    for key, value in storageSupply.items():
        localSupply[key] = localSupply.get(key, zeros) + value

    # Conversion output with the inputs taken from the connected group of
    # locations. Every time step stops iterating once it has converged.
    conversionOutput = {}
    isIterating = np.ones(numberOfTimeSteps, dtype=bool)
    for _ in range(_MAX_ITERATION):
        pooledSupply = {}
        for values in (localSupply, conversionOutput):
            for (commodity, loc), value in values.items():
                key = data.key(commodity, loc)
                pooledSupply[key] = pooledSupply.get(key, zeros) + value
        newOutput = {}
        for conv in data.conversions:
            for loc in conv.eligible:
                operation = np.full(
                    numberOfTimeSteps,
                    conv.capacity.get(loc, np.inf) * data.hoursPerTimeStep,
                )
                for commodity, factor in conv.inputs.items():
                    available = pooledSupply.get(data.key(commodity, loc), zeros)
                    operation = np.minimum(operation, available / factor)
                for commodity, factor in conv.outputs.items():
                    newOutput[(commodity, loc)] = newOutput.get(
                        (commodity, loc), zeros
                    ) + np.where(operation > 0, operation * factor, 0.0)
        keys = list(dict.fromkeys([*conversionOutput, *newOutput]))
        hasConverged = np.ones(numberOfTimeSteps, dtype=bool)
        for key in keys:
            hasConverged &= _isClose(
                conversionOutput.get(key, zeros), newOutput.get(key, zeros)
            )
        for key in keys:
            conversionOutput[key] = np.where(
                isIterating, newOutput.get(key, zeros), conversionOutput.get(key, zeros)
            )
        isIterating &= ~hasConverged
        if not isIterating.any():
            break

    totalLocalSupply = dict(localSupply)
    for key, value in conversionOutput.items():
        totalLocalSupply[key] = totalLocalSupply.get(key, zeros) + value
    locationDemand = {}
    for _, commodity, loc, values in data.demand:
        locationDemand[(commodity, loc)] = locationDemand.get(
            (commodity, loc), zeros
        ) + np.where(values > 0, values, 0.0)

    problemsPerTimeStep = defaultdict(list)
    for commodity in sorted({commodity for commodity, _ in locationDemand}):
        maxFlow, totalDemand = _maxTransportableFlow(
            data, commodity, totalLocalSupply, locationDemand
        )
        for t in np.flatnonzero(maxFlow + _TOL < totalDemand).tolist():
            problemsPerTimeStep[t].append(
                f"Time step {t}: the maximum transportable flow of {maxFlow[t]:.4g} "
                f"for the commodity '{commodity}' is smaller than the total demand of "
                f"{totalDemand[t]:.4g}. The supply or the transmission capacity of the "
                "network is insufficient."
            )
    return [p for t in sorted(problemsPerTimeStep) for p in problemsPerTimeStep[t]]


def _operationBounds(comp, data):
    """Return {location: (lower, upper)} of the operation of a source or sink
    summed over the time horizon. Lower bounds only stem from operationRateFix
    (with a capacity variable times the guaranteed capacity).
    """
    ip, hoursPerTimeStep = data.ip, data.hoursPerTimeStep
    rateFix = _timeSeries(comp, "OperationRateFix", ip)
    rateMax = _timeSeries(comp, "OperationRateMax", ip)

    def total(rate, loc):
        return (
            float(rate[loc].sum()) if rate is not None and loc in rate.columns else None
        )

    bounds = {}
    if comp.hasCapacityVariable:
        capacityMax = _upperCapacity(comp, ip)
        capacityMin = _guaranteedCapacity(comp, ip).fillna(0.0)
        for loc in sorted(_eligible(comp)):
            capMax = capacityMax.get(loc, np.inf)
            fixSum = total(rateFix, loc)
            if fixSum is not None:
                bounds[loc] = (
                    _safeProduct(capacityMin[loc], fixSum * hoursPerTimeStep),
                    _safeProduct(capMax, fixSum * hoursPerTimeStep),
                )
            else:
                maxSum = total(rateMax, loc)
                if maxSum is None:
                    maxSum = float(data.numberOfTimeSteps)
                bounds[loc] = (0.0, _safeProduct(capMax, maxSum * hoursPerTimeStep))
    else:
        for loc in sorted(_eligible(comp)):
            fixSum, maxSum = total(rateFix, loc), total(rateMax, loc)
            if fixSum is not None:
                bounds[loc] = (fixSum, fixSum)
            else:
                bounds[loc] = (0.0, np.inf if maxSum is None else maxSum)
    return bounds


def _balanceLimitContribution(comp, data):
    """Return {location: (lower, upper)} of the contribution of a component to
    its balance limit. Sources contribute their operation, sinks the negative
    operation, transmission at most the capacity of the adjacent links.
    """
    esM = data.esM
    sign = getattr(comp, "sign", None)
    if sign in (1, -1):
        return {
            loc: (lower, upper) if sign == 1 else (-upper, -lower)
            for loc, (lower, upper) in _operationBounds(comp, data).items()
        }
    if getattr(comp, "dimension", None) == Dimension.TWO:
        capacity = _upperCapacity(comp, data.ip)
        horizon = esM.numberOfTimeSteps * esM.hoursPerTimeStep
        maxChange = defaultdict(float)
        for edge in _eligible(comp):
            for loc in comp._mapC[edge]:
                maxChange[loc] += _safeProduct(capacity.get(edge, np.inf), horizon)
        return {loc: (-maxChange[loc], maxChange[loc]) for loc in esM.locations}
    return dict.fromkeys(esM.locations, (-np.inf, np.inf))


def _checkBalanceLimits(data):
    esM = data.esM
    table = _ipValue(esM.processedBalanceLimit, data.ip)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return []
    locations = sorted(esM.locations)
    lowerSum, upperSum, limitIDs = defaultdict(float), defaultdict(float), set()
    for comp in data.components:
        limitID = getattr(comp, "balanceLimitID", None)
        if limitID is None or limitID not in table.index:
            continue
        limitIDs.add(limitID)
        bounds = _balanceLimitContribution(comp, data)
        for loc in locations:
            lower, upper = bounds.get(loc, (0.0, 0.0))
            for column in (loc, "Total"):
                lowerSum[(limitID, column)] += lower
                upperSum[(limitID, column)] += upper

    problems = []
    for limitID, row in table.iterrows():
        # a limit without components may not be declared at all
        if limitID not in limitIDs:
            continue
        isLowerBound = bool(row.get("lowerBound", 0))
        for column in locations + ["Total"]:
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            value = float(value)
            reachableMin, reachableMax = (
                lowerSum[(limitID, column)],
                upperSum[(limitID, column)],
            )
            tolerance = _TOL * max(1.0, abs(value))
            where = "the total system" if column == "Total" else f"location '{column}'"
            if isLowerBound and reachableMax + tolerance < value:
                problems.append(
                    f"The balanceLimit '{limitID}' requires a balance of at least "
                    f"{value:.4g} in {where}, but the components with this "
                    f"balanceLimitID can reach at most {reachableMax:.4g}."
                )
            if not isLowerBound and reachableMin - tolerance > value:
                problems.append(
                    f"The balanceLimit '{limitID}' allows a balance of at most "
                    f"{value:.4g} in {where}, but the components with this "
                    f"balanceLimitID contribute at least {reachableMin:.4g}."
                )
    return problems


_CHECKS = {
    "checkCommodityReachability": _checkCommodityReachability,
    "checkBalanceLimits": _checkBalanceLimits,
    "checkJointInputDemandAggregated": _checkJointInputDemandAggregated,
    "checkTimeStepBalance": _checkTimeStepBalance,
    "checkJointInputDemandPerTimeStep": _checkJointInputDemandPerTimeStep,
}


def runInfeasibilityPrechecks(esM):
    """Run the infeasibility pre-checks described in the module docstring.

    A ValueError lists every detected problem, prefixed with the name of the
    check and, for several investment periods, the name of the period.

    :param esM: EnergySystemModel instance
    """
    periodData = [_PeriodData(esM, ip) for ip in esM.investmentPeriods]
    problems = []
    for name, check in _CHECKS.items():
        checkProblems = [
            (
                f"Investment period {esM.investmentPeriodNames[data.ip]}: "
                if len(periodData) > 1
                else ""
            )
            + problem
            for data in periodData
            for problem in check(data)
        ]
        problems += [f"{name}: {problem}" for problem in checkProblems]
        if not checkProblems:
            output(
                f"The infeasibility pre-check '{name}' passed.", esM.verboseLogLevel, 0
            )
    if problems:
        raise ValueError(
            "The infeasibility pre-checks detected that the model cannot be solved:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + "\n\nThe pre-checks can be deactivated by setting "
            "'runInfeasibilityPrechecks' to False."
        )
    output(
        "All infeasibility pre-checks passed. Note that they evaluate necessary "
        "conditions only and are no feasibility guarantee.",
        esM.verboseLogLevel,
        0,
    )
