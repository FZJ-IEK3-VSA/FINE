"""
Modeling to Generate Alternatives (MGA) on top of ETHOS.FINE.

Workflow
  1. Solve the cost-optimal problem with esM.optimize().
  2. Add a constraint: total cost <= optimal cost + slack * |optimal cost|.
  3. Replace FINE's cost objective with a weighted sum of capacity / operation
     variables (minimized or maximized).
  4. Re-solve the existing Pyomo model (declaresOptimizationProblem=False).



  variable: "capacity"  -> cap_<abbrv>          (all modeling classes)
            "operation" -> op_<abbrv>           (SourceSink, Conversion, Transmission)
            "charge"    -> chargeOp_<abbrv>     (Storage)
            "discharge" -> dischargeOp_<abbrv>  (Storage)
  Operation variables are aggregated to annual sums (period occurrences and
  number of years are taken into account, so time series aggregation works).
"""

import os
import re
import time

# import fine as fn
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
import pyomo.environ as pyo

varPatterns = {
    "capacity": "cap_{}",
    "operation": "op_{}",
    "charge": "chargeOp_{}",
    "discharge": "dischargeOp_{}",
}
operationKinds = ("operation", "charge", "discharge")
selections = ("custom", "allCapacity", "allCapacityAndOperation")


# --------------------------------------------------------------------------- #
# Index helpers. FINE index layout: (loc, compName, [ip], [p, t])
# --------------------------------------------------------------------------- #
def periodOccurrence(esM, ip, p):
    occ = esM.periodOccurrences
    if isinstance(occ, dict):  # FINE versions with investment periods
        return occ[ip][p]
    return occ[p]


def _ip_of(idx):
    rest = idx[2:]
    return rest[0] if len(rest) in (1, 3) else None


def annualFactor(esM, idx):
    """1 for capacity variables, occurrence / numberOfYears for operation variables."""
    rest = idx[2:]
    if len(rest) < 2:
        return 1.0
    return periodOccurrence(esM, _ip_of(idx), rest[-2]) / esM.numberOfYears


def weightedParts(esM, var, weightOf):
    """One pass over a Pyomo variable. weightOf(idx) returns a weight or None (skip)."""
    parts = []
    for idx in var:
        w = weightOf(idx)
        if w is None:
            continue
        parts.append(w * annualFactor(esM, idx) * var[idx])
    return parts


# --------------------------------------------------------------------------- #
# (a) user-defined terms
# --------------------------------------------------------------------------- #
def termParts(esM, term):
    pyM = esM.pyM
    comp = term["component"]
    kind = term["variable"]
    weight = float(term.get("weight", 1.0))
    locs = term.get("locations")
    ips = term.get("investmentPeriods")

    if comp not in esM.componentNames:
        raise KeyError(f"Unknown component '{comp}'.")
    if kind not in varPatterns:
        raise KeyError(f"Unknown variable '{kind}'. Use one of {list(varPatterns)}.")

    mdl = esM.componentModelingDict[esM.componentNames[comp]]
    var = pyM.find_component(varPatterns[kind].format(mdl.abbrvName))
    if var is None:
        available = [v.name for v in pyM.component_objects(pyo.Var)
                     if v.name.endswith("_" + mdl.abbrvName)]
        raise ValueError(f"'{kind}' is not available for '{comp}' "
                         f"({type(mdl).__name__}). Variables of this class: {available}")

    print(f"    Adding {kind} of '{comp}' (weight {weight}) ...", flush=True)

    def weightOf(idx):
        if idx[1] != comp:
            return None
        if locs is not None and idx[0] not in locs:
            return None
        if ips is not None and _ip_of(idx) not in ips:
            return None
        return weight

    parts = weightedParts(esM, var, weightOf)
    if not parts:
        raise ValueError(f"No variables matched term {term}.")
    return parts


# --------------------------------------------------------------------------- #
# (b), (c) all capacity [and operation] variables of the model
# --------------------------------------------------------------------------- #
def lookupWeight(weights, comp, kind, default):
    w = weights.get(comp, default)
    if isinstance(w, dict):  # per-variable override, e.g. {"capacity": 0.5}
        w = w.get(kind, default)
    return float(w)


def allVariableParts(esM, objSpec, includeOperation):
    pyM = esM.pyM
    default = float(objSpec.get("defaultWeight", 1.0))
    weights = objSpec.get("weights", {})
    exclude = set(objSpec.get("excludeComponents", []))
    unknown = (set(weights) | exclude) - set(esM.componentNames)
    if unknown:
        raise KeyError(f"Unknown components in weights/excludeComponents: {sorted(unknown)}")

    kinds = ("capacity",) + (operationKinds if includeOperation else ())
    parts = []
    for mdl in esM.componentModelingDict.values():
        print(f"    Adding variables of {type(mdl).__name__} ...", flush=True)
        for kind in kinds:
            varName = varPatterns[kind].format(mdl.abbrvName)
            var = pyM.find_component(varName)
            if var is None:  # e.g. no op_ variable for storage
                continue

            def weightOf(idx, kind=kind):
                comp = idx[1]
                if comp in exclude:
                    return None
                return lookupWeight(weights, comp, kind, default)

            parts += weightedParts(esM, var, weightOf)
    return parts


# --------------------------------------------------------------------------- #
# Objective, cost constraint, driver
# --------------------------------------------------------------------------- #
def objectiveParts(esM, objSpec):
    selection = objSpec.get("variableSelection", "custom")
    if selection == "custom":
        if not objSpec.get("terms"):
            raise ValueError(f"Objective '{objSpec.get('name')}' is 'custom' but has no terms.")
        parts = [p for t in objSpec["terms"] for p in termParts(esM, t)]
    elif selection == "allCapacity":
        parts = allVariableParts(esM, objSpec, includeOperation=False)
    elif selection == "allCapacityAndOperation":
        parts = allVariableParts(esM, objSpec, includeOperation=True)
    else:
        raise ValueError(f"variableSelection must be one of {selections}, got '{selection}'.")
    if not parts:
        raise ValueError(f"Objective '{objSpec.get('name')}' contains no variables.")
    return parts


def setMGAObjective(esM, objSpec):
    """Replace the active objective with the weighted MGA objective."""
    pyM = esM.pyM
    t0 = time.perf_counter()
    sense = objSpec.get("sense", "max")
    selection = objSpec.get("variableSelection", "custom")
    print(f"  Declaring MGA objective function ({sense}, variableSelection='{selection}') ...",
          flush=True)

    if pyM.find_component("Obj") is not None:
        print("    Removing previous objective function ...", flush=True)
        pyM.del_component("Obj")

    parts = objectiveParts(esM, objSpec)
    print(f"    Building objective expression from {len(parts)} variables ...", flush=True)
    pyoSense = pyo.maximize if sense.lower().startswith("max") else pyo.minimize
    pyM.Obj = pyo.Objective(expr=pyo.quicksum(parts),
                            sense=pyoSense)  # keep FINE's name "Obj"
    print(f"  MGA objective function declared ({time.perf_counter() - t0:.1f} s).", flush=True)


def addCostSlackConstraint(esM, costExpr, optimalCost, slack):
    pyM = esM.pyM
    t0 = time.perf_counter()
    bound = optimalCost + slack * abs(optimalCost)
    print(f"Declaring cost constraint: total cost <= {bound:.6g} "
          f"(optimal cost {optimalCost:.6g}, slack {slack:.1%}) ...", flush=True)
    if pyM.find_component("mgaCostConstraint") is not None:
        pyM.del_component("mgaCostConstraint")
    # abs() keeps the bound correct if the optimum is negative (e.g. export revenues)
    pyM.mgaCostConstraint = pyo.Constraint(expr=costExpr <= bound)
    print(f"Cost constraint declared ({time.perf_counter() - t0:.1f} s).", flush=True)


def _snapshot(esM, **summaryKwargs):
    """Copy the summaries, because every solve overwrites them in esM."""
    print("  Collecting optimization summaries ...", flush=True)
    return {name: esM.getOptimizationSummary(name, outputLevel=2, **summaryKwargs).copy()
            for name in esM.componentModelingDict}


def writeExcel(esM, outputDir, fileStem, excelKwargs):
    """Write the current esM results with FINE's writeOptimizationOutputToExcel."""
    t0 = time.perf_counter()
    safeStem = re.sub(r"[^\w\-]+", "_", fileStem)  # file-system safe name
    outputFileName = os.path.join(outputDir, safeStem)
    print(f"  Writing results to Excel: {outputFileName}.xlsx ...", flush=True)
    writeOptimizationOutputToExcel(esM, outputFileName=outputFileName, **excelKwargs)
    print(f"  Excel file written ({time.perf_counter() - t0:.1f} s).", flush=True)
    return outputFileName + ".xlsx"


def runMGA(
    esM, 
    spec, 
    summaryKwargs=None,
    save_mga_results_in_excel=False, 
    excelOutputDir="mga_results",
    excelFilePrefix="MGA", 
    excelKwargs={"optSumOutputLevel": 2, "optValOutputLevel": 2},
    **optimizeKwargs):
    """
    Solve cost-optimal, then each MGA objective in spec["objectives"].

    **Arguments:**

    :param spec: states the parameters required for the MGA optimization.

        E.g., User input (spec)
        {
            "slack": 0.10,                          # allowed cost increase (10 %)
            "objectives": [                         # one entry = one alternative, if multiple alternatives are needed,
                                                    # add multiple objectives with varying weights. Objectives should be
                                                    # in the form of (a), (b), or (c) as shown.
            # (a) user-defined variables
            {
                "name": "max_wind",
                "sense": "max",                     # "min" or "max"
                "variableSelection": "custom",      # default
                "terms": [
                {"component": "Wind (onshore)", "variable": "capacity",  "weight": 1.0},
                {"component": "PV",             "variable": "operation", "weight": 0.5,
                "locations": ["region1"],          # optional, default: all
                "investmentPeriods": [2030]},      # optional, default: all
                ],
            },
            # (b) all capacity variables of the model
            {
                "name": "min_all_capacity",
                "sense": "min",
                "variableSelection": "allCapacity",
                "defaultWeight": 1.0,                          # optional, default 1.0
                "weights": {"PV": 2.0,                         # optional overrides:
                            "Li-ion": {"capacity": 0.5}},      #   per component or per variable
                "excludeComponents": ["Electricity demand"],   # optional
            },
            # (c) all capacity and all operation variables of the model
            {
                "name": "min_all_cap_and_op",
                "sense": "min",
                "variableSelection": "allCapacityAndOperation",
            },
            ],
        }

    :type spec: dictionary

    : param summaryKwargs: summaryKwargs are passed to getOptimizationSummary (e.g. ip=2030)

        |br| * the default value is None
    :type summaryKwargs: dictionary

    :param save_mga_results_in_excel: if True, the cost-optimal solution and every MGA
        alternative are written with fn.writeOptimizationOutputToExcel to
        <excelOutputDir>/<excelFilePrefix>_costOptimal.xlsx and
        <excelOutputDir>/<excelFilePrefix>_<objective name>.xlsx

        |br| * the default value is False
    :type save_mga_results_in_excel: boolean

    :param excelOutputDir: defines <excelOutputDir> of the output Excel files

        |br| * the default value is "mga_results"
    :type excelOutputDir: string

    :param excelFilePrefix: defines <excelFilePrefix> of the output Excel files

        |br| * the default value is "MGA"
    :type excelFilePrefix: string

    :param excelKwargs: passed to writeOptimizationOutputToExcel

       |br| * the default value is {"optSumOutputLevel": 2, "optValOutputLevel": 2}
    :type excelKwargs: dictionary

    :param optimizeKwargs: optimizeKwargs are passed to esM.optimize (solver, timeSeriesAggregation, ...)

       |br| * the default value is {}
    :type optimizeKwargs: dictionary

    """
    tStart = time.perf_counter()
    summaryKwargs = summaryKwargs or {}
    excelKwargs = excelKwargs or {}
    if save_mga_results_in_excel:
        os.makedirs(excelOutputDir, exist_ok=True)
    optimizeKwargs.pop("declaresOptimizationProblem", None)
    nObj = len(spec["objectives"])

    print("=" * 70, flush=True)
    print(f"MGA run: slack {spec['slack']:.1%}, {nObj} alternative(s)", flush=True)
    if save_mga_results_in_excel:
        print(f"Excel output enabled: {os.path.abspath(excelOutputDir)}", flush=True)
    print("=" * 70, flush=True)

    # 1) cost-optimal solution
    print("Accessing the cost-optimal solution ...", flush=True)
    costExpr = esM.pyM.Obj.expr
    optimalCost = pyo.value(costExpr)
    print(f"Cost-optimal solution found: total cost = {optimalCost:.6g}", flush=True)
    results = {"costOptimal": {"cost": optimalCost,
                               "summary": _snapshot(esM, **summaryKwargs)}}
    if save_mga_results_in_excel:
        results["costOptimal"]["excelFile"] = writeExcel(
            esM, excelOutputDir, f"{excelFilePrefix}_costOptimal", excelKwargs)

    # 2) near-optimal cost constraint
    print("Adding the near-optimal cost constraint ...", flush=True)
    addCostSlackConstraint(esM, costExpr, optimalCost, spec["slack"])

    # 3) + 4) one solve per MGA objective, reusing the declared model
    print("Solving MGA alternatives ...", flush=True)
    for i, objSpec in enumerate(spec["objectives"], start=1):
        tObj = time.perf_counter()
        print("-" * 70, flush=True)
        print(f"Alternative {i}/{nObj}: '{objSpec['name']}'", flush=True)
        setMGAObjective(esM, objSpec)

        print("  Solving MGA problem ...", flush=True)
        esM.optimize(declaresOptimizationProblem=False, **optimizeKwargs)

        cost = pyo.value(costExpr)
        mgaValue = pyo.value(esM.pyM.Obj)
        print(f"  MGA objective value = {mgaValue:.6g}, total cost = {cost:.6g} "
              f"({(cost - optimalCost) / abs(optimalCost):+.2%} vs. cost-optimal)", flush=True)
        results[objSpec["name"]] = {
            "cost": cost,
            "mgaObjective": mgaValue,
            "summary": _snapshot(esM, **summaryKwargs),
        }
        if save_mga_results_in_excel:
            results[objSpec["name"]]["excelFile"] = writeExcel(
                esM, excelOutputDir, f"{excelFilePrefix}_{objSpec['name']}", excelKwargs)
        print(f"Alternative '{objSpec['name']}' done ({time.perf_counter() - tObj:.1f} s).",
              flush=True)

    print("=" * 70, flush=True)
    print(f"MGA run finished in {time.perf_counter() - tStart:.1f} s.", flush=True)
    print("=" * 70, flush=True)
    return results