import os
import re
import time
from pathlib import Path

from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
import pyomo.environ as pyo
from pyomo.opt import SolverStatus as opt_SolverStatus
from pyomo.opt import TerminationCondition as opt_TerminationCondition

"""
Modeling to Generate Alternatives (MGA) on top of ETHOS.FINE.

Workflow
  1. Solve the cost-optimal problem with esM.optimize().
  2. Add a constraint: total cost <= optimal cost + slack * |optimal cost|.
  3. Replace FINE's cost objective with a weighted sum of capacity / operation
     variables (minimized or maximized).
  4. Re-solve the existing Pyomo model (declaresOptimizationProblem=False).

  Operation variables are aggregated to annual sums (period occurrences and
  number of years are taken into account, so time series aggregation works).

  persistent=True (Gurobi only)
  The default path re-solves with esM.optimize(), which hands the complete
  Pyomo model to Gurobi again for every alternative. With persistent=True the
  model is transferred to a gurobi_persistent instance once; for every
  alternative only the objective is replaced (and the cost constraint is
  added once). For large models this saves the transfer time per alternative.
"""

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
    """Return how many times typical period p occurs, for investment period ip if applicable."""
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
    """Weighted objective terms for one user-defined term (see runMGA's 'custom' spec)."""
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
        available = [
            v.name
            for v in pyM.component_objects(pyo.Var)
            if v.name.endswith("_" + mdl.abbrvName)
        ]
        raise ValueError(
            f"'{kind}' is not available for '{comp}' "
            f"({type(mdl).__name__}). Variables of this class: {available}"
        )

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
    """Resolve the weight of a component/variable pair, honoring per-variable overrides."""
    w = weights.get(comp, default)
    if isinstance(w, dict):  # per-variable override, e.g. {"capacity": 0.5}
        w = w.get(kind, default)
    return float(w)


def allVariableParts(esM, objSpec, includeOperation):
    """Weighted objective terms for all capacity (and, optionally, operation) variables."""
    pyM = esM.pyM
    default = float(objSpec.get("defaultWeight", 1.0))
    weights = objSpec.get("weights", {})
    exclude = set(objSpec.get("excludeComponents", []))
    unknown = (set(weights) | exclude) - set(esM.componentNames)
    if unknown:
        raise KeyError(
            f"Unknown components in weights/excludeComponents: {sorted(unknown)}"
        )

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
    """Weighted objective terms for one MGA objective, dispatched on variableSelection."""
    selection = objSpec.get("variableSelection", "custom")
    if selection == "custom":
        if not objSpec.get("terms"):
            raise ValueError(
                f"Objective '{objSpec.get('name')}' is 'custom' but has no terms."
            )
        parts = [p for t in objSpec["terms"] for p in termParts(esM, t)]
    elif selection == "allCapacity":
        parts = allVariableParts(esM, objSpec, includeOperation=False)
    elif selection == "allCapacityAndOperation":
        parts = allVariableParts(esM, objSpec, includeOperation=True)
    else:
        raise ValueError(
            f"variableSelection must be one of {selections}, got '{selection}'."
        )
    if not parts:
        raise ValueError(f"Objective '{objSpec.get('name')}' contains no variables.")
    return parts


def setMGAObjective(esM, objSpec):
    """Replace the active objective with the weighted MGA objective."""
    pyM = esM.pyM
    t0 = time.perf_counter()
    sense = objSpec.get("sense", "max")
    selection = objSpec.get("variableSelection", "custom")
    print(
        f"  Declaring MGA objective function ({sense}, variableSelection='{selection}') ...",
        flush=True,
    )

    if pyM.find_component("Obj") is not None:
        print("    Removing previous objective function ...", flush=True)
        pyM.del_component("Obj")

    parts = objectiveParts(esM, objSpec)
    print(
        f"    Building objective expression from {len(parts)} variables ...", flush=True
    )
    pyoSense = pyo.maximize if sense.lower().startswith("max") else pyo.minimize
    pyM.Obj = pyo.Objective(
        expr=pyo.quicksum(parts), sense=pyoSense
    )  # keep FINE's name "Obj"
    print(
        f"  MGA objective function declared ({time.perf_counter() - t0:.1f} s).",
        flush=True,
    )


def addCostSlackConstraint(esM, costExpr, optimalCost, slack):
    """Constrain total cost to optimalCost + slack * |optimalCost|."""
    pyM = esM.pyM
    t0 = time.perf_counter()
    bound = optimalCost + slack * abs(optimalCost)
    print(
        f"Declaring cost constraint: total cost <= {bound:.6g} "
        f"(optimal cost {optimalCost:.6g}, slack {slack:.1%}) ...",
        flush=True,
    )
    if pyM.find_component("mgaCostConstraint") is not None:
        pyM.del_component("mgaCostConstraint")
    # abs() keeps the bound correct if the optimum is negative (e.g. export revenues)
    pyM.mgaCostConstraint = pyo.Constraint(expr=costExpr <= bound)
    print(f"Cost constraint declared ({time.perf_counter() - t0:.1f} s).", flush=True)


def _snapshot(esM, **summaryKwargs):
    """Copy the summaries, because every solve overwrites them in esM."""
    print("  Collecting optimization summaries ...", flush=True)
    return {
        name: esM.getOptimizationSummary(name, outputLevel=2, **summaryKwargs).copy()
        for name in esM.componentModelingDict
    }


def writeExcel(esM, outputDir, fileStem, excelKwargs):
    """Write the current esM results with FINE's writeOptimizationOutputToExcel."""
    t0 = time.perf_counter()
    safeStem = re.sub(r"[^\w\-]+", "_", fileStem)  # file-system safe name
    outputFileName = Path(outputDir) / safeStem
    print(f"  Writing results to Excel: {outputFileName}.xlsx ...", flush=True)
    writeOptimizationOutputToExcel(
        esM, outputFileName=str(outputFileName), **excelKwargs
    )
    print(f"  Excel file written ({time.perf_counter() - t0:.1f} s).", flush=True)
    return str(outputFileName) + ".xlsx"




# --------------------------------------------------------------------------- #
# Cost objective bookkeeping (shared by both solve paths)
# --------------------------------------------------------------------------- #
def _costExpression(esM):
    """Return FINE's cost expression and whether the model is still in its original state.

    The expression is stored on the Pyomo model the first time, so that repeated
    runMGA calls on the same esM keep using the cost objective even after Obj
    was replaced by an MGA objective.
    """
    pyM = esM.pyM
    stored = getattr(pyM, "_mgaCostExpr", None)
    if stored is None:
        pyM._mgaCostExpr = pyM.Obj.expr
        return pyM._mgaCostExpr, True
    return stored, False


def _restoreCostObjective(esM, costExpr):
    """Remove MGA objective / cost constraint of a previous run and restore the cost objective."""
    pyM = esM.pyM
    if pyM.find_component("mgaCostConstraint") is not None:
        pyM.del_component("mgaCostConstraint")
    if pyM.find_component("Obj") is not None:
        pyM.del_component("Obj")
    pyM.Obj = pyo.Objective(expr=costExpr, sense=pyo.minimize)


# --------------------------------------------------------------------------- #
# Persistent Gurobi path
# --------------------------------------------------------------------------- #
optimizeDefaults = {
    "timeSeriesAggregation": False,
    "relaxIsBuiltBinary": False,
    "logFileName": "",
    "threads": 3,
    "solver": "gurobi",
    "timeLimit": None,
    "optimizationSpecs": "",
    "warmstart": False,
    "relevanceThreshold": None,
}


def _optimizeSettings(optimizeKwargs):
    """Fill the esM.optimize keyword arguments with FINE's defaults."""
    unknown = set(optimizeKwargs) - set(optimizeDefaults) - {"includePerformanceSummary"}
    if unknown:
        raise TypeError(f"Unknown optimize arguments for persistent=True: {sorted(unknown)}")
    settings = {**optimizeDefaults, **optimizeKwargs}
    settings.pop("includePerformanceSummary", None)
    if settings["solver"] != "gurobi":
        raise ValueError(
            f"persistent=True requires solver='gurobi', got '{settings['solver']}'."
        )
    return settings


def _createPersistentSolver(esM, settings):
    """Create a gurobi_persistent instance with the same options esM.optimize would set."""
    wlsaccessid = os.environ.get("WLSACCESSID", "")
    wlssecret = os.environ.get("WLSSECRET", "")
    licenseid = os.environ.get("LICENSEID", "")
    if wlsaccessid and wlssecret and licenseid:
        # same WLS handling as esM.optimize, via Pyomo's managed Gurobi environment
        opt = pyo.SolverFactory(
            "gurobi_persistent",
            manage_env=True,
            options={
                "WLSACCESSID": wlsaccessid,
                "WLSSECRET": wlssecret,
                "LICENSEID": int(licenseid),
            },
        )
    else:
        opt = pyo.SolverFactory("gurobi_persistent")

    specs = settings["optimizationSpecs"] or ""
    if "LogToConsole=" not in specs and esM.verboseLogLevel == 2:
        specs += " LogToConsole=0"
    optionString = f"Threads={settings['threads']} {specs}"
    if settings["logFileName"]:
        optionString += f" logfile={settings['logFileName']}"
    opt.set_options(optionString)
    if settings["timeLimit"] is not None:
        opt.options["timelimit"] = settings["timeLimit"]
    return opt


def _solvePersistent(esM, opt, settings, label, acceptSuboptimal=True):
    """Solve the model held by the persistent solver and run FINE's post-processing.

    Gurobi's barrier sometimes stalls very close to the optimum and reports
    SUBOPTIMAL (status 13). Pyomo then does not load the solution and FINE
    generates no output, so the summaries silently keep the values of the
    previous solve. With acceptSuboptimal=True such a solution is loaded and
    post-processed like an optimal one, and a warning with the remaining
    primal-dual gap is printed. Returns (solverInfo, statusText).
    """
    # stores the optimize settings in esM.solverSpecs, like esM.optimize does
    timeStart, _, _ = esM._prepareOptimization(
        False,                                  # declaresOptimizationProblem
        settings["relaxIsBuiltBinary"],
        settings["timeSeriesAggregation"],
        settings["logFileName"],
        settings["threads"],
        "gurobi",
        settings["timeLimit"],
        settings["optimizationSpecs"],
        settings["warmstart"],
        settings["relevanceThreshold"],
        False,                                  # includePerformanceSummary
    )
    print(f"  Solving ({label}, persistent Gurobi) ...", flush=True)
    solverInfo = opt.solve(tee=True, warmstart=settings["warmstart"], load_solutions=False)
    esM.solverSpecs["solvetime"] = time.time() - timeStart
    esM.solverSpecs["solver"] = "gurobi_persistent"

    grbModel = opt._solver_model
    grbStatus = grbModel.Status                 # 2 = OPTIMAL, 13 = SUBOPTIMAL
    statusText = {2: "optimal", 13: "suboptimal"}.get(grbStatus, f"gurobi status {grbStatus}")

    if grbStatus == 13 and acceptSuboptimal and grbModel.SolCount > 0:
        primal = grbModel.ObjVal
        try:                                    # not available for every LP / Gurobi version
            dual = grbModel.ObjBound
            gapText = f"relative gap {abs(primal - dual) / max(abs(primal), 1e-10):.1e}"
        except Exception:
            gapText = "gap: see the last barrier iteration in the Gurobi log"
        print(f"  WARNING: '{label}' ended SUBOPTIMAL (barrier stalled). Accepting the solution: "
              f"objective {primal:.8g}, {gapText}.", flush=True)
        solverInfo.solver.status = opt_SolverStatus.ok
        solverInfo.solver.termination_condition = opt_TerminationCondition.optimal
        statusText = f"suboptimal (accepted; {gapText})"
    elif grbStatus == 13:
        print(f"  WARNING: '{label}' ended SUBOPTIMAL and acceptSuboptimal=False: "
              "no solution loaded, the summaries keep the previous values.", flush=True)

    if solverInfo.solver.termination_condition == opt_TerminationCondition.optimal:
        opt.load_vars()
        dual = esM.pyM.find_component("dual")
        if dual is not None and isinstance(dual, pyo.Suffix) and dual.import_enabled():
            try:
                opt.load_duals()
            except Exception as e:              # duals are optional for the summaries
                print(f"  Note: duals could not be loaded ({e}).", flush=True)
    else:
        print(f"  WARNING: '{label}' did not solve ({statusText}); "
              "the summaries keep the values of the previous solve.", flush=True)

    esM._runPostprocessing(solverInfo, timeStart)
    esM.solverSpecs["mgaSolveStatus"] = statusText
    return solverInfo, statusText


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def runMGA(
    esM,
    spec,
    summaryKwargs=None,
    save_mga_results_in_excel=False,
    excelOutputDir="mga_results",
    excelFilePrefix="MGA",
    excelKwargs={"optSumOutputLevel": 2, "optValOutputLevel": 2},
    persistent=False,
    solverModelFile=None,
    optimalCost=None,
    acceptSuboptimal=True,
    **optimizeKwargs,
):
    """Solve cost-optimal, then each MGA objective in spec["objectives"].

    **Arguments:**

    :param spec: states the parameters required for the MGA optimization.

        ``spec`` has two top-level keys:

        - ``"slack"``: allowed relative cost increase over the cost-optimal
          solution (e.g. ``0.10`` for 10 %).
        - ``"objectives"``: a list of MGA objectives. Each entry describes one
          alternative to generate; to generate several alternatives, add
          several objectives (e.g. with varying weights). Every objective is
          a dictionary with a ``"name"``, a ``"sense"`` (``"min"`` or
          ``"max"``), and a ``"variableSelection"`` that selects which
          variables make up the objective. Three selections are available,
          shown below as examples (a), (b), and (c).

        (a) ``"custom"`` (default) - objective built from explicitly listed
        (component, variable) terms::

            {
                "name": "max_wind",
                "sense": "max",
                "variableSelection": "custom",
                "terms": [
                    {"component": "Wind (onshore)", "variable": "capacity", "weight": 1.0},
                    {
                        "component": "PV",
                        "variable": "operation",
                        "weight": 0.5,
                        "locations": ["region1"],        # optional, default: all
                        "investmentPeriods": [2030],      # optional, default: all
                    },
                ],
            }

        (b) ``"allCapacity"`` - objective built from the capacity variable of
        every component in the model::

            {
                "name": "min_all_capacity",
                "sense": "min",
                "variableSelection": "allCapacity",
                "defaultWeight": 1.0,                      # optional, default: 1.0
                "weights": {                                # optional per-component/-variable overrides
                    "PV": 2.0,
                    "Li-ion": {"capacity": 0.5},
                },
                "excludeComponents": ["Electricity demand"],  # optional
            }

        (c) ``"allCapacityAndOperation"`` - objective built from the capacity
        variable and the operation/charge/discharge variable of every
        component in the model::

            {
                "name": "min_all_cap_and_op",
                "sense": "min",
                "variableSelection": "allCapacityAndOperation",
            }

        Putting it together, a full ``spec`` combining all three examples
        looks like::

            spec = {
                "slack": 0.10,
                "objectives": [
                    {...},  # (a), (b), or (c) as shown above
                    {...},
                    {...},
                ],
            }

        Each ``variable`` must match the modeling class of its ``component``:

        ========== ==============================================
        variable    valid component classes
        ========== ==============================================
        "capacity"  all modeling classes
        "operation" SourceSink, Conversion, Transmission
        "charge"    Storage
        "discharge" Storage
        ========== ==============================================

        E.g. a storage component has no "operation" variable, only "charge"
        and/or "discharge".

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

    :param persistent: if True, the model is transferred to a gurobi_persistent solver
        instance once and every MGA alternative only replaces the objective function.
        If False, every alternative is solved with esM.optimize(), which transfers the
        complete model to the solver again. Requires solver="gurobi".

        * esM already optimized (cost-optimal solution available): the solution is reused,
          the model is transferred once and only the alternatives are solved.
        * esM declared but not optimized, or not declared at all: the problem is declared
          if necessary and the cost-optimal solution is computed with the persistent solver.

        |br| * the default value is False
    :type persistent: boolean

    :param solverModelFile: only with persistent=True. If given, the Gurobi model is written
        to this file right after the transfer (e.g. "model.mps.bz2"), so that it can be solved
        or tuned outside Python (gurobi_cl, gurobipy.read).

        |br| * the default value is None
    :type solverModelFile: string

    :param optimalCost: cost-optimal objective value of exactly this model (e.g. stored after an
        earlier cost-optimal run). If given, the cost-optimal solve is skipped and this value is
        used for the near-optimal cost constraint; results["costOptimal"]["summary"] is then None.

        |br| * the default value is None
    :type optimalCost: float

    :param acceptSuboptimal: only with persistent=True. If Gurobi's barrier ends SUBOPTIMAL
        (stalled close to the optimum), load and post-process that solution anyway and print the
        remaining primal-dual gap. Without this the summaries would silently keep the values of the
        previous solve. The status of every solve is stored in results[name]["status"].

        |br| * the default value is True
    :type acceptSuboptimal: boolean

    :param optimizeKwargs: optimizeKwargs are passed to esM.optimize (solver, timeSeriesAggregation, ...).
        With persistent=True the same arguments are used for the persistent solver.

       |br| * the default value is {}
    :type optimizeKwargs: dictionary

    """
    tStart = time.perf_counter()
    summaryKwargs = summaryKwargs or {}
    excelKwargs = excelKwargs or {}
    if save_mga_results_in_excel:
        Path(excelOutputDir).mkdir(parents=True, exist_ok=True)
    optimizeKwargs.pop("declaresOptimizationProblem", None)
    if solverModelFile and not persistent:
        raise ValueError("solverModelFile requires persistent=True.")
    nObj = len(spec["objectives"])

    print("=" * 70, flush=True)
    print(
        f"MGA run: slack {spec['slack']:.1%}, {nObj} alternative(s)"
        + (", persistent Gurobi" if persistent else ""),
        flush=True,
    )
    if save_mga_results_in_excel:
        print(f"Excel output enabled: {Path(excelOutputDir).resolve()}", flush=True)
    print("=" * 70, flush=True)

    opt = None
    try:
        # 0) model and solver set-up
        if persistent:
            settings = _optimizeSettings(optimizeKwargs)
            if esM.pyM is None:
                print("Declaring optimization problem ...", flush=True)
                esM.declareOptimizationProblem(
                    timeSeriesAggregation=settings["timeSeriesAggregation"],
                    relaxIsBuiltBinary=settings["relaxIsBuiltBinary"],
                    relevanceThreshold=settings["relevanceThreshold"],
                )
        elif esM.pyM is None:
            # nothing declared yet: declare and solve cost-optimal with esM.optimize
            print("Declaring and solving the cost-optimal problem ...", flush=True)
            esM.optimize(declaresOptimizationProblem=True, **optimizeKwargs)

        costExpr, firstRun = _costExpression(esM)
        if not firstRun:
            # a previous runMGA call replaced Obj: restore the cost objective
            print("Restoring the cost objective of a previous MGA run ...", flush=True)
            _restoreCostObjective(esM, costExpr)

        # cost-optimal solution can be reused if esM was optimized and nothing was changed since
        reuseCostOptimal = firstRun and esM.objectiveValue is not None

        if persistent:
            print("Transferring the model to Gurobi (once) ...", flush=True)
            t0 = time.perf_counter()
            opt = _createPersistentSolver(esM, settings)
            opt.set_instance(esM.pyM)
            print(f"Model transferred ({time.perf_counter() - t0:.1f} s).", flush=True)
            if solverModelFile:
                print(f"Writing Gurobi model to {solverModelFile} ...", flush=True)
                opt._solver_model.write(str(solverModelFile))

        # 1) cost-optimal solution
        providedCost = optimalCost
        costStatus = "provided"
        if providedCost is not None:
            print(f"Using the provided cost-optimal value {providedCost:.6g} "
                  "(cost-optimal solve skipped) ...", flush=True)
        elif reuseCostOptimal:
            print("Accessing the cost-optimal solution ...", flush=True)
            costStatus = "reused"
        else:
            print("Solving the cost-optimal problem ...", flush=True)
            if persistent:
                _, costStatus = _solvePersistent(esM, opt, settings, "cost-optimal",
                                                 acceptSuboptimal)
            else:
                esM.optimize(declaresOptimizationProblem=False, **optimizeKwargs)
                costStatus = esM.solverSpecs.get("terminationCondition", "")
        optimalCost = providedCost if providedCost is not None else pyo.value(costExpr)
        print(f"Cost-optimal solution: total cost = {optimalCost:.6g}", flush=True)
        results = {
            "costOptimal": {
                "cost": optimalCost,
                "status": costStatus,
                "summary": None if providedCost is not None else _snapshot(esM, **summaryKwargs),
            }
        }
        if save_mga_results_in_excel and providedCost is None:
            results["costOptimal"]["excelFile"] = writeExcel(
                esM, excelOutputDir, f"{excelFilePrefix}_costOptimal", excelKwargs
            )

        # 2) near-optimal cost constraint
        print("Adding the near-optimal cost constraint ...", flush=True)
        addCostSlackConstraint(esM, costExpr, optimalCost, spec["slack"])
        if persistent:
            opt.add_constraint(esM.pyM.mgaCostConstraint)

        # 3) + 4) one solve per MGA objective, reusing the declared model
        print("Solving MGA alternatives ...", flush=True)
        for i, objSpec in enumerate(spec["objectives"], start=1):
            tObj = time.perf_counter()
            print("-" * 70, flush=True)
            print(f"Alternative {i}/{nObj}: '{objSpec['name']}'", flush=True)
            setMGAObjective(esM, objSpec)

            if persistent:
                opt.set_objective(esM.pyM.Obj)  # only the objective changes in Gurobi
                _, status = _solvePersistent(esM, opt, settings, objSpec["name"],
                                             acceptSuboptimal)
            else:
                print("  Solving MGA problem ...", flush=True)
                esM.optimize(declaresOptimizationProblem=False, **optimizeKwargs)
                status = esM.solverSpecs.get("terminationCondition", "")

            cost = pyo.value(costExpr)
            mgaValue = pyo.value(esM.pyM.Obj)
            print(
                f"  MGA objective value = {mgaValue:.6g}, total cost = {cost:.6g} "
                f"({(cost - optimalCost) / abs(optimalCost):+.2%} vs. cost-optimal)",
                flush=True,
            )
            results[objSpec["name"]] = {
                "cost": cost,
                "status": status,
                "mgaObjective": mgaValue,
                "summary": _snapshot(esM, **summaryKwargs),
            }
            if save_mga_results_in_excel:
                results[objSpec["name"]]["excelFile"] = writeExcel(
                    esM, excelOutputDir, f"{excelFilePrefix}_{objSpec['name']}", excelKwargs
                )
            print(
                f"Alternative '{objSpec['name']}' done ({time.perf_counter() - tObj:.1f} s).",
                flush=True,
            )
    finally:
        if opt is not None and hasattr(opt, "close"):
            opt.close()  # releases the Gurobi model (and environment, if managed)

    print("=" * 70, flush=True)
    print(f"MGA run finished in {time.perf_counter() - tStart:.1f} s.", flush=True)
    print("=" * 70, flush=True)
    return results
