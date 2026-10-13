"""Multi-stage stochastic optimization of the 08b example model, runnable in parallel with MPI.

This script builds exactly the model of ``08b_Multi-Stage_Stochastic_Optimization.ipynb``
(two regions, 8760 hourly time steps, six investment periods, a 3-stage scenario tree with
four scenarios) and solves it with the multi-stage stochastic programming module.

Run it serially::

    python 08b_Multi-Stage_Stochastic_Optimization_parallel.py

or in parallel, with one MPI process (rank) per scenario::

    mpiexec -n 4 python 08b_Multi-Stage_Stochastic_Optimization_parallel.py --threads 3

Only progressive hedging (``--method ph``) runs in parallel across scenarios. The extensive
form (``--method ef``) is a single solve; it uses parallelism only inside the solver, so run
it without ``mpiexec`` and give it all threads.

Options::

    --method {ph,ef}         solution method (default: ph)
    --resolution {tsa,full}  12 typical days or full 8760 h (default: tsa)
    --typical-days N         number of typical days for --resolution tsa (default: 12)
    --rho R                  progressive hedging penalty parameter (default: 0.1)
    --threads T              Gurobi threads per process (default: cores / processes)
"""

import os

# Required on some Windows machines to avoid native crashes from conflicting MKL/OpenMP
# runtimes. It must be set before numpy is imported, and setting it here makes it apply to
# every MPI process regardless of how the shell was configured.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from mpi4py import MPI  # noqa: E402

import fine as fn  # noqa: E402
import fine.expansionModules.multiStageStochasticProgram as mssp  # noqa: E402

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def log(*args):
    """Print on rank 0 only, so that the output is not repeated by every process."""
    if rank == 0:
        print(*args, flush=True)


# ------------------------------------------------------------------------- arguments

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--method", choices=["ph", "ef"], default="ph")
parser.add_argument("--resolution", choices=["tsa", "full"], default="tsa")
parser.add_argument("--typical-days", type=int, default=12)
parser.add_argument("--rho", type=float, default=0.1)
parser.add_argument("--threads", type=int, default=None)
args = parser.parse_args()

threads = args.threads or max(1, (os.cpu_count() or 1) // size)
if args.method == "ef" and size > 1:
    log(
        "Note: the extensive form is a single solve and does not use the MPI processes; "
        "run it without mpiexec."
    )

# ------------------------------------------------------------------------------ data

dataPath = Path(__file__).resolve().parent.parent / "data" / "multiRegion" / "SpatialData"

# Two of the eight regions of the multi-regional example data stand in for "north" (the
# windiest region) and "south" (the region with the highest demand).
regions = {"north": "cluster_6", "south": "cluster_0"}

windData = pd.read_excel(dataPath / "Wind" / "maxOperationRateOnshore_el.xlsx", index_col=0)
demandData = pd.read_excel(dataPath / "Demands" / "electricityDemand_GWh_el.xlsx", index_col=0)

windCapacityFactor = pd.DataFrame(
    {loc: windData[region].to_numpy() for loc, region in regions.items()}
)
demandShape = pd.DataFrame(
    {
        loc: (demandData[region] / demandData[region].mean()).to_numpy()
        for loc, region in regions.items()
    }
)

numberOfTimeSteps = 8760
hoursPerTimeStep = 1


def demand(northLevel, southLevel):
    """Electricity demand per time step for the given average load per region in GW."""
    return (
        demandShape * pd.Series({"north": northLevel, "south": southLevel})
    ) * hoursPerTimeStep


def gasPriceFromEuroPerMWh(euroPerMWh):
    """Convert a gas price in Euro/MWh into FINE's cost unit per GWh."""
    return euroPerMWh * 1e-6


# ------------------------------------------------------------------------ base model

esM = fn.EnergySystemModel(
    locations={"north", "south"},
    commodities={"electricity", "naturalGas"},
    numberOfTimeSteps=numberOfTimeSteps,
    commodityUnitsDict={
        "electricity": r"GW$_{el}$",
        "naturalGas": r"GW$_{CH_{4},LHV}$",
    },
    hoursPerTimeStep=hoursPerTimeStep,
    costUnit="1e9 Euro",
    startYear=2020,
    numberOfInvestmentPeriods=6,
    investmentPeriodInterval=5,
    lengthUnit="km",
    verboseLogLevel=2,
)

esM.add(
    fn.Source(
        esM,
        name="Wind",
        commodity="electricity",
        hasCapacityVariable=True,
        operationRateMax=windCapacityFactor,
        capacityMax=pd.Series({"north": 60.0, "south": 40.0}),
        investPerCapacity=1.20,
        opexPerCapacity=1.20 * 0.02,
        interestRate=0.05,
        economicLifetime=25,
        technicalLifetime=25,
    )
)
esM.add(
    fn.Source(
        esM,
        name="GasPurchase",
        commodity="naturalGas",
        hasCapacityVariable=False,
        commodityCost=gasPriceFromEuroPerMWh(30),  # placeholder, set per scenario node
    )
)
esM.add(
    fn.Conversion(
        esM,
        name="GasTurbine",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={"naturalGas": -1 / 0.6, "electricity": 1.0},
        hasCapacityVariable=True,
        investPerCapacity=0.60,
        opexPerCapacity=0.60 * 0.03,
        interestRate=0.05,
        economicLifetime=25,
        technicalLifetime=25,
    )
)
esM.add(
    fn.Transmission(
        esM,
        name="Grid",
        commodity="electricity",
        hasCapacityVariable=True,
        distances=pd.DataFrame(
            [[0, 400], [400, 0]], index=["north", "south"], columns=["north", "south"]
        ),
        losses=0.0001,
        investPerCapacity=0.001,
        interestRate=0.05,
        economicLifetime=40,
        technicalLifetime=40,
    )
)
esM.add(
    fn.Sink(
        esM,
        name="Demand",
        commodity="electricity",
        hasCapacityVariable=False,
        operationRateFix=demand(30.0, 20.0),  # placeholder, set per scenario node
    )
)

# --------------------------------------------------------------------- scenario tree

stageInvestmentPeriods = [[0], [1, 2], [3, 4, 5]]


def node(parent, probability, demandValue, gasPrice):
    return {
        "parent": parent,
        "probability": probability,
        "values": {
            "Demand": {"operationRateFix": demandValue},
            "GasPurchase": {"commodityCost": gasPrice},
        },
    }


def perPeriod(function, values):
    return {year: function(*value) for year, value in zip((2035, 2040, 2045), values)}


growthDemand = perPeriod(demand, [(38.0, 25.0), (40.0, 26.5), (42.0, 28.0)])
stagnationDemand = perPeriod(demand, [(31.0, 21.0), (31.5, 21.5), (32.0, 22.0)])
gas = gasPriceFromEuroPerMWh

tree = {
    "today": node(None, 1.0, demand(30.0, 20.0), gas(30)),
    "growth": node("today", 0.65, demand(34.5, 23.0), gas(38)),
    "stagnation": node("today", 0.35, demand(30.0, 20.0), gas(27)),
    "growth_gasCrisis": node(
        "growth", 0.30, growthDemand, {2035: gas(105), 2040: gas(120), 2045: gas(115)}
    ),
    "growth_cheapGas": node(
        "growth", 0.70, growthDemand, {2035: gas(30), 2040: gas(28), 2045: gas(27)}
    ),
    "stagnation_gasCrisis": node(
        "stagnation", 0.20, stagnationDemand, {2035: gas(100), 2040: gas(110), 2045: gas(105)}
    ),
    "stagnation_cheapGas": node(
        "stagnation", 0.80, stagnationDemand, {2035: gas(26), 2040: gas(25), 2045: gas(24)}
    ),
}
scenarioTree = mssp.ScenarioTree(tree)

# ----------------------------------------------------------------------------- solve

tsa = args.resolution == "tsa"
log(
    f"Solving with {args.method.upper()} on {size} process(es), {threads} Gurobi thread(s) "
    f"each, {'%d typical days' % args.typical_days if tsa else 'full 8760 h resolution'}"
    + (f", rho={args.rho}" if args.method == "ph" else "")
)

start = time.time()
results = mssp.optimizeMultiStageStochastic(
    esM,
    scenarioTree,
    stageInvestmentPeriods=stageInvestmentPeriods,
    method=args.method,
    solver="gurobi",
    solverOptions={"Threads": threads},
    mpisppyOptions=(
        {"PHIterLimit": 200, "defaultPHrho": args.rho, "convthresh": 1e-6}
        if args.method == "ph"
        else None
    ),
    timeSeriesAggregation=tsa,
    temporalAggregationSpecs=(
        {"n_clusters": args.typical_days, "period_duration": 24, "segments": None}
        if tsa
        else None
    ),
)
elapsed = comm.allreduce(time.time() - start, op=MPI.MAX)

# --------------------------------------------------------------------------- results

# Every process only holds the scenarios it solved. Collect the commissioning decisions of
# all scenarios on rank 0.
localRows = {}
for scenarioName, scenarioEsM in results.scenarioModels.items():
    for mdlName, abbrv, compName in [
        ("SourceSinkModel", "srcSnk", "Wind"),
        ("ConversionModel", "conv", "GasTurbine"),
    ]:
        variable = getattr(scenarioEsM.pyM, "commis_" + abbrv)
        localRows[(compName, scenarioName)] = {
            scenarioEsM.investmentPeriodNames[ip]: sum(
                variable[key].value for key in variable if key[1] == compName and key[2] == ip
            )
            for ip in scenarioEsM.investmentPeriods
        }
allRows = comm.gather(localRows, root=0)

if rank == 0:
    rows = {key: value for part in allRows for key, value in part.items()}
    order = [(c, s) for s in scenarioTree.scenarioNames for c in ("Wind", "GasTurbine")]
    commissioningTable = pd.DataFrame(rows).T.loc[order].round(2)
    commissioningTable.index.names = ["component", "scenario"]

    firstStage = pd.Series(
        {
            f"{compName} ({loc})": round(value, 3)
            for (_, compName, loc, _), value in results.firstStageDecisions().items()
            if abs(value) > 1e-6
        }
    ).sort_index()

    print(f"\nwall-clock time        : {elapsed:.1f} s")
    print(f"expected NPV           : {results.objectiveValue:.4f} {esM.costUnit}")
    print("\nfirst-stage decisions (commissioned in 2020):")
    print(firstStage.to_string())
    print("\ncommissioning per scenario:")
    print(commissioningTable.to_string())
