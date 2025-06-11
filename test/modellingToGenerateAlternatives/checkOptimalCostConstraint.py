import pandas as pd
import fine as fn
import copy

from pyomo.environ import *

esM = fn.EnergySystemModel(
    locations={"Region1","Region2"},
    commodities={"electricity","heat"},
    numberOfTimeSteps=6,
    commodityUnitsDict={"electricity": r"GW$_{el}$", "heat": r"GW$_{heat}$"},
    hoursPerTimeStep=1,
    costUnit="1e9 Euro",
    lengthUnit="km",
    verboseLogLevel=0
)

demandProfile = [
    0.6,
    0.7,
    0.9,
    1,
    0.9,
    0.8,
]

demand = pd.DataFrame(
    [[u * 10, u * 20] for u in demandProfile],
    index=range(6),
    columns=["Region1", "Region2"],
)

esM.add(
    fn.Sink(
        esM=esM,
        name="Electricity demand",
        commodity="electricity",
        hasCapacityVariable=False,
        operationRateFix=demand,
    )
)

generationProfile = [
    0.05,
    0.15,
    0.2,
    0.0,
    0.8,
    0.7
]
operationRateFix = pd.DataFrame(
    [[u*4, u*8] for u in generationProfile],
    index=range(6),
    columns=["Region1", "Region2"],
)
capacityMax = pd.Series([100, 120], index=["Region1", "Region2"])
investPerCapacity, opexPerCapacity = 100, 10
interestRate, economicLifetime = 0.08, 25
esM.add(
    fn.Source(
        esM=esM,
        name="wind",
        commodity="electricity",
        hasCapacityVariable=True,
        operationRateFix=operationRateFix,
        capacityFix=capacityMax,
        investPerCapacity=investPerCapacity,
        opexPerCapacity=opexPerCapacity,
        interestRate=interestRate,
        economicLifetime=economicLifetime,
    )
)

esM.add(
    fn.Conversion(
        esM=esM,
        name="Heat Pump",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={"electricity": -1/2.5, "heat": 1},
        hasCapacityVariable=True,
        investPerCapacity=0.76,
        opexPerCapacity=0.00646,
        interestRate=0.06,
        economicLifetime=20,
    )
)

esM.add(
    fn.Storage(
        esM=esM,
        name="Li-ion batteries",
        commodity="electricity",
        hasCapacityVariable=True,
        chargeEfficiency=0.95,
        cyclicLifetime=10000,
        dischargeEfficiency=0.95,
        selfDischarge=1 - (1 - 0.03) ** (1 / (30 * 24)),
        chargeRate=1,
        dischargeRate=1,
        doPreciseTsaModeling=False,
        investPerCapacity=0.151,
        opexPerCapacity=0.002,
        interestRate=0.08,
        economicLifetime=22,
    )
)

acCables = pd.DataFrame(data =[[0,10],[10,0]], index=["Region1","Region2"], columns=["Region1","Region2"])
print(acCables)

esM.add(
    fn.Transmission(
        esM=esM,
        name="AC cables",
        commodity="electricity",
        hasCapacityVariable=True,
        capacityFix=acCables,
    )
)

esM.optimize(
    timeSeriesAggregation=False,
    optimizationSpecs="OptimalityTol=1e-3 method=2 cuts=0 MIPGap=5e-3",
)

esM.objectiveValue = copy.deepcopy(esM.pyM.Obj())
esM.slack = 0.1
esM.iterations = 2
fn.Mgaoptimize.calculateBeta(esM, random_seed=False)

fn.Mgaoptimize.declareMGAOptimizationProblem(esM, iteration=1, sense="maximize")


print("Optimum cost value:", esM.objectiveValue)
print("Slack value:", esM.slack)
print("optimum Cost Constraint:", esM.objectiveValue*(1+esM.slack))

print(esM.pyM.component("optimalCostConstraint").expr)

# optimizer = opt.SolverFactory("gurobi")
# solver_info = optimizer.solve(
#     esM.pyM,
#     warmstart=False,
#     tee=True,
# )

# for cname, cdata in esM.pyM.component_map(Constraint, active=True).items():
#     print(f"\nConstraint component: {cname}")
#     for index in cdata:
#         print(f"  Index: {index}")
#         print(f"  Expression: {cdata[index].expr}")
# print(esM.pyM.Obj.display())

# fn.Mgaoptimize.mgaOptimize(
#     esM,
#     timeSeriesAggregation = False,
#     solver='gurobi',
#     optimizationSpecs="OptimalityTol=1e-3 method=2 cuts=0 MIPGap=5e-3",
#     declaresOptimizationProblem=True, 
#     warmstart=False,
#     threads=0,
#     slack = 0.1,
#     iterations = 1,
#     random_seed = False,
#     writeSolutionsasExcels = False,
#     getOptimizationSummary = False,
#     outputLevel = 2,
#     operationRateinOutput = False
