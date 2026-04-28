import fine as fn
import numpy as np
import pandas as pd
import pytest

from fine.utils import ImplementedSolvers


@pytest.fixture(scope="function")
def cyclic_lifetime_esM():
    """Minimal 2-location, 3-IP system designed to force battery cycling.

    Solar generates only in period t=0; demand occurs only in period t=1.
    The battery is the cheapest way to shift energy, but its annual charge
    volume is bounded by cyclicLifetime. An expensive backup source keeps
    the model feasible when the battery is fully constrained.
    """
    locations = {"Loc1"}
    # 2 time steps × 4380 h → numberOfYears = 1, giving intuitive annual values
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity"},
        numberOfTimeSteps=10,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=8760/10,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=5,
        investmentPeriodInterval=1,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )

    # Solar: only generates in the first half-year (t=0)
    solar_profile = pd.DataFrame(
        {"Loc1": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]},
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Solar",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=solar_profile,
            investPerCapacity=0.1,
            interestRate=0.0,
            economicLifetime=10,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=100,
        )
    )

    # Expensive backup ensures feasibility when the battery is constrained
    esM.add(
        fn.Source(
            esM=esM,
            name="Backup",
            commodity="electricity",
            hasCapacityVariable=False,
            opexPerOperation=1e4,
        )
    )

    return esM


def test_cyclic_lifetime_constraint_multi_ip(cyclic_lifetime_esM):
    """Verify the per-vintage cyclicLifetime constraint.

    The constraint sums charge throughput over ALL active IPs for each
    commissioning vintage and checks that the total does not exceed
    commisVar × (SoCmax - SoCmin) × cyclicLifetime.
    """
    esM = cyclic_lifetime_esM
    cyclic_lifetime = 8
    economic_lifetime = 10

    esM.add(
        fn.Storage(
            esM=esM,
            name="Battery",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=1.0,
            dischargeEfficiency=1.0,
            chargeRate=1,
            dischargeRate=1,
            investPerCapacity=1.0,
            opexPerCapacity=0.0,
            # interestRate=0.0,
            economicLifetime=economic_lifetime,
            cyclicLifetime=cyclic_lifetime,
        )
    )

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    battery = esM.getComponent("Battery")

    # Read per-vintage Pyomo variables directly from the solved model
    pyM = esM.pyM
    chargeOpCommis_var = getattr(pyM, "chargeOpCommis_stor")
    commis_var = getattr(pyM, "commis_stor")
    commisConstrSet1 = getattr(pyM, "chargeOpCommisConstrSet1_stor")

    # Build (loc, compName, commis) → [active ips] mapping
    active_ips = {}
    for loc, compName, commis, ip in commisConstrSet1:
        active_ips.setdefault((loc, compName, commis), []).append(ip)

    total_commis = 0.0

    for (loc, compName, commis), ips in active_ips.items():
        commis_val = commis_var[loc, compName, commis].value or 0.0
        total_commis += commis_val

        # Reproduce the constraint LHS: total charge over all active IPs in real years
        total_charge = (
            sum(
                (chargeOpCommis_var[loc, compName, commis, ip, p, t].value or 0.0)
                * esM.periodOccurrences[ip][p]
                for ip in ips
                for p, t in pyM.intraYearTimeSet
            )
            * esM.investmentPeriodInterval
            / esM.numberOfYears
        )

        first_ip = min(ips)
        soc_max = battery.processedStateOfChargeMax[first_ip][loc].max()
        soc_min = battery.processedStateOfChargeMin[first_ip][loc].min()
        limit = commis_val * (soc_max - soc_min) * cyclic_lifetime

        assert total_charge <= limit + 1e-4, (
            f"cyclicLifetime violated for commis={commis} at {loc}: "
            f"total_charge={total_charge:.6f} > limit={limit:.6f}"
        )

    assert total_commis > 0, "No battery capacity commissioned — constraint check vacuous"
