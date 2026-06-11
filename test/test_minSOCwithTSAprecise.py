import pandas as pd
import pyomo.environ as pyomo
import pytest


@pytest.mark.parametrize(
    "has_capacity_variable, has_segmentation",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_minSOCwithTSAprecise_at_min_soc_compensates_self_discharge(
    single_node_test_esM,
    has_capacity_variable,
    has_segmentation,
):
    """Test the minimum state-of-charge constraint created by
    StorageModel.minSOCwithTSAprecise() using the conftest EnergySystemModel.

    The test verifies the boundary case where the storage starts exactly at its
    minimum allowed state of charge. Due to self-discharge, the effective state
    of charge decreases over time. The intra-period stateOfCharge variable is
    set to exactly compensate this loss.

    The resulting minimum SOC constraint must be exactly binding.
    """
    esM = single_node_test_esM

    loc = "Location"
    comp_name = "Pressure tank"

    ip = 0
    p_inter = 0
    period = 0
    t = 2

    cap_value = 100
    min_soc_value = 0.33
    self_discharge = 0.1
    hours_per_time_step = 1
    segment_start_time = 4

    esM.hoursPerTimeStep = hours_per_time_step
    esM.periods = [p_inter]
    esM.timeStepsPerPeriod = [t]
    esM.periodsOrder = {
        ip: {
            p_inter: period,
        }
    }

    esM.segmentStartTime = {
        ip: pd.Series(
            data=[segment_start_time],
            index=pd.MultiIndex.from_tuples([(period, t)]),
        )
    }

    storage_model = esM.componentModelingDict[esM.componentNames[comp_name]]
    storage_component = storage_model.componentsDict[comp_name]
    abbrv_name = storage_model.abbrvName

    storage_component.hasCapacityVariable = has_capacity_variable
    storage_component.selfDischarge = self_discharge
    storage_component.processedStateOfChargeMin = {
        ip: {
            loc: pd.DataFrame(
                data=[[min_soc_value]],
                index=[p_inter],
                columns=[t],
            )
        }
    }

    if has_segmentation:
        exponent = segment_start_time * hours_per_time_step
    else:
        exponent = t * hours_per_time_step

    if has_capacity_variable:
        min_soc_absolute = cap_value * min_soc_value
    else:
        min_soc_absolute = min_soc_value

    soc_inter_value = min_soc_absolute
    soc_value = min_soc_absolute - min_soc_absolute * (1 - self_discharge) ** exponent

    pyM = pyomo.ConcreteModel()
    pyM.hasSegmentation = has_segmentation

    setattr(
        pyM,
        f"varSetPrecise_{abbrv_name}",
        pyomo.Set(
            initialize=[(loc, comp_name, ip)],
            dimen=3,
        ),
    )

    setattr(
        pyM,
        f"socInterSet_{abbrv_name}",
        pyomo.Set(
            initialize=[(loc, comp_name, ip, p_inter)],
            dimen=4,
        ),
    )

    setattr(
        pyM,
        f"stateOfChargeInterPeriods_{abbrv_name}",
        pyomo.Var(
            getattr(pyM, f"socInterSet_{abbrv_name}"),
            initialize=soc_inter_value,
        ),
    )

    setattr(
        pyM,
        f"stateOfCharge_{abbrv_name}",
        pyomo.Var(
            [(loc, comp_name, ip, period, t)],
            initialize=soc_value,
        ),
    )

    setattr(
        pyM,
        f"cap_{abbrv_name}",
        pyomo.Var(
            [(loc, comp_name, ip)],
            initialize=cap_value,
        ),
    )

    soc_inter = getattr(pyM, f"stateOfChargeInterPeriods_{abbrv_name}")
    soc = getattr(pyM, f"stateOfCharge_{abbrv_name}")
    cap = getattr(pyM, f"cap_{abbrv_name}")

    soc_inter[loc, comp_name, ip, p_inter].fix(soc_inter_value)
    soc[loc, comp_name, ip, period, t].fix(soc_value)
    cap[loc, comp_name, ip].fix(cap_value)

    storage_model.minSOCwithTSAprecise(pyM, esM)

    constraint = getattr(pyM, f"ConstrSOCMinPrecise_{abbrv_name}")[
        loc,
        comp_name,
        ip,
        p_inter,
        t,
    ]

    expected_soc_after_self_discharge = (
        soc_inter_value * (1 - self_discharge) ** exponent
    )
    expected_self_discharge_loss = min_soc_absolute - expected_soc_after_self_discharge

    assert hasattr(pyM, f"ConstrSOCMinPrecise_{abbrv_name}")
    assert len(getattr(pyM, f"ConstrSOCMinPrecise_{abbrv_name}")) == 1

    assert soc_inter_value == pytest.approx(min_soc_absolute)
    assert soc_value == pytest.approx(expected_self_discharge_loss)

    body_value = pyomo.value(constraint.body)

    if constraint.lower is not None:
        lower_value = pyomo.value(constraint.lower)
        assert body_value == pytest.approx(lower_value)

    if constraint.upper is not None:
        upper_value = pyomo.value(constraint.upper)
        assert body_value == pytest.approx(upper_value)

    assert constraint.equality is False
