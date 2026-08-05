"""Unit tests for the stacked layout the Zarr format is built on.

stackComponents and unstackComponents are an inverse pair over plain datasets, so
they can be checked without writing anything.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import fine.IOManagement.xarrayIO as xrIO
from fine.IOManagement import utilsIO


@pytest.mark.parametrize(
    "dims, expected",
    [
        ((), "0d_"),
        (("dim_0",), "0d_"),
        (("space",), "1d_"),
        (("space", "space_2"), "2d_"),
        (("time",), "ts_"),
        (("time", "space"), "ts_"),
        (("time", "space", "space_2"), "ts_"),
    ],
)
def test_netcdfPrefixForDims(dims, expected):
    """Every shape maps to the prefix the netCDF layout gives it."""
    assert utilsIO.netcdfPrefixForDims(dims) == expected


def _assertPrefixRuleHolds(esM):
    """Every canonical Input name has to follow the prefix rule.

    This is what lets the unstack rebuild the netCDF name from the mask alone. If
    it fails, it names the exact parameter.
    """
    datasets = xrIO.convertOptimizationInputToDatasets(esM)["Input"]
    checked = 0
    for model, components in datasets.items():
        for component, dataset in components.items():
            for name, variable in dataset.data_vars.items():
                prefix = str(name)[:3]
                indexNames = utilsIO.indexNamesOfVariable(variable, prefix)
                assert utilsIO.netcdfPrefixForDims(indexNames) == prefix, (
                    f"{model}/{component}/{name} has index names {indexNames}"
                )
                checked += 1
    assert checked, "the model has to carry at least one variable"


def test_prefix_rule_agrees_with_the_canonical_names(minimal_test_esM):
    _assertPrefixRuleHolds(minimal_test_esM)


def test_prefix_rule_agrees_with_the_canonical_names_single_node(single_node_test_esM):
    """A single location model is the hard case.

    squeeze() collapses "space" into a scalar coordinate, and the merge that
    follows attaches that coordinate to every variable of the component, including
    the 0d ones. The dimensions alone are therefore empty for both a 0d_ and a 1d_
    parameter, which is why indexNamesOfVariable takes the prefix as well.
    """
    _assertPrefixRuleHolds(single_node_test_esM)


def _assertUnstackUndoesStack(esM):
    """unstack(stack(x)) == x, per model class."""
    datasets = xrIO.convertOptimizationInputToDatasets(esM)["Input"]
    for model, components in datasets.items():
        stacked = utilsIO.stackComponents(components, prefixed=True)
        if stacked is None:  # a model class the esM holds no component of
            continue
        rebuilt = utilsIO.unstackComponents(stacked)

        assert list(rebuilt) == list(components), model
        for component, original in components.items():
            assert set(rebuilt[component].data_vars) == set(original.data_vars), (
                f"{model}/{component}"
            )
            for name, variable in original.data_vars.items():
                result = rebuilt[component][name]
                assert list(result.dims) == list(variable.dims), (
                    f"{model}/{component}/{name}"
                )
                # object has no Zarr dtype, so the stack casts it. Compare the
                # cast of the original, not the original.
                expected = (
                    utilsIO._castObjectArray(variable)
                    if variable.dtype == object
                    else variable
                )
                np.testing.assert_array_equal(
                    np.asarray(result.values, dtype=expected.dtype),
                    expected.values,
                    err_msg=f"{model}/{component}/{name}",
                )


def test_unstack_undoes_stack(minimal_test_esM):
    _assertUnstackUndoesStack(minimal_test_esM)


def test_unstack_undoes_stack_single_node(single_node_test_esM):
    _assertUnstackUndoesStack(single_node_test_esM)


def test_stackComponents_of_nothing():
    """A model class with no component stacks to nothing, not to an error."""
    assert utilsIO.stackComponents({}, prefixed=True) is None


def test_a_time_only_parameter_keeps_its_shape():
    """A time-only parameter next to a time-space parameter of the same name.

    The old format recorded both shapes as "ts_" and then called dropna on every
    dimension, so the time-only one was widened to time x space and never came
    back.
    """
    timeOnly = xr.DataArray(
        [1.0, 2.0], dims=["time"], coords={"time": [0, 1]}, name="ts_operationRateMax"
    )
    timeSpace = xr.DataArray(
        [[1.0, 3.0], [2.0, 4.0]],
        dims=["time", "space"],
        coords={"time": [0, 1], "space": ["R1", "R2"]},
        name="ts_operationRateMax",
    )
    components = {
        "A": xr.Dataset({"ts_operationRateMax": timeOnly}),
        "B": xr.Dataset({"ts_operationRateMax": timeSpace}),
    }

    rebuilt = utilsIO.unstackComponents(
        utilsIO.stackComponents(components, prefixed=True)
    )

    assert rebuilt["A"]["ts_operationRateMax"].dims == ("time",)
    np.testing.assert_array_equal(
        rebuilt["A"]["ts_operationRateMax"].values, [1.0, 2.0]
    )
    assert rebuilt["B"]["ts_operationRateMax"].dims == ("time", "space")


def test_an_absent_parameter_stays_absent():
    """The presence mask is what tells an absent parameter from a NaN one."""
    components = {
        "A": xr.Dataset({"0d_investPerCapacity": xr.DataArray(1.0)}),
        "B": xr.Dataset(
            {
                "0d_investPerCapacity": xr.DataArray(2.0),
                "0d_opexPerCapacity": xr.DataArray(3.0),
            }
        ),
    }

    rebuilt = utilsIO.unstackComponents(
        utilsIO.stackComponents(components, prefixed=True)
    )

    assert set(rebuilt["A"].data_vars) == {"0d_investPerCapacity"}
    assert set(rebuilt["B"].data_vars) == {
        "0d_investPerCapacity",
        "0d_opexPerCapacity",
    }


def test_the_group_attribute_decides_the_names():
    """A group written from plain names comes back with plain names."""
    components = {"A": xr.Dataset({"capacity": xr.DataArray(1.0)})}

    stacked = utilsIO.stackComponents(components, prefixed=False)
    assert stacked.attrs[utilsIO.PREFIXED_ATTRIBUTE] is False
    assert set(utilsIO.unstackComponents(stacked)["A"].data_vars) == {"capacity"}


def test_per_component_variable_attributes_survive_the_stack():
    """The unit of a result variable differs per component, so concat would lose it."""
    components = {
        "A": xr.Dataset(
            {"TAC": xr.DataArray(1.0, attrs={"TAC": "[Euro/a]"})},
        ),
        "B": xr.Dataset(
            {"TAC": xr.DataArray(2.0, attrs={"TAC": "[1e3 Euro/a]"})},
        ),
    }

    rebuilt = utilsIO.unstackComponents(
        utilsIO.stackComponents(components, prefixed=False)
    )

    assert rebuilt["A"]["TAC"].attrs == {"TAC": "[Euro/a]"}
    assert rebuilt["B"]["TAC"].attrs == {"TAC": "[1e3 Euro/a]"}


def test_a_scalar_coordinate_is_expanded_and_squeezed_back():
    """squeeze() has to be undone before the concat and redone after the unstack."""
    squeezed = (
        pd.Series([1.5], index=pd.Index(["R1"], name="space")).to_xarray().squeeze()
    )
    components = {"A": xr.Dataset({"1d_interestRate": squeezed})}

    stacked = utilsIO.stackComponents(components, prefixed=True)
    assert "space" in stacked.dims

    rebuilt = utilsIO.unstackComponents(stacked)["A"]["1d_interestRate"]
    assert rebuilt.dims == ()
    assert rebuilt.coords["space"].item() == "R1"
    assert rebuilt.item() == 1.5
