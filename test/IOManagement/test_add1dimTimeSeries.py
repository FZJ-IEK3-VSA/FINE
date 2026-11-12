"""addDFVariablesToXarray converts the time series of 1dim components one
component at a time (_add1dimTimeSeriesToXarray). Compared here with the
previous algorithm (stack all components, concat, sort_index().to_xarray(),
sel/squeeze per component), kept below as the reference."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fine.IOManagement import utilsIO


def _reference(component_dict, df_iteration_dict):
    xr_ds = {
        cls: {comp: xr.Dataset() for comp in comps}
        for cls, comps in component_dict.items()
    }
    for variable, tuples in df_iteration_dict.items():
        df_dict = {}
        for cls, comp in tuples:
            s = component_dict[cls][comp][variable].stack()
            if "Period" in s.index.names:
                s = s.droplevel(0)
            s.index.set_names("time", level=0, inplace=True)
            s.index.set_names("space", level=1, inplace=True)
            df_dict[f"{cls}; {comp}"] = s
        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True)
        ds = xr.Dataset()
        ds[f"ts_{variable}"] = df_variable.sort_index().to_xarray()
        for key in df_variable.index.get_level_values(0).unique():
            cls, comp = key.split("; ")
            part = ds.sel(component=key).squeeze().reset_coords(
                names=["component"], drop=True
            )
            xr_ds[cls][comp] = xr.merge([xr_ds[cls][comp], part])
    return xr_ds


def _new(component_dict, df_iteration_dict):
    xr_ds = {
        cls: {comp: xr.Dataset() for comp in comps}
        for cls, comps in component_dict.items()
    }
    return utilsIO.addDFVariablesToXarray(
        xr_ds, component_dict, df_iteration_dict, {}, []
    )


def _assert_same(component_dict, df_iteration_dict):
    expected = _reference(component_dict, df_iteration_dict)
    result = _new(component_dict, df_iteration_dict)
    for cls in expected:
        for comp in expected[cls]:
            xr.testing.assert_identical(result[cls][comp], expected[cls][comp])
            assert list(result[cls][comp].variables) == list(
                expected[cls][comp].variables
            )
            for var in expected[cls][comp].variables:
                assert (
                    result[cls][comp][var].dtype == expected[cls][comp][var].dtype
                ), (cls, comp, var)


def _frame(rng, times, locs, nan_share=0.0, dtype="float64"):
    values = rng.normal(size=(len(times), len(locs)))
    df = pd.DataFrame(values, index=pd.Index(times), columns=list(locs))
    if dtype != "float64":
        df = (df * 100).round().astype(dtype)
    if nan_share:
        df = df.mask(rng.random(df.shape) < nan_share)
    return df


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_mixed_components(rng):
    T = range(24)
    cd = {
        "Source": {
            "pv": {"operationRateMax": _frame(rng, T, ["DE", "AT", "FR"])},
            "wind": {"operationRateMax": _frame(rng, T, ["PL", "DE"], 0.3)},
            # all NaN: no entry, not in the grid
            "empty": {"operationRateMax": _frame(rng, T, ["DE"], 1.0)},
        },
        "Sink": {
            "demand": {
                "operationRateFix": _frame(rng, T, ["DE", "AT"], dtype="int64"),
                "operationRateMax": _frame(rng, T, ["UK"], dtype="int64"),
            }
        },
    }
    dfi = {
        "operationRateMax": [
            ("Source", "pv"), ("Source", "wind"), ("Source", "empty"), ("Sink", "demand")
        ],
        "operationRateFix": [("Sink", "demand")],
    }
    _assert_same(cd, dfi)


def test_int_full_grid_keeps_int(rng):
    T = range(5)
    cd = {"Sink": {
        "a": {"x": _frame(rng, T, ["DE", "AT"], dtype="int64")},
        "b": {"x": _frame(rng, T, ["AT", "DE"], dtype="int64")},
    }}
    _assert_same(cd, {"x": [("Sink", "a"), ("Sink", "b")]})


def test_squeeze_single_location_and_time(rng):
    cd = {"Source": {
        "a": {"x": _frame(rng, [3], ["DE"])},
        "b": {"y": _frame(rng, range(4), ["DE"])},
        "c": {"y": _frame(rng, [0], ["DE", "FR"])},
    }}
    _assert_same(cd, {"x": [("Source", "a")], "y": [("Source", "b"), ("Source", "c")]})


def test_unsorted_times_and_locations(rng):
    cd = {"Source": {
        "a": {"x": _frame(rng, [5, 1, 3], ["FR", "AT", "DE"], 0.2)},
        "b": {"x": _frame(rng, [2, 1], ["UK", "AT"])},
    }}
    _assert_same(cd, {"x": [("Source", "a"), ("Source", "b")]})


def test_period_level(rng):
    df = _frame(rng, range(6), ["DE", "AT"])
    df.index = pd.MultiIndex.from_product([[0], range(6)], names=["Period", None])
    cd = {"Source": {"a": {"x": df}, "b": {"x": _frame(rng, range(6), ["DE"])}}}
    _assert_same(cd, {"x": [("Source", "a"), ("Source", "b")]})


def test_float32_with_float64(rng):
    cd = {"Source": {
        "a": {"x": _frame(rng, range(4), ["DE"]).astype("float32")},
        "b": {"x": _frame(rng, range(4), ["AT"])},
        "c": {"y": _frame(rng, range(4), ["AT"], 0.5).astype("float32")},
    }}
    _assert_same(cd, {"x": [("Source", "a"), ("Source", "b")], "y": [("Source", "c")]})


def test_object_values_fall_back(rng):
    df = _frame(rng, range(3), ["DE", "AT"]).astype(object)
    cd = {"Source": {"a": {"x": df}, "b": {"x": _frame(rng, range(3), ["DE"])}}}
    _assert_same(cd, {"x": [("Source", "a"), ("Source", "b")]})
