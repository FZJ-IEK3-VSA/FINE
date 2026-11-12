import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fine.IOManagement.utilsIO import lazySeriesToXarray


def _sparse_series(levels, names, share, seed=0, dtype="float64", shuffle=True):
    rng = np.random.default_rng(seed)
    full = pd.MultiIndex.from_product(levels, names=names)
    keep = rng.random(len(full)) < share
    index = full[keep]
    series = pd.Series(rng.normal(size=len(index)).astype(dtype), index=index)
    if shuffle:
        series = series.sample(frac=1, random_state=seed)
    return series


def _check(series, **kwargs):
    expected = xr.DataArray.from_series(series)
    result = lazySeriesToXarray(series, **kwargs)
    xr.testing.assert_identical(result.compute(), expected)
    assert result.dtype == expected.dtype
    for dim in expected.dims:
        assert result.indexes[dim].equals(expected.indexes[dim])
        assert result.indexes[dim].dtype == expected.indexes[dim].dtype
    return result


LOCS = ["DE1", "AT1", "FR1", "PL1", "UKN"]


@pytest.mark.parametrize("block_bytes", [8, 200, 128 * 2**20])
def test_time_space_space2(block_bytes):
    series = _sparse_series(
        [range(30), LOCS, LOCS], ["time", "space", "space_2"], 0.2
    )
    result = _check(series, block_bytes=block_bytes)
    assert isinstance(result.data, da.Array)


def test_time_not_first_and_leading_dim():
    # (component, time, space, space_2) as in the NetCDF Input
    series = _sparse_series(
        [["Transmission; a", "Transmission; b"], range(7), LOCS, LOCS],
        ["component", "time", "space", "space_2"],
        0.3,
    )
    _check(series, block_bytes=100)
    # block dim missing: cut along the first level
    _check(series.rename_axis(["c", "t", "s", "s2"]), block_bytes=100)


def test_unused_levels_and_names():
    series = _sparse_series([range(5), LOCS, LOCS], ["time", "space", "space_2"], 0.4)
    # a level value without data: from_series drops it
    series = series[series.index.get_level_values("space") != "PL1"]
    assert "PL1" in series.index.levels[1]
    series.name = "operationVariablesOptimum"
    _check(series)
    series.index = series.index.set_names([None, "space", None])
    _check(series)


def test_single_time_step_and_full_grid():
    _check(_sparse_series([[0], LOCS, LOCS], ["time", "space", "space_2"], 0.5))
    # no missing values: dtype kept
    _check(
        _sparse_series(
            [range(3), LOCS, LOCS], ["time", "space", "space_2"], 1.1, dtype="float32"
        )
    )
    _check(
        _sparse_series(
            [range(3), LOCS, LOCS], ["time", "space", "space_2"], 0.5, dtype="float32"
        )
    )


def test_integers():
    ints = _sparse_series([range(4), LOCS, LOCS], ["time", "space", "space_2"], 0.5)
    ints = (ints * 1000).astype("int64")
    result = _check(ints, block_bytes=100)
    assert isinstance(result.data, da.Array) and result.dtype == np.float64
    full = _sparse_series([range(4), LOCS], ["time", "space"], 1.1)
    result = _check((full * 1000).astype("int64"))
    assert result.dtype == np.int64


def test_fallback():
    bools = _sparse_series([range(4), LOCS], ["time", "space"], 0.5) > 0
    result = _check(bools)
    assert not isinstance(result.data, da.Array)
    flat = pd.Series([1.0, 2.0], index=pd.Index(["a", "b"], name="space"))
    _check(flat)
    empty = _sparse_series([range(4), LOCS], ["time", "space"], 0.0)
    _check(empty)
