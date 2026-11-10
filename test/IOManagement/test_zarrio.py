"""Round trip of writeDatasetsToZarr / readZarrToDatasets under zarr 2 and zarr 3."""

import json
import os
from copy import deepcopy

import pytest
import xarray as xr
import zarr

import fine.IOManagement.xarrayIO as xrIO

ZARR_MAJOR = int(zarr.__version__.split(".")[0])


def _write_and_capture(datasets, path, monkeypatch, **kwargs):
    """Run writeDatasetsToZarr and return {store path: dataset handed to zarr}."""
    written = {}
    original_to_zarr = xr.Dataset.to_zarr

    def recording_to_zarr(self, store, *args, **kw):
        written[str(store)] = self
        return original_to_zarr(self, store, *args, **kw)

    monkeypatch.setattr(xr.Dataset, "to_zarr", recording_to_zarr)
    xrIO.writeDatasetsToZarr(datasets, output_zarr_path=str(path), **kwargs)
    monkeypatch.setattr(xr.Dataset, "to_zarr", original_to_zarr)
    return written


def _read_back(read, store, root):
    parts = store[len(str(root)) + 1:].split("/")
    node = read
    for p in parts:
        node = node[p]
    return node


def _store_format(store):
    """Zarr format of the group at ``store`` (format 3 has zarr.json, 2 has .zgroup)."""
    meta = os.path.join(store, "zarr.json")
    if not os.path.exists(meta):
        meta = os.path.join(store, ".zgroup")
    with open(meta) as f:
        return json.load(f)["zarr_format"]


def _array_compressor(store, var):
    """(codec name, cname, clevel) of the first compressor of ``var``."""
    meta_v3 = os.path.join(store, var, "zarr.json")
    if os.path.exists(meta_v3):
        with open(meta_v3) as f:
            codecs = json.load(f)["codecs"]
        blosc = next(c for c in codecs if c["name"] == "blosc")
        return "blosc", blosc["configuration"]["cname"], blosc["configuration"]["clevel"]
    with open(os.path.join(store, var, ".zarray")) as f:
        comp = json.load(f)["compressor"]
    return comp["id"], comp["cname"], comp["clevel"]


@pytest.fixture
def optimized_zarr_datasets(minimal_test_esM):
    esM = deepcopy(minimal_test_esM)
    esM.optimize()
    return xrIO.writeEnergySystemModelToDatasets(esM, zarrFormat=True)


@pytest.mark.parametrize("replace_fill_value", [False, True])
def test_zarr_round_trip(optimized_zarr_datasets, tmp_path, monkeypatch, replace_fill_value):
    path = tmp_path / "esm.zarr"
    written = _write_and_capture(
        optimized_zarr_datasets, path, monkeypatch, replace_fill_value=replace_fill_value
    )
    assert any("/Results/" in s for s in written)
    assert any("/Input/" in s for s in written)

    read = xrIO.readZarrToDatasets(path, lazy_load=False)
    for store, ds in written.items():
        back = _read_back(read, store, path)
        xr.testing.assert_equal(back, ds.load())


def test_zarr_default_format_and_compressor(optimized_zarr_datasets, tmp_path, monkeypatch):
    """Default: zarr format 2 with Blosc zstd level 5 under zarr 2 and zarr 3."""
    path = tmp_path / "esm.zarr"
    written = _write_and_capture(optimized_zarr_datasets, path, monkeypatch)
    for store, ds in written.items():
        assert _store_format(store) == 2
    store, ds = next((s, d) for s, d in written.items() if "/Results/" in s and d.data_vars)
    var = next(iter(ds.data_vars))
    assert _array_compressor(store, var) == ("blosc", "zstd", 5)


def test_zarr_writes_every_chunk(optimized_zarr_datasets, tmp_path):
    """zarr 3 skips chunks equal to the fill value unless told otherwise; for
    integer/boolean variables (fill value null) zarr 2 then reads garbage."""
    import math

    path = tmp_path / "esm.zarr"
    xrIO.writeDatasetsToZarr(optimized_zarr_datasets, output_zarr_path=str(path))
    n_arrays = 0
    for root, _dirs, files in os.walk(path):
        if ".zarray" not in files:
            continue
        with open(os.path.join(root, ".zarray")) as f:
            meta = json.load(f)
        expected = math.prod(
            math.ceil(s / c) for s, c in zip(meta["shape"], meta["chunks"])
        ) if meta["shape"] else 1
        chunks = [f for f in files if not f.startswith(".")]
        assert len(chunks) == expected, root
        n_arrays += 1
    assert n_arrays > 0


@pytest.mark.skipif(ZARR_MAJOR < 3, reason="zarr format 3 needs zarr >= 3")
def test_zarr_format_3_round_trip(optimized_zarr_datasets, tmp_path, monkeypatch):
    path = tmp_path / "esm.zarr"
    written = _write_and_capture(optimized_zarr_datasets, path, monkeypatch, zarr_format=3)
    read = xrIO.readZarrToDatasets(path, lazy_load=False)
    for store, ds in written.items():
        assert _store_format(store) == 3
        xr.testing.assert_equal(_read_back(read, store, path), ds.load())
    store, ds = next((s, d) for s, d in written.items() if "/Results/" in s and d.data_vars)
    assert _array_compressor(store, next(iter(ds.data_vars))) == ("blosc", "zstd", 5)


@pytest.mark.skipif(ZARR_MAJOR >= 3, reason="only zarr 2 rejects zarr format 3")
def test_zarr_format_3_needs_zarr_3(optimized_zarr_datasets, tmp_path):
    with pytest.raises(ValueError, match="zarr format 3"):
        xrIO.writeDatasetsToZarr(
            optimized_zarr_datasets, output_zarr_path=str(tmp_path / "esm.zarr"), zarr_format=3
        )
