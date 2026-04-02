import copy
import pytest
import pandas as pd
import fine as fn
import fine.IOManagement.xarrayIO as xrIO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ip_dim_esM(perfectForesight_test_esM):
    """esM with ip-dependent and ip-independent 0d and 1d input parameters."""
    esM = copy.deepcopy(perfectForesight_test_esM)

    # ip-dependent 0d: investPerCapacity varies per investment period
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity={2020: 600, 2025: 550, 2030: 500, 2035: 450, 2040: 400},
            opexPerCapacity=10,  # ip-independent 0d (same component, mixed case)
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # ip-dependent 1d: capacityMax is a pd.Series per location, different per ip
    esM.add(
        fn.Source(
            esM=esM,
            name="WindOnshore",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax={
                2020: pd.Series({"PerfectLand": 10.0, "ForesightLand": 8.0}),
                2025: pd.Series({"PerfectLand": 12.0, "ForesightLand": 10.0}),
                2030: pd.Series({"PerfectLand": 15.0, "ForesightLand": 12.0}),
                2035: pd.Series({"PerfectLand": 18.0, "ForesightLand": 15.0}),
                2040: pd.Series({"PerfectLand": 20.0, "ForesightLand": 18.0}),
            },
            investPerCapacity=1200,
            opexPerCapacity=24,
            interestRate=0.05,
            economicLifetime=20,
        )
    )

    # ip-independent 1d: capacityMax is a single pd.Series, constant across all ips
    esM.add(
        fn.Source(
            esM=esM,
            name="SolarConstant",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=pd.Series({"PerfectLand": 100.0, "ForesightLand": 80.0}),
            investPerCapacity=800,
            interestRate=0.05,
            economicLifetime=25,
        )
    )

    # Hydrogen sink to make the model feasible
    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                index=[0, 1],
                columns=["PerfectLand", "ForesightLand"],
                data=[[500, 500], [500, 500]],
            ),
        )
    )

    return esM


@pytest.fixture
def ip_dim_datasets(ip_dim_esM):
    """Xarray datasets written from ip_dim_esM."""
    return xrIO.writeEnergySystemModelToDatasets(ip_dim_esM)


@pytest.fixture
def ip_dim_esM_rt(ip_dim_datasets):
    """ESM reconstructed from ip_dim_datasets (write → read)."""
    return xrIO.convertDatasetsToEnergySystemModel(ip_dim_datasets)


# ---------------------------------------------------------------------------
# Write-side tests — 0d variables
# ---------------------------------------------------------------------------


def test_0d_ip_dep_written_with_ip_dim(ip_dim_datasets):
    """ip-dependent 0d variable is stored as a single DataArray with an ip dimension."""
    electrolyzer_ds = ip_dim_datasets["Input"]["Conversion"]["Electrolyzer"]
    assert "0d_investPerCapacity" in electrolyzer_ds.data_vars, (
        f"Available vars: {list(electrolyzer_ds.data_vars)}"
    )
    assert "ip" in electrolyzer_ds["0d_investPerCapacity"].dims


def test_0d_ip_dep_no_per_ip_split_vars(ip_dim_datasets):
    """No legacy per-ip split variables (e.g. '0d_investPerCapacity.0') must exist."""
    electrolyzer_ds = ip_dim_datasets["Input"]["Conversion"]["Electrolyzer"]
    split_vars = [v for v in electrolyzer_ds.data_vars if "investPerCapacity." in v]
    assert split_vars == [], f"Unexpected per-ip split variables: {split_vars}"


def test_0d_ip_dep_values_correct(ip_dim_datasets):
    """Each ip slice of the 0d DataArray matches the input value."""
    da = ip_dim_datasets["Input"]["Conversion"]["Electrolyzer"]["0d_investPerCapacity"]
    expected = {"2020": 600.0, "2025": 550.0, "2030": 500.0, "2035": 450.0, "2040": 400.0}
    for ip_str, val in expected.items():
        assert float(da.sel(ip=ip_str).values) == pytest.approx(val), (
            f"ip={ip_str}: expected {val}, got {float(da.sel(ip=ip_str).values)}"
        )


def test_0d_ip_independent_has_no_ip_dim(ip_dim_datasets):
    """ip-independent scalar on the same component must not gain an ip dimension."""
    electrolyzer_ds = ip_dim_datasets["Input"]["Conversion"]["Electrolyzer"]
    assert "0d_opexPerCapacity" in electrolyzer_ds.data_vars
    assert "ip" not in electrolyzer_ds["0d_opexPerCapacity"].dims


# ---------------------------------------------------------------------------
# Write-side tests — 1d variables
# ---------------------------------------------------------------------------


def test_1d_ip_dep_written_with_ip_dim(ip_dim_datasets):
    """ip-dependent 1d variable is stored as a single DataArray with space and ip dims."""
    wind_ds = ip_dim_datasets["Input"]["Source"]["WindOnshore"]
    assert "1d_capacityMax" in wind_ds.data_vars, (
        f"Available vars: {list(wind_ds.data_vars)}"
    )
    da = wind_ds["1d_capacityMax"]
    assert "ip" in da.dims
    assert "space" in da.dims


def test_1d_ip_dep_no_per_ip_split_vars(ip_dim_datasets):
    """No legacy per-ip split variables (e.g. '1d_capacityMax.0') must exist."""
    wind_ds = ip_dim_datasets["Input"]["Source"]["WindOnshore"]
    split_vars = [v for v in wind_ds.data_vars if "capacityMax." in v]
    assert split_vars == [], f"Unexpected per-ip split variables: {split_vars}"


def test_1d_ip_dep_values_correct(ip_dim_datasets):
    """Spot-check values in the 1d ip-dimensioned DataArray."""
    da = ip_dim_datasets["Input"]["Source"]["WindOnshore"]["1d_capacityMax"]
    assert float(da.sel(ip="2020", space="PerfectLand").values) == 10.0
    assert float(da.sel(ip="2020", space="ForesightLand").values) == 8.0
    assert float(da.sel(ip="2040", space="PerfectLand").values) == 20.0
    assert float(da.sel(ip="2040", space="ForesightLand").values) == 18.0


def test_1d_ip_independent_has_no_ip_dim(ip_dim_datasets):
    """ip-independent 1d variable (constant Series) must not gain an ip dimension."""
    solar_ds = ip_dim_datasets["Input"]["Source"]["SolarConstant"]
    assert "1d_capacityMax" in solar_ds.data_vars, (
        f"Available vars: {list(solar_ds.data_vars)}"
    )
    da = solar_ds["1d_capacityMax"]
    assert "ip" not in da.dims
    assert "space" in da.dims


# ---------------------------------------------------------------------------
# Round-trip tests (write → read → write)
#
# Strategy: write ip_dim_esM to datasets, read back to esM_rt, write esM_rt
# again to datasets_rt, then assert structure and values are preserved.
# This exercises both add0dVariableToDict and add1dVariableToDict read-back.
# ---------------------------------------------------------------------------


def test_0d_ip_dep_roundtrip(ip_dim_esM_rt):
    """After read-back, 0d ip-dep variable is still ip-dimensioned with correct values."""
    datasets_rt = xrIO.writeEnergySystemModelToDatasets(ip_dim_esM_rt)
    da = datasets_rt["Input"]["Conversion"]["Electrolyzer"]["0d_investPerCapacity"]
    assert "ip" in da.dims
    expected = {"2020": 600.0, "2025": 550.0, "2030": 500.0, "2035": 450.0, "2040": 400.0}
    for ip_str, val in expected.items():
        assert float(da.sel(ip=ip_str).values) == val, (
            f"Roundtrip ip={ip_str}: expected {val}, got {float(da.sel(ip=ip_str).values)}"
        )


def test_0d_ip_independent_roundtrip(ip_dim_esM_rt):
    """After read-back, ip-independent scalar survives without an ip dim and correct value."""
    datasets_rt = xrIO.writeEnergySystemModelToDatasets(ip_dim_esM_rt)
    electrolyzer_ds = datasets_rt["Input"]["Conversion"]["Electrolyzer"]
    assert "0d_opexPerCapacity" in electrolyzer_ds.data_vars
    da = electrolyzer_ds["0d_opexPerCapacity"]
    assert "ip" not in da.dims
    assert float(da.values) == 10.0


def test_1d_ip_dep_roundtrip(ip_dim_esM_rt):
    """After read-back, 1d ip-dep variable is still ip+space-dimensioned with correct values."""
    datasets_rt = xrIO.writeEnergySystemModelToDatasets(ip_dim_esM_rt)
    da = datasets_rt["Input"]["Source"]["WindOnshore"]["1d_capacityMax"]
    assert "ip" in da.dims
    assert "space" in da.dims
    assert float(da.sel(ip="2020", space="PerfectLand").values) == 10.0
    assert float(da.sel(ip="2040", space="ForesightLand").values) == 18.0


def test_1d_ip_independent_roundtrip(ip_dim_esM_rt):
    """After read-back, ip-independent 1d variable survives without ip dim and correct values."""
    datasets_rt = xrIO.writeEnergySystemModelToDatasets(ip_dim_esM_rt)
    solar_ds = datasets_rt["Input"]["Source"]["SolarConstant"]
    assert "1d_capacityMax" in solar_ds.data_vars, (
        f"Available vars: {list(solar_ds.data_vars)}"
    )
    da = solar_ds["1d_capacityMax"]
    assert "ip" not in da.dims
    assert float(da.sel(space="PerfectLand").values) == 100.0
    assert float(da.sel(space="ForesightLand").values) == 80.0

