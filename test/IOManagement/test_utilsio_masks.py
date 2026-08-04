"""Unit tests for the dimension mask layer that the Zarr format is built on.

These functions are pure, so they can be checked without writing anything.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fine.IOManagement import utilsIO


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, utilsIO.DIMENSION_SCALAR),
        (1.5, utilsIO.DIMENSION_SCALAR),
        ("a name", utilsIO.DIMENSION_SCALAR),
        (True, utilsIO.DIMENSION_SCALAR),
        (["ID1", "ID2"], utilsIO.DIMENSION_SCALAR),
        (
            pd.Series([1.0, 2.0], index=pd.Index(["R1", "R2"], name="space")),
            utilsIO.DIMENSION_SPACE,
        ),
        (pd.Series([1.0], index=pd.Index([0], name="time")), utilsIO.DIMENSION_TIME),
    ],
)
def test_inferParameterDimension(value, expected):
    """Every parameter shape maps to its own code."""
    assert utilsIO.inferParameterDimension(value) == expected


def test_inferParameterDimension_multi_index():
    """A parameter with named index levels is read from those names."""
    spaceSpace = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([("R1", "R2")], names=["space", "space_2"]),
    )
    timeSpace = pd.Series(
        [1.0], index=pd.MultiIndex.from_tuples([(0, "R1")], names=["time", "space"])
    )
    timeSpaceSpace = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples(
            [(0, "R1", "R2")], names=["time", "space", "space_2"]
        ),
    )
    unknown = pd.Series([1.0], index=pd.Index(["x"], name="something"))

    assert utilsIO.inferParameterDimension(spaceSpace) == (
        utilsIO.DIMENSION_SPACE_SPACE2
    )
    assert utilsIO.inferParameterDimension(timeSpace) == utilsIO.DIMENSION_TIME_SPACE
    assert utilsIO.inferParameterDimension(timeSpaceSpace) == (
        utilsIO.DIMENSION_TIME_SPACE_SPACE2
    )
    assert utilsIO.inferParameterDimension(unknown) == utilsIO.DIMENSION_UNKNOWN


def test_createWasNoneMask_marks_only_the_none_parameters():
    """The mask is what lets a None come back as None instead of as NaN."""
    component_dict = {
        "Source": {"PV": {"name": "PV", "capacityMax": None, "investPerCapacity": 1.0}}
    }
    mask = utilsIO.createWasNoneMask(component_dict)
    assert mask["Source"]["PV"] == {
        "name": False,
        "capacityMax": True,
        "investPerCapacity": False,
    }


def test_replaceNoneValuesForXarray_does_not_change_its_input():
    """The caller's component dict has to survive the replacement unchanged."""
    component_dict = {"Source": {"PV": {"capacityMax": None}}}
    replaced = utilsIO.replaceNoneValuesForXarray(component_dict)

    assert component_dict["Source"]["PV"]["capacityMax"] is None
    assert np.isnan(replaced["Source"]["PV"]["capacityMax"])


def test_parameter_masks_survive_a_round_trip_through_xarray():
    """What addParameterMasksToXarray writes, extractParameterMasksFromXarray reads."""
    dimensions = {"Source": {"PV": {"name": 0, "capacityMax": 2}}}
    wasNone = {"Source": {"PV": {"name": False, "capacityMax": True}}}
    xr_dss = {"Source": {"PV": xr.Dataset()}}

    utilsIO.addParameterMasksToXarray(xr_dss, dimensions, wasNone)
    readDimensions, readWasNone = utilsIO.extractParameterMasksFromXarray(
        xr_dss["Source"]["PV"]
    )

    assert readDimensions == dimensions["Source"]["PV"]
    assert readWasNone == wasNone["Source"]["PV"]


def test_extractParameterMasksFromXarray_without_a_mask():
    """A dataset that carries no mask reads as no parameters, not as an error."""
    assert utilsIO.extractParameterMasksFromXarray(xr.Dataset()) == ({}, {})
