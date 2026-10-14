"""Data loader of the aggregation example.

The input data is shared between several examples. It is stored one time in
``examples/data/multiRegion``. See ``examples/exampleData.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from exampleData import MULTI_REGION_DATA_PATH, getMultiRegionData  # noqa: E402

#: Path to the input data of this example.
INPUT_DATA_PATH = MULTI_REGION_DATA_PATH


def getData(engine="openpyxl"):
    """Get example data for the aggregation example."""
    return getMultiRegionData(engine=engine)
