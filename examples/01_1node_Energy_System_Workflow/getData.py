"""Data loader of the one node example.

The input data is shared between several examples. It is stored one time in
``examples/data/oneNode``. See ``examples/exampleData.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from exampleData import ONE_NODE_DATA_PATH, getOneNodeData  # noqa: E402

#: Path to the input data of this example.
INPUT_DATA_PATH = ONE_NODE_DATA_PATH


def getData(engine="openpyxl"):
    """Get example data for the one node example."""
    return getOneNodeData(engine=engine)
