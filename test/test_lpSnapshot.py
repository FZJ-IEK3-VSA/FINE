"""Compare the LP files of test fixtures with stored references.

A refactoring that must not change the optimization model keeps these files
identical. Set UPDATE_GOLDEN=1 to regenerate the references intentionally.
"""

import difflib
import gzip
import os
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

LP_GOLDEN_DIR = Path(__file__).parent / "data" / "golden" / "lp"


def _assert_lp_snapshot(esM, reference_name, tmp_path):
    """Compare the LP file of an esM with a stored reference.

    References ending in .gz are stored compressed.
    """
    esM.declareOptimizationProblem()
    actual_path = tmp_path / reference_name.removesuffix(".gz")
    esM.pyM.write(
        str(actual_path),
        io_options={"symbolic_solver_labels": True, "file_determinism": 20},
    )
    # Text mode with universal newlines: Windows writes CRLF, Linux LF.
    actual = actual_path.read_text()
    reference_path = LP_GOLDEN_DIR / reference_name
    compressed = reference_path.suffix == ".gz"

    if os.environ.get("UPDATE_GOLDEN", "").lower() in {"1", "true", "yes"}:
        LP_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        if compressed:
            # mtime=0 keeps the file identical when nothing changed.
            with gzip.GzipFile(reference_path, "wb", mtime=0) as f:
                f.write(actual.encode())
        else:
            reference_path.write_text(actual, newline="\n")
        return

    regenerate = (
        "Regenerate it intentionally with: UPDATE_GOLDEN=1 pytest "
        f"test/test_lpSnapshot.py -k {reference_name.split('.')[0]}"
    )
    assert reference_path.exists(), (
        f"Missing LP reference {reference_path}. {regenerate}"
    )
    if compressed:
        with gzip.open(reference_path, "rt") as f:
            expected = f.read()
    else:
        expected = reference_path.read_text()

    if actual != expected:
        diff = difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            "reference",
            "actual",
            lineterm="",
        )
        pytest.fail(
            f"LP file differs from {reference_path}.\n"
            f"Actual file: {actual_path}\n{regenerate}\n" + "\n".join(islice(diff, 40))
        )


@pytest.mark.parametrize(
    "fixture_name, reference_name",
    [
        ("minimal_test_esM", "minimal_test_esM.lp"),
        ("single_node_test_esM", "single_node_test_esM.lp"),
        ("dsm_test_esM", "dsm_test_esM.lp"),
        ("balanceLimitConstraint_test_esM", "balanceLimitConstraint_test_esM.lp.gz"),
        ("perfectForesight_test_esM", "perfectForesight_test_esM.lp"),
    ],
)
def test_lp_snapshot(fixture_name, reference_name, request, tmp_path):
    # balanceLimitConstraint_test_esM draws random input data.
    np.random.seed(0)
    esM = request.getfixturevalue(fixture_name)
    if isinstance(esM, tuple):
        esM = esM[0]
    _assert_lp_snapshot(esM, reference_name, tmp_path)
