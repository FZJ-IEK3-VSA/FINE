import pytest


def test_matplot():
    """Test if matplotlib works."""
    import matplotlib.pyplot as plt

    plt.plot([1, 2, 3, 4])


@pytest.mark.skip("not yet implemented")
def test_operation_transmission_plot():
    """Tests whether abstract transmission operation function works"""


@pytest.mark.skip("not yet implemented")
def test_operational_commodity_balance_plots():
    """Tests whether abstract ... function works"""
