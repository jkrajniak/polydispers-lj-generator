import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests"""
    import numpy as np

    np.random.seed(42)
