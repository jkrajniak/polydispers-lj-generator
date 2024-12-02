import numpy as np
import pytest

from polydispers.stats import sz_distribution_inverse_transform


def test_sz_distribution_mean():
    """Test if generated distribution has correct mean"""
    mn = 50
    pdi = 1.2
    size = 10000
    tolerance = 0.1  # 10% tolerance due to random sampling

    samples = sz_distribution_inverse_transform(mn, pdi, size)
    actual_mean = np.mean(samples)

    assert abs(actual_mean - mn) / mn < tolerance


def test_sz_distribution_pdi():
    """Test if generated distribution has correct PDI"""
    mn = 50
    pdi = 1.2
    size = 10000
    tolerance = 0.1  # 10% tolerance due to random sampling

    samples = sz_distribution_inverse_transform(mn, pdi, size)

    # Calculate actual PDI
    mw = np.mean(samples * samples) / np.mean(samples)  # Weight-average molecular weight
    actual_pdi = mw / np.mean(samples)

    assert abs(actual_pdi - pdi) / pdi < tolerance


def test_invalid_parameters():
    """Test if function handles invalid parameters correctly"""
    with pytest.raises(ValueError):
        sz_distribution_inverse_transform(mn=-50, pdi=1.2, size=1000)  # Invalid Mn

    with pytest.raises(ValueError):
        sz_distribution_inverse_transform(mn=50, pdi=0.8, size=1000)  # Invalid PDI (must be >= 1)

    with pytest.raises(ValueError):
        sz_distribution_inverse_transform(mn=50, pdi=1.2, size=0)  # Invalid size
