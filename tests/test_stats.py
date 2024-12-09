import numpy as np
import pytest

from polydispers.input_config import InputConfig
from polydispers.stats import sz_distribution_inverse_transform


@pytest.fixture
def basic_config():
    config = InputConfig(
        num_chains=100, mn=1000, pdi=2.0, box_size=50.0, output_dir="test_output", seed=42, polymer="test_polymer"
    )
    return config


@pytest.fixture
def repeat_unit_config():
    config = InputConfig(
        num_chains=100, mn=1000, pdi=2.0, box_size=50.0, output_dir="test_output", seed=42, polymer="test_polymer"
    )
    config.repeat_unit_topology = "AB"
    config.bead_masses = {"A": 10, "B": 20}
    return config


def test_basic_distribution(basic_config):
    size = 1000
    weights, units = sz_distribution_inverse_transform(basic_config, size=size)

    # Check if mean is close to Mn (within 5%)
    assert abs(np.mean(weights) - basic_config.mn) / basic_config.mn < 0.05

    # Check if PDI is close to target (within 5%)
    calculated_pdi = np.mean(weights * weights) / (np.mean(weights) * np.mean(weights))
    assert abs(calculated_pdi - basic_config.pdi) / basic_config.pdi < 0.05


def test_repeat_unit_distribution(repeat_unit_config):
    size = 1000
    weights, units = sz_distribution_inverse_transform(repeat_unit_config, size=size)
    repeat_unit_mass = 30  # A(10) + B(20)

    # Check if all weights are multiples of repeat unit mass
    assert all(weight % repeat_unit_mass == 0 for weight in weights)

    # Check if mean is close to Mn (within 5%)
    assert abs(np.mean(weights) - repeat_unit_config.mn) / repeat_unit_config.mn < 0.05


def test_monodisperse_system():
    config = InputConfig(
        num_chains=100,
        mn=1000,
        pdi=1.0001,  # Using exactly 1.0 might cause numerical issues
        box_size=50.0,
        output_dir="test_output",
        seed=42,
        polymer="test_polymer",
    )

    size = 100
    weights, units = sz_distribution_inverse_transform(config, size=size)

    # In a monodisperse system, all weights should be very close to Mn
    assert np.allclose(weights, config.mn, rtol=0.01)
    assert len(np.unique(units)) == 1  # All chains should have same number of units


def test_repeat_unit_monodisperse():
    config = InputConfig(
        num_chains=100,
        mn=900,  # Chosen to be divisible by repeat unit mass
        pdi=1.0001,
        box_size=50.0,
        output_dir="test_output",
        seed=42,
        polymer="test_polymer",
    )
    config.repeat_unit_topology = "AAB"
    config.bead_masses = {"A": 10, "B": 20}

    size = 100
    weights, units = sz_distribution_inverse_transform(config, size=size)
    repeat_unit_mass = 40  # 2*A(10) + B(20)

    # Check if all weights are the same and are multiples of repeat unit mass
    assert all(weight % repeat_unit_mass == 0 for weight in weights)
    assert len(np.unique(weights)) == 1
    assert len(np.unique(units)) == 1


def test_invalid_pdi():
    config = InputConfig(
        num_chains=100,
        mn=1000,
        pdi=0.5,  # PDI must be >= 1
        box_size=50.0,
        output_dir="test_output",
        seed=42,
        polymer="test_polymer",
    )

    with pytest.raises(ValueError):
        sz_distribution_inverse_transform(config)


def test_zero_size():
    config = InputConfig(
        num_chains=100, mn=1000, pdi=2.0, box_size=50.0, output_dir="test_output", seed=42, polymer="test_polymer"
    )

    weights, units = sz_distribution_inverse_transform(config, size=0)
    assert len(weights) == 0
    assert len(units) == 0
