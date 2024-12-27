import numpy as np
import pytest

from polydispers.input_config import InputConfig
from polydispers.stats import sz_distribution_inverse_transform


def create_test_config(num_chains: int = 100, mn: float = 10000, pdi: float = 1.2, repeat_unit_topology: str = "AB"):
    """Helper function to create test configuration"""
    return InputConfig(
        num_chains=num_chains,
        mn=mn,
        pdi=pdi,
        box_size=100.0,
        output_dir="test_output",
        seed=42,
        polymer={
            "bond_length": 0.85,
            "bead_radius": 1.0,
            "repeat_unit_topology": repeat_unit_topology,
            "bead_types": {"A": {"mass": 1.0, "type_id": 1}, "B": {"mass": 1.0, "type_id": 2}},
        },
    )


def test_monodisperse_system():
    """Test generation of monodisperse system (PDI = 1.0)"""
    config = create_test_config(pdi=1.0)
    molecular_weights, chain_lengths = sz_distribution_inverse_transform(config)

    assert len(molecular_weights) == config.num_chains
    assert len(chain_lengths) == config.num_chains
    assert np.allclose(molecular_weights, [config.mn / config.num_chains] * config.num_chains)
    assert len(set(chain_lengths)) == 1  # All chains should have the same length


def test_repeat_unit_monodisperse():
    """Test if repeat unit topology is correctly handled in monodisperse case"""
    config = create_test_config(pdi=1.0, repeat_unit_topology="AABBAA")
    molecular_weights, chain_lengths = sz_distribution_inverse_transform(config)

    # Each repeat unit has 6 beads, each with mass 1.0
    repeat_unit_mass = sum(config.polymer.bead_types[bead].mass for bead in config.polymer.repeat_unit_topology)
    expected_chain_length = int(np.ceil(config.mn / (config.num_chains * repeat_unit_mass)))

    assert all(length == expected_chain_length for length in chain_lengths)


def test_invalid_pdi():
    """Test if function raises error for invalid PDI values"""
    with pytest.raises(ValueError):
        config = create_test_config(pdi=0.9)  # PDI must be >= 1.0
        sz_distribution_inverse_transform(config)


def test_zero_size():
    """Test if function handles zero system size correctly"""
    with pytest.raises(ValueError):
        config = create_test_config(num_chains=0)
        sz_distribution_inverse_transform(config)


def test_basic_distribution():
    """Test basic properties of the generated distribution"""
    config = create_test_config()
    molecular_weights, chain_lengths = sz_distribution_inverse_transform(config)

    # Check if we got the requested number of chains
    assert len(molecular_weights) == config.num_chains
    assert len(chain_lengths) == config.num_chains

    # Check if total molecular weight is close to target
    assert np.isclose(np.sum(molecular_weights), config.mn, rtol=0.1)

    # Check if PDI is close to target
    mw = np.mean(molecular_weights * molecular_weights) / np.mean(molecular_weights)
    mn = np.mean(molecular_weights)
    actual_pdi = mw / mn
    assert np.isclose(actual_pdi, config.pdi, rtol=0.2)  # Allow 20% tolerance due to finite sampling


def test_repeat_unit_distribution():
    """Test if repeat unit topology is correctly handled in polydisperse case"""
    config = create_test_config(repeat_unit_topology="AABBAA")
    molecular_weights, chain_lengths = sz_distribution_inverse_transform(config)

    # Each repeat unit has 6 beads, each with mass 1.0
    repeat_unit_mass = sum(config.polymer.bead_types[bead].mass for bead in config.polymer.repeat_unit_topology)

    # Check if molecular weights are multiples of repeat unit mass
    for mw in molecular_weights:
        assert np.isclose(mw % repeat_unit_mass, 0, atol=1e-10)
