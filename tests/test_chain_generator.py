import numpy as np
import pytest

from polydispers.chain_generator import generate_kremer_grest_chain
from polydispers.input_config import InputConfig, PolymerConfig


def create_test_config(
    num_repeat_units: int, bond_length: float = 0.85, bead_radius: float = 1.0, box_size: float = 50.0
):
    """Helper function to create test configuration"""
    return InputConfig(
        num_chains=1,
        mn=1000,
        pdi=1.0,
        box_size=box_size,
        output_dir="test_output",
        seed=42,
        polymer=PolymerConfig(
            bond_length=bond_length,
            bead_radius=bead_radius,
            repeat_unit_topology="A",
            bead_types={"A": {"mass": 1.0, "type_id": 1}},
        ),
    )


def test_chain_length():
    """Test if generated chain has correct number of beads"""
    num_repeat_units = 10
    config = create_test_config(num_repeat_units)

    coordinates = generate_kremer_grest_chain(config, num_repeat_units)

    # Since repeat_unit_topology is "A", each repeat unit is one bead
    expected_length = num_repeat_units * len(config.polymer.repeat_unit_topology)
    assert len(coordinates) == expected_length
    assert coordinates.shape == (expected_length, 3)


def test_bond_lengths():
    """Test if bonds are within expected length"""
    num_repeat_units = 20
    bond_length = 0.85
    config = create_test_config(num_repeat_units, bond_length=bond_length)
    tolerance = 1e-6

    coordinates = generate_kremer_grest_chain(config, num_repeat_units)

    # Check consecutive bond lengths
    total_beads = num_repeat_units * len(config.polymer.repeat_unit_topology)
    for i in range(total_beads - 1):
        delta = coordinates[i + 1] - coordinates[i]
        actual_length = np.linalg.norm(delta)
        assert abs(actual_length - bond_length) < tolerance


def test_bead_overlap():
    """Test if beads don't overlap"""
    num_repeat_units = 15
    bead_radius = 1.0
    config = create_test_config(num_repeat_units, bead_radius=bead_radius)

    coordinates = generate_kremer_grest_chain(config, num_repeat_units)

    # Check distances between non-bonded beads
    total_beads = num_repeat_units * len(config.polymer.repeat_unit_topology)
    for i in range(total_beads):
        for j in range(i + 2, total_beads):  # Skip consecutive beads
            delta = coordinates[j] - coordinates[i]
            distance = np.linalg.norm(delta)
            assert distance >= 2 * bead_radius  # Beads shouldn't overlap


def test_periodic_boundary_conditions():
    """Test if coordinates respect periodic boundary conditions"""
    num_repeat_units = 10
    box_size = 5.0  # Small box to force PBC
    config = create_test_config(num_repeat_units, box_size=box_size)

    coordinates = generate_kremer_grest_chain(config, num_repeat_units)

    # All coordinates should be within box bounds
    assert np.all(coordinates >= 0)
    assert np.all(coordinates < box_size)


def test_invalid_inputs():
    """Test if function handles invalid inputs correctly"""
    with pytest.raises(ValueError):
        config = create_test_config(0)  # Invalid chain length
        generate_kremer_grest_chain(config, 0)

    with pytest.raises(ValueError):
        config = create_test_config(10, bond_length=-1.0)  # Invalid bond length
        generate_kremer_grest_chain(config, 10)

    with pytest.raises(ValueError):
        config = create_test_config(10, bead_radius=-1.0)  # Invalid bead radius
        generate_kremer_grest_chain(config, 10)
