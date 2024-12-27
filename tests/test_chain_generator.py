import numpy as np

from polydispers.chain_generator import generate_kremer_grest_chain
from polydispers.input_config import InputConfig


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
        polymer={
            "bond_length": bond_length,
            "bead_radius": bead_radius,
            "repeat_unit_topology": "A",
            "bead_types": {"A": {"mass": 1.0, "type_id": 1}},
        },
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
