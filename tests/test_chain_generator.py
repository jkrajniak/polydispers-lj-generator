import numpy as np
import pytest

from polydispers.chain_generator import generate_kremer_grest_chain


def test_chain_length():
    """Test if generated chain has correct number of beads"""
    chain_length = 10
    bond_length = 0.85
    bead_radius = 1.0
    box_size = 50.0

    coordinates = generate_kremer_grest_chain(
        chain_length=chain_length,
        bond_length=bond_length,
        bead_radius=bead_radius,
        box_size=box_size,
    )

    assert len(coordinates) == chain_length
    assert coordinates.shape == (chain_length, 3)


def test_bond_lengths():
    """Test if bonds are within expected length"""
    chain_length = 20
    bond_length = 0.85
    bead_radius = 1.0
    box_size = 50.0
    tolerance = 1e-6

    coordinates = generate_kremer_grest_chain(
        chain_length=chain_length,
        bond_length=bond_length,
        bead_radius=bead_radius,
        box_size=box_size,
    )

    # Check consecutive bond lengths
    for i in range(chain_length - 1):
        delta = coordinates[i + 1] - coordinates[i]
        actual_length = np.linalg.norm(delta)
        assert abs(actual_length - bond_length) < tolerance


def test_bead_overlap():
    """Test if beads don't overlap"""
    chain_length = 15
    bond_length = 0.85
    bead_radius = 1.0
    box_size = 50.0

    coordinates = generate_kremer_grest_chain(
        chain_length=chain_length,
        bond_length=bond_length,
        bead_radius=bead_radius,
        box_size=box_size,
    )

    # Check distances between non-bonded beads
    for i in range(chain_length):
        for j in range(i + 2, chain_length):  # Skip consecutive beads
            delta = coordinates[j] - coordinates[i]
            distance = np.linalg.norm(delta)
            assert distance >= 2 * bead_radius  # Beads shouldn't overlap


def test_periodic_boundary_conditions():
    """Test if coordinates respect periodic boundary conditions"""
    chain_length = 10
    bond_length = 0.85
    bead_radius = 1.0
    box_size = 5.0  # Small box to force PBC

    coordinates = generate_kremer_grest_chain(
        chain_length=chain_length,
        bond_length=bond_length,
        bead_radius=bead_radius,
        box_size=box_size,
    )

    # All coordinates should be within box bounds
    assert np.all(coordinates >= 0)
    assert np.all(coordinates < box_size)


def test_invalid_inputs():
    """Test if function handles invalid inputs correctly"""
    with pytest.raises(ValueError):
        generate_kremer_grest_chain(
            chain_length=0,  # Invalid chain length
            bond_length=0.85,
            bead_radius=1.0,
            box_size=50.0,
        )

    with pytest.raises(ValueError):
        generate_kremer_grest_chain(
            chain_length=10,
            bond_length=-1.0,  # Invalid bond length
            bead_radius=1.0,
            box_size=50.0,
        )

    with pytest.raises(ValueError):
        generate_kremer_grest_chain(
            chain_length=10,
            bond_length=0.85,
            bead_radius=-1.0,  # Invalid bead radius
            box_size=50.0,
        )
