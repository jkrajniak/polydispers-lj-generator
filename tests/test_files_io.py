import os
import tempfile

import numpy as np
import pytest

from polydispers.files_io import (
    read_topology_file,
    write_lammps_data,
    write_topology_file,
    write_xyz_file,
)


@pytest.fixture
def sample_coordinates():
    return np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])


@pytest.fixture
def sample_chain_description():
    return [(1, "C", 1, "MOL"), (2, "C", 1, "MOL"), (3, "C", 1, "MOL")]


@pytest.fixture
def sample_bond_list():
    return [(1, 2), (2, 3)]


def test_xyz_file_writing(sample_coordinates):
    """Test XYZ file writing"""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        write_xyz_file(tmp.name, sample_coordinates)

        # Read and verify file contents
        with open(tmp.name, "r") as f:
            lines = f.readlines()
            assert len(lines) == len(sample_coordinates) + 2  # Header + comment + coordinates
            assert int(lines[0].strip()) == len(sample_coordinates)  # Number of atoms

            # Check coordinates
            for i, (x, y, z) in enumerate(sample_coordinates):
                line = lines[i + 2].strip().split()
                assert float(line[1]) == x
                assert float(line[2]) == y
                assert float(line[3]) == z

    os.unlink(tmp.name)


def test_topology_file_io(sample_chain_description, sample_bond_list):
    """Test topology file writing and reading"""
    box_size = 50.0

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        write_topology_file(tmp.name, box_size, [3])  # One chain of length 3

        # Read and verify
        chain_desc, bonds, read_box_size = read_topology_file(tmp.name)

        assert read_box_size == box_size
        assert len(chain_desc) == len(sample_chain_description)
        assert len(bonds) == len(sample_bond_list)

    os.unlink(tmp.name)


def test_lammps_data_writing(sample_coordinates, sample_chain_description, sample_bond_list):
    """Test LAMMPS data file writing"""
    box_size = 50.0

    with tempfile.NamedTemporaryFile(suffix=".data", delete=False) as tmp:
        write_lammps_data(
            tmp.name,
            sample_coordinates,
            sample_chain_description,
            sample_bond_list,
            box_size,
        )

        # Read and verify file contents
        with open(tmp.name, "r") as f:
            content = f.read()
            assert "LAMMPS data file" in content
            assert f"{len(sample_coordinates)} atoms" in content
            assert f"{len(sample_bond_list)} bonds" in content
            assert f"0.0 {box_size} xlo xhi" in content

    os.unlink(tmp.name)


def test_invalid_file_operations():
    """Test handling of invalid file operations"""
    invalid_coords = np.array([[0, 0]])  # Invalid shape

    with pytest.raises(ValueError):
        with tempfile.NamedTemporaryFile(suffix=".xyz") as tmp:
            write_xyz_file(tmp.name, invalid_coords)

    with pytest.raises(FileNotFoundError):
        read_topology_file("nonexistent_file.yaml")
