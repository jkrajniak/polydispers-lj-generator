import os
import tempfile

import numpy as np

from polydispers.files_io import (
    read_topology_file,
    write_lammps_data,
    write_topology_file,
)
from polydispers.input_config import InputConfig, PolymerConfig
from polydispers.topology_config import BeadType, ChainDescription
from polydispers.topology_config import PolymerConfig as TopologyPolymerConfig
from polydispers.topology_config import TopologyConfig


def test_topology_file_io():
    """Test writing and reading topology file"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        # Create test data
        config = InputConfig(
            num_chains=10,
            mn=1000,
            pdi=1.2,
            box_size=100.0,
            output_dir="test_output",
            seed=42,
            polymer=PolymerConfig(
                bond_length=0.85,
                bead_radius=1.0,
                repeat_unit_topology="AB",
                bead_types={"A": {"mass": 1.0, "type_id": 1}, "B": {"mass": 1.0, "type_id": 2}},
            ),
        )
        chain_lengths = [10, 20, 15, 12, 8, 18, 14, 16, 11, 9]

        # Write topology file
        write_topology_file(tmp.name, config, chain_lengths)

        # Read topology file
        topology = read_topology_file(tmp.name)

        # Verify contents
        assert topology.box_size == config.box_size
        assert topology.chain_description.repeat_unit_topology == config.polymer.repeat_unit_topology
        assert topology.chain_description.chain_lengths == chain_lengths
        assert len(topology.polymer.bead_types) == len(config.polymer.bead_types)
        for bead_type in topology.polymer.bead_types.values():
            assert isinstance(bead_type, BeadType)

    # Clean up
    os.unlink(tmp.name)


def test_lammps_data_writing():
    """Test writing LAMMPS data file"""
    with tempfile.NamedTemporaryFile(suffix=".data", delete=False) as tmp:
        # Create test data
        topology = TopologyConfig(
            box_size=100.0,
            chain_description=ChainDescription(repeat_unit_topology="AB", chain_lengths=[10, 20]),
            polymer=TopologyPolymerConfig(
                bead_types={"A": BeadType(mass=1.0, type_id=1), "B": BeadType(mass=1.0, type_id=2)}
            ),
        )

        # Create some test coordinates
        total_beads = sum(
            len(topology.chain_description.repeat_unit_topology) * n for n in topology.chain_description.chain_lengths
        )
        coordinates = np.random.rand(total_beads, 3) * topology.box_size

        # Write LAMMPS data file
        write_lammps_data(tmp.name, topology, coordinates)

        # Verify file exists and has content
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0

    # Clean up
    os.unlink(tmp.name)
