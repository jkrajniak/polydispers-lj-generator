import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np

from polydispers.files_io import (
    read_topology_file,
    write_lammps_data,
    write_lammps_input,
    write_packmol_input,
    write_topology_file,
    write_xyz_file,
)
from polydispers.input_config import InputConfig
from polydispers.system_generator import generate_polymer_system

__all__ = ["GeneratedSystem", "LammpsFiles", "generate_polymer_files", "prepare_lammps_files"]


@dataclass
class GeneratedSystem:
    """Result of polymer system generation"""

    output_dir: str
    topology_file: str
    packmol_input_file: str
    chain_files: List[str]
    instructions_file: str
    chain_lengths: List[int]


@dataclass
class LammpsFiles:
    """Result of LAMMPS file preparation"""

    data_file: str
    input_file: str


def generate_polymer_files(config: InputConfig) -> GeneratedSystem:
    """Generate polymer system files based on configuration.

    Args:
        config: Input configuration

    Returns:
        GeneratedSystem containing paths to all generated files
    """
    np.random.seed(config.seed)
    random.seed(config.seed)

    output_dir = (
        f"{config.output_dir}/chains_{config.polymer.repeat_unit_topology}_num_chains_{config.num_chains}_"
        f"Mn_{config.mn}_PDI_{config.pdi}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Generate polymer system
    polymer_system, chain_lengths = generate_polymer_system(config)

    # Save each chain in a separate xyz file
    chain_files = []
    for i, chain in enumerate(polymer_system):
        chain_file = f"{output_dir}/chain_{i}.xyz"
        write_xyz_file(chain_file, chain)
        chain_files.append(chain_file)

    # Write packmol input
    packmol_file = f"{output_dir}/packmol_input.txt"
    output_xyz = f"{output_dir}/lj.xyz"
    write_packmol_input(output_xyz, chain_files, config.box_size, packmol_file)

    # Write topology file
    topology_file = f"{output_dir}/topology.yaml"
    write_topology_file(topology_file, config, chain_lengths)

    # Write instructions
    instructions_file = f"{output_dir}/instructions.sh"
    with open(instructions_file, "w") as f:
        f.write("#! /bin/bash\n")
        f.write(f"packmol < {packmol_file}\n")
        f.write(f"polydispers lammps --topology-file {topology_file} --coordinates {output_xyz}\n")
        f.write(f"lmp -in {output_dir}/lj.data\n")

    return GeneratedSystem(
        output_dir=output_dir,
        topology_file=topology_file,
        packmol_input_file=packmol_file,
        chain_files=chain_files,
        instructions_file=instructions_file,
        chain_lengths=chain_lengths,
    )


def prepare_lammps_files(topology_file: str, coordinates_file: str) -> LammpsFiles:
    """Prepare LAMMPS input files from topology and coordinates.

    Args:
        topology_file: Path to the topology YAML file
        coordinates_file: Path to the coordinates XYZ file

    Returns:
        LammpsFiles containing paths to generated LAMMPS files
    """
    basedir = os.path.dirname(coordinates_file)
    filename = os.path.basename(coordinates_file)

    topology_data = read_topology_file(topology_file)
    coordinates = np.loadtxt(coordinates_file, skiprows=2, usecols=(1, 2, 3))

    data_file = os.path.join(basedir, filename.split(".")[0] + ".data")
    in_file = os.path.join(basedir, filename.split(".")[0] + ".in")

    write_lammps_data(
        data_file, coordinates, topology_data.chain_description, topology_data.bond_list, topology_data.box_size
    )

    write_lammps_input(in_file, data_file, topology_data)

    return LammpsFiles(data_file=data_file, input_file=in_file)
