import os
import random

import click
import numpy as np

from polydispers.files_io import (
    read_topology_file,
    write_lammps_data,
    write_lammps_input,
    write_packmol_input,
    write_topology_file,
    write_xyz_file,
)
from polydispers.input_config import load_config
from polydispers.system_generator import generate_polymer_system


@click.group()
def cli():
    """Generate and prepare polymer systems for LAMMPS simulation."""
    pass


@cli.command()
@click.option("--config", type=str, default="input_config.yaml", help="Path to the input configuration file.")
def generate(config):
    """Generate a polymer system with specified parameters."""

    config = load_config(config)

    np.random.seed(config.seed)
    random.seed(config.seed)

    output_dir = config.output_dir
    box_size = config.box_size

    num_chains = config.num_chains
    mn = config.mn
    pdi = config.pdi
    bond_length = config.polymer.bond_length
    bead_radius = config.polymer.bead_radius

    print("Polydispers Generator, version 0.1.0")
    print("-" * 100)
    print(f"Number of chains: {num_chains}")
    print(f"Number-average molecular weight: {mn}")
    print(f"Polydispersity index: {pdi}")
    print(f"Bond length: {bond_length}")
    print(f"Bead radius: {bead_radius}")
    print(f"Box size: {box_size}")
    print("-" * 100)
    print("Generating polymer system...")

    polymer_system, chain_lengths = generate_polymer_system(config)

    output_dir = f"{output_dir}/chains_{num_chains}_Mn{mn}_PDI{pdi}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each of the chain in a separate gro file
    filenames = []
    for i, chain in enumerate(polymer_system):
        # write_pdb_file(f"{output_dir}/chain_{i}.pdb", chain, single_chain_box_size)
        write_xyz_file(f"{output_dir}/chain_{i}.xyz", chain)
        filenames.append(f"{output_dir}/chain_{i}.xyz")

    write_packmol_input(f"{output_dir}/lj.xyz", filenames, box_size, f"{output_dir}/packmol_input.txt")

    # Write topology file
    write_topology_file(f"{output_dir}/topology.yaml", config, chain_lengths)

    print(f"\nTopology file written to {output_dir}/topology.yaml")
    print(f"Packmol input file written to {output_dir}/packmol_input.txt\n")

    # Next steps
    print("-" * 100)
    print("Next steps:")
    print("-" * 100)
    print(f"1. Run packmol with input file {output_dir}/packmol_input.txt\n")
    print(f"packmol < {output_dir}/packmol_input.txt\n")
    print("2. Prepare LAMMPS input files for the generated polymer system.\n")
    print(
        (
            f"polydispers lammps --topology-file {output_dir}/topology.yaml "
            f"--coordinates {output_dir}/lj.xyz "
            f"--output-file {output_dir}/lj.data\n"
        )
    )
    print(f"3. Run lammps with data file {output_dir}/lj.data\n")
    print(f"lmp -in {output_dir}/lj.data\n")

    # The instruction in form of a bash script is written to a file
    with open(f"{output_dir}/instructions.sh", "w") as f:
        f.write("#! /bin/bash\n")
        f.write(f"packmol < {output_dir}/packmol_input.txt\n")
        f.write(
            f"polydispers lammps --topology-file {output_dir}/topology.yaml " f"--coordinates {output_dir}/lj.xyz" "\n"
        )
        f.write(f"lmp -in {output_dir}/lj.data\n")
    print(f"Instructions written to {output_dir}/instructions.sh")
    print(f"You can now run the script {output_dir}/instructions.sh to prepare the system.\n")


@cli.command()
@click.option("--topology-file", type=str, required=True, help="Path to the topology file.")
@click.option("--coordinates", type=str, required=True, help="Path to the coordinates file.")
def lammps(topology_file, coordinates):
    """Prepare LAMMPS input files for the generated polymer system."""
    basedir = os.path.dirname(coordinates)
    filename = os.path.basename(coordinates)

    chain_description, bond_list, box_size = read_topology_file(topology_file)
    coordinates = np.loadtxt(coordinates, skiprows=2, usecols=(1, 2, 3))

    data_file = os.path.join(basedir, filename.split(".")[0] + ".data")
    in_file = os.path.join(basedir, filename.split(".")[0] + ".in")

    write_lammps_data(data_file, coordinates, chain_description, bond_list, box_size)
    click.echo(f"LAMMPS data file written to {data_file}")

    write_lammps_input(in_file, data_file)
    click.echo(f"LAMMPS input file written to {in_file}")


if __name__ == "__main__":
    cli()
