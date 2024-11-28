import argparse
import os
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from polydispers.chain_generator import generate_kremer_grest_chain
from polydispers.files_io import (
    read_topology_file,
    write_lammps_data,
    write_lammps_input,
    write_packmol_input,
    write_topology_file,
    write_xyz_file,
)
from polydispers.stats import sz_distribution_inverse_transform


def generate_polymer_system(
    num_chains, Mn, PDI, box_size, bond_length=1.0, bead_radius=1.0, disable_pbc=False
) -> tuple[np.array, list[int]]:
    """
    Generates a system of Kremer-Grest chains with Schulz-Zimm
    molecular weight distribution in 3D space.

    Args:
      num_chains: Number of chains to generate.
      Mn: Number-average molecular weight.
      PDI: Polydispersity index.
      bond_length: Equilibrium bond length for FENE potential.
      bead_radius: Radius of the beads.
      box_size: Size of the cubic box.
      disable_pbc: Whether to disable periodic boundary conditions.
    Returns:
        A tuple with the numpy array of shape (N, 3) containing the bead coordinates and a list of chain lengths.
    """
    chain_lengths = sz_distribution_inverse_transform(Mn, PDI, size=num_chains).astype(int)
    print(chain_lengths)
    print(f"Sum of chain lengths: {sum(chain_lengths)}, difference: {sum(chain_lengths) - Mn}")
    polymer_system = []

    # Generate chains in parallel
    fn_generator = partial(
        generate_kremer_grest_chain,
        bond_length=bond_length,
        bead_radius=bead_radius,
        box_size=box_size,
        disable_pbc=disable_pbc,
    )
    with Pool(processes=4) as pool:
        polymer_system = list(tqdm(pool.imap(fn_generator, chain_lengths), total=num_chains))
    return polymer_system, chain_lengths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a polymer system with a given number of chains, molecular weight, and polydispersity index."
        )
    )
    parser.add_argument("--num_chains", type=int, default=1, help="Number of chains to generate.")
    parser.add_argument("--Mn", type=float, default=50, help="Number-average molecular weight.")
    parser.add_argument("--PDI", type=float, default=1.2, help="Polydispersity index.")
    parser.add_argument("--box-size", type=float, default=500, help="Size of the cubic box. (Angstrom)")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--disable-pbc", action="store_true", help="Disable periodic boundary conditions.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Example usage:
    num_chains = args.num_chains
    Mn = args.Mn
    PDI = args.PDI
    bond_length = 0.85  # Example bond length
    bead_radius = 1.0  # Bead radius
    box_size = args.box_size

    single_chain_box_size = box_size

    polymer_system, chain_lengths = generate_polymer_system(
        num_chains, Mn, PDI, single_chain_box_size, bond_length, bead_radius, args.disable_pbc
    )

    output_dir = f"{args.output_dir}/chains_{num_chains}_Mn{Mn}_PDI{PDI}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each of the chain in a separate gro file
    filenames = []
    for i, chain in enumerate(polymer_system):
        # write_pdb_file(f"{output_dir}/chain_{i}.pdb", chain, single_chain_box_size)
        write_xyz_file(f"{output_dir}/chain_{i}.xyz", chain)
        filenames.append(f"{output_dir}/chain_{i}.xyz")

    print(f"Box size: {single_chain_box_size}")
    write_packmol_input(f"{output_dir}/lj.xyz", filenames, box_size, f"{output_dir}/packmol_input.txt")

    # Write topology file
    write_topology_file(f"{output_dir}/topology.yaml", chain_lengths)

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
            f"polydispers-lammps --topology-file {output_dir}/topology.yaml "
            f"--box-size {box_size} --coordinates {output_dir}/lj.xyz "
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
            f"polydispers-lammps --topology-file {output_dir}/topology.yaml "
            f"--box-size {box_size} --coordinates {output_dir}/lj.xyz "
            f"--output-file {output_dir}/lj.data\n"
        )
        f.write(f"lmp -in {output_dir}/lj.data\n")
    print(f"Instructions written to {output_dir}/instructions.sh")
    print(f"You can now run the script {output_dir}/instructions.sh to prepare the system.\n")


def prepare_lammps():
    parser = argparse.ArgumentParser("Prepare LAMMPS input files for the generated polymer system.")
    parser.add_argument("--topology-file", type=str, required=True, help="Path to the topology file.")
    parser.add_argument("--coordinates", type=str, required=True, help="Path to the coordinates file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output LAMMPS data file.")
    parser.add_argument("--box-size", type=float, required=True, help="Size of the cubic box.")

    args = parser.parse_args()

    box_size = args.box_size
    chain_description, bond_list = read_topology_file(args.topology_file)

    coordinates = np.loadtxt(args.coordinates, skiprows=2, usecols=(1, 2, 3))

    data_file = args.output_file.split(".")[0] + ".data"
    in_file = args.output_file.split(".")[0] + ".in"

    # Write LAMMPS data file
    write_lammps_data(data_file, coordinates, chain_description, bond_list, box_size)

    # Write LAMMPS input file
    write_lammps_input(in_file, data_file)
