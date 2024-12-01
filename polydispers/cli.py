import os
import random
from functools import partial
from multiprocessing import Pool

import click
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


@click.group()
def cli():
    """Generate and prepare polymer systems for LAMMPS simulation."""
    pass


@cli.command()
@click.option("--num-chains", type=int, default=1, help="Number of chains to generate.")
@click.option("--mn", type=float, default=50, help="Number-average molecular weight.")
@click.option("--pdi", type=float, default=1.2, help="Polydispersity index.")
@click.option("--box-size", type=float, default=500, help="Size of the cubic box. (Angstrom)")
@click.option("--output-dir", type=str, default=".", help="Output directory.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--disable-pbc", is_flag=True, help="Disable periodic boundary conditions.")
def generate(num_chains, mn, pdi, box_size, output_dir, seed, disable_pbc):
    """Generate a polymer system with specified parameters."""
    np.random.seed(seed)
    random.seed(seed)

    bond_length = 0.85
    bead_radius = 1.0
    single_chain_box_size = box_size

    polymer_system, chain_lengths = generate_polymer_system(
        num_chains, mn, pdi, single_chain_box_size, bond_length, bead_radius, disable_pbc
    )

    output_dir = f"{output_dir}/chains_{num_chains}_Mn{mn}_PDI{pdi}"

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
    write_topology_file(f"{output_dir}/topology.yaml", box_size, chain_lengths)

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
