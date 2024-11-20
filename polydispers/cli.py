import argparse
from functools import partial
import os
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from polydispers.chain_generator import generate_kremer_grest_chain
from polydispers.stats import sz_distribution_inverse_transform
from polydispers.files_io import write_pdb_file, write_packmol_input, write_xyz_file, write_topology_file, \
    read_topology_file


def generate_polymer_system(num_chains, Mn, PDI, box_size, bond_length=1.0, bead_radius=1.0) -> tuple[np.array, list[int]]:
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
    
  Returns:
      A tuple with the numpy array of shape (N, 3) containing the bead coordinates and a list of chain lengths.
  """
    chain_lengths = sz_distribution_inverse_transform(Mn, PDI, size=num_chains).astype(int)
    print(chain_lengths)
    print(f"Sum of chain lengths: {sum(chain_lengths)}, difference: {sum(chain_lengths) - Mn}")
    polymer_system = []

    # Generate chains in parallel
    fn_generator = partial(generate_kremer_grest_chain, bond_length=bond_length,
                           bead_radius=bead_radius, box_size=box_size)
    with Pool(processes=4) as pool:
        polymer_system = list(tqdm(pool.imap(fn_generator, chain_lengths),
                                   total=num_chains))
    return polymer_system, chain_lengths



def main():
    parser = argparse.ArgumentParser(
        description="Generate a polymer system with a given number of chains, molecular weight, and polydispersity index.")
    parser.add_argument("--num_chains", type=int, default=1, help="Number of chains to generate.")
    parser.add_argument("--Mn", type=float, default=50, help="Number-average molecular weight.")
    parser.add_argument("--PDI", type=float, default=1.2, help="Polydispersity index.")
    parser.add_argument("--box-size", type=float, default=500, help="Size of the cubic box. (Angstrom)")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

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

    polymer_system, chain_lengths = generate_polymer_system(num_chains, Mn, PDI, single_chain_box_size, bond_length,
                                                            bead_radius)

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
    write_packmol_input("lj.xyz", filenames, box_size, filetype="xyz")

    # Write topology file
    write_topology_file(f"{output_dir}/topology.yaml", chain_lengths)


def prepare_lammps():
    parser = argparse.ArgumentParser("Prepare LAMMPS input files for the generated polymer system.")
    parser.add_argument("--topology-file", type=str, required=True, help="Path to the topology file.")
    parser.add_argument("--coordinates", type=str, required=True, help="Path to the coordinates file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output LAMMPS data file.")

    args = parser.parse_args()

    chain_description, bond_list = read_topology_file(args.topology_file)

    coordinates = np.loadtxt(args.coordinates, skiprows=2, usecols=(1, 2, 3))
    # Write LAMMPS data file
    with open(args.output_file, 'w+') as out_lmp:
        out_lmp.write("LAMMPS data file\n\n")
        out_lmp.write(f"{len(coordinates)} atoms\n")
        out_lmp.write(f"{len(bond_list)} bonds\n")
        out_lmp.write("\n")
        out_lmp.write("1 atom types\n")
        out_lmp.write("1 bond types\n")
        out_lmp.write("\n")
        out_lmp.write("0.0 500.0 xlo xhi\n")
        out_lmp.write("0.0 500.0 ylo yhi\n")
        out_lmp.write("0.0 500.0 zlo zhi\n")
        out_lmp.write("\n")
        out_lmp.write("Atoms\n\n")
        for i, (atom_id, _, res_id, _) in enumerate(chain_description, start=1):
            x, y, z = coordinates[i - 1]
            out_lmp.write(f"{i} {res_id} {res_id} {x} {y} {z}\n")
        out_lmp.write("\n")
        out_lmp.write("Bonds\n\n")
        for i, (atom_i, atom_j) in enumerate(bond_list, start=1):
            out_lmp.write(f"{i} 1 {atom_i} {atom_j}\n")
