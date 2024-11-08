import argparse
from functools import partial
import os
import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import uniform
from tqdm import tqdm
from multiprocessing import Pool


import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import uniform

import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import uniform

import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import uniform


def sz_distribution_inverse_transform(Mn, PDI, size=1):
  """
  Generates molecular weights from the Schulz-Zimm distribution 
  using inverse transform sampling with the Newton-Raphson method.

  Args:
    Mn: Number-average molecular weight.
    PDI: Polydispersity index.
    size: Number of molecular weights to generate.

  Returns:
    An array of molecular weights.
  """
  Mn_local = Mn / size
  z = 1 / (PDI - 1)
  u = np.random.uniform(0, 1, size=size)

  def sz_cdf(x):
    return gammainc(z + 1, (z + 1) * x / Mn_local)

  def sz_cdf_derivative(x):
    return ((z + 1) / Mn_local) * ((z + 1) * x / Mn_local)**z * np.exp(-(z + 1) * x / Mn_local) / gamma(z + 1)

  molecular_weights = np.zeros(size)
  for i in range(size):
    x = Mn_local  # Initial guess
    tolerance = 1e-6
    max_iterations = 100
    for _ in range(max_iterations):
      x_next = x - (sz_cdf(x) - u[i]) / sz_cdf_derivative(x)
      if abs(x_next - x) < tolerance:
        break
      x = x_next
    molecular_weights[i] = x

  return molecular_weights


def generate_kremer_grest_chain(chain_length, bond_length, bead_radius):
  """
  Generates a single linear Kremer-Grest chain in 3D space, preventing 
  bead overlap. Assumes beads are spheres with the given radius and
  a Lennard-Jones potential with sigma = 1.0.

  Args:
    chain_length: Number of beads in the chain.
    bond_length: Equilibrium bond length for FENE potential.
    bead_radius: Radius of the beads.

  Returns:
    A NumPy array of shape (chain_length, 3) with the bead coordinates.
  """
  coordinates = np.zeros((chain_length, 3))
  coordinates[0] = np.array([0.0, 0.0, 0.0])  # Place the first bead at the origin
  
  for i in range(1, chain_length):
    # print(f"Generating bead {i} / {chain_length} of chain ({i/chain_length*100:.2f}%)")
    valid_position = False
    count = 0
    while not valid_position:
# Get direction from previous beads
      if i > 1:
        direction = coordinates[i-1] - coordinates[i-2]
        direction = direction / np.linalg.norm(direction)
        
        # Add small perturbation of 1-2 degrees
        base_angle = np.random.uniform(1, 2)  # Base angle in degrees
        angle = base_angle + (count / 1000) * 10  # Increase up to 10 degrees
        angle = angle * np.pi / 180  # Convert to radians
        
        # Generate random rotation axis
        random_axis = np.random.randn(3)
        random_axis = random_axis / np.linalg.norm(random_axis)
        
        # Apply Rodrigues' rotation formula
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        direction = (direction * cos_theta + 
                      np.cross(random_axis, direction) * sin_theta +
                      random_axis * np.dot(random_axis, direction) * (1 - cos_theta))
      else:  # This are the first two beads.
        phi = np.random.uniform(0, np.pi)
        theta = np.random.uniform(0, 2 * np.pi)
        direction = np.array([
          np.sin(phi) * np.cos(theta), 
          np.sin(phi) * np.sin(theta), 
          np.cos(phi)
        ])
      next_position = coordinates[i-1] + bond_length * direction

      # Check for overlap with previous beads (considering bead radius)
      if i > 1:  # Only check for overlaps if there are at least 2 beads
        distances = np.linalg.norm(coordinates[:i-1] - next_position, axis=1)
        if np.all(distances >= bead_radius):  # Use bead_radius as the threshold
          valid_position = True
      else:
        valid_position = True  # No need to check for the second bead
      
      np.random.seed(i*count+i)
      
      count += 1
    
    coordinates[i] = next_position
    
  return coordinates


def generate_polymer_system(num_chains, Mn, PDI, bond_length=1.0, bead_radius=1.0):
  """
  Generates a system of Kremer-Grest chains with Schulz-Zimm 
  molecular weight distribution in 3D space.

  Args:
    num_chains: Number of chains to generate.
    Mn: Number-average molecular weight.
    PDI: Polydispersity index.
    bond_length: Equilibrium bond length for FENE potential.
    bead_radius: Radius of the beads.

  Returns:
    A list of NumPy arrays, where each array contains the coordinates 
    of a single chain.
  """
  chain_lengths = sz_distribution_inverse_transform(Mn, PDI, size=num_chains).astype(int)
  print(chain_lengths)
  print(f"Sum of chain lengths: {sum(chain_lengths)}, difference: {sum(chain_lengths) - Mn}")
  polymer_system = []
  
  # Generate chains in parallel
  fn_generator = partial(generate_kremer_grest_chain, bond_length=bond_length, 
                         bead_radius=bead_radius)
  with Pool(processes=2) as pool:
    polymer_system = list(tqdm(pool.imap(fn_generator, chain_lengths), 
                               total=num_chains))
  return polymer_system


def write_gro_file(filename: str, 
                   coordinates: np.ndarray,
                   box_size: float,
                   atomname: str = "C",
                   resname: str = "MOL",
                   title: str = "Generated by gro_writer") -> None:
    """
    Write a GRO format file with simplified inputs.
    
    Args:
        filename: Output .gro filename
        coordinates: Numpy array of shape (N, 3) containing x,y,z coordinates in nm
        box_size: Cubic box size in nm
        atomname: Name of all atoms (default: "C")
        resname: Name of all residues (default: "MOL")
        title: Title string
    """
    n_atoms = len(coordinates)
    
    with open(filename, 'w') as f:
        # Write title line
        f.write(f"{title}\n")
        
        # Write number of atoms
        f.write(f"{n_atoms}\n")
        
        # Write atom information
        for i, (x, y, z) in enumerate(coordinates, start=1):
            resid = 1
            atomid = i
            line = f"{resid:>5d}{resname:<5s}{atomname:>5s}{atomid:>5d}"
            line += f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f.write(line + '\n')
        
        # Write box vectors (cubic box)
        f.write(f"{box_size:10.5f}{box_size:10.5f}{box_size:10.5f}\n")


def write_pdb_file(filename: str,
                   coordinates: np.ndarray,
                   box_size: float,
                   atomname: str = "C",
                   resname: str = "MOL",
                   title: str = "Generated by pdb_writer") -> None:
    """
    Write a PDB format file with simplified inputs.
    
    Args:
        filename: Output .pdb filename
        coordinates: Numpy array of shape (N, 3) containing x,y,z coordinates in nm
        box_size: Cubic box size in nm
        atomname: Name of all atoms (default: "C")
        resname: Name of all residues (default: "MOL")
        title: Title string
  
    PDB Format Reference:
    ATOM/HETATM (1-6)    Record name
    atom number (7-11)    Integer
    atom name (13-16)     Atom name
    alt loc (17)          Alternate location indicator
    residue name (18-20)  Residue name
    chain (22)            Chain identifier
    residue number (23-26) Integer
    x (31-38)            Real(8.3) Orthogonal coordinates for X
    y (39-46)            Real(8.3) Orthogonal coordinates for Y
    z (47-54)            Real(8.3) Orthogonal coordinates for Z
    occupancy (55-60)     Real(6.2)
    temp factor (61-66)   Real(6.2)
    segment id (73-76)    Segment identifier
    element (77-78)       Element symbol
    charge (79-80)        Charge
    """
    # Convert nm to Angstroms for PDB format
    coords_ang = coordinates * 10.0
    box_ang = box_size * 10.0
    resid = 1
    
    with open(filename, 'w') as f:
        # Write title
        f.write(f"TITLE     {title}\n")
        f.write(f"REMARK    Generated by pdb_writer\n")
        
        # Write crystal structure (box size)
        # Format: CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
        f.write(f"CRYST1{box_ang:9.3f}{box_ang:9.3f}{box_ang:9.3f}  90.00  90.00  90.00 P 1           1\n")
        
        # Write atom coordinates
        for i, (x, y, z) in enumerate(coords_ang, start=1):
            # Format atom line according to PDB standard
            line = (f"ATOM  {i:5d}  {atomname:<3s} {resname:3s} A{resid:4d}"
                   f"{x:12.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
                   f"      {resname:<4s}{atomname:>2s}  \n")
            f.write(line)
        
        # Write END record
        f.write("END\n")


def write_packmol_input(output_filename: str,
                        filenames: list[str],
                        box_size: float) -> None:
  with open("packmol_input.txt", "w") as f:
    f.write("tolerance 2.0\n")
    f.write("filetype pdb\n")
    f.write(f"output {output_filename}\n")
    for filename in filenames:
      f.write(f"""
structure {filename}
  number 1
  inside box 0 0 0 {box_size} {box_size} {box_size}
  resnumbers 0
end structure
      """)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Generate a polymer system with a given number of chains, molecular weight, and polydispersity index.")
  parser.add_argument("--num_chains", type=int, default=1, help="Number of chains to generate.")
  parser.add_argument("--Mn", type=float, default=50, help="Number-average molecular weight.")
  parser.add_argument("--PDI", type=float, default=1.2, help="Polydispersity index.")
  parser.add_argument("--output-dir", type=str, default=".", help="Output directory.")
  return parser.parse_args()

def main():
  args = parse_args()
  
  # Example usage:
  num_chains = args.num_chains
  Mn = args.Mn
  PDI = args.PDI
  bond_length = 0.85  # Example bond length
  bead_radius = 1.0  # Bead radius

  polymer_system = generate_polymer_system(num_chains, Mn, PDI, bond_length, 
                                           bead_radius)
  
  output_dir = f"{args.output_dir}/chains_{num_chains}_Mn{Mn}_PDI{PDI}"
  
  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)
  
  max_box_size = 0
  
  # Save each of the chain in a separate gro file
  filenames = []
  for i, chain in enumerate(polymer_system):
    box_size = 2 * bead_radius * np.ceil(np.max(chain))*10
    write_gro_file(f"{output_dir}/chain_{i}.gro", chain, box_size)
    write_pdb_file(f"{output_dir}/chain_{i}.pdb", chain, box_size)
    filenames.append(f"{output_dir}/chain_{i}.pdb")
    max_box_size = max(max_box_size, box_size)

  print(f"Max box size: {max_box_size}")

  # Let's prepare input for packmol
  
  # Write the final box size to a file
  
  write_packmol_input("lj.pdb", filenames, max_box_size)
  
if __name__ == "__main__":
  main()

# --- Further processing ---
# (Same as before)