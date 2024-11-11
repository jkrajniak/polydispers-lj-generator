import argparse
from functools import partial
import os
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from scipy.special import gamma, gammainc


from files_io import write_pdb_file, write_packmol_input, write_xyz_file


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


def generate_kremer_grest_chain(chain_length: int, bond_length: float, bead_radius: float,
                                box_size: float):
  """
  Generates a single linear Kremer-Grest chain in 3D space, preventing 
  bead overlap. Assumes beads are spheres with the given radius and
  a Lennard-Jones potential with sigma = 1.0.

  Args:
    chain_length: Number of beads in the chain.
    bond_length: Equilibrium bond length for FENE potential.
    bead_radius: Radius of the beads.
    box_size: Size of the cubic box.

  Returns:
    A NumPy array of shape (chain_length, 3) with the bead coordinates.
  """
  coordinates = np.zeros((chain_length, 3))
  coordinates[0] = np.array([0, 0, 0])  # Start in box center
    
  def apply_pbc(pos):
    """Apply periodic boundary conditions"""
    return pos
    return pos - box_size * np.floor(pos / box_size)

  def get_minimum_image_distance(pos1, pos2):
      """Calculate minimum image distance between two points"""
      delta = pos1 - pos2
      return delta
      delta = delta - box_size * np.round(delta / box_size)
      return delta
    
  
  for i in range(1, chain_length):
    print(f"Generating bead {i} / {chain_length} of chain ({i/chain_length*100:.2f}%)", end="\r")
    valid_position = False
    count = 0
    while not valid_position:
      if i > 1:
        delta = get_minimum_image_distance(coordinates[i-1], coordinates[i-2])
        direction = delta / np.linalg.norm(delta)
        
        # Add small perturbation and ensure angle is in 1-180 range
        base_angle = np.random.uniform(0, 30)  # Base angle in degrees
        angle = (base_angle + (count / 1000) * 10) % 180  # Increase up to 10 degrees, wrap to 1-180
        angle = max(1, angle)  # Ensure minimum of 1 degree
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
      next_position = apply_pbc(next_position)

      # Check for overlap with previous beads (considering bead radius)
      if i > 1:  # Only check for overlaps if there are at least 2 beads
        valid_position = True
        for j in range(i-2):
            delta = get_minimum_image_distance(next_position, coordinates[j])
            distance = np.linalg.norm(delta)
            if distance < 2 * bead_radius:
                valid_position = False
                break
      else:
        valid_position = True  # No need to check for the second bead
      count += 1
    
    coordinates[i] = next_position
    
  return coordinates


def generate_polymer_system(num_chains, Mn, PDI, box_size, bond_length=1.0, bead_radius=1.0):
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
    A list of NumPy arrays, where each array contains the coordinates 
    of a single chain.
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
  return polymer_system



def parse_args():
  parser = argparse.ArgumentParser(
    description="Generate a polymer system with a given number of chains, molecular weight, and polydispersity index.")
  parser.add_argument("--num_chains", type=int, default=1, help="Number of chains to generate.")
  parser.add_argument("--Mn", type=float, default=50, help="Number-average molecular weight.")
  parser.add_argument("--PDI", type=float, default=1.2, help="Polydispersity index.")
  parser.add_argument("--box-size", type=float, default=500, help="Size of the cubic box. (Angstrom)")
  parser.add_argument("--output-dir", type=str, default=".", help="Output directory.")
  parser.add_argument("--seed", type=int, default=42, help="Random seed.")
  return parser.parse_args()

def main():
  args = parse_args()
  
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
  
  polymer_system = generate_polymer_system(num_chains, Mn, PDI, single_chain_box_size, bond_length, 
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
  
if __name__ == "__main__":
  main()

# --- Further processing ---
# (Same as before)