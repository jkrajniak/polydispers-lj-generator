import argparse
from functools import partial
import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import uniform
from tqdm import tqdm
from multiprocessing import Pool


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
  z = 1 / (PDI - 1)
  u = uniform.rvs(size=size)

  def sz_cdf(x):
    return gammainc(z + 1, (z + 1) * x / Mn)

  def sz_cdf_derivative(x):
    return ((z + 1) / Mn) * ((z + 1) * x / Mn)**z * np.exp(-(z + 1) * x / Mn) / gamma(z + 1)

  molecular_weights = np.zeros(size)
  for i in range(size):
    x = Mn  # Initial guess
    tolerance = 1e-6
    max_iterations = 100
    for _ in range(max_iterations):
      x_next = x - (sz_cdf(x) - u[i]) / sz_cdf_derivative(x)
      if abs(x_next - x) < tolerance:
        break
      x = x_next
    molecular_weights[i] = x

  return molecular_weights


def generate_kremer_grest_chain(chain_length, bond_length=1.0, bead_radius=1.0):
  """
  Generates a single linear Kremer-Grest chain in 3D space, preventing 
  bead overlap. Assumes beads are spheres with the given radius.

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
    valid_position = False
    while not valid_position:
      # Generate random direction in 3D
      phi = np.random.uniform(0, np.pi)
      theta = np.random.uniform(0, 2 * np.pi)
      direction = np.array([
          np.sin(phi) * np.cos(theta), 
          np.sin(phi) * np.sin(theta), 
          np.cos(phi)
      ])
      next_position = coordinates[i-1] + bond_length * direction

      # Check for overlap with previous beads (considering bead radius)
      distances = np.linalg.norm(coordinates[:i] - next_position, axis=1)
      if np.all(distances >= 2 * bead_radius):
        valid_position = True

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
  polymer_system = []
  
  # Generate chains in parallel
  fn_generator = partial(generate_kremer_grest_chain, bond_length=bond_length, 
                         bead_radius=bead_radius)
  with Pool(processes=2) as pool:
    polymer_system = list(tqdm(pool.imap(fn_generator, chain_lengths), 
                               total=num_chains))
  return polymer_system



def parse_args():
  parser = argparse.ArgumentParser(description="Generate a polymer system with a given number of chains, molecular weight, and polydispersity index.")
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
  
  # Save each of the chain in a separate gro file
  for i, chain in enumerate(polymer_system):
    with open(f"{args.output_dir}/chain_{i}.gro", "w") as f:
      f.write(f"{len(chain)}\n")
      for bead in chain:
        f.write(f"{bead[0]:.3f} {bead[1]:.3f} {bead[2]:.3f}\n")


if __name__ == "__main__":
  main()

# --- Further processing ---
# (Same as before)