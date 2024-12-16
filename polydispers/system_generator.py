from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from polydispers.chain_generator import generate_kremer_grest_chain
from polydispers.input_config import InputConfig
from polydispers.stats import sz_distribution_inverse_transform


def generate_polymer_system(config: InputConfig) -> tuple[np.array, list[int]]:
    """
    Generates a system of Kremer-Grest chains with Schulz-Zimm
    molecular weight distribution in 3D space.

    Args:
        config: Input configuration containing system parameters
    Returns:
        A tuple with the numpy array of shape (N, 3) containing the bead coordinates
        and a list of chain lengths.
    """
    molecular_weights, chain_lengths = sz_distribution_inverse_transform(config)

    # Calculate repeat unit mass for clearer output
    repeat_unit_mass = sum(config.polymer.bead_types[bead].mass for bead in config.polymer.repeat_unit_topology)

    print(f"Repeat unit mass: {repeat_unit_mass}")
    print(f"Molecular weights: {molecular_weights}")
    print(f"Chain lengths (repeat units): {chain_lengths}")
    print(f"Total beads: {sum(len(config.polymer.repeat_unit_topology) * n for n in chain_lengths)}")
    print(f"Sum of molecular weights: {sum(molecular_weights)}")
    print(f"Target total molecular weight (Mn): {config.mn}")
    print(f"Difference from target: {sum(molecular_weights) - config.mn}")

    # Generate chains in parallel
    fn_generator = partial(generate_kremer_grest_chain, config)
    with Pool(processes=4) as pool:
        polymer_system = list(tqdm(pool.imap(fn_generator, chain_lengths), total=config.num_chains))
    return polymer_system, chain_lengths
