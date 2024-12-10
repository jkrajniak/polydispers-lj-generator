import numpy as np
from scipy.special import gamma, gammainc

from polydispers.input_config import InputConfig


def sz_distribution_inverse_transform(config: InputConfig):
    """
    Generates chain lengths in terms of repeat units from the Schulz-Zimm distribution
    using inverse transform sampling with the Newton-Raphson method.

    Args:
        config: Input configuration containing Mn (total system molecular weight), PDI, and topology information
        size: Number of chains to generate

    Returns:
        Tuple of (molecular_weights, num_repeat_units)
    """
    # Calculate mass of one repeat unit
    repeat_unit_mass = sum(config.polymer.bead_types[bead].mass for bead in config.polymer.repeat_unit_topology)

    # Convert total Mn to target number of repeat units
    # We divide by repeat_unit_mass because we want the number of repeat units
    target_n = config.mn / repeat_unit_mass

    num_chains = config.num_chains

    # Use target_n for the distribution calculations
    z = 1 / (config.pdi - 1)
    u = np.random.uniform(0, 1, size=num_chains)

    def sz_cdf(x):
        return gammainc(z + 1, (z + 1) * x / target_n)

    def sz_cdf_derivative(x):
        return ((z + 1) / target_n) * ((z + 1) * x / target_n) ** z * np.exp(-(z + 1) * x / target_n) / gamma(z + 1)

    # Generate number of repeat units
    num_repeat_units = np.zeros(num_chains, dtype=int)
    for i in range(num_chains):
        x = target_n  # Initial guess
        tolerance = 1e-6
        max_iterations = 100
        for _ in range(max_iterations):
            x_next = x - (sz_cdf(x) - u[i]) / sz_cdf_derivative(x)
            if abs(x_next - x) < tolerance:
                break
            x = x_next

        # Round to nearest integer for number of repeat units
        num_repeat_units[i] = round(x)

    # Calculate molecular weights
    molecular_weights = num_repeat_units * repeat_unit_mass

    # Scale to match target total molecular weight while keeping integer repeat units
    scale_factor = config.mn / np.sum(molecular_weights)
    num_repeat_units = np.round(num_repeat_units * scale_factor).astype(int)

    # Adjust one chain length to match total exactly if needed
    total_mass = np.sum(num_repeat_units * repeat_unit_mass)
    if total_mass != config.mn:
        diff_repeat_units = int(round((config.mn - total_mass) / repeat_unit_mass))
        # Add the difference to the longest chain to minimize impact on distribution
        idx = np.argmax(num_repeat_units)
        num_repeat_units[idx] += diff_repeat_units

    # Calculate final molecular weights
    molecular_weights = num_repeat_units * repeat_unit_mass

    return molecular_weights, num_repeat_units
