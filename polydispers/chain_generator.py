import numpy as np

from polydispers.input_config import InputConfig


def generate_kremer_grest_chain(config: InputConfig, num_repeat_units: int):
    """
    Generates a single linear Kremer-Grest chain in 3D space, preventing
    bead overlap. Assumes beads are spheres with the given radius and
    a Lennard-Jones potential with sigma = 1.0.

    Args:
      config: Input configuration.
      num_repeat_units: Number of repeat units in the chain.
    Returns:
      A NumPy array of shape (chain_length, 3) with the bead coordinates.
    """
    disable_pbc = True
    coordinates = np.zeros((num_repeat_units * config.polymer.repeat_unit_length, 3))
    coordinates[0] = np.array([0, 0, 0])  # Start in box center

    def apply_pbc(pos):
        """Apply periodic boundary conditions"""
        if disable_pbc:
            return pos
        return pos - config.box_size * np.floor(pos / config.box_size)

    def get_minimum_image_distance(pos1, pos2):
        """Calculate minimum image distance between two points"""
        delta = pos1 - pos2
        if disable_pbc:
            return delta
        delta = delta - config.box_size * np.round(delta / config.box_size)
        return delta

    for i in range(1, num_repeat_units * config.polymer.repeat_unit_length):
        print(
            (
                f"Generating bead {i} / {num_repeat_units * config.polymer.repeat_unit_length} "
                f"of chain ({i / (num_repeat_units * config.polymer.repeat_unit_length) * 100:.2f}%)"
            ),
            end="\r",
        )
        valid_position = False
        count = 0
        while not valid_position:
            if i > 1:
                delta = get_minimum_image_distance(coordinates[i - 1], coordinates[i - 2])
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
                direction = (
                    direction * cos_theta
                    + np.cross(random_axis, direction) * sin_theta
                    + random_axis * np.dot(random_axis, direction) * (1 - cos_theta)
                )
            else:  # This are the first two beads.
                phi = np.random.uniform(0, np.pi)
                theta = np.random.uniform(0, 2 * np.pi)
                direction = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
            next_position = coordinates[i - 1] + config.polymer.bond_length * direction
            next_position = apply_pbc(next_position)

            # Check for overlap with previous beads (considering bead radius)
            if i > 1:  # Only check for overlaps if there are at least 2 beads
                valid_position = True
                for j in range(i - 2):
                    delta = get_minimum_image_distance(next_position, coordinates[j])
                    distance = np.linalg.norm(delta)
                    if distance < 2 * config.polymer.bead_radius:
                        valid_position = False
                        break
            else:
                valid_position = True  # No need to check for the second bead
            count += 1

        coordinates[i] = next_position

    return coordinates
