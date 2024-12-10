import os
from dataclasses import dataclass

import yaml


@dataclass
class BeadTypeConfig:
    mass: float
    type_id: int


@dataclass
class PolymerConfig:
    bond_length: float
    bead_radius: float
    repeat_unit_topology: str
    bead_types: dict[str, BeadTypeConfig]

    def __post_init__(self):
        for bead_type, config in self.bead_types.items():
            self.bead_types[bead_type] = BeadTypeConfig(**config)

    @property
    def repeat_unit_mass(self):
        return sum(self.bead_types[bead].mass for bead in self.repeat_unit_topology)

    @property
    def repeat_unit_length(self):
        return len(self.repeat_unit_topology)


@dataclass
class InputConfig:
    num_chains: int
    mn: float
    pdi: float
    box_size: float
    output_dir: str
    seed: int
    polymer: PolymerConfig

    def __post_init__(self):
        self.output_dir = os.path.abspath(self.output_dir)
        self.polymer = PolymerConfig(**self.polymer)

    @classmethod
    def from_string(cls, config_string: str) -> "InputConfig":
        config = yaml.safe_load(config_string)
        return cls(**config)

    @classmethod
    def from_file(cls, config_path: str) -> "InputConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


def load_config(config_path: str) -> InputConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return InputConfig(**config)
