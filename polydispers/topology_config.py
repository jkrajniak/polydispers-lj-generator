from dataclasses import dataclass
from typing import Dict


@dataclass
class BeadType:
    """Configuration for a single bead type"""

    mass: float
    type_id: int


@dataclass
class Bond:
    """Description of a bond between two atoms"""

    atom_i: int  # Index of first atom (1-based for LAMMPS)
    atom_j: int  # Index of second atom (1-based for LAMMPS)


@dataclass
class ChainDescription:
    """Description of the chain topology"""

    repeat_unit_topology: str
    chain_lengths: list[int]


@dataclass
class PolymerConfig:
    """Configuration for the polymer"""

    bead_types: Dict[str, BeadType]


@dataclass
class TopologyConfig:
    """Complete topology configuration"""

    box_size: float
    chain_description: ChainDescription
    polymer: PolymerConfig

    @classmethod
    def from_dict(cls, data: dict) -> "TopologyConfig":
        """Create TopologyConfig from a dictionary (e.g., loaded from YAML)"""
        chain_desc = ChainDescription(
            repeat_unit_topology=data["chain_description"]["repeat_unit_topology"],
            chain_lengths=data["chain_description"]["chain_lengths"],
        )

        bead_types = {name: BeadType(**props) for name, props in data["polymer"]["bead_types"].items()}

        polymer = PolymerConfig(bead_types=bead_types)

        return cls(box_size=data["box_size"], chain_description=chain_desc, polymer=polymer)

    def to_dict(self) -> dict:
        """Convert TopologyConfig to a dictionary (for YAML dumping)"""
        return {
            "box_size": self.box_size,
            "chain_description": {
                "repeat_unit_topology": self.chain_description.repeat_unit_topology,
                "chain_lengths": self.chain_description.chain_lengths,
            },
            "polymer": {
                "bead_types": {
                    name: {"mass": bead.mass, "type_id": bead.type_id} for name, bead in self.polymer.bead_types.items()
                }
            },
        }
