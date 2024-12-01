# Polydispers System Generator

A Python tool for generating polydisperse polymer systems with Schulz-Zimm molecular weight distribution for LAMMPS simulations.

## Features

- Generate polymer systems with specified polydispersity index (PDI) and number-average molecular weight (Mn)
- Parallel chain generation for improved performance
- Automatic LAMMPS input file generation
- Integration with Packmol for system packing
- Support for periodic boundary conditions

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jkrajniak/polydispers-lj-generator.git
cd polydispers-lj-generator
```

2. Install the package:

```bash
pip install .
```

## Dependencies

- numpy
- scipy
- tqdm
- pyyaml
- click
- packmol (external dependency)
- LAMMPS (external dependency)

## Usage

### 1. Generate a Polymer System

```bash
polydispers generate \
    --num-chains 100 \
    --mn 50 \
    --pdi 1.2 \
    --box-size 500 \
    --output-dir ./output \
    --seed 42
```

Options:
- `--num-chains`: Number of polymer chains to generate
- `--mn`: Number-average molecular weight
- `--pdi`: Polydispersity index
- `--box-size`: Size of the cubic simulation box (Angstrom)
- `--output-dir`: Output directory for generated files
- `--seed`: Random seed for reproducibility
- `--disable-pbc`: Flag to disable periodic boundary conditions

### 2. Prepare LAMMPS Input Files

```bash
polydispers lammps \
    --topology-file output/topology.yaml \
    --coordinates output/lj.xyz
```

Options:
- `--topology-file`: Path to the generated topology file
- `--coordinates`: Path to the coordinates file

## Output Files

The generator creates several files in the output directory:
- `chain_*.xyz`: Individual chain coordinates
- `topology.yaml`: System topology information
- `packmol_input.txt`: Packmol input file
- `lj.xyz`: Final packed system coordinates
- `lj.data`: LAMMPS data file
- `lj.in`: LAMMPS input script
- `instructions.sh`: Shell script with next steps

## Workflow

1. Generate the initial system using `polydispers generate`
2. Run Packmol to pack the system
3. Generate LAMMPS input files using `polydispers lammps`
4. Run the LAMMPS simulation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
