import os
import shutil
import subprocess
import sys
from typing import Optional

import click
import numpy as np

from polydispers.core import (
    GeneratedSystem,
    LammpsFiles,
    generate_polymer_files,
    prepare_lammps_files,
)
from polydispers.input_config import load_config


def check_command(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def find_lammps_executable() -> Optional[str]:
    """Find available LAMMPS executable in PATH."""
    lammps_variants = ["lmp", "lmp_serial", "lmp_mpi"]
    for variant in lammps_variants:
        if check_command(variant):
            return variant
    return None


def check_requirements():
    """Check if required external tools are available."""
    missing_tools = []

    # Check packmol
    if not check_command("packmol"):
        missing_tools.append("packmol")

    # Check LAMMPS
    if not find_lammps_executable():
        missing_tools.append("LAMMPS (lmp, lmp_serial, or lmp_mpi)")

    if missing_tools:
        click.echo("Error: Missing required external tools:", err=True)
        for tool in missing_tools:
            click.echo(f"  - {tool}", err=True)
        click.echo("\nPlease install these tools and make sure they are in your PATH.", err=True)
        sys.exit(1)


@click.group()
def cli():
    """Generate and prepare polymer systems for LAMMPS simulation."""
    pass


def calculate_sz_parameters(target_chain_length, pdi):
    """Calculate Schulz-Zimm distribution parameters."""
    if pdi == 1.0:
        return None, None  # Return None for monodisperse case
    k = 1.0 / (pdi - 1.0)  # shape parameter
    theta = target_chain_length / (k + 1)  # scale parameter
    return k, theta


def print_distribution_statistics(chain_lengths, config):
    target_chain_length = np.mean(chain_lengths)
    k, theta = calculate_sz_parameters(target_chain_length, config.pdi)

    print("\nDistribution statistics:")
    print(f"Number of chains: {len(chain_lengths)}")
    print(f"Mean chain length: {np.mean(chain_lengths):.2f}")
    print(f"PDI: {config.pdi:.2f}")

    # Only print Schulz-Zimm parameters if not monodisperse
    if config.pdi != 1.0:
        print(f"Schulz-Zimm k parameter: {k:.2f}")
        print(f"Schulz-Zimm theta parameter: {theta:.2f}")


@cli.command()
@click.option("--config", type=str, default="input_config.yaml", help="Path to the input configuration file.")
def generate(config):
    """Generate a polymer system with specified parameters."""
    config = load_config(config)

    print("Polydispers Generator, version 0.1.0")
    print("-" * 100)
    print(f"Number of chains: {config.num_chains}")
    print(f"Number-average molecular weight: {config.mn}")
    print(f"Polydispersity index: {config.pdi}")
    print(f"Bond length: {config.polymer.bond_length}")
    print(f"Bead radius: {config.polymer.bead_radius}")
    print(f"Box size: {config.box_size}")
    print("-" * 100)
    print("Generating polymer system...")

    result = generate_polymer_files(config)

    # Get chain lengths and calculate statistics
    chain_lengths = np.array(result.chain_lengths)
    print_distribution_statistics(chain_lengths, config)

    print(f"\nTopology file written to {result.topology_file}")
    print(f"Packmol input file written to {result.packmol_input_file}\n")

    # Next steps
    print("-" * 100)
    print("Next steps:")
    print("-" * 100)
    print(f"1. Run packmol with input file {result.packmol_input_file}\n")
    print(f"packmol < {result.packmol_input_file}\n")
    print("2. Prepare LAMMPS input files for the generated polymer system.\n")
    print((f"polydispers lammps --topology-file {result.topology_file} " f"--coordinates {result.output_dir}/lj.xyz\n"))
    print(f"3. Run lammps with data file {result.output_dir}/lj.data\n")
    print(f"lmp -in {result.output_dir}/lj.data\n")

    print(f"Instructions written to {result.instructions_file}")
    print(f"You can now run the script {result.instructions_file} to prepare the system.\n")


@cli.command()
@click.option("--topology-file", type=str, required=True, help="Path to the topology file.")
@click.option("--coordinates", type=str, required=True, help="Path to the coordinates file.")
def lammps(topology_file, coordinates):
    """Prepare LAMMPS input files for the generated polymer system."""
    result = prepare_lammps_files(topology_file, coordinates)
    click.echo(f"LAMMPS data file written to {result.data_file}")
    click.echo(f"LAMMPS input file written to {result.input_file}")


@cli.command()
@click.option("--config", type=str, default="input_config.yaml", help="Path to the input configuration file.")
def flow(config):
    """Interactive workflow from generation to LAMMPS simulation."""

    def run_command(cmd, cwd=None):
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=cwd)
            return True
        except subprocess.CalledProcessError as e:
            click.echo(f"Error running command: {e}", err=True)
            if click.confirm("Command failed. Continue anyway?", default=False):
                return False
            sys.exit(1)

    def check_file_exists(filepath, description):
        """Check if file exists and ask to skip if it does."""
        if os.path.exists(filepath):
            click.echo(f"Found existing {description} at {filepath}")
            return click.confirm("Skip this step?", default=True)
        return False

    # Check external tools before starting
    check_requirements()

    # Find LAMMPS executable
    lammps_exe = find_lammps_executable()
    if lammps_exe != "lmp":  # If not the default
        click.echo(f"Found LAMMPS executable: {lammps_exe}")
        if not click.confirm(f"Use {lammps_exe}?", default=True):
            available_variants = [exe for exe in ["lmp", "lmp_serial", "lmp_mpi"] if check_command(exe)]
            if not available_variants:
                click.echo("No other LAMMPS variants found in PATH.", err=True)
                sys.exit(1)
            lammps_exe = click.prompt(
                "Choose LAMMPS executable", type=click.Choice(available_variants), default=available_variants[0]
            )

    result = None
    lammps_result = None

    # Step 1: Generate polymer system
    if click.confirm("Generate polymer system?", default=True):
        click.echo("Generating polymer system...")
        config = load_config(config)
        result = generate_polymer_files(config)
        # Print distribution statistics
        chain_lengths = np.array(result.chain_lengths)
        print_distribution_statistics(chain_lengths, config)
        click.echo(f"Polymer system generated in {result.output_dir}")
    else:
        # If skipping generation, need to load existing config and find files
        config = load_config(config)
        output_dir = f"{config.output_dir}/chains_{config.num_chains}_Mn{config.mn}_PDI{config.pdi}"
        result = GeneratedSystem(
            output_dir=output_dir,
            topology_file=f"{output_dir}/topology.yaml",
            packmol_input_file=f"{output_dir}/packmol_input.txt",
            chain_files=[f"{output_dir}/chain_{i}.xyz" for i in range(config.num_chains)],
            instructions_file=f"{output_dir}/instructions.sh",
            chain_lengths=[],  # Empty list since we don't have the chain lengths when skipping generation
        )

    # Step 2: Run packmol
    packed_xyz = f"{result.output_dir}/lj.xyz"
    if not check_file_exists(packed_xyz, "packed system"):
        if click.confirm("Run packmol to pack the polymer chains?", default=True):
            click.echo("Running packmol...")
            success = run_command(f"packmol < {result.packmol_input_file}")
            if success:
                click.echo("Packmol completed successfully")

    # Step 3: Prepare LAMMPS files
    lammps_data = f"{result.output_dir}/lj.data"
    lammps_input = f"{result.output_dir}/lj.in"
    if not check_file_exists(lammps_data, "LAMMPS data file"):
        if click.confirm("Prepare LAMMPS input files?", default=True):
            click.echo("Preparing LAMMPS files...")
            lammps_result = prepare_lammps_files(result.topology_file, packed_xyz)
            click.echo(f"LAMMPS files prepared: {lammps_result.data_file}")
    else:
        lammps_result = LammpsFiles(data_file=lammps_data, input_file=lammps_input)

    # Step 4: Run LAMMPS
    thermo_file = f"{result.output_dir}/thermo.dat"
    if not check_file_exists(thermo_file, "LAMMPS output"):
        if click.confirm("Run LAMMPS simulation?", default=True):
            click.echo("Running LAMMPS simulation...")
            run_command(f"{lammps_exe} -in {lammps_result.input_file}")

    click.echo("Workflow completed!")


if __name__ == "__main__":
    cli()
