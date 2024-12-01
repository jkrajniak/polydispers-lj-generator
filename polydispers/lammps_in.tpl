# LAMMPS input script for polymer simulation
units           lj
atom_style      molecular
pair_style      lj/cut 2.5
boundary        p p p

# Set special bonds for FENE potential
special_bonds   lj 0.0 1.0 1.0

read_data       $output_dir/lj.data

# Set mass for atom type 1
mass            1 1.0

# Initialize velocities at T=1.0
velocity        all create 1.0 12345 dist gaussian

# Reset time
reset_timestep  0

# Define pair and bond styles
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0

bond_style      fene
bond_coeff      1 30.0 1.5 1.0 1.0

# Compute energies
compute         pe all pe/atom
compute         pair all pair lj/cut
compute         ebond all pe/atom bond
compute         epair all pe/atom pair
compute         total_pe all reduce sum c_pe
compute         bond_pe all reduce sum c_ebond
compute         pair_pe all reduce sum c_epair

# Output settings with reduced precision
thermo          1000
thermo_style    custom step temp press density c_total_pe vol lx ly lz
thermo_modify   format float %.3f
thermo_modify   format 3 %.6f  # special format for density

# Save detailed thermo data to file with reduced precision
fix             thermo all print 1000 &
                "%d %.3f %.3f %.6f %.2f %.2f %.2f" &
                file $output_dir/thermo.dat &
                title "# Step Temp Press Density TotalPE BondE PairE" &
                screen no

# Trajectory output with reduced precision
dump            1 all custom 1000 $output_dir/traj.lammpstrj id type mol x y z
dump_modify     1 format float %.4f

# First minimize to fix any bad contacts or stretched bonds
minimize        1.0e-4 1.0e-6 1000 10000

# -------- NPT Equilibration with Berendsen --------
timestep        0.005  # Small timestep for stability

# Berendsen thermostat and barostat (separately)
fix mynpt all npt temp 1 1 100 iso 1 1 100

# Run NPT equilibration
run             1000000

# Remove Berendsen fixes
unfix           mynpt

# -------- NVT Equilibration --------
# timestep        0.005
fix             nvt all nvt temp 5.0 5.0 500.0
run             50000
