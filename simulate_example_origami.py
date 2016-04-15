#!/usr/env python

"""Run a basic simulation of the example origami system."""

from lattice_origami_domains import *

# Specificy initial configuration by setting input file and step number
input_file = JSONInputFile('simple_loop_linear.json')
step = 0

# Set conditions
temp = 300

# Cation concentration
cation_M = 1

# Setup origami system object
origami_system = OrigamiSystemEight(input_file, step, temp, cation_M)

# Specify moves to be used and associated probabilities
move_settings = {MOVETYPE.EXCHANGE_STAPLE: 0.4,
                 MOVETYPE.REGROW_STAPLE: 0.2,
                 MOVETYPE.REGROW_SCAFFOLD_AND_BOUND_STAPLES: 0.2,
                 MOVETYPE.ROTATE_ORIENTATION_VECTOR: 0.2}

# Specify output file type and name
output_file_name = 'simple_loop.hdf5'
output_file = HDF5OutputFile(output_file_name, origami_system)

# Setup up simulation
sim = GCMCBoundStaplesSimulation(origami_system, move_settings, output_file)

# Run
N = 10000
sim.run(N)
