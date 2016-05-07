#!/usr/env python

"""Run a basic simulation of the example origami system."""

from lattice_origami_domains import *

# Specificy initial configuration by setting input file and step number
#input_file = JSONInputFile('simple_loop_linear.json')
#input_file = JSONInputFile('simple_loop.json')
input_file = JSONInputFile('cyclic_example.json')
step = 0

# Set conditions
temp = 300

# Cation concentration
cation_M = 1

# Setup origami system object
#origami_system = OrigamiSystemEight(input_file, step, temp, cation_M)
origami_system = OrigamiSystemSixteen(input_file, step, temp, cation_M)

# Specify moves to be used and associated probabilities
#move_settings = {MOVETYPE.EXCHANGE_STAPLE: 0.4,
#                 MOVETYPE.REGROW_STAPLE: 0.2,
#                 MOVETYPE.REGROW_SCAFFOLD: 0.2,
#                 MOVETYPE.ROTATE_ORIENTATION_VECTOR: 0.2}

move_settings = {MOVETYPE.CB_CONSERVED_TOPOLOGY: 1}

# Specify output file type and name
#output_file_name = 'simple_loop_replica-0.hdf5'
output_file_name = 'cyclic_example_replica-0.hdf5'
config_write_freq = 1
count_write_freq = 1

output_file = HDF5OutputFile(output_file_name, origami_system,
        config_write_freq=config_write_freq,
        count_write_freq=count_write_freq)

# Setup up simulation
sim = GCMCSimulation(origami_system, move_settings, output_file)

# Run
N = 1000
sim.run(N)
