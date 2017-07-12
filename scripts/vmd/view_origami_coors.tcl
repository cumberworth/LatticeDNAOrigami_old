# Requires that the structure and coordinates be loaded before sourcing
# Requires the variable filebase has been set (for loading states and ores)

# Load data, fill attributes, setup callbacks
set states [load_matrix_as_lists $filebase.states]
color_states $states
set ores_raw [load_matrix_as_lists $filebase.ores]
set ores [unpack_ores $ores_raw]
set num_scaffold_domains [calc_num_scaffold_domains]
trace variable vmd_frame(0) w draw_graphics_trace
trace variable vmd_frame(0) w update_radii_trace

# Setup representation
mol delrep 0 0
mol default style vdw
mol default color user
mol rep vdw
mol addrep top
animate goto start
animate goto end
mol modcolor 0 0 user
animate goto start

# For some reason if I put this earlier it won't load the script properly
display orthographic
