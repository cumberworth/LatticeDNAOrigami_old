# Requires that the structure and coordinates be loaded before sourcing
# Requires the variable filebase has been set (for loading states and ores)

set ores_raw [load_matrix_as_lists $filebase.ores]
set ores [unpack_ores $ores_raw]
set num_scaffold_domains [calc_num_scaffold_domains]
trace variable vmd_frame(0) w update_colors_trace
trace variable vmd_frame(0) w draw_graphics_trace
trace variable vmd_frame(0) w update_radii_trace

create_legend
set states [load_matrix_as_lists $filebase.states]
mol delrep 0 0
create_domain_reps
animate goto start

# For some reason if I put this earlier it won't load the script properly
display projection orthographic
