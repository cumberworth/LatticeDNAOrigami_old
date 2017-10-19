# Requires the variables filebase and libdir have been set

source $libdir/liborigami.tcl

set origami [mol new $filebase.vsf]
set ores_raw [load_matrix_as_lists $filebase.ores]
set ores [unpack_ores $ores_raw]
set num_scaffold_domains [calc_num_scaffold_domains]
mol delrep 0 0
create_domain_reps
mol addfile $filebase.vcf type vcf waitfor all
create_legend
axes location off
display projection orthographic
mol top $origami
set states [load_matrix_as_lists $filebase.states]
trace variable vmd_frame(0) w update_colors_trace
trace variable vmd_frame(0) w draw_graphics_trace
trace variable vmd_frame(0) w update_radii_trace

animate goto start
