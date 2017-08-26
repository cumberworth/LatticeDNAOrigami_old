# Shitty pipe

set vmd_file_dir [lindex $argv 0]
set filebase [lindex $argv 1]

source $vmd_file_dir/liborigami.tcl

#after 10000
set origami [mol new $filebase.vsf]
set num_scaffold_domains [calc_num_scaffold_domains]
mol delrep 0 0
create_domain_reps
animate read vcf $filebase.vcf waitfor all $origami
animate read vcf $filebase.vcf waitfor all $origami
create_legend
axes location off
display projection orthographic
mol top $origami

trace variable vmd_frame(0) w update_frame_trace

animate speed 0.1
animate forward
