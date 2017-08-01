# Functions for visualizing lattice DNA origami simulations
#
# origami must be set to the relevant mol id
# Note this assumes throughout all staples are two domains

# Color scheme and labels
set labels(scaffold_domain) "Unbound scaffold domain"
set colors(scaffold_domain) 10
set labels(staple_domain) "Unbound staple domain"
set colors(staple_domain) 31
set labels(bound_domain) "Bound domain"
set colors(bound_domain) 19
set labels(misbound_domain) "Misbound domain"
set colors(misbound_domain) 8
set labels(scaffold_next_domain) "Scaffold next domain vector"
set colors(scaffold_next_domain) 10
set labels(staple_next_domain) "Staple next domain vector"
set colors(staple_next_domain) 31
set labels(scaffold_ore) "Scaffold orientation vector"
set colors(scaffold_ore) 29
set labels(staple_ore) "Staple orientation vector"
set colors(staple_ore) 4

proc create_legend {} {
    # Create legend
    global colors
    global labels
    global legend
    if {[info exists legend]} {
        mol delrep all $legend
    }
    set legend [mol new]
    mol fix $legend
    set x 0.5
    set cur_y 1.3
    set incr 0.1
    foreach index {scaffold_domain staple_domain bound_domain misbound_domain \
            scaffold_next_domain staple_next_domain scaffold_ore staple_ore} {
        graphics $legend color $colors($index)
        graphics $legend text "$x $cur_y 0" $labels($index) size 1
        set cur_y [expr $cur_y - $incr]
    }
}

proc load_matrix_as_lists {filename} {
    # Load a matrix as a list of lists
    set f [open $filename r]
    set raw [read $f]
    close $f
    set lines [split $raw "\n"]
    set last_index [expr [llength $lines] - 1]
    set lines [lreplace $lines $last_index $last_index]

    return $lines
}

proc unpack_ores {raw_mat} {
    # Take a list of lists matrix and package the ores into a further list
    set ores {}
    foreach step $raw_mat {
        set step_ores {}
        for {set i 0} {$i != [llength $step]} {incr i 3} {
            set ore {}
            for {set j $i} {$j != [expr $i + 3]} {incr j} {
                lappend ore [lindex $step $j]
            }
            lappend step_ores $ore
        }
        lappend ores $step_ores
    }

    return $ores
}

proc calc_num_scaffold_domains {} {
    # Calculate number of scaffold domains
    global origami
    set numatoms [molinfo $origami get numatoms]
    set num_scaffolds 0
    for {set atom_i 0} {$atom_i != $numatoms} {incr atom_i} {
        set atom [atomselect $origami "index $atom_i"]
        if {[$atom get type] != "staple"} {
            incr num_scaffolds
        } else {
            break
        }
    }

    return $num_scaffolds
}

proc calc_num_staples {frame} {
    # Calculate number of staples for given frame
    global num_scaffold_domains
    global states
    global origami
    set numatoms [molinfo $origami get numatoms]
    set num_staple_domains 0
    for {set i $num_scaffold_domains} {$i != $numatoms} {incr i} {
        set state [lindex [lindex $states $frame] $i]
        if {$state != -1} {
            incr num_staple_domains
        } else {
            break
        }
    }

    return [expr $num_staple_domains / 2]
}

proc create_domain_reps {} {
    global origami
    mol rep vdw
    set numatoms [molinfo $origami get numatoms]
    for {set i 0} {$i != $numatoms} {incr i} {
        mol addrep $origami
        mol modselect $i $origami "index $i"
    }
}

proc update_colors {} {
    # Color the domains based on their binding states (and type)
    global colors
    global states
    global origami
    set numatoms [molinfo $origami get numatoms]
    set frame [molinfo $origami get frame]
    for {set i 0} {$i != $numatoms} {incr i} {
        set state [lindex [lindex $states $frame] $i]
        set atom [atomselect $origami "index $i" frame $frame]

        # Unbound domains
        if {$state == 1} {
            set type [$atom get type]
            if {$type == "scaffold"} {
                mol modcolor $i $origami ColorID $colors(scaffold_domain)
            } elseif {$type == "staple"} {
                mol modcolor $i $origami ColorID $colors(staple_domain)
            }

        # Fully bound domains
        } elseif {$state == 2} {
            mol modcolor $i $origami ColorID $colors(bound_domain)

        # Misbound domains
        } elseif {$state == 3} {
            mol modcolor $i $origami ColorID $colors(misbound_domain)
        }
    }
}

# Not sure how to pass arguments properly with callbacks, so just use globals
# It's what they do in the examples in the VMD docs

proc update_radii {} {
    # Set undefined domains' radii to 0 for current frame
    global states
    global origami
    set frame [molinfo $origami get frame]
    set numatoms [molinfo $origami get numatoms]
    for {set atom_i 0} {$atom_i != $numatoms} {incr atom_i} {
        set atom [atomselect $origami "index $atom_i" frame $frame]
        set state [lindex [lindex $states $frame] $atom_i]
        if {$state == 0 || $state == -1} {
            $atom set radius 0
        } else {
            $atom set radius 0.25
        }
    }
}

proc draw_3d_vector {origin vector color} {
    # Draw vector from origin
    global origami
    graphics $origami color $color
    set end [vecadd $origin $vector]
    set middle [vecadd $origin [vecscale 0.8 [vecsub $end $origin]]]
    graphics $origami cylinder $origin $middle radius 0.05 resolution 10
    graphics $origami cone $middle $end radius 0.15 resolution 10
}

proc draw_next_domain_vectors {} {
    # Calculate and draw next domain vectors for current frame
    # Must clear previous first
    global colors
    global num_scaffold_domains
    global origami
    global states
    set frame [molinfo $origami get frame]
    set num_staples [calc_num_staples $frame]

    # Draw scaffold vectors
    set d1 [atomselect $origami "index 0" frame $frame]
    set d1_coors [lindex [$d1 get {x y z}] 0]
    for {set i 0} {$i != $num_scaffold_domains - 1} {incr i} {
        set d2_i [expr $i + 1]
        set d2 [atomselect $origami "index $d2_i" frame $frame]
        set d2_coors [lindex [$d2 get {x y z}] 0]
        set diff [vecsub $d2_coors $d1_coors]
        set vector [vecscale $diff 0.75]
        set state1 [lindex [lindex $states $frame] $i]
        set state2 [lindex [lindex $states $frame] $d2_i]
        if {$state1 != 0 && $state2 != 0} {
            draw_3d_vector $d1_coors $vector $colors(scaffold_next_domain)
        }
        set d1_coors $d2_coors
    }

    # Draw staple vectors
    for {set i 0} {$i != $num_staples} {incr i} {
        set d1_i [expr 2*$i + $num_scaffold_domains]
        set d2_i [expr $d1_i + 1]
        set d1 [atomselect $origami "index $d1_i" frame $frame]
        set d1_coors [lindex [$d1 get {x y z}] 0]
        set d2 [atomselect $origami "index $d2_i" frame $frame]
        set d2_coors [lindex [$d2 get {x y z}] 0]
        set diff [vecsub $d2_coors $d1_coors]
        set vector [vecscale $diff 0.75]
        set state1 [lindex [lindex $states $frame] $d1_i]
        set state2 [lindex [lindex $states $frame] $d2_i]
        if {$state1 != 0 && $state2 != 0} {
            draw_3d_vector $d1_coors $vector $colors(staple_next_domain)
        }
    }
}

proc draw_ore_vectors {} {
    # Draw orientation vectors for current frame
    # Must clear previous first
    global colors
    global ores
    global num_scaffold_domains
    global origami
    global states
    set frame [molinfo $origami get frame]
    set num_staples [calc_num_staples $frame]

    # Draw scaffold vectors
    for {set i 0} {$i != $num_scaffold_domains} {incr i} {
        set state [lindex [lindex $states $frame] $i]
        if {$state == 0} {
            continue
        }
        set d [atomselect $origami "index $i" frame $frame]
        set d_coors [lindex [$d get {x y z}] 0]
        set d_ore [lindex [lindex $ores $frame] $i]
        set vector [vecscale $d_ore 0.5]
        draw_3d_vector $d_coors $vector $colors(scaffold_ore)
    }

    # Draw staple vectors
    set num_staple_domains [expr 2*$num_staples]
    set num_domains [expr $num_scaffold_domains + $num_staple_domains]
    for {set i $num_scaffold_domains} {$i != $num_domains} {incr i} {
        set state [lindex [lindex $states $frame] $i]
        if {$state == 0} {
            continue
        }
        set d [atomselect $origami "index $i" frame $frame]
        set d_coors [lindex [$d get {x y z}] 0]
        set d_ore [lindex [lindex $ores $frame] $i]
        set vector [vecscale $d_ore 0.5]
        draw_3d_vector $d_coors $vector $colors(staple_ore)
    }
}

proc update_frame {} {
    # Load new configuration and delete previous
    global origami
    global filebase
    global states
    global ores
    animate delete beg 0 $origami

    # Save visulation state
    foreach mol [molinfo list] {
        set viewpoints($mol) [molinfo $mol get {
            center_matrix rotate_matrix scale_matrix global_matrix}]
    }
    animate read vcf $filebase.vcf waitfor all $origami
    animate read vcf $filebase.vcf waitfor all $origami

    # Return to previous visulation state
    foreach mol [molinfo list] {
        molinfo $mol set {center_matrix rotate_matrix scale_matrix
            global_matrix} $viewpoints($mol)
    }

    set states [load_matrix_as_lists $filebase.states]
    lappend states [lindex $states 0]
    set ores_raw [load_matrix_as_lists $filebase.ores]
    set ores [unpack_ores $ores_raw]
    lappend ores [lindex $ores 0]

    graphics $origami delete all
    update_colors
    update_radii
    draw_next_domain_vectors
    draw_ore_vectors
}

proc update_frame_trace {args} {
    update_frame
}

proc draw_all_vectors {} {
    # Clear all previous graphics and draw vectors for current frame
    global origami
    graphics $origami delete all
    draw_next_domain_vectors
    draw_ore_vectors
}

proc update_colors_trace {args} {
    update_colors
}

proc update_radii_trace {args} {
    update_radii
}

proc draw_graphics_trace {args} {
    # Have to have all graphics stuff in one callback because of deletion
    draw_all_vectors
}
