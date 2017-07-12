# Functions for visualizing lattice DNA origami simulations
#
# Note this assumes throughout all staples are two domains

# Color scheme 
set colors(scaffold_domain) 10
set colors(staple_domain) 0
set colors(bound_domain) 0
set colors(misbound_domain) 0
set colors(scaffold_next_domain) 0
set colors(staple_next_domain) 0
set colors(scaffold_ore) 0
set colors(staple_ore) 0

proc load_matrix_as_lists {filename} {
    # Load a matrix as a list of lists
    set f [open $filename r]
    set raw [read $f]
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
    set numatoms [molinfo top get numatoms]
    set num_scaffolds 0
    for {set atom_i 0} {$atom_i != $numatoms} {incr atom_i} {
        set atom [atomselect top "index $atom_i"]
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
    set numatoms [molinfo top get numatoms]
    set num_staple_domains 0
    for {set i $num_scaffold_domains} {$i != $numatoms} {incr i} {
        set state [lindex [lindex $states $frame] $i]
        if {$state != 0} {
            incr num_staple_domains
        } else {
            break
        }
    }

    return [expr $num_staple_domains / 2]
}

proc color_states {states} {
    # Fill the user2 attribute for all atoms at all frames
    global colors
    set numatoms [molinfo top get numatoms]
    set numframes [molinfo top get numframes]
    for {set frame 0} {$frame != $numframes} {incr frame} {
        for {set atom_i 0} {$atom_i != $numatoms} {incr atom_i} {
            set atom [atomselect top "index $atom_i" frame $frame]
            set state [lindex [lindex $states $frame] $atom_i]

            # Unbound domains
            if {$state == 1} {
                set type [$atom get type]
                if {$type == "scaffold"} {
                    $atom set user2 $colors(scaffold_domain)
                } elseif {$type == "staple"} {
                    $atom set user2 $colors(staple_domain)
                }

            # Fully bound domains
            } elseif {$state == 2} {
                $atom set user2 $colors(bound_domain)

            # Misbound domains
            } elseif {$state == 3} {
                $atom set user2 $colors(misbound_domain)
            }
        }
    }
}

# Not sure how to pass arguments properly with callbacks, so just use globals
# It's what they do in the examples in the VMD docs

proc update_radii {} {
    # Set undefined domains' radii to 0 for current frame
    global states
    set frame [molinfo top get frame]
    set numatoms [molinfo top get numatoms]
    for {set atom_i 0} {$atom_i != $numatoms} {incr atom_i} {
        set atom [atomselect top "index $atom_i" frame $frame]
        set state [lindex [lindex $states $frame] $atom_i]
        if {$state == 0} {
            $atom set radius 0
        } else {
            $atom set radius 0.25
        }
    }
}

proc draw_3d_vector {origin vector color} {
    # Draw vector from origin
    graphics top color $color
    set end [vecadd $origin $vector]
    set middle [vecadd $origin [vecscale 0.8 [vecsub $end $origin]]]
    graphics top cylinder $origin $middle radius 0.05 resolution 10
    graphics top cone $middle $end radius 0.15 resolution 10
}

proc draw_next_domain_vectors {} {
    # Calculate and draw next domain vectors for current frame
    # Must clear previous first
    global colors
    global num_scaffold_domains
    set frame [molinfo top get frame]
    set num_staples [calc_num_staples $frame]

    # Draw scaffold vectors
    set d1 [atomselect top "index 0" frame $frame]
    set d1_coors [lindex [$d1 get {x y z}] 0]
    for {set i 0} {$i != $num_scaffold_domains - 1} {incr i} {
        set d2_i [expr $i + 1]
        set d2 [atomselect top "index $d2_i" frame $frame]
        set d2_coors [lindex [$d2 get {x y z}] 0]
        set diff [vecsub $d2_coors $d1_coors]
        set vector [vecscale $diff 0.75]
        draw_3d_vector $d1_coors $vector $colors(scaffold_next_domain)
        set d1_coors $d2_coors
    }

    # Draw staple vectors
    for {set i 0} {$i != $num_staples} {incr i} {
        set d1_i [expr 2*$i + $num_scaffold_domains]
        set d2_i [expr $d1_i + 1]
        set d1 [atomselect top "index $d1_i" frame $frame]
        set d1_coors [lindex [$d1 get {x y z}] 0]
        set d2 [atomselect top "index $d2_i" frame $frame]
        set d2_coors [lindex [$d2 get {x y z}] 0]
        set diff [vecsub $d2_coors $d1_coors]
        set vector [vecscale $diff 0.75]
        draw_3d_vector $d1_coors $vector $colors(staple_next_domain)
    }
}

proc draw_ore_vectors {} {
    # Draw orientation vectors for current frame
    # Must clear previous first
    global colors
    global ores
    global num_scaffold_domains
    set frame [molinfo top get frame]
    set num_staples [calc_num_staples $frame]

    # Draw scaffold vectors
    for {set i 0} {$i != $num_scaffold_domains} {incr i} {
        set d [atomselect top "index $i" frame $frame]
        set d_coors [lindex [$d get {x y z}] 0]
        set d_ore [lindex [lindex $ores $frame] $i]
        set vector [vecscale $d_ore 0.5]
        draw_3d_vector $d_coors $vector $colors(scaffold_ore)
    }

    # Draw staple vectors
    set num_staple_domains [expr 2*$num_staples]
    set num_domains [expr $num_scaffold_domains + $num_staple_domains]
    for {set i $num_scaffold_domains} {$i != $num_domains} {incr i} {
        set d [atomselect top "index $i" frame $frame]
        set d_coors [lindex [$d get {x y z}] 0]
        set d_ore [lindex [lindex $ores $frame] $i]
        set vector [vecscale $d_ore 0.5]
        draw_3d_vector $d_coors $vector $colors(staple_ore)
    }
}

proc draw_all_vectors {} {
    # Clear all previous graphics and draw vectors for current frame
    graphics top delete all
    draw_next_domain_vectors
    draw_ore_vectors
}

proc update_radii_trace {args} {
    # For vmd callback
    update_radii
}

proc draw_graphics_trace {args} {
    # Have to have all graphics stuff in one callback because of deletion
    draw_all_vectors
}
