## create histogram of quantity for a trajectory; plot using multiplot
# atomlist = [list atom1 atom2 atom3 atom4]
# Multiplot: http://www.ks.uiuc.edu/Research/vmd/plugins/multiplot/

# example:  with 500 model PDB from trypsin/benzamidine traj loaded as mol 7,
#  histtraj 7 [list 3224 3225 3226 3227]
# Compare: Proteins 58, 407 (2005); clearly, my simulation
#  was not long enough to sample the symmetric + and - conformations
# ... or it could be because they used a different PDB structure!!!

proc trajhist { molid atomlst } {
  package require multiplot

  set dihedrals [list]
  set n [molinfo $molid get numframes]
  # get dihedral for each frame
  for { set i 0 } { $i < $n } { incr i } {
    lappend dihedrals [measure dihed $atomlst frame $i]
  }
  # generate histogram
  set lmin [min $dihedrals]
  set lmax [max $dihedrals]
  set nbins [expr $n/20]

  set hist [lrepeat [expr 2*$nbins] 0]
  foreach d $dihedrals {
    set idx [expr int(round($nbins*($d-$lmin)/($lmax-$lmin)))]
    if {$idx >= $nbins} then { set idx [expr $nbins-1] }
    lset hist [expr 2*$idx] [expr [lindex $hist [expr 2*$idx]]+1]
    lset hist [expr 2*$idx+1] [expr [lindex $hist [expr 2*$idx+1]]+1]
  }

  # create x axis
  set bins [list]
  for { set jj 0 } { $jj < $nbins } { incr jj } {
    lappend bins [expr $lmin + ($jj/double($nbins))*($lmax-$lmin)]
    lappend bins [expr $lmin + (($jj+1)/double($nbins))*($lmax-$lmin)]
  }

  # this will create and show the plot
  set plothandle [multiplot -x $bins -y $hist -title "Trajectory Histogram" -lines -linewidth 3 -ymin 0 -plot]
  return
}

proc min { x } {
  set minv [lindex $x 0]
  foreach e $x {
    if {$e < $minv} then {
      set minv $e
    }
  }
  return $minv
}

proc max { x } {
  set maxv [lindex $x 0]
  foreach e $x {
    if {$e > $maxv} then {
      set maxv $e
    }
  }
  return $maxv
}

