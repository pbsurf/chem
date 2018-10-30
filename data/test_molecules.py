# test_molecules.py - geometry and inputs for tests

# MM types are for Amber force fields (OPLS-AA is 63 and 64)
water_tip3p_xyz = """3 TIP3P Water
     1  O      0.000000    0.000000    0.000000     2001     2     3
     2  H     -0.239987    0.000000    0.926627     2002     1
     3  H      0.957200    0.000000    0.000000     2002     1
"""
water = water_tip3p_xyz

water_dimer_tip3p_xyz = """6  TIP3P Water Dimer
     1  O      0.000000    0.000000    0.000000     2001     2     3
     2  H     -0.239987    0.000000    0.926627     2002     1
     3  H      0.957200    0.000000    0.000000     2002     1
     4  O     -0.506021    0.000000    2.700023     2001     5     6
     5  H     -0.817909    0.756950    3.195991     2002     4
     6  H     -0.817909   -0.756950    3.195991     2002     4
"""

C2H3F_noFF_xyz = """6  C2H3F
     1  C     -0.061684    0.673790    0.000000    0     2     3     4
     2  C     -0.061684   -0.726210    0.000000    0     1     5     6
     3  F      1.174443    1.331050    0.000000    0     1
     4  H     -0.927709    1.173790    0.000000    0     1
     5  H     -0.927709   -1.226210    0.000000    0     2
     6  H      0.804342   -1.226210    0.000000    0     2
"""
C2H3F = C2H3F_noFF_xyz

# TODO: I think old (Tinker 4.2) OPLS-AA corresponds to SMOOTH-AA in recent versions of Tinker (verified with `diff`)
ethanol_old_oplsaa_xyz = """9  Ethanol
     1  C     -1.21044700   -0.24241400   -0.02178400    77     2     4     5     6
     2  C      0.07852300    0.56030800    0.04748400    96     1     3     7     8
     3  O      1.23676000   -0.26099500   -0.11099300    93     2     9
     4  H     -1.25533200   -0.79514000   -0.96272300    82     1
     5  H     -2.08478500    0.41216300    0.04607500    82     1
     6  H     -1.26447600   -0.96121000    0.80255600    82     1
     7  H      0.12738300    1.12979300    0.98661900    95     2
     8  H      0.12688100    1.27874200   -0.77385600    95     2
     9  H      1.24779000   -0.88375500    0.63506800    94     3
"""
ethanol = ethanol_old_oplsaa_xyz


# of course we can write some simple code to assemble GAMESS inp
gamess_RHF_6_311G_EandG_Boys_inp = \
""" $CONTRL SCFTYP=RHF RUNTYP=GRADIENT ICHARG=0 MULT=1 NPRINT=3 LOCAL=BOYS MAXIT=100 $END
 $SYSTEM MWORDS=128 $END
 $BASIS GBASIS=N311 NGAUSS=6 NDFUNC=1 NPFUNC=1 $END
 $GUESS GUESS=HCORE $END
 $SCF DIIS=.TRUE. ETHRSH=1.0 $END
 $DATA
"""
