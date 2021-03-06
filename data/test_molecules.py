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

ethanol_gaff_xyz = """9  molden generated ambfor .xyz
1  c3     -1.208   -0.243   -0.031  901  0.000      2      4      5      6
2  c3      0.089    0.560    0.035  902  0.000      1      3      7      8
3  oh      1.215   -0.315   -0.107  903  0.000      2      9
4  hc     -1.297   -0.796   -0.991  904  0.000      1
5  hc     -2.097    0.419    0.053  905  0.000      1
6  hc     -1.273   -0.988    0.792  906  0.000      1
7  h1      0.160    1.117    0.994  907  0.000      2
8  h1      0.114    1.311   -0.783  908  0.000      2
9  ho      1.298   -0.828    0.686  909  0.000      3
"""

ethane_gaff_xyz = """8  ethane (molden ambfor xyz)
1  c3     -1.208   -0.243   -0.031  801  0.000      2      3      4      5
2  c3      0.089    0.560    0.035  802  0.000      1      6      7      8
3  hc     -1.297   -0.796   -0.991  803  0.000      1
4  hc     -2.097    0.419    0.053  804  0.000      1
5  hc     -1.273   -0.988    0.792  805  0.000      1
6  hc      0.160    1.117    0.994  806  0.000      2
7  hc      0.114    1.311   -0.783  807  0.000      2
8  hc      1.215   -0.315   -0.107  808  0.000      2
"""

# these are both MM3 optimized
ch3cho_xyz = ("7  CH3CHO w/ Amber96 Tinker types, GAFF names  \n"   # MM3 types:
" 1  c     0.216434   -0.239836    0.022589    342  2  3  7   \n"   # 3
" 2  c3    0.041247   -0.042250    1.496848    340  1  4  5  6\n"   # 1
" 3  o     1.273466   -0.143112   -0.555929    343  1         \n"   # 7
" 4  hc   -0.670708    0.790679    1.698814    341  2         \n"   # 5
" 5  hc   -0.360705   -0.966829    1.971194    341  2         \n"   # 5
" 6  hc    1.009413    0.204927    1.989829    341  2         \n"   # 5
" 7  ha   -0.702574   -0.492745   -0.564006    341  1         \n")  # 5

ch2choh_xyz = ("7  CH2CHOH w/ Amber96 Tinker types, GAFF names\n"    # MM3 types:
" 1  c2    0.231311   -0.117468    0.202122    342  2  3  7   \n"    #  2
" 2  c2   -0.336100   -0.188389    1.410112    340  1  4  5   \n"    #  2
" 3  oh    1.465427   -0.622004   -0.056621    343  1  6      \n"    #  6
" 4  ha   -1.343088    0.222649    1.588799    341  2         \n"    #  5
" 5  ha    0.192087   -0.661201    2.253726    341  2         \n"    #  5
" 6  ho    1.994120    0.037261   -0.538351    341  3         \n"    # 73
" 7  ha   -0.308197    0.346592   -0.641855    341  1         \n")   #  5

hcooh_xyz = """5 emin -8.287  AMBFOR/AMBMD generated .xyz (amber/gaff param.)
1  h5     2.07864  -4.95969   2.12292  -23  0.098      2
2  c2     1.66099  -4.11544   1.59679   -4  0.475      1      3      4
3  o      1.86183  -3.93186   0.36790  -49 -0.419      2
4  oh     1.29103  -3.06027   2.31321  -50 -0.353      2      5
5  ho     1.12452  -2.33975   1.68018  -27  0.199      4
"""
