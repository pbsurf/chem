## full chorismate mutase reaction path example

from chem.molecule import *
from chem.io import *
from chem.vis.chemvis import *

# https://www.cp2k.org/howto:biochem_qmmm
ligA_xyz = """24  chorismate
  1 C       49.2110    26.9920    85.5530
  2 H       49.1050    28.0330    85.7020
  3 H       48.8320    26.6060    84.6410
  4 C       49.8460    26.1960    86.4430
  5 C       50.0590    24.7460    86.2630
  6 O       50.4810    24.0440    87.2160
  7 O       49.8330    24.2110    85.1490
  8 O       50.3730    26.8140    87.5890
  9 C       51.7280    27.3660    87.5930
 10 H       51.8840    27.9250    88.5040
 11 C       51.8170    28.2920    86.3980
 12 H       51.3990    29.2560    86.5300
 13 C       52.1840    27.8630    85.1670
 14 C       52.8190    26.5610    85.0450
 15 H       53.0370    26.2220    84.0630
 16 C       53.1170    25.8090    86.1190
 17 H       53.5700    24.8590    86.0130
 18 C       52.7610    26.1980    87.5410
 19 H       52.3530    25.3480    88.0710
 20 O       54.0020    26.6280    88.1730
 21 H       54.4570    27.1880    87.5220
 22 C       51.9410    28.6780    83.9640
 23 O       51.9100    29.9280    84.0860
 24 O       51.8570    28.1570    82.8230
"""

ligA = load_molecule(ligA_xyz, center=True)
ligA.set_bonds(guess_bonds(ligA.r, ligA.znuc))

# ligA - TSA chain A (zero-based indices):
# 21 : 13539 (C carboxyl)
# 12 : 13529 (C ring)
# 10 : 13535 (C ring)
# 13 : 13530 (C ring)
# 7 : 13536 (O)
# 4 : 13542 (moving carboxyl C)
# 19 : 13533 (hydroxyl O)

crystal_pdb = load_molecule('2CHT.pdb')
crystal_mol = load_molecule('2CHT.xyz', charges='generate')
cm_all = copy_residues(crystal_mol, crystal_pdb)
cm = cm_all.extract_atoms('/A,B,C/*/*')

align_mol(ligA, crystal_pdb.r[[13539, 13535, 13530]], sel=[21, 10, 13], sort=None)
qmatoms = cm.append_atoms(ligA.atoms, residue="LIG")

# what is the atom density in bulk of protein? around active site (including substrate)?
# compared to, e.g., sum_{atoms i} 4*pi*(covalent radius_i)^3/3

import pdb; pdb.set_trace()

# radial pair distribution fn
from chem.analysis import *
#cm = quick_load('2CHT.pdb').extract_atoms('/A,B,C/*/*')
cmbins, cmrdf = calcRDF(cm)
amif = quick_load('2E2L.pdb').extract_atoms('/A/*/*')
amifbins, amifrdf = calcRDF(amif)
import matplotlib.pyplot as plt
plt.plot(amifbins[:-1], amifrdf, cmbins[:-1], cmrdf)
#plt.xlabel("r (Angstroms)")
#plt.ylabel("probability")
plt.show()


qmmm = QMMM(cm, qmatoms=qmatoms, qm_opts=Bunch(prog='pyscf'), prefix='cm1', savefiles=False)
for ii in qmmm.qmatoms:
  qmmm.mol.atoms[ii].qmbasis = '6-31G*'

pyscf_EandG(cm, qmatoms=qmatoms)


a = 5.0  # 5 x 5 x 5 Ang bins
extents = cm.extents()
range = extents[1] - extents[0]
nbins = np.array(np.ceil(range/a), dtype=np.uint)
counts = np.zeros(nbins)
for atom in cm.atoms:
  counts[ tuple(np.int_((atom.r - extents[0])/a)) ] += 1 if atom.znuc > 1 else 0  # += atom.znuc
density = counts/(a*a*a)
histr = np.histogram(np.ravel(density), bins=np.max(counts)+1)
# max ~0.09 heavy atoms/Ang^3
# water at 1 g/mL: .056 mol/mL -> 0.056*NA/10^24 Ang^3 -> 3.37e22/1e24 -> 3.37e-2 -> 0.03 heavy atoms/Ang^3


vis = Chemvis(Mol(cm, [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='protein'), VisGeom(style='lines', sel='not protein'), VisGeom(style='licorice', radius=0.5, sel=qmatoms) ]), fog=True).run()


# active site
vis = Chemvis(Mol(cm, [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='any(mol.dist(a, 6098) < 8 for a in resatoms)'), VisGeom(style='licorice', radius=0.5, sel=qmatoms) ]), fog=True).run()

