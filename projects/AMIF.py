from chem.molecule import *
from chem.io import *
from chem.vis.chemvis import *


ARF_xyz = """6  molden generated tinker .xyz (mm3 param.)
 1  C    -0.074912   -0.221230    0.018657      3  2  5  6
 2  N     0.110525    0.201884    1.310047      9  1  3  4
 3  H     0.966905   -0.092052    1.796416     28  2
 4  H    -0.596220    0.795644    1.763857     28  2
 5  O     0.682820   -0.916477   -0.611296      7  1
 6  H    -1.028766    0.140260   -0.440185      5  1
"""

#HETATM15014  N   ARF A1335      -2.192  -8.398  36.932  1.00 40.34           N
#HETATM15015  C   ARF A1335      -3.032  -9.153  37.491  1.00 40.81           C
#HETATM15016  O   ARF A1335      -4.060  -8.629  38.150  1.00 36.70           O

crystal_pdb = load_molecule('2E2L.pdb')
crystal_mol = load_molecule('2E2L.xyz', charges='generate')
amif_all = copy_residues(crystal_mol, crystal_pdb)
amif = amif_all.extract_atoms('/A/*/*')  # 2E2L is a trimer, just keep one chain

arf_pdb = crystal_pdb.select(pdb='A 1335 N,C,O')
arf = load_molecule(ARF_xyz)
align_mol(arf, crystal_pdb.r[arf_pdb], sel=[1,0,4], sort=None)
arf_atoms = amif.append_atoms(arf.atoms, residue="LIG")

import pdb; pdb.set_trace()

from scipy.spatial.ckdtree import cKDTree
load_tinker_params(amif, '2E2L.xyz', charges=True, vdw=True)
mol = amif
kd = cKDTree(mol.r)
for aidx in [0, 1, 4]:
  qidx = arf_atoms[aidx]  #qmatoms[1]
  B1 = amif.get_bonded(qidx)
  exclude = frozenset(flatten(B1 + [amif.get_bonded(ii) for ii in B1]))
  qatom = mol.atoms[qidx]
  dists, locs = kd.query(qatom.r, k=12)
  for ii in locs:
    if ii not in exclude:
      r = mol.bond((qidx, ii))
      r_lj = 0.5*(qatom.lj_r0 + mol.atoms[ii].lj_r0)
      print("%s (%d) - %s (%d): dist = %f = %f * vdW dist" % (qatom.name, qidx, mol.atoms[ii].name, ii, r, r/r_lj))


vis = Chemvis(Mol(amif, [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='sidechain and any(mol.dist(a, 5356) < 8 for a in resatoms)'), VisGeom(style='licorice', radius=0.5, sel=arf_atoms) ]), fog=True).run()

# extract active site
theo1_sel = amif.select('any(mol.dist(a, %d) < 8 for a in resatoms)' % arf_atoms[0])
theo1 = amif.extract_atoms(theo1_sel)

