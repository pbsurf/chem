from chem.molecule import *


if 0: ##__name__ == "__main__":
#~   mol = molecule("water");
#~   bonds, angles, diheds = mol.get_internals();
#~   for b in bonds:
#~     print b, mol.bond(b);
#~   for b in angles:
#~     print b, mol.angle(b);
#~   for b in diheds:
#~     print b, mol.dihedral(b);

  # check gradient calculations
  r = np.random.rand(2,3);
  bond0, grad = calc_dist(r, grad=True);
  dr = np.random.rand(2,3)*0.01;
  bond1 = calc_dist(r + dr);
  print "bond0: {}; bond1: {}; err vs. dr*grad: {}".format(bond0, bond1, bond1 - (bond0 + np.sum(grad*dr)));

  r = np.random.rand(3,3);
  angle0, grad = calc_angle(r, grad=True);
  dr = np.random.rand(3,3)*0.01;
  angle1 = calc_angle(r + dr);
  print "angle0: {}; angle1: {}; err vs. dr*grad: {}".format(angle0, angle1,
      angle1 - (angle0 + np.sum(grad*dr)));

  r = np.random.rand(4,3);
  dihed0, grad = calc_dihedral(r, grad=True);
  dr = np.random.rand(4,3)*0.01;
  dihed1 = calc_dihedral(r + dr);
  print "dihed0: {}; dihed1: {}; err vs. dr*grad: {}".format(dihed0, dihed1,
      dihed1 - (dihed0 + np.sum(grad*dr)));

  r = np.random.rand(4,3);
  dihed0, grad = cos_dihedral(r, grad=True);
  dr = np.random.rand(4,3)*0.01;
  dihed1 = cos_dihedral(r + dr);
  print "cosdihed0: {}; cosdihed1: {}; err vs. dr*grad: {}; sum(grad): {}".format(dihed0, dihed1,
      dihed1 - (dihed0 + np.sum(grad*dr)), np.sum(grad));


## Object for using lambda with Molecule.select()
## ... this is slower than fn produced by decode_atom_sel_str()!!!
# Examples:
# - VisGeom(... sel=lambda a: a.resnum in nearres)
# - freeze = mol.select(lambda a: a.protein and (a.extbackbone or a.resnum not in nearres))
# - kd = cKDTree(mol.r[mol.select(ligin)]); nearres8 = mol.select(lambda a: a.sidechain and a.within(8.0, kd))

# other ideas:
# - mol.atomsel returns generator for AtomSel objects?
# - just add mol and idx attrs to Atom so it can be used directly?  huge mess when extracting and removing?
# - use regex to replace variable names in select string with "a.<name>" and use AtomSel
# Notes:
# - any additional args to select() interfere with passing select strings/lambdas to other fns

# add to Molecule.atomsres():
#if callable(sel):
#  atomsel = AtomSel(self)
#  return [ii for ii,res in enumerate(mol.residues) if any(sel(atomsel.update(jj)) for jj in res.atoms)]

class AtomSel:
  attrlambdas = dict(
    backbone=lambda a: a._protein() and a.atom.name in PDB_BACKBONE,
    extbackbone=lambda a: a._protein() and a.atom.name in PDB_EXTBACKBONE,
    sidechain=lambda a: a._protein() and a.atom.name not in PDB_NOTSIDECHAIN,
    polymer=lambda a: a.res().name in PDB_STANDARD,
    protein=lambda a: a._protein(),
    water=lambda a: a.res().name == 'HOH',
    pdb_resid=lambda a: str(a.res().pdb_num),
    pdb_resnum=lambda a: int(a.res().pdb_num),
    residue=lambda a: a.res()
  )

  def __init__(self, mol):
    self.mol = mol

  # or do this instead of recreating obj:
  def _update(self, idx, atom):
    self.idx, self.atom = idx, atom
    return self

  def within(self, d, other):
    if hasattr(other, 'query'):  # support kd-tree
      return other.query([self.atom.r], distance_upper_bound=d)[0][0] < d
    other = [other] if type(other) is int else other
    return any(calc_dist(self.atom.r, self.mol.atoms[b].r) <= d for b in other)

  def res(self):
    return self.mol.residues[self.atom.resnum]

  def _protein(self):
    return self.res().name in PDB_PROTEIN

  def __getattr__(self, attr):
    val = getattr(self.atom, attr, self)  # self is used to indicate missing attribute (faster than try/except)
    if val is not self: return val
    val = self.attrlambdas.get(attr, self)
    if val is not self: return val(self)
    return getattr(self.res(), attr[3:] if attr[:3] == 'res' else attr)  # let this generate exception


def fast_select(mol, sel):
  if type(sel) is str:
    vars = "name,znuc,resnum,residue,resname,resatoms,chain,pdb_resnum,pdb_resid,polymer,protein,backbone,extbackbone,sidechain,water,within".split(',')
    for var in vars:
      sel = re.sub(r'\b%s\b' % var, 'a.' + var, sel)
    sel = eval('lambda a: ' + sel)
  atomsel = AtomSel(mol)
  return [ ii for ii,atom in enumerate(mol.atoms) if sel(atomsel._update(ii,atom)) ]


mol = mol0_AI

from timeit import timeit

timeit(lambda: fast_select(mol, 'sidechain'), number=100)  # 0.8s
## ... ha ha ha, more like SLOW_SELECT!!!
# timeit(lambda: fast_select(mol, 'znuc == 8'), number=100) is a bit faster, but residue attrs are slower

timeit(lambda: mol.select('sidechain'), number=100)  # 0.5s

timeit(lambda: [ii for ii,a in enumerate(mol.atoms) if mol.residues[a.resnum].name in PDB_PROTEIN and a.name not in PDB_NOTSIDECHAIN ], number=100)  # 0.2s
