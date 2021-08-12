import numpy as np
import copy, re
from .basics import *
from .data.elements import ELEMENTS
from .data.pdb_bonds import PDB_STANDARD, PDB_PROTEIN


# To consider:
# 1. Would it be better to store molecule attributes as arrays of arrays, and have __getattr__/__setattr__ for
#  Atom and Residue objects?  They would then need to know their index in Molecule (at least for setattr)
#  - previously supported mol[ii] as short version of mol.atoms[ii], but with residues, this is too ambiguous
# 1. Dealing with `active` in many methods is a bit inelegant ... any alternatives?  Create separate Molecule
#  object for active atoms?
# 1. Should we add methods to Molecule to wrap non-method fns (and try to move even more non-wrapper methods
#   out of Molecule)?
# Possible future work:
# 1. Z-matrix ... we do not need or want z-matrix for link atom opt (?)
#  - see https://github.com/mcocdawc/chemcoord for z-matrix -> cartesian (in general, non-trivial due to
#   accumulation of numerical errors for angles and dihedrals); or just see DLC.toxyz()?
#  - cartesian -> z-matrix is trivial, see QMForge intmodel.py (also CCP1GUI zmatrix.py)
# build z-matrix ... order of atoms can be specified (so we can determine which bond distances will be used)
#  entry: <dist atom> <dist> <angle atom> <angle> <dihed atom> <dihed>
#  atom n: dist to n-1, angle to n-1 dist atom, dihed to n-1 angle atom


## Atom class

class Atom:
  """name (string), znuc: atomic number, r: position vector, mmtype: MM atom type (usually int),
    mmq: MM partial charge, mmconnect: list of atom numbers for bonded atoms
  """
  def __init__(self, name='', znuc=0, r=None, mmtype=0, mmq=0.0, mmconnect=None, resnum=None, **kwargs):
    self.name = name
    self.znuc = znuc
    # don't use np.asarray because we want our own copy of r
    self.r = np.array(r, dtype=np.float64) if r is not None else []
    self.mmtype = mmtype
    self.mmq = mmq
    self.mmconnect = mmconnect if mmconnect is not None else []
    self.resnum = resnum
    for key,val in kwargs.items():
      setattr(self, key, val)

  def __repr__(self):
    attrs = ["%s=%r" % (a, getattr(self,a)) for a in dir(self) \
        if not a.startswith('__') and not callable(getattr(self,a))]
    return "Atom(%s)" % ', '.join(attrs)


## Residue Class

class Residue:
  def __init__(self, name='', atoms=None, pdb_num=None, chain=None, het=False):
    self.name = name
    self.atoms = [] if atoms is None else atoms
    self.pdb_num = pdb_num
    self.chain = chain
    self.het = het

  def __repr__(self):
    attrs = ["%s=%r" % (a, getattr(self,a)) for a in dir(self) \
        if not a.startswith('__') and not callable(getattr(self,a))]
    return "Residue(%s)" % ', '.join(attrs)


## Molecule class

class Molecule:
  # class variables
  _atomattrs = ['name', 'znuc', 'mmtype', 'mmq', 'mmconnect', 'resnum', 'lj_eps', 'lj_r0']
  _resattrs = ['resname', 'resatoms', 'pdb_num', 'chain', 'het']

  def __init__(self, atoms=None, residues=None, r=None, znuc=None, bonds=None, pbcbox=None, header=None):
    if r is not None and znuc is not None and atoms is None:
      atoms = [Atom(r=ra, znuc=za) for ra,za in zip(r, znuc)]
    elif atoms is not None and atoms.__class__ == Molecule:
      residues = copy.deepcopy(atoms.residues) if residues is None else residues
      atoms = copy.deepcopy(atoms.atoms)
      pbcbox = copy.deepcopy(atoms.pbcbox)
    self.atoms = atoms if atoms is not None else []
    self.residues = residues if residues is not None else []
    if r is not None and znuc is None:
      self.r = r
    if bonds is not None:
      self.set_bonds(bonds)
    self.pbcbox = pbcbox  # 3 dimensions for PBC box centered at origin
    self.header = header


  def listatoms(self, exclude=None):
    """ return list of atom indices, excluding any in `exclude` """
    # user should probably use a more general fn like difference(mol.listatoms(), exclude)
    if exclude:
      exclude = frozenset(exclude)
      return [ii for ii in range(len(self.atoms)) if ii not in exclude]
    return range(len(self.atoms))


  def enumatoms(self):
    return enumerate(self.atoms)


  # Maybe we should just rename bond() to dist()?
  def dist(self, a1, a2=None):
    a1, a2 = (a1,a2) if a2 is not None else a1
    return calc_dist(self.atoms[a1].r, self.atoms[a2].r)


  def extents(self, pad=0.0):
    return get_extents(self.r, pad)


  def append_atoms(self, atoms, r=None, residue=None):
    """ append Molecule, list of atoms, or single atom in `atoms` to this Molecule; positions passed in `r`, if
      provided, will replace those from `atoms`.  Residue object or name can be passed in `residue` for Atom(s),
      or int to add atoms to existing residue
    """
    offset = len(self.atoms)
    res_offset = None
    resnum = None
    if hasattr(atoms, 'atoms'):
      # molecule
      mol, atoms = atoms, atoms.atoms
      if residue is None:
        res_offset = len(self.residues)
        self.residues.extend(
            [setattrs(copy.copy(res), atoms=[ii + offset for ii in res.atoms]) for res in mol.residues])
    else:
      if hasattr(atoms, 'name'):
        # single atom
        residue = atoms.name if residue is None else residue
        atoms = [atoms]
    if type(residue) is int:
      resnum = residue
      self.residues[resnum].atoms.extend(range(offset, offset + len(atoms)))
    elif residue is not None:
      residue = Residue(name=residue, het=True) if type(residue) is str else copy.copy(residue)
      residue.atoms = range(offset, offset + len(atoms))
      resnum = len(self.residues)
      self.residues.append(residue)
    # reshape to provide error check and handle single atom case
    r = [a.r for a in atoms] if r is None else np.reshape(r, (len(atoms),3))
    self.atoms.extend([setattrs(copy.copy(a), r=r[ii], mmconnect=[jj + offset for jj in a.mmconnect],
        resnum=(a.resnum + res_offset if a.resnum is not None and res_offset is not None else resnum))
        for ii,a in enumerate(atoms)])
    return range(offset, len(self.atoms))  # list of appended atoms


  def extract_atoms(self, sel=None, resnums=None):
    """ return a new Molecule containing atoms specified by `sel` or residue numbers `resnums` """
    if sel is not None:
      atom_idxs = select_atoms(self, sel) if callable(sel) or type(sel) is str else sel
      resnums = sorted(list(set([getattr(self.atoms[ii], 'resnum', None) for ii in atom_idxs])))
    elif resnums is not None:
      resnums = [resnums] if type(resnums) is int else resnums
      atom_idxs = [ii for resnum in resnums for ii in self.residues[resnum].atoms]

    resnum_map = dict(zip(resnums, range(len(resnums))))
    idx_map = dict(zip(atom_idxs, range(len(atom_idxs))))
    atoms = [setattrs(copy.copy(self.atoms[ii]), resnum=resnum_map.get(self.atoms[ii].resnum, None),
        mmconnect=[idx_map[ii] for ii in self.atoms[ii].mmconnect if ii in idx_map]) for ii in atom_idxs]
    residues = [setattrs(copy.copy(self.residues[ii]),
        atoms=[idx_map[ii] for ii in self.residues[ii].atoms if ii in idx_map]) for ii in resnums]
    return Molecule(atoms, residues)


  def remove_atoms(self, sel):
    """ remove atoms specified by `sel` (in-place) """
    idxs = select_atoms(self, sel) if callable(sel) or type(sel) is str else sel
    # process indices from largest to smallest index so it remains valid
    idxs.sort(reverse=True)
    for ii in idxs:
      for atom in self.atoms:
        atom.mmconnect = [a - 1 if a > ii else a for a in atom.mmconnect if a != ii]
      for residue in self.residues:
        residue.atoms = [a - 1 if a > ii else a for a in residue.atoms if a != ii]
      # remove residue if now empty
      resnum = self.atoms[ii].resnum
      del self.atoms[ii]
      if resnum is not None and not self.residues[resnum].atoms:
        for jj in range(ii+1, len(self.residues)):
          for a in self.residues[jj].atoms:
            self.atoms[a].resnum -= 1
        del self.residues[resnum]

    return self


  def set_residue(self, name):
    self.residues = [Residue(name=name, atoms=self.listatoms(), het=True)]
    self.resnum = [0]*len(self.atoms)
    return self


  def select(self, *args, **kwargs):
    return select_atoms(self, *args, **kwargs)


  def get_bonded(self, qmatoms):
    """ get atoms directly bonded to `qmatoms`; useful for getting frontier MM atoms to set them inactive """
    qmatoms = [qmatoms] if type(qmatoms) is int else qmatoms
    return list(set([jj for ii in qmatoms for jj in self.atoms[ii].mmconnect if jj not in qmatoms]))


  def get_connected(self, idx, active=None):
    """ get all atoms connected directly or indirectly to atom number `idx` - i.e. get molecule for `idx` """
    active = set(active if active is not None else self.listatoms())
    conn = set([idx])
    def helper(ii):
      for ii in self.atoms[ii].mmconnect:
        if ii not in conn and ii in active:
          conn.add(ii)
          helper(ii)
    helper(idx)
    return list(conn)


  def get_nearby(self, qmatoms, radius):
    """ return atoms within radius of any atom in qmatoms, excluding atoms in qmatoms """
    nearby = []
    for ii in self.listatoms():
      if ii not in qmatoms:
        for jj in qmatoms:
          if norm(self.atoms[ii].r - self.atoms[jj].r) <= radius:
            nearby.append(ii)
            break
    # ---
    return nearby


  def set_bonds(self, bonds, replace=True):
    """ set mmconnect from list of pairs specifying bonded atoms; returns self for chaining """
    if replace:
      for atom in self.atoms:
        atom.mmconnect = []
    for bond in bonds:
      self.atoms[bond[0]].mmconnect.append(bond[1])
      self.atoms[bond[1]].mmconnect.append(bond[0])
    return self


  def get_bonds(self, active=None):
    """ because of angle and diheds, get_internals() can be slow for large molecules; use this method to get
    just bonds
    """
    if active is not None:
      return [ (ii, jj) for ii in active for jj in self.atoms[ii].mmconnect if jj > ii and jj in active ]
    else:
      return [ (ii, jj) for ii,a in self.enumatoms() for jj in a.mmconnect if jj > ii ]


  def get_internals(self, active=None, inclM1=False):
    """ returns bonds, angles, diheds - lists of atom number pairs, triples, quads (tuples)
    if list of active atoms is given, all other atoms are ignored (but angles and diheds involving atoms
    bonded to active atoms are included if inclM1 is True)
    """
    A = self.atoms  # alias
    active = frozenset(active or self.listatoms())
    inclfn = (lambda ii, kk: kk > ii or kk not in active) if inclM1 else (lambda ii, kk: kk > ii and kk in active)
    bonds = [ (ii, jj) for ii in active
      for jj in A[ii].mmconnect if jj > ii and jj in active ]
    angles = [ (ii, jj, kk) for ii in active
      for jj in A[ii].mmconnect if jj in active
      for kk in A[jj].mmconnect if inclfn(ii, kk) ]  #kk > ii and kk in active ]
    diheds = [ (ii, jj, kk, ll) for ii in active
      for jj in A[ii].mmconnect if jj in active
      for kk in A[jj].mmconnect if kk != ii and kk in active
      for ll in A[kk].mmconnect if ll != jj and inclfn(ii, ll) ]  #ll > ii and ll in active ]
    # angles and dihed are generated from bonds
    return bonds, angles, diheds


  ## geometry manipulation
  # Methods for manipulating molecular geometry by changing
  #  bond lengths, angles, and dihedrals.  Only atoms on one side
  #  side of bond/angle/dihedral are moved; see MMTK for code to
  #  avoid "overall translation/rotation" (is this just equiv to
  #  aligning to original?)
  # It should be clear from this code that performing manipulations
  #  in Cartesian coords can't be much (if at all) harder than with z-matrix
  # There's no reason to use calc_bond/angle - the calculations are one-liners
  # See: MMTK InternalCoordinates.py
  # TODO: option to retain (i.e., memoize) partition, since it may be time
  #  consuming to calculate for large molecules (?)

  ## TODO: use set() instead of list here!
  def partition_mol(self, a1, a2):
    # we include other atom in list to accomplish
    #  isolation, then remove after fragment is built
    f1 = self.buildfrag(a1, [ a2 ])[1:]
    f2 = self.buildfrag(a2, [ a1 ])[1:]
    # make sure intersection is empty
    assert len([x for x in f1 if x in f2]) == 0,\
      "Unable to partition molecule; is (%d, %d) bond part of cyclic structure?" % (a1, a2)
    return f1, f2


  def buildfrag(self, ii, frag):
    frag.append(ii)
    for jj in self.atoms[ii].mmconnect:
      if jj not in frag:
        self.buildfrag(jj, frag)
    return frag


  def coord(self, atoms):
    """ Wrapper for bond, angle, dihed that uses length of atom list to dispatch """
    if len(atoms) == 2: return self.bond(atoms)
    elif len(atoms) == 3: return self.angle(atoms)
    elif len(atoms) == 4: return self.dihedral(atoms)
    else: raise ValueError('Atom list must have length 2, 3, or 4')


  def bond(self, bond, newlen=None, rel=False, move_frag=True):
    """ bond is pair of atoms specifying the bond
    only fragment containing 2nd atom in bond is translated
    rel indicates newlen is relative to current
    """
    bondvec = self.atoms[bond[1]].r - self.atoms[bond[0]].r
    oldlen = norm(bondvec)
    if newlen is not None:
      f1, f2 = self.partition_mol(bond[0], bond[1]) if move_frag else ([], [bond[1]])
      # translation vector
      delta = (newlen - (not rel and oldlen or 0))
      deltavec =  delta*bondvec/oldlen
      for ii in f2:
        self.atoms[ii].r += deltavec
      return oldlen + delta
    else:
      return oldlen


  # probably should use radians instead of degrees for new value! (fpr dihedral() too)
  def angle(self, angle, newdeg=None, rel=False):
    """ angle is triple of atoms specifying angle
    only fragment containing 3rd atom of angle is rotated
    rel indicates new angle is relative to current
    """
    v12 = self.atoms[angle[0]].r - self.atoms[angle[1]].r
    v23 = self.atoms[angle[1]].r - self.atoms[angle[2]].r
    # normalize
    v12 /= norm(v12)
    v23 /= norm(v23)
    # result is in radians (pi - converts to interior bond angle)
    olddeg = np.pi - np.arccos(np.dot(v12, v23))
    if newdeg is not None:
      f1, f2 = self.partition_mol(angle[1], angle[2])
      delta = newdeg*np.pi/180 - (not rel and olddeg or 0)
      rotmat = rotation_matrix(np.cross(v23, v12), delta)
      rotcent = self.atoms[angle[1]].r
      for ii in f2:
        self.atoms[ii].r = np.dot(rotmat, self.atoms[ii].r - rotcent) + rotcent
      return olddeg + delta
    else:
      return olddeg


  def dihedral(self, dihed, newdeg=None, rel=False):
    """ dihed is set of 4 atoms specifying dihedral
    only fragment containing 3rd atom of dihed is rotated
    rel indicates new angle is relative to current
    """
    # we'll need v2 and v3 twice
    v1, v2, v3, v4 = ( self.atoms[dihed[0]].r, self.atoms[dihed[1]].r,
      self.atoms[dihed[2]].r, self.atoms[dihed[3]].r )
    olddeg = calc_dihedral(v1, v2, v3, v4)
    if newdeg is not None:
      f1, f2 = self.partition_mol(dihed[1], dihed[2])
      delta = newdeg*np.pi/180 - (not rel and olddeg or 0)
      rotmat = rotation_matrix(v2 - v3, -delta)
      rotcent = self.atoms[dihed[2]].r
      for ii in f2:
        self.atoms[ii].r = np.dot(rotmat, self.atoms[ii].r - rotcent) + rotcent
      return (olddeg + delta) - 2*np.pi*np.trunc( 0.5 + (olddeg + delta)/2/np.pi )
    else:
      return olddeg


  ## operator overloading

  def __repr__(self):
    sr = lambda a: a if type(a) is str or safelen(a) < 10 else ("[<%d>]" % len(a))
    attrs = ["%s=%r" % (a, sr(getattr(self,a))) for a in dir(self) \
        if not a.startswith('_') and not callable(getattr(self,a))]
    return ("Molecule(%s)" % ', '.join(attrs)).replace('\n', ' ')


  # .r is frequently accessed - we should make each Atom.r be a view of one row of saved r matrix!
  def __getattr__(self, attr):
    """ molecule.<attr>; except for special cases, returns array atoms[:].attr """
    if attr.startswith('_'):
      raise AttributeError(attr)
    if attr == 'r':
      return np.array([ self.atoms[ii].r for ii in self.listatoms() ])
    if attr == 'natoms':
      return len(self.atoms)
    if attr == 'nresidues':
      return len(self.residues)
    if attr == 'bonds':
      return self.get_bonds()
    if attr == 'mass':
      return np.array([ELEMENTS[a.znuc].mass for a in self.atoms])
    #if self.atoms and hasattr(self.atoms[0], attr):  # infinite loop if self.atoms not set
    if attr in self._atomattrs:
      return np.array([ getattr(atom, attr) for atom in self.atoms ])
    if attr in self._resattrs:
      resattr = attr[3:] if attr[:3] == 'res' else attr
      return np.array([ getattr(res, resattr) for res in self.residues ])
    raise AttributeError(attr)


  def __setattr__(self, attr, val):
    """ molecule.<attr> = val  """
    if attr == 'r':
      assert len(val) == len(self.atoms), "Incorrect length!"
      for ii in range(len(self.atoms)):
        self.atoms[ii].r = np.array(val[ii])  # copy
    #elif attr == 'bonds':
    #  self.set_bonds(val)
    elif attr in self._atomattrs:
      assert len(val) == len(self.atoms), "Incorrect length!"
      for ii in range(len(self.atoms)):
        setattr(self.atoms[ii], attr, val[ii])
    elif attr in self._resattrs:
      assert len(val) == len(self.residues), "Incorrect length!"
      resattr = attr[3:] if attr[:3] == 'res' else attr
      for ii in range(len(self.residues)):
        setattr(self.residues[ii], resattr, val[ii])
    else:
      self.__dict__[attr] = val


## Utility fns for atom selection, etc.
# Functions which act more or less symmetrically on two molecules shouldn't be part of
#  molecule class.  Also, most functions which rely on only coordinates (i.e., don't
#  need connectivity info, etc) should not be part of molecule class and should work
#  with simple lists of coordinates

def get_header(mol):
  header = getattr(mol, 'header', None)
  if not header:
    import time
    today = time.strftime("%d-%b-%y", time.localtime()).upper()
    header = "CUSTOM MOLECULE                         %s   XXXX" % today
  return header


# this will be very inefficient if used repeatedly
def residue_chain(mol, resnum):
  """ determine chain number of start, stop residue index for residue indexed by `resnum` """
  chain, het = mol.residues[resnum].chain, mol.residues[resnum].het
  start, stop = resnum, resnum
  while start > 0 and mol.residues[start-1].chain == chain and mol.residues[start-1].het == het:
    start -= 1
  while stop < len(mol.residues) and mol.residues[stop].chain == chain and mol.residues[stop].het == het:
    stop += 1
  # count unique chains before start
  chain_idx = len(set([mol.residues[ii].chain for ii in range(start-1)]))
  return chain, chain_idx, start, stop


def mol_fragments(mol, active=None):
  """ return set of unconnected fragments from `active` (default, all) atoms in `mol` """
  remaining = set(active if active is not None else mol.listatoms())
  frags = []
  while remaining:
    frags.append(mol.get_connected(remaining.pop(), active))
    remaining -= set(frags[-1])
  return frags


# any reason for option to only connect non-hydrogens?
def fragment_connections(r, frags):
  """ returns set of minimum distance connections needed to fully connect set of fragments specified by
    `frags` - list of lists of indexes into `r` (as returned by mol_fragments())
  """
  from scipy.spatial.ckdtree import cKDTree
  r_frags = [ [r[ii] for ii in frag] for frag in frags ]
  kdtrees = [cKDTree(r_frag) for r_frag in r_frags[:-1]]
  connect = []  # list to hold "bonds" needed to connect fragments
  unconnected = set(range(len(frags)-1))
  r_connected = r_frags[-1]
  idx_connected = list(frags[-1])
  while unconnected:
    min_dist = np.inf
    for ii,kd in enumerate(kdtrees):
      if ii in unconnected:
        dists, idxs = kd.query(r_connected)
        min_idx = dists.argmin()
        if dists[min_idx] < min_dist:
          min_dist = dists[min_idx]
          closest_frag = ii
          closest_pair = (idx_connected[min_idx], frags[ii][idxs[min_idx]])
    # connect next fragment
    connect.append(closest_pair)
    unconnected.remove(closest_frag)
    r_connected.extend(r_frags[closest_frag])
    idx_connected.extend(frags[closest_frag])
  return connect


def nearest_pairs(r_array, N):
  """ return list of pairs (i,j), i < j for `N` nearest neighbors of points `r_array` """
  from scipy.spatial.ckdtree import cKDTree
  if N > len(r_array) - 1:
    print("Warning: requested %d nearest neighbors but only %d points" % (N, len(r_array)))
  ck = cKDTree(r_array)
  dists, locs = ck.query(r_array, min(len(r_array), N+1))  # query returns same point, so query for N+1 points
  return [ (ii, jj) for ii, loc in enumerate(locs) for jj in loc if jj > ii ]


def generate_internals(bonds):  #, impropers=False):
  """ generate angles, dihedrals, and improper torsions (w/ 2nd atom central atom) from a set of bonds """
  max_idx = max([x for bond in bonds for x in bond]) if bonds else -1
  connect = [ [] for ii in range(max_idx + 1) ]
  for bond in bonds:
    connect[bond[0]].append(bond[1])
    connect[bond[1]].append(bond[0])
  angles = [ (ii, jj, kk) for ii in range(len(connect))
    for jj in connect[ii]
    for kk in connect[jj] if kk > ii ]
  diheds = [ (ii, jj, kk, ll) for ii in range(len(connect))
    for jj in connect[ii]
    for kk in connect[jj] if kk != ii
    for ll in connect[kk] if ll != jj and ll > ii ]
  imptor = [ (ii, jj, kk, ll) for ii in range(len(connect))
    for jj in connect[ii]
    for kk in connect[jj] if kk != ii
    for ll in connect[jj] if ll != kk and ll > ii ]
  return angles, diheds, imptor


def pdb_repr(mol, ii, sep=" "):
  """ get PDB string (chain res_num atom_name) for atom `ii` in `mol` """
  a = mol.atoms[ii]
  res = mol.residues[a.resnum] if a.resnum is not None else None
  return sep.join([res.chain, res.pdb_num, a.name]) if res is not None else a.name


def init_qmatoms(mol, sel, qmbasis):
  sel = mol.select(sel)
  for ii in sel:
    mol.atoms[ii].qmbasis = qmbasis
  return sel


# consider returning a class with __call__ instead so we can set __repr__ to print sel string!
def decode_atom_sel_str(sel):
  """ unqualifed exec() can't be in same function as a nested fn or lambda """
  locs = {}  # needed for Python 3
  exec("""
def sel_fn(atom, mol, idx):
  name = atom.name
  znuc = atom.znuc
  resnum = atom.resnum
  residue, resname, resatoms, chain, pdb_resnum, pdb_resid = None, '', [], None, None, None
  try:
    residue = mol.residues[resnum]
    resname = residue.name
    resatoms = residue.atoms
    chain = residue.chain
    pdb_resid = residue.pdb_num
    pdb_resnum = int(pdb_resid)
  except: pass
  # booleans
  polymer = resname in PDB_STANDARD
  protein = resname in PDB_PROTEIN
  backbone = protein and name in ['C', 'N', 'CA']
  extbackbone = protein and name in ['C', 'N', 'CA', 'O', 'H', 'H1', 'H2', 'H3', 'OXT', 'HXT']
  sidechain = protein and name not in ['C', 'N', 'O', 'H', 'H1', 'H2', 'H3', 'OXT', 'HXT']
  water = resname == 'HOH'
  return """ + sel, globals(), locs)
  # ---
  return locs['sel_fn']


def decode_atom_sel(sel):
  """ convert atom selection string `sel` to function (just returns `sel` if not a string) """
  if type(sel) is str:
    return decode_atom_sel_str(sel)
  try:
    if type(sel[0]) is int:
      sel_set = frozenset(sel)
      return lambda atom, mol, idx: idx in sel_set
  except: pass
  return sel


# PDB selection syntax - succinct atom selection with PDB fields - similar to pymol selection macros:
# '/chain/pdb_resnum/atoms'; field can be left blank to select all; leading / is only required for
#  '/chain' and '/chain/pdb_resnum'; without it, string interpreted as 'atoms' and 'pdb_resnum/atoms'
# fields are comma separate lists of values, optionally preceeded by '~' to negate
# spaces can be used as separator instead of /, in which case * can be used to select all
# multiple selections can be joined with ';'
# should we support range for resnums, e.g 100-110? ... non-trivial because of insert codes
# e.g.: "A/57,102,195/~C,N,CA,O,H,HA; I/5/CA,C,HA,O; I 6 CA,N,H,HA"
#  all equivalent: '/A//', '/A/', '/A', 'A//', 'A * *', '/A *', '/A/*', '/A/*/*', ...
# Can be passed as `pdb` keyword arg or in `sel` if first char is / or *

# make this a method of Molecule?
def select_atoms(mol, sel=None, sort=None, pdb=None):
  if sort is True:
    sort = lambda atom, mol, ii: (atom.resnum, atom.name, ii)  # default sort key
  elif sort is not None:
    sort = decode_atom_sel(sort)
  pdb = sel if type(sel) is str and (sel.startswith('/') or sel.startswith('*')) else pdb
  if pdb:
    selected = []
    pdbsels = re.sub(r'\s*,\s*', ',', pdb).split(';')  # remove spaces around commas
    for pdbsel in pdbsels:
      ss = pdbsel.strip().replace('/', ' ').split()  #re.split(r'\s*/\s*|\s+', pdbsel.strip())  # split on space or /
      ss = ss[1:] if not ss[0] else ss
      if not ss: continue  # empty selection
      assert len(ss) <= 3, "Too many fields in " + pdbsel
      # extend to contain all three fields; blank or * means match all
      full_ss = (['']*(3-len(ss)) + ss)  #if ss[0] else (ss[1:] + ['']*(3-len(ss)+1))
      res_field = 'resname' if full_ss[1][-3:].isalpha() else 'pdb_resid'  # support residue name or number
      atomsel = " and ".join(
          (f + " not in " + str(s[1:].split(','))) if s[0] == '~' else (f + " in " + str(s.split(','))) \
          for f, s in zip(['chain', res_field, 'name'], full_ss) if s and s != '*')
      # should we join with 'or' and create a single selection string instead?
      selected.extend(select_atoms(mol, atomsel if atomsel else 'True'))  # '*' will give empty atomsel
  elif safelen(sel) > 0 and type(sel[0]) is int:
    selected = sel  # preserve order
  else:
    sel_fn = decode_atom_sel(sel)
    selected = [ii for ii, atom in enumerate(mol.atoms) if sel_fn is None or sel_fn(atom, mol, ii)]
  return sorted(selected, key=lambda ii: sort(mol[ii], mol, ii)) if sort is not None else selected


# How might we support final box extents + total number of copies?
# - this related to sphere packing, which is non-trivial
# - maybe see https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
# - we could try low-discrepancy sequence to generate points, but still some probability of clash, and
#  a single clash can break minimization
# - consider removing shuffling or making it optional
def tile_mol(mol, extents, shape, rand_rot=False, rand_offset=0):
  """ tile `mol` with box `extents` symmetrically about its center `shape` times (x,y,z), optionally with
    random rotation if `rand_rot`=True and random translation by 0 to +/-0.5 * `random_offset`
  """
  extents = np.asarray([extents]*3 if np.isscalar(extents) else extents)
  extents = np.array([-0.5*extents, 0.5*extents]) if np.size(extents) == 3 else extents
  unitcell = extents[1] - extents[0]
  offset = 0.5*(np.array(shape)-1)
  r0 = mol.r
  tiled = Molecule()
  ijks = [(ii, jj, kk) for ii in range(shape[0]) for jj in range(shape[1]) for kk in range(shape[2])]
  np.random.shuffle(ijks)
  # I think this could be vectorized
  for ijk in ijks:
    r = np.dot(r0, random_rotation()) if rand_rot else r0
    r = r + (ijk - offset) * unitcell
    r = r + rand_offset*(np.random.random(3) - 0.5) if rand_offset else r
    tiled.append_atoms(mol, r)
  return tiled


## Analysis fns - these probably should be moved to a separate file

# note that unweighted average of all positions is the centroid while weighted average is the center of mass
def center_of_mass(mol):
  weights = np.array([ELEMENTS[z].mass for z in mol.znuc])
  return np.sum(weights[:,None]*mol.r, axis=0)/np.sum(weights)


def calc_RMSD(mol1, mol2, atoms=None, grad=False):
  """ Return RMS displacement (RMSD) between atoms of mol1 and atoms of mol2 and, if grad=True, gradient wrt
    first set of coordinates
  """
  # handle case of molecule objects
  r1 = getattr(mol1, 'r', mol1)
  r2 = getattr(mol2, 'r', mol2)
  dr = (r1 - r2) if atoms is None else (r1[atoms] - r2[atoms])
  natoms = np.size(dr)/3
  rmsd = np.sqrt(np.sum(dr*dr)/natoms)
  return (rmsd, dr/(natoms*rmsd)) if grad else rmsd


def alignment_matrix(subject, ref, weights=1.0):
  """ return 4x4 tranformation matrix providing least squares (RMSD) optimal overlap of `subject` points
    with `ref` points; optionally, `weights` for each point can be provided
  Refs: BioPython SVDSuperimposer class, personal notes, Kabsch algorithm (Wikipedia),
   J. Comp. Chem. 25, 1849 (SVD and quaternion approaches)
  """
  if np.size(weights) > 1:
    totmass = np.sum(weights)
    # we assume weights is a row vector
    weights = np.transpose(np.tile(weights, (3,1)))
  else:
    weights = 1.0  # prevent invalid scalar weight
    totmass = np.size(subject)/3  # number of atoms

  # move both centers of mass to origin - this is necessary for finding rotation
  CMsubject = np.sum(weights * subject, 0)/totmass
  CMref = np.sum(weights * ref, 0)/totmass
  subject = subject - CMsubject
  ref = ref - CMref
  # calculate correlation matrix (division by totmass isn't actually needed)
  corr = np.dot(np.transpose(subject), weights*ref)/totmass
  u, w, vT = np.linalg.svd(corr)
  # calculate optimal rotation matrix
  rot = np.dot(u, vT)
  # check for reflection and undo if present
  if np.linalg.det(rot) < 0:
    vT[2] = -vT[2]
    rot = np.dot(u, vT)

  return np.dot(translation_matrix(-CMsubject), np.dot(to4x4(rot), translation_matrix(CMref)))


def align_atoms(subject, ref, weights=1.0, sel=None):
  """ return optimal alignment of points `subject` (or `subject`.r if a Molecule) to `target` (or `target`.r)
    with optional `weights`. A list of atoms indices `sel` to use for alignment can be passed
  """
  # handle case of molecule objects
  subject = getattr(subject, 'r', subject)
  ref = getattr(ref, 'r', ref)
  if sel is not None:
    weights = setitem(np.zeros(len(subject)), sel, 1.0 if np.isscalar(weights) else weights[sel])
  return apply_affine(alignment_matrix(subject, ref, weights), subject)


# note that sel+sort may give different indices for mol and ref, so taking a single list of indices would
#  not be equivalent
def align_mol(mol, ref, sel=None, sort='atom.resnum, atom.name'):
  if hasattr(ref, 'atoms'):
    ref = ref.r[ select_atoms(ref, sel, sort=sort) ] if sel else ref.r
  align = mol.r[ select_atoms(mol, sel, sort=sort) ] if sel else mol.r
  if len(align) == len(ref):
    M = alignment_matrix(align, ref)
    mol.r = apply_affine(M, mol.r)
  else:
    print("Error: number of alignment atoms does not match reference for ", mol)
  return mol


# Usage: mol.set_bonds(guess_bonds(mol))
def guess_bonds(r_array, z_array=None, tol=0.1):
  """ return list of covalent bonds ((i,j), i < j) for atoms with positions r and atomic numbers z """
  from scipy.spatial.ckdtree import cKDTree
  if z_array is None:
    r_array, z_array = r_array.r, r_array.znuc
  # find all pairs within 5 Ang; largest covalent radius is 2.35 (Cs)
  ck = cKDTree(r_array)
  pairs = ck.query_pairs(5)
  bonds = []
  for i,j in pairs:
    #rval = 1.3*(ELEMENTS[z_array[i]].cov_radius + ELEMENTS[z_array[j]].cov_radius) ... more standard
    rval = tol + ELEMENTS[z_array[i]].cov_radius + ELEMENTS[z_array[j]].cov_radius
    dr = r_array[i] - r_array[j]
    if np.dot(dr,dr) < rval*rval:
      bonds.append( (i, j) if i < j else (j, i) )
  return bonds
