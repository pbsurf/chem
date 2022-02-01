# Tinker interface

import numpy as np
import os, subprocess, time, sys
if sys.version_info >= (3, 0):
  from io import StringIO
else:
  from cStringIO import StringIO

from ..basics import *
from ..molecule import Atom, Residue, Molecule, get_header
from ..data.elements import ELEMENTS


TINKER_PATH = os.getenv('TINKER_PATH')

# See save_zip/load_zip for handling of .arc - multiple concatenated xyz files (aka Tinker archive)
def parse_xyz(xyz, inorganic=False):
  """ create Molecule from file or string `xyz` with format: [index] name x y z [mm_type [mm_charge] bonded]
    first line: natoms [comment]; second line: first atom, comment, blank, or PBC box; "Cl, Ca, Na, etc.
    interpreted as carbon, nitrogen unless inorganic==True
  """
  # file object, contents string, or filename string
  fxyz = xyz if type(xyz) is not str else StringIO(xyz) if '\n' in xyz else open(xyz, 'r')
  l = fxyz.readline().split(None, 1)
  if not l:
    return None
  if l[0] == "[AMBFOR]":
    l = fxyz.readline().split(None, 1)  # support Molden AMBFOR .xyz
  natoms = int(l[0])
  header = l[1].strip() if len(l) > 1 else ''
  l = fxyz.readline().split()
  mol = Molecule(header=header)
  # for generic (non-Tinker) xyz, line 2 is comment; for Tinker xyz with periodic box, line 2 specifies box
  #  and values contain '.', for which isdigit() returns False
  if not l or not l[0].isdigit():
    try: mol.pbcbox = np.array([float(w) for w in l][:3])  # Tinker PBC: a b c angle_a angle_b angle_c
    except: pass
    l = fxyz.readline().split()
  # Tinker xyz doesn't necessarily start atom numbering at 1
  offset = int(l[0]) if l[0].isdigit() else 1
  last_res_id, last_chain_id = None, None  # for our xyz + pdb hack
  while len(l) > 3:
    if len(l) == 4:
      # to support generic xyz format
      l = [0] + l + [0]
    name = l[1]
    if '_' in name:
      # support our hack cramming residue info in atom name (Tinker ignores name, just needs line len < 120)
      extname = name.split('_')
      is_hetatm = name.endswith('_')
      name, res_type, chain_id, res_id = ([s for s in extname if s] + [None]*4)[:4]
      # we'll require two character elements be written to pdbhack .xyz in title case
      elem = name[:2] if len(name) > 1 and name[:2] in ELEMENTS else name[0].upper()  #name.title() if is_hetatm
      if len(elem) > 1:
        name = elem.upper() + name[2:]  # use upper case internally
      if res_id != last_res_id or chain_id != last_chain_id:
        last_res_id, last_chain_id = res_id, chain_id
        mol.residues.append(Residue(name=res_type, pdb_num=res_id, chain=chain_id, het=is_hetatm))
      mol.residues[-1].atoms.append(len(mol.atoms))  # haven't added atom yet
    else:
      # (some) xyz files use forcefield atom name, which may include extra characters
      elem2 = name[:2].title()
      elem = elem2[0] if elem2[0] in 'HCNOPS' and (elem2 not in ELEMENTS or not inorganic) else elem2
    has_mmq = len(l) > 6 and not l[6].isdigit()  # float at l[6] is MMQ
    mol.atoms.append(Atom(
      name=name,
      r=np.array(l[2:5], 'd'),
      znuc=ELEMENTS[elem].number if elem in ELEMENTS else 0,
      mmtype=(int(l[5]) if l[5].isdigit() else l[5]) if len(l) > 5 else 0,
      mmq=float(l[6]) if has_mmq else 0.0,
      # note conversion to zero-based indexing
      mmconnect=[int(x) - offset for x in l[(7 if has_mmq else 6):]],
      resnum=len(mol.residues) - 1 if mol.residues else None
    ))
    if len(mol.atoms) == natoms and fxyz is xyz:  # to support .arc format
      break
    l = fxyz.readline().split()

  if len(mol.atoms) != natoms:
    print("Warning: %d atoms found in xyz file but %d expected" % (len(mol.atoms), natoms))
  if fxyz is not xyz:
    fxyz.close()
  return mol


def parse_mol2(xyz, inorganic=False):
  """ parse mol2 format (used by Amber) - similar to xyz by different enough to justify separate fn """
  fxyz = xyz if type(xyz) is not str else StringIO(xyz) if '\n' in xyz else open(xyz, 'r')
  skiptoline(fxyz, "@<TRIPOS>ATOM", strip=True)
  mol = Molecule()
  res_id = None
  l = fxyz.readline().split()
  while len(l) >= 5:
    elem2 = l[1][:2].title()
    elem = elem2[0] if elem2[0] in 'HCNOPS' and (elem2 not in ELEMENTS or not inorganic) else elem2
    if len(l) > 6:
      res_id, last_res_id = int(l[6]), res_id
      if res_id != last_res_id:
        res_type = l[7] if len(l) > 7 else 'UNK'
        mol.residues.append(Residue(name=res_type, pdb_num=res_id, chain='Z', het=True))

    mol.atoms.append(Atom(
      name=l[1],
      r=np.array(l[2:5], 'd'),
      znuc=ELEMENTS[elem].number if elem in ELEMENTS else 0,
      mmtype=(int(l[5]) if l[5].isdigit() else l[5]) if len(l) > 5 else 0,
      mmq=float(l[8]) if len(l) > 8 else 0.0,
      # note conversion to zero-based indexing
      resnum=len(mol.residues) - 1 if mol.residues else None
    ))
    l = fxyz.readline().split()
  if l[0] != "@<TRIPOS>BOND":
    skiptoline(fxyz, "@<TRIPOS>BOND", strip=True)
  l = fxyz.readline().split()
  while len(l) == 4:
    mol.add_bonds( (int(l[1])-1, int(l[2])-1) )
    l = fxyz.readline().split()

  return mol


def load_tinker_params(mol, filename=None, key=None, charges=True, vdw=False, bonded=False):
  """ set .mmq and/or .lj_r0,.lj_eps attributes on atoms of mol """
  cleanup = False
  if filename is None:
    prefix = "mmtmp_%d" % time.time()
    filename = prefix + ".xyz"
    # don't need bonds to get charge or vdW params
    write_tinker_xyz(mol, filename, noconnect=(None if bonded else mol.listatoms()))
    if key is not None:
      mmkey = prefix + ".key"
      write_tinker_key(key, mmkey)
    elif not os.path.exists('tinker.key'):
      raise ValueError("Tinker key not specified and tinker.key not present!")
    cleanup = True

  analysis = subprocess.check_output([os.path.join(TINKER_PATH, 'analyze'), filename, 'P'])
  f = StringIO(analysis)
  try:
    UNIT = 1.0/KCALMOL_PER_HARTREE
    if bonded:
      mol.mm_stretch, mol.mm_bend, mol.mm_torsion, mol.mm_imptor = [], [], [], []
      skiptoline(f, ' Bond Stretching Parameters :')
      b = skiptoline(f, '1', strip=True).split()
      while len(b) > 2:
        if len(b) > 4:
          mol.mm_stretch.append(( [int(b[1])-1, int(b[2])-1], float(b[3])*UNIT, float(b[4]) ))
        b = f.readline().split()
      l = f.readline().strip()

      if l == 'Angle Bending Parameters :':
        b = skiptoline(f, '1', strip=True).split()
        while len(b) > 3:
          if len(b) > 5:
            mol.mm_bend.append(( [int(b[1])-1, int(b[2])-1, int(b[3])-1], float(b[4])*UNIT, float(b[5])*np.pi/180 ))
          b = f.readline().split()
        l = f.readline().strip()

      if l == 'Improper Torsion Parameters :':
        b = skiptoline(f, '1', strip=True).split()
        while len(b) > 4:
          if len(b) > 7:
            mol.mm_imptor.append(( [int(b[1])-1, int(b[2])-1, int(b[3])-1, int(b[4])-1],
                [(float(b[5])*UNIT, float(b[6])*np.pi/180, float(b[7]))] ))
          b = f.readline().split()
        l = f.readline().strip()

      if l == 'Torsional Angle Parameters :':
        b = skiptoline(f, '1', strip=True).split()
        while len(b) > 4:
          if len(b) > 6:
            terms = []
            for amp,pps in chunks(b[5:], 2):
              pp = pps.split('/')
              terms.append( (float(amp)*UNIT, float(pp[0])*np.pi/180, float(pp[1])) )  # amplitude, phase, periodicity
            mol.mm_torsion.append( ([int(b[1])-1, int(b[2])-1, int(b[3])-1, int(b[4])-1], terms) )
          b = f.readline().split()
        l = f.readline().strip()
    else:
      skiptoline(f, ' Van der Waals Parameters :')

    if vdw:
      line = skiptoline(f, '1', strip=True)
      for atom in mol.atoms:
        lj = line.split()
        atom.lj_r0 = 2.0*float(lj[2])  # "Size" from Tinker is r0/2
        atom.lj_eps = float(lj[3])*UNIT
        line = f.readline()
    else:
      skiptoline(f, ' Atomic Partial Charge Parameters :')

    if charges:
      line = skiptoline(f, '1', strip=True)
      for atom in mol.atoms:
        q = line.split()
        atom.mmq = float(q[2])
        line = f.readline()
  except:
    print("Error getting MM params: Tinker analyze failed:\n\n" + analysis)
    raise

  if cleanup:
    try:
      os.remove(filename)
      if key is not None:
        os.remove(mmkey)
    except: pass


def write_tinker_params(mol):
  """ serialize MM params in Tinker format """
  UNIT = 1.0/KCALMOL_PER_HARTREE
  ids = lambda b: tuple(mol.atoms[a].mmtype for a in b)
  s = []
  for ii,a in enumerate(mol.atoms):
    s.append('atom %d %d %s "GAFF atom" %d %.3f %d' %
        (a.mmtype, a.mmtype, a.name, a.znuc, ELEMENTS[a.znuc].mass, len(a.mmconnect)))  # mm class = mm type
  for p in mol.mm_stretch:
    s.append("bond %d %d  %.2f %.4f" % (ids(p[0]) + (p[1]/UNIT, p[2])))
  for p in mol.mm_bend:
    s.append("angle %d %d %d  %.2f %.2f" % (ids(p[0]) + (p[1]/UNIT, p[2]*180/np.pi)))
  for p in mol.mm_torsion:
    s.append("torsion %d %d %d %d  " % ids(p[0])
        + '  '.join("%.3f %.1f %d" % (q[0]/UNIT, q[1]*180/np.pi, q[2]) for q in p[1]))
  for p in mol.mm_imptor:
    s.append("imptors %d %d %d %d  " % ids(p[0])
        + '  '.join("%.3f %.1f %d" % (q[0]/UNIT, q[1]*180/np.pi, q[2]) for q in p[1]))
  for a in mol.atoms:
    s.append("vdw %d  %.4f %.6f" % (a.mmtype, a.lj_r0/2, a.lj_eps/UNIT))
  for a in mol.atoms:
    s.append("charge %d  %.4f" % (a.mmtype, a.mmq))
  return '\n'.join(s)


# simple xyz reader to read just geometry from xyz file
def read_tinker_geom(filename, pbc=False, last=False):
  """ Read all (or only last if `last` is True) geometries from TINKER .xyz or .arc file """
  XYZs = []
  PBCs = []
  with open(filename, 'r') as input:
    l = input.readline().split()
    while True:
      try:
        natoms = int(l[0])
      except:
        break
      l = input.readline().split()
      # skip 2nd line if it doesn't start with int
      if not l[0].isdigit():
        try: PBCs.append([float(w) for w in l][:3])  # Tinker PBC: a b c angle_a angle_b angle_c
        except: pass
        l = input.readline().split()
      XYZs.append([])
      for ii in range(natoms): # and len(l) > 4:  -- exception on len(l) < 5
        XYZs[-1].append( [float(x) for x in l[2:5]] )
        l = input.readline().split()

  if last:
    XYZs, PBCs = XYZs[-1], (PBCs[-1] if PBCs else [])
  elif len(XYZs) == 1 and not filename.endswith(".arc"):
    XYZs, PBCs = XYZs[0], (PBCs[0] if PBCs else [])
  return (np.array(XYZs), np.array(PBCs)) if pbc else np.array(XYZs)


def read_tinker_energy(input, components=False):
  """ Read energy from TINKER analyze E (or D) """
  keystr = " Total Potential Energy :"
  f = StringIO(input) if '\n' in input else open(input, 'r')
  x = f.readline()
  while len(x) != 0 and x[:len(keystr)] != keystr:
    x = f.readline()
  E = float(x.split()[4]) / KCALMOL_PER_HARTREE
  # now read components of energy
  if components:
    keystr = " Energy Component Breakdown :"
    x = f.readline()
    while len(x) != 0 and x[:len(keystr)] != keystr:
      x = f.readline()
    # skip blank line
    f.readline()
    Ecomps = {}
    x = f.readline().split()
    while len(x) >= 3:
      Ecomps["Emm_" + "_".join(x[:-2])] = float(x[-2]) / KCALMOL_PER_HARTREE
      x = f.readline().split()
  f.close()
  return (E, Ecomps) if components else E


def read_tinker_grad(input):
  """ parses output of TINKER testgrad, which provides energy (in kcal/mol) and gradient (in kcal/mol/Ang) """
  energystr = " Total Potential Energy :"
  gradstr = " Cartesian Gradient Breakdown over Individual Atoms :"
  f = StringIO(input) if '\n' in input else open(input, 'r')
  line = skiptoline(f, energystr)
  if not line:
    print(input)
    raise ValueError("Unable to read energy from Tinker output")
  E = float(line.split()[4]) / KCALMOL_PER_HARTREE
  skiptoline(f, gradstr, extra=3)
  G = []
  while True:
    l = f.readline().split()
    if len(l) < 5:
      if len(l) > 1:
        print("Warning: unexpected line in Tinker gradient output: ", l)  # >=1e6 or <=-1e5 means no room for spaces
      break
    # testgrad doesn't print grad for inactive atoms, so insert zeros for any missing rows; G will still be
    #  too short if last atom(s) inactive, but no way to get total number of atoms from testgrad output
    G.extend([ [0,0,0] for ii in range(int(l[1]) - len(G) - 1) ])
    G.append([float(x) for x in l[2:5]])
  G = np.array(G) / KCALMOL_PER_HARTREE
  f.close()
  return E, G


def read_tinker_hess(filename):
  """ Construct Hessian from .hes file generated by TINKER testhess """
  hdiag, hoffd = [], []
  input = open(filename, 'r')
  input.readline()
  input.readline()
  input.readline()
  l = input.readline().split()
  while len(l) > 0:
    hdiag.extend([float(x) for x in l])
    l = input.readline().split()
  # create matrix to hold hessian
  hess = np.diag(hdiag)
  # now for the off diag elements
  ii = 0
  x = input.readline()
  while len(x) > 0:
    input.readline()
    l = input.readline().split()
    while len(l) > 0:
      hoffd.extend([float(x) for x in l])
      l = input.readline().split()
    # fill in hessian - the idea here is to cause NumPy errors if there is a parsing problem
    hess[ii, (ii+1):] = np.array(hoffd)
    hess[(ii+1):, ii] = np.array(hoffd)
    ii += 1
    hoffd = []
    x = input.readline()
  input.close()
  return hess


def read_tinker_interactions(input):
  """ read output of `analyze D` - breakdown of individual interactions """
  f = StringIO(input) if '\n' in input else open(input, 'r')
  E = Bunch()
  while True:
    skiptoline(f, " Individual", extra=3)
    tokens = f.readline().split()
    if not tokens:
      break
    type = tokens[0].lower()
    natoms = dict(angle=3, improper=4, torsion=4, solvate=1).get(type, 2)
    E[type] = {}
    while tokens:
      atoms = tuple(int(a.split('-')[0])-1 for a in tokens[1:1+natoms])
      atoms = atoms[::-1] if atoms[-1] < atoms[0] else atoms
      E[type][atoms] = float(tokens[-1])/KCALMOL_PER_HARTREE  # energy is always last item on line
      tokens = f.readline().split()
  f.close()
  return E


def write_tinker_xyz(mol, filename=None, r=None, noconnect=None, title=None, pdbhack=False, writemmq=False):
  def namehack(a):
    if not pdbhack or a.resnum is None:
      return a.name
    res = mol.residues[a.resnum]
    # use title case for two character elements
    name = a.name[:2].title() + a.name[2:] if len(ELEMENTS[a.znuc].symbol) > 1 else a.name
    return "{:_<4}_{}_{}_{}{}".format(name,  #ELEMENTS[a.znuc].symbol.upper() if res.het else a.name
        res.name or 'XXX', res.chain or 'Z', res.pdb_num or (a.resnum + 1), "_" if res.het else "")
  # idea of noconnect is to omit bonds for (hetatm) qmatoms so we don't need extra Tinker params
  noconnect = frozenset(noconnect if noconnect else [])
  def bonded(ii, a):
    return "" if ii in noconnect else " ".join(["%5d" % (x+1) for x in a.mmconnect])

  r = mol.r if r is None else r
  # allow non-integer mmtype only for pdbhack case
  fmt = " %5d  %-16s %11.6f %11.6f %11.6f %5s%s%s\n" if pdbhack else " %5d  %-3s %11.6f %11.6f %11.6f %5d%s%s\n"
  # since Tinker mmtype > 0, we'll just use bool(mmtype); gap in numbering is OK
  lines = [fmt % ( ii+1, namehack(a), r[ii][0], r[ii][1], r[ii][2], a.mmtype, (' %7.4f '%a.mmq) if writemmq else ' ',
      bonded(ii, a) ) for ii, a in mol.enumatoms() if a.mmtype or pdbhack ]
  pbc = ("    %f %f %f 90.0 90.0 90.0\n" % tuple(mol.pbcbox)) if mol.pbcbox is not None else ""
  lines[0] = ("  %d  %s\n%s" % (len(lines), title or get_header(mol), pbc)) + lines[0]
  if filename is None:
    return "".join(lines)
  with open(filename, 'w') as f:
    f.write("".join(lines))


def write_xyz(mol, *args, **kwargs):
  return write_tinker_xyz(mol, *args, pdbhack=True, **kwargs)


# inactive atoms are assigned to a group for which intra-group interactions are set to 0; just using inactive
#  keyword isn't good because testgrad won't print gradient for inactive atoms
def write_tinker_key(key, filename, inactive=None, charges=None):
  with open(filename, 'w') as f:
    f.write(key)
    if inactive:
      # tinker only reads first ~100 chars of line, so split group across lines as needed
      f.write("\n".join("group 1   " + " ".join("%d" % (x+1) for x in chunk) for chunk in chunks(inactive, 10)))
      f.write("\ngroup-select 1 1 0.0\n\n")
    if charges:
      # negative arg for TINKER charge keyword applies value to individual atoms instead of atom type
      f.write("".join(["charge %4d %7.3f\n" % (-ii-1, q) for ii,q in charges]))


def make_tinker_key(ff='amber96', digits=8):
  pfile = os.path.abspath(os.path.join(TINKER_PATH, "../params/%s.prm" % ff))
  return "parameters     %s\ndigits %d\n" % (pfile, digits)


# should we take a generic keyopts dict instead of inactive and charges?
# for grad and components, pass True to calculate, False to skip but include in returns, and None to exclude
#  from returns ... this is done to ease unpacking
# An alternative would be to require a dict to passed in to be filled with components
def tinker_EandG(mol, r=None, prefix=None, key=None,
    inactive=None, charges=None, noconnect=None, title=None, grad=True, components=None, cleanup=False):
  """ return energy and optionally gradient and components of energy of `mol` calculated by Tinker """
  if prefix is None:
    prefix = "mmtmp_%d" % time.time()
    cleanup = True
  mminp = prefix + ".xyz"
  mmkey = prefix + ".key"

  write_tinker_xyz(mol, mminp, r=r, title=title, noconnect=noconnect)
  # writing key file can be skipped if prefix + ".key" file has already been written
  if key is not None:
    write_tinker_key(key, mmkey, inactive=inactive, charges=charges)
  else:
    assert not inactive and not charges, "key must be specified to use inactive atoms or charges"

  if grad:
    # Y: yes analytical gradient, N: no numerical gradient, N: no breakdown by component
    Gout = subprocess.check_output([os.path.join(TINKER_PATH, "testgrad"), mminp, "Y", "N", "N"])
    Emm, Gmm = read_tinker_grad(Gout)
  if components is not None or not grad:
    Eout = subprocess.check_output([os.path.join(TINKER_PATH, "analyze"), mminp, "E"])
    if components is None:
      Emm = read_tinker_energy(Eout)
    else:
      Emm, Emmcomp = read_tinker_energy(Eout, components=True)
      components.update(Emmcomp)

  if cleanup:
    try:
      os.remove(mminp)
      if key is not None:
        os.remove(mmkey)
    except: pass

  return Emm, (Gmm if grad else None)
