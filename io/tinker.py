# Tinker interface

import numpy as np
import os, subprocess, time
from cStringIO import StringIO

from ..basics import *
from ..molecule import Atom, Molecule, get_header
from ..data.elements import ELEMENTS


TINKER_PATH = os.getenv('TINKER_PATH')

# TODO: support multiple concatenated xyz files (aka Tinker archive)
def parse_xyz(xyz, charges=None):
  """ XYZ data read from file or string `xyz`. Tinker and generic (name and position only) xyz is
    supported.  MM charges will be read from `charges` or a ".charges" file if present, assumed to contain
    lines of the form "<index> <don't care> <MM force field charge>"
  """
  # xyz file
  if '\n' in xyz:
    fxyz = StringIO(xyz)
    filename = None
  else:
    filename = xyz
    fxyz = open(filename, 'r')
  l = fxyz.readline().split(None, 1)
  natoms = int(l[0])
  header = l[1].strip() if len(l) > 1 else ''
  l = fxyz.readline().split()
  # for generic (non-Tinker) xyz, line 2 is comment; for Tinker xyz with periodic box, line 2 specifies box
  #  and values contain '.', for which isdigit() returns False
  if not l[0].isdigit():
    l = fxyz.readline().split()
  # Tinker xyz doesn't necessarily start atom numbering at 1
  offset = int(l[0])
  # charge file is optional
  fq = None
  q = [0, 0, 0]
  if charges == 'generate':
    analysis = subprocess.check_output([os.path.join(TINKER_PATH, 'analyze'), filename, 'P'])
    idx = analysis.find('Atomic Partial Charge Parameters :')
    charges = analysis[idx:]
  if charges is not None:
    fq = StringIO(charges)
  elif filename is not None:
    try:
      charge_file = '.'.join(filename.split('.')[:-1]) + '.charges'
      fq = open(charge_file, 'r')
    except IOError: pass
  if fq:
    # skip to first charge
    lq = fq.readline()
    while lq and not lq.strip().startswith('1'):
      lq = fq.readline()
    q = lq.split()

  mol = Molecule()
  while len(l) > 3 and len(q) > 2:
    if len(l) == 4:
      # to support generic xyz format
      l = [0] + l + [0]
    # (some) xyz files use forcefield atom name, which may include extra characters
    elem = l[1][0] if l[1][0] in 'HCNOS' else l[1]
    mol.atoms.append(Atom(
      name=l[1],
      r=np.array(l[2:5], 'd'),
      znuc=ELEMENTS[elem].number,
      mmq=float(q[2]),
      mmtype=int(l[5]),
      # note conversion to zero-based indexing
      mmconnect=[int(x) - offset for x in l[6:]]
    ))
    l = fxyz.readline().split()
    q = fq.readline().split() if fq else q
  mol.header = header
  if len(mol.atoms) != natoms:
    print("Warning: only %d atoms found in xyz file but %d expected" % (len(mol.atoms), natoms))
  # ---
  fxyz.close()
  if fq:
    fq.close()
  return mol


# simple xyz reader to read just geometry from xyz file
def read_tinker_geom(filename):
  """ Read cartesian geom from TINKER .xyz file """
  XYZ = []
  with open(filename, 'r') as input:
    input.readline()  # skip first line
    l = input.readline().split()
    # skip 2nd line if it doesn't start with int
    if not l[0].isdigit():
      l = input.readline().split()
    while len(l) > 4:
      XYZ.append( [float(x) for x in l[2:5]] )
      l = input.readline().split()
  return np.array(XYZ)


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
    natoms = dict(angle=3, improper=4, torsion=4).get(type, 2)
    E[type] = {}
    while tokens:
      atoms = tuple(int(a.split('-')[0])-1 for a in tokens[1:1+natoms])
      atoms = atoms[::-1] if atoms[-1] < atoms[0] else atoms
      E[type][atoms] = float(tokens[-1])/KCALMOL_PER_HARTREE  # energy is always last item on line
      tokens = f.readline().split()
  f.close()
  return E


def write_tinker_xyz(mol, filename, r=None, title=None):
  # since Tinker mmtype > 0, we'll just use bool(mmtype); gap in numbering is OK
  r = mol.r if r is None else r
  lines = [" %5d  %-3s %11.6f %11.6f %11.6f %5d %s\n" % ( ii+1, a.name, r[ii][0], r[ii][1], r[ii][2], a.mmtype,
      " ".join(["%5d" % (x+1) for x in a.mmconnect]) ) for ii, a in mol.enumatoms() if a.mmtype ]
  with open(filename, 'w') as f:
    f.write("  %d  %s\n" % (len(lines), title or get_header(mol)))
    f.write("".join(lines))


# inactive atoms are assigned to a group for which intra-group interactions are set to 0; just using inactive
#  keyword isn't good because testgrad won't print gradient for inactive atoms
def write_tinker_key(key, filename, inactive=None, charges=None):
  with open(filename, 'w') as f:
    f.write(key)
    if inactive:
      # tinker only reads first ~100 chars of line, so split group across lines as needed
      f.write("\n".join("group 1   " + " ".join("%d" % x for x in chunk) for chunk in chunks(inactive, 10)))
      f.write("\ngroup-select 1 1 0.0\n\n")
    if charges:
      # negative arg for TINKER charge keyword applies value to individual atoms instead of atom type
      f.write("".join(["charge %4d %7.3f\n" % (-ii, q) for ii,q in charges]))


# should we take a generic keyopts dict instead of inactive and charges?
# for grad and components, pass True to calculate, False to skip but include in returns, and None to exclude
#  from returns ... this is done to ease unpacking
# An alternative would be to require a dict to passed in to be filled with components
def tinker_EandG(mol, r=None, prefix=None, key=None, inactive=None, charges=None, title=None,
    grad=True, components=None, cleanup=False):
  """ return energy and optionally gradient and components of energy of `mol` calculated by Tinker """
  if prefix is None:
    prefix = "mmtmp_%d" % time.time()
    cleanup = True
  mminp = prefix + ".xyz"
  mmkey = prefix + ".key"

  write_tinker_xyz(mol, mminp, r=r, title=title)
  # writing key file can be skipped if prefix + ".key" file has already been written
  if key is not None:
    write_tinker_key(key, mmkey, inactive=inactive, charges=charges)

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
