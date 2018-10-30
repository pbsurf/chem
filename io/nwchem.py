import os, time
from . import cclib_open
from ..basics import *
from ..data.elements import ELEMENTS


NWCHEM_PATH = os.getenv('NWCHEM_PATH', '')

# For QM/MM calc:
# print low necessary to prevent failure with large number of charges; also makes output file much smaller
#nwchem_cclib(trypsin, task="print low\ntask scf gradient", qmatoms=qmatoms, caps=caps, charges=charges, basis=True)

# Write PDB for NWChem MM (or QM/MM); NWChem MM doesn't seem to support inactive atoms properly (i.e. like
#  Tinker), so we can't reliably compare QM/MM results between NWChem and our code
# not handled: need to make sure 2H, 3H, 4H used for N-terminal residues
def nwchem_pdb(mol, filename):
  """ write a PDB file that can be read by NWChem """
  # temporarily renumber PDB residues, since NWChem doesn't support insert codes
  old_pdb_nums = mol.pdb_num
  mol.pdb_num = [str(ii+1) for ii in range(mol.nresidues)]
  write_pdb(mol, filename, format='NWCHEM')
  mol.pdb_num = old_pdb_nums


def read_nwchem_mm(filename, coords=False):
  """ read E and G (and optionally coords) from NWChem MM or QM/MM .out file """
  E, G, R = None, [], []
  with open(filename) as f:
    l = skiptoline(f, " Energy total")
    E = float(l.split()[2])
    skiptoline(f, " Solute coordinates and forces", extra=3)
    fields = f.readline().split()
    while len(fields) == 10:
      R.append([float(x) for x in fields[4:7]])
      G.append([float(x) for x in fields[7:10]])
      fields = f.readline().split()
  return (E, G, R) if coords else (E, G)


# Point charges (Bq) must be included in geometry section to get gradient on them - doesn't seem to be any way
#  to get NWChem to print grad for charges in bq section - not even in binary RTDB (.db) file
def write_nwchem_inp(mol, filename, task, header='', r=None, qmatoms=None, caps=[], charges=[], basis=None):
  """ Write NWChem input file to `filename` for `qmatoms` in `mol`, `caps`, and `charges`.  `basis` can be
    string identifying basis to be used for all atoms, True to use qmbasis for each atom, or false if basis
    information is included in `header`.  NWChem `task` (including "task") passed separately since it must go
    at end of file
  """
  # "start" is required to prevent NWChem from running in "restart" mode if .db file is present
  strs = ["start", header, ""]
  bstrs = ["  * library " + basis] if type(basis) is str else []
  strs.append("geometry noautoz noprint nocenter noautosym")
  # QM atoms
  qmatoms = mol.listatoms() if qmatoms is None else qmatoms
  r = mol.r if r is None else r
  for ii, qi in enumerate(qmatoms):
    a = mol.atoms[qi]
    # NWChem requires atom name start with element name
    name = ELEMENTS[a.znuc].symbol.upper()
    if basis == True and getattr(a, 'qmbasis', None):
      name += str(ii)
      bstrs.append("  " + name + " library " + a.qmbasis)
    strs.append("  %-6s      %13.8f %13.8f %13.8f" % (name, r[qi][0], r[qi][1], r[qi][2]))

  # link/cap atoms
  for ii, cap in enumerate(caps):
    name = ELEMENTS[getattr(cap, 'znuc', 1)].symbol + "_L"
    if basis == True and getattr(cap, 'qmbasis', None):
      name += str(ii)
      bstrs.append("  " + name + " library " + cap.qmbasis)
    strs.append("  %-6s      %13.8f %13.8f %13.8f" % (name, cap.r[0], cap.r[1], cap.r[2]))

  # point charges - NWChem requires charges with same name to have same charge, so use unique names
  for ii, q in enumerate(charges):
    strs.append("  Bq%-5d     %13.8f %13.8f %13.8f  charge %13.8f" % (ii, q.r[0], q.r[1], q.r[2], q.qmq))
  # end geometry
  strs.append("end")

  # must explicitly specify total charge if it is not an integer
  if charges:
    strs.append("\ncharge " + str(sum(q.qmq for q in charges)))

  if bstrs:
    strs.append("\nbasis noprint")
    strs.extend(bstrs)
    strs.append("end")

  strs.extend(['', task, ''])
  with open(filename, 'w') as f:
    f.write("\n".join(strs))  # writelines() doesn't add '\n'!


# May want to try getting gradient from binary .db file (use print debug to see its format) to get more
#  significant digits

def nwchem_cclib(mol, r=None, prefix=None, cleanup=False, **kwargs):
  """ generate NWChem input, run NWChem, and return resulting log parsed by cclib """
  # some duplication with gamess_cclib() here
  if prefix is None:
    prefix = "qmtmp_%d" % time.time()
    cleanup = True

  # nwchem generates a bunch of extra files with go into directory it's run from by default
  splitfix = prefix.rsplit('/', 1)
  qmdir, fprefix = (splitfix[0] + '/', splitfix[1]) if len(splitfix) > 1 else ('', prefix)
  qminp = fprefix + ".nw"
  qmout = fprefix + ".log"

  write_nwchem_inp(mol, qmdir + qminp, r=r, **kwargs)
  # changed rungms to clear scratch files instead of complaining
  qmcmd = "cd ./%s && %s %s > %s 2> %s.stderr" % (qmdir, os.path.join(NWCHEM_PATH, 'nwchem'), qminp, qmout, fprefix)
  assert os.system(qmcmd) == 0, "Error running QM program: " + qmcmd
  mol_cc = cclib_open(qmdir + qmout)

  if cleanup and prefix:
    try: os.system("rm " + prefix + ".*")
    except: pass

  return mol_cc
