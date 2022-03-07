import numpy as np
from io import StringIO
from ..basics import *
from ..molecule import Atom, Residue, Molecule
from ..data.elements import ELEMENTS


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


def get_header(mol):
  """ return a header for use when writing molecule to PDB or XYZ file """
  header = getattr(mol, 'header', None)
  if not header:
    import time
    today = time.strftime("%d-%b-%y", time.localtime()).upper()
    header = "CUSTOM MOLECULE                         %s   XXXX" % today
  return header


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
  with open(filename, 'x') as f:
    f.write("".join(lines))


def write_xyz(mol, *args, **kwargs):
  return write_tinker_xyz(mol, *args, pdbhack=True, **kwargs)
