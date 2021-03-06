# PDB (Protein Databank) file I/O
# Notes:
# - because PDB format is line-oriented (and tabular), parsing is fairly trivial; adapting output of a
#  preexisting parser would have been more work than just writing this
# - PDB spec requires, and we assume, that CONECT records exists for S-S bonds, so we ignore SSBOND records
# - we will typically be loading PDB files generated by, e.g., Tinker, as opposed to directly from RCSB

# Tinker modules which handle PDB parsing, adding hydrogens, and atom type assignment:
#  initres.f: residue names to biotype codes
#  readpdf.f: mapping PDB atom names to standard names (e.g. 3HG2 -> HG23)
#  pdbxyz.f: create atoms, add hydrogens

from __future__ import absolute_import  # so that import pdb chooses global
import numpy as np
import os, subprocess, copy

from ..basics import setattrs, read_file
from ..molecule import Atom, Residue, Molecule, guess_bonds
from ..data.elements import ELEMENTS
from ..data.pdb_data import *
from .xyz import get_header

TWO_LETTER_ELEMENTS = frozenset([ELEMENTS[z].symbol.upper() for z in range(1, 83) if len(ELEMENTS[z].symbol) == 2])
DUMMY_ATOM = Atom(name='X', znuc=0, mmconnect=[])


def download_pdb(id, dir=''):
  """ download PDB `id` from RCSB to directory `dir` (default: current directory) """
  assert len(id) == 4, "Invalid PDB ID"
  dest = os.path.join(dir, id + ".pdb.gz")
  # piping through gunzip directly creates problem of removing file on error
  os.system("wget -O {1} https://files.rcsb.org/download/{0}.pdb.gz && gunzip {1}".format(id, dest))


def apply_pdb_bonds(mol):
  """ populate atom.mmconnect lists with bonds for standard PDB residues """
  from ..data.pdb_bonds import PDB_BONDS
  prevCidx, prevO3idx, prev_chain = None, None, None
  for residx, residue in enumerate(mol.residues):
    if residue.chain != prev_chain:
      prevCidx, prevO3idx = None, None
    prev_chain = residue.chain
    stdbonds = PDB_BONDS.get(residue.name)
    if not stdbonds:
      # unknown residue - bonds should already have been created with CONECT records
      prevCidx, prevO3idx, prev_chain = None, None, None
      continue
    # get mapping from atom names to numbers for this residue
    resatoms = residue.atoms
    name_to_idx = dict([(mol.atoms[ii].name, ii) for ii in resatoms])
    # '-C' is 'C' of previous residue; "-O3'" is O3' of previous nucleotide
    if prevCidx:
      name_to_idx['-C'] = prevCidx
    if prevO3idx:
      name_to_idx["-O3'"] = prevO3idx
    prevCidx = name_to_idx.get('C')
    prevO3idx = name_to_idx.get("O3'")
    # hydrogens on terminal N are named H1,H2,H3 in most PDB files, but PDB_BONDS uses H instead of H1
    if residue.name in PDB_PROTEIN and 'H1' in name_to_idx and 'H' not in name_to_idx:
      name_to_idx['H'] = name_to_idx.pop('H1')
    # iterate over name_to_idx instead of resatoms to handle '-C', etc correctly
    for n0, i0 in name_to_idx.items():
      if n0 in stdbonds:
        mol.atoms[i0].mmconnect.extend([name_to_idx[n1] for n1 in stdbonds[n0] if n1 in name_to_idx])
      elif residue.name != 'HOH':
        # Tinker uses H for both water hydrogens (instead of H1, H2) and writes its own CONECT records
        print("Warning: residue %d (%s) contains invalid atom %s (%d)" % (residx, residue.name, n0, i0))
  # Remove any duplicates from mmconnect
  for atom in mol.atoms:
    atom.mmconnect = list(set(atom.mmconnect))
  return mol


def fix_tinker_pdb(mol):
  # fix names for water hydrogens
  for residue in mol.residues:
    if residue.name == 'HOH' and len(residue.atoms) == 3:
      h1, h2 = [mol.atoms[ii] for ii in residue.atoms if mol.atoms[ii].znuc == 1]
      h1.name, h2.name = 'H1', 'H2'
  # Tinker `xyzpdb` doesn't include CONECT for disulfide bonds!
  sulfurs = [ii for ii, atom in mol.enumatoms() if atom.znuc == 16]
  if len(sulfurs) > 1:
    disulfides = guess_bonds(mol.r[sulfurs], mol.znuc[sulfurs])
    mol.add_bonds([(sulfurs[i], sulfurs[j]) for i,j in disulfides])
  return mol


def copy_residues(mol, mol_pdb):
  """ copy residues to `mol` from `mol_pdb`, which is assumed to have all heavy (non-hydrogen) atoms at the
    same positions as `mol`.  H atoms in standard residues are assigned names based on order in mmconnect
  """
  # original idea was to get argsort() mol.r and mol_pdb.r, but np.argsort doesn't take a key fn!
  from scipy.spatial.ckdtree import cKDTree
  from ..data.pdb_bonds import PDB_BONDS
  ck = cKDTree(mol_pdb.r)  # more efficient to build kd-tree with smaller set and query with larger set
  dists, locs = ck.query(mol.r, distance_upper_bound=0.001)
  mol.residues = [setattrs(copy.copy(res), atoms=[]) for res in mol_pdb.residues]
  for ii in range(len(locs)):
    if dists[ii] > 0.001:
      continue
    atom_pdb = mol_pdb.atoms[locs[ii]]
    atom = mol.atoms[ii]
    res = mol.residues[atom_pdb.resnum]
    atom.name = atom_pdb.name
    atom.znuc = atom_pdb.znuc  # determining element (esp. Ca) from .pdbb more reliable than .xyz
    atom.resnum = atom_pdb.resnum
    res.atoms.append(ii)
    if atom.znuc > 1 and res.name in PDB_BONDS:
      # process H atoms bonded to this atom and not present in mol_pdb
      h_names = [name for name in PDB_BONDS[res.name][atom.name] if name[0] == 'H']
      for jj in atom.mmconnect:
        if mol.atoms[jj].znuc == 1 and dists[jj] > 0.001:
          mol.atoms[jj].name = h_names.pop(0)
          mol.atoms[jj].resnum = atom.resnum
          res.atoms.append(jj)
  return mol


# Ref: http://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
# - Because PDB TER records get atom ("record") numbers, offset between PDB atom number and sequential atom
#  number will be different for different atoms in general.  And PDB atom numbers are used in CONECT records.
#  Thus, we pad atom list with dummy atoms, including for atom 0, to keep PDB and sequential numbers equal
#  and remove these atoms when parsing is complete
RES_RENAME = dict(CYX='CYS', HID='HIS', HIE='HIS')

def parse_pdb(PDB, fix_tinker=True, bonds=False, alt_loc='A', bfactors=False):
  """ PDB data read from file or string (assumed if contains \n). Problems with Tinker generated PDB files
    fixed if `fix_tinker` is true.  Returns Molecule object or, if PDB data contains multiple models, list of
    Molecule objects
  """
  is_tinker = True
  last_res_id, last_chain_id, chain_id = '0 ','',''
  mols = []
  mol = Molecule()
  lines = PDB.splitlines() if '\n' in PDB else read_file(PDB).splitlines()
  for line in lines:
    if line.startswith('ENDMDL') or (line.strip() == 'END' and len(mol.atoms) > 0):
      mol = mol.remove_atoms([ii for ii,a in mol.enumatoms() if a.znuc == 0])
      mol = fix_tinker_pdb(mol) if is_tinker and fix_tinker else mol
      mols.append(apply_pdb_bonds(mol) if bonds else mol)  #copy_residues(hydrogens[len(mols)], mol) if hydrogens
      mol = Molecule()
    elif line.startswith('HETATM') or line.startswith('ATOM'):
      # alternate location id - for now, require user to specify which alt_loc to use
      atom_alt_loc = line[16].strip()
      if atom_alt_loc and atom_alt_loc != alt_loc:
        continue
      atom = Atom()
      is_hetatm = line[0] == 'H'
      atom_num = int(line[6:11])
      atom.name = line[12:16].strip()
      # Element
      element = line[76:78].strip()
      if not element:
        # ATOM (but not HETATM) is always single letter element (H,C,N,O,S)
        if is_hetatm and len(atom.name) > 1 and atom.name[0:2] in TWO_LETTER_ELEMENTS:
          element = atom.name[0:2]
        else:
          # old style PDBs place second number for hydrogens in 1st column of name field
          element = atom.name[1] if atom.name[0].isdigit() else atom.name[0]
      atom.znuc = ELEMENTS[element.capitalize()].number
      # Residue
      res_type = line[17:21].strip()
      res_type = RES_RENAME.get(res_type, res_type)
      if not is_hetatm and res_type not in PDB_STANDARD:  #PDB_BONDS:
        print("Warning: unknown residue %s for ATOM %d" % (res_type, atom_num))
      elif is_hetatm and res_type in PDB_STANDARD:
        # prepend 'HET ' to standard residue listed as HETATMs, indicating it is not part of a polymer
        res_type = 'HET ' + res_type
      last_chain_id, chain_id = chain_id, line[21].strip()
      # RCSB PDB residues are numbered to preserve numbering for functionally important residues between
      #  different variants of a protein; missing residues in a given variant produce gaps in numbering, while
      #  inserts are identified by adding an insert code without changing number.
      res_id = line[22:27]  # residue number (line[22:26]) + insert code (line[26])
      if res_id != last_res_id or chain_id != last_chain_id or not res_id[:-1].strip():
        if not res_id[:-1].strip():
          print("Warning: invalid or missing residue ID for PDB atom %d" % atom_num)
        elif chain_id == last_chain_id and int(res_id[:-1]) < int(last_res_id[:-1].strip() or 0):
          print("Warning: smaller residue number %s following %s on chain %s" % (res_id, last_res_id, chain_id))
        last_res_id = res_id
        mol.residues.append(Residue(name=res_type, pdb_num=res_id.strip(), chain=chain_id, het=is_hetatm))
      mol.residues[-1].atoms.append(atom_num)
      atom.resnum = len(mol.residues) - 1
      # Position, etc
      atom.r = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
      #atom.occupancy = float(line[54:60].strip() or 100)
      b_factor = line[60:66].strip()
      if bfactors and b_factor:
        atom.b_factor = float(b_factor)
      #at.segi = line[72:76].strip()
      atom.mmconnect = []
      # add any dummy atoms necessary to get correct atom number
      mol.atoms.extend([DUMMY_ATOM]*(atom_num - len(mol.atoms)))
      mol.atoms.append(atom)
    elif line.startswith('CONECT'):
      # according to the PDB spec, CONECT records should be given for both atoms of bond,
      #  so we don't need to do atoms[a1].mmconnect.append(a0), etc
      # Note that there may be more than one CONECT record for an atom
      if is_tinker and fix_tinker:
        # Tinker doesn't format CONECT records correctly if >10K atoms
        tokens = line.split()
        a0 = int(tokens[1])
        mol.atoms[a0].mmconnect.extend([int(t) for t in tokens[2:6]])
      else:
        a0 = int(line[6:11])
        mol.atoms[a0].mmconnect.append(int(line[11:16]))
        try:
          mol.atoms[a0].mmconnect.append(int(line[16:21]))
          mol.atoms[a0].mmconnect.append(int(line[21:26]))
          mol.atoms[a0].mmconnect.append(int(line[26:31]))
        except: pass
      # we will ignore the additional fields for hydrogen bonds and salt bridges
    elif line.startswith('HEADER'):
      mol.header = line[10:].strip()
    # Tinker PDB header has only has empty SOURCE and COMPND lines after 1st line (HEADER)
    elif line.startswith('SOURCE') and line.strip() != 'SOURCE':
      is_tinker = False
    elif line.startswith('COMPND') and line.strip() != 'COMPND':
      is_tinker = False
    elif line.startswith('SEQRES') or line.startswith('REMARK'):
      is_tinker = False

  return mols if len(mols) > 1 else mols[0]


## writing

def write_pdb(mol, filename=None, r=None, title=None, format='TINKER'):
  """ write molecule as PDB file; residues missing a PDB residue number will be assigned to chain Z """
  lines = ["HEADER    %s\nCOMPND\nSOURCE\nREMARK   1 FORMAT: %s\n" % (title or get_header(mol), format)]
  idx_to_serial = []
  serial = 1
  r = mol.r if r is None else r
  for ii,atom in mol.enumatoms():
    res = mol.residues[atom.resnum]
    sym = ELEMENTS[atom.znuc].symbol.upper()
    # PDB atom name - attempt to find alias if different format
    name = atom.name
    resname = res.name
    if format != 'TINKER':
      try: name = PDB_ALIASES['*'][atom.name][format]
      except: pass
      try: name = PDB_ALIASES[res.name][atom.name][format]
      except: pass
      name = atom.name if name == '.' or name == '?' else name
      # for mixed case element names (Na, Cl, Ca, Mg, etc.), change residue name too
      if len(res.atoms) == 1 and len(atom.name) == 2 and res.name == atom.name:
        resname = name

    name = (' ' + name) if len(name) < 4 and not name[0].isdigit() and len(sym) == 1 else name
    hetatm = "HETATM" if res.het else "ATOM"
    if res.pdb_num:
      pdb_id = res.pdb_num + ' ' if res.pdb_num[-1].isdigit() else res.pdb_num
    else:
      pdb_id = str(atom.resnum + 1) + ' '  # trailing space is for the insertion code column
    lines.append("%-6s%5d %-4s %3s %1s%5s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n" % (hetatm, serial,
        name, resname, res.chain or 'Z', pdb_id, r[ii][0], r[ii][1], r[ii][2], 1.0, 0.0, sym))
    idx_to_serial.append(serial)
    serial += 1
    try: nextres = mol.residues[mol.atoms[ii+1].resnum]
    except: nextres = None
    if not res.het and (not nextres or nextres.het or nextres.chain != res.chain):
      lines.append("TER   %5d      %3s %1s%5s\n" % (serial, resname, res.chain or 'Z', pdb_id))
      serial += 1
  # CONECT records for HETATMs and disulfides
  for ii, atom in mol.enumatoms():
    res = mol.residues[atom.resnum]
    if res.het and res.name != 'HOH':
      # need to write ATOM -> HETATM connect record since we don't check ATOMs
      for jj in atom.mmconnect:
        res2 = mol.residues[mol.atoms[jj].resnum]
        if not res2.het or res2.name == 'HOH':
          lines.append("CONECT%5d%5d\n" % (idx_to_serial[jj], idx_to_serial[ii]))
      for jj in range(0, len(atom.mmconnect), 4):
        conect_to = "".join([ "%5d" % idx_to_serial[b] for b in atom.mmconnect[jj:jj+4] ])
        lines.append("CONECT%5d%s\n" % (idx_to_serial[ii], conect_to))
    elif res.name == 'CYS' and atom.name == 'SG':
      sg2 = next((jj for jj in atom.mmconnect if mol.atoms[jj].name == 'SG'), None)
      if sg2:
        lines.append("CONECT%5d%5d\n" % (idx_to_serial[ii], idx_to_serial[sg2]))
  lines.append("END\n")
  if filename is None:
    return "".join(lines)
  with open(filename, 'w') as f:
    f.write("".join(lines))
