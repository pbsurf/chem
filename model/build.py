import os
from ..basics import *
from ..molecule import *
from ..io import load_molecule


def download_pdb(id):
  assert len(id) == 4, "Invalid PDB ID"
  os.system("wget https://files.rcsb.org/download/{0}.pdb.gz && gunzip {0}.pdb.gz".format(id))


def _download_build_data():
  from chem.io import write_xyz
  from zipfile import ZipFile, ZIP_DEFLATED
  os.chdir(DATA_PATH)
  if not os.path.exists("bbindep_2015.txt"):
    # site is currently down but can't find this file anywhere else
    os.system("wget http://www.dynameomics.org/external/rotamer/bbindep_2015.zip && unzip bbindep_2015.zip")
  if not os.path.exists("aa-variants-v1.cif"):
    os.system("wget https://ftp.wwpdb.org/pub/pdb/data/monomers/aa-variants-v1.cif.gz && gunzip aa-variants-v1.cif.gz")
  if not os.path.exists('components.cif'):
    os.system("wget https://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif.gz && gunzip components.cif.gz")
  cif = []
  with open('components.cif') as f:
    openmm_app.pdbxfile.PdbxReader(f).read(cif)

  with ZipFile('components.zip', 'w', ZIP_DEFLATED) as zip:
    for res in cif:
      try:
        mol = load_cif_res(res)
        zip.writestr(mol.header, write_xyz(mol))
      except:  # errors usually due to missing coords in both model and ideal coords
        print("Error loading %s" % res.getObj('chem_comp').getValue('id'))


def load_cif_res(res):
  cc = res.getObj('chem_comp')
  resname = cc.getValue('three_letter_code')
  header = cc.getValue('id')  # used as dict key
  useideal = cc.getValue('pdbx_ideal_coordinates_missing_flag') == 'N'  #('pdbx_model_coordinates_missing_flag') == 'Y'
  coords = 'pdbx_model_Cartn_%s_ideal' if useideal else 'model_Cartn_%s'
  cca = res.getObj('chem_comp_atom')
  _a = cca.getAttributeIndex('atom_id')  # name
  _e = cca.getAttributeIndex('type_symbol')  # element
  _x, _y, _z = [cca.getAttributeIndex(coords % c) for c in 'xyz']
  unquote = lambda s: s[1:-1] if s[0] == '"' else s
  atoms = [Atom(unquote(row[_a]), row[_e], r=[row[_x], row[_y], row[_z]]) for row in cca.getRowList()]
  nametoidx = { a.name: ii for ii,a in enumerate(atoms) }
  # bonds
  bonds = None
  ccb = res.getObj('chem_comp_bond')
  if ccb is not None:
    _1 = ccb.getAttributeIndex('atom_id_1')
    _2 = ccb.getAttributeIndex('atom_id_2')
    bonds = [(nametoidx[row[_1]], nametoidx[row[_2]]) for row in ccb.getRowList()]
  # create molecule
  return Molecule(atoms, bonds=bonds, header=header).set_residue(resname, het=(resname not in PDB_PROTEIN))


def load_compound(name):
  from zipfile import ZipFile
  with ZipFile(DATA_PATH + '/components.zip', 'r') as zip:
    try:
      return load_molecule(zip.read(name).decode('utf-8'))
    except:
      return None


NME_xyz = """3 NME
  1  N    2.281  26.213 12.804  0  2 3
  2  HN1  2.033  25.273 12.493  0  1
  3  HN2  3.080  26.184 13.436  0  1
"""

TIP3P_xyz = """3 TIP3P Water
  1  O      0.000000    0.000000    0.000000     0     2     3
  2  H1    -0.239987    0.000000    0.926627     0     1
  3  H2     0.957200    0.000000    0.000000     0     1
"""

class PDB_RES_T:
  """ lazy loading class for CIF data """
  def __init__(self, file):
    self.file = file
    self._data, self._idx = None, None

  def _load(self):
    from chem.io.openmm import openmm_app
    cif = []
    with open(self.file) as f:
      openmm_app.pdbxfile.PdbxReader(f).read(cif)
    self._data = { mol.header: mol for mol in (load_cif_res(res) for res in cif) }
    # NH2 cap for C-terminal to support lone residues w/ Amber99
    self._data['NME'] = load_molecule(NME_xyz, residue='NME')
    self._data['HOH'] = load_molecule(TIP3P_xyz, residue='HOH')
    self._data['GLH'] = self._data['GLU']
    self._data['GLU'] = self._data['GLU_LL_DHE2']
    self._data['ASH'] = self._data['ASP']
    self._data['ASP'] = self._data['ASP_LL_DHD2']
    self._data['HID'] = self._data['HIS_LL_DHE2']
    self._data['HIE'] = self._data['HIS_LL_DHD1']

  def __call__(self, res=None):
    if self._data is None: self._load()
    return self._data if res is None else Molecule(self._data[res]) if res in self._data else None  # copy!

  def index(self, res=None, atom=None):
    """ get index of `atom` (name) in residue `res` """
    if self._idx is None:
      self._idx = {res: {a.name: ii for ii,a in enumerate(mol.atoms)} for res,mol in self().items()}
    return self._idx if res is None else self._idx.get(res) if atom is None else self._idx.get(res, {}).get(atom)


PDB_RES = PDB_RES_T(DATA_PATH + '/aa-variants-v1.cif')


# build a peptide
# phi, psi for common folds - from build_seq.py from pldserver1.biochem.queensu.ca/~rlc/work/pymol/
# - alpha helix: -57,-47; beta (antiparallel) sheet -139,-135; parallel sheet -119,-113; 3/10 helix -40.7,30
# refs: Tinker protein.f:676
def add_residue(mol, res, phi, psi):  #, chain='*'):
  """ add residue `res` to `mol` with `phi`, `psi` angles in degrees """
  nextmol = PDB_RES(res) if type(res) is str else res
  nextres = nextmol.residues[0]
  if not mol.residues:  # first residue?
    mol.append_atoms(nextmol)
    c2, n2, ca2, oxt2 = res_select(mol, mol.residues[-1], 'C,N,CA,OXT')
    mol.dihedral([n2, ca2, c2, oxt2], psi)
    return mol
  # find the id of the last residue in mol
  lastres = mol.residues[-1]  #[mol.atoms[mol.select(chain + ' * CA')[-1]].resnum]
  if lastres.pdb_num is None:
    lastres.pdb_num = 1
  nextres.chain = lastres.chain
  nextres.pdb_num = int(lastres.pdb_num) + 1
  oxt1, n2 = res_select(mol, lastres, 'OXT')[0], res_select(nextmol, nextres, 'N')[0]
  nextmol.remove_atoms(res_select(nextmol, nextres, 'H2,H3', squash=True))
  for ii in res_select(nextmol, nextres, 'H1', squash=True):  # rename H1 to H
    nextmol.atoms[ii].name = 'H'
  nextmol.r = nextmol.r - nextmol.atoms[n2].r + mol.atoms[oxt1].r
  # remove -OH from mol, -H from res, bond C to N, set peptide bond angle and dihed, set psi and phi
  mol.remove_atoms(res_select(mol, lastres, 'OXT,HXT', squash=True))
  mol.append_atoms(nextmol)
  nextres = mol.residues[-1]
  c1, n1, ca1, o1 = res_select(mol, lastres, 'C,N,CA,O')
  c2, n2, ca2, oxt2, h2 = res_select(mol, nextres, 'C,N,CA,OXT,H')
  mol.add_bonds([c1, n2])  # peptide bond
  mol.bond([c1, n2], 1.32)
  mol.angle([ca1, c1, n2], 112.7)
  if ca2 is None and nextres.name == 'NME':  # support NH2 C-terminal cap (NME)
    hn1, hn2 = res_select(mol, nextres, 'HN1,HN2')
    mol.angle([c1, n2, hn1], 118.0)
    mol.angle([c1, n2, hn2], 118.0)
    mol.dihedral([o1, c1, n2, hn1], 180.0, incl3=False)
    mol.dihedral([o1, c1, n2, hn2], 0.0, incl3=False)
  else:
    mol.angle([c1, n2, ca2], 121.0)
    mol.dihedral([ca1, c1, n2, ca2], 180)  # "omega" angle, usually 180 deg due to partial dbl bond nature of peptide bond
    mol.dihedral([c1, n2, ca2, c2], phi)
    mol.dihedral([n2, ca2, c2, oxt2], psi)  # oxt2 ~ n3
    # fix position of H
    mol.angle([ca2, n2, h2], 121.0)
    mol.dihedral([c2, ca2, n2, h2], phi - 180.0, incl3=False)  # move only h2, not c1
  return mol


# print(np.array(get_bb_angles(mol))/np.pi*180)
def get_bb_angles(mol, chain='*'):
  """ get peptide backbone angles """
  c = mol.select(chain + ' * C', sort=True)
  n = mol.select(chain + ' * N', sort=True)
  ca = mol.select(chain + ' * CA', sort=True)
  #print(c, n, ca)
  #assert all(mol.atoms[c[ii]].resnum == mol.atoms[n[ii]].resnum
  #    and mol.atoms[c[ii]].resnum == mol.atoms[ca[ii]].resnum for ii in range(len(c))), "Bad atom order"
  phi = [ mol.dihedral([c[ii-1], n[ii], ca[ii], c[ii]]) for ii in range(1, len(c)) ]
  psi = [ mol.dihedral([n[ii], ca[ii], c[ii], n[ii+1]]) for ii in range(len(c)-1) ]
  omega = [ mol.dihedral([ca[ii], c[ii], n[ii+1], ca[ii+1]]) for ii in range(len(c)-1) ]
  return phi, psi, omega


def sort_chains(mol):
  """ sort residues by chain """
  new2old = sorted(range(len(mol.residues)), key=lambda ii: (mol.residues[ii].chain, ii))
  old2new = argsort(new2old)
  for a in mol.atoms:
    a.resnum = old2new[a.resnum]
  mol.residues = [mol.residues[ii] for ii in new2old]
  return mol


def sort_atoms(mol):
  """ sort atoms of `mol` by resnum and canonical order within residue """
  new2old = sorted(mol.listatoms(), key=lambda ii: (mol.atoms[ii].resnum,
      PDB_RES.index().get(mol.atomres(ii).name, {}).get(mol.atoms[ii].name, -1)))
  old2new = argsort(new2old)
  #out = Molecule(mol)
  for res in mol.residues:
    res.atoms = []
  mol.atoms, oldatoms = [], mol.atoms
  for new,old in enumerate(new2old):
    a = oldatoms[old]
    mol.atoms.append(a)
    a.mmconnect = [old2new[ii] for ii in a.mmconnect]
    mol.residues[a.resnum].atoms.append(new)
  return mol


# replace sidechain
def mutate_residue(mol, resnum, newres):
  """ replace residue number `resnum` in `mol` with residue (name or object) `newres` """
  newres = PDB_RES(newres) if type(newres) is str else Molecule(newres)
  # align backbone of new residue to old
  resnum = resnum if type(resnum) is int else mol.atoms[ mol.select(resnum)[0] ].resnum
  oldbb = res_select(mol, resnum, 'C,CA,N')
  newbb = res_select(newres, 0, 'C,CA,N')
  newca = newbb[1]
  newres.r = apply_affine(alignment_matrix(newres.r[newbb], mol.r[oldbb]), newres.r)
  # replace sidechain (GLY has HA2,HA3 instead of HA,CB)
  bbnames = set(['C','N','O','H','H1','H2','H3','OXT','HXT','CA','HA','HA2'])
  mol.remove_atoms([ii for ii in mol.residues[resnum].atoms if mol.atoms[ii].name not in bbnames])
  s1 = next(ii for ii in newres.atoms[newca].mmconnect if newres.atoms[ii].name not in ['HA','HA2','N','C'])
  schead = newres.atoms[s1].name
  notsc, sc = newres.partition_mol(newca, s1)
  newres.remove_atoms(notsc)
  mol.append_atoms(newres, residue=resnum)
  mol.add_bonds(res_select(mol, resnum, 'CA,%s' % schead))
  mol.residues[resnum].name = newres.residues[0].name
  ha = res_select(mol, resnum, 'HA,HA2', squash=True)[0]
  mol.atoms[ha].name = 'HA2' if schead == 'HA3' else 'HA'  # fix GLY
  return sort_atoms(mol)  #if sort else mol


# another option is OpenMM Modeller.addHydrogens()
# for each missing atom, align heavy atom and its bound neighbors, then place missing atom
# TODO:
# - ARG hydrogens aren't planar
def add_hydrogens(mol):
  """ add hydrogens and bonds to `mol` based on residues """
  prevchain = False
  chargedres = dict(ASP=['HD2'], GLU=['HE2'], HIS=['HD1'])  # hydrogens normally absent at pH 7.4
  for resnum, res in enumerate(mol.residues):
    nextres = getitem(mol.residues, resnum+1)
    if res.name not in PDB_PROTEIN:
      ref = PDB_RES(res.name) or load_compound(res.name)
    elif prevchain != res.chain:
      ref = PDB_RES(res.name + '_LSN3')  # N-terminal
    elif nextres is None or nextres.chain != res.chain or nextres.name not in PDB_PROTEIN:
      ref = PDB_RES(res.name + '_LEO2')  # C-terminal
    else:
      ref = PDB_RES(res.name + '_LL')
    if ref is None:
      print("No template for residue %s" % res.name)
      continue
    refnametoidx = {a.name: ii for ii,a in enumerate(ref.atoms)}
    resnametoidx = {mol.atoms[ii].name: ii for ii in res.atoms}
    # missing atoms
    missing = (set(refnametoidx.keys()) - set(resnametoidx.keys()))
    # can't use the "_L_DHxx" variants because the H atom may be included in the PDB (so we need to add bond)
    missing -= set(chargedres.get(res.name, []))
    if res.name in ['CYS', 'CYX'] and any(mol.atoms[ii].resnum != resnum
        for ii in mol.atoms[resnametoidx['SG']].mmconnect):
      missing -= set(['HG'])  # or we could use 'CYS_*_DHG' variant
    single = len(res.atoms) == 1
    if single:  # for HOH; note affine transforms are applied from right!
      a0 = mol.atoms[res.atoms[0]]
      align = np.dot(translation_matrix(-ref.atoms[refnametoidx[a0.name]].r),
          np.dot(to4x4(random_rotation()), translation_matrix(a0.r)))
    for m in missing:
      ma = ref.atoms[refnametoidx[m]]
      if ma.znuc > 1:
        print("Heavy atom %s missing in residue %d (%s %s %s)" % (m, resnum, res.name, res.chain, res.pdb_num))
      if not single:
        M2 = set(ma.mmconnect)
        resalign = []
        while len(resalign) < 3 and len(resalign) < len(res.atoms):
          M2 = M2 | set(ref.get_bonded(M2))
          resalign = [resnametoidx[ref.atoms[ii].name] for ii in M2 if ref.atoms[ii].name in resnametoidx]
        refalign = [refnametoidx[mol.atoms[ii].name] for ii in resalign]
        align = alignment_matrix(ref.r[refalign], mol.r[resalign])
      mol.append_atoms(ma, r=apply_affine(align, ma.r), residue=resnum)  # this makes copy of atom
      mol.atoms[-1].mmconnect = []
    # bonds
    resnametoidx = {mol.atoms[ii].name: ii for ii in res.atoms}  # or should we update for every atom added?
    refbonds = [(ref.atoms[a1].name, ref.atoms[a2].name) for (a1, a2) in ref.get_bonds()]
    resbonds = [(resnametoidx.get(n1), resnametoidx.get(n2)) for (n1, n2) in refbonds]
    mol.add_bonds([(a1,a2) for (a1,a2) in resbonds if a1 is not None and a2 is not None])
    if prevchain == res.chain and prevC and res.name in PDB_PROTEIN:
      N2, H2, CA2 = [ resnametoidx.get(a) for a in ['N', 'H', 'CA'] ]
      mol.add_bonds( (prevC, N2) )  # peptide bond
      # fix position of H (except for PRO)
      if H2 is not None:
        mol.dihedral([prevC, CA2, N2, H2], 180.0, incl3=False)
        mol.angle([prevC, N2, H2], 118.0)
    prevC = resnametoidx['C'] if res.name in PDB_PROTEIN else None
    prevchain = res.chain

  return sort_atoms(mol)


# rotamers
# see www.cgl.ucsf.edu/chimera/docs/ContributedSoftware/rotamers/framerot.html
ROTAMERS = {}

def _load_rotamers():
  import csv
  ROTAMERS['_stats'] = {}
  with open(DATA_PATH + '/bbindep_2015.txt', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
      ROTAMERS.setdefault(row[0], []).append([ float(s) for s in row[5:9] if s.strip() != '-' ])
      ROTAMERS['_stats'].setdefault(row[0], []).append(float(row[9])/100)
  # sort by likelihood
  for res,ps in ROTAMERS['_stats'].items():
    ord = argsort(ps)[::-1]
    ROTAMERS[res] = [ROTAMERS[res][ii] for ii in ord]
    ROTAMERS['_stats'][res] = [ROTAMERS['_stats'][res][ii] for ii in ord]


def get_rotamers(res=None):
  if not ROTAMERS: _load_rotamers()
  return ROTAMERS.get(res, []) if res is not None else ROTAMERS


def set_rotamer(mol, resnum, angles):
  XG = res_select(mol, resnum, 'CG,CG1,OG,OG1,SG', squash=True)
  XD = res_select(mol, resnum, 'CD,CD1,ND1,OD1,SD', squash=True)  # CD1: LEU; ND1: HIS; OD1: ASN; SD: MET
  XE = res_select(mol, resnum, 'CE,NE,OE1', squash=True)  # CE: MET, LYS; NE: ARG; OE1: GLU, GLN
  XZ = res_select(mol, resnum, 'CZ,NZ', squash=True)  # CZ: ARG; NZ: LYS
  atoms = res_select(mol, resnum, 'N,CA,CB') + XG + XD + XE + XZ
  for ii,ang in enumerate(angles):
    mol.dihedral(atoms[ii:ii+4], ang)
  return mol
