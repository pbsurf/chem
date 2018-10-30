"""
Connectivity for atoms (including hydrogens) of standard PDB residues (including nucleic acids)
Generated from
  https://github.com/pandegroup/openmm/blob/master/wrappers/python/simtk/openmm/app/data/residues.xml
with code below

Similar files can be found by searching github with e.g. "hg21 hb3 ..."; pymol/modules/chempy/bonds.py
 is interesting as it indicates double bonds

https://github.com/rlabduke/reduce/blob/master/reduce_wwPDB_het_dict.txt - 40MB of HET residue connectivity

Authoritative source for residue info is http://www.wwpdb.org/data/ccd
... the components.cif file here is probably the format we'd want to support if we implemented adding missing
atoms here
"""

if __name__ == "__main__":
  import pprint
  import xml.etree.ElementTree as ET
  tree = ET.parse('residues.xml')
  residues = tree.getroot()
  bonds = {}
  for residue in residues.iter('Residue'):
    resbonds = {}
    for bond in residue.iter('Bond'):
      a1, a2 = bond.get('from'), bond.get('to')
      resbonds.setdefault(a1, []).append(a2)
      resbonds.setdefault(a2, []).append(a1)
    bonds[residue.get('name')] = resbonds
  with open('pdb_bonds.out', 'w') as f:
    f.write(pprint.pformat(bonds, indent=2))


## Residues
PDB_PROTEIN = set(['ILE', 'SER', 'UNK', 'GLY', 'HIS', 'TRP',  'PCA', 'GLU', 'AIB', 'CYS', 'ASP', \
    'FOR', 'THR', 'LYS', 'PRO', 'PHE', 'ALA', 'ORN', 'MET', 'GLN', 'ARG', 'VAL', 'ASN', 'TYR', 'LEU'])
PDB_RNA = set(['A', 'C', 'G', 'U'])
PDB_DNA = set(['DA', 'DC', 'DG', 'DT'])
PDB_STANDARD = set.union(PDB_PROTEIN, PDB_RNA, PDB_DNA)


## PDB atom names (mostly for hydrogen) used by different programs
# goal here is to defer processing of data file until first use ... not sure if this is the best approach
# - should we just make PDB_ALIASES a fn instead of a table?
class PDB_ALIASES_T:
  FIELDS = 'RES3 RES1 BMRB SC PDB UCSF MSI XPLOR SYBYL MIDAS TINKER AMBER GROMACS NWCHEM HINTS'.split()

  def load_aliases(self):
    import os
    self.ALIASES = {}
    with open(os.path.join(os.path.dirname(__file__), 'atom_name.tbl')) as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
          t = dict(zip(self.FIELDS, line.split()))
          if t['TINKER'] != '.' and t['TINKER'] != '?':
            if t['RES3'] not in self.ALIASES:
              self.ALIASES[t['RES3']] = {}
            self.ALIASES[t['RES3']][t['TINKER']] = t

  def __getitem__(self, idx):
    if not hasattr(self, 'ALIASES'):
      self.load_aliases()
    return self.ALIASES[idx]

PDB_ALIASES = PDB_ALIASES_T()


## Bonds for standard residues
PDB_BONDS = \
{ 'A': { "-O3'": ['P'],
         "C1'": ["C2'", "H1'", 'N9', "O4'"],
         'C2': ['H2', 'N1', 'N3'],
         "C2'": ["C1'", "C3'", "H2'", "O2'"],
         "C3'": ["C2'", "C4'", "H3'", "O3'"],
         'C4': ['C5', 'N3', 'N9'],
         "C4'": ["C3'", "C5'", "H4'", "O4'"],
         'C5': ['C4', 'C6', 'N7'],
         "C5'": ["C4'", "H5'", "H5''", "O5'"],
         'C6': ['C5', 'N1', 'N6'],
         'C8': ['H8', 'N7', 'N9'],
         "H1'": ["C1'"],
         'H2': ['C2'],
         "H2'": ["C2'"],
         "H3'": ["C3'"],
         "H4'": ["C4'"],
         "H5'": ["C5'"],
         "H5''": ["C5'"],
         'H61': ['N6'],
         'H62': ['N6'],
         'H8': ['C8'],
         "HO2'": ["O2'"],
         "HO3'": ["O3'"],
         "HO5'": ["O5'"],
         'HOP2': ['OP2'],
         'HOP3': ['OP3'],
         'N1': ['C2', 'C6'],
         'N3': ['C2', 'C4'],
         'N6': ['C6', 'H61', 'H62'],
         'N7': ['C5', 'C8'],
         'N9': ["C1'", 'C4', 'C8'],
         "O2'": ["C2'", "HO2'"],
         "O3'": ["C3'", "HO3'"],
         "O4'": ["C1'", "C4'"],
         "O5'": ["C5'", 'P', "HO5'"],
         'OP1': ['P'],
         'OP2': ['HOP2', 'P'],
         'OP3': ['HOP3', 'P'],
         'P': ["-O3'", "O5'", 'OP1', 'OP2', 'OP3']},
  'ACE': { 'C': ['CH3', 'H', 'O'],
           'CH3': ['C', 'H1', 'H2', 'H3'],
           'H': ['C'],
           'H1': ['CH3'],
           'H2': ['CH3'],
           'H3': ['CH3'],
           'O': ['C']},
  'AIB': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB1', 'CB2', 'N'],
           'CB1': ['CA', 'HB11', 'HB12', 'HB13'],
           'CB2': ['CA', 'HB21', 'HB22', 'HB23'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HB11': ['CB1'],
           'HB12': ['CB1'],
           'HB13': ['CB1'],
           'HB21': ['CB2'],
           'HB22': ['CB2'],
           'HB23': ['CB2'],
           'HO2': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HO2']},
  'ALA': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'HB1', 'HB2', 'HB3'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB1': ['CB'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'ARG': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD': ['CG', 'HD2', 'HD3', 'NE'],
           'CG': ['CB', 'CD', 'HG2', 'HG3'],
           'CZ': ['NE', 'NH1', 'NH2'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD2': ['CD'],
           'HD3': ['CD'],
           'HE': ['NE'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HH11': ['NH1'],
           'HH12': ['NH1'],
           'HH21': ['NH2'],
           'HH22': ['NH2'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'NE': ['CD', 'CZ', 'HE'],
           'NH1': ['CZ', 'HH11', 'HH12'],
           'NH2': ['CZ', 'HH21', 'HH22'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'ASN': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CG': ['CB', 'ND2', 'OD1'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD21': ['ND2'],
           'HD22': ['ND2'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'ND2': ['CG', 'HD21', 'HD22'],
           'O': ['C'],
           'OD1': ['CG'],
           'OXT': ['C', 'HXT']},
  'ASP': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CG': ['CB', 'OD1', 'OD2'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD2': ['OD2'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OD1': ['CG'],
           'OD2': ['CG', 'HD2'],
           'OXT': ['C', 'HXT']},
  'C': { "-O3'": ['P'],
         "C1'": ["C2'", "H1'", 'N1', "O4'"],
         'C2': ['N1', 'N3', 'O2'],
         "C2'": ["C1'", "C3'", "H2'", "O2'"],
         "C3'": ["C2'", "C4'", "H3'", "O3'"],
         'C4': ['C5', 'N3', 'N4'],
         "C4'": ["C3'", "C5'", "H4'", "O4'"],
         'C5': ['C4', 'C6', 'H5'],
         "C5'": ["C4'", "H5'", "H5''", "O5'"],
         'C6': ['C5', 'H6', 'N1'],
         "H1'": ["C1'"],
         "H2'": ["C2'"],
         "H3'": ["C3'"],
         "H4'": ["C4'"],
         'H41': ['N4'],
         'H42': ['N4'],
         'H5': ['C5'],
         "H5'": ["C5'"],
         "H5''": ["C5'"],
         'H6': ['C6'],
         "HO2'": ["O2'"],
         "HO3'": ["O3'"],
         "HO5'": ["O5'"],
         'HOP2': ['OP2'],
         'HOP3': ['OP3'],
         'N1': ["C1'", 'C2', 'C6'],
         'N3': ['C2', 'C4'],
         'N4': ['C4', 'H41', 'H42'],
         'O2': ['C2'],
         "O2'": ["C2'", "HO2'"],
         "O3'": ["C3'", "HO3'"],
         "O4'": ["C1'", "C4'"],
         "O5'": ["C5'", 'P', "HO5'"],
         'OP1': ['P'],
         'OP2': ['HOP2', 'P'],
         'OP3': ['HOP3', 'P'],
         'P': ["-O3'", "O5'", 'OP1', 'OP2', 'OP3']},
  'CYS': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'HB2', 'HB3', 'SG'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HG': ['SG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT'],
           'SG': ['CB', 'HG']},
  'DA': { "-O3'": ['P'],
          "C1'": ["C2'", "H1'", 'N9', "O4'"],
          'C2': ['H2', 'N1', 'N3'],
          "C2'": ["C1'", "C3'", "H2'", "H2''"],
          "C3'": ["C2'", "C4'", "H3'", "O3'"],
          'C4': ['C5', 'N3', 'N9'],
          "C4'": ["C3'", "C5'", "H4'", "O4'"],
          'C5': ['C4', 'C6', 'N7'],
          "C5'": ["C4'", "H5'", "H5''", "O5'"],
          'C6': ['C5', 'N1', 'N6'],
          'C8': ['H8', 'N7', 'N9'],
          "H1'": ["C1'"],
          'H2': ['C2'],
          "H2'": ["C2'"],
          "H2''": ["C2'"],
          "H3'": ["C3'"],
          "H4'": ["C4'"],
          "H5'": ["C5'"],
          "H5''": ["C5'"],
          'H61': ['N6'],
          'H62': ['N6'],
          'H8': ['C8'],
          "HO3'": ["O3'"],
          "HO5'": ["O5'"],
          'HOP2': ['OP2'],
          'HOP3': ['OP3'],
          'N1': ['C2', 'C6'],
          'N3': ['C2', 'C4'],
          'N6': ['C6', 'H61', 'H62'],
          'N7': ['C5', 'C8'],
          'N9': ["C1'", 'C4', 'C8'],
          "O3'": ["C3'", "HO3'"],
          "O4'": ["C1'", "C4'"],
          "O5'": ["C5'", 'P', "HO5'"],
          'OP1': ['P'],
          'OP2': ['HOP2', 'P'],
          'OP3': ['HOP3', 'P'],
          'P': ["-O3'", "O5'", 'OP1', 'OP2', 'OP3']},
  'DC': { "-O3'": ['P'],
          "C1'": ["C2'", "H1'", 'N1', "O4'"],
          'C2': ['N1', 'N3', 'O2'],
          "C2'": ["C1'", "C3'", "H2'", "H2''"],
          "C3'": ["C2'", "C4'", "H3'", "O3'"],
          'C4': ['C5', 'N3', 'N4'],
          "C4'": ["C3'", "C5'", "H4'", "O4'"],
          'C5': ['C4', 'C6', 'H5'],
          "C5'": ["C4'", "H5'", "H5''", "O5'"],
          'C6': ['C5', 'H6', 'N1'],
          "H1'": ["C1'"],
          "H2'": ["C2'"],
          "H2''": ["C2'"],
          "H3'": ["C3'"],
          "H4'": ["C4'"],
          'H41': ['N4'],
          'H42': ['N4'],
          'H5': ['C5'],
          "H5'": ["C5'"],
          "H5''": ["C5'"],
          'H6': ['C6'],
          "HO3'": ["O3'"],
          "HO5'": ["O5'"],
          'HOP2': ['OP2'],
          'HOP3': ['OP3'],
          'N1': ["C1'", 'C2', 'C6'],
          'N3': ['C2', 'C4'],
          'N4': ['C4', 'H41', 'H42'],
          'O2': ['C2'],
          "O3'": ["C3'", "HO3'"],
          "O4'": ["C1'", "C4'"],
          "O5'": ["C5'", 'P', "HO5'"],
          'OP1': ['P'],
          'OP2': ['HOP2', 'P'],
          'OP3': ['HOP3', 'P'],
          'P': ["-O3'", "O5'", 'OP1', 'OP2', 'OP3']},
  'DG': { "-O3'": ['P'],
          "C1'": ["C2'", "H1'", 'N9', "O4'"],
          'C2': ['N1', 'N2', 'N3'],
          "C2'": ["C1'", "C3'", "H2'", "H2''"],
          "C3'": ["C2'", "C4'", "H3'", "O3'"],
          'C4': ['C5', 'N3', 'N9'],
          "C4'": ["C3'", "C5'", "H4'", "O4'"],
          'C5': ['C4', 'C6', 'N7'],
          "C5'": ["C4'", "H5'", "H5''", "O5'"],
          'C6': ['C5', 'N1', 'O6'],
          'C8': ['H8', 'N7', 'N9'],
          'H1': ['N1'],
          "H1'": ["C1'"],
          "H2'": ["C2'"],
          "H2''": ["C2'"],
          'H21': ['N2'],
          'H22': ['N2'],
          "H3'": ["C3'"],
          "H4'": ["C4'"],
          "H5'": ["C5'"],
          "H5''": ["C5'"],
          'H8': ['C8'],
          "HO3'": ["O3'"],
          "HO5'": ["O5'"],
          'HOP2': ['OP2'],
          'HOP3': ['OP3'],
          'N1': ['C2', 'C6', 'H1'],
          'N2': ['C2', 'H21', 'H22'],
          'N3': ['C2', 'C4'],
          'N7': ['C5', 'C8'],
          'N9': ["C1'", 'C4', 'C8'],
          "O3'": ["C3'", "HO3'"],
          "O4'": ["C1'", "C4'"],
          "O5'": ["C5'", 'P', "HO5'"],
          'O6': ['C6'],
          'OP1': ['P'],
          'OP2': ['HOP2', 'P'],
          'OP3': ['HOP3', 'P'],
          'P': ["-O3'", "O5'", 'OP1', 'OP2', 'OP3']},
  'DT': { "-O3'": ['P'],
          "C1'": ["C2'", "H1'", 'N1', "O4'"],
          'C2': ['N1', 'N3', 'O2'],
          "C2'": ["C1'", "C3'", "H2'", "H2''"],
          "C3'": ["C2'", "C4'", "H3'", "O3'"],
          'C4': ['C5', 'N3', 'O4'],
          "C4'": ["C3'", "C5'", "H4'", "O4'"],
          'C5': ['C4', 'C6', 'C7'],
          "C5'": ["C4'", "H5'", "H5''", "O5'"],
          'C6': ['C5', 'H6', 'N1'],
          'C7': ['C5', 'H71', 'H72', 'H73'],
          "H1'": ["C1'"],
          "H2'": ["C2'"],
          "H2''": ["C2'"],
          'H3': ['N3'],
          "H3'": ["C3'"],
          "H4'": ["C4'"],
          "H5'": ["C5'"],
          "H5''": ["C5'"],
          'H6': ['C6'],
          'H71': ['C7'],
          'H72': ['C7'],
          'H73': ['C7'],
          "HO3'": ["O3'"],
          "HO5'": ["O5'"],
          'HOP2': ['OP2'],
          'HOP3': ['OP3'],
          'N1': ["C1'", 'C2', 'C6'],
          'N3': ['C2', 'C4', 'H3'],
          'O2': ['C2'],
          "O3'": ["C3'", "HO3'"],
          'O4': ['C4'],
          "O4'": ["C1'", "C4'"],
          "O5'": ["C5'", 'P', "HO5'"],
          'OP1': ['P'],
          'OP2': ['HOP2', 'P'],
          'OP3': ['HOP3', 'P'],
          'P': ["-O3'", "O5'", 'OP1', 'OP2', 'OP3']},
  'FOR': { '-C': ['N'],
           'C': ['H1', 'H2', 'O'],
           'H1': ['C'],
           'H2': ['C'],
           'H3': ['N'],
           'N': ['-C', 'H3'],
           'O': ['C']},
  'G': { "-O3'": ['P'],
         "C1'": ["C2'", "H1'", 'N9', "O4'"],
         'C2': ['N1', 'N2', 'N3'],
         "C2'": ["C1'", "C3'", "H2'", "O2'"],
         "C3'": ["C2'", "C4'", "H3'", "O3'"],
         'C4': ['C5', 'N3', 'N9'],
         "C4'": ["C3'", "C5'", "H4'", "O4'"],
         'C5': ['C4', 'C6', 'N7'],
         "C5'": ["C4'", "H5'", "H5''", "O5'"],
         'C6': ['C5', 'N1', 'O6'],
         'C8': ['H8', 'N7', 'N9'],
         'H1': ['N1'],
         "H1'": ["C1'"],
         "H2'": ["C2'"],
         'H21': ['N2'],
         'H22': ['N2'],
         "H3'": ["C3'"],
         "H4'": ["C4'"],
         "H5'": ["C5'"],
         "H5''": ["C5'"],
         'H8': ['C8'],
         "HO2'": ["O2'"],
         "HO3'": ["O3'"],
         "HO5'": ["O5'"],
         'HOP2': ['OP2'],
         'HOP3': ['OP3'],
         'N1': ['C2', 'C6', 'H1'],
         'N2': ['C2', 'H21', 'H22'],
         'N3': ['C2', 'C4'],
         'N7': ['C5', 'C8'],
         'N9': ["C1'", 'C4', 'C8'],
         "O2'": ["C2'", "HO2'"],
         "O3'": ["C3'", "HO3'"],
         "O4'": ["C1'", "C4'"],
         "O5'": ["C5'", 'P', "HO5'"],
         'O6': ['C6'],
         'OP1': ['P'],
         'OP2': ['HOP2', 'P'],
         'OP3': ['HOP3', 'P'],
         'P': ["-O3'", "O5'", 'OP1', 'OP2', 'OP3']},
  'GLN': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD': ['CG', 'NE2', 'OE1'],
           'CG': ['CB', 'CD', 'HG2', 'HG3'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HE21': ['NE2'],
           'HE22': ['NE2'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'NE2': ['CD', 'HE21', 'HE22'],
           'O': ['C'],
           'OE1': ['CD'],
           'OXT': ['C', 'HXT']},
  'GLU': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD': ['CG', 'OE1', 'OE2'],
           'CG': ['CB', 'CD', 'HG2', 'HG3'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HE2': ['OE2'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OE1': ['CD'],
           'OE2': ['CD', 'HE2'],
           'OXT': ['C', 'HXT']},
  'GLY': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'HA2', 'HA3', 'N'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA2': ['CA'],
           'HA3': ['CA'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'HIS': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD2': ['CG', 'HD2', 'NE2'],
           'CE1': ['HE1', 'ND1', 'NE2'],
           'CG': ['CB', 'CD2', 'ND1'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD1': ['ND1'],
           'HD2': ['CD2'],
           'HE1': ['CE1'],
           'HE2': ['NE2'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'ND1': ['CE1', 'CG', 'HD1'],
           'NE2': ['CD2', 'CE1', 'HE2'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'HOH': { 'H1': ['O'], 'H2': ['O'], 'O': ['H1', 'H2']},
  'ILE': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG1', 'CG2', 'HB'],
           'CD1': ['CG1', 'HD11', 'HD12', 'HD13'],
           'CG1': ['CB', 'CD1', 'HG12', 'HG13'],
           'CG2': ['CB', 'HG21', 'HG22', 'HG23'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB': ['CB'],
           'HD11': ['CD1'],
           'HD12': ['CD1'],
           'HD13': ['CD1'],
           'HG12': ['CG1'],
           'HG13': ['CG1'],
           'HG21': ['CG2'],
           'HG22': ['CG2'],
           'HG23': ['CG2'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'LEU': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD1': ['CG', 'HD11', 'HD12', 'HD13'],
           'CD2': ['CG', 'HD21', 'HD22', 'HD23'],
           'CG': ['CB', 'CD1', 'CD2', 'HG'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD11': ['CD1'],
           'HD12': ['CD1'],
           'HD13': ['CD1'],
           'HD21': ['CD2'],
           'HD22': ['CD2'],
           'HD23': ['CD2'],
           'HG': ['CG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'LYS': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD': ['CE', 'CG', 'HD2', 'HD3'],
           'CE': ['CD', 'HE2', 'HE3', 'NZ'],
           'CG': ['CB', 'CD', 'HG2', 'HG3'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD2': ['CD'],
           'HD3': ['CD'],
           'HE2': ['CE'],
           'HE3': ['CE'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HXT': ['OXT'],
           'HZ1': ['NZ'],
           'HZ2': ['NZ'],
           'HZ3': ['NZ'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'NZ': ['CE', 'HZ1', 'HZ2', 'HZ3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'MET': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CE': ['HE1', 'HE2', 'HE3', 'SD'],
           'CG': ['CB', 'HG2', 'HG3', 'SD'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HE1': ['CE'],
           'HE2': ['CE'],
           'HE3': ['CE'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT'],
           'SD': ['CE', 'CG']},
  'NH2': { '-C': ['N'], 'HN1': ['N'], 'HN2': ['N'], 'N': ['-C', 'HN1', 'HN2']},
  'NME': { '-C': ['N'],
           'C': ['H1', 'H2', 'H3', 'N'],
           'H': ['N'],
           'H1': ['C'],
           'H2': ['C'],
           'H3': ['C'],
           'HN2': ['N'],
           'N': ['-C', 'C', 'H', 'HN2']},
  'ORN': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD': ['CG', 'HD2', 'HD3', 'NE'],
           'CG': ['CB', 'CD', 'HG2', 'HG3'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD2': ['CD'],
           'HD3': ['CD'],
           'HE1': ['NE'],
           'HE2': ['NE'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'NE': ['CD', 'HE1', 'HE2'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'PCA': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD': ['CG', 'N', 'OE'],
           'CG': ['CB', 'CD', 'HG2', 'HG3'],
           'H': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'CD', 'H', 'H3'],
           'O': ['C'],
           'OE': ['CD'],
           'OXT': ['C', 'HXT']},
  'PHE': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD1': ['CE1', 'CG', 'HD1'],
           'CD2': ['CE2', 'CG', 'HD2'],
           'CE1': ['CD1', 'CZ', 'HE1'],
           'CE2': ['CD2', 'CZ', 'HE2'],
           'CG': ['CB', 'CD1', 'CD2'],
           'CZ': ['CE1', 'CE2', 'HZ'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD1': ['CD1'],
           'HD2': ['CD2'],
           'HE1': ['CE1'],
           'HE2': ['CE2'],
           'HXT': ['OXT'],
           'HZ': ['CZ'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'PRO': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD': ['CG', 'HD2', 'HD3', 'N'],
           'CG': ['CB', 'CD', 'HG2', 'HG3'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD2': ['CD'],
           'HD3': ['CD'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'CD', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'SER': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'HB2', 'HB3', 'OG'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HG': ['OG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OG': ['CB', 'HG'],
           'OXT': ['C', 'HXT']},
  'THR': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG2', 'HB', 'OG1'],
           'CG2': ['CB', 'HG21', 'HG22', 'HG23'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB': ['CB'],
           'HG1': ['OG1'],
           'HG21': ['CG2'],
           'HG22': ['CG2'],
           'HG23': ['CG2'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OG1': ['CB', 'HG1'],
           'OXT': ['C', 'HXT']},
  'TRP': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD1': ['CG', 'HD1', 'NE1'],
           'CD2': ['CE2', 'CE3', 'CG'],
           'CE2': ['CD2', 'CZ2', 'NE1'],
           'CE3': ['CD2', 'CZ3', 'HE3'],
           'CG': ['CB', 'CD1', 'CD2'],
           'CH2': ['CZ2', 'CZ3', 'HH2'],
           'CZ2': ['CE2', 'CH2', 'HZ2'],
           'CZ3': ['CE3', 'CH2', 'HZ3'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD1': ['CD1'],
           'HE1': ['NE1'],
           'HE3': ['CE3'],
           'HH2': ['CH2'],
           'HXT': ['OXT'],
           'HZ2': ['CZ2'],
           'HZ3': ['CZ3'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'NE1': ['CD1', 'CE2', 'HE1'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'TYR': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB2', 'HB3'],
           'CD1': ['CE1', 'CG', 'HD1'],
           'CD2': ['CE2', 'CG', 'HD2'],
           'CE1': ['CD1', 'CZ', 'HE1'],
           'CE2': ['CD2', 'CZ', 'HE2'],
           'CG': ['CB', 'CD1', 'CD2'],
           'CZ': ['CE1', 'CE2', 'OH'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB2': ['CB'],
           'HB3': ['CB'],
           'HD1': ['CD1'],
           'HD2': ['CD2'],
           'HE1': ['CE1'],
           'HE2': ['CE2'],
           'HH': ['OH'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OH': ['CZ', 'HH'],
           'OXT': ['C', 'HXT']},
  'U': { "-O3'": ['P'],
         "C1'": ["C2'", "H1'", 'N1', "O4'"],
         'C2': ['N1', 'N3', 'O2'],
         "C2'": ["C1'", "C3'", "H2'", "O2'"],
         "C3'": ["C2'", "C4'", "H3'", "O3'"],
         'C4': ['C5', 'N3', 'O4'],
         "C4'": ["C3'", "C5'", "H4'", "O4'"],
         'C5': ['C4', 'C6', 'H5'],
         "C5'": ["C4'", "H5'", "H5''", "O5'"],
         'C6': ['C5', 'H6', 'N1'],
         "H1'": ["C1'"],
         "H2'": ["C2'"],
         'H3': ['N3'],
         "H3'": ["C3'"],
         "H4'": ["C4'"],
         'H5': ['C5'],
         "H5'": ["C5'"],
         "H5''": ["C5'"],
         'H6': ['C6'],
         "HO2'": ["O2'"],
         "HO3'": ["O3'"],
         "HO5'": ["O5'"],
         'HOP2': ['OP2'],
         'HOP3': ['OP3'],
         'N1': ["C1'", 'C2', 'C6'],
         'N3': ['C2', 'C4', 'H3'],
         'O2': ['C2'],
         "O2'": ["C2'", "HO2'"],
         "O3'": ["C3'", "HO3'"],
         'O4': ['C4'],
         "O4'": ["C1'", "C4'"],
         "O5'": ["C5'", 'P', "HO5'"],
         'OP1': ['P'],
         'OP2': ['HOP2', 'P'],
         'OP3': ['HOP3', 'P'],
         'P': ["-O3'", "O5'", 'OP1', 'OP2', 'OP3']},
  'UNK': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG', 'HB1', 'HB2'],
           'CG': ['CB', 'HG1', 'HG2', 'HG3'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB1': ['CB'],
           'HB2': ['CB'],
           'HG1': ['CG'],
           'HG2': ['CG'],
           'HG3': ['CG'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']},
  'VAL': { '-C': ['N'],
           'C': ['CA', 'O', 'OXT'],
           'CA': ['C', 'CB', 'HA', 'N'],
           'CB': ['CA', 'CG1', 'CG2', 'HB'],
           'CG1': ['CB', 'HG11', 'HG12', 'HG13'],
           'CG2': ['CB', 'HG21', 'HG22', 'HG23'],
           'H': ['N'],
           'H2': ['N'],
           'H3': ['N'],
           'HA': ['CA'],
           'HB': ['CB'],
           'HG11': ['CG1'],
           'HG12': ['CG1'],
           'HG13': ['CG1'],
           'HG21': ['CG2'],
           'HG22': ['CG2'],
           'HG23': ['CG2'],
           'HXT': ['OXT'],
           'N': ['-C', 'CA', 'H', 'H2', 'H3'],
           'O': ['C'],
           'OXT': ['C', 'HXT']}}
