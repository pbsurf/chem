"""
Basic PDB data - standard residue names, atoms names, etc
"""

## Residues
PDB_AMINO = set(['ILE', 'SER', 'GLY', 'HIS', 'TRP',
    'GLU', 'CYS', 'ASP', 'THR', 'LYS', 'PRO', 'PHE', 'ALA', 'MET', 'GLN', 'ARG', 'VAL', 'ASN', 'TYR', 'LEU'])
PDB_PROTEIN = PDB_AMINO | set(['UNK', 'PCA', 'AIB', 'FOR', 'ORN'])
PDB_RNA = set(['A', 'C', 'G', 'U'])
PDB_DNA = set(['DA', 'DC', 'DG', 'DT'])
PDB_STANDARD = set.union(PDB_PROTEIN, PDB_RNA, PDB_DNA)

# 1 letter code <-> 3 letter code
_AA1 = list("ACDEFGHIKLMNPQRSTVWY")
_AA3 = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
AA_1TO3 = dict(zip(_AA1,_AA3))
AA_3TO1 = dict(zip(_AA3,_AA1))

# atoms
PDB_BACKBONE = set(['C', 'N', 'CA'])
PDB_EXTBACKBONE = set(['C', 'N', 'CA', 'O', 'H', 'H1', 'H2', 'H3', 'OXT', 'HXT'])
PDB_NOTSIDECHAIN = PDB_EXTBACKBONE - set(['CA'])


# Grantham distance: amino acid similarity measure based on composition, polarity, molecular volume
# - see en.wikipedia.org/wiki/Amino_acid_replacement
# - data from www.genome.jp/dbget-bin/www_bget?aaindex:GRAR740104 via github and `pr -t -e4`
_GRANTHAM_TABLE = """
      A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
  A   0   112 111 126 195 91  107 60  86  94  96  106 84  113 27  99  58  148 112 64
  R   112 0   86  96  180 43  54  125 29  97  102 26  91  97  103 110 71  101 77  96
  N   111 86  0   23  139 46  42  80  68  149 153 94  142 158 91  46  65  174 143 133
  D   126 96  23  0   154 61  45  94  81  168 172 101 160 177 108 65  85  181 160 152
  C   195 180 139 154 0   154 170 159 174 198 198 202 196 205 169 112 149 215 194 192
  Q   91  43  46  61  154 0   29  87  24  109 113 53  101 116 76  68  42  130 99  96
  E   107 54  42  45  170 29  0   98  40  134 138 56  126 140 93  80  65  152 122 121
  G   60  125 80  94  159 87  98  0   98  135 138 127 127 153 42  56  59  184 147 109
  H   86  29  68  81  174 24  40  98  0   94  99  32  87  100 77  89  47  115 83  84
  I   94  97  149 168 198 109 134 135 94  0   5   102 10  21  95  142 89  61  33  29
  L   96  102 153 172 198 113 138 138 99  5   0   107 15  22  98  145 92  61  36  32
  K   106 26  94  101 202 53  56  127 32  102 107 0   95  102 103 121 78  110 85  97
  M   84  91  142 160 196 101 126 127 87  10  15  95  0   28  87  135 81  67  36  21
  F   113 97  158 177 205 116 140 153 100 21  22  102 28  0   114 155 103 40  22  50
  P   27  103 91  108 169 76  93  42  77  95  98  103 87  114 0   74  38  147 110 68
  S   99  110 46  65  112 68  80  56  89  142 145 121 135 155 74  0   58  177 144 124
  T   58  71  65  85  149 42  65  59  47  89  92  78  81  103 38  58  0   128 92  69
  W   148 101 174 181 215 130 152 184 115 61  61  110 67  40  147 177 128 0   37  88
  Y   112 77  143 160 194 99  122 147 83  33  36  85  36  22  110 144 92  37  0   55
  V   64  96  133 152 192 96  121 109 84  29  32  97  21  50  68  124 69  88  55  0
"""

_GRANTHAM_DIST = {}
def GRANTHAM_DIST(res):
  if not _GRANTHAM_DIST:
    lines = _GRANTHAM_TABLE.splitlines()
    cols = lines[1].split()
    for row in lines[2:]:
      vals = row.split()
      t = _GRANTHAM_DIST.setdefault(AA_1TO3[vals[0]], {})
      for col, val in zip(cols, vals[1:]):
        t[AA_1TO3[col]] = int(val)
  return _GRANTHAM_DIST[res]


# ref: http://blanco.biomol.uci.edu/Whole_residue_HFscales.txt
# columns: \delta G for water to bilayer interface; \delta G for water to octanol; difference
# Note that larger value means more hydrophilic
RESIDUE_HYDROPHOBICITY = {
  'ALA' : 0.17 , # 0.50  0.33
  'ARG' : 0.81 , # 1.81  1.00  # ARG+
  'ASN' : 0.42 , # 0.85  0.43
  'ASP' : 1.23 , # 3.64  2.41  # ASP-
  #'ASP0' : -0.07, # 0.43  0.50  # aka ASH
  'CYS' : -0.24, # -0.02 0.22
  'GLN' : 0.58 , # 0.77  0.19
  'GLU': 2.02 , # 3.63  1.61  # GLU-
  #'GLU0' : -0.01, # 0.11  0.12  # aka GLH
  'GLY' : 0.01 , # 1.15  1.14
  #'HIS+': 0.96 , # 2.33  1.37
  'HIS' : 0.17 , # 0.11  -0.06  # HIS0
  'ILE' : -0.31, # -1.12 -0.81
  'LEU' : -0.56, # -1.25 -0.69
  'LYS' : 0.99 , # 2.80  1.81  # LYS+
  'MET' : -0.23, # -0.67 -0.44
  'PHE' : -1.13, # -1.71 -0.58
  'PRO' : 0.45 , # 0.14  -0.31
  'SER' : 0.13 , # 0.46  0.33
  'THR' : 0.14 , # 0.25  0.11
  'TRP' : -1.85, # -2.09 -0.24
  'TYR' : -0.94, # -0.71 0.23
  'VAL' : 0.07   # -0.46 -0.53
}


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
