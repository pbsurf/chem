# Amber GAFF atom type assignment: antechamber/atomtype.c (iterates through lines in ATOMTYPE_GFF.DEF)
#  Refs: 10.1016/j.jmgm.2005.12.005 (Antechamber), 10.1002/jcc.20035 (GAFF)

# This is our failed attempt to assign atom types with a few rules and elimination of all candidates with
#  missing stretch or bend parameters
## ... doesn't work; e.g., c -c2-os is missing for chorismate; antechamber/parmchk handles missing params

gaff_helper_str = """
c    X,X,X               Sp2 C carbonyl group (O or S)
c1   X,X                 Sp C
c2   X,X,X               Sp2 C
c3   X,X,X,X             Sp3 C
ca   X,X,X               Sp2 C in pure aromatic systems
cp   X,X,X               Head Sp2 C that connect two rings in biphenyl sys.
cq   X,X,X               Head Sp2 C that connect two rings in biphenyl sys. identical to cp
cc   X,X,X               Sp2 carbons in non-pure aromatic systems
cd   X,X,X               Sp2 carbons in non-pure aromatic systems, identical to cc
ce   X,X,X               Inner Sp2 carbons in conjugated systems
cf   X,X,X               Inner Sp2 carbons in conjugated systems, identical to ce
cg   X,X                 Inner Sp carbons in conjugated systems
ch   X,X                 Inner Sp carbons in conjugated systems, identical to cg
cx   X,X,X,X             Sp3 carbons in triangle systems
cy   X,X,X,X             Sp3 carbons in square systems
cu   X,X,X               Sp2 carbons in triangle systems
cv   X,X,X               Sp2 carbons in square systems
cz   N,N,N               Sp2 carbon in guanidine group
h1   C                   H bonded to aliphatic carbon with 1 electrwd. group
h2   C                   H bonded to aliphatic carbon with 2 electrwd. group
h3   C                   H bonded to aliphatic carbon with 3 electrwd. group
h4   C                   H bonded to non-sp3 carbon with 1 electrwd. group
h5   C                   H bonded to non-sp3 carbon with 2 electrwd. group
ha   C                   H bonded to aromatic carbon
hc   C                   H bonded to aliphatic carbon without electrwd. group
hn   N                   H bonded to nitrogen atoms
ho   O                   Hydroxyl group
hp   P                   H bonded to phosphate
hs   S                   Hydrogen bonded to sulphur
hw   O                   Hydrogen in water
hx   C                   H bonded to C next to positively charged group
n    X,X,X               Sp2 nitrogen in amide groups
n1   X,X                 Sp N
n2   X,X                 aliphatic Sp2 N with two connected atoms
n3   X,X,X               Sp3 N with three connected atoms
n4   X,X,X,X             Sp3 N with four connected atoms
na   X,X,X               Sp2 N with three connected atoms
nb   X,X,X               Sp2 N in pure aromatic systems
nc   X,X,X               Sp2 N in non-pure aromatic systems
nd   X,X,X               Sp2 N in non-pure aromatic systems, identical to nc
ne   X,X,X               Inner Sp2 N in conjugated systems
nf   X,X,X               Inner Sp2 N in conjugated systems, identical to ne
nh   X,X,X               Amine N connected one or more aromatic rings
no   O,O,X               Nitro N
o    X                   Oxygen with one connected atom
oh   H,X                 Oxygen in hydroxyl group
os   X,X                 Ether and ester oxygen
ow   H                   Oxygen in water
"""

gaff_helper = { l[0]: l[1].split(',') for line in gaff_helper_str.splitlines() if (l := line.split()) and l[0][0] != '#' }


def gaff_special(mol, ii, cand):
  a = mol.atoms[ii]
  if a.znuc not in [1,6]:  return cand
  cand = set(cand)
  if a.znuc == 6:
    cand -= set(['cl'])
    if 'c' in cand and any(mol.atoms[jj].znuc in [8,16] and len(mol.atoms[jj].mmconnect) == 1 for jj in a.mmconnect):
      return ['c']
    cand -= set(['c'])
    # keep only possibly aromatic cycles; note that every cycle appears twice (one for each direction)
    cycles = list(dfs_path(mol.mmconnect, ii, ii))
    aromatic = [c for c in cycles if all(len(mol.atoms[jj].mmconnect) < 4 for jj in c)]
    if len(aromatic) < 4:  cand -= set(['cp', 'cq'])
    if len(aromatic) == 0:  cand -= set(['ca', 'cc', 'cd'])
    lencycles = [len(c) for c in cycles]
    if not lencycles or min(lencycles) > 3:  cand -= set(['cx', 'cu'])
    if not lencycles or min(lencycles) > 4:  cand -= set(['cy', 'cv'])
    #'ce' 'cf': (mol.atoms[jj].znuc == 6 and len( for jj in a.mmconnect)  - conjugated
  elif a.znuc == 1:
    a1 = mol.atoms[a.mmconnect[0]]
    if a1.znuc == 6:
      elewd = sum(mol.atoms[jj].znuc in [7,8,9,17,35] for jj in a1.mmconnect)
      if elewd != 3:  cand -= set(['h3'])
      if elewd != 2:  cand -= set(['h2', 'h5'])
      if elewd != 1:  cand -= set(['h1', 'h4'])
      if elewd != 0:  cand -= set(['hc'])  #'ha'
      if len(a1.mmconnect) < 4:  cand -= set(['h1', 'h2', 'h3', 'hc'])
      if not any(mol.atoms[jj].znuc == 7 and len(mol.atoms[jj].mmconnect) == 4 for jj in a1.mmconnect):
        cand -= set(['hx'])
    else:
      cand -= set(['h1', 'h2', 'h3', 'h4', 'h5', 'ha', 'hc', 'hx'])

  return list(cand)


gaff = load_amber_dat(DATA_PATH + '/amber/gaff.dat')

#mISJ = load_compound('ISJ').remove_atoms([22,25])
mol = mISJ
# choose all candidates for element
candtypes = [ [n for n in gaff.vdw.keys() if n[0].upper() == ELEMENTS[znuc].symbol ] for znuc in mol.znuc ]
# eliminate candidates based on number of connected atoms
candtypes = [ [c for c in candtypes[ii] if c not in gaff_helper or len(gaff_helper[c]) == len(a.mmconnect)] for ii,a in mol.enumatoms() ]
# eliminate candidates based on special cases
candtypes = [ gaff_special(mol, ii, cand) for ii, cand in enumerate(candtypes) ]
# eliminate any candidates for which strench or bend parameters are missing
prevsize = np.inf
ncand = lambda x: sum(len(x) for x in candtypes)
while prevsize > 0 and ncand(candtypes) < prevsize:
  prevsize = ncand(candtypes)
  candtypes = [ [c1 for c1 in candtypes[ii] if all( any( ((c1+'-'+c2) in gaff.stretch or (c2+'-'+c1) in gaff.stretch)  for c2 in candtypes[jj] ) for jj in a.mmconnect)] for ii,a in mol.enumatoms() ]  # if jj > ii
  candtypes = [ [c1 for c1 in candtypes[ii] if all( any( ((c3+'-'+c1+'-'+c2) in gaff.bend or (c2+'-'+c1+'-'+c3) in gaff.bend) for c2 in candtypes[jj] for c3 in candtypes[kk] ) for jj in a.mmconnect for kk in a.mmconnect if kk > jj )] for ii,a in mol.enumatoms() ]


# see what bends are missing
for ii,a in mol.enumatoms():
  print("%d: %r:" % (ii, candtypes[ii]))
  for c1 in candtypes[ii]:
    for jj in a.mmconnect:
      for kk in a.mmconnect:
        if kk > jj:
          for c2 in candtypes[jj]:
            for c3 in candtypes[kk]:
              if not ((c3+'-'+c1+'-'+c2) in gaff.bend or (c2+'-'+c1+'-'+c3) in gaff.bend):
                print("Missing: " + (c2+'-'+c1+'-'+c3))



import random
mol = Molecule(mISJ)
mol.mmtype = [random.choice(x) for x in candtypes]
set_mm_params(mol, gaff)


# single point RHF/6-31G* takes 500s ... calculate RESP w/o optimization?
#def openmm_gaff_prepare(mol):
mISJ = resp_prepare(mISJ, avg=[[7,8], [13,14], [22,23]], netcharge=-2.0, maxiter=0)
write_xyz(mISJ, 'ISJ_gaff.xyz', writemmq=True)
mPRE = resp_prepare(mPRE, avg=[[8,9], [14,15], [17,20], [16,21], [22,23]], netcharge=-2.0, maxiter=0)
write_xyz(mPRE, 'PRE_gaff.xyz', writemmq=True)
