import os, subprocess
from chem.basics import *
from chem.mm import *
from chem.io import *
from chem.io.tinker import *
from chem.fep import *

# Refs:
# - https://sites.google.com/site/amoebaworkshop/intro - EtOH solvation

ethanol_gaff = """9  molden generated ambfor .xyz
1  c3     -1.208   -0.243   -0.031  901  0.000      2      4      5      6
1  c3      0.089    0.560    0.035  902  0.000      1      3      7      8
1  oh      1.215   -0.315   -0.107  903  0.000      2      9
1  hc     -1.297   -0.796   -0.991  904  0.000      1
1  hc     -2.097    0.419    0.053  905  0.000      1
1  hc     -1.273   -0.988    0.792  906  0.000      1
1  h1      0.160    1.117    0.994  907  0.000      2
1  h1      0.114    1.311   -0.783  908  0.000      2
1  ho      1.298   -0.828    0.686  909  0.000      3
"""

ethane_gaff = """8  ethane (molden ambfor xyz)
1  c3     -1.208   -0.243   -0.031  801  0.000      2      3      4      5
2  c3      0.089    0.560    0.035  802  0.000      1      6      7      8
3  hc     -1.297   -0.796   -0.991  803  0.000      1
4  hc     -2.097    0.419    0.053  804  0.000      1
5  hc     -1.273   -0.988    0.792  805  0.000      1
6  hc      0.160    1.117    0.994  806  0.000      2
7  hc      0.114    1.311   -0.783  807  0.000      2
8  hc      1.215   -0.315   -0.107  808  0.000      2
"""

# start interactive
import pdb; pdb.set_trace()

filename = 'ethanol0'
#filename = 'ethane0'
try:
  mol = load_molecule(filename + '.xyz')
  ligparm_key = read_file(filename + '.key')
except:
  mol = gaff_prepare(load_molecule(ethanol_gaff), avg=[[3,4,5], [6,7]], mmtype0=901)
  #mol = gaff_prepare(load_molecule(ethane_gaff), constr=[[0,1]], avg=[[2,3,4], [5,6,7]], mmtype0=801)
  write_xyz(mol, filename + '.xyz')
  ligparm_key = write_mm_params(mol)
  write_file(filename + '.key', ligparm_key)


# tinker key setup
implsolv_key = '\n\n'.join([make_tinker_key('amber96'), "solvate GBSA", ligparm_key])
# `rattle water` freezes water internal geometry; should we use `ewald` instead of default of 9 nm cutoffs?
explsolv_key = '\n\n'.join([make_tinker_key('amber96'), "rattle water", ligparm_key])

T0 = 300  # Kelvin
beta = 1/(BOLTZMANN*T0)

try:
  mol = load_molecule(filename + "_solv.xyz")
except:
  mol = solvate_prepare(mol, explsolv_key, T0, solute_res='LIG')
  write_xyz(mol, filename + "_solv.xyz")


# MD runs for lambda = 1 -> lambda = 0
# Other relevant keywords: vdw-annihilate - also apply lambda to intra-ligand vdW interaction
# TODO: will we need soft-core potentials (for explicit solvent)?
ligatoms = mol.select('* LIG *')
# ligand <atom number list>: atom type -> 0 at lambda == 0
ligand_key = "ligand " + ','.join(str(x+1) for x in ligatoms)
# we can use mutate to make atoms appear as well as disappear (but note Tinker doesn't modify bonded terms)
# mutate <atom number> <atom type at lambda == 0> <atom type at lambda == 1> -- see Tinker mutate.f
#ligand_key = "\n".join("mutate %d 0 %d" % (x+1, mol.atoms[x].mmtype) for x in ligatoms)

# contrary to docs, "lambda" doesn't seem to work - have to use "ele-lambda" and "vdw-lambda" (even if equal)
ele_lambda = [1.0, 0.9, 0.8, 0.70, 0.6, 0.50, 0.4, 0.3, 0.2, 0.1, 0.0]
vdw_lambda = [1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.2, 0.0]

common_key = explsolv_key  #implsolv_key
keys = ['\n\n'.join([common_key, ligand_key,
    "ele-lambda %f\nvdw-lambda %f" % (el, vl)]) for el, vl in zip(ele_lambda, vdw_lambda)]
nlambda = len(keys)

mdsteps = 1200
warmup = 2  # number of initial *frames* to discard

for ii,key in enumerate(keys):
  filename = "fe%02d" % ii
  if os.path.exists(filename + ".arc"):
    continue
  write_tinker_xyz(mol, filename + ".xyz")
  write_file(filename + ".key", key)
  # NVT: ..., "2", str(T0)  NPT: ..., "4", str(T0), 1.0 (w/ explicit solvent; 1.0 atm)
  subprocess.check_call([os.path.join(TINKER_PATH, "dynamic"),
      filename + ".xyz", str(mdsteps), "1.0", "0.1", "4", str(T0), "1.0"])  # dt (fs), save interval (ps), NVT, T (K)

# confirm that solute is invisible to solvent at final step (look for solvent molecules overlapping solute)
# - should also confirm decrease in PBC box size
if 0:
  fefinal = read_tinker_geom('fe%02d.arc' % (nlambda - 1))
  vis = Chemvis(Mol(mol, fefinal, [ VisGeom(style='licorice') ]), wrap=False).run()

## Tinker vdw-lambda only works for BUFFERED-14-7 vdW type used for Amoeba - not supported for Amber's LJ 12-6
# Options:
# - use Amoeba instead of Amber just to finish this stuff
# - OpenMM, Conda, Python 3
# - modify Tinker ... seems LJ calc is repeated many times, so probably not an option


tinker_fep(nlambda, T0)
# cleanup
#os.system("rm fe*.*")

# - Tinker BAR seems to agree w/ our calc (for first lambda step at least)
# Ethanol NVT:
# Implicit solvent: BAR: 9.616824 kcal/mol; FEP: 9.613122 kcal/mol; FEP(rev): -9.620093 kcal/mol
# Explicit solvent: BAR: 6.211831 kcal/mol; FEP: 6.138846 kcal/mol; FEP(rev): -6.312527 kcal/mol
# - this is free energy change for removing LIG, so negative of solvation free energy
# - actual solvation free energy of EtOH: -3.94 kcal/mol - nist.gov/programs-projects/solvation-free-energies

# Ethane (actual solvation free energy: +0.697 kcal/mol):
# NVT: BAR: -0.009030 kcal/mol; FEP: -0.009268 kcal/mol; FEP(rev): 0.008790 kcal/mol
# NPT: BAR: -0.017925 kcal/mol; FEP: -0.018657 kcal/mol; FEP(rev): 0.017189 kcal/mol
# ... at least the sign is correct (barely)!
# - remember that thermostat is stochastic, so results will not be identical for repeated runs

# NEXT: MC? or ligand binding free energy? then QM/MM free energy?
# - maybe do full protein, then compare to frozen active site only?
# - need error estimates on free energy!

# Free energy w/ constrained/restrained MD:
# - https://www.cp2k.org/exercises:2018_uzh_acpc2:prot_fol - for a distance constraint, Lagrange mult. from SHAKE/RATTLE gives constraint force, TI of which gives free energy difference between two values of constraint
# - see also 10.1063/1.477419 and www.vasp.at/wiki/index.php/Constrained_molecular_dynamics

# QM/MM FE:
# - http://chryswoods.com/embo2014/QM_MM.html (Sire)
# - http://www.cse.scitech.ac.uk/ccg/software/chemshell/manual/hyb_fep.html

# - also try pymbar


# too slow, but useful documentation
if 0:
  dEup, dEdn = [[] for _ in range(nlambda)], [[] for _ in range(nlambda)]
  if os.path.exists('dEs.h5'):
    dEup, dEdn = read_hdf5('dEs.h5', 'dEup', 'dEdn')
  else:
    dEup, dEdn = [[] for _ in range(nlambda)], [[] for _ in range(nlambda)]
    for ii in range(nlambda):
      arc = read_tinker_geom("fe%02d.arc" % ii)
      for r in arc[warmup:]:
        Eii, _ = tinker_EandG(mol, r=r, key=keys[ii], grad=False)
        Edn, _ = tinker_EandG(mol, r=r, key=keys[ii-1], grad=False) if ii > 0 else (Eii, 0)
        Eup, _ = tinker_EandG(mol, r=r, key=keys[ii+1], grad=False) if ii < 9 else (Eii, 0)
        dEup[ii].append(beta*(Eup - Eii))
        dEdn[ii].append(beta*(Edn - Eii))

    dEup, dEdn = np.array(dEup), np.array(dEdn)
    write_hdf5('dEs.h5', dEup=dEup, dEdn=dEdn)


## Mutate ethanol <-> ethane
# ... doesn't look like this will work unfortunately: Tinker doesn't apply lambda to bonded interactions
#  (except in standalone `alchemy` program) - only electrostatic and vdW are modified (w/ ele-lambda and
#  vdw-lambda keywords)

# Assembled from ethanol0.xyz and ethane0.xyz
mutate_gaff = """10  ethane <-> ethanol
 1  c3     -1.193368   -0.256468   -0.033938  801  0.000      2      3      4      5
 2  c3      0.119218    0.522610    0.024413  802  0.000      1      6      7      8      9
 3  hc     -1.288641   -0.795476   -0.971415  803  0.000      1
 4  hc     -2.048358    0.406717    0.053173  804  0.000      1
 5  hc     -1.255579   -0.981700    0.771437  805  0.000      1
 6  hc      0.218394    1.054918    0.965463  806  0.000      2
 7  hc      0.177675    1.254082   -0.775777  807  0.000      2
 8  hc      0.973660   -0.139682   -0.071355  808  0.000      2
 9  oh      1.220393   -0.258322   -0.093038  903  0.000      2     10
10  ho      1.266540   -0.860157    0.636833  909  0.000      9
"""

mutmol = load_molecule(mutate_gaff)
# find bonded terms between appearing/disappearing atoms
some = lambda x: any(x) and not all(x)
bonds, angles, diheds = mutmol.get_internals()
mmt = mutmol.mmtype
pbonds = list(set([tuple(mmt[a] for a in b) for b in bonds if some(mmt[a] < 900 for a in b)]))
pangles = list(set([tuple(mmt[a] for a in b) for b in angles if some(mmt[a] < 900 for a in b)]))
pdiheds = list(set([tuple(mmt[a] for a in b) for b in diheds if some(mmt[a] < 900 for a in b)]))
# create null interactions between appearing/disappearing atoms to satisfy Tinker
# note that there won't be vdw, coulomb interactions between atoms 8,9,10 since they are bonded
sbonds = ["bond %d %d  0.00 1.5000" % p for p in pbonds]
sangles = ["angle %d %d %d  0.00 109.50" % p for p in pangles]  # Tinker doesn't like angle = 0
sdiheds = ["torsion %d %d %d %d  0.000 0.0 2" % p for p in pdiheds]
zero_key = "\n".join(sbonds + sangles + sdiheds)

# ethane, ethanol params
ligparm_etoh = read_file('ethanol0.key')
ligparm_ethane = read_file('ethane0.key')

# ethane <-> ethanol mutations
muts = [(801, 901), (802, 902),
    (803, 904), (804, 905), (805, 906), (806, 907), (807, 908), (808, 0), (0, 903), (0, 909)]
mutate_key = "\n".join("mutate %d %d %d" % (ii+1, ab[0], ab[1]) for ii,ab in enumerate(muts))

mutbase_key = '\n\n'.join([make_tinker_key('amber96'), "rattle water", ligparm_ethane, ligparm_etoh, zero_key, mutate_key])

mol = solvate_prepare(mutmol, mutbase_key + "\n\nlambda 1.0\n", T0)
