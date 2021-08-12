from chem.basics import *
from chem.mm import *
from chem.io import *
from chem.io.tinker import *
from chem.theo import *
from chem.fep import *

# What we should be doing:
# - prepare with AmberTools
# - convert to tinker with github.com/emleddin/research-scripts/tree/main/tinker-params (uses parmed)
#  - other options are amber2lammps.py; OpenMM can read Amber prmtop directly!

if 0:
  parm = load_amber_dat('../common/amber/parm99.dat')
  p99sb = load_amber_frcmod('../common/amber/frcmod.ff99SB')
  parm.torsion = dict(parm.torsion, **p99sb.torsion)  # in general we'd need to merge stretch, bend, etc. too

  # Couldn't create single residue w/ NH3+ and COO- in Amber; combined charges from Tinker GLU backbone and
  #  Amber GLH sidechain (used all_amino03.lib, but Tinker uses charges from all_amino94.lib), adjusted N3 to
  #  give zero total charge
  glh = load_molecule('glh1.xyz')
  set_mm_params(glh, parm, mmtype0=801)
  glh_key = write_mm_params(mol)


# start interactive
import pdb; pdb.set_trace()


hcooh_xyz = """5 emin -8.287  AMBFOR/AMBMD generated .xyz (amber/gaff param.)
1  h5     2.07864  -4.95969   2.12292  -23  0.098      2
2  c2     1.66099  -4.11544   1.59679   -4  0.475      1      3      4
3  o      1.86183  -3.93186   0.36790  -49 -0.419      2
4  oh     1.29103  -3.06027   2.31321  -50 -0.353      2      5
5  ho     1.12452  -2.33975   1.68018  -27  0.199      4
"""

filename = 'hcooh'
try:
  glh = load_molecule(filename + '.xyz')
  glh_key = read_file(filename + '.key')
except:
  glh = gaff_prepare(load_molecule(hcooh_xyz, residue='GLH'), mmtype0=801)
  write_xyz(glh, filename + '.xyz')
  glh_key = write_mm_params(glh)
  write_file(filename + '.key', hcooh_key)


filename = 'ch2choh'
#filename = 'ch3cho'
try:
  mol = load_molecule(filename + '.xyz')
  ligparm_key = read_file(filename + '.key')
except:
  mol = gaff_prepare(load_molecule(ch2choh_xyz, residue='LIG'), avg=[[3,4]], mmtype0=901)
  #mol = gaff_prepare(load_molecule(ch3cho_xyz, residue='LIG'), avg=[[3,4,5]], mmtype0=901)
  write_xyz(mol, filename + '.xyz')
  ligparm_key = write_mm_params(mol)
  write_file(filename + '.key', ligparm_key)

# add the host
ligatoms = mol.listatoms()
mol.append_atoms(glh)

from chem.vis.chemvis import *
vis = Chemvis(Mol(mol, [ VisGeom(style='licorice') ])).run()

## need to position GLH and LIG properly!
# - how well do they stick together in MD?

# ideally, minimize distance between specified pairs w/o overlapping vdw radii
# - could we just do minimization w/ attraction between pairs?
# - issue in general case would be interlocking between mols preventing relaxation
#  - global opt (over 6 rigid body DOF) to minimize LJ energy (repel only, except pairs)? then local relaxation?
#  - rotations about centroid of guest dock atoms, displacements are offset between centroids of guest and host dock atoms
# - make vdW strength of pairs much stronger? how? just set all other pairs to repel only?


def dock(mol, ligatoms, ligalign, hostalign, niter=1000, maxdr=5.0, kmsd=0.01):
  r = mol.r
  rpiv = np.mean(r[ligalign], axis=0)
  molLJ = MolLJ(mol)
  rgd = Rigid(r[ligatoms], pivot=rpiv)
  rgd.init()

  def EandG(rlig):
    r[ligatoms] = rlig
    E, G = molLJ(r)
    dr = r[ligalign] - r[hostalign]
    Emsd = 0.5*kmsd*np.sum(dr*dr, axis=0)/len(dr)
    Gmsd = kmsd*dr/len(dr)
    G[ligalign] += Gmsd
    return E + Emsd, G

  Es, Rs = np.zeros(niter), np.tile(r, (niter, 1, 1))
  for ii in range(niter):
    randrot = 2*np.pi*np.random.rand()*normalize(np.random.randn(3))  # random rotation
    randdr = maxdr*np.random.rand()*normalize(np.random.randn(3))  # random position inside maxdr sphere
    rgd.update(np.hstack([rpiv + randdr, randrot]))
    res, r1 = moloptim(EandG, rgd.xyzs(), coords=rgd, raiseonfail=False, maxiter=20)
    Es[ii], Rs[ii][ligatoms] = res.fn, r1

  sortE = np.argsort(Es)
  return Es[sortE], Rs[sortE]


Edock, rdock = dock(mol, ligatoms, [], [])
vis = Chemvis(Mol(mol, rdock, [ VisGeom(style='licorice') ])).run()



T0 = 300  # Kelvin
beta = 1/(BOLTZMANN*T0)

# Use Amber ff99sb from now on (2006 tweaks to Amber ff99)!
explsolv_key = '\n\n'.join(make_tinker_key('amber99sb'), "rattle water", ligparm_key)

try:
  mol = load_molecule(filename + "_solv.xyz")
except:
  mol = solvate_prepare(mol, explsolv_key, T0)
  write_xyz(mol, filename + "_solv.xyz")


## we'll probably need restraining potential to hold LIG near GLH as lambda -> 0

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
      filename + ".xyz", str(mdsteps), "1.0", "0.1", "4", str(T0), "1.0"])  # dt (fs), save interval (ps)

# confirm that disappearing atoms are invisible to solvent at final step (look for solvent molecules overlapping)
# - should also confirm decrease in PBC box size
if 0:
  fefinal = read_tinker_geom('fe%02d.arc' % (nlambda - 1))
  vis = Chemvis(Mol(mol, fefinal, [ VisGeom(style='licorice') ]), wrap=False).run()


# reads nlambda files fe%02d.arc from current directory
tinker_fep(nlambda, T0)
