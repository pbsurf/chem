from io import StringIO
from chem.basics import *
from chem.mm import *
from chem.io import *
from chem.io.openmm import *
from chem.theo import *
from chem.fep import *
from chem.opt.optimize import *
from chem.data.test_molecules import ch3cho_xyz, ch2choh_xyz, hcooh_xyz
from chem.vis import *

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

  filename = 'hcooh'
  try:
    glh = load_molecule(filename + '.xyz')
    glh_key = read_file(filename + '.ff.xml')
  except:
    glh = resp_prepare(load_molecule(hcooh_xyz, residue='GLH'))
    glh.mmtype = glh.name
    write_xyz(glh, filename + '.xyz')
    glh_key = openmm_resff(glh)
    write_file(filename + '.ff.xml', glh_key)


# GLU_LFZW: w/ HE2, GLU_LFZW_DHE2: w/o HE2
filename = 'glh'
try:
  glh = load_molecule(filename + '.xyz')
except:
  from chem.model.build import *
  # AMBER doesn't support terminal GLH, so we need GLY at both ends
  glh = add_residue(Molecule(), 'GLY_LFZW', -57, -47)
  glh = add_residue(glh, 'GLU_LFZW', -57, -47)
  glh = add_residue(glh, 'GLY_LFZW', -57, -47)
  ff = openmm_app.ForceField('amber99sb.xml', 'tip3p.xml')
  ctx = openmm_EandG_context(glh, ff)
  res, glh.r = moloptim(partial(openmm_EandG, ctx=ctx), mol=glh, ftol=2E-07)  #, coords=XYZ(mol, h_atoms))
  write_xyz(glh, filename + '.xyz')


filename = 'ch2choh'
#filename = 'ch3cho'
try:
  mol = load_molecule(filename + '.xyz')
  ligparm_key = read_file(filename + '.ff.xml')
except:
  mol = resp_prepare(load_molecule(ch2choh_xyz, residue='LIG'), avg=[[3,4]])
  #mol = resp_prepare(load_molecule(ch3cho_xyz, residue='LIG'), avg=[[3,4,5]])
  mol.mmtype = mol.name
  write_xyz(mol, filename + '.xyz')
  ligparm_key = openmm_resff(mol)
  write_file(filename + '.ff.xml', ligparm_key)


# add the host
mol.r = mol.r - np.mean(mol.r, axis=0)
ligatoms = mol.listatoms()
mol.append_atoms(glh, r=glh.r - np.mean(glh.r, axis=0))

mol.dihedral((1, 0, 2, 5), 90.0)  # change H6 position for better docking
#vis = Chemvis(Mol(mol, [ VisGeom(style='licorice') ])).run()

## need to position GLH and LIG properly!
## we'll probably need restraining potential to hold LIG near GLH as lambda -> 0
# - how well do they stick together in MD?

# ideally, minimize distance between specified pairs w/o overlapping vdw radii
# - could we just do minimization w/ attraction between pairs?
# - issue in general case would be interlocking between mols preventing relaxation
#  - global opt (over 6 rigid body DOF) to minimize LJ energy (repel only, except pairs)? then local relaxation?
#  - rotations about centroid of guest dock atoms, displacements are offset between centroids of guest and host dock atoms
# - make vdW strength of pairs much stronger? how? just set all other pairs to repel only?


## A more general dock() would also randomize major diheds (and probably drop the ligalign/hostalign stuff)
# - aside: autodock scoring fn is simple: https://github.com/ttjoseph/mmdevel/blob/master/VinaScore/vinascore.py
def major_diheds(mol):
  """ get dihedrals with more than 4 atoms on each side """
  _, _, diheds = mol.get_internals(active=mol.select(ligin))
  dbnds = np.unique([sorted(d[1:3]) for d in diheds], axis=0).tolist()
  minpar = []
  for b in dbnds:
    try:
      l1, l2 = mol.partition(b[0], b[1])
      minpar.append(min(len(l1), len(l2)))
    except:
      minpar.append(0)
  ord = argsort(minpar)[::-1]
  pardiheds = [next(d for d in diheds if sorted(d[1:3]) == dbnds[ii]) for ii in ord if minpar[ii] > 4]
  minpar = [minpar[ii] for ii in ord if minpar[ii] > 4]
  return pardiheds, minpar


def dock(mol, ligatoms, ligalign, hostalign, niter=1000, maxdr=5.0, kmsd=0.01, usecharge=False):
  r = mol.r
  rpiv = np.mean(r[ligalign], axis=0)
  molLJ = NCMM(mol, qq=usecharge)
  rgd = Rigid(r[ligatoms], pivot=rpiv)
  rgd.init()

  def EandG(rlig):
    r[ligatoms] = rlig
    E, G = molLJ(r)
    dr = r[ligalign] - r[hostalign]
    Emsd = 0.5*kmsd*np.sum(dr*dr)/len(dr)
    Gmsd = kmsd*dr/len(dr)
    G[ligalign] += Gmsd
    print("E: %f; Emsd: %f" % (E, Emsd))
    return E + Emsd, G[ligatoms]

  Es, Rs = np.zeros(niter), np.tile(r, (niter, 1, 1))
  for ii in range(niter):
    randrot = 2*np.pi*np.random.rand()*normalize(np.random.randn(3))  # random rotation
    randdr = maxdr*np.random.rand()*normalize(np.random.randn(3))  # random position inside maxdr sphere
    rgd.update(np.hstack([rpiv + randdr, randrot]))
    res, r1 = moloptim(EandG, rgd.xyzs(), coords=rgd, raiseonfail=False, maxiter=50, verbose=False, mon=None)
    Es[ii], Rs[ii][ligatoms] = res.fun, r1

  sortE = np.argsort(Es)
  return Es[sortE], Rs[sortE]


# Use Amber ff99sb from now on (2006 tweaks to Amber ff99)!
ff = openmm_app.ForceField('amber99sb.xml', 'tip3p.xml', DATA_PATH + '/gaff.xml', StringIO(ligparm_key)) #, StringIO(glh_key))
openmm_load_params(mol, ff=ff, charges=True, vdw=True)

dock2_h5 = 'dock2.h5'
if os.path.exists(dock2_h5):
  Epose, rpose = read_hdf5(dock2_h5, 'Epose', 'rpose')
else:
  Edock, rdock = dock(mol, ligatoms, [5, 1], res_select(mol, 2, 'OE1,HE2'))
  Epose, rpose = cluster_poses(Edock, rdock, ligatoms)
  write_hdf5(dock2_h5, Epose=Epose, rpose=rpose)
  #vis = Chemvis(Mol(mol, rpose, [ VisGeom(style='licorice') ]), wrap=False).run()

mol.r = rpose[0]

# start interactive
import pdb; pdb.set_trace()


T0 = 300  # Kelvin
try:
  mol = load_molecule(filename + "_solv.xyz")
except:
  mol = solvate_prepare(mol, ff, T0)
  write_xyz(mol, filename + "_solv.xyz")
  # check box visually
  #if 0:
  #  from chem.vis.chemvis import *
  #  verts0, normals, indices = cube_separate(flat=False)  # returns unit cube: (0..1,0..1,0..1)
  #  vertices = np.dot(verts0, np.diag(mol.pbcbox)) - 0.5*solvent.pbcbox
  #  # for a general transformation, we'd need to apply to normals then renormalize
  #  cubemol = Bunch(r=vertices, normals=normals, colors=[(255, 0, 0, 127)], indices=indices)
  #  vis = Chemvis([ Mol(mol, [ VisGeom(style='licorice') ]), Mol(cubemol, [VisTriangles()]) ]).run()



## what are you trying to do?
# - compute delta G of bindng
# ... obviously not going to stay associated at 300K
#  - lower temp? (ice!?!)
#  - add a restraint (then subtract out contribution to dG)?
#  ... then what?  What can we adjust in this system?  compare ch3cho and ch2choh?
#  - forget this and work with a full cavity?

# how does this fit into enzyme design?
# - want to be able to calculate binding delta G for reactants and products and adjust, e.g, by making changes in binding cavity but not right at active site that make displacement of solvent easier
# - making cavity more hydrophobic might have bigger effect than tweaking geometry

# ~7.5 kBT binding energy (in vacuum - difference is total MM energy for bound and separated)
# - free energy cost of constraining position of LIG? ... for 1nm^3 box, 55 amu, S_rot ~ 20 kB, S_tr ~ 15 kB

# - measure binding energy needed to keep molecules together in MD?
#  - start w/ large value, ramp down during simulation until separation, average results from multiple runs
#  - how to adjust binding w/o otherwise affecting simulation? don't want to mess w/ charge or vdw params; maybe could add vdw exception; add separate restraint force? what form? truncated harmonic or square well?
#  - use large molecules w/ many contacts (e.g. protein dimer) and turn off charges for contacts? but then we're making it hydrophobic; seems like large molecules w/ result in too long simulation time and difficult to interpret results

# - play with binding in a cavity (use real enzyme? ... yes)
#  - seems like cavity would reduce binding energy needed to hold LIG since solvent atoms won't be bumping into it as frequently
# ... so basically start by studying ligand binding in a real system
## - we could try trypsin ... remove residues until LIG unbinds


# would our design system be able to invent, e.g, the catalytic triad?  Or would we be using predefined library of motifs and mostly tweaking geometry around LIG?
# - could building system that can design triad be a something we could work toward?
# - at a minimum, user has to specify reactants and products
# - recall: we find TS, guess some config of charges to stabilize TS, find most appropriate residues to provide the charge config, refine
# ... but non-covalent catalysis seems like such a small subset of reactions that we shouldn't restrict ourselves to it even at this early stage; could non-covalent catalysis be treated as a special case of covalent catalysis?
# - how would we explore covalent catalysis pathways?

# We have a starting point for non-covalent design ... what would be an equivalent starting point for covalent design?
# - non-covalent TS stabilization would still be applicable as well (e.g. triad oxy-anion hole)
# ... I don't think we're in a good spot to think about this until we go beyond intra-molecular reactions!



from openmmtools import alchemy
from openmmtools.integrators import HMCIntegrator

basesystem = ff.createSystem(top, nonbondedMethod=openmm_app.PME,
    nonbondedCutoff=min(0.5*min(mol.pbcbox), 10)*UNIT.angstrom, constraints=openmm_app.HBonds)
alchregion = alchemy.AlchemicalRegion(alchemical_atoms=ligatoms)
alchfactory = alchemy.AbsoluteAlchemicalFactory()
system = alchfactory.create_alchemical_system(basesystem, alchregion)
alchstate = alchemy.AlchemicalState.from_system(system)

def alch_lambdas(ctx, l_ele, l_vdw):
  alchstate.lambda_electrostatics, alchstate.lambda_sterics = l_ele, l_vdw
  alchstate.apply_to_context(ctx)

# for each step, randomizes velocities then takes nsteps Verlet steps before accepting or rejecting state
intgr = HMCIntegrator(T0*UNIT.kelvin, nsteps=100, timestep=2*UNIT.femtoseconds)  # incr timestep to decr acceptance

ctx = openmm.Context(system, intgr)
ctx.setPositions(mol.r*UNIT.angstrom)
if mol.pbcbox is not None:
  ctx.setPeriodicBoxVectors(*[v*UNIT.angstrom for v in np.diag(mol.pbcbox)])

# have to turn off electrostatics before vdW with soft-core LJ - MD fails for, e.g.,
#  lambda_ele = lambda_vdw = 0.5 due to LIG atom coming too close to solvent atom
lambda_ele = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
lambda_vdw = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
nlambda = len(lambda_ele)

mdsteps = 25000  # total steps
sampsteps = 100  # steps between samples - see autocorrelation time check below
nsamps = mdsteps//sampsteps
E_kln = np.zeros((nlambda, nlambda, nsamps))  #, np.float64)
#Rs = np.zeros((nlambda, nsamps, mol.natoms, 3))
kT = UNIT.AVOGADRO_CONSTANT_NA * UNIT.BOLTZMANN_CONSTANT_kB * T0*UNIT.kelvin  # kB*T in OpenMM units

# start interactive
import pdb; pdb.set_trace()

for kk in range(nlambda):
  print("%s: running %d MD steps for lambda_ele %f, lambda_vdw %f" % (time.strftime('%X'), mdsteps, lambda_ele[kk], lambda_vdw[kk]))
  for jj in range(nsamps):
    alch_lambdas(ctx, lambda_ele[kk], lambda_vdw[kk])
    intgr.step(1 if useMC else sampsteps)
    for ll in range(nlambda):  # for MBAR
      alch_lambdas(ctx, lambda_ele[ll], lambda_vdw[ll])
      E_kln[kk,ll,jj] = ctx.getState(getEnergy=True).getPotentialEnergy()/kT

#print("MC acceptance rate: %f" % intgr.acceptance_rate)  #(naccept/nsamps)

warmup = 5
E_kln = E_kln[:,:,warmup:]
write_hdf5("mbar_%.0f.h5" % time.time(), E_kln=E_kln)

# pymbar
from pymbar import MBAR, BAR
beta = 1/(KCALMOL_PER_HARTREE*BOLTZMANN*T0)  # 1/kB/T0 in kcal/mol

mbar = MBAR(E_kln, [len(E[0]) for E in E_kln])  #[nsamps - warmup]*nlambda)
res_mbar = mbar.computeEntropyAndEnthalpy(return_dict=True)
