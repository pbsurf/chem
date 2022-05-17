from chem.io import *
from chem.io.openmm import *
from chem.model.build import *
from chem.model.prepare import *
from chem.opt.optimize import *
from chem.fep import *

from chem.vis.chemvis import *


def openmm_fep(mol, lig, ff, T0, state=None, mdsteps=25000, sampsteps=100,
    useMC=False, warmup=5, ele_steps=5, vdw_steps=5, rst_steps=0, restraint=None, dt=4.0, verbose=False):
  """ compute free energy difference for turing off non-bonded interactions between `ligatoms` and rest of
    `mol` at temperature `T0` with openmm force field `ff` using `ele_steps` "lambda" steps to turn off
    electrostatic interaction followed by `vdw_steps` lambda steps to turn off vdW interaction, with `mdsteps`
    MD steps for each lambda step, sampling energy every `sampsteps` MD steps.  Hybrid/hamiltonian Monte Carlo
    is used if `useMC` is true.
  """
  from openmmtools import alchemy
  from openmmtools.integrators import HMCIntegrator, VelocityVerletIntegrator
  from pymbar import MBAR, BAR

  ligatoms = mol.select(lig)
  basesystem = ff.createSystem(openmm_top(mol), nonbondedMethod=openmm_app.PME,
      nonbondedCutoff=min(0.5*min(mol.pbcbox), 10)*UNIT.angstrom, constraints=openmm_app.HBonds)
  alchregion = alchemy.AlchemicalRegion(alchemical_atoms=ligatoms)
  alchfactory = alchemy.AbsoluteAlchemicalFactory()
  system = alchfactory.create_alchemical_system(basesystem, alchregion)
  alchstate = alchemy.AlchemicalState.from_system(system)
  if restraint is not None:
    system.addForce(restraint)

  def alch_lambdas(ctx, l_ele, l_vdw, l_rst):
    alchstate.lambda_electrostatics, alchstate.lambda_sterics = l_ele, l_vdw
    alchstate.apply_to_context(ctx)
    if restraint is not None:
      ctx.setParameter('lambda_restraints', l_rst)

  if useMC:
    # for each step, randomizes velocities then takes nsteps Verlet steps before accepting or rejecting state
    # - increase timestep (increasing energy error of integrator) to decrease acceptance
    intgr = HMCIntegrator(T0*UNIT.kelvin, nsteps=sampsteps, timestep=dt*UNIT.femtoseconds)
    #intgr = VelocityVerletIntegrator(dt*UNIT.femtoseconds)
  else:
    intgr = openmm.LangevinMiddleIntegrator(T0*UNIT.kelvin, 1/UNIT.picosecond, dt*UNIT.femtoseconds)

  if state is None: state = Bunch()
  ctx = openmm.Context(system, intgr)  #, openmm.Platform.getPlatformByName("Reference"))
  ctx.setPositions(state.get('r', mol.r)*UNIT.angstrom)
  if mol.pbcbox is not None:
    ctx.setPeriodicBoxVectors(*[v*UNIT.angstrom for v in np.diag(mol.pbcbox)])
  if state.get('v') is None:
    ctx.setVelocitiesToTemperature(T0*UNIT.kelvin)
  else:
    ctx.setVelocities(state.v*(UNIT.angstrom/UNIT.picosecond))

  openmm_load_params(mol, system)  # get charges
  ligq = sum(mol.atoms[ii].mmq for ii in ligatoms)
  if abs(ligq) > 0.01:
    print("Warning: ligatoms have net charge of %.3f" % ligq)

  # have to turn off electrostatics before vdW with soft-core LJ - MD fails for, e.g.,
  #  lambda_ele = lambda_vdw = 0.5 due to LIG atom coming too close to solvent atom
  if type(ele_steps) is int:
    lambda_ele = [1.0]*rst_steps + [(1 - float(ii)/ele_steps) for ii in range(ele_steps)] + [0.0]*(vdw_steps+1)
    lambda_vdw = [1.0]*rst_steps + [1.0]*ele_steps + [(1 - float(ii)/vdw_steps) for ii in range(vdw_steps+1)]
    lambda_rst = [(float(ii)/rst_steps) for ii in range(rst_steps)] + [1.0]*(ele_steps + vdw_steps + 1)
  else:
    lambda_ele, lambda_vdw, lambda_rst = ele_steps, vdw_steps, rst_steps
  nlambda = len(lambda_ele)

  nsamps = mdsteps//sampsteps
  if state.get('Rs') is True:  # optional recording of positions
    state.Rs = [ [0]*nsamps for ii in range(nlambda) ]  #np.zeros((nlambda, nsamps, mol.natoms, 3))
  if 'E_kln' not in state:
    state.E_kln = np.zeros((nlambda, nlambda, nsamps))
  kT = UNIT.AVOGADRO_CONSTANT_NA * UNIT.BOLTZMANN_CONSTANT_kB * T0*UNIT.kelvin  # kB*T in OpenMM units
  try:
    for kk in range(state.get('idx_lambda', 0), nlambda):
      print("%s: running %d MD steps for lambda_ele %f, lambda_vdw %f" % (time.strftime('%X'), mdsteps, lambda_ele[kk], lambda_vdw[kk]))
      for jj in range(state.get('idx_samps', 0), nsamps):
        alch_lambdas(ctx, lambda_ele[kk], lambda_vdw[kk], lambda_rst[kk])
        intgr.step(1 if useMC else sampsteps)
        if 'Rs' in state:
          mmstate = ctx.getState(getPositions=True, enforcePeriodicBox=True)
          state.Rs[kk][jj] = mmstate.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
        for ll in range(nlambda):  # for MBAR
          alch_lambdas(ctx, lambda_ele[ll], lambda_vdw[ll], lambda_rst[kk])
          state.E_kln[kk,ll,jj] = ctx.getState(getEnergy=True).getPotentialEnergy()/kT
        if verbose:
          if useMC:
            print("MC acceptance rate: %f" % intgr.acceptance_rate)  #(naccept/(1+jj)))  #
          else:
            print(".", end='', flush=True)
      state.idx_samps = 0
  except (KeyboardInterrupt, Exception) as e:
    #print("Handling: %r" % e)
    mmstate = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    state.r = mmstate.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
    state.v = None if useMC else mmstate.getVelocities(asNumpy=True).value_in_unit(UNIT.angstrom/UNIT.picosecond)
    state.idx_lambda = kk
    state.idx_samps = jj
    raise e

  state.E_kln = state.E_kln[:,:,warmup:]
  beta = 1/(KCALMOL_PER_HARTREE*BOLTZMANN*T0)  # 1/kB/T0 in kcal/mol
  mbar = MBAR(state.E_kln, [len(E[0]) for E in state.E_kln])  #[nsamps - warmup]*nlambda)
  res_mbar = mbar.computeEntropyAndEnthalpy(return_dict=True)
  print("MBAR: dF = %f (%f) kcal/mol; dH = %f kcal/mol; T0*dS = %f kcal/mol" % (res_mbar['Delta_f'][0][-1]/beta,
      res_mbar['dDelta_f'][0][-1]/beta, res_mbar['Delta_u'][0][-1]/beta, res_mbar['Delta_s'][0][-1]/beta))
  print("MBAR steps (kcal/mol): %s" % (np.diff(res_mbar['Delta_f'][0])/beta))

  return res_mbar['Delta_f'][0][-1]/beta, res_mbar


def run_fep(file, *args, **kwargs):
  import pickle
  if os.path.exists(file + '.h5'):
    return read_hdf5(file + '.h5', 'dF')
  state = pickle.loads(read_file(file + '.pickle', 'rb')) if os.path.exists(file + '.pickle') else Bunch() #Rs=True
  try:
    dF, res_mbar = openmm_fep(*args, **kwargs)
    write_hdf5(file + '.h5', dF=dF, E_kln=state.E_kln)
    return dF
  except:
    write_file(file + '.pickle', pickle.dumps(state))


# energy matches energy computed from context
#print("E: %f -> %f" % (intgr.getGlobalVariableByName('Eold')/kT.value_in_unit(UNIT.kilojoules_per_mole),
#    intgr.getGlobalVariableByName('Enew')/kT.value_in_unit(UNIT.kilojoules_per_mole)))


def openmm_mc(ctx, T0, nsteps, sampsteps=100):
  """ Run nsteps/sampsteps iterations of Hamiltonian Monte Carlo (randomize velocities, sampsteps of NVE
    integration, Metropolis acceptance) for OpenMM context `ctx` at temperature `T0`, returning position and
    potential energy recorded for iteration
  """
  intgr = ctx.getIntegrator()
  Es, Rs = [], []
  nsteps, sampsteps = int(nsteps), int(sampsteps)  # so user can pass, e.g. 1E6 (which is a float)
  accept, naccept = 0, 0
  kT = UNIT.AVOGADRO_CONSTANT_NA * UNIT.BOLTZMANN_CONSTANT_kB * T0*UNIT.kelvin  # kB*T in OpenMM units
  R0 = ctx.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
  try:
    for ii in range(nsteps//sampsteps):
      ctx.setVelocitiesToTemperature(T0*UNIT.kelvin)
      mmstate = ctx.getState(getEnergy=True)
      U0, K0 = mmstate.getPotentialEnergy()/kT, mmstate.getKineticEnergy()/kT  # U0 should be unchanged
      intgr.step(sampsteps)
      mmstate = ctx.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=True)
      R1, U1, K1 = mmstate.getPositions(asNumpy=True), mmstate.getPotentialEnergy()/kT, mmstate.getKineticEnergy()/kT
      E1, E0 = U1+K1, U0+K0
      accept = 1 if E1 <= E0 or np.exp(-(E1 - E0)) > np.random.rand() else 0
      naccept += accept
      print("E/kT (K+U): %f (%f + %f) -> %f (%f + %f); acceptance: %f" % ( E0, K0, U0, E1, K1, U1, (naccept/(ii+1)) ))
      if accept:
        R0, U0, K0 = R1, U1, K1
      else:
        ctx.setPositions(R0)  #*UNIT.angstrom)
      Es.append((U0*kT).value_in_unit(UNIT.kilojoule_per_mole)/KJMOL_PER_HARTREE)
      #Es[-1] += (K0*kT).value_in_unit(UNIT.kilojoule_per_mole)/KJMOL_PER_HARTREE
      Rs.append(R0.value_in_unit(UNIT.angstrom))
  except (KeyboardInterrupt, Exception) as e:
    print("openmm_mc terminated by exception: ", e)
  return np.asarray(Rs), np.asarray(Es)



ff = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml')

T0 = 300  # Kelvin
filename = '1MCT'  #_A'
if os.path.exists(filename + "_solv.xyz"):
  mol = load_molecule(filename + "_solv.xyz")
else:
  mol = load_molecule('../1MCT.pdb', bonds=False)
  mol = add_hydrogens(mol)
  # 1MCT.pdb has both HE2 and HD1 on HIS A 57
  mol.remove_atoms('/A/57/HE2')  # /A/195/HG - SER HD
  mol.remove_atoms('not protein')
  #mol.remove_atoms('/I//')
  ctx = openmm_EandG_context(mol, ff)
  h_atoms = mol.select('znuc == 1')
  res, mol.r = moloptim(partial(openmm_EandG, ctx=ctx), mol=mol, coords=XYZ(mol, h_atoms), maxiter=100, raiseonfail=False)
  badpairs, badangles = geometry_check(mol)
  protonation_check(mol)
  mol = solvate_prepare(mol, ff, T0, neutral=True, solvent_chain='Z')  #, eqsteps=1)
  write_xyz(mol, filename + "_solv.xyz")


def get_molI(mol):
  molI = mol.extract_atoms('/I/4,5,6/*')
  mutate_residue(molI, 0, 'GLY')
  mutate_residue(molI, 2, 'GLY')
  return add_hydrogens(molI)


fileA = "A_solv.xyz"
if os.path.exists(fileA):
  molA = load_molecule(fileA)
else:
  molA = mol.extract_atoms('/A/*/*')
  molA.append_atoms(get_molI(mol))
  molA = solvate_prepare(molA, ff, T0, neutral=True, solvent_chain='Z')
  write_xyz(molA, fileA)
  append_hdf5('md_restart.h5', v_molA=molA.md_vel)


fileI = "I_solv.xyz"
if os.path.exists(fileI):
  molI = load_molecule(fileI)
else:
  molI = solvate_prepare(get_molI(mol), ff, T0, neutral=True, solvent_chain='Z')
  write_xyz(molI, fileI)
  append_hdf5('md_restart.h5', v_molI=molI.md_vel)


# pretty hacky, but this isn't worth the effort of implementing something more complicated
# - adjust termination threshold and err norm reduction threshold as needed
# - openmmtools restraints use mass-weighting, so we will too
def get_host_restrained(mol, r, ligatoms, hostatoms):
  from scipy.spatial.ckdtree import cKDTree
  ligcentroid = np.sum([mol.mass(ii)*r[ii] for ii in ligatoms], axis=0)/np.sum([mol.mass(ii) for ii in ligatoms])
  kd = cKDTree(r[ligatoms])
  dists, locs = kd.query(r[hostatoms], k=[2])
  nearest = np.asarray(hostatoms)[np.argsort(np.ravel(dists))]
  centatoms, centroid, totmass = [], 0, 0
  for jj in range(10):
    for ii in nearest[:50]:
      if ii in centatoms:
        continue
      nextcent = (totmass*centroid + mol.mass(ii)*r[ii])/(totmass + mol.mass(ii))
      if not centatoms or norm(nextcent - ligcentroid) < 0.85*norm(centroid - ligcentroid):
        #if (len(centatoms) < 1 or np.dot(r[ii] - ligcentroid, centroid - ligcentroid) < 0) and ii not in centatoms:
        centroid = nextcent  #(totmass*centroid + mol.mass(ii)*r[ii])/(totmass + mol.mass(ii))
        totmass += mol.mass(ii)
        centatoms.append(ii)
        err = norm(centroid - ligcentroid)
        print("err = %f" % err)
        if len(centatoms) >= 8 and err < 0.5:
          print("Final centroid offset: %f A" % norm(centroid - ligcentroid))
          return centatoms
  print("Final centroid offset: %f A" % norm(centroid - ligcentroid))
  return centatoms

mol = None

# start interactive
import pdb; pdb.set_trace()


if 1:  #os.path.exists('restraint.h5'):
  #ligrstr, hostrstr, k_rstr = read_hdf5('restraint.h5', 'ligrstr', 'hostrstr', 'k_rstr')
  hostrstr = [2493, 2501, 2560, 2766, 2519, 2763, 2823, 2764]
  ligrstr = [3257, 3260, 3261, 3262, 3263, 3264, 3265, 3266]
  k_rstr = 10.0  #1.3874335023511761
else:
  Rs, Es = read_hdf5('md2.h5', 'Rs', 'Es')
  rmsf = calc_RMSF(Rs)
  ligrstr = molA.select("chain == 'I' and pdb_resnum == 5 and sidechain and znuc > 1")
  hosthvy = [ii for ii in molA.select("chain == 'A' and znuc > 1") if rmsf[ii] < 1.25]
  hostrstr = get_host_restrained(molA, np.mean(Rs, axis=0), ligrstr, hosthvy)

  M = molA.mass()
  offsets = np.linalg.norm(np.sum(M[hostrstr,None]*Rs[:,hostrstr,:], axis=1)/np.sum(M[hostrstr])
      - np.sum(M[ligrstr,None]*Rs[:,ligrstr,:], axis=1)/np.sum(M[ligrstr]), axis=1)
  #plot(offsets, bins=10, fn='hist', xlabel='centroid offset (A)', ylabel='occurences')  # examine distribution
  kT = UNIT.AVOGADRO_CONSTANT_NA * UNIT.BOLTZMANN_CONSTANT_kB * T0*UNIT.kelvin  # kB*T in OpenMM units
  k_rstr = kT.value_in_unit(UNIT.kilocalories_per_mole)/(0.5 + np.std(offsets))**2
  #write_hdf5('restraint.h5', ligrstr=ligrstr, hostrstr=hostrstr, k_rstr=k_rstr)

from openmmtools.forces import HarmonicRestraintForce
bind_rst = HarmonicRestraintForce(k_rstr*UNIT.kilocalories_per_mole/UNIT.angstrom**2, hostrstr, ligrstr)

# l_0_0.h5 - Rs from lambda = 0 w/ no restraint (I flies away)
# l_0_0_rstr.h5 - Rs from lambda = 0 w/ 10 kcal/mol/A^2 restraint

state = Bunch(idx_lambda=11, Rs=True)
dF, res_mbar = openmm_fep(molA, molA.select('/I//;/252/CL-'), ff, T0, state, rst_steps=1, restraint=bind_rst, verbose=1)  #mdsteps=25000

#state = Bunch()
#dF, res_mbar = openmm_fep(molA, molA.select('/I//;/252/CL-'), ff, T0, state)
#dF, res_mbar = openmm_fep(molI, molI.select('/I//;//CL-'), ff, T0, state)


#~ if 0:
#~   Rs, Es = openmm_dynamic(ctx, 1E4, sampsteps=1E2)
#~   write_hdf5('traj1.hd5', Rs=Rs, Es=Es)
#~ else:
#~   mol = load_molecule('1MCT_A_solv.xyz')
#~   Rs, Es = read_hdf5('traj1.hd5', 'Rs', 'Es')

mol = molA
Rs, Es = read_hdf5('md2.h5', 'Rs', 'Es')

od2,og,cg1 = mol.select('/A/189/OD2;/A/190/OG;/A/213/CG1')  #/I/5/NE
pocket = 'any(mol.dist(a, %d) < 6 for a in resatoms)' % od2


  #vis = Chemvis(Mol(mol, Rs, [ VisGeom(style='lines', sel='protein'), VisGeom(style='licorice', sel=pocket, cache=False) ], update_mol=True), wrap=False).run()
  #vis.select([od2]).set_view(center=True)





vis = Chemvis(Mol(mol, Rs, [ VisGeom(style='licorice', sel='not protein'), VisGeom(style='lines', sel='protein'), VisGeom(style='spacefill', sel=nhsel) ], update_mol=True), wrap=False).run()


# list N closest waters (and their distance) per frame
nearhoh = [ sorted([ (norm(r[ii] - r[od2]), rmsf[ii], ii, a.resnum) for ii,a in enumerate(mol.atoms) if a.znuc == 8 and mol.atomres(ii).name == 'HOH' and norm(r[ii] - r[od2]) < 10 ])[:8] for r in Rs ]

# distances covered by atoms during traj
dr = np.diff(Rs, axis=0)
d = np.sum(np.linalg.norm(dr - np.round(dr/mol.pbcbox)*mol.pbcbox, axis=2), axis=0)
#hist, bins = np.histogram(d, bins='auto')
#plot(bins[:-1], hist)

# find waters that aren't moving much ... all near protein as expected
stillwater = [ii for ii in np.argsort(d) if mol.atoms[ii].znuc == 8 and mol.atomres(ii).name == 'HOH']

if 0:
  # dist from stillwaters to protein (for a random frame)
  from scipy.spatial.ckdtree import cKDTree
  r = Rs[100]
  kd = cKDTree(r[selAI])
  dists, locs = kd.query(r[stillwater], k=[2])
  # MM-GBSA binding energy of stillwaters to protein ... closest waters mosly have dE_binding < 0, while others
  #  consistently have ~0.1
  stillres = [mol.atoms[ii].resnum for ii in stillwater]
  np.mean([dE_binding(mol, ffgbsa, selAI, mol.residues[res].atoms, r=Rs[::20]) for res in stillres[:100:10]], axis=1)


## Examine distribution of energies and RMSDs (cross? to initial?) for different sampling techniques
# - esp. RMSD for chain I and nearby atoms
# - dihedrals more informative than RMSD?

# - correlation between rmsf?


#openmm_fep(molA, molA.select('/I//'), ff, T0, useMC=2)



np.corrcoef(calc_RMSF(Rs[:len(Rs)//2]), calc_RMSF(Rs[len(Rs)//2:]))[0,1]

np.corrcoef(calc_RMSF(RsMD2), calc_RMSF(RsMC2))[0,1]

# don't try to render rmsf for solvent ... VMware will crash!
#vis = Chemvis(Mol(molA, [ VisGeom(sel='/A//'), VisGeom(sel='/I//'), VisGeom(sel='/Z//'), VisGeom(sel='protein and znuc > 1', radius=rmsf, style='spacefill', coloring=color_by_constant, colors=decode_color('#7fffff00')) ]), wrap=False).run()

dF_solv = run_fep('solv_fep', molI, molI.select('/I//;//CL-'), ff, T0)

dF_bind = run_fep('bind_fep', molA, molA.select('/I//;/252/CL-'), ff, T0, rst_steps=1, restraint=bind_rst)



# HMC acceptance rate very sensitive to details of integration ... maybe because energy error looks oscillatory
from openmmtools.integrators import HMCIntegrator, GHMCIntegrator, VelocityVerletIntegrator
#mcintgr = HMCIntegrator(T0*UNIT.kelvin, nsteps=100, timestep=1.0*UNIT.femtoseconds)
#mcintgr = GHMCIntegrator(T0*UNIT.kelvin, timestep=2.0*UNIT.femtoseconds)
vvintgr = VelocityVerletIntegrator(2*UNIT.femtoseconds)

ctx = openmm_MD_context(molA, ff, T0, intgr=vvintgr)  #, rigidWater=True)
Rs, Es = openmm_mc(ctx, T0, 20000, sampsteps=100)


# MD
ctx = openmm_MD_context(molA, ff, T0)
ctx.setVelocities(read_hdf5('md_restart.h5', 'v_molA')*(UNIT.angstrom/UNIT.picosecond))
Rs, Es = openmm_dynamic(ctx, 20000, sampsteps=100)

write_hdf5('md2.h5', Rs=Rs, Es=Es)

# MC
ctx = openmm_MD_context(molA, ff, T0, intgr=mcintgr)
Rs, Es = openmm_dynamic(ctx, 200, sampsteps=1)  #00)

write_hdf5('mc2.h5', Rs=Rs, Es=Es)

# .h5 files (A_solv.xyz): mc2 - HMCIntegrator, 100 x 1.0fs steps; md2 - LangevinMiddleIntegrator, 4fs, 100 steps/sample
# gmc: GHMCIntegrator, 2fs, 100 steps/sample; mc3: openmm_mc, 2 fs, 100 steps





# NEXT:
# - go over papers, make notes on entropy for restraints and MM-PBSA
# ... recent review: https://www.frontiersin.org/articles/10.3389/fmolb.2021.712085/full#B177
# - http://getyank.org/latest/algorithms.html

# - get very basic ligand binding FEP working
#  - test restraint w/ fully decoupled LIG (first try no restraint); try Boresch if we need orientational restraint?
#  ... restraint keeps LIG from flying off, but not sure how useful w/o orientational restraint; in any case, if orientation is maintained w/ lambda_vdw = 0.2 we're probably OK
#  - Boresch - see github.com/choderalab/yank/blob/master/Yank/restraints.py
#  - free energy corrections for restraints: github.com/choderalab/openmmtools/blob/master/openmmtools/forces.py

# Acceleration/enhanced sampling:
# - metadynamics (turn on potentials to push away from current phase space position): https://www.cp2k.org/howto:biochem_qmmm
# - attach-pull-release (like it sounds?): https://github.com/slochower/pAPRika
# - replica exchange - run multiple simulations at different temperatures, periodically exchange coords between instances w/ Metropolis acceptance criteria; 10.3390/e16010163
# - umbrella sampling

# issues we won't pursue for now:
# - what's going on with HMC ... do we just need to run for much longer???
#  - https://github.com/kyleabeauchamp/openmmtools/blob/hmc/openmmtools/hmc_integrators.py



# restraint force:
# forces offered by openmmtools don't restain orientation, only position, so correction to free energy is only
#  from translational (and not rotational) entropy: dG = kB*T*ln(vol_std/vol_restrain), where vol_std is
#  N_A/1L - i.e. volume per molecule at 1M concentration.  In the simplest case of a square well (i.e.
#  particle in a box), vol_restrain is just the box volume

# always-on, flat bottom restraint (such that it won't have any effect until other forces are very weak
# vs. harmonic (or flat-bottom?) restraint ramped up as other forces turned off? or before (so we have a check of its effect)

# how to choose atoms for restaint?
# - few? pick one atom of LIG (closest to center?), then two or three atoms of host w/ centroid near chosen LIG atom
# - many? all LIG atoms, then subset of host atoms nearest LIG w/ centroid sufficiently close to centroid of LIG: take all host atoms w/in some distance, then add others iff they move centroid closer to LIG centroid
#  - use average positions over an MD run instead of single set of positions
#  - for this specific system, we might use only ARG sidechain instead of all atoms (but for the usual case of a more fully buried LIG, we could use all atoms)

# alternatives:
# - OpenMM RMSDForce, ramped on, off (instead of analytical calculation of restraint contribution) ... I think this can only be used to constrain conformation (sometimes desired), not position/orientation (in the case that host can move)
# - should we include orientational restraint?
#  - if we find that position-only works fine except maybe at lambda=0, maybe just go with that
#  - just use positional restraints on 2 - 3 parts of LIG?
#  - yank uses Boresch restraint - dist, 2 angles, 3 diheds restrained for 3 host + 3 LIG atoms
#  - test w/ idx_lambda = nlambda - 1
#  - OpenMM CustomCVForce lets us define any force we want
# - for MC, reject samples outside specified box

# MM-PBSA: typically single simulation is run w/ whole system in explicit solvent, then interaction energies of host, LIG, and host+LIG calculated w/ implicit solvent (PBSA/GBSA)

# HEY! since ARG has +1 charge, decoupling will change net charge of system ... options:
# - also scale chage of a Cl- ion
# - use ASP or GLU instead of a GLY to make LIG neutral
# - use -COOH instead of -COO- at C-terminal ... don't think Amber will like this

# we can try vinylfep w/ restraints ... would orientational restraints be desirable or necessary in this case?

# compare numerical calculation for restraint \delta G to analytical; numerical calc in solvent or vacuum would be instructive - similar to our SHO FEP test!

# decoupling (alchemical) approach is only applicable for small ligands - for large ligands (where decoupling would result in significant change in density of system/leave a large void, resulting in long equilib time), PMF (w/ umbrella sampling) can be used

# calculations needed for ligand binding \delta G:
# - LIG solvation (switch interactions in solvent)
# - LIG decoupling (from host + solvent) (w/ restraints; analytical adjustment for restraints)





def NCMM_forces(mol, r=None, sel=None, sel2=None, Gthresh=10.0/KCALMOL_PER_HARTREE):
  from chem.mm import NCMM
  E = {}
  NCMM(mol)(mol, r, components=E)  #vdw=False,
  sel = range(mol.natoms) if sel is None else mol.select(sel)
  sel2 = sel if sel2 is None else mol.select(sel2)
  pairs = [ (i,j) for i in range(mol.natoms) for j in range(i+1, mol.natoms)
      if norm(E['Gqq'][i,j]) > Gthresh and i in sel and j in sel2 ]
  mol.Ebreakdown = { 'charge': {(i,j): E['Eqq'][i,j] for i,j in pairs},
      'vdw-lj': {(i,j): E['Evdw'][i,j] for i,j in pairs} }
  return pairs


#vis = Chemvis(Mol(mol, [ VisGeom(style='licorice', sel='/A/*/*', coloring=scalar_coloring('mmq', [-1,1])), VisGeom(style='licorice', sel='/I/*/*', coloring=scalar_coloring('mmq', [-1,1])), VisGeom(style='lines', sel='not protein') ]), wrap=False).run()

#vis = Chemvis(Mol(mol, [ VisBackbone(style='tubemesh'), VisGeom(style='licorice', sel='/A/*/*'), VisGeom(style='licorice', sel='/I/*/*'), VisGeom(style='lines', sel='not protein'), VisContacts(partial(NCMM_forces, sel='/A/*/*', sel2='/I/*/*', Gthresh=20.0/KCALMOL_PER_HARTREE), style='lines', dash_len=0.1, colors=Color.light_grey) ]), wrap=False).run()

#vis = Chemvis(Mol(mol, [ VisBackbone(style='tubemesh'), VisGeom(style='licorice', sel='protein'), VisGeom(style='lines', sel='not protein') ]), wrap=False).run()

#vis.select(pdb='A 57,102,195 *').set_view()




# examine contacts between A and I chains ... ARG 227 contributes -0.18 H of total -0.36 H electrostatic interaction
if 0:
  from chem.mm import NCMM
  openmm_load_params(mol, ff=ff, charges=True, vdw=True)
  mol = mol.extract_atoms('not water')
  E = {}
  NCMM(mol)(mol, components=E)
  chainA = mol.select('chain != "I"')
  resI = [ ii for ii,res in enumerate(mol.residues) if res.chain == 'I' ]
  Ei = [ (resnum, sum(E['Eqq'][i,j] + E['Evdw'][i,j] for i in mol.residues[resnum].atoms for j in chainA))
      for resnum in resI ]

# mutate ARG 227 (to GLU?), see if I dissociates (then 228, 229, 250)
# remove disulfides from I, see if A 195 OG - I 5 CA distance decreases

# since openmm_EandG_context uses NoCutoff instead of PME, it's super slow for large system
#ctx = openmm_EandG_context(mol, ff)
if 0:
  mol = mutate_residue(mol, '/I/5/', 'GLU_LL_DHE2')
  active = mol.select('/I/5/~CA,C,N')

  ctx = openmm_MD_context(mol, ff, T0)
  res, mol.r = moloptim(partial(openmm_EandG, ctx=ctx), mol=mol, coords=XYZ(mol, active), maxiter=10, raiseonfail=False)
  ctx.setPositions(mol.r*UNIT.angstrom)

# more drastic
mol.remove_atoms('/I/1,2,3,4,5,6,7,8/*;/I/9/H')
mol = add_hydrogens(mol)  # fix CYS I 20 and N-terminal NH3 ... we need an option to only process selected residues
#mol = mutate_residue(mol, '/I/20/', 'CYS_LL')  # restore HG for broken disulfide bond
ctx = openmm_MD_context(mol, ff, T0, p0=1.0)  # const. pressure

#import cProfile  cProfile.run("vis.run()", sort='cumtime') ... VisBackbone is slow
vis = Chemvis(Mol(mol, [ VisBackbone(style='tubemesh'), VisGeom(style='licorice', sel='/A/*/*'), VisGeom(style='licorice', sel='/I/*/*'), VisGeom(style='lines', sel='not protein') ]), wrap=False).run()

Rs, Es = openmm_dynamic(ctx, 1E4, sampsteps=1E2, vis=vis)





# completely disable A - I electrostatic interactions and see how long until they separate?


# we want to look at the cavity around ARG I 5
# ... what is the point?  Don't we just want to do standard ligand binding free energy w/ I 4,5,6 ???
