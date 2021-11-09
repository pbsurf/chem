from io import StringIO
from chem.basics import *
from chem.mm import *
from chem.io import load_molecule, write_xyz, write_hdf5
from chem.io.openmm import *
from chem.fep import *
from chem.data.test_molecules import ethanol_gaff_xyz, ethane_gaff_xyz

# Refs:
# - openmm.org/tutorials/alchemical-free-energy/
# - sites.google.com/site/amoebaworkshop/intro - EtOH solvation
# - www.ks.uiuc.edu/Training/Workshop/Urbana_2010A/lectures/TCBG-2010.pdf - slides
# - www.ks.uiuc.edu/Training/Tutorials/namd/PLB/tutorial-protein-ligand.pdf - ligand binding free energy tutorial
# - dasher.wustl.edu/ponder/papers/jcc-38-2047-17.pdf - Tinker-OpenMM Amoeba free energy; host-ligand restraint
# - dasher.wustl.edu/ponder/papers/wang-thesis.pdf - Tinker Amoeba free energy
# More
# - 10.33011/livecoms.1.1.5957 - MD best practices
# - 10.33011/livecoms.2.1.18378 / arxiv.org/pdf/2008.03067.pdf - Alchemical free energy best practices
# - github.com/samplchallenges - blind challenges for binding free energies and more; SAMPL6 paper 10.1007/s10822-020-00290-5 is useful

T0 = 300  # Kelvin

filename = 'ethanol0'
#filename = 'ethane0'
try:
  mol = load_molecule(filename + '.xyz')
  ligparm_key = read_file(filename + '.ff.xml')
except:
  mol = resp_prepare(load_molecule(ethanol_gaff_xyz, residue='LIG'), avg=[[3,4,5], [6,7]])
  #mol = resp_prepare(load_molecule(ethane_gaff_xyz, residue='LIG'), constr=[[0,1]], avg=[[2,3,4], [5,6,7]])
  mol.mmtype = mol.name
  write_xyz(mol, filename + '.xyz')
  ligparm_key = openmm_resff(mol)
  write_file(filename + '.ff.xml', ligparm_key)


ff = openmm_app.ForceField('amber99sb.xml', 'tip3p.xml', '/home/mwhite/qc/gaff.xml', StringIO(ligparm_key))

try:
  mol = load_molecule(filename + "_solv.xyz")
except:
  from chem.qmmm.prepare import water_box, solvate
  side = np.max(np.diff(mol.extents(pad=6.0), axis=0))
  solvent = water_box(side, mmtypes=['OW','HW','HW'])
  mol.r = mol.r - np.mean(mol.extents(), axis=0)
  mol = solvate(mol, solvent, d=2.0, solute_res='LIG')
  mol.r = mol.r + 0.5*side  # OpenMM centers box at side/2 instead of origin

  top = openmm_top(mol)
  sys = ff.createSystem(top, nonbondedMethod=openmm_app.CutoffPeriodic,  #PME,
      nonbondedCutoff=min(0.5*min(mol.pbcbox), 10)*UNIT.angstrom, constraints=openmm_app.HBonds)
  #sys.addForce(MonteCarloBarostat(1*bar, T0*kelvin))
  intgr = openmm.LangevinMiddleIntegrator(T0*UNIT.kelvin, 1/UNIT.picosecond, 0.004*UNIT.picoseconds)
  ctx = openmm.Context(sys, intgr)  #sim = Simulation(top, sys, intgr)
  ctx.setPositions(mol.r*UNIT.angstrom)  # Angstroms to nm
  openmm.LocalEnergyMinimizer.minimize(ctx, maxIterations=100)
  intgr.step(5000)
  simstate = ctx.getState(getPositions=True, enforcePeriodicBox=True)
  mol.r = simstate.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
  mol.pbcbox = np.diag(simstate.getPeriodicBoxVectors(asNumpy=True).value_in_unit(UNIT.angstrom))
  write_xyz(mol, filename + "_solv.xyz")


ligatoms = mol.select('* LIG *')
top = openmm_top(mol)

# openmmtools supports PME - superior to reaction field electrostatics (alch_setup)
if 1:
  from openmmtools import alchemy

  basesystem = ff.createSystem(top, nonbondedMethod=openmm_app.PME,
      nonbondedCutoff=min(0.5*min(mol.pbcbox), 10)*UNIT.angstrom, constraints=openmm_app.HBonds)
  alchregion = alchemy.AlchemicalRegion(alchemical_atoms=ligatoms)
  alchfactory = alchemy.AbsoluteAlchemicalFactory()
  system = alchfactory.create_alchemical_system(basesystem, alchregion)
  alchstate = alchemy.AlchemicalState.from_system(system)

  def alch_lambdas(ctx, l_ele, l_vdw):
    alchstate.lambda_electrostatics, alchstate.lambda_sterics = l_ele, l_vdw
    alchstate.apply_to_context(ctx)

else:
  system = ff.createSystem(top, nonbondedMethod=openmm_app.CutoffPeriodic,  #NoCutoff
      nonbondedCutoff=min(0.5*min(mol.pbcbox), 10)*UNIT.angstrom, constraints=openmm_app.HBonds)
  # constant pressure ... vdW sampling is too noisy to tell if this is making a difference
  #system.addForce(openmm.MonteCarloBarostat(1*UNIT.bar, T0*UNIT.kelvin))
  # Simulation.reporters doesn't do anything fancy - just calculates steps to next report and runs that many steps
  #sim.reporters.append(PDBReporter('output.pdb', 1000))
  alch_setup(system)

  def alch_lambdas(ctx, l_ele, l_vdw):
    ctx.setParameter('lambda_ele', l_ele)
    ctx.setParameter('lambda_vdw', l_vdw)

# Monte Carlo ... openmmtools docs give examples of other enhanced sampling techniques, e.g. relica exchange
# HMC scales better with dimension than plain (random-walk) MC - see arxiv.org/pdf/1206.1901.pdf
useMC = True  # Monte Carlo
if useMC:
  # HMC (Hybrid MC aka Hamiltonian MC) randomizes velocities, measures total energy, then applies Metropolis
  #  criterion to total energy difference dE after N steps of microcanonical evolution; for leap frog
  #  integrator w/ step size dt, energy error dE ~ dt^2.  Large acceptance rates (closer to 1.0 than 0.5) are
  #  typical and desired.  openmmtools HMCIntegrator has ~0.85 ... for some reason, trying to do the same
  #  thing with setVelocitiesToTemperature() and VelocityVerletIntegrator() (so that position and velocity,
  #  and hence PE and KE, are in sync) gives ~0.995 due to a few kB*T decr in total energy with first step !?!
  # To use MC moves, samplers from openmmtools.mcmc with alchemy, looks like we use CompoundThermodynamicState
  # ... but not clear what these abstractions add - might as well just use YANK
  from openmmtools.integrators import HMCIntegrator, VelocityVerletIntegrator

  # for each step, randomizes velocities then takes nsteps Verlet steps before accepting or rejecting state
  intgr = HMCIntegrator(T0*UNIT.kelvin, nsteps=100, timestep=2*UNIT.femtoseconds)  # incr timestep to decr acceptance
  #intgr = VelocityVerletIntegrator(2*UNIT.femtoseconds)
else:
  # seems we have to create Context after finishing setup of System
  intgr = openmm.LangevinMiddleIntegrator(T0*UNIT.kelvin, 1/UNIT.picosecond, 0.004*UNIT.picoseconds)
  #intgr = openmm.VerletIntegrator(0.002*UNIT.picoseconds)

ctx = openmm.Context(system, intgr)
ctx.setPositions(mol.r*UNIT.angstrom)
if mol.pbcbox is not None:
  ctx.setPeriodicBoxVectors(*[v*UNIT.angstrom for v in np.diag(mol.pbcbox)])
#ctx.setVelocitiesToTemperature(T0*UNIT.kelvin)
openmm_load_params(mol, system)  # get charges

# have to turn off electrostatics before vdW with soft-core LJ - MD fails for, e.g.,
#  lambda_ele = lambda_vdw = 0.5 due to LIG atom coming too close to solvent atom
lambda_ele = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
lambda_vdw = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
nlambda = len(lambda_ele)

mdsteps = 25000  # total steps
sampsteps = 100  # steps between samples - see autocorrelation time check below
nsamps = mdsteps//sampsteps
E_kln = np.zeros((nlambda, nlambda, nsamps))  #, np.float64)
#dEup, dEdn = np.zeros((nlambda, nsamps)), np.zeros((nlambda, nsamps))
#Rs = np.zeros((nlambda, nsamps, mol.natoms, 3))
kT = UNIT.AVOGADRO_CONSTANT_NA * UNIT.BOLTZMANN_CONSTANT_kB * T0*UNIT.kelvin  # kB*T in OpenMM units

# start interactive
import pdb; pdb.set_trace()

for kk in range(nlambda):
  print("%s: running %d MD steps for lambda_ele %f, lambda_vdw %f" % (time.strftime('%X'), mdsteps, lambda_ele[kk], lambda_vdw[kk]))
  for jj in range(nsamps):
    alch_lambdas(ctx, lambda_ele[kk], lambda_vdw[kk])
    #openmm.LocalEnergyMinimizer.minimize(ctx, maxIterations=100)
    #ctx.setVelocitiesToTemperature(T0*UNIT.kelvin)
    intgr.step(1 if useMC else sampsteps)
    #Rs[kk, jj] = ctx.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
    #Ekk = ctx.getState(getEnergy=True).getPotentialEnergy()
    #if kk+1 < nlambda:
    #  alch_lambdas(ctx, lambda_ele[kk+1], lambda_vdw[kk+1])
    #  dEup[kk, jj] = (ctx.getState(getEnergy=True).getPotentialEnergy() - Ekk)/kT
    #if kk > 0:
    #  alch_lambdas(ctx, lambda_ele[kk-1], lambda_vdw[kk-1])
    #  dEdn[kk, jj] = (ctx.getState(getEnergy=True).getPotentialEnergy() - Ekk)/kT
    for ll in range(nlambda):  # for MBAR
      alch_lambdas(ctx, lambda_ele[ll], lambda_vdw[ll])
      E_kln[kk,ll,jj] = ctx.getState(getEnergy=True).getPotentialEnergy()/kT
  if useMC:
    print("MC acceptance rate: %f" % intgr.acceptance_rate)  #(naccept/nsamps)


warmup = 5
E_kln = E_kln[:,:,warmup:]
#dEup, dEdn = dEup[:,:,warmup:], dEdn[:,:,warmup:]
#write_hdf5("fep_%.0f.h5" % time.time(), dEup=dEup, dEdn=dEdn)
write_hdf5("mbar_%.0f.h5" % time.time(), E_kln=E_kln)

dEup = np.array([E_kln[k, k+1 if k+1 < nlambda else k] - E_kln[k, k] for k in range(nlambda)])
dEdn = np.array([E_kln[k, k-1 if k-1 >= 0 else k] - E_kln[k, k] for k in range(nlambda)])
# our BAR and FEP calc
fep_results(dEup, dEdn, T0)

# pymbar
from pymbar import MBAR, BAR
beta = 1/(KCALMOL_PER_HARTREE*BOLTZMANN*T0)  # 1/kB/T0 in kcal/mol

mbar = MBAR(E_kln, [len(E[0]) for E in E_kln])  #[nsamps - warmup]*nlambda)
#dFs_mbar = mbar.getFreeEnergyDifferences(return_dict=True)
res_mbar = mbar.computeEntropyAndEnthalpy(return_dict=True)
print("MBAR: dF = %f (%f) kcal/mol; dH = %f kcal/mol; T0*dS = %f kcal/mol" % (res_mbar['Delta_f'][0][-1]/beta,
    res_mbar['dDelta_f'][0][-1]/beta, res_mbar['Delta_u'][0][-1]/beta, res_mbar['Delta_s'][0][-1]/beta))
print("MBAR steps (kcal/mol): %s" % (np.diff(res_mbar['Delta_f'][0])/beta))

dFs = np.array([ BAR(dEup[ii], dEdn[ii+1]) for ii in range(len(dEup)-1) ])/beta
dF = np.sum(dFs[:,0])
print("pymbar BAR: dF = %f;\ndF steps:\n%r\nddF steps:\n%r" % (dF, dFs[:,0], dFs[:,1]))

# FEP and BAR enthalpy and entropy
dF_fepup = np.sum([fep(dEup[ii])/beta for ii in range(len(dEup)-1)])
dF_fepdn = np.sum([fep(dEdn[ii+1])/beta for ii in range(len(dEdn)-1)])
dH_fepup = np.sum([fep_enthalpy(E_kln[k,k], E_kln[k,k+1])/beta for k in range(nlambda-1)])
dH_fepdn = np.sum([fep_enthalpy(E_kln[k+1,k+1], E_kln[k+1,k])/beta for k in range(nlambda-1)])
TdS_fepup = dH_fepup - dF_fepup
TdS_fepdn = dH_fepdn - dF_fepdn
print("FEP (kcal/mol): fwd: dF = %f, dH = %f, T0*dS = %f; rev: dF = %f, dH = %f, T0*dS = %f" %
    (dF_fepup, dH_fepup, TdS_fepup, dF_fepdn, dH_fepdn, TdS_fepdn))

dF_bar = np.array([bar(dEup[ii], dEdn[ii+1])/beta for ii in range(len(dEup)-1)] )
dH_bar = np.array([bar_enthalpy(E_kln[k,k], E_kln[k+1,k], E_kln[k,k+1], E_kln[k+1,k+1])/beta for k in range(nlambda-1)])
TdS_bar = dH_bar - dF_bar
print("BAR: dF = %f kcal/mol; dH = %f kcal/mol; T0*dS = %f kcal/mol\nBAR steps dF (kcal/mol): %s" %
    (np.sum(dF_bar), np.sum(dH_bar), np.sum(TdS_bar), dF_bar))

# visualizing distributions
hup = [np.histogram(dEup[ii]) for ii in range(nlambda-1)]
hdn = [np.histogram(-dEdn[ii+1]) for ii in range(nlambda-1)]
for ii in range(nlambda-1):
  plot(hup[ii][1][:-1], hup[ii][0], hdn[ii][1][:-1], hdn[ii][0], subplots=(5,2), subplot=ii)


## debugging stuff

# HMC w/o openmmtools ... seems to work, just don't understand decrease of total energy on first intgr step
#  naccept = 0
#
#    ctx.setVelocitiesToTemperature(T0*UNIT.kelvin)  # this applies velocity constraints
#    st = ctx.getState(getEnergy=True)
#    E0 = st.getPotentialEnergy()/kT + st.getKineticEnergy()/kT
#    #intgr.step(100)
#    pe, ke = np.zeros(1000), np.zeros(1000)  #st.getPotentialEnergy()/kT, st.getKineticEnergy()/kT
#    #print("Initial PE: %f; KE: %f; E: %f" % (pe, ke, pe+ke))
#    for n in range(1000):
#      st = ctx.getState(getEnergy=True)
#      pe[n], ke[n] = st.getPotentialEnergy()/kT, st.getKineticEnergy()/kT
#      intgr.step(1)
#      #print("PE: %f; KE: %f; E: %f" % (pe, ke, pe+ke))
#
#    # I believe the high freq peak is H2O bend (freq of KE,PE oscillation is twice mode freq of ~1500/cm)
#    plot(range(1000), pe - np.mean(pe), range(1000), ke - np.mean(ke))
#    plot(np.fft.fftfreq(len(pe)), np.abs(np.fft.fft(pe - np.mean(pe))))
#
#
#    st = ctx.getState(getEnergy=True)
#    E1 = st.getPotentialEnergy()/kT + st.getKineticEnergy()/kT
#    #print("E1: %f; E0 %f; metropolis: %f; naccept: %d" % (E1, E0, np.exp(-(E1 - E0)), naccept))
#    if E1 <= E0 or np.exp(-(E1 - E0)) > np.random.rand() or jj == 0:
#      naccept += 1
#      Rcur = ctx.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
#    else:
#      ctx.setPositions(Rcur)
#
#    Rs[kk, jj] = Rcur.value_in_unit(UNIT.angstrom)


#print(ctx.getState(
#    getPositions=True, enforcePeriodicBox=True).getPeriodicBoxVectors(asNumpy=True).value_in_unit(UNIT.angstrom))

# -0.1308658_7230492316 H (-343.5883 kJ/mol, -137.747 kB*T); coulomb gives -0.1308658_6469427886 H
#alchem.addInteractionGroup(set(ligatoms), set(range(system.getNumParticles())) - set(ligatoms))
#alchem.setForceGroup(1)
#ctx.setParameter('lambda_vdw', 0.0)
#ctx.getState(getEnergy=True, groups=set([1])).getPotentialEnergy() #_.value_in_unit(UNIT.kilojoule_per_mole)/KJMOL_PER_HARTREE
#NCMM(mol, lj=False, cutoff=min(mol.pbcbox)/2)(mol.r)

# let's check autocorrelation time
if 0:
  E = np.zeros(2000)
  for ii in range(len(E)):
    intgr.step(10)
    E[ii] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(UNIT.kilojoule_per_mole)/KJMOL_PER_HARTREE

  # 80 steps (4fs steps)
  print("Integrated autocorr. time %f" % int_autocorr_time(E))

# confirm that solute is invisible to solvent when lambda == 0
if 0:
  ctx.setParameter('lambda_ele', 0.0)
  ctx.setParameter('lambda_vdw', 0.0)
  Rs, Es = openmm_dynamic(ctx, 2500, 100)
  from chem.vis import *
  vis = Chemvis(Mol(mol, Rs, [VisGeom(style='licorice') ]), wrap=False).run()

# explore failure when lambda_vdw < 1 and lambda_ele > 0
if 0:
  ctx.setParameter('lambda_ele', 0.5)
  ctx.setParameter('lambda_vdw', 0.5)
  Rs, Es, Gs = openmm_dynamic(ctx, 2000, 1, grad=True)  # blows up

# try different electrostatic interaction cutoffs
# - main observation is magnitude of dE incr w/ increasing cutoff dist, so larger value seen w/ PME is plausible
if 0:
  f_str += "Eele = lam_ele*138.935456*q1q2*select(step(rcut - r), 1.0/r + krf*r^2 - crf, 0.0);"
  f_str += "krf = {c0}/rcut^3; crf = {c1}/rcut;".format(c0=(eps - 1)/(2*eps + 1), c1=3*eps/(2*eps + 1))
  alchem.addGlobalParameter('rcut', rcut_nm)
  # ...
  cutoffs = np.linspace(0.8, 1.0, 9)*rcut_nm
  dEcuts = np.zeros((len(cutoffs), nsamps))
  for jj in range(nsamps):
    ctx.setParameter('rcut', rcut_nm)
    intgr.step(sampsteps)
    for kk, rcut in enumerate(cutoffs):
      ctx.setParameter('rcut', rcut)
      ctx.setParameter('lambda_ele', 0.0)
      E0 = ctx.getState(getEnergy=True).getPotentialEnergy()
      ctx.setParameter('lambda_ele', 1.0)
      dEcuts[kk, jj] = (ctx.getState(getEnergy=True).getPotentialEnergy() - E0)/kT

## aside - test QMMM with OpenMM
# github.com/CCQC/janus/blob/master/janus/mm_wrapper/openmm_wrapper.py is a good ref
qmbasis = '6-31G*'
qmatoms = [0,3,4,5]
init_qmatoms(mol, qmatoms, qmbasis)

from chem.qmmm.qmmm1 import QMMM
# subtract: OpenMM can't match residue for fragment ... we could try setting masses to zero instead
qmmm = QMMM(mol, qmatoms=qmatoms, mm_key=ff,
    mm_opts=Bunch(newcharges=False),
    capopts=Bunch(basis=qmbasis, placement='rel', g_CH=0.714),
    chargeopts=Bunch(scaleM1=1.0, adjust='dipole', dipolecenter=0.75, dipolesep=0.5))

E,G = qmmm.EandG(mol.r)

# check QM/MM setup
mqq = Molecule(atoms=[Atom(name=q.name, r=q.r, mmq=q.qmq) for q in qmmm.charges if q.qmq != 0])
mcap = Molecule(atoms=[Atom(name='HL', r=c.r) for c in qmmm.caps])
from chem.vis import *
vis = Chemvis([Mol(qmmm.mol, VisGeom(style='lines')), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))), Mol(mcap, VisGeom(style='lines'))], fog=True).run()
