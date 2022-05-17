import openmm
import openmm.app #as openmm_app
import openmm.unit as UNIT
from ..basics import *
from ..molecule import Residue

EUNIT = UNIT.kilojoule_per_mole*KJMOL_PER_HARTREE

def openmm_top(mol):
  """ generate OpenMM topology from chem Molecule """
  assert len(mol.residues) > 0, "Residues required to created OpenMM topology"
  top = openmm.app.Topology()
  # preserve order of chains (Python 3.7+ - use OrderedDict for Python 2)
  chains = list(dict.fromkeys(res.chain for res in mol.residues))  #set()
  topchn = {chain: top.addChain(chain) for chain in chains}
  topres = [top.addResidue(res.name, topchn[res.chain]) for res in mol.residues]
  topatm = [top.addAtom(a.name,
      openmm.app.Element.getByAtomicNumber(a.znuc), topres[a.resnum or 0]) for a in mol.atoms]
  for a1,a2 in mol.get_bonds():
    top.addBond(topatm[a1], topatm[a2])
  if mol.pbcbox is not None:
    top.setUnitCellDimensions(mol.pbcbox*UNIT.angstrom)  # Angstroms to nm
  assert all(atop.element.atomic_number == amol.znuc for atop,amol in zip(top.atoms(), mol.atoms)), \
      "OpenMM atom order does not match Molecule!"
  return top


def openmm_resff(mol, residues=None):
  """ generate OpenMM force field w/ residue for mol, containing atomic charges """
  import xml.etree.ElementTree as ET
  root = ET.Element("ForceField")
  xmlresidues = ET.SubElement(root, "Residues")
  residues = range(len(mol.residues)) if residues is None else residues
  for residx in residues:
    residue = mol.residues[residx]
    xmlresidue = ET.SubElement(xmlresidues, "Residue", name=residue.name)
    for ii in residue.atoms:
      ET.SubElement(xmlresidue,
          "Atom", name="X" + str(ii+1), type=mol.atoms[ii].mmtype, charge=("%.4f" % mol.atoms[ii].mmq))
    for a1,a2 in mol.get_bonds(active=residue.atoms):
      ET.SubElement(xmlresidue, "Bond", atomName1="X" + str(a1+1), atomName2="X" + str(a2+1))
    #etree.SubElement(residue, "ExternalBond", atomName=bond.atom1.name)
  # serialize to XML string
  return ET.tostring(root, encoding='unicode').replace('>', '>\n')  # no easy way to pretty print


def openmm_make_inactive(mol, sys, inactive):
  """ disable bonded and 1-4 vdW interactions involving only atoms in `inactive` and set charges to zero """
  inactive = sorted(inactive)
  forces = {f.__class__: f for f in sys.getForces()}
  bndforce, angforce, torforce, nbforce = forces[openmm.HarmonicBondForce], \
      forces[openmm.HarmonicAngleForce], forces[openmm.PeriodicTorsionForce], forces[openmm.NonbondedForce]
  # bonded forces
  for ii in range(bndforce.getNumBonds()):
    bnd = bndforce.getBondParameters(ii)
    if all(a in inactive for a in bnd[:2]):
      bnd[-1] = 0.0  # set force constant to zero
      bndforce.setBondParameters(ii, *bnd)
  for ii in range(angforce.getNumAngles()):
    ang = angforce.getAngleParameters(ii)
    if all(a in inactive for a in ang[:3]):
      ang[-1] = 0.0  # set force constant to zero
      angforce.setAngleParameters(ii, *ang)
  for ii in range(torforce.getNumTorsions()):
    tor = torforce.getTorsionParameters(ii)
    if all(a in inactive for a in tor[:4]):
      tor[-1] = 0.0  # set force constant to zero
      torforce.setTorsionParameters(ii, *tor)
  # QM charge - (QM+MM) charge interaction handled at QM level
  for ii in inactive:
    charge, sigma, epsilon = nbforce.getParticleParameters(ii)
    nbforce.setParticleParameters(ii, charge*0, sigma, epsilon)
  # exclusions for QM-QM vdW
  bonds, angles, _ = mol.get_internals(active=inactive)
  excluded = frozenset(bonds + [(a[0], a[2]) for a in angles])
  for ii, a1 in enumerate(inactive):
    for a2 in inactive[ii+1:]:
      if (a1, a2) not in excluded:  # a1 < a2 in excluded
        nbforce.addException(a1, a2, 0.0, 0.34, 0.0, replace=True)


# vdw works for AMBER but not CHARMM, which uses custom non-bonded force for LJ
def openmm_load_params(mol, ff=None, charges=True, vdw=False, bonded=False):
  sys = ff.createSystem(openmm_top(mol), nonbondedMethod=openmm.app.NoCutoff) if hasattr(ff, 'createSystem') else ff
  forces = {f.__class__: f for f in sys.getForces()}
  bndforce, angforce, torforce, nbforce = forces[openmm.HarmonicBondForce], \
      forces[openmm.HarmonicAngleForce], forces[openmm.PeriodicTorsionForce], forces[openmm.NonbondedForce]
  for ii, atom in enumerate(mol.atoms):
    charge, sigma, eps = nbforce.getParticleParameters(ii)
    if charges:
      atom.mmq = charge/UNIT.elementary_charge
    if vdw:
      atom.lj_r0 = sigma/UNIT.angstrom * 2**(1/6.0)
      atom.lj_eps = eps/EUNIT

  if bonded:
    mol.mm_stretch, mol.mm_bend, mol.mm_torsion, mol.mm_imptor = [], [], [], []
    for ii in range(bndforce.getNumBonds()):
      a1, a2, d, k = bndforce.getBondParameters(ii)
      # openmm uses 0.5*k*(x-x0)^2 whereas we use k*(x-x0)^2 (as in the AMBER spec)
      mol.mm_stretch.append(( [a1, a2], 0.5*k/(EUNIT/UNIT.angstrom**2), d/UNIT.angstrom ))
    for ii in range(angforce.getNumAngles()):
      a1, a2, a3, d, k = angforce.getAngleParameters(ii)
      mol.mm_bend.append(( [a1, a2, a3], 0.5*k/(EUNIT/UNIT.radian**2), d/UNIT.radian ))
    for ii in range(torforce.getNumTorsions()):
      a1, a2, a3, a4, period, phase, k = torforce.getTorsionParameters(ii)
      mol.mm_torsion.append(( [a1, a2, a3, a4], [ (k/EUNIT, phase/UNIT.radian, period) ] ))


# since we may have active != bondactive, just pass both instead of making a fn to fix up MM params for
#  extract_atoms/remove_atoms
def openmm_create_system(mol, active=None, bondactive=None, nbargs={}):
  active = mol.listatoms() if active is None else sorted(mol.select(active))
  bondactive = frozenset(active if bondactive is None else mol.select(bondactive))
  idx_map = dict(zip(active, range(len(active))))
  map_idxs = lambda bb: tuple(idx_map[b] for b in bb)
  bonds, angles, diheds = mol.get_internals()

  sys = openmm.System()
  bndforce = openmm.HarmonicBondForce()
  angforce = openmm.HarmonicAngleForce()
  torforce = openmm.PeriodicTorsionForce()
  nbforce = openmm.NonbondedForce()

  for bnd in mol.mm_stretch:
    if all(b in bondactive for b in bnd[0]):
      bndforce.addBond(*map_idxs(bnd[0]), bnd[2]*UNIT.angstrom, bnd[1]/0.5*(EUNIT/UNIT.angstrom**2))
  for ang in mol.mm_bend:
    if all(b in bondactive for b in ang[0]):
      angforce.addAngle(*map_idxs(ang[0]), ang[2]*UNIT.radian, ang[1]/0.5*(EUNIT/UNIT.radian**2))
  for tor in mol.mm_torsion:
    if all(b in bondactive for b in tor[0]):
      assert len(tor[1]) == 1, "Add support for non-openmm torsions"
      torforce.addTorsion(*map_idxs(tor[0]), tor[1][0][2], tor[1][0][1]*UNIT.radian, tor[1][0][0]*EUNIT)
  assert not getattr(mol, 'mm_imptor', None), "Need to do mm_torsion + mm_imptor!"

  for ii in active:
    a = mol.atoms[ii]
    sys.addParticle(ELEMENTS[a.znuc].mass)
    nbforce.addParticle(a.mmq*UNIT.elementary_charge, a.lj_r0/2**(1/6.0)*UNIT.angstrom, a.lj_eps*EUNIT)

  mapped_bonds = [ map_idxs(bb) for bb in bonds if bb[0] in idx_map and bb[1] in idx_map ]
  nbforce.createExceptionsFromBonds(mapped_bonds, 1/1.2, 1/2.0)
  for name,val in nbargs.items():
    getattr(nbforce, 'set' + name.title())(val)

  sys.addForce(bndforce)
  sys.addForce(angforce)
  sys.addForce(torforce)
  sys.addForce(nbforce)
  return sys


# freezemass = 0 works for LocalEnergyMinimizer, whereas huge freezemass works for constraints
def openmm_MD_context(mol, ff, T0,
    p0=None, dt=4, intgr=None, r0=None, v0=None, freeze=[], freezemass=1E10, forces=[], **kwargs):
  """ create OpenMM context for `mol`, w/ force field `ff`, temperature `T0` (Kelvin), and optionally,
    pressure `p0` (bar), timestep `dt` (fs, default 4fs), integrator `intgr` (default: LangevinMiddleIntegrator),
    with initial  positions and velocities r0 (Ang) and v0 (Ang/ps - default random thermal velocities for T0).
    Atoms specified by `freeze` are fixed by setting mass to `freezemass` (default 1E10 - use 0 instead for
    LocalEnergyMinimizer).  Additional `forces` added to system.  Other keyword args passed to ff.createSystem()
  """
  isPBC = mol.pbcbox is not None
  nbmethod = openmm.app.PME if isPBC else openmm.app.NoCutoff
  sysargs = dict(dict(constraints=openmm.app.HBonds, rigidWater=True, nonbondedMethod=nbmethod), **kwargs)
  sys = ff.createSystem(openmm_top(mol), **sysargs) if hasattr(ff, 'createSystem') else ff
  for force in forces:
    sys.addForce(force)
  assert freeze is not None, "use freeze = [] instead of freeze = None"
  for ii in mol.select(freeze):
    sys.setParticleMass(ii, freezemass)
  if p0 is not None:
    sys.addForce(openmm.MonteCarloBarostat(p0*UNIT.bar, T0*UNIT.kelvin))
  if intgr is None:
    intgr = openmm.LangevinMiddleIntegrator(T0*UNIT.kelvin, 1/UNIT.picosecond, dt*UNIT.femtoseconds)
  ctx = openmm.Context(sys, intgr)
  ctx.setPositions((mol.r if r0 is None else r0)*UNIT.angstrom)
  if np.ndim(v0) > 1:
    ctx.setVelocities(v0*(UNIT.angstrom/UNIT.picosecond))
  else:
    ctx.setVelocitiesToTemperature(T0*UNIT.kelvin)
  if isPBC:
    ctx.setPeriodicBoxVectors(*[v*UNIT.angstrom for v in np.diag(mol.pbcbox)])
  return ctx


# Notes on MD failures ("Particle coordinate is nan"):
# - excessively strained system may fail w/ our default timestep
# - get positions from current state and see which are NaN
# - for rapid failures, use sampsteps=1 and examine last few frames
#  - could we save last velocity to help quickly reproduce failure for less rapid cases?
def openmm_dynamic(ctx, nsteps, sampsteps=100, grad=False, vel=False, sampfn=None, vis=None, verbose=1):
  """ Run OpenMM context `ctx` for `nsteps` and return position, potential energy, and, optionally, gradient
    (if `grad` is True) recorded every `sampsteps` steps (also call `sampfn` and update `vis`, if provided).
  """
  intgr = ctx.getIntegrator()
  isPBC = ctx.getSystem().usesPeriodicBoundaryConditions()
  Es, Rs, Gs, Vs = [], [], [], []
  nsteps, sampsteps = int(nsteps), int(sampsteps)  # so user can pass, e.g. 1E6 (which is a float)
  try:
    for ii in range(nsteps//sampsteps):
      intgr.step(sampsteps)
      if sampfn:
        sampfn(ii, ctx)
      state = ctx.getState(getEnergy=True, getPositions=True,
          getForces=grad, getVelocities=vel, enforcePeriodicBox=isPBC)
      Es.append(state.getPotentialEnergy()/EUNIT)
      #Es[-1] += state.getKineticEnergy()/EUNIT
      Rs.append(state.getPositions(asNumpy=True)/UNIT.angstrom)
      if grad:
        Gs.append(-state.getForces(asNumpy=True)/(EUNIT/UNIT.angstrom))
      if vel:
        Vs.append(state.getVelocities(asNumpy=True)/(UNIT.angstrom/UNIT.picosecond))
      if vis:
        vis.refresh(r=Rs[-1], repaint=True)
      if verbose > 0:
        print('.', end='', flush=True)
  except (KeyboardInterrupt, Exception) as e:
    print("openmm_dynamic terminated by %s: %s" % (e.__class__.__name__, e))
  print('\a')  # beep!
  return tuple(np.asarray(a) for a in (Rs, Es, Gs, Vs) if len(a) > 0)


# OpenMM doesn't support Hessian calculations; see github.com/leeping/forcebalance/blob/master/src/openmmio.py
#  for Hessian calc via finite differencing; also see github.com/Hong-Rui/Normal_Mode_Analysis

def openmm_EandG_context(mol, ff=None, inactive=None, charges=None, epsilons=None, **kwargs):
  """ create an OpenMM context for single point energy and gradient calculations for `mol` using force field
    (default Amber99sb) or system `ff`; `inactive` atoms are made inactive (see openmm_make_inactive);
    charges, vdW epsilon parameters overridden by lists of (atom number, value) pairs `charges`, `epsilons`
  """
  #sysargs = dict(dict(nonbondedMethod=openmm.app.NoCutoff), **kwargs)
  ff = openmm.app.ForceField('amber99sb.xml', 'tip3p.xml') if ff is None else ff
  sys = ff.createSystem(openmm_top(mol), **kwargs) if hasattr(ff, 'createSystem') else ff
  for ii,f in enumerate(sys.getForces()):
    f.setForceGroup(ii)
  if inactive is not None:
    openmm_make_inactive(mol, sys, inactive)
  if charges is not None:
    nbforce = next(force for force in sys.getForces() if type(force) == openmm.NonbondedForce)
    for ii,q in charges:
      _, sigma, epsilon = nbforce.getParticleParameters(ii)
      nbforce.setParticleParameters(ii, q, sigma, epsilon)
  if epsilons is not None:
    nbforce = next(force for force in sys.getForces() if type(force) == openmm.NonbondedForce)
    for ii,eps in epsilons:
      q, sigma, _ = nbforce.getParticleParameters(ii)
      nbforce.setParticleParameters(ii, q, sigma, eps)
  # integrator required to create Context
  intgr = openmm.VerletIntegrator(1.0*UNIT.femtoseconds)  #openmm.CustomIntegrator(0)
  return openmm.Context(sys, intgr)


def openmm_EandG(mol, r=None, ff=None, ctx=None, grad=True, components=None, **kwargs): #noconnect=None,
  if ctx is None:
    ctx = openmm_EandG_context(mol, ff, **kwargs)
  else:
    assert ff is None and not kwargs, "Cannot specify ff, inactive, charges, or other context params w/ ctx"
  ctx.setPositions((getattr(mol, 'r', mol) if r is None else r)*UNIT.angstrom)
  if components is not None:
    for ii,f in enumerate(ctx.getSystem().getForces()):
      state = ctx.getState(getEnergy=True, getForces=grad, groups=set([ii]))
      components['E_' + f.__class__.__name__] = state.getPotentialEnergy()/EUNIT
      if grad:
        components['G_' + f.__class__.__name__] = -state.getForces(asNumpy=True)/(EUNIT/UNIT.angstrom)
  state = ctx.getState(getEnergy=True, getForces=grad)
  E = state.getPotentialEnergy()/EUNIT
  G = -state.getForces(asNumpy=True)/(EUNIT/UNIT.angstrom) if grad else None
  return E, G


class OpenMM_EandG:
  def __init__(self, mol, ff=None, inactive=None, charges=None, **kwargs):
    self.ctx = ff if type(ff) is openmm.Context else openmm_EandG_context(mol, ff, inactive, charges, **kwargs)
    self.inactive, self.charges = inactive, charges

  def __call__(self, mol, r=None, inactive=None, charges=None, grad=True, components=None):
    if inactive is not None or charges is not None:
      assert inactive == self.inactive and charges == self.charges, "charges and inactive cannot be changed!"
    return openmm_EandG(mol, r, ctx=self.ctx, grad=grad, components=components)


# can't extend openmm.app.ForceField due to problem with use of inspect.signature() in ArgTracker
class OpenMM_ForceField:  #(openmm.app.ForceField):
  """ Wrapper class to override default args for ForceField.createSystem() """
  def __init__(self, *args, **kwargs):
    self.sysargs = kwargs
    self.ff = args[0] if hasattr(args[0], 'createSystem') else openmm.app.ForceField(*args)  #super(OpenMM_ForceField, self).__init__(*args)

  def createSystem(self, top, **kwargs):
    return self.ff.createSystem(top, **dict(self.sysargs, **kwargs))  #super(OpenMM_ForceField, self)


# replace nonbonded force of OpenMM system with custom nonbonded force for alchemical transformation
def alch_setup(system):
  # OpenMM uses 4*\eps*( (\sigma/r)^12 - (\sigma/r)^6 ) for LJ w/ \sigma in nm and \eps in kJ/mol in
  #  NonbondedForce section of FF XML file, while Amber, Tinker use \eps*( (r0/r)^12 - 2*(r0/r)^6 ), where
  #  param file stores r0/2 in Angstrom and \eps in kcal/mol; r0 = 2^(1/6)*sigma
  nbidx, nbforce = next((ii, f) for ii, f in enumerate(system.getForces()) if type(f) == openmm.NonbondedForce)

  # CustomNonbondedForce for alchemical reaction field electrostatics and soft-core vdW
  # hard cutoff for electrostatics is a disaster since energy is highly dependent on cutoff (note that NoCutoff
  #  just turns off PBC for nonbonded forces), so we need to taper to zero at cutoff, and reaction field approx.
  #  (treating space beyond cutoff as uniform dielectric) used in NonbondedForce is a physically reasonable way
  # Trying to keep NonbondedForce for solvent-solvent interaction turned into a mess (might be better now that
  #  we use reaction field in our custom force), so we remove from system and replace with:
  # - alchem (CustomNonbondedForce): all atoms, 1-2, 1-3, 1-4 excluded; lambda for solute - solvent
  # - alch14 (CustomBondForce): solute - solute 1-4 interaction; see: github.com/openmm/openmm/issues/2698
  # openmmtools uses the parameter offset feature of NonbondedForce to apply lambda to PME electrostatics
  # ref: docs.openmm.org/latest/userguide/theory.html
  eps = nbforce.getReactionFieldDielectric()
  rcut_nm = nbforce.getCutoffDistance().value_in_unit(UNIT.nanometer)
  krf = (eps - 1)/(2*eps + 1)/rcut_nm**3  # reaction field parameters
  crf = 3*eps/(2*eps + 1)/rcut_nm
  # note that variables must be defined *after* all uses (string is split on ';' and reversed)
  f_str = "Eele + Evdw; Evdw = lam_vdw*4*epsilon*x*(x-1.0);"  #x = (sigma/r)^6;
  f_str += "Eele = lam_ele*138.935456*q1q2*(1.0/r + {krf}*r^2 - {crf});".format(krf=krf, crf=crf)
  f_str += "x = 1.0/(0.5*(1.0-lam_vdw) + (r/sigma)^6); lam_ele = 1.0 - useLambda*(1.0-lambda_ele);"
  f_str += "lam_vdw = 1.0 - useLambda*(1.0-lambda_vdw); useLambda = abs(solute1 - solute2);"  #1-delta(solute1-solute2)
  f_str += "q1q2 = charge1*charge2; sigma = 0.5*(sigma1 + sigma2); epsilon = sqrt(epsilon1*epsilon2);"
  alchem = openmm.CustomNonbondedForce(f_str)
  alchem.addGlobalParameter('lambda_ele', 1.0)
  alchem.addGlobalParameter('lambda_vdw', 1.0)
  alchem.addPerParticleParameter('solute')
  alchem.addPerParticleParameter('charge')
  alchem.addPerParticleParameter('sigma')
  alchem.addPerParticleParameter('epsilon')
  for ii in range(system.getNumParticles()):
    charge, sigma, epsilon = nbforce.getParticleParameters(ii)
    alchem.addParticle([1.0 if ii in ligatoms else 0.0, charge, sigma, epsilon])

  # Bonded force for 1-4 interactions
  f14_str = "138.935456*chargeprod/r + 4*epsilon*x*(x - 1.0); x = (sigma/r)^6;"
  alch14 = openmm.CustomBondForce(f14_str)
  alch14.addPerBondParameter('chargeprod')
  alch14.addPerBondParameter('sigma')
  alch14.addPerBondParameter('epsilon')
  for ii in range(nbforce.getNumExceptions()):
    # note that these params already include 1-4 scale factors
    a1, a2, chargeprod, sigma, epsilon = nbforce.getExceptionParameters(ii)
    alchem.addExclusion(a1, a2)
    if chargeprod != 0 or epsilon != 0:
      alch14.addBond(a1, a2, [chargeprod, sigma, epsilon])

  alchem.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)  #nbforce.getNonbondedMethod()
  alchem.setCutoffDistance(rcut_nm)  #nbforce.getCutoffDistance()
  alchem.setSwitchingDistance(nbforce.getSwitchingDistance())
  alchem.setUseSwitchingFunction(nbforce.getUseSwitchingFunction())
  alchem.setUseLongRangeCorrection(False)  #???
  #nbforce.setUseDispersionCorrection(False)  # ... otherwise I think we'd need separate custom force for vdW
  #nbforce.setForceGroup(1); alchem.setForceGroup(2); alch14.setForceGroup(3)
  system.removeForce(nbidx)
  system.addForce(alchem)
  system.addForce(alch14)
