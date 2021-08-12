import simtk.openmm as openmm
import simtk.openmm.app as openmm_app
import simtk.unit as UNIT
from ..basics import *
from ..molecule import Residue


def openmm_top(mol):
  """ generate OpenMM topology from chem Molecule """
  assert len(mol.residues) > 0, "Residues required to created OpenMM topology"
  top = openmm_app.Topology()
  chains = set(res.chain for res in mol.residues)
  topchn = {chain: top.addChain(chain) for chain in chains}
  topres = [top.addResidue(res.name, topchn[res.chain]) for res in mol.residues]
  topatm = [top.addAtom(a.name,
      openmm_app.Element.getByAtomicNumber(a.znuc), topres[a.resnum or 0]) for a in mol.atoms]
  for a1,a2 in mol.get_bonds():
    top.addBond(topatm[a1], topatm[a2])
  if mol.pbcbox is not None:
    top.setUnitCellDimensions(mol.pbcbox*UNIT.angstrom)  # Angstroms to nm
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


def openmm_load_params(mol, sys, charges=True, vdw=False):  #, charges=True, vdw=False, bonded=False):
  nbforce = next(force for force in sys.getForces() if type(force) == openmm.NonbondedForce)
  for ii, atom in enumerate(mol.atoms):
    charge, sigma, eps = nbforce.getParticleParameters(ii)
    if charges:
      atom.mmq = charge.value_in_unit(UNIT.elementary_charge)
    if vdw:
      atom.lj_r0 = sigma.value_in_unit(UNIT.angstrom) * 2**(1/6.0)
      atom.lj_eps = eps.value_in_unit(UNIT.kilojoule_per_mole)/KJMOL_PER_HARTREE


# for tracking down "Particle coordinate is nan" error
def openmm_dynamic(ctx, nsteps, sampsteps=100, grad=False):
  """ Run OpenMM context `ctx` for `nsteps` and return position, energy, and, optionally, gradient recorded
    every sampsteps
  """
  sys, intgr = ctx.getSystem(), ctx.getIntegrator()
  Es, Rs, Gs = [], [], []
  try:
    for ii in range(nsteps//sampsteps):
      intgr.step(sampsteps)
      state = ctx.getState(getEnergy=True, getPositions=True, getForces=grad, enforcePeriodicBox=True)
      Es.append(state.getPotentialEnergy().value_in_unit(UNIT.kilojoule_per_mole)/KJMOL_PER_HARTREE)
      Rs.append(state.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom))
      if grad:
        Gs.append(-state.getForces(asNumpy=True).value_in_unit(UNIT.kilojoule_per_mole/UNIT.angstrom)/KJMOL_PER_HARTREE)
  except Exception as e:
    print("OpenMM exception: ", e)
  return (np.asarray(Rs), np.asarray(Es), np.asarray(Gs)) if grad else (np.asarray(Rs), np.asarray(Es))


# OpenMM doesn't support Hessian calculations; see github.com/leeping/forcebalance/blob/master/src/openmmio.py
#  for Hessian calc via finite differencing; also see github.com/Hong-Rui/Normal_Mode_Analysis

def openmm_EandG_context(mol, ff=None, inactive=None, charges=None):
  """ create an OpenMM context for single point energy and gradient calculations """
  top = openmm_top(mol)  #noconnect=noconnect
  ff = openmm_app.ForceField('amber99sb.xml', 'tip3p.xml') if ff is None else ff
  sys = ff.createSystem(top, nonbondedMethod=openmm_app.NoCutoff)
  openmm_load_params(mol, sys)  # should we do this after overriding charges?
  if inactive is not None:
    openmm_make_inactive(mol, sys, inactive)
  if charges is not None:
    nbforce = next(force for force in sys.getForces() if type(force) == openmm.NonbondedForce)
    for ii,q in charges:
      _, sigma, epsilon = nbforce.getParticleParameters(ii)
      nbforce.setParticleParameters(ii, q, sigma, epsilon)
  # integrator required to create Context
  intgr = openmm.VerletIntegrator(1.0*UNIT.femtoseconds)
  return openmm.Context(sys, intgr)


def openmm_EandG(mol, r=None, ff=None, ctx=None, inactive=None, charges=None, grad=True, components=None): #noconnect=None,
  if ctx is None:
    ctx = openmm_EandG_context(mol, ff, inactive, charges)
  else:
    assert ff is None and inactive is None and charges is None, "Cannot specify ff, inactive, or charges w/ ctx"
  ctx.setPositions((mol.r if r is None else r)*UNIT.angstrom)
  state = ctx.getState(getEnergy=True, getForces=True)
  E = state.getPotentialEnergy().value_in_unit(UNIT.kilojoule_per_mole)/KJMOL_PER_HARTREE
  gunit = UNIT.kilojoule_per_mole/UNIT.angstrom
  G = -state.getForces(asNumpy=True).value_in_unit(gunit)/KJMOL_PER_HARTREE if grad else None
  return E, G


class OpenMM_EandG:
  def __init__(self, mol, ff=None, inactive=None, charges=None):
    self.ctx = openmm_EandG_context(mol, ff, inactive, charges)
    self.inactive, self.charges = inactive, charges

  def __call__(self, mol, r=None, inactive=None, charges=None, components=None):
    if inactive is not None or charges is not None:
      assert inactive == self.inactive and charges == self.charges, "charges and inactive cannot be changed!"
    return openmm_EandG(mol, r, ctx=self.ctx, components=components)


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
