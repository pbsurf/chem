import numpy as np
from .basics import *
from .molecule import *


## MM potentials
# original ref is QMMM.qm_EandG() - use this fn there instead (in which case maybe move this somewhere else)
# also see: github.com/dspoel/Toy-MD

# 1/(4*pi*eps0) == 1 in atomic units; use these only if trying to match AMBER or openmm numbers exactly for some reason
AMBER_qqkscale = 332.0522173/(ANGSTROM_PER_BOHR*KCALMOL_PER_HARTREE)  # 0.999965380578321
OPENMM_qqkscale = 332.06371335990205/(ANGSTROM_PER_BOHR*KCALMOL_PER_HARTREE)  # 1.0000000006209415

# hard cutoff should never actually be used!
def coulomb(rq, mq, mask=None, mask14=None, scale14=1.0, kscale=1.0, pbcbox=None, cutoff=None, hess=False, pairsout=None):
  """ return charge-charge energy, gradient, and Hessian (if hess=True) for charges `mq` at positions `rq`,
    excluding interaction between pairs listed in `mask`, and scaling interaction between pairs in `mask14`
    by `scale14`; coulomb constant is scaled by `kscale` (to accommodate force fields using different values)
  """
  Ke = kscale*ANGSTROM_PER_BOHR  # Coulomb constant in our units
  dr = rq[:,None,:] - rq[None,:,:]
  if pbcbox is not None:
    dr = dr - np.round(dr/pbcbox)*pbcbox  # != np.fmod(dr, pbcbox)
  dd = np.sqrt(np.sum(dr*dr, axis=2))
  if cutoff:
    dd[dd > cutoff] = np.inf
  if mask is not None:
    dd[mask[:,0], mask[:,1]] = np.inf  # mask.T does not work
    dd[mask[:,1], mask[:,0]] = np.inf
  np.fill_diagonal(dd, np.inf)
  invdd = 1/dd
  qq = mq[:,None]*mq[None,:]  # broadcasting seems to be faster than np.outer()
  # 1-4 scale factor for atoms separated by 3 bonds
  if mask14 is not None and scale14 != 1.0:
    qq[mask14[:,0], mask14[:,1]] *= scale14
    qq[mask14[:,1], mask14[:,0]] *= scale14
  # E and G ... you had the wrong sign here!!!
  Eqq = 0.5*Ke*np.sum(qq*invdd)
  Gqq = -Ke*np.sum((qq*(invdd**3))[:,:,None]*dr, axis=1)
  if pairsout is not None:
    pairsout['Eqq'], pairsout['Gqq'] = Ke*qq*invdd, -Ke*(qq*(invdd**3))[:,:,None]*dr
  if not hess:
    return Eqq, Gqq
  # Hessian
  ddr = np.einsum('ijk,ijl->ijkl', dr, dr)
  e4 = np.zeros_like(ddr)
  np.einsum('ijkk->ijk', e4)[:] = 1.0
  # off-diagonal components of Hessian
  Hij = (qq*(invdd**3))[:,:,None,None]*e4 + (-qq*(3*invdd**5))[:,:,None,None]*ddr  # Nx3 x Nx3
  # diagonal of Hessian
  np.einsum('iijk->ijk', Hij)[:] = -np.sum(Hij, axis=1)  # generally true for pairwise interaction (?)
  return Eqq, Gqq, Hij*Ke


# see QMMM.qm_EandG()
def lj_EandG(r, r0, depth, mask=None, mask14=None, scale14=1.0, pbcbox=None, cutoff=None, repel=None, hess=False, pairsout=None):
  """ return Lennard-Jones 6-12 van der Waal energy, gradient, and Hessian (if hess=True) for atoms at
    positions `r` with LJ parameters `r0` (equilib dist) and `depth` (\eps_0), excluding interaction between
    pairs listed in `mask`, and scaling interaction between pairs in `mask14` by `scale14`; only repulsive
    r**-12 term will be included for pairs in `repel`
  """
  dr = r[:,None,:] - r[None,:,:]
  if pbcbox is not None:
    dr = dr - np.round(dr/pbcbox)*pbcbox  # != np.fmod(dr, pbcbox)
  dr2 = np.sum(dr*dr, axis=2)
  #dr[mask] = 0  # for grad ... shouldn't be necessary since elements of dr2 are zeroed
  if cutoff:
    dr2[dr2 > cutoff*cutoff] = np.inf
  if mask is not None:
    dr2[mask[:,0], mask[:,1]] = np.inf  # mask.T does not work
    dr2[mask[:,1], mask[:,0]] = np.inf
  np.fill_diagonal(dr2, np.inf)
  idr2 = 1.0/dr2
  # Lorentz-Berthelot combining rules - en.wikipedia.org/wiki/Combining_rules
  r0 = 0.5*(r0[:,None] + r0[None,:]) if np.size(r0) > 1 else r0
  depth = np.sqrt(depth[:,None]*depth[None,:]) if np.size(depth) > 1 else depth
  # 1-4 scale factor for atoms separated by 3 bonds
  if mask14 is not None and scale14 != 1.0:
    depth[mask14[:,0], mask14[:,1]] *= scale14
    depth[mask14[:,1], mask14[:,0]] *= scale14
  # option to turn off attractive component of interaction
  repelmask = 1.0
  if repel is not None:
    repelmask = np.ones_like(idr2)
    repelmask[repel[:,0], repel[:,1]] = 0
    repelmask[repel[:,1], repel[:,0]] = 0
  # E and G
  lj2 = (r0*r0)*idr2
  lj6 = lj2*lj2*lj2
  Elj = 0.5*np.sum(depth*(lj6*lj6 - 2*repelmask*lj6))
  Glj = -12.0*np.sum((depth*(lj6*lj6 - repelmask*lj6)*idr2)[:,:,None]*dr, axis=1)
  #-12.0*np.einsum('ij,ijk->ik', depth*(lj6*lj6 - repelmask*lj6)*idr2, dr)
  if pairsout is not None:
    pairsout['Evdw'] = depth*(lj6*lj6 - 2*repelmask*lj6)
    pairsout['Gvdw'] = -12.0*(depth*(lj6*lj6 - repelmask*lj6)*idr2)[:,:,None]*dr
  if not hess:
    return Elj, Glj
  # Hessian
  assert repel is None, "repel not yet supported w/ Hessian"  # should be an easy fix though
  ddr = np.einsum('ijk,ijl->ijkl', dr, dr)
  e4 = np.zeros_like(ddr)
  np.einsum('ijkk->ijk', e4)[:] = 1.0
  # off-diagonal components of Hessian
  Hij = -12*( (depth*(14*lj6*lj6 - 8*lj6)*idr2*idr2)[:,:,None,None]*ddr
      - (depth*(lj6*lj6 - lj6)*idr2)[:,:,None,None]*e4 )
  # diagonal of Hessian
  np.einsum('iijk->ijk', Hij)[:] = -np.sum(Hij, axis=1)
  return Elj, Glj, Hij


# LJ potential (as callable class returning E and G) for use with QMMM
class MolLJ:
  def __init__(self, mol, r0=None, depth=None, repel=None, rigid=False, scale14=1/2.0, cutoff=None):
    """ create energy and gradient calculator for Lennard-Jones interactions in mol; repel is list of atom
      pairs for which only repulsive term of potential should be included; scale14 is scaling value
      for 1-4 interaction (default = 1/2 for AMBER) """
    #r0 = np.array([ELEMENTS[a.znuc].vdw_radius for a in theo_ch2o.atoms])
    self.r0 = np.array([a.lj_r0 for a in mol.atoms]) if r0 is None else r0
    self.depth = np.array([a.lj_eps for a in mol.atoms]) if depth is None else depth
    self.repel = np.asarray(repel) if repel is not None else None
    bonds, angles, diheds = mol.get_internals()
    self.maskidx = np.array([ [ii,jj] for ii in mol.listatoms() for jj in mol.get_connected(ii) ]) if rigid \
        else np.array(bonds + [(a[0], a[2]) for a in angles])  #+ [(ii, ii) for ii in range(mol.natoms)])
    self.mask14 = np.array([(a[0], a[3]) for a in diheds])
    self.scale14 = scale14
    self.pbcbox = mol.pbcbox
    self.cutoff = cutoff

  def __call__(self, mol, r=None, hess=False):
    r = getattr(mol, 'r', mol) if r is None else r
    pbcbox = getattr(mol, 'pbcbox', self.pbcbox)
    return lj_EandG(r, self.r0, self.depth, mask=self.maskidx, mask14=self.mask14, scale14=self.scale14,
        repel=self.repel, pbcbox=pbcbox, cutoff=self.cutoff, hess=hess)


# non-contact (Coulomb and/or vdW) potential for use with QMMM
class NCMM:
  def __init__(self, mol, qq=True, lj=True, qqscale14=1/1.2, ljscale14=1/2.0, qqkscale=1.0, cutoff=None):
    """ create energy and gradient calculator for Coulomb (unless qq=False) and Lennard-Jones (unless
      lj=False) interactions in mol; qqscale14 and ljscale14 are scaling values for 1-4 interaction (w/
      defaults appropriate for AMBER), and qqkscale is scale factor for Coulomb constant (see above)
    """
    self.qq, self.lj = qq, lj
    self.mmq = mol.mmq if qq else []
    self.r0, self.depth = (mol.lj_r0, mol.lj_eps) if lj else ([], [])
    bonds, angles, diheds = mol.get_internals()
    self.mask = np.array(bonds + [(a[0], a[2]) for a in angles])
    self.mask14 = np.array([(a[0], a[3]) for a in diheds])
    self.qqscale14, self.ljscale14, self.qqkscale = qqscale14, ljscale14, qqkscale
    self.pbcbox, self.cutoff = mol.pbcbox, cutoff

  def __call__(self, mol, r=None, inactive=None, charges=None, components=None):
    r = getattr(mol, 'r', mol) if r is None else r  # handle (mol), (mol, r), and (r)
    pbcbox = getattr(mol, 'pbcbox', self.pbcbox)
    q = self.mmq if charges is None else np.array(self.mmq)  # copy iff we have to make changes
    if charges is not None:
      for rq in charges:
        q[rq[0]] = rq[1]
    mask = self.mask if inactive is None else np.vstack(
        (self.mask, [(ii,jj) for ii in inactive for jj in inactive if jj > ii]) )
    Eqq, Gqq = coulomb(r, q, mask=mask, mask14=self.mask14, scale14=self.qqscale14,
        kscale=self.qqkscale, pbcbox=pbcbox, cutoff=self.cutoff, pairsout=components) if self.qq else (0,0)
    Elj, Glj = lj_EandG(r, self.r0, self.depth, mask=mask, mask14=self.mask14,
        scale14=self.ljscale14, pbcbox=pbcbox, cutoff=self.cutoff, pairsout=components) if self.lj else (0,0)
    return Eqq + Elj, Gqq + Glj


def mmbonded(mol, r=None, inactive=None, components=None):
  """ return stretch, bend, torsion, and improper torsion components of MM energy and gradient for mol """
  r = mol.r if r is None else r
  inactive = frozenset(inactive) if inactive is not None else None
  Ebond, Eangle, Etors, Eimptor = 0,0,0,0
  G = np.zeros_like(r)
  if components is not None:
    components.update(Ebond=[], Eangle=[], Etors=[], Eimptor=[])

  # mm_bonds: [ ([atom1, atom2], spring constant, equilb length), ... ]
  for b in mol.mm_stretch:
    if not (inactive and all(a in inactive for a in b[0])):
      d, gd = calc_dist(r[b[0]], grad=True)
      eb = b[1]*(d - b[2])**2
      if components is not None: components['Ebond'].append(eb)
      Ebond += eb
      G[b[0]] += 2*b[1]*(d - b[2])*gd

  # mm_angles: [ ([atom1, atom2, atom3], spring constant, equilb angle), ... ]
  for b in mol.mm_bend:
    if not (inactive and all(a in inactive for a in b[0])):
      a, ga = calc_angle(r[b[0]], grad=True)
      eb = b[1]*(a - b[2])**2
      if components is not None: components['Eangle'].append(eb)
      Eangle += eb
      G[b[0]] += 2*b[1]*(a - b[2])*ga

  # mm_torsions: [ ([atom1, atom2, atom3, atom4], [(amplitude, phase, periodicity), ...]), ... ]
  for b in mol.mm_torsion:
    if not (inactive and all(a in inactive for a in b[0])):
      a, ga = calc_dihedral(r[b[0]], grad=True)
      eb = sum(c[0]*(1 + np.cos(c[2]*a - c[1])) for c in b[1])
      if components is not None: components['Etors'].append(eb)
      Etors += eb
      for c in b[1]:
        G[b[0]] += -c[0]*np.sin(c[2]*a - c[1])*c[2]*ga

  # mm_imptors: [ ([atom1, atom2, atom3, atom4], [(amplitude, phase, periodicity), ...]), ... ]
  for b in mol.mm_imptor:
    if not (inactive and all(a in inactive for a in b[0])):
      a, ga = calc_dihedral(r[b[0]], grad=True)
      eb = sum(c[0]*(1 + np.cos(c[2]*a - c[1])) for c in b[1])
      if components is not None: components['Eimptor'].append(eb)
      Eimptor += eb
      for c in b[1]:
        G[b[0]] += -c[0]*np.sin(c[2]*a - c[1])*c[2]*ga

  return Ebond + Eangle + Etors + Eimptor, G


# To make this less "simple" we'd need to support PBC w/ Ewald (PME) sum for Coulomb force
# Possible refs: github.com/SimonTreu/particlesim , github.com/chemlab/chemlab  (ewald*.py in both)
# mmbonded() is slow
class SimpleMM:
  def __init__(self, mol, ncmm=None):
    self.ncmm = ncmm if ncmm is not None else NCMM(mol)
    self.mol = mol

  def __call__(self, mol=None, r=None, inactive=None, charges=None, components=None):
    if hasattr(mol, 'atoms'):  #r is not None:
      assert id(mol) == id(self.mol), "SimpleMM can only be used with self.mol"
    r = mol if r is None else r
    Enc, Gnc = self.ncmm(self.mol, r, inactive, charges, components)
    Ebd, Gbd = mmbonded(self.mol, r, inactive, components)
    return Enc + Ebd, Gnc + Gbd


def test_MM(mol, r0=None, key=None, niter=100, printcomp=True):
  """ randomly translate atoms and compare energy from Tinker and our SimpleMM """
  from chem.io.tinker import tinker_EandG
  simple_EandG = SimpleMM(mol)
  r0 = r0 if r0 is not None else mol.r
  r = r0
  tinkercomp, simplecomp = {}, {}
  for ii in range(niter):
    t0 = time.clock()
    E0, G0 = tinker_EandG(mol, r, key=key, grad=False, components=tinkercomp)
    t1 = time.clock()
    E1, G1 = simple_EandG(mol, r, components=simplecomp)
    t2 = time.clock()
    print("Tinker E (%f s): %.9f, Simple MM E (%f s): %.9f; diff: %.12f" % (t1-t0, E0, t2-t1, E1, E1-E0))
    if printcomp:
      print("Tinker components: %r" % tinkercomp)
      print("Simple MM components: %r\n" % simplecomp)
    r = r0 + 0.1*np.random.random(np.shape(r0))


class SimpleMD:
  def __init__(self, EandG, m, r0, kB, T0=300, v0=None, dt=0.2, therm_steps=100):
    """ Classical dynamics simulation (i.e. molecular dynamics) of system w/ energy and gradient fn EandG,
      masses m, initial positions r0, initial velocities v0 (initialized according to Maxwell-Boltzmann at
      temperature T0 if omitted), with time step dt and thermostat time constant therm_steps*dt (or no
      thermostat if !therm_steps).  Units are set via kB - because our usual units of Hartree and Angstrom
      would require time in units of sqrt(Hartree/amu/Ang^2) ~ 1.95fs, we require kB be passed to make clear
      what units are in use.  One choice, w/ kB in kJ/mol/K, is: mass (m): amu (Dalton); time (dt): ps;
      distance (r0): nm; velocity (v0): nm/ps, energy (EandG): kJ/mol = amu*nm^2/ps^2.
    """
    self.EandG = EandG
    self.m = m
    self.r = r0
    self.kB = kB
    self.T0 = T0
    # Maxwell-Boltzmann distribution
    self.v = np.random.normal(0, np.sqrt(kB*T0/m), size=r0.shape) if v0 is None else v0
    self.a = None
    self.dt = dt
    self.therm_steps = therm_steps
    self.dEbath = 0
    self.Ndim = np.size(r0)
    self.Rs = []


  # refs:
  # - github.com/molmod/yaff/blob/master/yaff/sampling/nvt.py
  # - arxiv.org/pdf/0803.4397.pdf and arxiv.org/pdf/0803.4060.pdf
  # - github.com/lammps/lammps/blob/master/tools/i-pi/ipi/engine/thermostats.py
  # - github.com/whitead/Simple-MD
  def csvr_thermostat(self, v):
    """ return velocities adjusted by CSVR (canonical stochastic velocity rescaling?) thermostat (Bussi and
      Parrinello) towards target temperature self.T0
    """
    c = np.exp(-1.0/self.therm_steps) #np.exp(-dt/tau)
    R = np.random.normal(0, 1)
    S = np.sum(np.random.normal(0, 1, self.Ndim - 1)**2)
    KE = 0.5*np.sum(self.m*(v*v))  # instantaneous temperature is T = KE/(0.5*kB*Ndim)
    KEavg = 0.5*self.Ndim*self.kB*self.T0
    rate = (1-c)*KEavg/self.Ndim/KE  # KEavg/KE == T0/T
    alpha = np.sign(R + np.sqrt(c/rate))*np.sqrt(c + (S + R**2)*rate + 2*R*np.sqrt(c*rate))
    self.dEbath += (1 - alpha**2)*KE  # energy exchanged w/ bath
    return alpha*v


  # does not give proper canonical distribution, and can cause "flying ice cube" effect (energy shifted to
  #  low freq. modes) - CSVR should be used instead unless fully deterministic simulation is needed
  def berendsen_thermostat(self, v):
    """ return velocities adjusted by Berendsen thermostat: dT/dt = (T0 - T)/tau w/ tau = therm_steps*dt """
    KE = 0.5*np.sum(self.m*(v*v))
    T = KE/(0.5*self.kB*self.Ndim)  # instantaneous temperature
    c = np.sqrt(1.0 + 1.0/self.therm_steps*(self.T0/T - 1))
    return c*v


  def run(self, nsteps=1000, nsave=100, mon=None):
    """ integrate equation of motion (dv/dt = F/m) using velocity Verlet algorithm for nsteps, saving
      coordinates every nsave steps and calling mon function if specified; returns saved coordinates
    """
    r, v, a, dt = self.r, self.v, self.a, self.dt  # avoid lots of "self."
    self.Rs = []  #np.zeros((int(nsteps/nsave),) + np.shape(r)) if nsave else None
    self.step = 0
    if a is None:
      E,G = self.EandG(r)  # potential energy
      a = -G/self.m

    for ii in range(1, nsteps+1):
      self.step = ii
      # state: r(t), v(t)
      r = r + v*dt + (0.5*dt*dt)*a
      # state: r(t + dt), v(t)
      E,G = self.EandG(r)  # potential energy
      anew = -G/self.m
      v = v + 0.5*dt*(a + anew)
      a = anew
      # state: r(t + dt), v(t + dt)
      if self.therm_steps:
        v = self.csvr_thermostat(v)

      if nsave and ii % nsave == 0:
        self.Rs.append(r)  #self.Rs[int(ii/nsave)-1] = r
        if mon:
          mon(self, r, v, E)

    self.r, self.v, self.a = r, v, a
    return self.Rs


# Beeman's method
#r = r + v*dt + (4*a - aprev)*dt*dt/6
#v = v + (2*anew + 5*a - aprev)*dt/6


# atomic number -> value dicts for GBSA parameters
GBSA_RADIUS = { 6: 1.7, 7: 1.55, 8: 1.5, 9: 1.5, 14: 2.1, 15: 1.85, 16: 1.8, 17: 1.7 }  # Ang
GBSA_SCREEN = { 1: 0.85, 6: 0.72, 7: 0.79, 8: 0.85, 9: 0.88, 15: 0.86, 16: 0.96 }  # unitless

# GBSA implicit solvation energy - basically cut and paste of GBSAOBC1Force from openmm customgbforces.py
# only energy, no derivative - so only useful for single point calcs for now
def gbsa_E(mol, r=None, eps_solu=1.0, eps_solv=78.5, pairsout=None):
  """ return GBSA solvation energy for `mol` (currently using AMBER/OpenMM OBC1 method) """
  # H: 1.2 (1.3 if bonded to N); all other elements, use lookup table (or default of 1.5)
  roffset = 0.009  # nm
  or1 = np.array([ 0.1*((1.3 if mol.atoms[a.mmconnect[0]].znuc == 7 else 1.2)
      if a.znuc == 1 else GBSA_RADIUS.get(a.znuc, 1.5)) - roffset for a in mol.atoms ])  # note conversion to nm!
  # scaled radii
  sr2 = np.array([ GBSA_SCREEN.get(a.znuc, 0.8) for a in mol.atoms ]) * or1

  mmq = mol.mmq
  rq = (mol.r if r is None else r)*0.1  # convert to nm
  dr = rq[:,None,:] - rq[None,:,:]
  r = np.sqrt(np.sum(dr*dr, axis=2))
  np.fill_diagonal(r, 1.0)  # 0, np.inf, and np.nan are all problematic
  # HCT analytical approx for \int_{vdW} r^-4 d^3r - 10.1021/jp961710n (1996)
  L = np.fmax(np.abs(r - sr2), or1[:,None])
  U = r + sr2
  Iij = (r+sr2-or1 > 0) * 0.5*(1/L - 1/U + 0.25*(r - sr2**2/r)*(1/(U**2) - 1/(L**2)) + 0.5*np.log(L/U)/r)
  np.fill_diagonal(Iij, 0)
  I = np.sum(Iij, axis=1)
  # OBC formula to get generalized Born radii - 10.1002/prot.20033 (2004)
  or1ex = or1 + roffset
  #B = 1/(1/or1 - I)  # HCT (Amber igb=1)
  psi = I*or1
  B = 1/(1/or1 - np.tanh(0.8*psi + 2.909125*psi**3)/or1ex)  # OBC1 (Amber igb=2)
  #B = 1/(1/or1 - np.tanh(1.0*psi - 0.8*psi**2 + 4.85*psi**3)/or1ex)  # OBC2 (Amber igb=5)
  # SA energy
  Esa = (28.3919551 * (or1ex + 0.14)**2 * (or1ex/B)**6)/KJMOL_PER_HARTREE  # ACE SA
  # GB energy
  BB = B[:,None]*B[None,:]
  qq = mmq[:,None]*mmq[None,:]
  np.fill_diagonal(r, 0)  # this lets Egb expr. work for diagonal terms as well
  fgb = np.sqrt(r**2 + BB*np.exp(-r**2/(4*BB)))
  Egb = 0.5*np.sum(-138.935485*(1/eps_solu - 1/eps_solv)*qq/fgb, axis=1)/KJMOL_PER_HARTREE
  if pairsout is not None:
    pairsout['Egb'], pairsout['Esa'] = Egb, Esa
  return np.sum(Egb + Esa)  #0.5*Egb + Esa)/KJMOL_PER_HARTREE  # Egb, Esa


## untested

# idea for coulomb_atom(), lj_atom() is to quickly update energy after a single-atom Monte Carlo move
# ... but they don't support mask14 yet!

def coulomb_atom(idx, rq, mq, mask=None):
  """ coulomb energy and gradient for interaction between atom `idx` and all other atoms """
  dr = rq - rq[idx]
  dd = np.sqrt(np.sum(dr*dr, axis=-1))
  if mask is not None:
    dd[mask] = np.inf
  dd[idx] = np.inf
  invdd = 1.0/dd
  Eqq = 0.5*ANGSTROM_PER_BOHR*np.sum(mq*invdd)
  Gqq = -ANGSTROM_PER_BOHR*np.sum((mq*(invdd**3))[:,None]*dr, axis=1)
  return Eqq, Gqq


def lj_atom(idx, r, r0, depth):
  """ Lennard-Jones energy and gradient for interaction between atom `idx` and all other atoms """
  dr = r - r[idx]
  dr2 = np.sum(dr*dr, axis=-1)
  if mask is not None:
    dr2[mask] = np.inf
  dr2[idx] = np.inf
  idr2 = 1.0/dr2
  # Lorentz-Berthelot combining rules
  r0 = 0.5*(r0[idx] + r0) if np.size(r0) > 1 else r0
  depth = np.sqrt(depth[idx]*depth) if np.size(depth) > 1 else depth
  # E and G
  lj2 = (r0*r0)*idr2
  lj6 = lj2*lj2*lj2
  Elj = 0.5*np.sum(depth*(lj6*lj6 - 2*lj6))
  Glj = -12.0*np.sum((depth*(lj6*lj6 - lj6)*idr2)[:,None]*dr, axis=1)
  return Elj, Glj


## move this stuff to mm/amber.py
_tleap_in = """source leaprc.gaff
mods = loadAmberParams mol.frcmod
mol = loadMol2 mol.mol2
saveAmberParm mol mol.prmtop mol.inpcrd
quit"""

#from openmm.app.amberprmtopfile import AmberPrmtopFile
#AmberPrmtopFile("{0}.prmtop".format(res)).createSystem()

# refs:
# - AmberTools manual
# - http://ambermd.org/tutorials/basic/tutorial4b/
# - https://docs.bioexcel.eu/2020_06_09_online_ambertools4cp2k/
# - https://github.com/ParmEd/ParmEd/issues/1109
def antechamber_prepare(mol, res, netcharge=0):
  import parmed
  from .io import write_pdb
  dir = '{0}_amber'.format(res)
  os.mkdir(dir)
  os.chdir(dir)
  write_pdb(mol, 'mol.pdb')
  os.system("antechamber -i mol.pdb -fi pdb -o mol.mol2 -fo mol2 -c bcc -nc %d -s 2" % netcharge)
  os.system("parmchk2 -i mol.mol2 -f mol2 -o mol.frcmod")
  write_file("mol.tleap.in", _tleap_in)
  os.system("tleap -f mol.tleap.in")
  # now load with parmed to convert to OpenMM XML
  amber = parmed.load_file('mol.prmtop', 'mol.inpcrd')
  # unique_atom_types needed to prevent openmm errors when loading multiple force field files
  omm = parmed.openmm.OpenMMParameterSet.from_parameterset(
      parmed.amber.AmberParameterSet.from_structure(amber), unique_atom_types=True)
  #omm = parmed.openmm.OpenMMParameterSet.from_structure(amber)
  omm.residues.update(parmed.modeller.ResidueTemplateContainer.from_structure(amber).to_library())
  os.chdir('..')
  omm.write('%s.ff.xml' % res)
  amber.save('%s.mol2' % res)  #  seems mol2 is the only format for which parmed will include bonds
  return amber


# assign simple gaff types to allow for basic relaxation (e.g., as starting point for QM geom optim)
# try Tinker basic.prm force field?
def gaff_trivial(mol):
  for a in mol.atoms:
    conn = [mol.atoms[jj].znuc for jj in a.mmconnect]
    if a.znuc == 1:
      a1 = mol.atoms[a.mmconnect[0]]
      if a1.znuc == 6:
        elewd = sum(mol.atoms[jj].znuc in [7,8,9,17,35] for jj in a1.mmconnect)
        a.mmtype = ['hc','h1','h2','h3'][elewd]
      else:
        a.mmtype = {7:'hn', 8:'ho'}[a1.znuc]
    elif a.znuc == 6: a.mmtype = [None,'c1','c2','c3'][len(conn)-1]
    elif a.znuc == 7: a.mmtype = ['n1','n2','n3','n4'][len(conn)-1]
    elif a.znuc == 8: a.mmtype = 'o' if len(conn) == 1 else 'oh' if 1 in conn else 'os'


# AMBER parameter (dat) file notes:
# - multiple torsion terms get separate lines (negative value of periodicity for all but the last)
# - ref: https://github.com/ParmEd/ParmEd/blob/master/parmed/amber/parameters.py
# previously, we messed w/ the case of atom names here ... instead, mess with atom name/mmtype as needed

def _download_amber_data():
  # ff99SB.xml differs from amber99sb.xml in that charges are set in residue defs instead of defining a unique
  #  atom type for every atom, so it more closely matches structure of AMBER param files
  #os.system("wget https://github.com/openmm/openmmforcefields/blob/master/amber/ffxml/ff99SB.xml")
  os.chdir(DATA_PATH)
  os.system("wget https://github.com/openmm/openmmforcefields/tree/master/amber/gaff/ffxml/gaff.xml")
  os.system("mkdir amber")
  os.chdir(DATA_PATH + "/amber")
  os.system("wget https://github.com/choderalab/ambermini/raw/master/share/amber/dat/reslib/leap/all_amino94.lib")
  os.system("wget https://github.com/choderalab/ambermini/raw/master/share/amber/dat/reslib/leap/all_aminont94.lib")
  os.system("wget https://github.com/choderalab/ambermini/raw/master/share/amber/dat/reslib/leap/all_aminoct94.lib")
  os.system("wget https://github.com/choderalab/ambermini/raw/master/share/amber/dat/leap/parm/parm99.dat")
  os.system("wget https://github.com/choderalab/ambermini/raw/master/share/amber/dat/leap/parm/frcmod.ff99SB")
  os.system("wget https://github.com/choderalab/ambermini/raw/master/share/amber/dat/leap/parm/gaff.dat")
  # or https://github.com/openmm/openmmforcefields/raw/master/amber/gaff/dat/gaff-1.81.dat


def load_amber_dat(filename):
  """ load AMBER parameter *.dat file """
  smash = lambda s: s.replace(' ', '')
  parm = Bunch(stretch={}, bend={}, torsion={}, imptor={}, vdw={})
  vdw_equiv = {}
  with open(filename, 'r') as file:
    for line in file:
      if not line.strip(): break  # skip atom list
    next(file, '')  # skip list of hydrophobic atoms
    for line in file:
      a = line[5:].split()
      if not a: break
      parm.stretch[smash(line[:5])] = a[:2]  # kcal/mol/Ang^2 ; Ang
    for line in file:
      a = line[8:].split()
      if not a: break
      parm.bend[smash(line[:8])] = a[:2]  # kcal/mol/rad^2 ; degrees
    for line in file:
      a = line[11:].split()
      if not a: break  # divisor (int) ; kcal/mol ; phase (deg) ; multiplicity (int)
      parm.torsion.setdefault(smash(line[:11]), []).append(a[:4])
    for line in file:
      a = line[11:].split()
      if not a: break
      parm.imptor[smash(line[:11])] = a[:3]
    for line in file:
      if not line.strip(): break  # skip 10-12 terms
    for line in file:
      a = line.split()
      if not a: break
      vdw_equiv[a[0]] = a[1:]
    next(file, '')  # skip "MOD4 ... RE"
    for line in file:
      a = line.split()
      if not a: break
      parm.vdw[a[0]] = a[1:3]  # Ang ; kcal/mol
      for eqv in vdw_equiv.get(a[0], []):
        parm.vdw[eqv] = a[1:3]

  return parm


def load_amber_frcmod(filename):
  """ load AMBER parameter frcmod.* file """
  smash = lambda s: s.replace(' ', '')
  parm = Bunch(stretch={}, bend={}, torsion={}, imptor={}, vdw={})
  with open(filename, 'r') as file:
    for line in file:
      if line.startswith('MASS'): break
    for line in file:
      if not line.strip(): break  # skip atom list
    line = next(file, '')
    if line.startswith('BOND'):
      for line in file:
        a = line[5:].split()
        if not a: break
        parm.stretch[smash(line[:5])] = a[:2]
      line = next(file, '')
    if line.startswith('ANGL'):
      for line in file:
        a = line[8:].split()
        if not a: break
        parm.bend[smash(line[:8])] = a[:2]
      line = next(file, '')
    if line.startswith('DIHE'):
      for line in file:
        a = line[11:].split()
        if not a: break
        parm.torsion.setdefault(smash(line[:11]), []).append(a[:4])
      line = next(file, '')
    if line.startswith('IMPR'):
      for line in file:
        a = line[11:].split()
        if not a: break
        parm.imptor[smash(line[:11])] = a[:3]
      line = next(file, '')
    if line.startswith('NONB'):
      for line in file:
        a = line.split()
        if not a: break
        parm.vdw[a[0]] = a[1:3]

  return parm


def load_amber_reslib(*files):
  """ load residue data from AMBER reslib files """
  reslib = {}
  currres = None
  for filename in files:
    with open(filename, 'r') as file:
      for line in file:
        if line.startswith("!entry."):
          l = line.split('.', 2)
          if l[2].startswith("unit.atoms table"):
            currres = reslib.setdefault(l[1], {})
          else:
            currres = None
        elif currres is not None:
          l = line.split()
          currres[l[0][1:-1]] = ( l[1][1:-1], float(l[-1]) )  # name, atom type, charge
  return reslib


def residue_terminal(mol, resnum):
  """ return 'N' if residue `resnum` is N-terminal, 'C' if C-terminal, 'NC' if lone, '' otherwise """
  res = mol.residues[resnum]
  if res.name not in PDB_PROTEIN:
    return ''
  term = ''
  prevres = mol.residues[resnum-1] if resnum > 0 else None
  if prevres is None or prevres.chain != res.chain or prevres.name not in PDB_PROTEIN:
    term += 'N'
  nextres = mol.residues[resnum+1] if resnum+1 < len(mol.residues) else None
  if nextres is None or nextres.chain != res.chain or nextres.name not in PDB_PROTEIN:
    term += 'C'
  return term


# we no longer assign numeric mmtypes for Tinker ... as easy as mol.mmtype = np.arange(mol.natoms) + mmtype0
#  - mmtype0 must be <1000 (Tinker has maxclass=1000 by default) and, for AMBER, >648
def set_mm_params(mol, parm, reslib=None, allowmissing=False):
  """ load MM params for a molecule `mol` from Amber parameters `parm` """
  UNIT = 1.0/KCALMOL_PER_HARTREE
  mol.mm_stretch, mol.mm_bend, mol.mm_torsion, mol.mm_imptor = [], [], [], []
  bonds, angles, diheds = mol.get_internals()
  # charge and mmtype from reslib
  if reslib is not None:
    for ii,res in enumerate(mol.residues):
      resdat = reslib[residue_terminal(mol, ii) + res.name]
      for jj in res.atoms:
        mol.atoms[jj].mmtype, mol.atoms[jj].mmq = resdat[mol.atoms[jj].name]
  mmtypes = mol.mmtype  #np.array([a.name.upper() for a in mol.atoms])
  # helper fns
  getparm = lambda p, a: p.get('-'.join(a), None) or p.get('-'.join(a[::-1]), None)
  def checkparm(p, b):
    if p is None:
      print("Missing MM parameters for {}; atoms {}".format('-'.join(mmtypes[list(b)]), b))
      assert allowmissing, "Missing parameters not permitted!"
    return p is not None
  # ---
  for b in bonds:
    p = getparm(parm.stretch, mmtypes[list(b)])
    if not checkparm(p, b): continue
    mol.mm_stretch.append(( list(b), float(p[0])*UNIT, float(p[1]) ))
  for b in angles:
    p = getparm(parm.bend, mmtypes[list(b)])
    if not checkparm(p, b): continue
    mol.mm_bend.append(( list(b), float(p[0])*UNIT, float(p[1])*np.pi/180 ))
  for b in diheds:
    ps = getparm(parm.torsion, mmtypes[list(b)]) or getparm(parm.torsion, ['X', mmtypes[b[1]], mmtypes[b[2]], 'X'])
    if not checkparm(ps, b): continue
    # when comparing, note that Tinker amber*.prm torsions are already divided by the p[0] value
    l = [( float(p[1])/float(p[0])*UNIT, float(p[2])*np.pi/180, abs(float(p[3])) ) for p in ps if float(p[1]) != 0.0]
    if l:
      mol.mm_torsion.append((list(b), l))
  # can't iterate over imptors from generate_internals() - need to choose the most specific term for each atom
  # ref: Tinker kimptor.f
  impparm = lambda nx, z: parm.imptor.get('-'.join(['X']*nx + [mmtypes[y] for y in z[nx:]]), None)
  for c,atom in enumerate(mol.atoms):
    if len(atom.mmconnect) != 3: continue
    a,b,d = atom.mmconnect  # 3rd atom is the central atom in Amber and Tinker convention
    w = [[a,b,c,d], [b,a,c,d], [a,d,c,b]]
    for numx in [0,1,2]:
      # imptor w/ 0<->3 swap is equiv. up to sign (and Amber imptor terms are always even)
      res = [ impparm(numx, z) or impparm(numx, [z[3], z[1], z[2], z[0]]) for z in w ]
      hits = [(z,p) for z,p in zip(w, res) if p is not None]
      for z,p in hits:
        #print("imptor = %r, abcd = %s, x = %s, nhits = %d" % (z, '-'.join(mmtypes[z]), '-'.join(['X']*numx + [mmtypes[y] for y in z[numx:]]), len(hits)))
        mol.mm_imptor.append(( list(z), [(float(p[0])*UNIT/len(hits), float(p[1])*np.pi/180, abs(float(p[2])))] ))
      if hits: break
  # vdW
  for ii,a in enumerate(mol.atoms):
    p = parm.vdw.get(mmtypes[ii], None)
    a.lj_r0 = 2.0*float(p[0])
    a.lj_eps = float(p[1])*UNIT

  return mol
