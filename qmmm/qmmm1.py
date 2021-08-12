# QMMM driver class

# TODO:
# 1. probably should have separate get_charges/update_charges as we do for caps
# 1. try Micro-it: use RESP to assign MM charges to QM atoms, then do full MM minimization between QM steps
# 1. get_charges(): shift to intermediate pos w/o dipole?
# Later still:
# 1. get energy components (from analyze?) and warn if inactive-active VDW exceeds
#  some threshold (or changes by more than a threshold over calculation)?
# 1. QM/QM microit or alternating step geom opt (to defeat nonlinear scaling)?
# 1. smooth transition from QM grad to MM grad (gradmix) seems interesting ... try to determine best single
#  point energy method based on agreement with this?
# 1. separate iteration number from file number and always include iteration number in inpnote?

# Other QM/MM codes:
# * Chemshell/PyChemshell
# * comp.chem.umn.edu/qmmm - very useful manual
# * github.com/CCQC/janus - OpenMM, Psi4; clean python code

# NOTES:
# 1. ensuring net charge of MM system is unaltered: mm_EandG option to distribute net charge of nocharge
#  atoms over nocharge atoms rather than settting their charge to zero (enabled by default). Alternative
#  would be to place net charge on one atom, the others being zeroed.
# 1. In a perfect world, we'd get rid of self.mol and just use mol passed to EandG; in reality, there is some
#  molecule specific state, e.g. cap atom parameters.  Also, since QM/MM isn't a fully automated black box,
#  passing a bunch of different molecules to EandG isn't a realistic use case

# All atoms included in QM energy should be set to inactive in TINKER - TINKER will exclude energy terms iff
#  all atoms involved are inactive; frontier MM atoms can optionally be set to inactive as well (this will
#  remove Q-M bond stretch terms, angle terms involving more than 1 QM atom, and torsions involving more than
#  2 QM atoms) ... technically, we use TINKER group/group-select to set intra-group interactions to 0 for
#  inactive atoms (see tinker.py)

import os, copy, time
import numpy as np
from ..molecule import *
from ..io.pyscf import pyscf_EandG
from ..io.tinker import tinker_EandG
from ..io.openmm import openmm_EandG, OpenMM_EandG


class QMMM:
  prefix = ""
  embed = 'elec'  # or 'mech'
  savefiles = False
  iternum = 0
  #reuselogs = False
  qm_opts = {}
  # mm_opts defaults: newcharges==True to conserve total MM charge; m1inactive==True to make M1 inactive,
  #  so e.g. M1-Q1 bond, Q2-Q1-M1 angle, etc. ignored
  mm_opts = {}
  mm_key = None
  qmatoms = None
  qm_charge = 0  # (total qmatoms Znuc - num. electrons for QM calc); not in qm_opts due to use in mm_EandG
  chargeopts = {}  # dictionary of options to be passed to get_charge
  capopts = {}
  prev_cclib = None  # previous cclib data object
  moguess = None  # array of MO coeffs to be used for guess, or 'prev' to use prev_cclib.mocoeffs
  charge_cutoff = 0.5E-6  # charges below this cutoff are not included
  components = True  # save components to Ecomp, Gcomp?
  maxgradsum = 1.0E-7  # generate warning if sum of total gradient exceeds this value (in Hartree/Ang)


  def __init__(self, mol, **kargs):
    self.set(**kargs)
    self.set_molecule(mol)


  def set(self, **kargs):
    oldvals = {}
    for key, val in kargs.items():
      # omit default so getattr() throws exception if key doesn't match existing/allowed members
      oldvals[key] = getattr(self, key)
      setattr(self, key, val)
    return oldvals


  def set_molecule(self, mol):
    self.mol = mol
    if type(self.qmatoms) is str:
      self.qmatoms = select_atoms(mol, self.qmatoms)
    self.caps = self.get_caps(mol, self.qmatoms, **self.capopts)
    mmq_qm = sum(mol.atoms[a].mmq for a in self.qmatoms)
    print("Net MM charge of QM region: %f" % mmq_qm)


  def EandG(self, mol=None, r=None, dograd=1, **kargs):
    """ single point QM/MM energy and gradient; additional arguments are used to set qmmm data members for
      duration of call to EandG() only
    """
    if mol is not None:
      if mol.__class__ == Molecule:
        assert id(mol) == id(self.mol), "QMMM.EandG() can only be used with self.mol"
      else:
        assert r is None, "Invalid Molecule object passed to QMMM.EandG()"
        r = mol
    r = self.mol.r if r is None else r
    oldvals = self.set(**kargs)
    Eqm, Emm, G, Gqm, Gmm = 0.0, 0.0, None, 0.0, 0.0
    self.Ecomp, self.Gcomp = (Bunch(), Bunch()) if self.components else (None, None)
    self.timing = {}
    t0 = time.time()
    # MM
    if len(self.qmatoms) < self.mol.natoms:
      inactive = self.qmatoms  # qmmethod == 'lmo' and self.qmEatoms or self.qmatoms
      nocharge = self.embed == 'elec' and inactive or []
      # note Gmm will always include all atoms in original order
      Emm, Gmm = self.mm_EandG(self.mol, r,
          modcharge=nocharge, inactive=inactive, grad=dograd, **self.mm_opts)
    t1 = time.time()
    # QM
    if len(self.qmatoms) > 0:
      self.charges = []
      if self.embed == 'elec' and len(self.qmatoms) < self.mol.natoms:
        self.charges = self.get_charges(self.mol, r, self.qmatoms, **self.chargeopts)
        if not self.charges:
          print("*** WARNING: electrostatic embedding specified but MM charges not set!")
      self.update_caps(r, self.caps)
      Eqm, Gqm = self.qm_EandG(self.mol, r,
          self.qmatoms, self.caps, self.charges, charge=self.qm_charge, grad=dograd, **self.qm_opts)
    E = Eqm + Emm
    # combine gradients
    if dograd:
      G = Gmm + Gqm
      # translational invariance requires sum of gradients be zero
      Gsum = np.sum(G, axis=0)
      if np.any(np.abs(Gsum) > self.maxgradsum):
        print("*** WARNING: gradient sum is too large: %s" % Gsum)  # should we just assert?
    if self.Ecomp is not None:
      self.Ecomp.update(Eqm=Eqm, Emm=Emm)
    if self.Gcomp is not None:
      self.Gcomp.update(Gqm=Gqm, Gmm=Gmm)
    if self.savefiles:  #or self.reuselogs:
      self.iternum += 1
    self.timing.update(MM=t1-t0, QM=time.time()-t1)
    # restore data members
    self.set(**oldvals)
    return (E, G) if dograd else E


  # For now, cap atom dist can be 1.09 Ang (fixed; Methane C-H) or g*(Q1 - M1 dist) (g = 0.709 in NWChem)
  # NOTE: these are only valid for carbon Q1 atoms, and preferably with carbon M1 as well
  # better options:
  # - fix dist at equilib Q-H distance from a QM geom opt (seems appropriate if frontier MM atom is active) or
  # - g = equilib Q-H dist/equilb Q-M dist (seems appropriate if frontier MM atom is inactive);
  #  equilb Q-M dist for MM force field available from tinker `analyze P <Q atom> <M atom>`; or we could
  #  try using bond length from initial geometry in self.mol, assuming it is MM optimized
  #  ... another option would be to just use covalent radii, which doesn't seem too far off
  #  For Q-H dist, we could require user specify MM type for cap atom, then read from TINKER param file, or
  #   create an MM model for QM region w/ caps, and use `analyze P` or optimize it and use actual distances
  def get_caps(self, mol, qmatoms, basis=None, placement='abs', g_CH=0.709, d0_CH=1.09):
    """ Given QM region specified by `qmatoms`, use MM connectivity information to generate array of cap atom
      objects. We assume only single bonds on QM-MM boundary!
    """
    caps = []
    for ii in qmatoms:
      for jj in mol.atoms[ii].mmconnect:
        if jj not in qmatoms:
          # if either Q1 or M1 is not carbon, cap must be configured manually
          CC = mol.atoms[ii].znuc == 6 and mol.atoms[jj].znuc == 6
          g = g_CH if CC and placement == 'rel' else None
          d0 = d0_CH if CC and placement == 'abs' else None
          caps.append( Bunch(Q1=ii, M1=jj, g=g, d0=d0, qmbasis=basis) )

    return caps


  def update_caps(self, r, caps):
    """ Update `cap` atom positions and Jacobians d r_L/d r_Q and d r_L/d r_M based on atom positions `r`
      Ref: Senn, Thiel 2006 review (10.1007/128_2006_084), eqns. 10, 17
    """
    for cap in caps:
      assert cap.qmbasis, "cap atom between %d and %d is missing basis!" % (cap.Q1, cap.M1)
      assert cap.g or cap.d0, "cap atom between %d and %d not configured - " \
          "caps must be manually configured if Q1 or M1 is not carbon" % (cap.Q1, cap.M1)
      vMQ = r[cap.M1] - r[cap.Q1]
      rMQ = norm(vMQ)
      rcap = r[cap.Q1] + vMQ*(cap.g if cap.g else cap.d0/rMQ)
      Gmm = cap.g*np.eye(3) if cap.g else (cap.d0/rMQ)*np.eye(3) - (cap.d0/rMQ**3)*np.outer(vMQ, vMQ)
      cap.update(r=rcap, J=[(cap.M1, Gmm), (cap.Q1, np.eye(3) - Gmm)])


  def find_cap(self, a1, a2=None):
    a1, a2 = (a1,a2) if a2 is not None else a1
    for cap in self.caps:
      if (cap.Q1 == a1 and cap.M1 == a2) or (cap.Q1 == a2 and cap.M1 == a1):
        return cap

  # MM frontier atom (M1) charge adjustment options:
  # - delete M1 charges; doesn't preserve total charge
  # preserve total charge only:
  # - distribute M1 charge evenly to M2 atoms
  # preserve total charge and M1-M2 dipole (default):
  # - add M1 charge to M2, then create dipole with moment p = r_12 * q_1/N_2
  #  centered at dipolecenter (relative to M1) using two charges separated
  #  by dipolesep; if one of charges lands on M2, it is simply combined with it.
  #  If dipoleabspos=True, dipolesep and dipolecenter are measured in Angstroms,
  #  otherwise, as a fraction of r_12 (which will typically be 1 - 1.5 Ang)
  # Note that GAMESS will fail if any charges are closer than 0.1 Ang
  # Trying to get a small optimization by putting in atoms instead of creating
  #  charges list makes writeGAMESS and combinegrads too messy

  # Should handle pathological cases (which should be avoided if possible)
  #  of multiple Q1 atoms bonded to same M1, multiple M1 bonded to same M2,
  #  bonds between M1 atoms ... actually we should print warning in these cases
  def get_charges(self, mol, r, qmatoms, charge_atoms=None, radius=None,
      adjust='dipole', scaleM1=1.0, combcutoff=0.001, dipolecenter=0.5, dipolesep=0.25,
      dipoleabspos=False, byconnectivity=True, byproximity=False, proxcutoff=2.5):
    """ Generate point charges for QM calculation w/ electrostatic embedding
    charge_atoms: explicit list of MM atoms to include
    radius: exclude charges not within this distance of a QM atom
      set to None or False to include all charges
    adjust:
     'Z0' or 'L0' or 'none': no frontier charge adjustment
     'Z1' or 'L1' or 'delete': delete M1 charges
     'shift': shift M1 charge to M2; do not preserve dipole
     'dipole': shift M1 charge to M2 and preserve M1-M2 dipole
    dipolecenter and dipolesep specify position of point charges
      added when shift=dipole
    combcutoff: shift=dipole charges within this dist from M2 are
      combined with M2; this usually doesn't need to be changed
    scaleM1: prescale M1 charge to account for Q1-M1 bond deletion
     (typically 0.5 or, for no scaling, 1.0).
    byconnectivity: include in M1 all atoms connected to Q1 atoms
    byproximity: include in M1 all atoms within proxcutoff (Ang) of
      a QM atom - may be useful for H bonds, etc

    Results are returned as a list of atom objects, with relevant fields .r,
     .name, .qmq (the charge), and .J (the Jacobians wrt real atoms).
    """
    M1 = mol.get_bonded(qmatoms) if byconnectivity else []
    M1 = list(set(M1 + mol.get_nearby(qmatoms, cutoff))) if byproximity else M1
    # create list of QM charges, then adjust M1 and M2 atoms
    if charge_atoms is None:
      charge_atoms = mol.get_nearby(qmatoms, radius) if radius else mol.listatoms(exclude=qmatoms)
    charges = dict([ (ii, Bunch(name=mol.atoms[ii].name,
        r=r[ii], qmq=mol.atoms[ii].mmq, J=[(ii, 1.0)])) for ii in charge_atoms ])
    dipoles = []
    # adjustment for M1 atoms
    for ii in M1:
      # M1 scaling
      if adjust=='Z0' or adjust=='L0':
        charges[ii].qmq = scaleM1 * mol.atoms[ii].mmq
      # M1 deletion
      elif adjust=='Z1' or adjust=='L1' or adjust=='delete':
        charges[ii].qmq = 0
      # M1 shift
      elif adjust=='shift':
        charges[ii].qmq = 0
        M2 = [kk for kk in mol.atoms[ii].mmconnect if (kk not in qmatoms and kk not in M1)]
        q2adj = scaleM1*mol.atoms[ii].mmq/len(M2)
        for jj in M2:
          charges[jj].qmq += q2adj
      # M1 shift and M1-M2 dipole preservation
      elif adjust=='dipole':
        charges[ii].qmq = 0
        M2 = [kk for kk in mol.atoms[ii].mmconnect if (kk not in qmatoms and kk not in M1)]
        for jj in M2:
          r1, r2 = r[ii], r[jj]
          r21 = r2 - r1  # bond vector
          d21 = norm(r21)
          v0 = r21/d21 if dipoleabspos else r21
          J0 = np.eye(3)/d21 - np.outer(r21, r21)/d21**3 if dipoleabspos else np.eye(3)
          # dipole charge positions
          rdipp = r1 + v0*(dipolecenter + dipolesep/2)
          rdipn = r1 + v0*(dipolecenter - dipolesep/2)
          # Jacobian
          Jp = [(jj, J0*(dipolecenter + dipolesep/2)), (ii, np.eye(3) - J0*(dipolecenter + dipolesep/2))]
          Jn = [(jj, J0*(dipolecenter - dipolesep/2)), (ii, np.eye(3) - J0*(dipolecenter - dipolesep/2))]
          # charge to add to M2 and dipole charges
          q2adj = scaleM1*mol.atoms[ii].mmq/len(M2)
          qdip = q2adj*d21/norm(rdipp - rdipn)
          # combine dipole charge with M2 atom if very close
          if norm(rdipp - r2) < combcutoff:
            q2adj += qdip
          else:
            dipoles.append(Bunch(qmq=qdip, r=rdipp, J=Jp, name=mol.atoms[ii].name + mol.atoms[jj].name + 'p'))
          if norm(rdipn - r2) < combcutoff:
            q2adj += -qdip
          else:
            dipoles.append(Bunch(qmq=-qdip, r=rdipn, J=Jn, name=mol.atoms[ii].name + mol.atoms[jj].name + 'n'))
          # we now have final value for M2 charge
          charges[jj].qmq += q2adj
    # ---
    return [q for q in list(charges.values()) + dipoles if abs(q.qmq) > self.charge_cutoff]


  def mm_EandG(self, mol, r, modcharge=None, newcharges=True, charges=[],
      inactive=None, m1inactive=True, noconnect=None, grad=False, prog='openmm', subtract=False):
    """ Call out to MM program (currenly only Tinker) to get energy, gradient, and/or energy components
      modcharge: list of atoms on which to modify MM charge (typically qmEatoms, since QM-MM charge-charge
        interaction is calculated at QM level and thus should not be included in MM calc)
      newcharges: list of new charges
        - or True to distribute difference between net charge on modcharge atoms and self.qm_charge over
        modcharge atoms to preserve total MM charge (since QM system can only have integer charge, often
        zero) - all these atoms should be set inactive, so that interactions between them are ignored
        - or None/False to set all modcharge atoms charges to zero
      charges: list of (atom number, charge) tuples for overriding MM charges (for non-QM atoms)
      inactive: list of atoms to make inactive in MM calc
      m1inactive: if True, all atoms bonded to explicit inactive atoms (M1 atoms if inactive == qmatoms) are
        also made inactive; to be used with moving cap atoms
      noconnect: list of atoms for which bonds are ignored (so that MM bonded params are not needed)
      subtract: calculate MM energy by subtracting energy of inactive atoms from energy of entire system;
        sanity check and future support for MM codes w/o inactive; should not be used with noconnect due to
        huge vdW energies.  Note this is not the same as "subtractive QM/MM" - it is just an alternative way
        to compute MM energy when some atoms are "inactive"
    """
    if inactive and m1inactive:
      inactive = inactive + mol.get_bonded(inactive)
    if modcharge:
      islist = safelen(newcharges) > 0  # we want an explicit error if length is incorrect
      modqq = (sum(mol.atoms[ii].mmq for ii in modcharge)
          - self.qm_charge)/len(modcharge) if newcharges == True else 0.0
      charges = charges + [(ii, newcharges[ii] if islist else modqq) for ii in modcharge]

    if prog == 'tinker':
      prefix = "%s_%03dmm" % (self.prefix, self.iternum)
      title = "QMMM iter %d" % self.iternum
      def tinker_prog(mol, r, inactive=None, charges=None, components=None):
        return tinker_EandG(mol, r, prefix=prefix, key=self.mm_key, inactive=inactive,
            charges=charges, noconnect=noconnect, title=title, grad=grad, components=components)
      prog = tinker_prog
    elif prog == 'openmm':
      prog = openmm_EandG if subtract else OpenMM_EandG(mol, self.mm_key, inactive, charges)
      self.mm_opts['prog'] = prog  #if not subtract

    # subtractive calculation not needed for Tinker since it supports inactive atoms, but useful sanity check
    # For Tinker, self.mm_key must not contain references to atom numbers
    if subtract and inactive:
      assert not noconnect, "noconnect not supported with subtract"
      comp = {} if self.Ecomp is not None else None
      Eall, Gall = prog(mol, r, inactive=None, charges=charges, components=comp)
      inactive = sorted(inactive)  # make debugging slightly easier
      invmap = {v: ii for ii, v in enumerate(inactive)}
      submol = mol.extract_atoms(inactive)
      subr = r[inactive] if r is not None else None
      subq = [(invmap[chg[0]], chg[1]) for chg in charges if chg[0] in invmap]
      #subnc = [invmap[a] for a in noconnect if a in invmap] if noconnect is not None else None
      subcomp = {} if self.Ecomp is not None else None
      Esub, Gsub = prog(submol, subr, inactive=None, charges=subq, components=subcomp)
      if self.Ecomp is not None:
        self.Ecomp.update({k: v - subcomp.get(k, 0) for k,v in comp.items()})
      if not grad:
        return Eall - Esub, None
      Gall[inactive] -= Gsub
      return Eall - Esub, Gall
    else:
      return prog(mol, r, inactive=inactive, charges=charges, components=self.Ecomp)


  # Interaction energy between QM background charges is handled at MM level, so must be subtracted from QM
  #  energy if included (as with GAMESS, but not NWChem unless "geometry bqbq" is used)
  # - even if we wanted to zero all MM point charges instead, this would be wrong - most MM force fields do
  #  not include 1-2 and 1-3 electrostatic or vdW interactions (i.e., implicit in bond and angle terms)
  def qm_EandG(self, mol, r, qmatoms, caps, charges, grad=True,
      prog='pyscf', charge=0, inp=None, scffn=None, basis=None, write_basis=True):
    """ caps, charges: array of atoms objects for cap atoms and point charges.  For caps and charges, we
      expect a J field holding list of tuples of the form (i, J), where i is the index of a real atom and J
      is the Jacobian wrt atom i.
      basis: default basis for atoms w/o qmbasis attribute
    """

    title = "QMMM iter %d" % self.iternum
    t0 = time.time()
    moguess = self.prev_cclib.mocoeffs[0] if self.moguess == 'prev' and self.prev_cclib else self.moguess
    if prog == 'pyscf':
      Eqm, Gqm, scn = pyscf_EandG(mol, r, qmatoms=qmatoms,
          charges=charges, caps=caps, qm_charge=charge, scffn=scffn, basis=basis)
      self.prev_pyscf = scn
    elif prog == 'gamess':
      from ..io.gamess import gamess_cclib
      from ..io import cclib_EandG
      prefix = "%s_%03dqm" % (self.prefix, self.iternum)
      mol_cc = gamess_cclib(mol, r, prefix, header=inp, title=title,
          qmatoms=qmatoms, charges=charges, caps=caps, moguess=moguess, write_basis=write_basis)
      Eqm, Gqm = cclib_EandG(mol_cc)
      self.prev_cclib = mol_cc
    elif prog == 'nwchem':
      from ..io.nwchem import nwchem_cclib
      from ..io import cclib_EandG
      # NWChem a bit faster than GAMESS
      # "noprint mulliken" necessary to prevent crash w/ BG charges; "print low" at global level also works
      prefix = "%s_%03dqm" % (self.prefix, self.iternum)
      task = "\n".join(["scf", "  noprint mulliken", "end", "task scf gradient"])
      mol_cc = nwchem_cclib(mol, r, prefix, task=task, header='title "%s"' % title,
          qmatoms=qmatoms, charges=charges, caps=caps, basis=write_basis)
      Eqm, Gqm = cclib_EandG(mol_cc)
      self.prev_cclib = mol_cc
    else:
      assert 0, "Invalid QM program: " + prog
    self.timing['QM external'] = time.time() - t0

    # compute charge-charge E and G to remove from QM E and G, since these are already included in MM E and G
    #  Eqq = \sum_{i,j>i} q_i*q_j/|r_i - r_j|;  Gqq_i = -q_i*\sum_j q_j*(r_i - r_j)/|r_i - r_j|**3
    # non-vectorized calculation was slow for large number of charges. Note pyscf doesn't include charge-charge terms
    if charges and prog != 'pyscf':
      assert 0, "You had the wrong sign for Eqq - why didn't sanity checks catch this?!?"
      rq = np.asarray([c.r for c in charges])
      mq = np.asarray([c.qmq for c in charges])
      #Eqq, Gqq = coulomb(rq, mq)
      dr = rq[:,None,:] - rq[None,:,:]
      dd = np.sqrt(np.sum(dr*dr, axis=2))
      np.fill_diagonal(dd, np.inf)
      invdd = 1/dd
      qq = mq[:,None]*mq[None,:]  # broadcasting seems to be faster than np.outer()
      Eqq = 0.5*ANGSTROM_PER_BOHR*np.sum(qq*invdd)
      Gqq = -ANGSTROM_PER_BOHR*np.sum((qq*(invdd**3))[:,:,None]*dr, axis=1)
    else:
      Eqq, Gqq = 0.0, np.zeros(len(charges))

    if self.Ecomp is not None:
      self.Ecomp.update(Eqm0=Eqm, Eqmcorr=Eqq)
    if not grad:
      return Eqm + Eqq, None

    # We assume that the order of Gqm is [qmatoms, cap atoms, point charges] - this is order written by
    #  write_gamess_inp
    dimG = (mol.natoms, 3)
    Gqmqm, Gcap, Gcharge = np.zeros(dimG), np.zeros(dimG), np.zeros(dimG)
    ii = 0
    for aa in qmatoms:
      Gqmqm[aa] = Gqm[ii]
      ii += 1
    # redistribute gradient on QM caps to real atoms
    for atom in caps:
      for aa, gg in atom.J:
        Gcap[aa] += np.dot(gg, Gqm[ii])
      ii += 1
    # gradient on QM point charges
    for jj, a1 in enumerate(charges):
      Gcorr = Gqm[ii] + Gqq[jj]
      for aa, gg in a1.J:
        Gcharge[aa] += np.dot(gg, Gcorr)
      ii += 1

    if self.Gcomp is not None:
      self.Gcomp.update(Gqm0=Gqm, Gqmqm=Gqmqm, Gcap=Gcap, Gcharge=Gcharge)
    return Eqm + Eqq, Gqmqm + Gcap + Gcharge


## Untested old code

  # microitEandG:
  # * QM calc yielding E, G, grid
  # * external MM opt. using RESP from grid to give new MM geom
  # To be handled by calling code:
  # * new QM geom from G
  # * repeat with new QM and MM geoms
  def microitEandG(self, Elast=np.inf, **kargs):
    """ E and G for one (macro) step, using external MM optimizer
    Elast: if energy E for current geom is greater than Elast, MM optimization
     step is skipped (on the assumption that prev opt step current geom will
     be rejected by optimizer.
    """
    atoms = self.mol.atoms
    # QM calc only
    E, G, ESP, ESPgrid, Ecomp, Gcomp = self.EandG(mmexec=None, dograd=1, docomp=1, doESP=1, **kargs)
    if E > Elast:
      return E, G[qmatoms]
    # calculate RESP charges, conserving net MM charge of QM region
    netq = sum([atoms[ii].mmq for ii in self.qmatoms])
    qESP = resp(centers=atoms.r[qmatoms], grid=ESPgrid, phi0=ESP, netcharge=netq)
    # calculate gradient on MM atoms due to ESP charges on QM atoms
    G_ESP = np.empty( (len(mmatoms), 3) )
    for ii in range(len(mmatoms)):
      r21 = atoms.r[qmatoms] - atoms[ii].r
      G_ESP[ii] = ANGSTROM_PER_BOHR * atoms[ii].mmq * np.sum( 0, qESP*r21/(np.sum(1, r21**2)**1.5) )

    # run MM optimization
    mminp, mmEout = self.writeTINKER(atoms, modcharge=qmatoms,
      newcharges=qESP, inactive=inactive)
    assert os.system(self.mmexec + "optimize " + mminp + " E > " + mmEout)==0, \
      "Error running MM program: " + self.mmexec
    # read new MM geom
    rmm = mm_geom(mminp + "_2")
    # TINKER won't overwrite files; we must remove them ourselves
    os.remove(mminp + "_2")
    # update MM atom position...I don't like this code
    mmatoms = [ii for ii in self.mol.listatoms() if ii not in self.qmatoms]
    for ii,aa in enumerate(mmatoms):
      atoms[aa].r = rmm[ii]
    # ---
    return E, G[qmatoms]


  def MMmicroit(self, rmm, charges=True, Gmmqm=0):
    """
    Single MM microit step for use with internal (i.e. python) optimizer
     - update MM atom positions, then get E and G
    rmm is array of MM atom positions
    charges is array of ESP charge values for qmatoms
    Gmmqm is force on MM atoms due to QM atoms (incl just MM atoms)
     added to MM gradient; should only include MM atoms
    """
    mmatoms = [ii for ii in self.mol.listatoms() if ii not in self.qmatoms]
    # update MM atom position...I don't like this code
    for ii,aa in enumerate(mmatoms):
      self.mol.atoms[aa].r = rmm[ii]

    # for microit, all qmatoms are inactive (e.g., not just qmEatoms)
    mminp, mmEout = self.writeTINKER(self.mol, modcharge=qmatoms,
      newcharges=charges, inactive=self.qmatoms)
    #assert os.system(self.mmexec + "analyze " + mminp + " E > " + mmEout)==0, \
    #  "Error running MM program: " + self.mmexec
    assert os.system(self.mmexec + "testgrad " + mminp + " Y N > " + mmGout)==0, \
      "Error running MM program: " + self.mmexec
    #Emm = self.mm_energy(mmEout)
    Emm, Gmm = self.mm_grad(mmGout)
    # adjust gradient, then remove QM atoms
    return Emm, Gmm[mmatoms] + Gmmqm


  # gradmix feature (averaging of QM and MM gradients near boundary) is completely
  #  untested and probably stupid; keep in mind that all QM atoms have an MM grad
  #  contribution from vdW - with gradmix, this is actually scaled down!
  # Unless QM region happens to be nearly spherical, it doesn't make sense to weigh
  #  mixing using distance - only makes sense to do it by connectivity
  # Rather than assume order of Gqm, we could require cap and point charge atom objects to
  #  have a .qmidx field which is set by writeQM

  def gradmix(self, Gqm, Gmm, caps=None, charges=None, gradmix=[], components=False):
    """ Gqm, Gmm: gradients from QM and MM computations as 2D (? x 3) arrays
      gradmix: list of factors 0 <= a_i <= 1 such that the final grad on atom j in set Qi (determined by
        connectivity - Q1 = frontier QM atoms, etc.) is G_j = (1 - a_i)*Gqm_j + a_i*Gmm_j (by default
        a_i = 0, all i). QM atoms for which a_i > 0 should be MM active!
    """
    if len(gradmix) > 0:
      Gqm1 = np.array(Gqm)
      # alternative would be to create two Gmix arrays (both with 1s everywhere but on
      #  Qn atoms) to multiply Gqm1 and Gmm
      Gmm1 = np.array(Gmm)  # don't touch caller's Gmm
      # generate Q1
      # TODO: need generic fn in molecule.py: getconnected - restricted to
      #  specified subset and/or cache Q1, M1, etc
      Qn = []
      qmatoms = self.qmatoms
      for ii in qmatoms:
        for jj in self.mol.atoms[ii].mmconnect:
          if jj not in qmatoms:
            Qn.append(ii)
            break
      Gmm1[Qn] *= gradmix[0]  # * np.array([1.0, 1.0, 1.0])
      Gqm1[Qn] *= 1 - gradmix[0]
      for p in gradmix[1:]:
        # get next layer of QM atoms
        Qnext = []
        qmatoms = [ii for ii in qmatoms if ii not in Qn]
        for ii in Qn:
          for jj in self.mol.atoms[ii].mmconnect:
            if jj in qmatoms:
              Qnext.append(ii)
              break
        Qn = Qnext
      Gmm1[Qn] *= p
      Gqm1[Qn] *= 1 - p
    else:
      Gmm1 = Gmm  # alias

    #comps = {"Gqm": Gqm0, "Gcap": Gcap, "Gcharge": Gcharge, "Gmm": Gmm1}  if components else {}
    return Gqm1 + Gmm1


  def testwrite(self, **kargs):
    """ test input file generation """
    oldvals = self.set(**kargs)
    mminp, mmout = self.writeTINKER(self.mol, nocharge=self.qmatoms,
      inactive=self.qmatoms)
    qminp, qmout = self.writeGAMESS(self.mol, self.qmatoms,
      charges=self.get_charges(self.mol, self.qmatoms, **self.chargeopts) )
    self.set(oldvals)


  # TODO: support QM region with net charge
  # - should we suppress vdW interaction between qmEatoms and qmatoms\qmEatoms?
  # cclib atom numbering is one based
  def qm_lmoE(self, qmout, charges, caps, grad=False):
    """ QM energy from fns in lmo_overlap.py; must use NPRINT=3 in GAMESS inp """
    mol = lmo_overlap.makePyQMol( cclib_open(qmout) )
    lmomethod = self.qmmethod.lower().split("|")[1:] + [None, None]
    if lmomethod[0] == 'lmo':
      # arrays of energy expvals for MOs and LMOs
      moE, lmoE, enuke = lmo_overlap.mo_Eexpval(mol)
      # convert to indexing in inp file (and thus mol object)
      # - qmatoms are first, everything else follows, qmEatoms is a subset of qmatoms
      qmEatoms = [ ii for ii,a in enumerate(self.qmatoms) if a in self.qmEatoms ]
      #invatoms = [ ii for ii in range(mol.natom) if ii not in qmEatoms ]
      # final electronic energy is \sum_lmos <lmo|H|lmo> * prob_in_qmEatoms(lmo)
      lmoregn = lmo_overlap.orderbyregion(mol, qmEatoms)
      if lmomethod[1] == 'fuzzy':
        Eelec = np.sum(lmoregn * lmoE)
      else:
        # include exactly 1.0, 0.5, or 0.0 contribution from each LMO to
        #  ensure no net charge contribution
        znucreg = sum([self.mol.atoms[ii].znuc for ii in self.qmEatoms])
        lmoorder = np.argsort(lmoregn)
        Eelec = np.sum(lmoE[ lmoorder[-int(znucreg/2):] ])
        #Eelec = sum([ lmoE[ii] for ii in lmoorder[-int(znucreg/2):] ])
        if znucreg != 2*int(znucreg/2):  # odd?
          Eelec += 0.5*lmoE[ lmoorder[-int(znucreg/2)-1] ]
    elif lmomethod[0] == 'eda':
      # TODO: eda_mayer
      if lmomethod[1] == 'mo':
        Eeda = lmo_overlap.trivialEDA(mol, mol.mocoeffs[0][0:mol.nocc])
      elif lmomethod[1] == 'lmo':
        Eeda = lmo_overlap.trivialEDA(mol, mol.lmocoeffs[0][0:mol.nocc])
      Eelec = np.sum(Eeda[:,qmEatoms])*2.0 - np.sum(Eeda[:,qmEatoms][qmEatoms])
    else:
      assert False, "Invalid QM method"
    # ---
    # Eelec includes interaction between electrons on qmEatoms and
    #  all nuclei, incl. link atoms, and all background charges, so we must
    #  calculate interaction of the same with nuclei of qmEatoms.
    Enuke = 0.0
    for ii in self.qmEatoms:
      a1 = self.mol.atoms[ii]
      for bq in charges:
        # TODO: verify that qmq is correct (was mmq)
        Enuke += a1.znuc*bq.qmq/norm(a1.r - bq.r)
      for a2 in caps:
        Enuke += a1.znuc*1.0/norm(a1.r - a2.r)  # znuc = 1 for H
      for jj in self.qmatoms:
        if ii == jj: continue
        # prevent double counting of interactions between qmEatoms
        elif jj in self.qmEatoms: q2 = 0.5*self.mol.atoms[jj].znuc
        else: q2 = self.mol.atoms[jj].znuc
        Enuke += a1.znuc*q2/norm(a1.r - self.mol.atoms[jj].r)
    Enuke *= ANGSTROM_PER_BOHR
    #enukereg = lmo_overlap.calc_enuke(mol, qmEatoms)/2 + lmo_overlap.calc_enuke(mol, qmEatoms, invatoms)
    # ---
    return Eelec + Enuke, {'Eqm_elec': Eelec, 'Eqm_nuke': Enuke}


  def mm_energy(self, mmout):
    """ obtain MM energy, given log file; only supports TINKER """
    return read_tinker_energy(mmout, components=True)


  def mm_grad(self, mmout):
    """ obtain MM gradient (and energy), given output of testgrad; only supports TINKER """
    return read_tinker_grad(mmout)


  def mm_geom(self, mmout):
    """ obtain MM geometry, given .xyz file; only supports TINKER """
    return read_tinker_geom(mmout)
