import random
#from chem.io import *
from chem.io.openmm import *
from chem.model.build import *
from chem.model.prepare import *
from chem.opt.optimize import *
from chem.opt.dlc import HDLC, DLC
from chem.fep import *
from chem.mm import *
from chem.analyze import *
from chem.data.pdb_data import AA_3TO1, PDB_EXTBACKBONE


## dE_search2 (dE_search is in trypsin3.py) - global search mutating specified residues to optimize binding energy
def relax_bind(mol, E0, ff, host='protein', lig='not protein', active=None, mmargs={}, optargs={}):
  optargs = dict(dict(gtol=0.04, maxiter=50, raiseonfail=False, verbose=-100), **optargs)
  r0 = mol.r
  active = mol.select(active) if active is not None else None
  waters = [Rigid(r0, res.atoms, pivot=None) for res in mol.residues if res.name == 'HOH']
  coords = HDLC(mol, [XYZ(r0, active=active, atoms=mol.select('not water'))] + waters) \
      if waters else XYZ(r0, active=active)
  res, r = moloptim(OpenMM_EandG(mol, ff, **mmargs), r0, coords=coords, **optargs)
  return (np.inf if res.fun > E0 + 0.5 else dE_binding(mol, ff, host, lig, r)), r


# calculate per-residue energy - in vacuum, not solvent!
# ... not sure this is really useful - doesn't seem particularly correlated w/ binding energy

def get_coulomb_terms(rq1, mq1, rq2, mq2, kscale):
  dr = rq1[:,None,:] - rq2[None,:,:]
  dd = np.sqrt(np.sum(dr*dr, axis=2))
  #np.fill_diagonal(dd, np.inf)
  return kscale*ANGSTROM_PER_BOHR*(mq1[:,None]*mq2[None,:])/dd


def sidechain_Eqq(mol, nearres, lig):
  """ return electrostatic energy between atoms `lig` and sidechains of each residue `nearres` """
  ligatoms = [mol.atoms[ii] for ii in mol.select(lig)]
  rq1, mq1 = np.array([a.r for a in ligatoms]), np.array([a.mmq for a in ligatoms])
  Eres = []
  for resnum in nearres:
    atoms = [mol.atoms[ii] for ii in mol.residues[resnum].atoms if mol.atoms[ii].name not in PDB_EXTBACKBONE]
    rq2, mq2 = np.array([a.r for a in atoms]), np.array([a.mmq for a in atoms])
    Eqq = get_coulomb_terms(rq1, mq1, rq2, mq2, OPENMM_qqkscale)
    Eres.append(np.sum(Eqq))
  return np.asarray(Eres)


# perform mutations, replacing worst candidate in population if result is better
# - this makes it possible to accept some mutations that are worse than original, at a cost of loss of diversity
# ... therefore we assume dE_search will be run multiple times for better diversity

def dE_search2(molAI, molAZ, bindAI, bindAZ, nearres, candidates, mutweight=None, Npop=20, maxiter=250):
  molAIhist, molAZhist, dEhist = [None]*Npop, [None]*Npop, [np.inf]*Npop
  dEmaxidx = 0
  # favor candidates with more rotamers
  candweights = l1normalize([np.ceil(np.sqrt(len(get_rotamers(c)) or 4)) for c in candidates])
  try:
    for ii in range(maxiter):
      if dEhist[dEmaxidx] == np.inf or (dEhist[dEmaxidx] > 0 and ii < 2*Npop):
        mutAI, mutAZ = Molecule(molAI), (Molecule(molAZ) if molAZ is not None else None)
        # # shuffle so mutations are applied in random order ... doesn't matter anymore since we only relax once
        mutres = random.sample(nearres, len(nearres))
      else:
        mutAI, mutAZ = Molecule(molAIhist[ii%Npop]), (Molecule(molAZhist[ii%Npop]) if molAZ is not None else None)
        mutres = random.choices(nearres, weights=(mutweight(mutAI) if mutweight is not None else None))
      # mutate, choose random rotamer
      for resnum in mutres:
        newres = random.choices(candidates, weights=candweights)[0]
        mutate_residue(mutAI, resnum, newres)
        rotamers = get_rotamers('HIE' if newres == 'HIS' else newres)  # fix this
        if rotamers:
          set_rotamer(mutAI, resnum, random.choice(rotamers))  #rotamers[:10] ... limit to most common?
      # relax and calc dE_binding
      deAI, r1 = bindAI(mutAI)
      if deAI >= dEhist[dEmaxidx]:  #0:  -- let's save positive binding energy theos for possible inspection
        print("%03d: %s: failed (deAI = %f)" % (ii, ''.join(AA_3TO1[mutAI.residues[rn].name] for rn in nearres), deAI))
        continue  #import pdb; pdb.set_trace()
        #tooclose = cKDTree(r1).query_pairs(0.9, output_type='ndarray')
        #dd = [mutAI.bond(b) for b in tooclose]
        #np.atomsres(np.ravel(tooclose[np.argsort(dd)]))  -- try different rotamer for problematic residue
      mutAI.r = r1
      # mutate and relax for 2nd system, if specified
      if mutAZ is not None:
        for resnum in mutres:
          mutate_residue(mutAZ, resnum, mutAI.extract_atoms(residues=resnum))
        deAZ, mutAZ.r = bindAZ(mutAZ)
      else:
        deAZ = 0
      # accept/reject
      dE = deAI - deAZ
      accept = deAZ <= 0 and dE < dEhist[dEmaxidx]  #deAI < 0 and
      print("%03d: %s: %f - %f = %f%s" % (ii, ''.join(AA_3TO1[mutAI.residues[rn].name] for rn in nearres),
          deAI, deAZ, dE, (' (replacing %f)' % dEhist[dEmaxidx]) if accept else ''))
      if accept:
        dEhist[dEmaxidx] = dE
        molAIhist[dEmaxidx] = mutAI
        molAZhist[dEmaxidx] = mutAZ
        dEmaxidx = max(range(Npop), key=lambda kk: dEhist[kk])
  except KeyboardInterrupt as e:  #(, Exception) as e:
    pass

  ord = argsort(dEhist)
  return Bunch(molAI=[molAIhist[ii] for ii in ord], molAZ=[molAZhist[ii] for ii in ord], dE=[dEhist[ii] for ii in ord])


## prep_solvate() and prep_relax() replace solvate_prepare()
def prep_solvate(mol, ff=None, pad=6.0, center=None, side=None,
    solute_res=None, solvent_chain=None, neutral=False, overfill=1.5):
  """ if neutral is non-zero int, system will have at least abs(neutral) positive (if neutral > 0) or negative
    ions (if neutral < 0) - useful for FEP where we need counterions to decouple along with ligand
  """
  # make a cubic water box
  side = np.max(np.diff(mol.extents(pad=pad), axis=0)) if side is None else side
  solvent = water_box(side, overfill=overfill)
  solvent.r = solvent.r + 0.5*side  # OpenMM centers box at side/2 instead of origin
  ions = []
  if neutral:
    # Amber names for Na, Cl
    na_plus = Molecule(atoms=[Atom(name='Na+', znuc=11, r=[0,0,0], mmq=1.0)]).set_residue('Na+')
    cl_minus = Molecule(atoms=[Atom(name='Cl-', znuc=17, r=[0,0,0], mmq=-1.0)]).set_residue('Cl-')
    openmm_load_params(mol, ff=ff, charges=True)  # needed
    net_charge = round(np.sum(mol.mmq))
    if type(neutral) is int:
      ions.extend( [na_plus if neutral > 0 else cl_minus]*abs(neutral) )  # add requested ions
      net_charge += neutral
    ions.extend( [cl_minus if net_charge > 0 else na_plus]*abs(net_charge) )  # add ions to neutralize

  # d=2.0 matches solvent-solute distance distribution after MD fairly well
  center = np.mean(mol.extents(), axis=0) if center is None else center
  return solvate(mol, solvent, r_solute=(mol.r - center + 0.5*side),
      d=2.0, ions=ions, solute_res=solute_res, solvent_chain=solvent_chain)  #_p=na_plus, ion_n=cl_minus)


def prep_relax(mol, ff, T0=None, p0=None, freeze=[], eqsteps=5000):  #, **kwargs):
  isPBC = mol.pbcbox is not None
  ctx0 = openmm_MD_context(mol, ff, T0, freeze=freeze, freezemass=0, constraints=None)  #, **kwargs)
  openmm.LocalEnergyMinimizer.minimize(ctx0, maxIterations=200)
  mol.r = ctx0.getState(getPositions=True, enforcePeriodicBox=isPBC).getPositions(asNumpy=True)/UNIT.angstrom
  badpairs, badangles = geometry_check(mol)
  if eqsteps > 0:
    ctx = openmm_MD_context(mol, ff, T0=T0, p0=p0, freeze=freeze)  #, **sysargs)
    openmm_dynamic(ctx, eqsteps)  # use openmm_dynamic to show progress  #ctx.getIntegrator().step(eqsteps)
    simstate = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=isPBC)
    mol.r = simstate.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
    if isPBC:
      mol.pbcbox = np.diag(simstate.getPeriodicBoxVectors(asNumpy=True).value_in_unit(UNIT.angstrom))
    mol.md_vel = simstate.getVelocities(asNumpy=True).value_in_unit(UNIT.angstrom/UNIT.picosecond)
  return mol


## mutate multiple residues
def prep_mutate(mol, resnums, newress):
  if np.ndim(resnums) == 0: resnums = [resnums]
  if np.ndim(newress) == 0: newress = [newress]*len(resnums)
  mut = Molecule(mol)
  for resnum, newres in zip(resnums, newress):
    if newres is not None:
      mutate_residue(mut, resnum, newres)
  return mut


## modify molecule with chemical reaction
def reactmol(mol, breakbonds, makebonds):
  """ modify `mol` by removing bonds in `breakbonds`, adding bonds in `makebonds` and relaxing with SimpleMM """
  bonds = mol.get_bonds()
  for b in breakbonds:
    bonds.remove(b if b[1] > b[0] else b[::-1])
  for b in makebonds:
    bonds.append(b)
  mol.set_bonds(bonds)
  # generic MM relaxation
  gaff = load_amber_dat(DATA_PATH + '/amber/gaff.dat')
  gaff_trivial(mol)
  set_mm_params(mol, gaff, allowmissing=True)
  res, mol.r = moloptim(SimpleMM(mol), mol, maxiter=200)
  return mol


# maxiter=100: we need to optimize better than for combined calc!
def dE_binding_sep(mol, ff, host, lig, active=None, r=None, maxiter=100):
  """ dE_binding w/ host, lig, and host+lig systems relaxed separately """
  if r is None: r = mol.r
  selA, selI = mol.select(host), mol.select(lig)
  selAI = sorted(selA+selI)
  sels = [selAI, selA, selI]
  activeset = frozenset(mol.select(active)) if active is not None else None
  eAI, eA, eI = [ moloptim(OpenMM_EandG(mol.extract_atoms(sel), ff), r[sel],
      coords=XYZ(r[sel], active=[ii for ii,a in enumerate(sel) if a in activeset]) if activeset else None,
      maxiter=maxiter, raiseonfail=False, verbose=-2)[0].fun for sel in sels ]
  return (eAI - eA - eI)*KCALMOL_PER_HARTREE


# - this feels (slightly) more reasonable than using fixed number of waters (based on size of lig)
# - we should be using proper solvent exclusion - vdW radius of each atom (or znuc > 1?) + 1.4 Ang solvent radius
# - rlig is lig position from system w/ lig present; lig should not be present in mol,Rs
def dE_binding_hoh(mol, ff, Rs, rlig, host='protein', d=2.8, Nwat=10000):
  """ calculate dE_binding for water molecules in frames `Rs` with O within `d` (default 2.8 Ang) of any point
    in `rlig`; to use fixed number `Nwat` of waters instead, pass a sufficiently large value for `d`
  """
  Rs1 = [Rs] if np.ndim(Rs) < 3 else Rs
  ow = mol.select('water and znuc > 1')
  nearhoh = [ mol.atomsres( mol.get_nearby(rlig, d, active=ow, r=r, sort=True)[:Nwat] ) for r in Rs1 ]
  print([ len(nw) for nw in nearhoh ])
  dE = [ dE_binding(mol, ff, host, mol.resatoms(nw), r) for nw,r in zip(nearhoh, Rs1) ]
  return dE[0] if np.ndim(Rs) < 3 else dE


# calculate "reaction path" for binding by using optimization w/ constraint instead of MD
# ... seems to work reasonably
def expel_mm(mol, ff, lig, bindvec, freeze=[], dist=12.0, nsteps=12):
  ligatoms = mol.select(lig)
  freeze = mol.select(freeze)
  com = center_of_mass(mol, ligatoms)
  centlig = mol.get_nearby([com], 2.8, active=ligatoms, sort=True)[0]  # lig atoms closest to lig COM

  # we actually want frozen host atom closest to COM - bindvec
  #anchres = mol.atomsres( mol.get_nearby(ligin, 5.0, active='backbone', sort=True)[0] )[0]
  #anchatoms = mol.resatoms(anchres)  ## TODO: only sidechain
  #anchatom = res_select(mol, anchres, 'CA')[0]
  hits = mol.get_nearby(ligin, 5.0, active=freeze)
  linedist = [norm(np.cross(mol.atoms[ii].r - com, normalize(bindvec))) for ii in hits]
  anchatom = hits[np.argmin(linedist)]
  anchatoms = [anchatom]  #mol.resatoms(mol.atomsres(anchatom))

  rcbond = [anchatom, centlig]
  dlc = DLC(mol, atoms=ligatoms + anchatoms, bonds=[rcbond],
      autobonds='none', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
  dlc.constraint(dlc.bond(rcbond))
  a2 = mol.listatoms(exclude=dlc.atoms)
  hdlc = HDLC(mol, [XYZ(mol, active=sorted(set(a2) - set(freeze)), atoms=a2), dlc])

  rc0 = mol.bond(rcbond)
  rcs = np.linspace(rc0, rc0+dist, nsteps)
  RCr, RCe = [None]*nsteps, [None]*nsteps
  EandG = OpenMM_EandG(mol, ffpgbsa)
  for ii,rc in enumerate(rcs):
    mol.bond(rcbond, rc)
    res, r = moloptim(EandG, mol, coords=hdlc, maxiter=50, gtol=1E-4)
    RCr[ii], RCe[ii] = r, res.fun
    mol.r = r

  return RCr, RCe


# This is "steered molecular dynamics"
# - we could add support for pulling ligand into site (dist < 0), but this doesn't seem to have much luck
#  reproducing original ligand conformation
# - non-equilib work doesn't seem to be commonly used - one paper stated that correct free energy is dominated by
#  rarely-sampled low energy paths; in any case, a large number of runs are required, of which this fn does one
# see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7544513/
# - we return velocities to enable using this trajectory to initialize umbrella samping
def expel_md(mol, ffp, T0, lig, bindvec, freeze, dist=8.0, nsteps=20000):
  t_ramp = nsteps/100.0  # = 200.0
  offset = 4.0  # Ang
  Wneq = []  # work
  def metafn(ii, ctx):
    H0 = ctx.getState(getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy()/EUNIT
    ctx.setParameter('width', ((min(ii/t_ramp, 1.0)*dist) + offset)*0.1)
    H1 = ctx.getState(getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy()/EUNIT
    Wneq.append(H1 - H0)
  #metafn = lambda ii,ctx: ctx.setParameter('width', 1.5 - min(ii/t_ramp, 1.0)*1.0)
  #metafn = lambda ii,ctx: ctx.setParameter('width', min(ii/t_ramp, 1.0)*1.0)

  # harmonic shell allows for both positive and negative work
  force = openmm.CustomCentroidBondForce(1, "A*(d-width)^2; d = sqrt((x1-xc)^2+(y1-yc)^2+(z1-zc)^2); A = 8000.0")
  # Gaussian step barrier (positive work only)
  #force = openmm.CustomCentroidBondForce(1,
  #    "A*exp(-max(0, d-width)^2/(2*std^2)); d = sqrt((x1-xc)^2+(y1-yc)^2+(z1-zc)^2); A = 800.0; std = 0.1")
  #    "A*(1 - exp(-max(0, d-width)^2/(2*std^2))); d = sqrt((x1-xc)^2+(y1-yc)^2+(z1-zc)^2); A = 800.0; std = 0.1")
  # smoothstep: "A * x*x*(3 - 2*x); x = min(max((r - edge0)/(edge1 - edge0), 0.0), 1.0)"
  force.addPerBondParameter('xc')
  force.addPerBondParameter('yc')
  force.addPerBondParameter('zc')
  force.addGlobalParameter('width', offset*0.1)

  ligatoms = mol.select(lig)
  com = center_of_mass(mol, ligatoms)
  force.addGroup(ligatoms)
  force.addBond([0], (com - offset*normalize(bindvec))*0.1)  # offset center 4 Ang from COM; convert Angstrom to nm
  force.setForceGroup(31)

  ctx = openmm_MD_context(mol, ffp, T0, freeze=freeze, v0=mol.md_vel, forces=[force])
  #metafn(0, ctx)  #ctx.setParameter('width', 0.0)
  #ctx.getState(getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy() / EUNIT
  Rs, Es, Vs = openmm_dynamic(ctx, nsteps, 100, vel=True, sampfn=metafn)

  return Rs, Es, Vs, Wneq


# openmm_fep from trypsin2.py (w/ HMC option removed)
def fep_decouple(mol, ff, T0, lig, freeze=[], state=None, mdsteps=25000, sampsteps=100,
    ele_steps=5, vdw_steps=5, rst_steps=0, restraint=None):
  """ compute free energy difference for turing off non-bonded interactions between `ligatoms` and rest of
    `mol` at temperature `T0` with openmm force field `ff` using `ele_steps` "lambda" steps to turn off
    electrostatic interaction followed by `vdw_steps` lambda steps to turn off vdW interaction, with `mdsteps`
    MD steps for each lambda step, sampling energy every `sampsteps` MD steps.
  """
  from openmmtools import alchemy

  ligatoms = mol.select(lig)
  basesystem = ff.createSystem(openmm_top(mol), constraints=openmm.app.HBonds, rigidWater=True,
      nonbondedMethod=openmm.app.PME, nonbondedCutoff=min(0.5*min(mol.pbcbox), 10)*UNIT.angstrom)
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

  if state is None: state = Bunch()
  ctx = openmm_MD_context(mol, system, T0, freeze=freeze, r0=state.get('r'), v0=state.get('v'))

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
      print("%s: running %d MD steps for lambdas: ele %.2f, vdw %.2f, restraint %.2f" % (
          time.strftime('%X'), mdsteps, lambda_ele[kk], lambda_vdw[kk], lambda_rst[kk]))
      for jj in range(state.get('idx_samps', 0), nsamps):
        alch_lambdas(ctx, lambda_ele[kk], lambda_vdw[kk], lambda_rst[kk])
        ctx.getIntegrator().step(sampsteps)
        if 'Rs' in state:
          r1 = ctx.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)/UNIT.angstrom
          state.Rs[kk][jj] = r1
        #if vis is not None: vis.refresh(r=r1, repaint=True)
        for ll in range(nlambda):
          alch_lambdas(ctx, lambda_ele[ll], lambda_vdw[ll], lambda_rst[kk])
          state.E_kln[kk,ll,jj] = ctx.getState(getEnergy=True).getPotentialEnergy()/kT
        print(".", end='', flush=True)
      state.idx_samps = 0
      if kk+1 < nlambda:  # Note kBT = 0.6kcal/mol at 300K
        print("\nFEP dF = %.3f * kB*T" % fep(state.E_kln[kk, kk+1] - state.E_kln[kk, kk]))

  except (KeyboardInterrupt, Exception) as e:
    #print("Handling: %r" % e)
    mmstate = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    state.r = mmstate.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
    state.v = mmstate.getVelocities(asNumpy=True).value_in_unit(UNIT.angstrom/UNIT.picosecond)
    state.idx_lambda = kk
    state.idx_samps = jj
    #raise e

  return state  #.E_kln[:,:,warmup:]


def fep_results_full(E_kln, T0, warmup=5):
  """ collection of all out analysis for FEP calculations """
  import pymbar
  beta = 1/(KCALMOL_PER_HARTREE*BOLTZMANN*T0)  # 1/kB/T0 in kcal/mol
  E_kln = E_kln[:,:,warmup:]
  nlambda = len(E_kln)

  # our BAR and FEP calc
  dEup = np.array([E_kln[k, k+1 if k+1 < nlambda else k] - E_kln[k, k] for k in range(nlambda)])
  dEdn = np.array([E_kln[k, k-1 if k-1 >= 0 else k] - E_kln[k, k] for k in range(nlambda)])
  #fep_results(dEup, dEdn, T0)

  # pymbar
  mbar = pymbar.MBAR(E_kln, [len(E[0]) for E in E_kln])  #[nsamps - warmup]*nlambda)
  #dFs_mbar = mbar.getFreeEnergyDifferences(return_dict=True)
  res_mbar = mbar.computeEntropyAndEnthalpy(return_dict=True)
  print("MBAR: dF = %f (%f) kcal/mol; dH = %f kcal/mol; T0*dS = %f kcal/mol" % (res_mbar['Delta_f'][0][-1]/beta,
      res_mbar['dDelta_f'][0][-1]/beta, res_mbar['Delta_u'][0][-1]/beta, res_mbar['Delta_s'][0][-1]/beta))
  print("MBAR steps (kcal/mol): %s" % (np.diff(res_mbar['Delta_f'][0])/beta))

  dFs = np.array([ pymbar.BAR(dEup[ii], dEdn[ii+1]) for ii in range(len(dEup)-1) ])/beta
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


# TODO: modify expel_md to use this fn
# we could try using a CustomCVForce + CustomCentroidBondForce rather than separate fn to compute CV
def com_restraint(mol, lig, bindvec, offset=4.0, kspring=1.6E4):
  force = openmm.CustomCentroidBondForce(1, "A*(d-cv0)^2; d = sqrt((x1-xc)^2+(y1-yc)^2+(z1-zc)^2)")
  force.addPerBondParameter('xc')
  force.addPerBondParameter('yc')
  force.addPerBondParameter('zc')
  force.addGlobalParameter('cv0', offset*0.1)
  force.addGlobalParameter('A', kspring*0.5)

  ligatoms = mol.select(lig)
  origin = center_of_mass(mol, ligatoms) - offset*normalize(bindvec)
  force.addGroup(ligatoms)
  force.addBond([0], origin*0.1)
  force.setForceGroup(31)

  def cvval(mol, r=None):
    return norm(center_of_mass(mol, ligatoms, r) - origin)

  return force, cvval


# see https://github.com/choderalab/pymbar/blob/master/examples/umbrella-sampling-pmf/umbrella-sampling.py
# General collective variables refs: plumed.org, github.com/Colvars/colvars
def umbrella(mol, ff, T0, Rs0, Vs0, cvforce, cvval, steps, freeze=[], state=None, mdsteps=20000, sampsteps=100):
  """ Umbrella sampling """
  if state is None: state = Bunch()
  ctx = openmm_MD_context(mol, ff, T0, freeze=freeze, forces=[cvforce])

  # we could add option to infer range from  Rs0 (steps is int), but extreme min/max might not be good candidates
  dd = np.array([cvval(mol, r) for r in Rs0])
  r0idxs = [np.argmin(np.abs(dd - step)) for step in steps]  # configuration from Rs0 closest to desired one

  ncvsteps = len(steps)
  nsamps = mdsteps//sampsteps
  if state.get('Rs') is True:  # optional recording of positions
    state.Rs = [ [0]*nsamps for ii in range(ncvsteps) ]  #np.zeros((nlambda, nsamps, mol.natoms, 3))
  if 'E_kln' not in state:
    state.E_kln = np.zeros((ncvsteps, ncvsteps, nsamps))
    state.E0_kn = np.zeros((ncvsteps, nsamps))
    state.cv = np.zeros((ncvsteps, nsamps))
  kT = UNIT.AVOGADRO_CONSTANT_NA * UNIT.BOLTZMANN_CONSTANT_kB * T0*UNIT.kelvin  # kB*T in OpenMM units
  try:
    for kk in range(state.get('idx_cvstep', 0), ncvsteps):
      print("%s: running %d MD steps for step %f" % (time.strftime('%X'), mdsteps, steps[kk]))
      idx0 = r0idxs[kk]
      ctx.setPositions(Rs0[idx0]*UNIT.angstrom)
      ctx.setVelocities(Vs0[idx0]*(UNIT.angstrom/UNIT.picosecond))
      for jj in range(nsamps):
        ctx.setParameter('cv0', steps[kk]*0.1)
        ctx.getIntegrator().step(sampsteps)
        r1 = ctx.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)/UNIT.angstrom
        state.cv[kk,jj] = cvval(mol, r1)
        if 'Rs' in state:
          state.Rs[kk][jj] = r1
        E0 = ctx.getState(getEnergy=True, groups=set(range(32)) - {cvforce.getForceGroup()}).getPotentialEnergy()/kT
        state.E0_kn[kk,jj] = E0
        for ll, step in enumerate(steps):
          ctx.setParameter('cv0', step*0.1)
          state.E_kln[kk,ll,jj] = E0 + ctx.getState(
              getEnergy=True, groups={cvforce.getForceGroup()}).getPotentialEnergy()/kT
        print(".", end='', flush=True)
      if kk+1 < ncvsteps:  # Note kBT = 0.6kcal/mol at 300K
        print("\numbrella FEP dF = %.3f * kB*T" % fep(state.E_kln[kk, kk+1] - state.E_kln[kk, kk]))

  except (KeyboardInterrupt, Exception) as e:
    state.idx_cvstep = kk
    print("umbrella terminated by %s: %s" % (e.__class__.__name__, e))

  return state
