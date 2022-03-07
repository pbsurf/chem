import random
from chem.io import *
from chem.io.openmm import *
from chem.model.build import *
from chem.model.prepare import *
from chem.opt.optimize import *
from chem.opt.dlc import HDLC, DLC
from chem.fep import *
from chem.mm import *
from chem.vis.chemvis import *
from chem.data.pdb_data import AA_3TO1, PDB_EXTBACKBONE


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


def relax_bind(mol, E0, ff, host='protein', lig='not protein', active=None, optargs=dict(verbose=-100)):
  optargs = dict(dict(gtol=0.04, maxiter=50, raiseonfail=False, verbose=-2), **optargs)
  r0 = mol.r
  active = mol.select(active) if active is not None else None
  waters = [Rigid(r0, res.atoms, pivot=None) for res in mol.residues if res.name == 'HOH']
  coords = HDLC(mol, [XYZ(r0, active=active, atoms=mol.select('not water'))] + waters) \
      if waters else XYZ(r0, active=active)
  res, r = moloptim(OpenMM_EandG(mol, ff), r0, coords=coords, **optargs)
  return (np.inf if res.fun > E0 + 0.5 else dE_binding(mol, ff, host, lig, r)), r


# perform mutations, replacing worst candidate in population if result is better
# - this makes it possible to accept some mutations that are worse than original, at a cost of loss of diversity
# ... therefore we assume dE_search will be run multiple times for better diversity

def dE_search2(molAI, molAZ, bindAI, bindAZ, nearres, candidates, mutweight, Npop=20, maxiter=200):
  molAIhist, molAZhist, dEhist = [None]*Npop, [None]*Npop, [np.inf]*Npop
  dEmaxidx = 0
  # favor candidates with more rotamers
  candweights = l1normalize([np.ceil(np.sqrt(len(get_rotamers(c)) or 4)) for c in candidates])
  try:
    for ii in range(maxiter):
      if dEhist[dEmaxidx] == np.inf:
        mutAI, mutAZ = Molecule(molAI), (Molecule(molAZ) if molAZ is not None else None)
        mutres = nearres
      else:
        mutAI, mutAZ = Molecule(molAIhist[ii%Npop]), (Molecule(molAZhist[ii%Npop]) if molAZ is not None else None)
        mutres = random.choices(nearres, weights=mutweight(mutAI))

      # mutation
      for resnum in mutres:
        newres = random.choices(candidates, weights=candweights)[0]
        mutate_residue(mutAI, resnum, newres)
        rotamers = get_rotamers('HIE' if newres == 'HIS' else newres)  # fix this
        if not rotamers:
          deAI, r1 = bindAI(mutAI)
        else:
          for rotidx in np.random.permutation(len(rotamers))[:10]:
            angles = random.choice(rotamers)  #360*np.random(len(get_rotamer_angles())
            set_rotamer(mutAI, resnum, angles)
            deAI, r1 = bindAI(mutAI)
            if deAI < 0: break
        if deAI >= 0:  # this residue won't work, abort
          break
        mutAI.r = r1

      if deAI >= 0:  # mutations didn't work, try another candidate
        continue
      if mutAZ is not None:
        for resnum in mutres:
          mutate_residue(mutAZ, resnum, mutAI.extract_atoms(residues=resnum))
        deAZ, mutAZ.r = bindAZ(mutAZ)
      else:
        deAZ = 0

      dE = deAI - deAZ
      accept = deAI < 0 and deAZ <= 0 and dE <= dEhist[dEmaxidx]
      print("%03d: %s: %f - %f = %f%s" % (ii, ''.join(AA_3TO1[mutAI.residues[rn].name] for rn in nearres),
          deAI, deAZ, dE, ' (accept)' if accept else ''))
      if accept:
        dEhist[dEmaxidx] = dE
        molAIhist[dEmaxidx] = mutAI
        molAZhist[dEmaxidx] = mutAZ
        dEmaxidx = max(range(Npop), key=lambda kk: dEhist[kk])
  except KeyboardInterrupt as e:  #(, Exception) as e:
    pass

  ord = argsort(dEhist)
  return Bunch(molAI=[molAIhist[ii] for ii in ord], molAZ=[molAZhist[ii] for ii in ord], dE=[dEhist[ii] for ii in ord])


## ---

T0 = 300  # Kelvin

# print python traceback on segfault
# crash is in glfwCreateWindow -> glfwRefreshContextAttribs -(first call)-> glfwMakeContextCurrent -> previous.context.makeCurrent attempt
#import faulthandler; faulthandler.enable()

# aside: vary width of lines based on depth (or use thin licorice w/ lighting?); play with FOV(?) to increase perspective (Ctrl+D to increase FOV + E to decrease licorice radius) ... definitely worth seeing if we can get a better default or least a useful preset

# force field params for ligands
if not os.path.exists("PRE.mol2"):
  mISJ = load_compound('ISJ').remove_atoms([21,23])
  antechamber_prepare(mISJ, 'ISJ', netcharge=-2)

  # ISJ -> PRE: remove 0--10, add 2--15  (load_compound('PRE') has different atom order and other problems)
  mPRE = Molecule(mISJ)
  mPRE.residues[0].name = 'PRE'
  mPREbonds = mPRE.get_bonds()
  mPREbonds.remove((0,10))
  mPREbonds.append((2,15))
  mPRE.set_bonds(mPREbonds)

  # antechamber does (semi-empirical) QM geom opt, so we need to start from a reasonable geometry
  gaff = load_amber_dat(DATA_PATH + '/amber/gaff.dat')
  gaff_trivial(mPRE)
  set_mm_params(mPRE, gaff, allowmissing=True)
  res, mPRE.r = moloptim(SimpleMM(mPRE), mPRE, maxiter=200)

  antechamber_prepare(mPRE, 'PRE', netcharge=-2)


ff = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'ISJ.ff.xml', 'PRE.ff.xml', nonbondedMethod=openmm.app.PME)
ffgbsa = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'implicit/obc1.xml', 'ISJ.ff.xml', 'PRE.ff.xml', nonbondedMethod=openmm.app.CutoffNonPeriodic)

host, ligin, ligout = '/A,B,C//', '/I//', '/O//'

if os.path.exists("CMwt.zip"):
  molwt_AI, molwt_AO = load_zip("CMwt.zip", 'molwt_AI', 'molwt_AO')
else:
  cm_pdb = load_molecule('2CHT.pdb', bonds=False)
  mol = cm_pdb.extract_atoms('/A,B,C/~TSA,HOH/*')  # active sites are at interface of chains of the trimer
  mol.remove_atoms('*/HIS/HD1')  # 2CHT includes both HD1 and HE2 on all HIS
  #mol.remove_atoms('not protein')
  mol = add_hydrogens(mol)

  # this will place lig at active site between chains A and C
  mISJ = load_molecule('ISJ.mol2')
  mISJ.residues[0].chain = 'I'
  mISJ = align_mol(mISJ, cm_pdb.r[[13539, 13535, 13530]], sel=[6,1,3], sort=None)
  molwt_AI = Molecule(mol)
  molwt_AI.append_atoms(mISJ)
  molwt_AI.r = molwt_AI.r - np.mean(mol.extents(), axis=0)  # move to origin - after appending lig aligned to PDB pos!
  # E&G (ffgbsa): 10s w/ no cutoff; 0.5s w/ default 1nm cutoff
  molwt_AI.dihedral(res_select(molwt_AI, ligin, 'H1,C1,O11,C12'), newdeg=180, rel=True)
  molwt_AI.dihedral(res_select(molwt_AI, ligin, 'C1,O11,C12,C16'), newdeg=180, rel=True)
  #res, mISJ.r = moloptim(OpenMM_EandG(mISJ, ffgbsa), r0=mPRE.r, mol=mISJ, maxiter=200) -- this might be more general
  res, molwt_AI.r = moloptim(OpenMM_EandG(molwt_AI, ffgbsa), molwt_AI, maxiter=100)

  molwt_AO = Molecule(mol)
  mPRE = load_molecule('PRE.mol2')
  mPRE.residues[0].chain = 'O'  #'P'? 'J'?
  mPRE = align_mol(mPRE, cm_pdb.r[[13539, 13535, 13530]], sel=[6,1,3], sort=None)
  molwt_AO.append_atoms(mPRE)
  molwt_AO.r = molwt_AO.r - np.mean(mol.extents(), axis=0)
  res, molwt_AO.r = moloptim(OpenMM_EandG(molwt_AO, ffgbsa), molwt_AO, maxiter=100)

  save_zip('CMwt.zip', molwt_AI=molwt_AI, molwt_AO=molwt_AO)

  openmm_load_params(molwt_AI, ff=ff)
  protonation_check(molwt_AI)
  badpairs, badangles = geometry_check(molwt_AI)

  #molZ = solvate_prepare(mol, ff, T0, neutral=True, solvent_chain='Z')  #, eqsteps=1)
  #save_zip(filename + '_solv.zip', mol=mol, molZ=molZ, v_molZ=molZ.md_vel)  # save MD velocities from solvate_prepare for restart

# aside - protein monomer binding energy
#dE_binding(molwt_AI, ffgbsa, '/A,C//', '/B//')  #-197.9393432346021
#dE_binding(molwt_AI, ffgbsa, '/A,B//', '/C//')  #-194.9137667702444
#dE_binding(molwt_AI, ffgbsa, '/A//', '/B,C//')  #-189.40447204730478


#molZ = solvate_prepare(mol, ff, T0, neutral=True, solvent_chain='Z')  #, eqsteps=0)
#vis = Chemvis([ Mol(molZ, [ VisGeom(style='licorice') ]), Mol(box_triangles(molZ.pbcbox), [VisTriangles()]) ]).run()

# conventions: *wt_*: wild-type; *0_*: nearres -> GLY/ALA; *_AI: host + lig; *_AZ: host + some explicit waters
if os.path.exists('mutate0.zip'):
  nearres, candidates, dE0_AI, mol0_AI = load_zip('mutate0.zip', 'nearres', 'candidates', 'dE0_AI', 'mol0_AI')
else:
  #nearres = molwt_AI.atomsres(molwt_AI.get_nearby(ligin, 4.0, active='/A,B,C/*/~C,N,CA,O,H,HA'))
  #nearres = find_nearres(molwt_AI, '/A,B,C/*/~C,N,CA,O,H,HA', ligin)
  nearres = [5, 88, 288, 292, 304, 303]
  # note the inclusion of ASH and GLH
  #candidates = ['ALA', 'SER', 'VAL', 'THR', 'ILE', 'LEU', 'ASN', 'ASP', 'ASH', 'GLN', 'GLU', 'GLH' 'LYS', 'HIE', 'PHE', 'ARG', 'TYR', 'TRP']
  candidates = ['GLY', 'SER', 'THR', 'LEU', 'ASN', 'ASP', 'ASH', 'CYS', 'MET', 'GLN', 'GLU', 'GLH', 'LYS', 'HIE', 'ARG']
  dEwt_AI = dE_binding(molwt_AI, ffgbsa, host, ligin)  # -23.02561801518386
  dEwt_AO = dE_binding(molwt_AO, ffgbsa, host, ligout)  # -18.004081727063845

  mol0_AI, E0_AI = mutate_relax(molwt_AI, nearres, 'ALA', ff=ffgbsa, optargs=dict(maxiter=100, verbose=-1))
  dE0_AI = dE_binding(mol0_AI, ffgbsa, host, ligin)
  save_zip('mutate0.zip', nearres=nearres, candidates=candidates, dE0_AI=dE0_AI, mol0_AI=mol0_AI)


## individual mutations ... even w/ dE_rot180, unable to get any significant binding, so skip this step
#if os.path.exists('mutate2a.zip'):
#  de1AI, m1AI = load_zip('mutate2a.zip', 'de1AI', 'm1AI')
#  m1AI = np.reshape(m1AI, de1AI.shape)
#else:
#  de1AI, m1AI = dE_mutations(mol0_AI, nearres, candidates, ffgbsa, host, ligin, optargs=dict(gtol=0.04))
#  de1AI -= dE0_AI
#  save_zip('mutate2.zip', de1AI=de1AI, m1AI=np.ravel(m1AI))
#  # for ARG, GLU, HIS, LYS flip CA-CB dihedral 180 deg and try again, pick better result
#  de1AI_old = np.array(de1AI)
#  rotcand = ['GLN', 'GLU', 'LYS', 'HIS', 'ARG']
#  de1AI, m1AI = dE_rot180(mol0_AI, dE0_AI, de1AI, m1AI, nearres, rotcand, ffgbsa, host, ligin, optargs=dict(gtol=0.04))
#  save_zip('mutate2a.zip', de1AI=de1AI, m1AI=np.ravel(m1AI))


ffp = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'ISJ.ff.xml', 'PRE.ff.xml', ignoreExternalBonds=True)
ffpgbsa = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'implicit/obc1.xml', 'ISJ.ff.xml', 'PRE.ff.xml', ignoreExternalBonds=True)
#removeCMMotion=False


## create system of just residues w/in small radius (~8 Ang) of lig for faster MM calc
ligres = mol0_AI.atomsres(ligin)[0]
#nearres8 = sorted(find_nearres(mol0_AI, '/A,B,C/*/~C,N,CA,O,H,HA', ligin, 8.0))
nearres8 = mol0_AI.atomsres(mol0_AI.get_nearby(ligin, 8.0, active='/A,B,C/*/~C,N,CA,O,H,HA'))
nearres, nearres_full = [ii for ii,rn in enumerate(nearres8) if rn in nearres], nearres


# start interactive
import pdb; pdb.set_trace()

# Try dE_search2, w/ an eye toward where metadynamics might be useful
# - very low probability of random set of residues fitting ... identify problematic residues and try different rotatmer?
#  - add the residues one at a time? try different rotamers upon failure? - this might be better since multiple residues could be issue
# - start w/ smaller residues?


molp_AI = mol0_AI.extract_atoms(resnums=nearres8 + [ligres])
molp_AZ = None

nearres4 = molp_AI.atomsres(molp_AI.get_nearby(ligin, 4.0, active='/A,B,C/*/~C,N,CA,O,H,HA'))
ligatoms = molp_AI.select(ligin)
freeze = molp_AI.select('protein and (extbackbone or resnum not in %r)' % nearres4)

def mutate_weights(mol):
  # get electrostatic energy in Hartree between each nearres sidechain and lig
  if not hasattr(mol, 'sc_Eqq'):
    openmm_load_params(mol, ff=ffpgbsa)
    mol.sc_Eqq = sidechain_Eqq(mol, nearres, ligin)
    print("sidechain_Eqq: ", mol.sc_Eqq)
  # prefer residues w/ poor Eres for mutation
  return l1normalize((mol.sc_Eqq > -0.1) + 0.5)


if os.path.exists('search1.zip'):
  res = load_zip('search1.zip')
else:
  Ep_AI, _ = openmm_EandG(molp_AI, ff=ffpgbsa, grad=False)
  #Ep_AZ, _ = openmm_EandG(molp_AZ, ff=ffpart, grad=False)
  #active = '(not protein and not water) or (resnum in %r and not extbackbone)' % nearres
  active = '(resnum in %r and not extbackbone) or not protein' % nearres4
  bindAI = partial(relax_bind, E0=Ep_AI, ff=ffpgbsa, active=active)
  bindAZ = None  #partial(relax_bind, E0=Ep_AZ, ff=ffpgbsa, active=active)
  res = dE_search2(molp_AI, molp_AZ, bindAI, bindAZ, nearres, candidates, mutate_weights)
  save_zip('search1.zip', **res)  # save results


## NEXT:
# - MD calc for dE_search2 results - w/ lig and w/ waters
#  - binding energy for waters
#  - then try computing binding energy by pulling lig out of host

# - try pulling into site w/ implicit solvent - less variation in orientation?
# - compute binding free energy from our poor excuse for metadynamics
#  - https://iris.sissa.it/retrieve/handle/20.500.11767/110289/124927/main.pdf ?

bindvec = center_of_mass(molwt_AI, molwt_AI.select(ligin)) - center_of_mass(molwt_AI, molwt_AI.select(host))

# vector from COM of host to COM of lig (to approximate normal vector of binding site)
for mol0 in res.molAI:
  center = np.mean(mol0.extents(), axis=0) + 5.0*normalize(bindvec)

  mol = solvate_prepare(mol0, ffp, pad=6.0, center=center, solvent_chain='Z', neutral=True, eqsteps=None)
  ligatoms = mol.select(ligin)
  freeze = mol.select('protein and (extbackbone or resnum not in %r)' % nearres4)

  ctx0 = openmm_MD_context(mol, ffp, T0, freeze=freeze, freezemass=0, constraints=None, nonbondedMethod=openmm.app.PME, rigidWater=True)
  openmm.LocalEnergyMinimizer.minimize(ctx0, maxIterations=200)
  mol.r = ctx0.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)/UNIT.angstrom

  ctx = openmm_MD_context(mol, ffp, T0, freeze=freeze)
  Rs, Es = openmm_dynamic(ctx, 20000, 100)


# For explicit waters to be useful, we want to limit somehow to waters that would be displaced by lig (and
#  keep waters from leaving site, as happened w/ simple relaxation)
# Options/ideas:
# - try relaxation with restraint holding (few) waters near site
# - generate multiple candidates (w/ dE_search) using just lig, then do MD w/ and w/o lig in explicit water for each
# - dE_mutations: start w/ lig unbound, pull into site ... substantial computation time, so explore other avenues first


## crossover
# - not clear this adds much - less than 1 kcal/mol improvement (dE_search2 results were already pretty good)
# - might try random choice of crossover residues (i.e. w/o grouping)

# - then we need to see what fraction of cross-over offspring are better than both, one, or neither parent(s)?
# Then decide how to make use of cross-over:
# - generational: create new generation w/ 2 offspring per couple, then mutations, plus best parents (replacing worst offspring)
# - or? replace worse parent with offspring?

def randsplit(mol, nearres):
  rCA = np.array([ mol.atoms[res_select(mol, resnum, 'CA')[0]].r for resnum in nearres ])
  drCA = rCA[:,None,:] - rCA[None,:,:]
  resprox = np.argsort(np.sum(drCA*drCA, axis=2))  #ddCA = np.sqrt(np.sum(drCA*drCA, axis=2))
  idxsplit = random.randrange(len(nearres))
  nsplit = random.randrange(2, len(nearres)-1)  # at least two
  splitres = [ nearres[ii] for ii in resprox[idxsplit] ]
  return splitres[:nsplit]


def crossover(mol1, mol2, splitres):
  if type(mol1) is Molecule:
    mol1, mol2 = [mol1], [mol2]
  mut1, mut2 = [Molecule(mol) for mol in mol1], [Molecule(mol) for mol in mol2]
  for resnum in splitres:
    for ii in range(len(mut1)):
      mutate_residue(mut1[ii], resnum, mol2[ii].extract_atoms(resnums=resnum))
      mutate_residue(mut2[ii], resnum, mol1[ii].extract_atoms(resnums=resnum))
  return (mut1[0], mut2[0]) if len(mut1) == 1 else (mut1, mut2)


seq = lambda mol: ''.join(AA_3TO1[mol.residues[rn].name] for rn in nearres)
xres = []
for it in range(200):
  ii,jj = random.sample(range(len(res.molAI)), k=2)
  splitres = randsplit(res.molAI[ii], nearres)
  mx1, mx2 = crossover(res.molAI[ii], res.molAI[jj], splitres)
  dEx1, mx1.r = bindAI(mx1)
  dEx2, mx2.r = bindAI(mx2)
  xres.append( (ii, jj, splitres, dEx1, dEx2) )
  print("%s (%f) / %s (%f) -> %s (%f) / %s (%f) (split: %r)" % ( seq(res.molAI[ii]), res.dE[ii], seq(res.molAI[jj]), res.dE[jj], seq(mx1), dEx1, seq(mx2), dEx2, splitres ))
save_zip('crossover1.zip', xres)

sorted([min(dEx1, dEx2) - min(res.dE[ii], res.dE[jj]) for ii, jj, splitres, dEx1, dEx2 in xres])  # -> -6 kcal/mol


## pull lig out of host, and push back in

# vector from COM of host to COM of lig (to approximate normal vector of binding site)
bindvec = center_of_mass(molwt_AI, molwt_AI.select(ligin)) - center_of_mass(molwt_AI, molwt_AI.select(host))

molpwt_AI = molwt_AI.extract_atoms(resnums=nearres8 + [ligres])
nearres4 = molpwt_AI.atomsres(molpwt_AI.get_nearby(ligin, 4.0, active='/A,B,C/*/~C,N,CA,O,H,HA'))
center = np.mean(molpwt_AI.extents(), axis=0) + 5.0*normalize(bindvec)  #[-1, 1, -1])

mol = solvate_prepare(molpwt_AI, ffp, pad=6.0, center=center, solvent_chain='Z', neutral=True, eqsteps=None)
ligatoms = mol.select(ligin)
freeze = mol.select('protein and (extbackbone or resnum not in %r)' % nearres4)

ctx0 = openmm_MD_context(mol, ffp, T0, freeze=freeze, freezemass=0, constraints=None, nonbondedMethod=openmm.app.PME, rigidWater=True)
openmm.LocalEnergyMinimizer.minimize(ctx0, maxIterations=200)
mol.r = ctx0.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)/UNIT.angstrom

# openmm.app.metadynamics couldn't handle this (OOM - tabluated fn table too big?)

## Gaussian w/ incr variance to expel from site ... I think flat top Gaussian (below) is better
# - what about using dist from ref point as CV - how would that be different?
# note CustomCentroidBondForce actually computes mass-weighted centroid, i.e. center of mass
#periodicdistance(x1,y1,z1, xc,yc,zc)^2  #min(t_meta/10.0, 1.0)*  #; std = min(t_meta/t_tot, 1.0)*1.4; t_tot = 200.0
force = openmm.CustomCentroidBondForce(1, 'A*exp(-((x1 - xc)^2 + (y1 - yc)^2 + (z1 - zc)^2)/(2*std^2)); A = 800.0')

force.addPerBondParameter('xc')
force.addPerBondParameter('yc')
force.addPerBondParameter('zc')
force.addGlobalParameter('std', 1E-14)  # t_meta = 0 gives NaN for force

com = center_of_mass(mol, ligatoms)
force.addGroup(ligatoms)
force.addBond([0], (com - 5.0*normalize(bindvec))*0.1)  # offset center 5A from COM; convert Angstrom to nm
force.setForceGroup(31)

sys = ffp.createSystem(openmm_top(mol), constraints=openmm.app.HBonds, nonbondedMethod=openmm.app.PME, rigidWater=True)
sys.addForce(force)
ctx = openmm_MD_context(mol, sys, T0, freeze=freeze)

#metafn = lambda ii,ctx: ctx.setParameter('t_meta', max(1E-14, ii-10))
t_ramp = 200.0
max_w = 1.4  # nm
Emeta = []
def metafn(ii, ctx):
  ctx.setParameter('std', min(max(1E-14, ii-10)/t_ramp, 1.0)*max_w)
  Emeta.append( ctx.getState(getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy()/EUNIT )

Rs, Es, Vs = openmm_dynamic(ctx, 20000, 100, vel=True, sampfn=metafn)

#save_zip('solv_wt2.zip', mol=mol, Rs=Rs, Vs=Vs, Es=Es)


ctx.setVelocities(Vs[100]*(UNIT.angstrom/UNIT.picosecond))
ctx.setPositions(Rs[100]*UNIT.angstrom)

t_ramp = 200.0
max_w = 0.9  # nm
revfn = lambda ii,ctx: ctx.setParameter('std', min(max(1E-14, ii-10)/t_ramp, 1.0)*max_w)
Rs, Es = openmm_dynamic(ctx, 20000, 100, sampfn=revfn)
# ... looks like we'll need to add a force to push back into site


## flat top Gaussian (inverted to pull into site
force = openmm.CustomCentroidBondForce(1, "A*(1 - exp(-max(0, d-width)^2/(2*std^2))); d = sqrt((x1-xc)^2+(y1-yc)^2+(z1-zc)^2); A = 800.0; std = 0.1")
# smoothstep: "A * x*x*(3 - 2*x); x = min(max((r - edge0)/(edge1 - edge0), 0.0), 1.0)"

force.addPerBondParameter('xc')
force.addPerBondParameter('yc')
force.addPerBondParameter('zc')
force.addGlobalParameter('width', 0)

com = center_of_mass(mol, ligatoms)
force.addGroup(ligatoms)
force.addBond([0], (com - 5.0*normalize(bindvec))*0.1)  # offset center 5A from COM; convert Angstrom to nm
force.setForceGroup(31)

#force.setBondParameters(0, [0], (com + 11.0*normalize(bindvec))*0.1)
#force.updateParametersInContext(ctx)
sys = ffp.createSystem(openmm_top(mol), constraints=openmm.app.HBonds, nonbondedMethod=openmm.app.PME, rigidWater=True)
sys.addForce(force)
ctx = openmm_MD_context(mol, sys, T0, freeze=freeze)

t_ramp = 200.0
revfn = lambda ii,ctx: ctx.setParameter('width', 1.0 - min(max(0, ii-10)/t_ramp, 1.0)*0.5)

ctx.setParameter('width', 1.0)
ctx.setVelocities(V0)
ctx.setPositions(R0)

Rs, Es = openmm_dynamic(ctx, 20000, 100, sampfn=revfn)

#save_zip('solv_wt3.zip', mol=mol, Rs1=Rs1, Rs2=Rs2, Rs3=Rs3, Rs4=Rs)

dE_binding(mol, ffpgbsa, host, ligin, Rs[-1])

notwat = mol.select('not water')
res, r_opt = moloptim(OpenMM_EandG(molpwt_AI, ffpgbsa), Rs[-1][notwat], maxiter=50)
dE_binding(molpwt_AI, ffpgbsa, host, ligin, r_opt)


## OK, this seems reasonable ... should we try this with individual residues (i.e. all others ALA)?
# - but with less bulk in active site, lig can take almost any orientation


#~ ctx.setParameter('t_meta', 100.0)
#~ ctx.getState(getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy()/EUNIT
#~ ctx.setParameter('t_meta', 0)


## exploration w/ explicit waters
# if this doesn't work, try random placement as in solvate()
# ... maybe add a restraining potential to keep near lig pos?
if 0:
  molp_AI = mol0_AI.extract_atoms(resnums=nearres8 + [ligres])
  molp_AZ = molp_AI.extract_atoms('/~I//')
  tip3p = load_molecule(TIP3P_xyz, center=True, residue='HOH')
  for ii in molp_AI.select("chain == 'I' and znuc > 1"):
    molp_AZ.append_atoms(tip3p, r=np.dot(tip3p.r, random_rotation()) + molp_AI.atoms[ii].r)

  r0 = molp_AZ.r
  coords = HDLC(molp_AZ, [Rigid(r0, res.atoms, pivot=None) for res in molp_AZ.residues if res.name == 'HOH'] + [XYZ(r0, active=[], atoms=molp_AZ.select('not water'))])
  res, r1 = moloptim(OpenMM_EandG(molp_AZ, ffpart), mol=molp_AZ, coords=coords, maxiter=200)


molpwt_A = molwt_AI.extract_atoms(resnums=nearres8 + [ligres])  #[:-1])
mol = solvate_prepare(molpwt_A, ffp, pad=6.0, solvent_chain='Z', neutral=True, eqsteps=None)

molpwt_I = molwt_AI.extract_atoms(ligin)
rI = molpwt_I.r - np.mean(molpwt_A.extents(), axis=0) + 0.5*mol.pbcbox


freeze = mol.select('protein and (extbackbone or resnum not in %r)' % nearres)
ctx0 = openmm_MD_context(mol, ffp, T0, freeze=freeze, freezemass=0, constraints=None, nonbondedMethod=openmm.app.PME, rigidWater=True)
openmm.LocalEnergyMinimizer.minimize(ctx0, maxIterations=200)
mol.r = ctx0.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)/UNIT.angstrom
ctx = openmm_MD_context(mol, ffp, T0, freeze=freeze, nonbondedMethod=openmm.app.PME, rigidWater=True)
Rs, Es = openmm_dynamic(ctx, 10000, sampsteps=100)
#save_zip('solv_wt1.zip', mol=mol, Rs=Rs, Es=Es)

# openmm.LocalEnergyMinimizer is much faster ... also, rigid waters required for successful opt
# Rigid is slow with lots of waters - maybe look into using a single coord object for multiple rigid molecules
if 0:
  r0 = mol.r
  coords = HDLC(mol, [Rigid(r0, res.atoms, pivot=None) for res in mol.residues if res.name == 'HOH'] + [XYZ(r0, active=[], atoms=mol.select('not water'))])
  res, r1 = moloptim(OpenMM_EandG(mol, ffp, nonbondedMethod=openmm.app.PME), r0, coords=coords, maxiter=200)

# now get waters overlapping with lig
kd = cKDTree(rI)
ow = mol.select('* HOH O')
for r1 in Rs:
  dists, locs = kd.query(r1[ow], distance_upper_bound=6.0)
  waters = [mol.atoms[ow[ii]].resnum for ii,dist in enumerate(dists) if dist < 3.0]
  dE_wat = dE_binding(mol, ffpgbsa, host, [a for w in waters for a in mol.residues[w].atoms], r1)
  print("%d waters, dE = %f" % (len(waters), dE_wat))

# ... this looks reasonable - I think mean dE_binding for MD run could be useful, but looks like we do need a few thousand steps for water to fully fill site


## MORE
# - we need to set up a system with PRE!
# - explicit solvent w/o lig (for near waters) ... perhaps we should use all-GLY for this?
# ... should we populate all 3 sites to compare?
# - some way to penalize voids more strongly (might help single point calc get closer to MD result)? First step would be notes on GBSA method
## - find scaling factor which gives minimum RMSD between positions in explicit vs. implicit solvent MD, then maybe see what force needs to be added to implicit solvent case to match

# - save_zip: save string as .txt instead of .pkl
# - test multithreading for pyscf
# - should add a "charge" attribute to Molecule instead of having to keep track separately?
# - automated GAFF protonation check?


# how to create objective fn combining reactant and product ligs? (and later TS?)
# - don't want to just optimize difference between reactant and product
# - thresholds (min for reactant vs waters, max for product vs waters)
#  - but in general, we won't know what values are achieveable to help choose thresholds!
# specifically, w/ dE_search algo, how to decide whether to keep candidate if one value improves and the other worsens?
# - formula like ddE_AI**2 - ddE_AO**2?
# - require improvement in both? accept improvement in only one if other still meets threshold?
# - we need a single scalar if we want to sort candidates (e.g., so we can replace "worst" one)
# - rethink algorithm more broadly?
# - possibility to accept regression using Metropolis rule?

## - optimize for reactant and product separately, then ...
#  - pick nearest candidates from the two sets and ... ?


## Old

## maybe we shouldn't give up on deAI so easily - greatly simplifies the problem if it works!
# - start with lig slightly displaced, possibly with force pulling toward nearres (toward CAs?)
# - steered MD/umbrella sampling (or just local opt w/ pulling force) simulation of lig displacing waters?

# - maybe try some other systems to rebuild confidence in deAI, then return to CM

## w/o deAI (binding energies for individual residues), we need something else to guide dE_search
## ... let's cut and paste our new dE_search instead of trying to dedup
# - use genetic cross-over
# - get per-residue energies (see trypsin3.py) and favor changing residues w/ poor binding energy
#  - start with set of residues inited w/ mm params so we don't need to get params every time; only calc qq energy for lig and active residue sidechains
#  - combine residues w/ good binding energy in crossover?

# - try to shift position of lig by favoring large change in residue mass

# - almost naive global search: small mutations from best solns so far
# - do we know anything about structure of problem that will allow us to go beyond this?
#  - some level of independence between residues: so preferring changes to poorly binding residues, copying residues between candidates, etc. may be useful
#  -

# - total independence: just optimize separately
# - no independence (i.e. no correlation between individual residues and binding energy)
#  - try to find pattern ... e.g. pairs of residues? machine learning? try to extract pair correlations?  I think any correlations would be weak or more complicated
# ... I think cross-over favoring grouping of nearby residues is a reasonable approach here


# currently only interaction between dE_search candidates is favoring of more common residues
# - genetic algo cross-over
# - favor iterating on worst candidate?  (current approach may throw away diversity by replacing other candidates w/ minor variations of best candidate, e.g. mutations of residues w/ minimal effect on binding)
#  - but give up after ~5 iterations and move on to second worst (so keep list of active candidates sorted)


## how to integrate rotamers?
# - just pick random rotamer when mutating?
# - if we're using different rotamers, stats aggregated by residue type may not be very useful
# - should we give up keeping track of theos already seen?

# - we want to avoid replacing every candidate w/ a slightly modified version of the one that happens to be better initially
#  -

# what are we going to allow to move in partial system? only sidechains of active residues? all sidechains?
# - what constraints would we need to allow anything more to move?


# - test: adjust position to give more room: pass center or offset to solvate_prepare? ... done
#  - then run MD again ... done
# - see if we shut off expulsion force and continue simulation lig gets pulled back into site ... no
#  - if not, see if we can push back into site (reverse potential?) ... done, works
# - see if we get desired nearres conformations by doing this ... lig orientation varies greatly between runs; ARG A 7 does basically get forced into wild-type position as lig is pulled in
#  - if not, determine position of lig where nearres conformation is "locked-in" and ... ? go slower there?

revfn = lambda ii,ctx: ctx.setParameter('t_meta', max(1E-14, (ii if ii < 100 else (200-ii))-10))
Rs, Es = openmm_dynamic(ctx, 20000, 100, sampfn=revfn)

# - consider:
#  - favorable active site interaction w/ water: good dE_AZ
#  - less favorable: water moves out of active site and finds a different favorable place -> still good dE_AZ

# per-residue electrostatic interaction ... seems reasonable
openmm_load_params(molwt_AI, ff=ffgbsa, charges=True, vdw=True, bonded=False)
sidechain_Eqq(molwt_AI, nearres, ligin)


# OpenMM does not allow PME w/ implicit solvent
mol = Molecule(mol0_AI)
mol.pbcbox = np.array([np.max(np.diff(mol.extents(pad=10), axis=0))]*3)
ffgbsa.sysargs['nonbondedMethod'] = openmm.app.CutoffNonPeriodic  #PME
#ffgbsa.sysargs['nonbondedCutoff'] = 15*UNIT.angstrom
dE_binding(mol, ffgbsa, host, ligin)


# difference here vs. trypsin case is that none of the individual residues have a significant effect (and even if they did, they might not have significant effect on difference between reactant and product due to orientational freedom)
# ... NOT TRUE - ARG A 7 at least should have effect, but we can't position correctly
# - so just skip to dE_search


# what about VAL instead of ALA to get closer to "average" sidechain size?

#molALA_AI, eALA_AI = mutate_relax(molwt_AI, nearres, 'ALA', ff=ffgbsa, optargs=dict(maxiter=100, verbose=-1))
#molVAL_AI, eVAL_AI = mutate_relax(molwt_AI, nearres, 'VAL', ff=ffgbsa, optargs=dict(maxiter=100, verbose=-1))
#rmsdALA = calc_RMSD(molALA_AI, molwt_AI, 'backbone', align=True) ... 0.11
#rmsdVAL = calc_RMSD(molVAL_AI, molwt_AI, 'backbone', align=True) ... 0.10
#vis = Chemvis(Mol([molwt_AI,molALA_AI, molVAL_AI], [ VisGeom(style='lines', sel=host),  VisGeom(style='spacefill', sel=ligin), VisGeom(style='licorice', sel=('resnum in %r' % nearres)) ]), fog=True).run()


# rotate CA-CB dihedral of single-mutation candidates in deAI,mutAI by 180 deg and relax; choose conformation
#  with better dE_binding
def dE_rot180(molAI, dE0_AI, deAI, mutAI, nearres, rotcand, ff, host, lig, optargs={}):
  e0AI, _ = openmm_EandG(molAI, ff=ff, grad=False)
  for ii,resnum in enumerate(nearres):
    for jj in range(len(mutAI[ii])):
      resname = mutAI[ii][jj].residues[resnum].name
      if resname not in rotcand:
        continue
      mol = Molecule(mutAI[ii][jj])
      set_rotamer(mol, resnum, [180], rel=True)
      mutAI2, eAI2 = mutate_relax(mol, [], [], ff=ff, optargs=optargs)  # host + lig
      if eAI2 > e0AI + 0.5:  #0 or ebmaxAI > 0.05:  # clash
        print("%d -> %s failed with E = %f" % (resnum, resname, eAI2))
        continue
      deAI2 = dE_binding(mutAI2, ff, host, lig) - dE0_AI
      print("%s %d: %f -> %f" % (resname, resnum, deAI[ii,jj], deAI2))
      if deAI2 < deAI[ii,jj]:
        mutAI[ii][jj] = mutAI2
        deAI[ii,jj] = deAI2

  return deAI, mutAI


def dE_rotamers(molAI, resnum, rotamers, ff, host=None, lig=None, optargs={}):
  deAI = [np.inf]*len(rotamers)
  mutAI = [None]*len(rotamers)
  e0AI, _ = openmm_EandG(molAI, ff=ff, grad=False)
  mol = Molecule(molAI)
  for jj,angles in enumerate(rotamers):
    set_rotamer(mol, resnum, angles)
    mutAI[jj], eAI = mutate_relax(mol, [], [], ff=ff, optargs=optargs)  # host + lig
    if eAI > e0AI + 0.5:
      print("%d -> %s failed with E = %f" % (resnum, resname, eAI))
      continue
    if host is not None and lig is not None:
      deAI[jj] = dE_binding(mutAI[jj], ff, host, lig)
  return deAI, mutAI


## this code was attempt to see what was necessary to recover wild-type rotamer of ARG A 7
# - result was that wild-type rotamer was only recovered when lig position and other residues matched wild-type, but fortunately in this case the wild-type rotamer did give the lowest overall energy

# let's try with wild-type!
# ... finally, lowest energy config is wild-type config!
# ... but this depends on the precise position of the lig! won't work w/ other residues mutated and lig in different position

mol = molwt_AI
openmm_load_params(mol, ff=ff, charges=True, vdw=True, bonded=True)
A7 = mol.select('/A/7/*')
molA7 = mol.extract_atoms(A7)
nearres8 = find_nearres(mol, '/A,B,C/*/~C,N,CA,O,H,HA;/I//', A7, 8.0)
active = sorted([a for res in nearres8 for a in mol.residues[res].atoms])
activeA7 = [ii for ii,a in enumerate(active) if a in A7]
molactive = mol.extract_atoms(active)

sys = openmm_create_system(mol, active, bondactive=mol.get_connected(A7, 3))  #, bondactive=A7)

import itertools, random
torscan = list(itertools.product(np.arange(0, 360, 5), np.arange(0, 360, 5), np.arange(0, 360, 15), np.arange(0, 360, 30)))
random.Random(5).shuffle(torscan)  # shuffles in place
r_active = mol.r[active]
EandG = OpenMM_EandG(mol, sys)
coords = XYZ(r_active, active=activeA7)
result = []
r0_A7 = molA7.r
for ii,rot in enumerate(torscan):
  molA7.r = r0_A7  # prevent accumulation of numerical errors w/ repeated use of set_rotamer
  res, ropt = moloptim(EandG, S0=set_rotamer(molA7, 0, rot).r, coords=coords, maxiter=50, verbose=-100)
  result.append( (res.fun, res.x.reshape(-1,3)) )  #ropt[activeA7])

#save_zip('relaxA7.zip', result=result)  ... w/o lig
save_zip('relaxA7wtI.zip', result=result)

from chem.theo import cluster_poses
EsA7, RsA7 = np.array([ee for ee,_ in result]), np.array([ss for _,ss in result])
A7sc = molA7.select('* * ~C,N,CA,O,H,HA')
Ec, Rc = cluster_poses(EsA7[EsA7 < -1.0], RsA7[EsA7 < -1.0], A7sc, thresh=1.2)
Rs = np.array([ setitem(np.array(r_active), activeA7, rc) for rc in Rc ])

hbond = [calc_dist(rs[69], rs[256]) for rs in Rs]


  #r[activeA7] = set_rotamer(molA7, 0, rot).r
  #result.append( (rot, EandG(r, grad=False)[0]) )
  #if ii > 0 and ii%50000 == 0:
  #  save_zip('rotamerA7_%d.zip' % (ii//50000), result=result)


# how to visualize? volume vis? don't bother and just look at minima?
# - how to convert values at "randomly" placed points to values on grid?  for each grid point, get N nearest points and compute weighted average
# ... this is "interpolation on unstructured grid" - use scipy.interpolate.griddata()

Es = np.array([b for a,b in result])
Esort = np.argsort(Es)
Rs = [ setitem(np.array(r), activeA7, set_rotamer(molA7, 0, result[ii][0]).r) for ii in Esort[:10] ]

molactive = mol.extract_atoms(active)


# we'll need to do cluster_poses ... and two stages at that!
from chem.theo import cluster_poses

Rs = np.array([ set_rotamer(molA7, 0, result[ii][0]).r for ii in Esort[:10000] ])
A7sc = molA7.select('* * ~C,N,CA,O,H,HA')
cl1 = [ cluster_poses(Es[Esort[ii:ii+1000]], Rs[ii:ii+1000], A7sc, thresh=0.9) for ii in range(0, len(Rs), 1000) ]
Ec1, Rc1 = np.array([e for ee,_ in cl1 for e in ee]), np.array([s for _,ss in cl1 for s in ss])
Ec, Rc = cluster_poses(Ec1, Rc1, A7sc, thresh=1.2)
print(len(Rc))
RRs = [ setitem(np.array(r), activeA7, rc) for rc in Rc ]
# still not finding the desired pose among 10000 lowest energy poses
# - repeat w/o bonded energy terms?
# - relax ... let's try this
# - explicit test hydrogen bonds ... how?

# compare energy (differences) to calc w/ full molecule (this will check for errors in params as well as validity of approximation)
rall = mol.r
EandGall = OpenMM_EandG(mol, ff.ff)
Ers = np.array([ EandGall( setitem(np.array(rall), active, rs), grad=False )[0] for rs in RRs ])
print(Ers - Ec - (np.mean(Ers - Ec)))  # ... looks reasonable

wt5deg = np.array(get_rotamer_angles(molwt_AI, 5))*180/np.pi
r_wt5 = setitem(np.array(r), activeA7, set_rotamer(molA7, 0, wt5deg).r)




molA = m1AI[0][-1].extract_atoms('* ~ISJ *')
de5A, m5A = dE_rotamers(molA, 5, list(itertools.product([-90,90], repeat=4)), ffgbsa, optargs=dict(gtol=0.04))  #host, ligin,


wtres = [molwt_AI.residues[ii].name for ii in nearres]
mol0wt_AI, _ = mutate_relax(mol0_AI, nearres, wtres, ff=ffgbsa, optargs=dict(maxiter=100, verbose=-1))
dE0wt_AI = dE_binding(mol0wt_AI, ffgbsa, host, ligin)

m1AI_wt = [ m1AI[ii][candidates.index(resname)] if resname != 'ALA' else mol0_AI for ii,resname in enumerate(wtres)  ]
res1_wt = [ m.extract_atoms(resnums=nearres[ii]) for ii,m in enumerate(m1AI_wt) ]
mol1wt_AI, _ = mutate_relax(mol0_AI, nearres, res1_wt, ff=ffgbsa, optargs=dict(maxiter=100, verbose=-1))
dE1wt_AI = dE_binding(mol1wt_AI, ffgbsa, host, ligin)

# mutate wild type residues to ALA except for 1 and calc dE_binding ... ARG A 7 main contribution (-17) as expected
for ii,resnum in enumerate(nearres):
  mut0wt_AI = Molecule(molwt_AI)
  for rn in nearres[:ii] + nearres[ii+1:]:
    _ = mutate_residue(mut0wt_AI, rn, 'ALA')
  #mut0wt_AI, _ = mutate_relax(mol0wt_AI, nearres[:ii] + nearres[ii+1:], 'ALA', ff=ffgbsa, optargs=dict(gtol=0.04))
  dEmutwt_AI = dE_binding(mut0wt_AI, ffgbsa, host, ligin)
  print("%d %s: %f" % (resnum, mut0wt_AI.residues[resnum].name, dEmutwt_AI))



# ... still not getting to WT configuration ... should we try flipping diheds w/ complete site instead of single residues?
# - I think the fundamental issue is that ARG A 7 is held in position by hydrogen bond from HE to a backbone O (VAL A 91) - naive optimization will not be able to overcome the barrier (for bonded terms) to reach this position
#  - try more rotamers? ... didn't work ... need even more?

# ... ok, at least if we set WT rotamer manually it works ... would removing torsional barriers be sufficient
# - generate allowed positions (i.e. entire rotamer space) and constrain to that w/o barriers
# - just constrain based on distance (from CA)?
##  - theozyme w/ sidechain fragments constrained to max distance from set of positions?
#  - w/ N positions and constraint on max weight
# - strengthen interaction w/ lig?
# - clear dihedral bonded terms?
# - scan all torsions to get better idea of PES? complicated by cutoff - discard distant residues?
#  - how would we use this?  torsion scan for heavy residues for each position (w/ and/or w/o lig) ... replacing relaxation (or run relaxation near potential minima); could we use grad to interpolate values?  analytic calc of the Jacobian needed would be messy

# - (re)create forces including only active atoms (atoms of active residues)
# - use add_hydrogens to cap each peptide fragment



# in dE_search, include e.g. ARGX, ARG180, ARG', or ??? to indicate mutation is flipping CA-CB dihed instead of changing residue
# - always recorded as, e.g. ARG?

if 0:
  mol = add_residue(Molecule(), 'GLY_LFZW', -180, 135)
  mol = add_residue(mol, 'ARG_LFZW', 180, 180)
  mol = add_residue(mol, 'GLY_LFZW', -135, -180)
  openmm_load_params(mol, ff=ff, vdw=True, bonded=True)

  import itertools
  Rs,Es = [],[]
  for angles in itertools.product([-90,90], repeat=4):  #for angles in get_rotamers(mol.residues[1].name):
    _ = set_rotamer(mol, 1, angles)
    Rs.append(mol.r)
    Es.append(mmbonded(mol)[0])

  vis = Chemvis(Mol(mol, Rs, [VisGeom()])).run()

  # clustering doesn't seem to be picking the ones we want
  from chem.theo import cluster_poses
  _, Rc = cluster_poses(np.array(Es), np.array(Rs), mol.select('//NE,CZ'), thresh=0.9)  # '//CZ,NE'


mol = add_residue(Molecule(), 'GLY_LFZW', -180, 135)
mol = add_residue(mol, 'ARG_LFZW', 180, 180)
mol = add_residue(mol, 'GLY_LFZW', -135, -180)
Rs = [ set_rotamer(mol, 1, angles).r for angles in get_rotamers('ARG') ]
vis = Chemvis(Mol(mol, Rs, [VisGeom()])).run()
