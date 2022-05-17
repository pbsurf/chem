from chem.io import *
from chem.mmtheo import *
from chem.vis.chemvis import *


## ---

T0 = 300  # Kelvin

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

ffp = OpenMM_ForceField(ff.ff, ignoreExternalBonds=True)
ffpgbsa = OpenMM_ForceField(ffgbsa.ff, ignoreExternalBonds=True)  #removeCMMotion=False

host, ligin, ligout, ligall = '/A,B,C//', '/I//', '/O//', '/I,O//'

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


## create system of just residues w/in small radius (~7 Ang) of lig for faster MM calc
# - this looked slighly better than using residues w/in 2.8 Ang of nearres4 sidechains, since that got a bunch
#  of sidechains pointing away from lig - not helpful in constraining nearres
ligreswt = molwt_AI.atomsres(ligin)[0]
nearres7 = molwt_AI.atomsres(molwt_AI.get_nearby(ligin, 7.0, active='protein'))
nearres4 = molwt_AI.atomsres(molwt_AI.get_nearby(ligin, 4.0, active='sidechain'))  # 4 Ang
molpwt_AI = molwt_AI.extract_atoms(resnums=nearres7 + [ligreswt])
nearres = molpwt_AI.atomsres(molpwt_AI.get_nearby(ligin, 4.0, active='sidechain'))  # 4 Ang

ligatoms = molpwt_AI.select(ligin)
ligres = molpwt_AI.atomsres(ligatoms)[0]
freeze = 'protein and (extbackbone or resnum not in %r)' % nearres

# vector from COM of host to COM of lig (to approximate normal vector of binding site)
bindvec = center_of_mass(molwt_AI, molwt_AI.select(ligin)) - center_of_mass(molwt_AI, molwt_AI.select(host))
# center to use of PBC w/ partial molecule system
pcenter = np.mean(molpwt_AI.extents(), axis=0) + 5.0*normalize(bindvec)


## dE_search2

def mutate_weights(mol, ff, nearres, lig):
  # get electrostatic energy in Hartree between each nearres sidechain and lig
  if not hasattr(mol, 'sc_Eqq'):
    openmm_load_params(mol, ff=ff)
    mol.sc_Eqq = sidechain_Eqq(mol, nearres, lig)
    #print("sidechain_Eqq: ", mol.sc_Eqq)
  # prefer residues w/ poor Eres for mutation
  return l1normalize((mol.sc_Eqq > -0.1) + 0.5)

# vis = Chemvis(Mol(mol, [VisGeom(style='lines', sel='water'), VisGeom(sel='protein'), VisGeom(style='spacefill', sel=ligall)]), fog=True).run()

# openmm amber99sb doesn't include C or N terminal ASH, GLH
candidates = ['GLY', 'SER', 'THR', 'LEU', 'ASN', 'ASP', 'CYS', 'MET', 'GLN', 'GLU', 'LYS', 'HIE', 'ARG']

# start interactive
import pdb; pdb.set_trace()

# previous runs:
# - search1.zip: smaller limited set of nearres (see above)
# - search2.zip: partial structure, relaxed w/o freezing backbone
if os.path.exists('search3.zip'):
  theos = load_zip('search3.zip')
else:
  mol0_AI = prep_mutate(molwt_AI, nearres4, 'ALA')
  res0, mol0_AI.r = moloptim(OpenMM_EandG(mol0_AI, ffgbsa), mol0_AI, maxiter=200)
  E0_AI = res0.fun
  #mol0_AI, E0_AI = mutate_relax(molwt_AI, nearres4, 'ALA', ff=ffgbsa, optargs=dict(maxiter=200, verbose=-1))
  #Ep_AZ, _ = openmm_EandG(molp_AZ, ff=ffpart, grad=False)
  #active = '(resnum in %r and not extbackbone) or not protein' % nearres4
  bindAI = partial(relax_bind, E0=E0_AI, ff=ffgbsa)  #, mmargs=dict(nonbondedMethod=openmm.app.CutoffNonPeriodic))
  weightfn = partial(mutate_weights, ff=ffgbsa, nearres=nearres4, lig=ligin)
  mol0_AZ, bindAZ = None, None  #partial(relax_bind, E0=Ep_AZ, ff=ffpgbsa, active=active)
  theos = dE_search2(mol0_AI, mol0_AZ, bindAI, bindAZ, nearres4, candidates, weightfn)
  save_zip('search3.zip', **theos)  # save results


# common params
ntheos = len(theos.dE)
# use same PBC box dimensions for each system
molp0 = theos.molAI[0].extract_atoms(resnums=nearres7 + [ligreswt])
pbcside = np.max(np.diff(molp0.extents(pad=6.0), axis=0))


## ligand unbinding - (steered) MD
if os.path.exists('search3_smd.zip'):
  smd = load_zip('search3_smd.zip')
else:
  smd = Bunch(mols=[None]*ntheos, Rs=[None]*ntheos, Vs=[None]*ntheos, Wneq=[None]*ntheos)
  for ii,mol0 in enumerate(theos.molAI[:1]):  #molAImd; mol.md_vel = theos.molAImd_vel[ii]
    molp0 = mol0.extract_atoms(resnums=nearres7 + [ligreswt])
    mol = prep_solvate(molp0, ffp, pad=6.0, center=pcenter, solvent_chain='Z', neutral=True)
    mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=5000)
    Rs, Es, Vs, Wneq = expel_md(mol, ffp, T0, ligin, bindvec, freeze, dist=12.0, nsteps=20000)
    smd.mols[ii], smd.Rs[ii], smd.Vs[ii], smd.Wneq[ii] = mol, Rs, Vs, np.cumsum(Wneq)
    #np.sum(Wneq)*KCALMOL_PER_HARTREE  # 104, 88, 68, 69.9
    #plot(np.cumsum(Wneq)*KCALMOL_PER_HARTREE)

  save_zip('search3_smd.zip', **smd)  # save results


## umbrella sampling - initialized with trajectory from MD ligand unbinding
mol = smd.mols[0]
cvforce, cvval = com_restraint(mol, ligin, bindvec, kspring=2000)
steps = np.linspace(4.0, 4.0+12.0, 13)
umbres = umbrella(mol, ffp, T0, smd.Rs[0], smd.Vs[0], cvforce, cvval, steps=steps, freeze=freeze, mdsteps=10000)

# search3_umb.zip: kspring=1.6E4
save_zip('search3_umb2.zip', **umbres)

nbins = 24
bins = np.linspace(np.min(umbres.cv), np.max(umbres.cv), nbins+1)[1:]
#bin_n = np.digitize(umbres.cv, bins); [np.sum(bin_n == i) for i in range(nbins)]

mbar = pymbar.MBAR(umbres.E_kln, [len(E[0]) for E in umbres.E_kln])  #, verbose=1)
pmfres = mbar.computePMF(umbres.E0_kn, np.digitize(umbres.cv, bins), nbins, return_dict=True)

# check for overlap between steps
histbins = np.linspace(np.min(umbres.cv), np.max(umbres.cv), 100)
histo = [np.histogram(cvs, histbins)[0] for cvs in umbres.cv]
plot(histbins[:-1], np.transpose(histo))

# try FEP for analysis
fep_results_full(umbres.E_kln, T0, warmup=0)


## FEP calculation - turn off interactions between lig and rest of system
theos.E_kln = [None]*len(theos.dE)
for ii,mol0 in enumerate(theos.molAI[:2]):  #molAImd; mol.md_vel = theos.molAImd_vel[ii]
  molp0 = mol0.extract_atoms(resnums=nearres7 + [ligreswt])
  # lig charge is -2, so we need at least 2 positive ions to decouple along with it
  mol = prep_solvate(molp0, ffp, center=pcenter, side=pbcside, solvent_chain='Z', neutral=2)
  mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=5000)
  ligatoms = mol.select(ligin) + mol.select("name == 'Na+'")[:2]
  # we won't bother w/ restraint for now (w/ short MD runs) - see trypsin2.py for example
  theos.fepres[ii] = fep_decouple(mol, ffp, T0, ligatoms, freeze=freeze, mdsteps=20000)
  fep_results_full(theos.fepres[ii].E_kln, T0)

# search3_fep1.zip - forgot to include counterions (since lig is charged)
save_zip('search3_fep2.zip', **theos)

# FEP with just lig and solvent - only need to run this once of course
molp0 = mol0.extract_atoms(resnums=[ligreswt])
mol = prep_solvate(molp0, ffp, side=pbcside, solvent_chain='Z', neutral=True)  #center=pcenter,
mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=5000)

# check for complete decoupling
feptest = fep_decouple(mol, ffp, T0, ligin, freeze=freeze, mdsteps=20000, state=Bunch(Rs=True, idx_lambda=10))

theos.fepres_hoh = fep_decouple(mol, ffp, T0, ligin, freeze=freeze, mdsteps=20000)
fep_results_full(theos.fepres_hoh.E_kln, T0)


## replace ISJ w/ PRE
# - how to implement a threshold for min value of dE for PRE?
mPRE = molwt_AO.extract_atoms(ligout)
theos.dE_AO, theos.Rs_AO = [], []
for ii,mol0 in enumerate(theos.molAI):
  mol = mol0.extract_atoms(host)
  mol.append_atoms(mPRE)
  Ep_AO = 0  #, _ = openmm_EandG(molp_AI, ff=ffpgbsa, grad=False)
  dE, r1 = relax_bind(mol, Ep_AO, ffgbsa, active=active, optargs=dict(gtol=0.004, maxiter=200, verbose=-2))
  print("dE I - O: %f - %f = %f" % (theos.dE[ii], dE, theos.dE[ii] - dE))
  theos.dE_AO.append(dE)
  theos.Rs_AO.append(r1)

save_zip('search3_io.zip', **theos)  # save results


## MD - for now we will only run on a subset of candidate systems
# TODO: consider adding support to save_zip()/load_zip() for saving mol.md_vel to .md_vel.npy
theos.dEmd = [None]*len(theos.dE)
theos.molAImd = [None]*len(theos.dE)
theos.molAImd_vel = [None]*len(theos.dE)
for ii,mol0 in enumerate(theos.molAI[:4]):
  molp0 = mol0.extract_atoms(resnums=nearres7 + [ligreswt])
  mol = prep_solvate(molp0, ffp, pad=6.0, center=pcenter, solvent_chain='Z', neutral=True)
  mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=5000)
  ctx = openmm_MD_context(mol, ffp, T0, v0=mol.md_vel, freeze=freeze)
  Rs, Es = openmm_dynamic(ctx, 10000, 100)
  dEs = dE_binding(mol, ffpgbsa, host, ligin, Rs)
  theos.dEmd[ii] = dEs
  # save state after MD prep for future MD runs
  theos.molAImd[ii] = mol
  theos.molAImd_vel[ii] = mol.md_vel  # because save_zip will not include this when saving mol
  print("%d dE opt: %f; MD: %f +/- %f" % (ii, theos.dE[ii], np.mean(dEs), np.std(dEs)))

save_zip('search3_md.zip', **theos)  # save results


## MD for water binding
theos.dEhoh = [None]*len(theos.dE)
for ii,mol0 in enumerate(theos.molAI[:4]):
  molp0 = mol0.extract_atoms(resnums=nearres7)
  rlig = mol0.r[mol0.select("chain == 'I' and znuc > 1")] - molp0.r[0]
  mol = prep_solvate(molp0, ffp, pad=6.0, center=pcenter, solvent_chain='Z', neutral=True)
  rlig = rlig + mol.r[0]
  mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=5000)
  ctx = openmm_MD_context(mol, ffp, T0, v0=mol.md_vel, freeze=freeze)
  Rs, Es = openmm_dynamic(ctx, 10000, 100)
  dEs = dE_binding_hoh(mol, ffpgbsa, Rs, rlig)
  theos.dEhoh[ii] = dEs
  print("%d dE lig: %f; water: %f +/- %f" % (ii, theos.dE[ii], np.mean(dEs), np.std(dEs)))


## ligand unbinding - MM
theos.Es_mm = [None]*len(theos.dE)
for ii,mol0 in enumerate(theos.molAI[:4]):
  mol = mol0.extract_atoms(resnums=nearres7 + [ligreswt])
  Rs, Es = expel_mm(mol, ffpgbsa, ligin, bindvec, freeze, dist=12.0, nsteps=13)
  theos.Es_mm[ii] = Es




## clustering and comparing results from dE_search2
# ... try ansi_color() from seq_align.py?
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

seqs1 = [[(AA_3TO1[mol.residues[rn].name]) for rn in nearres4] for mol in theos.molAI]
seqs2 = [[(AA_3TO1[mol.residues[rn].name]) for rn in nearres4] for mol in theos2.molAI]

seqs = np.array(seqs1 + seqs2)
Ds = np.array([[np.sum(seq1 != seq2) for seq1 in seqs] for seq2 in seqs])

Z = linkage(squareform(Ds), method='single')
clust = fcluster(Z, len(seqs[0])/3, criterion='distance')  # also possible to specify number of clusters instead



## NEXT:
# - run dE_search2 a few more times ... different result, similar binding energy
# - include TS using SE QM (and maybe single point DFT/HF)
# - try another system!  let's pick one with uncharged ligand!

# Other thoughts and ideas:
# - FEP: looks like charged ligand might require smaller lambda steps to ensure sufficient overlap between steps!
# - what about using expel_mm (or expel_md) to refine bindvec?
#  - w/ a small number of trajectories, would we basically just pick the smallest barrier?
# - what if we ran expel_mm() from multiple starting geometries (e.g. sampled from MD)?  How would we combine the results?
# - can we make use of pulling out of site/(binding) reaction path for design? or just as a check?
# - reverse roles of I,O and make sure things make sense qualitatively?

# ... I think this is good enough for first attempt - now, how to combine the pieces: reactant binding, TS binding, product binding, water binding?


# - rank by TS - reactant (dE_search2); threshold on product; MD calc for reactant, product, water
# - maybe do fep for a few theozymes?

# - aside: some kind of MD w/ large motions instead of just thermal - large rotations of molecules, torsions?
# ... that's just Monte Carlo!  Maybe try HMC to provide better ensemble for dE_binding?



## openmm metadynamics with distance of lig COM from some point as collective variable
# ... looks like this would need to run for much longer to get useful results
# ref: /home/mwhite/miniconda3/lib/python3.9/site-packages/openmm/app/metadynamics.py

from openmm.app.metadynamics import Metadynamics, BiasVariable

# Metadynamics.step(Simulation) doesn't provide easy way to save positions
cvvals = []
def meta_step(ii, ctx, meta):
  cvpos = meta._force.getCollectiveVariableValues(ctx)
  cvvals.append(cvpos[0])
  energy = ctx.getState(getEnergy=True, groups={meta._force.getForceGroup()}).getPotentialEnergy()
  height = meta.height*np.exp(-energy/(UNIT.MOLAR_GAS_CONSTANT_R * meta._deltaT))
  meta._addGaussian(cvpos, height, ctx)


center = np.mean(molpwt_AI.extents(), axis=0) + 5.0*normalize(bindvec)
mol0 = prep_solvate(molpwt_AI, ffp, pad=6.0, center=center, solvent_chain='Z', neutral=True)
mol = prep_relax(mol0, ffp, T0, freeze, eqsteps=2000)

com = center_of_mass(mol, ligatoms)
force = openmm.CustomCentroidBondForce(1, 'sqrt((x1-xc)^2 + (y1-yc)^2 + (z1-zc)^2)')
force.addPerBondParameter('xc')
force.addPerBondParameter('yc')
force.addPerBondParameter('zc')
force.addGroup(ligatoms)
force.addBond([0], (com - 2.0*normalize(bindvec))*0.1)  # offset center 2A from COM; convert Angstrom to nm
force.setForceGroup(31)

cv = BiasVariable(force, 0*UNIT.angstrom, 15*UNIT.angstrom, 0.5*UNIT.angstrom)
sys = ffp.createSystem(openmm_top(mol), constraints=openmm.app.HBonds, nonbondedMethod=openmm.app.PME, rigidWater=True)
# constants inspired by openmm TestMetadynamics.py
meta = Metadynamics(sys, [cv], T0*UNIT.kelvin, 4.0, 1.0*UNIT.kilojoules_per_mole, 100)
#meta = Metadynamics(sys, [cv], T0*UNIT.kelvin, 4.0, 1000.0*UNIT.kilojoules_per_mole, 100)
ctx = openmm_MD_context(mol, sys, T0, freeze=freeze)
ctx.setVelocities(mol.md_vel*(UNIT.angstrom/UNIT.picosecond))
Rs, Es = openmm_dynamic(ctx, 20000, 100, sampfn=partial(meta_step, meta=meta))


## crossover between dE_search2 candidates
# ... not clear this adds much - less than 1 kcal/mol improvement (dE_search2 results were already pretty good)
# - might try random choice of crossover residues (i.e. w/o grouping)

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

# - then we need to see what fraction of cross-over offspring are better than both, one, or neither parent(s)?
# Then decide how to make use of cross-over:
# - generational: create new generation w/ 2 offspring per couple, then mutations, plus best parents (replacing worst offspring)
# - or? replace worse parent with offspring?


## OLD

## push out/pull in lig from site w/ flat top Gaussian
# ... looks like pulling in isn't going to be useful quantitatively
# - what if we use lower temperature?
#mol = prep_relax(Molecule(molpwt_AI), ffpgbsa, T0, freeze, eqsteps=2000) ... gbsa 2-3x slower than PME!


center = np.mean(molpwt_AI.extents(), axis=0) + 5.0*normalize(bindvec)
mol0 = prep_solvate(molpwt_AI, ffp, pad=6.0, center=center, solvent_chain='Z', neutral=True)
mol = prep_relax(mol0, ffp, T0, freeze, eqsteps=2000)



np.sum(Wneq)*KCALMOL_PER_HARTREE  # 104, 88, 68, 69.9
plot(np.cumsum(Wneq)*KCALMOL_PER_HARTREE)

simstate = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
r1 = simstate.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
v1 = simstate.getVelocities(asNumpy=True).value_in_unit(UNIT.angstrom/UNIT.picosecond)

#save_zip('push_wt2.zip', mol=mol, Rs=Rs, Es=Es, r1=r1, v1=v1, Wneq=Wneq)
# mol, Rs = load_zip('push_wt2.zip', 'mol', 'Rs')

dE_binding(mol, ffpgbsa, host, ligin, Rs[-1])

notwat = mol.select('not water')
res, r_opt = moloptim(OpenMM_EandG(molpwt_AI, ffpgbsa), Rs[-1][notwat], maxiter=50)
dE_binding(molpwt_AI, ffpgbsa, host, ligin, r_opt)

# explicit waters
# - solvate and relax multiple times (also compare to pulling lig out of site)
# - calc MM-GBSA binding energy for 1...N closest waters (to lig COM?)
# ... see if there is a qualitative difference between different number of waters

# list waters by proximity to lig COM

def dE_binding_hoh2(mol, ff, r, r0, nwat, host='protein'):
  ow = mol.select('water and znuc > 1')
  dd = np.linalg.norm(r[ow] - r0, axis=1)
  dsort = np.argsort(dd)
  dd = dd[dsort]
  nearhoh = [ mol.atoms[ow[ii]].resnum for ii in dsort ]
  return [ dE_binding(mol, ff, host, mol.resatoms(nearhoh[:ii]), r) for ii in nwat ]


# dE_binding varies from run-to-run (~1E-6) w/ GBSA (but not w/ no solvent)
dE_binding_hoh2(mol, ffpgbsa, Rs[-1], com, range(1,17))

# should we use overlap of vdW spheres to pick waters?  this will vary between frames ... is this OK if we are averaging?



#hits, dd = mol.get_nearby([com], 7.0, active='water and znuc > 1', r=Rs[-1], dists=True)
#mol.atomsres(hits)

# For explicit waters to be useful, we want to limit somehow to waters that would be displaced by lig (and
#  keep waters from leaving site, as happened w/ simple relaxation)
# Options/ideas:
# - try relaxation with restraint holding (few) waters near site
# - generate multiple candidates (w/ dE_search) using just lig, then do MD w/ and w/o lig in explicit water for each
# - dE_mutations: start w/ lig unbound, pull into site ... substantial computation time, so explore other avenues first


# Done:
# - MD calc for dE_search2 results - w/ lig and w/ waters
#  - binding energy for waters
#  - then try computing binding energy by pulling lig out of host (non-equilib work) (see below)
#  - try metadynamics properly ... need to run for long time to be useful
# - try pulling into site w/ implicit solvent - less variation in orientation? ... much slower, tried once, didn't get correct orientation
# - compute binding free energy from our poor excuse for metadynamics
#  - https://iris.sissa.it/retrieve/handle/20.500.11767/110289/124927/main.pdf ?


#mdres = openmm_dynamic(ctx, 12000, 1, grad=True, vel=True)

# debugging MD failures:
r1 = ctx.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)/UNIT.angstrom
#vis.select( unique(np.argwhere(np.isnan(r1))[:,0]) )
r1[np.isnan(r1)] = 0

## dE from pulling traj - see note below
# what is our CV? distance from some point?

com0 = center_of_mass(mol, ligatoms)
origin0 = (com0 - 5.0*normalize(bindvec))

# ...

t_ramp = 200.0
max_w = 1.4  # nm
Emeta = []  # pulling force strength
Smeta = []  # collective var
def metafn(ii, ctx):
  ctx.setParameter('std', min(max(1E-14, ii-10)/t_ramp, 1.0)*max_w)
  Emeta.append( ctx.getState(getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy()/EUNIT )
  r1 = ctx.getState(getPositions=True, enforcePeriodicBox=isPBC).getPositions(asNumpy=True)/UNIT.angstrom
  Smeta.append(center_of_mass(mol, ligatoms, r1))

Rs, Es = openmm_dynamic(ctx, 20000, 100, sampfn=metafn)

# bin Emeta values by Smeta and average
# - this has nothing to do with metadynamics and won't give anything like the free energy, but might still be
#  interesting (maybe it will approach free energy as we pull slower?)
Sbins = np.linspace(min(Smeta), max(Smeta), 20)
Ebins = [ np.mean(Emeta[Smeta >= Sbins[ii] and Smeta < Sbins[ii+1]]) for ii in range(len(Sbins)-1) ]


# anyway, I think we'd want to look at non-equilb free energy methods to get free energy from pulling
# - how do we separate the work of our pulling force?
# ... maybe use this fn: (have to do every time step? ... probably a reason this doesn't seem to be used as much as other methods)

Wneq = []  # work
def metafn(ii, ctx):
  H0 = ctx.getState(getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy()/EUNIT
  ctx.setParameter('std', min(max(1E-14, ii-10)/t_ramp, 1.0)*max_w)
  H1 = ctx.getState(getEnergy=True, groups={force.getForceGroup()}).getPotentialEnergy()/EUNIT
  Wmeta.append(H1 - H0)


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

ctx = openmm_MD_context(mol, ffp, T0, freeze=freeze, forces=[force])

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
ctx = openmm_MD_context(mol, ffp, T0, freeze=freeze, forces=[force])

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
# - we need to set up a system with PRE!  ... done
# - explicit solvent w/o lig (for near waters) ... perhaps we should use all-GLY for this? ... done
# ... should we populate all 3 sites to compare?
# - some way to penalize voids more strongly (might help single point calc get closer to MD result)? First step would be notes on GBSA method
# - find scaling factor which gives minimum RMSD between positions in explicit vs. implicit solvent MD, then maybe see what force needs to be added to implicit solvent case to match

# - test multithreading for pyscf
# - should we add a "charge" attribute to Molecule instead of having to keep track separately?
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

# - optimize for reactant and product separately, then ...
#  - pick nearest candidates from the two sets and ... ?


## Older

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


if 0:
  mol = add_residue(Molecule(), 'GLY_LFZW', -180, 135)
  mol = add_residue(mol, 'ARG_LFZW', 180, 180)
  mol = add_residue(mol, 'GLY_LFZW', -135, -180)
  Rs = [ set_rotamer(mol, 1, angles).r for angles in get_rotamers('ARG') ]
  vis = Chemvis(Mol(mol, Rs, [VisGeom()])).run()


  ligres = mol0_AI.atomsres(ligin)[0]
  #nearres8 = sorted(find_nearres(mol0_AI, '/A,B,C/*/~C,N,CA,O,H,HA', ligin, 8.0))
  nearres8 = mol0_AI.atomsres(mol0_AI.get_nearby(ligin, 8.0, active='/A,B,C/*/~C,N,CA,O,H,HA'))
  nearres, nearres_full = [ii for ii,rn in enumerate(nearres8) if rn in nearres], nearres

  nearmols = [ molwt_AI.extract_atoms(resnums=molwt_AI.atomsres(molwt_AI.get_nearby(ligin, radius, active='/A,B,C/*/~C,N,CA,O,H,HA')) + [ligres]) for radius in [2,3,4,5,6,7,8] ]


  nearres4 = molwt_AI.atomsres(molwt_AI.get_nearby(ligin, 4.0, active='sidechain'))

  nearres7 = molwt_AI.atomsres(molwt_AI.get_nearby(ligin, 7.0, active='protein'))
  # prevent lone residues
  #unique([ resnum+ii for resnum in pocket for ii in [-1,0,1] ])

  scatoms4 = molwt_AI.select('sidechain and resnum in %r' % nearres4)
  pocket = unique(nearres4 + molwt_AI.atomsres(molwt_AI.get_nearby(scatoms4, 2.8, active='protein')))

  nearmols = [ molwt_AI.extract_atoms(resnums=resnums + [ligres]) for resnums in [nearres4, nearres7, pocket] ]

  # for nearres: 4 Ang - 13 residues
  # for full model: 7 Ang - 29 residues

  resatoms4 = [ a for resnum in nearres4 for a in molwt_AI.residues[resnum].atoms ]
  scatoms4 = [ a for resnum in nearres4 for a in molwt_AI.residues[resnum].atoms if molwt_AI.atoms[a].name not in PDB_EXTBACKBONE]

  scatoms4 = molwt_AI.select('sidechain and resnum in %r' % nearres4)

  molwt_AI.get_nearby(ligin, 4.0, active='/A,B,C/*/~C,N,CA,O,H,HA')

  nearres4p4 = molwt_AI.atomsres(molwt_AI.get_nearby(resatoms4, 4.0, active='/A,B,C/*/~C,N,CA,O,H,HA'))

  nearmols4 = [ molwt_AI.extract_atoms(resnums = unique([ligres] + nearres4 + molwt_AI.atomsres(molwt_AI.get_nearby(scatoms4, radius, active='/A,B,C/*/~C,N,CA,O,H,HA')))) for radius in [0, 2.8, 3.0, 3.5, 4.0] ]

  # Try dE_search2, w/ an eye toward where metadynamics might be useful
  # - very low probability of random set of residues fitting ... identify problematic residues and try different rotatmer?
  #  - add the residues one at a time? try different rotamers upon failure? - this might be better since multiple residues could be issue
  # - start w/ smaller residues?


  molp_AI = mol0_AI.extract_atoms(resnums=nearres8 + [ligres])
  molp_AZ = None

## can't run MD on battery, ha ha; also, MD fails for res.molAI[0] (strained

mol0 = res.molAI[2]

# graft onto full protein, and relax ... this does seem to fix MD failures
mol1 = Molecule(molwt_AI)

for ii,resnum in enumerate(nearres7):
  if ii in nearres:
    mutate_residue(mol1, resnum, mol0.extract_atoms(resnums=ii))

for ii,jj in zip(mol1.residues[ligreswt].atoms, mol0.residues[ligres].atoms):
  mol1.atoms[ii].r = mol0.atoms[jj].r

res, mol1.r = moloptim(OpenMM_EandG(mol1, ffpgbsa, nonbondedMethod=openmm.app.CutoffNonPeriodic), mol1, maxiter=200)
mol0 = mol1.extract_atoms(resnums=nearres7 + [ligreswt])

dE_binding(mol0, ffpgbsa, host, ligin)

# res.molAI[1] failing due to LEU 12 (PDB A 90) ... also res.molAI[2]
# - maybe LYS 0, LYS 32 (PDB A 7, C 74) should be active?  review nearres choices?
# - don't allow lone residues in partial model?
## ... we need to rethink residues in nearres and freeze

# res.molAI[3], [4], [5] fail immediately (also due to A 90 - ARG)?
# ... all mols w/ ARG or LEU for A 90 seem to fail there
# molAI[8] w/ LYS works!

# how to serialize mol.md_vel?
# - save openmm state instead?
#  - why not? risk of openmm state becoming out of sync with mol
#  - we would need to pass openmm state object in addition to mol, so manually copy velocity to mol
# - use a different file format that allows arbitrary data fields (e.g. mmCIF)
#  - Molden [AMBFOR] style xyz? mol2? with section for velocities ... we don't have code to write mol2
#  - Tinker .dyn file? Amber restart file? ... why would you use one of these instead of openmm state?
# - write md_vel field to separate file in the zip
#  - name + ".md_vel.npy"
# - save with pickle if md_vel present? :-(

# is mol object the right place for velocity (as opposed to separate array)?
# - seems reasonable - analogous to mol.r (i.e., option of additional separate r arrays doesn't preclude a default one in mol)

# Need more study, thought on:
# - FEP w/ charged ligand ... do we really need dF per-step < kBT, in which case many lambda steps needed for charged lig; could we replace lig with charges instead of having them be separate?
# ... no, pretty sure there is no <kB*T limitation (that would be non-extensive) - we just need overlap between energy histograms
