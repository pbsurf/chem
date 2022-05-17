from chem.io import *
from chem.mmtheo import *
from chem.vis.chemvis import *


## Linalool dehydratase/isomerase
# - https://febs.onlinelibrary.wiley.com/doi/10.1002/1873-3468.12165

## ligand parameterization w/ GAFF
if not os.path.exists("LIN.mol2"):
  mGER = load_compound('64Z')  #.remove_atoms([21,23])
  mGER.residues[0].name = 'GER'
  antechamber_prepare(mGER, 'GER', netcharge=0)

  mLIN = reactmol(Molecule(mGER), breakbonds=[(6,9)], makebonds=[(4,9)])
  mLIN.residues[0].name = 'LIN'
  antechamber_prepare(mLIN, 'LIN', netcharge=0)


T0 = 300
host, ligin, ligout, ligall = '/A,B//', '/I//', '/O//', '/I,O//'

ff = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'GER.ff.xml', 'LIN.ff.xml')  #, nonbondedMethod=openmm.app.PME)
ffgbsa = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'implicit/obc1.xml', 'GER.ff.xml', 'LIN.ff.xml', nonbondedMethod=openmm.app.CutoffNonPeriodic)

ffp = OpenMM_ForceField(ff.ff, ignoreExternalBonds=True)
ffpgbsa = OpenMM_ForceField(ffgbsa.ff, ignoreExternalBonds=True)  #removeCMMotion=False

ffpgbsa10 = OpenMM_ForceField(ffgbsa.ff, nonbondedMethod=openmm.app.CutoffNonPeriodic, ignoreExternalBonds=True)

## initial molecule setup
if os.path.exists("LDIwt.zip"):
  molwt_AI, molwt_AO = load_zip("LDIwt.zip", 'molwt_AI', 'molwt_AO')
else:
  ldi_pdb = load_molecule('5HSS', bonds=False, download=False)  #=True
  mol = ldi_pdb.extract_atoms("protein and chain in 'AB'")  # active sites are at interface of chains
  #mol.remove_atoms('*/HIS/HD1')
  mol = add_hydrogens(mol)

  # align to lig in PDB - PDB A 402: O C5 C1
  rref = ldi_pdb.r[ res_select(ldi_pdb, '/A/402/*', 'O,C5,C1') ]

  mLIN = load_molecule('LIN.mol2')
  mLIN.residues[0].chain = 'I'
  mLIN = align_mol(mLIN, rref, sel=res_select(mLIN, 0, 'O,C5,C1'), sort=None)
  molwt_AI = Molecule(mol)
  molwt_AI.append_atoms(mLIN)
  molwt_AI.r = molwt_AI.r - np.mean(molwt_AI.extents(), axis=0)  # move to origin - after appending lig aligned to PDB pos!
  res, molwt_AI.r = moloptim(OpenMM_EandG(molwt_AI, ffgbsa), molwt_AI, maxiter=100)

  mGER = load_molecule('GER.mol2')
  mGER.residues[0].chain = 'O'
  mGER = align_mol(mGER, rref, sel=res_select(mGER, 0, 'O,C5,C1'), sort=None)
  molwt_AO = Molecule(mol)
  molwt_AO.append_atoms(mGER)
  molwt_AO.r = molwt_AO.r - np.mean(molwt_AO.extents(), axis=0)  # move to origin - after appending lig aligned to PDB pos!
  res, molwt_AO.r = moloptim(OpenMM_EandG(molwt_AO, ffgbsa), molwt_AO, maxiter=100)

  save_zip('LDIwt.zip', molwt_AI=molwt_AI, molwt_AO=molwt_AO)

  openmm_load_params(molwt_AI, ff=ff)
  protonation_check(molwt_AI)
  badpairs, badangles = geometry_check(molwt_AI)

#dEwt_AI = dE_binding(molwt_AI, ffgbsa, host, ligin)  #-20.932225237478857
#dEwt_AO = dE_binding(molwt_AO, ffgbsa, host, ligout)  #-23.328514065578446


## dE_search2
ligreswt = molwt_AI.atomsres(ligin)[0]
nearres7 = molwt_AI.atomsres(molwt_AI.get_nearby(ligin, 7.0, active='protein'))
nearres4 = molwt_AI.atomsres(molwt_AI.get_nearby(ligin, 4.0, active='sidechain'))  # 4 Ang

# openmm amber99sb doesn't include C or N terminal ASH, GLH
#candidates = ['GLY', 'SER', 'THR', 'LEU', 'ASN', 'ASP', 'CYS', 'MET', 'GLN', 'GLU', 'LYS', 'HIE', 'ARG']
candidates = "ALA CYS ASP GLU PHE GLY HIE ILE LYS LEU MET ASN GLN ARG SER THR VAL TRP TYR".split()


## create partial system of just residues w/in small radius of lig for faster MM calc
# - three layers: nearres4 (subject to mutation, backbone can move), nearres7 (backbone and sidechain can
#  move), nearres11 (only sidechains can move)
nearres11 = sorted(set(nearres7)
    | set(molwt_AI.atomsres(molwt_AI.get_nearby(molwt_AI.resatoms(nearres7), 4.0, active='protein'))))

molpwt_AI = molwt_AI.extract_atoms(resnums=nearres11 + [ligreswt])
nearres4p = molpwt_AI.atomsres(molpwt_AI.get_nearby(ligin, 4.0, active='sidechain'))  # 4 Ang
nearres7p = molpwt_AI.atomsres(molpwt_AI.get_nearby(ligin, 7.0, active='protein'))

active = 'not extbackbone or resnum in %r' % nearres7p


# what if dE_search2 fails to find hydrophobic pocket?
# - try using energy difference w/ explicit waters (multiple relaxations? MD?)
# ... first step will be MD calc w/ explicit waters w/ some candidates and wild-type
# - notes on GBSA
# - do we simply need a more heuristic potential (like Rosetta) to capture hydrophobic effects?


# water binding MD ... inconclusive I'd say
# - should be able to see some effect from Asn/Asp <-> Leu - multiple LEUs in wild-type to test this
# ... maybe a small effect but slow and pretty noisy (dE_binding_hoh: wt ~ -2 (std 3), ASN -5 - -8 (std 3)
# - maybe run longer MD (100K steps) and look at std(dE) vs number samples ... how many would we need


# compare GBSA energy for LEU vs. ASN/ASP (for whole sidechain? for heavy atoms? combine H atoms w/ heavy atoms?)
# 341 LEU: w/ lig: Egb =  0.00522; Esa = 0.00031; w/o lig: Egb =  0.00520; Esa = 0.00037 (no relaxation)
# 341 ASN: w/ lig: Egb = -0.00194; Esa = 0.00031; w/o lig: Egb = -0.00267; Esa = 0.00034
# 341 ASP: w/ lig : Egb: -0.102571; Esa: 0.000421 w/o lig: Egb: -0.104173; Esa: 0.000456
# 294 ASN: w/ lig : Egb: -0.001574; Esa: 0.000389 w/o lig: Egb: -0.002464; Esa: 0.000439
# 294 ASP: w/ lig : Egb: -0.123818; Esa: 0.000450 w/o lig: Egb: -0.126543; Esa: 0.000510
# - for all, Egb_sel(mol, ffp, ligall) Egb ~ 0.003, Esa ~ 0.001
# - not much different w/ OBC2 or HCT

# so it seems like GBSA isn't capturing displacement of solvent by lig, as expected; what's next?
# - try some different GBSA models ... no difference
#  - can we tweak model somehow to get desired behavior?
# - play w/ a simple Gaussian exclusion (EEF1) or sphere overlap calc?
#  - EEF1 doesn't appear to (directly) use MM charge!
# - look into PBSA?
# ... GB is usually aimed at reproducing PB, but PB probably doesn't capture displacement of solvent by lig either!

# - why put this effort into implicit solvent case?  by construction, it averages over solvent configurations, which is exactly what we want

# - what about calculating solvation energy by multiplying per-atom Born energy (using vdW radius) by SASA fraction?
#  - for comparing theozymes, entropic/hydrophobic surface area contribution won't be varying significantly and can be ignored
#  - only compute for residues near lig (yes, otherwise too slow)
#  - heavy atoms only?  (exclude hydrogens from SASA calc too)
#  - could we somehow limit to binding pocket?  e.g. grid only covers hemisphere facing lig
# ... if we're calculating w/ same geometry (i.e. replacing MM-GBSA), it shouldn't matter; performance is not an issue
# - we may want or need a scaling factor for this Born-SASA energy if we're adding it to other MM energy terms
#  - could we avoid this?  e.g. rank/filter based on Born-SASA separately?
#  - could we determine scaling factor from explicit solvent trajectories?

# - explicit solvent MD w/ and w/o lig; compare dE_binding, dE_binding_hoh (using vacuum, not GBSA?) to dE_binding using minimized structure and Born-SASA; do for several random theozymes
# - maybe try multiple minimized solvent boxes (w/o lig) instead of MD

# ... I don't think comparing total binding energy is going to be useful here - what about electrostatic energy between sidechains of individual residues and explicit solvent? all solvent molecules?  only N nearest?

# - xTB implicit solvation - 10.26434/chemrxiv.14555355.v1 - basically OBC2 but includes empirical H-bond term which takes the same form as ours: ~q^2 * SASA frac; Rosetta includes anisotropic solvation term to account for H-bonds
# ... this suggests we should use our expr with GBSA, not vacuum

# Born-SASA dE_binding:
# - E(AI) - E(A) - E(I); E(mol) = E_ff_vac(mol) + E_born_sasa(mol); E_born_sasa: all atoms? only nearres (all or only sidechain?) and lig?

# - note that we can compute per-residue Born-SASA energies to guide selection of residues to mutate

# QM continuum solvent review (10K+ citations): https://asset-pdf.scinapse.io/prod/1997151511/1997151511.pdf

# - alternative idea: for each grid point, do Born equation w/ radius equal to distance to nearest solvent found in the solid angle centered on the grid point; ideas:
#  - use explicit solvation and find nearest water molecule in solid angle? (use kd-tree to search in overlapping spheres along ray)
#  - raymarch, checking kd-tree at each point for sufficiently large void?

# Summary of attempt to calibrate born_sasa() (rename to hbond_sasa()?):
# - trying to calculate absolute binding energy with explicit solvent: if nothing else, number of waters to displace is uncertain
# - comparing to H bond energies: very noisy of course, but does support scale factor ~1 (i.e., not 0.1, not 10)
# ... basically, we just need to integrate into dE_search2 (using GBSA, not vac) and see what happens


## Summary/thoughts:
# - wild-type Ehb_AI and Ehb_A are both much closer to zero vs. random theozymes
# - GBSA does not capture hydrophobic effect well (because it doesn't capture blocking of solvent interactions well)
# - our H-bond term does a better job of capturing solvent blocking, but introducing it opens up a whole can of
#  worms, e.g., how to scale vs. rest of energy
# - energy fns (e.g. Rosetta) that introduce terms beyond standard MM force field obviously put a lot of effort into calibrating things, so we would either need to repeat that effort or just use Rosetta, etc. (a lot of effort has been put into calibrating GBSA too, of course)
# - the moment you go beyond MM, adding empirical or heuristic terms, it might be better to use ML

# Other ideas:
# - init dE_search2 w/ wild-type, see what it incorrectly favors and why, tweak and repeat

# how long for dE_search2 to find wild-type?
# - assuming it picks the correct individual mutation each time (and ignoring rotamers)
#  - (1 - 1/20)**13 = 0.51 ... say 10 tries on average to find right one ... so 10^N total

# "Nwat-MMGBSA" - www.ncbi.nlm.nih.gov/pmc/articles/PMC5844977/ - single MD run, just includes N nearest waters as part of host

# - ketosteroid isomerase: 1QJG

## NEXT:
# - let's try dE_binding_hoh() somehow in dE_search2()
# - more careful review of protein design literature
# - then it's time to work on CMtheo1.py (and LDItheo1.py?)

# how would we design active sites w/ ML?
# - break lig into chemical groups; separate ML models for environment of reacting and non-reacting groups from PDB

# We know, e.g., hydrophobic residues should go near hydrophobic groups of lig ... so random/global search is stupid
# ... time to bring back dE_mutations?  fix lig somehow?

# What about a simpler model, e.g. only 3-4 residue types (pos, neg, polar, non-polar) ... but also some way to capture shape

## explore algorithms for packing objects together in 3D; simple/toy ML system for this?

# Combinatorial optimization: https://github.com/google/or-tools


nearres = [ii for ii,resnum in enumerate(nearres7p) if resnum in nearres4p]
freeze = 'protein and (extbackbone or resnum not in %r)' % nearres
pcenter = None
pbcside = np.max(np.diff(molpwt_AI.extract_atoms(resnums=nearres7p).extents(pad=6.0), axis=0))  # 42.120256


# start interactive
import pdb; pdb.set_trace()

# previously, we dabbled w/ water binding but mostly assumed GBSA would account for it ... but that is clearly
#  not the case for this hydrophobic site, so we need to rethink

## How to sample multiple configurations w/ explicit water?  what could be integrated with dE_search2?
# - MD? ... slow (MM w/ FEP etc. is the correct way; we're trying to find a shortcut)
# - solvate, relax, dE_binding_hoh ... large variance
# - place water at each heavy atom position w/ random offset and rotation, then use antibunch to give desired number;
# ... seems to work reasonably, but would need some careful calibration of number of waters to keep

# solvate, relax, dE_binding_hoh
mol0 = molpwt_AI
rlig0 = mol0.r[mol0.select("chain == 'I' and znuc > 1")]

if 0:  # make host more hydrophilic
  mutate_residue(mol0, '/A/341/*', 'ASP', align=True)
  mutate_residue(mol0, '/A/294/*', 'HIE', align=True)

# lig + host + solvent; keep only lig and nearres7 (incl. nearres4)
molp0 = mol0.extract_atoms(resnums=nearres7p + [mol0.nresidues-1])  # lig is last residue
mol = prep_solvate(molp0, ffp, side=pbcside, center=pcenter, solvent_chain='Z', neutral=True)
mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=0)
#Rs, Es = openmm_dynamic(openmm_MD_context(mol, ffp, T0, v0=mol.md_vel, freeze=freeze), 20000, 100)
# include waters near lig (as part of host)
#nearhoh = mol.resatoms(mol.atomsres(mol.get_nearby(rlig, 2.8, active=mol.select('water and znuc > 1'))))
#hostatoms = mol.select(host) + nearhoh
dE_hostlig = dE_binding(mol, ffpgbsa, host, ligin)

# host + solvent
molp0 = mol0.extract_atoms(resnums=nearres7p)
mol = prep_solvate(molp0, ffp, side=pbcside, center=pcenter, solvent_chain='Z', neutral=True)
rlig = rlig0 - molp0.r[0] + mol.r[0]
mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=0)
#Rs, Es = openmm_dynamic(openmm_MD_context(mol, ffp, T0, v0=mol.md_vel, freeze=freeze), 20000, 100)
dE_hostsolv = dE_binding_hoh(mol, ffpgbsa, mol.r, rlig, host=host)  #, d=8.0, Nwat=16)

# lig + solvent
molp0 = mol0.extract_atoms(resnums=[mol0.nresidues-1])  # lig is last residue
mol = prep_solvate(molp0, ffp, side=pbcside, center=pcenter, solvent_chain='Z', neutral=True)
mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=0)
#Rs, Es = openmm_dynamic(openmm_MD_context(mol, ffp, T0, v0=mol.md_vel, freeze=freeze), 20000, 100)
r = mol.r
dE_ligsolv = dE_binding_hoh(mol, ffpgbsa, r, rlig=ligall, host=ligall)  #, d=8.0, Nwat=16)

print("dE_hostlig - dE_hostsolv - dE_ligsolv = %.3f - %.3f - %.3f = %.3f" % (
    dE_hostlig, dE_hostsolv, dE_ligsolv, dE_hostlig - dE_hostsolv - dE_ligsolv))


# replace lig with fixed number of waters
mol0 = molpwt_AI

tip3p = load_molecule(TIP3P_xyz, center=True, residue='HOH')

molp0 = mol0.extract_atoms(resnums=nearres7p)
rlig0 = mol0.r[mol0.select("chain == 'I' and znuc > 1")]

if 0:
  mutate_residue(molp0, '/A/341/*', 'ASN', align=True)  #'/A/294/*' ; 'ASP'
  mutate_residue(molp0, '/A/294/*', 'ASN', align=True)  #'/A/294/*' ; 'HIS'

for ii in range(10):
  molp1 = Molecule(molp0)
  rcs = rlig0 + 0.5*(np.random.rand(len(rlig0), 3) - 0.5)
  rcs = rcs[antibunch(rcs, 8)]
  _ = [ molp1.append_atoms(tip3p, r=np.dot(tip3p.r, random_rotation()) + rc) for rc in rcs ]
  mol = prep_solvate(molp1, ffp, side=pbcside, center=pcenter, solvent_chain='Z', neutral=True)
  rlig = rlig0 - molp0.r[0] + mol.r[0]
  mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=0)
  dE_hoh = dE_binding_hoh(mol, ffpgbsa, [mol.r], rlig)[0]
  print(dE_hoh)


## implicit solvation plus H-bonding term based on SASA

def born_sasa(mol, sel=None, r=None, eps_solv=78.5):
  """ return solvation energy in Hartree for each atom (using Born equation multiplied by SASA fraction) for
    atoms `sel` (default all) from Molecule `mol`
  """
  probe = 1.4
  SA = sasa(mol, sel, r, probe=probe)
  radii = np.array([ ELEMENTS[z].vdw_radius for z in mol.znuc ])
  SAfrac = SA/(4*np.pi*(radii + probe)**2)
  # Born equation - convert radii to Bohr to use cgs units
  return SAfrac * -(1 - 1/eps_solv) * mol.mmq**2 / (2*radii/ANGSTROM_PER_BOHR)


def mutate_weights_hb(mol, ff, nearres, lig, hbsel):
  # get electrostatic energy in Hartree between each nearres sidechain and lig
  if not hasattr(mol, 'sc_Eqq'):
    molA = Molecule(mol).remove_atoms(lig)
    Ehb = born_sasa(molA, sel=molA.select(hbsel))*KCALMOL_PER_HARTREE  # limit to sidechain?
    Ehbres = [ np.sum(Ehb[molA.resatoms(resnum)]) for resnum in nearres ]
    Eqq = sidechain_Eqq(mol, nearres, lig)*KCALMOL_PER_HARTREE
    # Eqq: more negative better (stronger binding to lig); Ehbres: less negative better (weaker binding to water)
    mol.sc_Eqq = Eqq - Ehbres
    #print("sidechain_Eqq: ", ' '.join(("%.1f" % e) for e in Eqq))
    #print("sidechain_Ehb: ", ' '.join(("%.1f" % e) for e in Ehbres))
  # should we calculate weights based on relative values of Eqq and Ehb separately and combine somehow?
  Erel = (mol.sc_Eqq - np.min(mol.sc_Eqq))/(np.max(mol.sc_Eqq) - np.min(mol.sc_Eqq))
  #print("weights: ", ', '.join(("%.2f" % e) for e in l1normalize(Erel + 0.25)))
  return l1normalize(Erel + 0.25)
  # prefer residues w/ poor Eres for mutation
  #return l1normalize((mol.sc_Eqq > -0.1) + 0.5)


def Ehbsel(mol, Rs, sel, hbsel):
  Rs1 = Rs if np.ndim(Rs) > 2 else [Rs]
  molp = mol.extract_atoms(sel)
  hbsel = molp.select(hbsel)
  Ehb = [ np.sum(born_sasa(molp, sel=hbsel, r=r[sel])) for r in Rs1 ] # limit to sidechain?
  return np.array(Ehb) if np.ndim(Rs) > 2 else Ehb[0]


def dE_binding_hb(mol, ff, host, lig, hbsel, r=None):
  if r is None: r = mol.r
  openmm_load_params(mol, ff)  # needed for Ehbsel; note that mmq may be missing from just mutated residues
  selA, selI = mol.select(host), mol.select(lig)
  sels = [sorted(selA+selI), selA, selI]
  eAI, eA, eI = [ mmgbsa(mol, r, sel, ff)*KCALMOL_PER_HARTREE for sel in sels ]
  hbAI, hbA, hbI = [ Ehbsel(mol, r, sel, hbsel)*KCALMOL_PER_HARTREE for sel in sels ]
  dEgbsa, dEhb = eAI - eA - eI, hbAI - hbA - hbI
  if np.ndim(r) < 3:
    print("dEhb: %.3f - %.3f - %.3f = %.3f; dEgbsa = %.3f (kcal/mol)" % (hbAI, hbA, hbI, dEhb, dEgbsa))
  return dEgbsa + dEhb


def relax_bind_hb(mol, E0, ffopt, ffbind, hbsel, host='protein', lig='not protein', active=None, mmargs={}, optargs={}):
  optargs = dict(dict(gtol=0.04, maxiter=50, raiseonfail=False, verbose=-100), **optargs)
  r0 = mol.r
  active = mol.select(active) if active is not None else None
  waters = [Rigid(r0, res.atoms, pivot=None) for res in mol.residues if res.name == 'HOH']
  coords = HDLC(mol, [XYZ(r0, active=active, atoms=mol.select('not water'))] + waters) \
      if waters else XYZ(r0, active=active)
  res, r = moloptim(OpenMM_EandG(mol, ffopt, **mmargs), r0, coords=coords, **optargs)
  return (np.inf if res.fun > E0 + 0.5 else dE_binding_hb(mol, ffbind, host, lig, hbsel, r)), r



# search1.zip: molwt_AI used; search2.zip: lig was frozen
if os.path.exists('search3.zip'):
  theos = load_zip('search3.zip')
else:
  mol0_AI = prep_mutate(molpwt_AI, nearres4p, 'ALA')
  coords = XYZ(mol0_AI, active=mol0_AI.select(active))
  res0, r0 = moloptim(OpenMM_EandG(mol0_AI, ffpgbsa10), mol0_AI, coords=coords, maxiter=100, verbose=-2)
  E0_AI, mol0_AI.r = res0.fun, r0
  hbsel = "chain == 'I' or (sidechain and resnum in %r)" % nearres4p
  # gtol=0.02 seems much harder to reach than 0.04
  bindAI = partial(relax_bind_hb, E0=E0_AI, ffopt=ffpgbsa10, ffbind=ffp, hbsel=hbsel, host=host, lig=ligall, active=active, optargs=dict(gtol=0.02, maxiter=60))
  weightfn = partial(mutate_weights_hb, ff=ffp, nearres=nearres4p, lig=ligin, hbsel=hbsel)
  mol0_AZ, bindAZ = None, None
  theos = dE_search2(mol0_AI, mol0_AZ, bindAI, bindAZ, nearres4p, candidates, weightfn, Npop=20)
  save_zip('search3.zip', **theos)  # save results



## analysis of per-atom GBSA energy from gbsa_E()
# GB implicit solvation energy of atoms `sel` in `mol` using MM charges from force field `ff`
def Egb_sel(mol, ff, sel):
  if mol.atoms[0].mmq is None:
    openmm_load_params(mol, ff)
  comp = Bunch()
  gbsa_E(mol, pairsout=comp)
  selatoms = mol.select(sel)
  print("Egb: %f; Esa: %f  (%s)" % (np.sum(comp.Egb[selatoms]), np.sum(comp.Esa[selatoms]), sel))


def Egb_sel2(mol, ff, sel):
  print('w/ lig : ', end='')
  Egb_sel(mol, ffp, sel)
  print('w/o lig: ', end='')
  Egb_sel(mol.extract_atoms('protein'), ffp, sel)



def get_coulomb_terms(rq1, mq1, rq2, mq2, kscale=1.0):
  dr = rq1[:,None,:] - rq2[None,:,:]
  return kscale*ANGSTROM_PER_BOHR*(mq1[:,None]*mq2[None,:])/np.sqrt(np.sum(dr*dr, axis=2))


mol = Molecule(molpwt_AI)
#mol = molpwt_AI.extract_atoms(ligall)  #'protein'

mdwt_AZ, Rswt_AZ = load_zip('mdwt_AZ1.zip', 'mol', 'Rs')

openmm_load_params(mdwt_AZ, ffp)

# for each atom of (sidechain? whole residue?) of (nearres4p? all residues?), find N nearest waters (using O atoms? all atoms?) and compute electrostatic energy (cumsum); also compute SASA frac, (times 1, mmq, and mmq^2) and compare


nearres = [ii for ii,resnum in enumerate(nearres7p) if resnum in nearres4p]
freeze = 'protein and (extbackbone or resnum not in %r)' % nearres


openmm_load_params(mol, ffp)
np.sum(born_sasa(mol, 'pdb_resnum == 341 and sidechain'))
np.sum(born_sasa(mol.extract_atoms('protein'), 'pdb_resnum == 341 and sidechain'))


np.sum(born_sasa(mol, 'resnum in %r and sidechain' % nearres4p))
np.sum(born_sasa(mol.extract_atoms('protein'), 'resnum in %r and sidechain' % nearres4p))

# ... mutate and repeat



np.sum(sasa(mol, 'pdb_resnum == 341 and sidechain'))
np.sum(sasa(mol.extract_atoms('protein'), 'pdb_resnum == 341 and sidechain'))
# ... much larger SASA w/o lig, as expected!


comp = Bunch()
EandG = OpenMM_EandG(mol, ffpgbsa)
EandG(mol, grad=False, components=comp)
print(comp)



Egb_sel2(mol, ffp, 'pdb_resnum == 341 and sidechain')
Egb_sel2(mol, ffp, 'pdb_resnum == 294 and sidechain')

Egb_sel(mol, ffp, ligall)


mutate_residue(mol, '/A/341/*', 'ASN', align=True)  #'/A/294/*' ; 'ASP'
coords = XYZ(mol, active=mol.select(active))
res0, mol.r = moloptim(OpenMM_EandG(mol, ffpgbsa10), mol, coords=coords, maxiter=100)

dE_binding(mol, ffpgbsa10, host, ligin, r=r0)  #wt: -21.560 ... w/ A 294 ASN: -22.361 ASP: -21.644; A 341 ASN: -21.145
# ... no indication hydrophobic effect is being captured
# ... GBSA energy (see dE_binding_comp below) is lower w/ polar/charged residues, as expected, but both w/ and w/o lig ... so not capturing effect of lig blocking from solvent?

dE_binding_comp(mol, ffpgbsa10, host, ligin, r=r0)




## dE_binding_sep - good check for excessive strain?
dEsep = [ dE_binding_sep(mol0, ffpgbsa10, host, ligin, active=active) for mol0 in enumerate(theos.molAI) ]



## MD - for now we will only run on a subset of candidate systems
for ii,mol0 in enumerate(theos.molAI[:4]):
  molp0 = mol0.extract_atoms(resnums=nearres7p + [mol0.nresidues-1])  # lig is last residue
  mol = prep_solvate(molp0, ffp, side=pbcside, center=pcenter, solvent_chain='Z', neutral=True)
  mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=5000)
  ctx = openmm_MD_context(mol, ffp, T0, v0=mol.md_vel, freeze=freeze)
  Rs, Es = openmm_dynamic(ctx, 20000, 100)
  dEs = dE_binding(mol, ffp, host, ligin, Rs)  #ffpgbsa
  print("%d dE opt: %f; MD: %f +/- %f" % (ii, theos.dE[ii], np.mean(dEs), np.std(dEs)))

save_zip('mdwt_AI1.zip', mol=mol, Rs=Rs)


dEs = dE_binding(mol, ffp, '/~I//', '/I//', Rs)  # -31.508

mdwt_AI, Rswt_AI = load_zip('mdwt_AI1.zip', 'mol', 'Rs')
mdwt_AZ, Rswt_AZ = load_zip('mdwt_AZ1.zip', 'mol', 'Rs')
dEs_AI = dE_binding(mdwt_AI, ffp, host, ligin, Rswt_AI)  # -30.805 (-33.533 w/ ~I/I)

dEs_AZ = dE_binding_hoh(mdwt_AZ, ffp, Rswt_AZ, rlig)  # -59.919


def dE_binding_sasa(mol, ff, host, lig, sasasel=None, r=None):
  dE0 = dE_binding(mol, ff, host, lig, r)


## MD for water binding
theos.dEhoh = [None]*len(theos.dE)
for ii,mol0 in enumerate(theos.molAI[:4]):
  pass

mol0 = molpwt_AI

molp0 = mol0.extract_atoms(resnums=nearres7p)
rlig = mol0.r[mol0.select("chain == 'I' and znuc > 1")] - molp0.r[0]
mol = prep_solvate(molp0, ffp, side=pbcside, center=pcenter, solvent_chain='Z', neutral=True)
rlig = rlig + mol.r[0]

solv0 = Molecule(mol)
mutate_residue(mol, '/A/341/*', 'ASN', align=True)  #'/A/294/*' ; 'ASP'

#len(mol.get_nearby(rlig, 2.8, active='water and znuc > 1'))
mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=5000)
ctx = openmm_MD_context(mol, ffp, T0, v0=mol.md_vel, freeze=freeze)
Rs, Es = openmm_dynamic(ctx, 10000, 100)
dEs = dE_binding_hoh(mol, ffpgbsa, Rs, rlig)
print("%d dE lig: %f; water: %f +/- %f" % (ii, theos.dE[ii], np.mean(dEs), np.std(dEs)))


# check solute - solvent distance
r = mol.r
kd = cKDTree(r[mol.select('water and znuc > 1')])
dists, locs = kd.query(r[mol.select('not water')])

radii = np.array([ ELEMENTS[z].vdw_radius for z in mol.znuc ])
d2 = dists - radii[mol.select('not water')]


## OLD ##

# remove overfilled waters after solvation ... cursory test suggests it doesn't make much difference
# - we should compare total waters between two approaches; also compare PBC box sizes after NPT!
overfill = 2
molp0 = mol0.extract_atoms(resnums=nearres7p)
rlig0 = mol0.r[mol0.select("chain == 'I' and znuc > 1")]
mol = prep_solvate(molp0, ffp, side=pbcside, center=pcenter, solvent_chain='Z', neutral=True, overfill=-overfill)
rlig = rlig0 - molp0.r[0] + mol.r[0]
# remove extra waters
ow = mol.select('water and znuc > 1')
keep = antibunch(mol.r[ow], (overfill - 1)*len(ow))
mol.remove_atoms(mol.resatoms([ mol.atoms[a].resnum for ii,a in enumerate(ow) if not keep[ii] ]))
mol.get_nearby(rlig, 2.8, active='water and znuc > 1')  #, r=r)


# let's plot sidechain born_sasa() vs. residue hydrophobicity
def unify_hydrogens(mol, sel):
  if mol.atoms[0].mmq is not None:
    for ii in mol.select(sel):
      mol.atoms[ mol.atoms[ii].mmconnect[0] ].mmq += mol.atoms[ii].mmq
  mol.remove_atoms(sel)  # this will fix up mmconnect
  return mol


Ehb = []
for resname in RESIDUE_HYDROPHOBICITY.keys():
  mol = add_residue(Molecule(), 'GLY', -180, 135)
  mol = add_residue(mol, resname, 180, 180)
  mol = add_residue(mol, 'GLY', -135, -180)
  openmm_load_params(mol, ffp)
  _, mol.r = moloptim(OpenMM_EandG(mol, ffpgbsa), mol, maxiter=100, verbose=-2)
  #unify_hydrogens(mol, 'protein and znuc == 1')
  Ehb.append( np.sum(born_sasa(mol, sel=mol.select('sidechain and resnum == 1')))*KCALMOL_PER_HARTREE )

# ... looks reasonable, except for LYS ... w/ unify_hydrogens() LYS is a little better, but ARG is no good
#list( zip(mol.name, mol.mmq, born_sasa(mol, sel=mol.select('sidechain and resnum == 1'))*KCALMOL_PER_HARTREE) )
list( zip(RESIDUE_HYDROPHOBICITY.keys(), Ehb) )
plot(RESIDUE_HYDROPHOBICITY.values(), Ehb, 'ob')


## attempt to calibrate born_sasa() by comparing with H-bond energies (DSSP)
# using just H bonds seems better - using all atoms includes lots of small energies which create noise in ratio
# ... this suggests factor is ~1 ... maybe we should go ahead and integrate into dE_search2 and have it print Ehb

# H-bond energy isn't really linear in SASA ... maybe plot H-bond energy vs. SASA (including atoms w/ no H-bond)
# ... mmq is of order 1 for all H-bonds, so we can just use Ehb directly

mol = mdwt_AZ
mmq = mol.mmq
prot = mol.select('protein')
mol_prot = mol.extract_atoms(prot)
for r in Rswt_AZ[::10]:

  ctx0 = openmm_MD_context(mol, ffp, T0, freeze=freeze, freezemass=0, constraints=None)  #, **kwargs)
  openmm.LocalEnergyMinimizer.minimize(ctx0, maxIterations=100)
  r = ctx0.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)/UNIT.angstrom

  Ehb_prot = born_sasa(mol_prot, r=r[prot])
  Ehb = setitem(np.zeros(mol.natoms), prot, Ehb_prot)
  Edssc = np.zeros(mol.natoms)

  Erel = []
  hbonds = find_h_bonds(mol, r=r)
  for h,acc in hbonds:
    aprot, asolv = (h, acc) if mol.atomres(acc).name == 'HOH' else (acc, h)
    if mol.atomres(aprot).name in PDB_AMINO and mol.atomres(asolv).name == 'HOH':
      E0 = h_bond_energy(mol, h, acc, r)
      E1 = np.sum(Ehb[ [aprot] + mol.atoms[aprot].mmconnect ])
      Edssc[aprot] += E0
      if E1 == 0:
        print("SASA = 0 for %s" % pdb_repr(mol, aprot))  #pdb.set_trace()
      else:
        Erel.append(E0/E1)
  plot(Edssc, Ehb, 'ob', block=True)
  #plot(Edssc[(Ehb != 0) & (Edssc != 0)]/Ehb[(Ehb != 0) & (Edssc != 0)], 'ob', block=True)
  print("E_dssc/E_hb: %f (%f) (%d H-bonds)" % ( np.mean(Erel), np.std(Erel), len(Erel) ))


def gbsa(mol, eps_solu=1.0, eps_solv=78.5):
  Egb, Esa, B = np.zeros(mol.natoms), np.zeros(mol.natoms), np.zeros(mol.natoms)
  #Lall, Uall, Iall = np.zeros((mol.natoms, mol.natoms)), np.zeros((mol.natoms, mol.natoms)), np.zeros((mol.natoms, mol.natoms))

  # H: 1.2 (1.3 if bonded to N); all other elements, use lookup table (or default of 1.5)
  # note conversion to nm!
  oradii = [ 0.1*((1.3 if mol.atoms[a.mmconnect[0]].znuc == 7 else 1.2)
      if a.znuc == 1 else GBSA_RADIUS.get(a.znuc, 1.5)) - 0.009 for a in mol.atoms ]
  # scaled radii
  sradii = [ GBSA_SCREEN.get(a.znuc, 0.8) * oradii[ii] for ii,a in enumerate(mol.atoms) ]

  for ii,a1 in enumerate(mol.atoms):
    I = 0
    or1 = oradii[ii]
    for jj,a2 in enumerate(mol.atoms):
      if jj == ii: continue
      sr2 = sradii[jj]
      r = calc_dist(a1.r, a2.r)*0.1
      L = max(abs(r - sr2), or1)
      U = r + sr2
      Iij = (r+sr2-or1 > 0) * 0.5*(1/L - 1/U + 0.25*(r - sr2**2/r)*(1/(U**2) - 1/(L**2)) + 0.5*np.log(L/U)/r)
      I += Iij
      #Lall[ii,jj], Uall[ii,jj], Iall[ii,jj] = L, U, Iij
    # now calculate energy for atom
    or1ex = or1 + 0.009
    psi = I*or1
    B[ii] = 1/(1/or1 - np.tanh(0.8*psi + 2.909125*psi**3)/or1ex)  #1.0*psi - 0.8*psi**2 + 4.85*psi**3
    Esa[ii] = 28.3919551 * (or1ex + 0.14)**2 * (or1ex/B[ii])**6

    Egb[ii] = -0.5*138.935485*(1/eps_solu - 1/eps_solv)*a1.mmq**2/B[ii]

    for jj in range(ii):
      a2 = mol.atoms[jj]
      r = calc_dist(a1.r, a2.r)*0.1
      fgb = np.sqrt(r**2 + B[ii]*B[jj]*np.exp(-r**2/(4*B[ii]*B[jj])))
      Egb[ii] += -138.935485*(1/eps_solu - 1/eps_solv)*a1.mmq*a2.mmq/fgb

  #print(Lall); print(Uall); print(Iall)
  return np.sum(Egb + Esa)/KJMOL_PER_HARTREE  # Egb, Esa


def mmgbsa_comp(mol, Rs, sel, ff, comp):
  selatoms = mol.select(sel)
  EandG = OpenMM_EandG(mol.extract_atoms(selatoms), ff)
  return EandG(Rs[selatoms], grad=False, components=comp)[0]

# note that we can't do lig = ~host in mmgbsa() since for general case we have host + lig + solvent
def dE_binding_comp(mol, ff, host, lig, r=None):
  if r is None: r = mol.r
  comps = [Bunch(), Bunch(), Bunch()]
  selA, selI = mol.select(host), mol.select(lig)
  eAI, eA, eI = [ mmgbsa_comp(mol, r, sel, ff, comps[ii]) for ii,sel in enumerate([sorted(selA+selI), selA, selI]) ]
  return (eAI - eA - eI)*KCALMOL_PER_HARTREE, comps  #np.mean() ... return array for better analysis


# we should plot dE_binding vs. optim step to see minimum number of optim steps we need
# ... smaller gtol helps a little, nothing huge
#EandG = OpenMM_EandG(mol, ff, **mmargs)
#def testfn(r1):
#  E,G = EandG(r1)
#  if E < E0 + 0.5:
#    print("E = %f; dE_binding = %f" % (E, dE_binding(mol, ff, host, lig, r1)))
#  return E,G
#
#res, r = moloptim(testfn, r0, coords=coords, **optargs)


# how to speed up first stage of dE_search2?
# - use partial system! (w/ two separate calls to dE_search2 or two separate fns) ... done
# - mutate all at once, detect problematic residues, try different rotatmers and repeat

# find pairs with largest Evdw ... turns out we can just check for nearest pairs
openmm_load_params(mutAI, ff=ffpgbsa10, charges=True, vdw=True, bonded=True)
comp = Bunch()
Encmm, Gncmm = NCMM(mol)(mol, components=comp)

worst = np.unravel_index(np.argmax(comp.Evdw), comp.Evdw.shape)  # + comp.Eqq
mol.atomsres(worst)  # problematic residues

bcomp = Bunch()
Emm, Gmm = mmbonded(mol, components=bcomp)

def get_openmm_gbsa_params():
  for ii,f in enumerate(EandG.ctx.getSystem().getForces()):
    if f.__class__.__name__ == 'CustomGBForce':
      for ii in range(mol.natoms):
        print(f.getParticleParameters(ii))


# try (and FAIL) to approximate SASA by subtracting sphere overlap
# ... this isn't going to work w/o accounting for multiple overlaps as in 10.1002/(SICI)1096-987X(19990130)20:2<217::AID-JCC4>3.0.CO;2-A
def sasa2_FAIL(mol, sel, r=None, probe=0.0):
  max_vdw_rad = 2.75  # Ang - for Na; we could instead do max([ELEMENTS[z].vdw_radius for z in unique(mol.znuc)])
  r = mol.r if r is None else r
  A = 0
  for ii in mol.select(sel):
    ai = ELEMENTS[mol.atoms[ii].znuc].vdw_radius + probe
    neighbors, dij = mol.get_nearby([ii], ai + max_vdw_rad + probe, r=r, dists=True)
    aj = np.array([ELEMENTS[mol.atoms[jj].znuc].vdw_radius + probe for jj in neighbors])
    A += 4*np.pi*ai**2 - np.sum( np.fmax(0, np.pi*ai*(2*ai - dij - (ai**2 - aj**2)/dij)) )
  return A


def compareE(mol, sel=None, r=None, N=10):
  r = mol.r if r is None else r
  selatoms = mol.select(sel)
  waters = np.array(mol.select('water and znuc > 1'))
  mmq = mol.mmq
  kd = cKDTree(r[waters])
  Ecum = np.zeros((len(selatoms), N))
  for ii,a in enumerate(selatoms):
    dists, locs = kd.query(r[a], k=N)
    nearwat = mol.resatoms(mol.atomsres(waters[locs[np.argsort(dists)]]))
    Eqq = get_coulomb_terms(r[[a]], mmq[[a]], r[nearwat], mmq[nearwat])
    Ecum[ii] = np.cumsum(np.sum(np.reshape(Eqq, (-1,3)), axis=-1))  # cumsum of per-residue energies
  return Ecum

# - implement Born-SASA fn (extract fn to get coverage frac from sasa()?), and simple code for getting per-residue energies; play around a bit to get sense of scale ... looks reasonable!  Seems like we might need a scale factor, maybe on the order of ~10
# - do MD on some random theozymes and try to get scale factors

# Born radii calc from 10.1021/jp961710n (HCT, 1996), with adjustment from 10.1002/prot.20033 (OBC, 2004)
# - https://github.com/openmm/openmm/blob/master/platforms/reference/src/SimTKReference/ReferenceObc.cpp
# - based on Tinker: https://github.com/TinkerTools/Tinker/blob/release/source/born.f
# NAMD calculation is more complex (6 part piecewise fn): https://www.ks.uiuc.edu/Research/namd/2.9/ug/node29.html


# - most codes implement GBSA implicit solvent (or none at all) - mostly just CHARMM that has other models
# - OTOH, if we can make something work w/ explicit waters w/o excessive CPU time, that would arguably be better

# EEF1 refs:
# - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5717763/ - Rosetta energy fn
# - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3987920/
#  - https://github.com/sbottaro/EEF1-SB/blob/master/solvpar_eef1sb.inp
# - https://www.plumed.org/doc-v2.6/user-doc/html/_e_e_f_s_o_l_v.html
#  - https://github.com/plumed/plumed2/blob/master/src/colvar/EEFSolv.cpp
# - https://open.library.ubc.ca/media/download/pdf/24/1.0166153/3 - MS thesis w/ overview of implicit solv. methods

# CustomGBForce examples:
# - https://github.com/choderalab/openmmtools/blob/main/openmmtools/testsystems.py#L4278

# - solvate(): what if we remove waters colliding w/ solute before removing overfilled waters? does average occupancy of cavity increase? ... but how many waters do we remove???
# ... e.g. if we overfill by 2x, remove half the remaining waters after adding solute

# what if we just use slightly smaller exclusion radius in solvate()?  determine from solute-solvent distances after MD?
# ... prep_solvate uses d=2.0, which actually matches distribution from MD pretty well
