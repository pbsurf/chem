from chem.io import *
from chem.io.openmm import *
from chem.model.build import *
from chem.model.prepare import *
from chem.opt.optimize import *
from chem.fep import *
from chem.mm import *

from chem.vis.chemvis import *
from scipy.spatial.ckdtree import cKDTree

# MM-GBSA binding energies
def mmgbsa(mol, Rs, sel, ff):
  selatoms = mol.select(sel)
  EandG = OpenMM_EandG(mol.extract_atoms(selatoms), ff)
  return np.array([ EandG(r[selatoms], grad=False)[0] for r in Rs ]) if np.ndim(Rs) > 2 \
      else EandG(Rs[selatoms], grad=False)[0]

# explicit solvent
ff = openmm_app.ForceField('amber99sb.xml', 'tip3p.xml')

# GBSA params for TIP3P water; radius and scale depend only on element so we've just copied values for H and O
from io import StringIO
tip3p_gbsa = """<ForceField>
 <GBSAOBCForce>
   <Atom type="tip3p-H" charge="0.417" radius="0.115" scale="0.85"/>
   <Atom type="tip3p-O" charge="-0.834" radius="0.148" scale="0.85"/>
 </GBSAOBCForce>
</ForceField>"""

# implicit solvent (including tip3p.xml allows HOH in implicit solvent)
ffgbsa = openmm_app.ForceField('amber99sb.xml', 'tip3p.xml', 'amber99_obc.xml', StringIO(tip3p_gbsa))

# other GBSA models: github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/internal/customgbforces.py


# maxiter=100: we need to optimize better than for combined calc!
def dE_binding_sep(mol, ff, host, lig, r=None, maxiter=100):
  if r is None: r = mol.r
  selA, selI = mol.select(host), mol.select(lig)
  egAI, egA, egI = [ OpenMM_EandG(mol.extract_atoms(sel), ff) for sel in [selA+selI, selA, selI] ]
  _, rAI = moloptim(egAI, r[selA+selI], maxiter=maxiter, raiseonfail=False, verbose=-2)
  _, rA = moloptim(egA, r[selA], maxiter=maxiter, raiseonfail=False, verbose=-2)
  _, rI = moloptim(egI, r[selI], maxiter=maxiter, raiseonfail=False, verbose=-2)
  eAI, eA, eI = egAI(rAI, grad=False)[0], egA(rA, grad=False)[0], egI(rI, grad=False)[0]
  return (eAI - eA - eI)*KCALMOL_PER_HARTREE


def dE_binding(mol, ff, host, lig, r=None):
  if r is None: r = mol.r
  selA, selI = mol.select(host), mol.select(lig)
  eAI, eA, eI = [ mmgbsa(mol, r, sel, ff) for sel in [selA+selI, selA, selI] ]
  return np.mean(eAI - eA - eI)*KCALMOL_PER_HARTREE


def atomic_E_bonded(mol, ff=None, ctx=None):
  if ff is not None or ctx is not None:
    openmm_load_params(mol, ff=ff, sys=(ctx.getSystem() if ctx else None), vdw=True, bonded=True)
  comp = {}
  #Emm, Gmm = SimpleMM(mol, ncmm=NCMM(mol, qqkscale=OPENMM_qqkscale))(components=comp)
  Emm, Gmm = mmbonded(mol, components=comp)
  Ebond, Eangle, Etors = np.zeros(mol.natoms), np.zeros(mol.natoms), np.zeros(mol.natoms)
  # assign half stretch energy to each atom
  for ii, eb in enumerate(comp['Ebond']):
    for a in mol.mm_stretch[ii][0]:
      Ebond[a] += eb/2
  # assign bend energy to central atom
  for ii, eb in enumerate(comp['Eangle']):
    Eangle[ mol.mm_bend[ii][0][1] ] += eb
  # assign half torsion energy to each central atom
  for ii, eb in enumerate(comp['Etors']):
    for a in mol.mm_torsion[ii][0][1:3]:
      Etors[a] += eb/2
  # openmm doesn't distinguish improper torsions ... if it did, we'd assign energy to 3rd atom
  #Enb = np.sum(comp['Eqq'], axis=1) + np.sum(comp['Evdw'], axis=1)
  return Ebond + Eangle + Etors


def set_closest_rotamer(mol, resnum, ligatoms):
  res = mol.residues[resnum]
  rotamers = get_rotamers('HIE' if res.name == 'HIS' else res.name)  ## FIX THIS
  if rotamers:
    r = mol.r
    ligkd = cKDTree(r[mol.select(ligatoms)])
    bbatoms = frozenset('C,N,CA,O,H,HA'.split(','))
    mindists = []
    for angles in rotamers:
      set_rotamer(mol, resnum, angles)
      scatoms = [ii for ii in res.atoms if mol.atoms[ii].name not in bbatoms]
      dists, locs = ligkd.query(r[scatoms], k=[2])
      mindists.append(min(dists))  #nearest = np.asarray(scatoms)[np.argsort(np.ravel(dists))]
    argmin = min(range(len(mindists)), key=lambda ii: mindists[ii])
    print("%s rotamer %r" % (res.name, rotamers[argmin]))
    set_rotamer(mol, resnum, rotamers[argmin])


from chem.opt.dlc import HDLC

def mutate_relax(mol, resnums, newress, ff, ligatoms=None, maxiter=50, verbose=-2):
  if np.ndim(resnums) == 0: resnums = [resnums]
  if np.ndim(newress) == 0: newress = [newress]*len(resnums)
  mut = Molecule(mol)
  for resnum, newres in zip(resnums, newress):
    if newres is None: continue
    mutate_residue(mut, resnum, newres)
    if ligatoms is not None:
      set_closest_rotamer(mut, resnum, ligatoms)
  ctx = openmm_EandG_context(mut, ff)  #, rigidWater=True)  #constraints=openmm_app.HBonds)

  coords = HDLC(mut, [Rigid(mut, res.atoms, pivot=None) if res.name == 'HOH' else
      XYZ(mut, atoms=res.atoms) for res in mut.residues])

  res, mut.r = moloptim(partial(openmm_EandG, ctx=ctx),
      mol=mut, coords=coords, gtol=1E-3, maxiter=maxiter, raiseonfail=False, verbose=verbose)

  # average MM bonded energy per atom for each residue ... seems less useful than per atom energy (thresh 0.01 or 0.005)
  #Eres = [np.mean([Ebd[a] for a in res.atoms]) for res in mutAZ.residues]
  Ebd = atomic_E_bonded(mut, ctx=ctx)
  Ebdmax = np.max(Ebd)
  if Ebdmax > 0.05:  # bonded energy of any atom > 0.05 Hartree?
    print("Most strained atoms:")
    for ii in np.argsort(Ebd)[-10:]:
      print("%4d %3d (%s) %f" % (ii, mut.atoms[ii].resnum, pdb_repr(mut, ii), Ebd[ii]))

  return mut, res.fun, Ebdmax

  # TIP3P hydrogens don't have vdW, so waters must be kept rigid; LocalEnergyMinimizer adds harmonic
  #  restraints to satisfy constraints; we could try moloptim + HDLC(XYZ + Rigid)
  #ctx.setPositions(mut.r)
  #openmm.LocalEnergyMinimizer.minimize(ctx, maxIterations=50)
  #state = ctx.getState(getEnergy=True, getPositions=True)  #, enforcePeriodicBox=True)
  #E1 = state.getPotentialEnergy().value_in_unit(UNIT.kilocalories_per_mole)
  #print("E = %f kcal/mol for %r -> %r" % (E1, resnums, newress))
  #if E1 > 0:  #res.fun > 0:  #res.fun > prevE + 500.0/KCALMOL_PER_HARTREE:  # clash?
  #  return None
  #mut.r = state.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
  #return mut

# equiv to np.full(dim, val).tolist()
def filled(dim, val=None):
  return [filled(dim[1:], val) for ii in range(dim[0])] if len(dim) > 1 else [val]*dim[0]


## Overview
# - set all to GLY initially
# - try mutations for each residue w/ all others GLY
#  - we expect contribution from each residue to be fairly indep.
#  - choose rotamer with atom closest to LIG ... seems to work well; an alternative might be to turn off non-bonded interactions between sidechain and rest of host (but not lig) for first stage of relaxation
# - form and test the most promising combinations (using knapsack algorithm w/ limited total heavy sidechain atoms)
#  - use binding energy in vacuum instead of implicit solvent?
#  - mutation (back) to wild-type shows both sequential and single-step approaches can work

## Given a design:
# - strained residues an issue - use indep minimized structures (or indep trajectories) as a sanity check
#  - I think we need to relax to tighter convergence for dE_binding_sep (or use trajectories instead) ... seems to help!

## Next:
# - similarity score for residues ... done
# - recalculating individual energies (w/ some non-GLY)
# - knapsack on subset
# - initial attempt at genetic algorithm


# residue proximity
rNR = np.array([molAI.atoms[ res_select(molAI, resnum, 'CA')[0] ].r for resnum in nearres])
drNR = rNR[:,None,:] - rNR[None,:,:]
dNR = np.sqrt(np.sum(drNR*drNR, axis=2))



# I think our biggest challenge is to make non-sequential approach work, by addressing issue w/ clashes
# - non-sequential also lets us handle binding of explicit waters

# - knapsack: if every randomized knapsack config uses same candidate for a residue, then set that candidate and recalc energies
# - make it easy to support knapsack on a subset? with GLY?
# - choose best candidate, recalc energies, then do knapsack?

# limit first stage (single mutation) to charged and polar residues only?
# - non-polar residues may mostly have effect w/ explicit waters (and maybe only w/ MD traj?)

# recalculate residues w/in certain distance of choosen candidate?

# make use of similarity between candidates? e.g. if non-polar AA is better than polar AA for a given residue, then makes more sense to try other non-polar AAs first
#  - number of sidechain atoms, net charge, dipole moment


# for each residue, try to determine if large/smaller, more/less polar is improving binding energy

# how would recombination work for knapsack combos?
# - cross-over vs. point mutations
#  - point mutations based on standalone energy? and/or similarity score?
#  - cross-over between pair of combos: bias choice based on choices of nearby (in space) residues?
#  - how else to choose groups for cross-over besides proximity?

# - with randomized deAI, update deAI w/ average from instances that produced accepted combos?


# - some kind of score based on proximity, observed correlation, etc. of pairs of residues? how would we use it?


## - set strongest binding residue, then repeat first stage (individual mutations) w/ candidates that had sufficient binding energy (< -2 ? < -4 ?)
# ... sequential w/ recalculation seems to work well

# - consider that interference between residues may not always involve a clash - e.g., one residue could push lig away from another
#  - assume we've found interfering residues ... then what?

# combine sequential and knapsack? run sequential w/ high threshold, then apply knapsack to remaining GLYs?

# - knapsack on subsets in proximity? but what about overlap between subsets?

# - calculate per-residue energies to guide recombination ("genetic" algo) of designed sites?

# - try some different residues for lig, e.g. GLU, GLN, TYR
# - chorismate mutase, optimizing for difference between reactant and product binding

## Notes/observations:
# - we can relax after removing lig to check *barrier* for binding, but seems that, with enough relaxation steps, protein expands so binding site is fairly relaxed rather than having a binding barrier
# - backbone looks to have significant interaction (when sidechain is GLY), so just working w/ sidechains is an oversimplification
# - for complete design, we don't necessarily want to maximize binding energy - we'd have a specific target value (and want to keep product binding energy low)


# One problem w/ sequential construction is that it's slow
# - given knapsack combinations, test effect of mutating each res back to GLY ... e.g ASP 170 doesn't relax to optimal position
#  - take note of significant RMSD differences for sidechains ... then what?
# - detect overlaps between residues and exclude the more weakly binding residue? ... but what about 3 way conflicts?
#  - add heuristic penality for overlap ... to which residue? both?
#  - assign residues to grid points; if conflict, exclude more weakly binding residue
#   - can this (pairwise exclusions or values?) be added to knapsack algorithm? ... process residues based on binding energy; for a given choice for current residue, exclude overlapping candidates for remaining residues ... by specifying occupied grid positions? explicitly checking for clashes?  this will add another dimension to subproblems
#   - knapsack over grid points instead of residues? how?
#  - set all residues (w/o relaxation), get all conflicts ... ?

# - quantify conflicts between residues/candidates as a first step? ... number of atoms within some threshold of other res?


# So, back to the question of how to avoid excessive heavy residues?
#  - should we penalize based on RMSD change caused by adding residue? ... at the very least we should print (backbone?) RMSD
#  - for this particular case, we could avoid 2/3 heavy residues just by setting a lower threshold for deAI (as high as -2 kcal/mol); scale threshold by number of (heavy) atoms?  ... but seems algorithm doesn't find anything better than GLY except for 2 residues!

# - what about freezing backbone for part of process?
# - constrain bond lengths and angles (so only torsions active)?




# - MM-GBSA w/ trajectory for final geometries?
# - explore single mutations from best candidate in the same manner as the two stages above?
#  - probably worth a try, but I don't expect much improvement
# - compute per-residue interaction energy and test replacements starting w/ worst residue
# ... per-residue interaction energy is a feature of AMBER MMPBSA.py!
# - genetic algo: combine best candidates to create new generation ... seems like most promising metaheuristic for this problem (vs., e.g., simulated annealing)



# - any way to enable continuous (instead of discrete) optimization? ... something like point charge theozyme?
# - how to do from scratch (w/o preexisting pocket)?  Identify charged LIG atoms and try residues w/ oppositely charged atoms (and hydrophobic residues near uncharged LIG atoms?)


# Observation: water has much more freedom to arrange itself in active site than ligand ... so non-polar residues in active site are important? ... does MM-PBSA handle this well?
# ... also suggests we might want to ignore HOH binding for initial stage (all GLY but one) since water will have more freedom to arrange vs. final active site
# ... maybe use only deAZ, then explore deAI - deAZ for single mutations from resulting site

# Aside: what is behavior of (explicit) waters in a hydrophobic pocket? not fixed?

# autodock scoring fn is simple: https://github.com/ttjoseph/mmdevel/blob/master/VinaScore/vinascore.py


molAI0 = load_molecule("A_solv.xyz").extract_atoms('/A,I//')
molAZ0 = load_molecule('A_near.xyz')

#vis = Chemvis(Mol(mol, [ VisGeom(style='lines', sel='/A//'),  VisGeom(style='spacefill', sel='/I//'), VisGeom(style='licorice', sel=('resnum in %r' % nearres)) ]), wrap=False).run()

if os.path.exists('mutate0.pickle'):
  nearres, candidates, dE0_AI, dE0_AZ, molAI, molAZ = read_pickle('mutate0.pickle',
      'nearres', 'candidates', 'dE0_AI', 'dE0_AZ', 'molAI', 'molAZ')
else:
  # we depend on system prepared in trypsin2.py
  mol = load_molecule("A_solv.xyz")
  # candidate residues: side chain proximity to I
  r = mol.r
  selAsc = mol.select("/A/~CYS/~C,N,CA,O,H,HA") # "chain == 'A' and sidechain")
  #selA, selI = mol.select('/A//'), mol.select('/I//')
  kd = cKDTree(r[mol.select('/I/5/~C,N,CA,O,H,HA')])
  dists, locs = kd.query(r[selAsc], distance_upper_bound=10)
  dsort = np.argsort(dists)
  nearest = np.asarray(selAsc)[dsort]
  nearest = nearest[ dists[dsort] < 4.0 ]
  nearres = list(dict.fromkeys(mol.atoms[a].resnum for a in nearest))

  from chem.data.pdb_bonds import PDB_AMINO
  # order candidates by residue mass ... original motivation was to be able to skip bigger residues once
  #  subsitution of lighter ones start failing ... but that concern didn't materialize
  candidates = sorted(list(PDB_AMINO - set(['PRO','GLY', 'MET','CYS'])),  #'TYR','TRP','PHE',
      key=lambda res: np.sum(PDB_RES()[res].mass()))
  #candidates = ['ILE', 'SER', 'HIS', 'GLU', 'ASP', 'THR', 'LYS', 'ALA', 'GLN', 'ARG', 'VAL', 'ASN', 'LEU'])

  # mutate all pocket residues to GLY
  molAI, E0_AI, _ = mutate_relax(molAI0, nearres, 'GLY', ff=ffgbsa, verbose=-1)
  molAZ, E0_AZ, _ = mutate_relax(molAZ0, nearres, 'GLY', ff=ffgbsa, verbose=-1)
  dE0_AI = dE_binding(molAI, ffgbsa, '/A//', '/I//')
  dE0_AZ = dE_binding(molAZ, ffgbsa, '/A//', '/Z//')

  write_pickle('mutate0.pickle',
      nearres=nearres, candidates=candidates, dE0_AI=dE0_AI, dE0_AZ=dE0_AZ, molAI=molAI, molAZ=molAZ)


candidates[candidates.index('HIS')] = 'HIE'  # HIE is more common than HID
dim = (len(nearres), len(candidates))

if 0:
  molAIwt, Ewt_AI, _ = mutate_relax(molAI0, [], [], ff=ffgbsa, verbose=-1)
  molAZwt, Ewt_AZ, _ = mutate_relax(molAZ0, [], [], ff=ffgbsa, verbose=-1)
  dEwt_AI = dE_binding(molAIwt, ffgbsa, '/A//', '/I//')  # -60.490 kcal/mol (-43.826 for molAI0)
  dEwt_AZ = dE_binding(molAZwt, ffgbsa, '/A//', '/Z//')  # (-21.541 kcal/mol for molAZ0)
  dEwt_vac = dE_binding(molAIwt, ff, '/A//', '/I//')  # -106.771 kcal/mol (-90.229 for molAI0)
  dEsepwt_AI = dE_binding_sep(molAI0, ffgbsa, '/A//', '/I//')  # -43.304 kcal/mol (w/ maxiter=100)
  for resnum in nearres: print("%d: %s" % (resnum, molAI0.residues[resnum].name))


if 0:
  Erot, Rrot = [], []
  trprot = get_rotamers('TRP')
  mut1 = mutate_residue(Molecule(molAZ), 193, 'TRP')
  for rot in trprot:
    mut1 = set_rotamer(mut1, 193, rot)
    m2, E = mutate_relax(mut1, [], [], ff=ffgbsa, verbose=-1)  # clashes w/ all rotatmers w/o relaxation
    Erot.append( E )  #openmm_EandG(mut1, ff=ffgbsa, grad=False)[0] )
    Rrot.append( m2.r )


if os.path.exists('mutate2.npz'):
  #nearres, candidates, deAI, deAZ = read_pickle('mutate2.pickle', 'nearres', 'candidates', 'deAI', 'deAZ')
  #deAI, deAZ = read_pickle('mutate2.pickle', 'deAI', 'deAZ')
  mAI = np.reshape([load_molecule(x.decode('utf-8'))
      for x in read_zip('mutate2.zip', *['mAI_%03d.xyz' % ii for ii in range(dim[0]*dim[1])])], dim)
  deAI, deVac = load_npz('mutate2.npz', 'deAI', 'deVac')  #'deAZ')
else:
  dE0_vac = dE_binding(molAI, ff, '/A//', '/I//')
  deAI, deVac = np.full(dim, np.inf), np.full(dim, np.inf)
  mAI, mAZ = filled(dim), filled(dim)  #[ [None]*len(candidates) for n in nearres ]

  for ii,resnum in enumerate(nearres):
    for jj,newres in enumerate(candidates):
      mAI[ii][jj], eAI, ebmaxAI = mutate_relax(molAI, resnum, newres, ff=ffgbsa, ligatoms='/I//')  # host + lig
      #mAZ[ii][jj], eAZ, ebmaxAZ = mutate_relax(molAZ, resnum, newres)  # host + explicit waters
      if eAI > 0 or ebmaxAI > 0.05:  # or eAZ[ii,jj] > 0:  # clash
        continue
      dE_AI = dE_binding(mAI[ii][jj], ffgbsa, '/A//', '/I//') - dE0_AI
      dE_vac = dE_binding(mAI[ii][jj], ff, '/A//', '/I//') - dE0_vac
      #dE_AZ = dE_binding(mAZ[ii][jj], ffgbsa, '/A//', '/Z//') - dE0_AZ
      deAI[ii,jj], deVac[ii,jj] = dE_AI, dE_vac
      #print("Binding energy %d -> %s: %f - %f = %f kcal/mol" % (resnum, newres, dE_AI, dE_AZ, dE_AI - dE_AZ))
      print("Binding energy %d -> %s: %f kcal/mol (%f vac)" % (resnum, newres, dE_AI, dE_vac))

  write_zip('mutate2.zip', **{'mAI_%03d.xyz' % ii : write_xyz(m) for ii,m in enumerate(np.ravel(mAI))})
  np.savez('mutate2.npz', deAI=deAI, deVac=deVac)  #deAZ=deAZ)

  vismols = np.ravel(mAI)[ np.argsort(np.ravel(deAI)) ]  #deVac
  vis = Chemvis(Mol(vismols, [ VisGeom(style='lines', sel='/A//'), VisGeom(style='spacefill', sel='/I//'), VisGeom(style='licorice', sel=('resnum in %r' % nearres)) ]), fog=True).run()


# best interaction given fixed atom number budget ... "multiple-choice knapsack problem"
def knapsack(values, weights, W, cache=None):
  N = len(values)
  if cache is None:
    cache = filled((N, W+1), False)  #np.full((len(values), W), -1)
  elif cache[N-1][W] is not False:
    return cache[N-1][W]
  #print("knapsack(%2d, %2d, %2d)" % (len(values), len(weights), W))
  if N > 1:
    idxss0 = [ (knapsack(values[1:], weights[1:], W - w, cache) if w <= W else None) for w in weights[0] ]
    idxss = [ ([ii] + idxs if idxs is not None else None) for ii,idxs in enumerate(idxss0) ]
  else:
    idxss = [ ([ii] if w <= W else None) for ii,w in enumerate(weights[0]) ]
  Vs = [ (np.sum(values[range(N), idxs]) if idxs is not None else -np.inf) for idxs in idxss ]
  argmax = max(range(len(Vs)), key=lambda ii: Vs[ii])
  cache[N-1][W] = idxss[argmax]
  return idxss[argmax]


if 0:
  deAI = load_npz('mutate4.npz', 'de2AI') - -24.28962706
  newres = mAI[1][7].extract_atoms(resnums=nearres[1])  # 170 ASP
  molAI, eAI, ebmaxAI = mutate_relax(molAI, nearres[1], newres, ff=ffgbsa)  # host + lig


## apply knapsack to subset
def knapsack_sample(mol, nearres, candidates, deAI, W0):
  nscatoms = [len(PDB_RES()[res].select('znuc > 1 and not extbackbone')) for res in candidates]

  resnames = [mol.residues[resnum].name for resnum in nearres]
  sel = [ii for ii,resnum in enumerate(nearres) if resnames[ii] == 'GLY']
  Wsc = sum([nscatoms[candidates.index[resname]] if resname != 'GLY'])

  # add GLY w/ value = 0, weight = 0
  vals = np.c_[np.zeros(len(sel)), -deAI[sel]]
  wgts = np.tile([0] + nscatoms, (len(sel),1))

  # - generating multiple candidates: add random offset to energies (values) ... could also randomize W
  theos0 = [ knapsack(vals + 3.0*(np.random.random(vals.shape)-0.5), wgts, (W0 - Wsc) + np.random.randint(-4, 8)) for ii in range(20) ]
  theos0 = np.unique(theos0, axis=0) - 1  # -1 to account for prepended 0,0 for GLY
  theos = np.full((len(theos0), len(nearres)), -1)
  theos[:,sel] = theos0
  return theos


def dE_theos(molAI, nearres, candidates, mAI, theos):
  mutAI = [None]*len(theos)
  deAI = np.full(len(theos), np.inf)
  for ii,idxs in enumerate(theos):
    newress = [mAI[jj][idx].extract_atoms(resnums=nearres[jj]) if idx >= 0 else None for jj,idx in enumerate(idxs)]
    mutAI[ii], eAI, ebmaxAI = mutate_relax(molAI, nearres, newress, ff=ffgbsa)  # host + lig
    if eAI > 0 or ebmaxAI > 0.05:  # or eAZ[ii,jj] > 0:  # clash
      continue
    deAI[ii] = dE_binding(mutAI[ii], ffgbsa, '/A//', '/I//')

  return deAI, mutAI


## genetic algorithm
# Generating new set of theos from current set
# - choose a subset (best half? probabilistically based on energy?), then partition into pairs
# or: choose theo randomly (weighted by energy), then choose partner randomly
# For each pair:
# - choose a residue randomly, then choose which parent to draw from
# - for subsequent residues, choose parent, weighted by choices of nearby residues
# - choose a different candidate (not from either parent) w/ some probability (based on amino acid similarity and deAI)

# - use clustering algo on residue pair distance to generate clusters?

# - simple non-genetic global search: take top half (N/2) of combos, apply random mutations to get N new candidates, repeat
# - inverse relationship between probability of mutation and number of combos w/ same candidate for the residue
#  - or greater weight to candidates found in existing set?

# - generate new combos as batch or sequentially?

# start interactive
import pdb; pdb.set_trace()

# range of values is 5 to 215
grantham_weights = np.zeros([len(candidates)]*2)
for ii,can in enumerate(candidates):
  for aaa,d in GRANTHAM_DIST[can].items():
    grantham_weights[ii][candidates.index(aaa)] = 1.0/d


def dE_search(mol, nearres, candidates, mAI, deAI0, theos, maxiter=200):
  N = len(theos)
  mutAI = [None]*maxiter
  deAI = np.full(maxiter, np.inf)
  idxhist, dehist, molhist = [], [], []
  try:
    for ii in range(maxiter):
      if ii < N:
        idxs = theos[ii]
      else:
        for jj,_ in enumerate(nearres):
          # favor more common residues for given position
          w0 = numpy.bincount([theo[jj] for theo in idxhist[:N]], minlength=len(candidates))/len(candidates)
          # favor similar residues
          w1 = grantham_weights[ idxhist[N%ii][jj] ]
          # favor residues w/ better binding energy
          w2 = 1 + 0.5*np.clip(-0.1*deAI0[jj], -1, 1)
          # combine terms
          weights[jj] = normalize(w0 + w1 + w2)

        for _ in range(100):
          idxs = [ np.random.choice(range(len(candidates)), p=weights[jj]) for jj,_ in enumerate(idxhist[N%ii]) ]
          if idxs not in idxhist: break

      newress = [mAI[jj][idx].extract_atoms(resnums=nearres[jj]) if idx >= 0 else None for jj,idx in enumerate(idxs)]
      mutAI, eAI, ebmaxAI = mutate_relax(molAI, nearres, newress, ff=ffgbsa)  # host + lig
      deAI = np.inf if eAI > 0 or ebmaxAI > 0.05 else dE_binding(mutAI, ffgbsa, '/A//', '/I//')

      if ii < N or deAI > dehist[demaxidx]:
        idxhist.append(idxs)
        dehist.append(deAI)
        molhist.append(mutAI)
        if ii == N-1:
          demaxidx = max(range(N), key=lambda kk: dehist[kk])
      else:
        idxhist.append(idxhist[demaxidx])
        dehist.append(dehist[demaxidx])
        idxhist[demaxidx] = idxs
        dehist[demaxidx] = deAI
        molhist[demaxidx] = mutAI
        demaxidx = max(range(N), key=lambda kk: dehist[kk])
  except KeyboardInterrupt as e:  #(, Exception) as e:
    pass

  return dehist[:N], molhist




# calculate binding energies for single mutations
def dE_mutations(molAI, nearres, candidates, mAI=None, skip=None, ligatoms=None):
  dim = (len(nearres), len(candidates))
  deAI = np.full(dim, np.inf)
  mutAI = filled(dim)
  for ii,resnum in enumerate(nearres):
    if molAI.residues[resnum].name != 'GLY':
      continue
    for jj,resname in enumerate(candidates):
      # compare to deAI for next residue instead of 0?  -4.0 and -2.0 give same result
      if skip is not None and skip[ii,jj]:
        continue
      newres = mAI[ii][jj].extract_atoms(resnums=[resnum]) if mAI is not None else resname
      mutAI[ii][jj], eAI, ebmaxAI = mutate_relax(molAI, resnum, newres, ff=ffgbsa, ligatoms=ligatoms)  # host + lig
      if eAI > 0 or ebmaxAI > 0.05:  # clash
        continue
      deAI[ii,jj] = dE_binding(mutAI[ii][jj], ffgbsa, '/A//', '/I//')

  return deAI, mutAI


#dE0_AI
# skip non-polar candidates?
if 0:
  deAI, mAI = dE_mutations(molAI, nearres, candidates, ligatoms='/I//')

mutAI, mutAZ = Molecule(molAI), Molecule(molAZ)

deAIseq = [deAI]
while np.min(deAIseq[-1]) < -5.0:
  ridx,cidx = np.unravel_index(np.argmin(deAIseq[-1]), deAIseq[-1].shape)
  deAIseq[-1][ridx,cidx]
  newres = mAI[ridx][cidx].extract_atoms(resnums=nearres[ridx])
  mutAI, eAI, ebmaxAI = mutate_relax(mutAI, nearres[ridx], newres, ff=ffgbsa)
  mutAZ, eAZ, ebmaxAZ = mutate_relax(mutAZ, nearres[ridx], newres, ff=ffgbsa)

  deAInew, _ = dE_mutations(mutAI, nearres, candidates, mAI, skip=(deAIseq[-1] > -4.0))
  deAIseq.append(deAInew - deAIseq[-1][ridx,cidx])

# remaining residues - use knapsack
theos = knapsack_sample(mutAI, nearres, candidates, deAI, 28)
deAIth, mAIth = dE_theos(mutAI, nearres, candidates, mAI, theos)
deAZth, mAZth = dE_theos(mutAZ, nearres, candidates, mAI, theos)

# global search
dE_search(mol, nearres, candidates, mAI, deAI0, theos)




if os.path.exists('mutate3.npz'):
  #mutAI, mutAZ = read_pickle('mutate3.pickle', 'mutAI', 'mutAZ')
  dE_AI, dE_AZ, dEsep_AI = load_npz('mutate3.npz', 'dE_AI', 'dE_AZ', 'dEsep_AI')
  mutAI = [load_molecule(x.decode('utf-8'))
      for x in read_zip('mutate3.zip', *['mutAI_%03d.xyz' % ii for ii in range(len(dE_AI))])]
else:
  # include GLY with 0 weight, 0 value (or weight > 0?)?
  #nscatoms = [PDB_RES()[res + '_LL'].natoms - 6 for res in candidates] # total sidechain atoms
  nscatoms = [len(PDB_RES()[res].select('znuc > 1 and not extbackbone')) for res in candidates]  # sidechain heavy atoms
  # number of (nearres) sidechain atoms for wild-type is 28 ... doesn't seem unreasonable since we're using wild-type cavity
  cidxs = knapsack(-deAI, np.tile(nscatoms, (9,1)), 28)
  for resnum,cidx in zip(nearres, cidxs): print("%d: %s" % (resnum, candidates[cidx]))
  # checks: np.sum(np.array(nscatoms)[cidxs]) ; np.sum(deAI[range(len(deAI)), cidxs])
  #  knapsack(-deAI[np.random.permutation(len(deAI))], np.tile(nscatoms, (9,1)), 28)

  # add GLY w/ value = 0, weight = 0
  sel = [ii for ii,resnum in enumerate(nearres) if resnum != 170]
  nearres = [nearres[ii] for ii in sel]
  ## also need to apply to mAI!
  vals = np.c_[-deAI[sel], np.zeros_like(nearres)]
  wgts = np.tile(nscatoms + [0], (len(nearres),1))

  # - generating multiple candidates: add random offset to energies (values) ... could also randomize W
  theos0 = [ knapsack(vals + 3.0*(np.random.random(vals.shape)-0.5), wgts, (28-4) + np.random.randint(-4, 8)) for ii in range(20) ]
  theos = np.unique(theos0, axis=0)
  # ... too many ALAs?

  dEsep0_AI = dE_binding_sep(molAI, ffgbsa, '/A//', '/I//')
  nt = len(theos)
  mutAI, mutAZ = [None]*nt, [None]*nt
  dE_AI, dE_AZ, dEsep_AI = np.full(nt, np.inf), np.full(nt, np.inf), np.full(nt, np.inf)
  for ii,idxs in enumerate(theos):
    newress = [mAI[jj][idx].extract_atoms(resnums=nearres[jj]) if idx >= 0 else None for jj,idx in enumerate(idxs)]
    mutAI[ii], eAI, ebmaxAI = mutate_relax(molAI, nearres, newress, ff=ffgbsa)  # host + lig
    mutAZ[ii], eAZ, ebmaxAZ = mutate_relax(molAZ, nearres, newress, ff=ffgbsa)  # host + explicit waters
    if eAI > 0 or ebmaxAI > 0.05:  # or eAZ[ii,jj] > 0:  # clash
      continue
    dE_AI[ii] = dE_binding(mutAI[ii], ffgbsa, '/A//', '/I//') - dE0_AI
    dE_AZ[ii] = dE_binding(mutAZ[ii], ffgbsa, '/A//', '/Z//') - dE0_AZ
    dEsep_AI[ii] = dE_binding_sep(mutAI[ii], ffgbsa, '/A//', '/I//') - dEsep0_AI
    dE = dE_AI[ii] - dE_AZ[ii]
    print("Binding energy %d: %f - %f = %f kcal/mol; (sep: %f)" % (ii, dE_AI[ii], dE_AZ[ii], dE, dEsep_AI[ii]))

  write_zip('mutate3.zip', **{'mutAI_%03d.xyz' % ii : write_xyz(m) for ii,m in enumerate(mutAI)})
  write_zip('mutate3z.zip', **{'mutAZ_%03d.xyz' % ii : write_xyz(m) for ii,m in enumerate(mutAZ)})
  np.savez('mutate3.npz', dE_AI=dE_AI, dE_AZ=dE_AZ, dEsep_AI=dEsep_AI)

  for ii,m in enumerate(mutAI):
    m.caption = "Binding energy %d: %f - %f = %f kcal/mol; (sep: %f)" % (ii, dE_AI[ii], dE_AZ[ii], dE_AI[ii] - dE_AZ[ii], dEsep_AI[ii])

  vis = Chemvis(Mol(mutAI, [ VisGeom(style='lines', sel='/A//'), VisGeom(style='spacefill', sel='/I//'), VisGeom(style='licorice', sel=('resnum in %r' % nearres)) ]), fog=True, verbose=True).run()


# calculate RMSD of sidechains to standalone case
# this tells us which residues are out of position, but not why; correlate w/ choices for other residues?
if 0:
  for ii,mol in enumerate(mutAI):
    print("\n%2d: dE_AI = %f" % (ii, dE_AI[ii]))
    for jj,resnum in enumerate(nearres):
      subj = mol.extract_atoms(resnums=resnum)
      resname = 'HIE' if subj.residues[0].name == 'HIS' else subj.residues[0].name
      ref = mAI[jj][candidates.index(resname)].extract_atoms(resnums=resnum)
      rmsd = calc_RMSD(align_mol(subj, ref, 'backbone'), ref, 'sidechain')
      print("%s %d: sidechain RMSD %f" % (resname, resnum, rmsd))


# calculate per-residue energy - in vacuum, not solvent!
# ... not sure this is really useful - doesn't seem particularly correlated w/ binding energy
if 0:
  for ii,mol in enumerate(mutAI):
    if dE_AI[ii] > 0: continue
    comp = {}
    openmm_load_params(mol, ff=ff, charges=True, vdw=True, bonded=False)
    Encmm, Gncmm = NCMM(mol, qqkscale=OPENMM_qqkscale)(mol, components=comp)
    selI = mol.select('/I//')
    # non-bonded interaction energy with lig, for each atom
    EnbI = np.sum(comp['Eqq'][:,selI], axis=1) + np.sum(comp['Evdw'][:,selI], axis=1)
    #EresI = [np.sum(EnbI[mol.residues[resnum].atoms]) for resnum in nearres]  # per residue
    print("\n%2d: dE_AI = %f" % (ii, dE_AI[ii]))
    for resnum in nearres:
      # do sidechain only?
      EresI = np.sum(EnbI[mol.residues[resnum].atoms])*KCALMOL_PER_HARTREE
      print("%s %d: %f" % (mol.residues[resnum].name, resnum, EresI))


# try mutating each res back to GLY ... w/ np.argmin(dE_AI), all are worse (170 -> GLY is almost the same)
# w/ 12 ... all give improvement, esp. 170 -> GLY; notably ASP 170 does not move to optimal position with
#  203 -> GLY ... stuck in local min?
if 0:
  mut2AI0 = mutAI[12]  #np.argmin(dE_AI)]
  mut2AI = [None]*len(nearres)
  dE2_AI = np.full(len(nearres), np.inf)
  for ii,resnum in enumerate(nearres):
    mut2AI[ii], eAI, ebmaxAI = mutate_relax(mut2AI0, resnum, 'GLY', ff=ffgbsa)
    dE2_AI[ii] = dE_binding(mut2AI[ii], ffgbsa, '/A//', '/I//') - dE0_AI
    print("Binding energy %d -> %s: %f kcal/mol" % (resnum, 'GLY', dE2_AI[ii]))


if 0:
  # try mutation to wild-type
  newresnames = [molAI0.residues[resnum].name for resnum in nearres]
  newresmols = [mAI[ii][candidates.index(name)].extract_atoms(resnums=nearres[ii]) for ii,name in enumerate(newresnames) if name != 'GLY']
  newresidxs = [nearres[ii] for ii,name in enumerate(newresnames) if name != 'GLY']
  mutAIwt, eAIwt, ebmaxAIwt = mutate_relax(molAI, newresidxs, newresmols, ff=ffgbsa, verbose=-1, maxiter=100)
  dE_binding(mutAIwt, ffgbsa, '/A//', '/I//')  # -51.702 (vs -60) ... main difference is orientation of GLN 173

  # build sequentially ... matches single step (above) well
  dEreswt = [deAI[ii][candidates.index(name)] for ii,name in enumerate(newresnames) if name != 'GLY']
  sum(dEreswt)  # -42.121 ... but what about dE0_AI !?!
  mutAIwt1 = molAI
  for ii in argsort(dEreswt):
    mutAIwt1, eAIwt1, ebmaxAIwt1 = mutate_relax(mutAIwt1, newresidxs[ii], newresmols[ii], ff=ffgbsa)
    dE_AIwt = dE_binding(mutAIwt1, ffgbsa, '/A//', '/I//')
    print("Binding energy %d -> %s: %f kcal/mol" % (newresidxs[ii], newresmols[ii].residues[0].name, dE_AIwt))


# sequential construction
if 0:  ##os.path.exists('mutate3.xyz'):
  mutAI0 = load_molecule('mutate3.xyz')
else:
  deAI = load_npz('mutate5.npz', 'de2AI')

  # sequential mutations ... can fill pocket w/ too many atoms
  #deAIpersc = deAI/nscatoms
  # sort candidates for each resnum by dE, sort resnums by min(dE)
  cansort = [argsort(x) for x in deAI]
  ressort = argsort([min(x) for x in deAI])

  #dE = deAI - deAZ
  results = [ "%3d (%2d) -> %s (%2d): %f (%f) kcal/mol" % (resnum, ii, newres, jj, deAI[ii,jj], deVac[ii,jj]) for ii,resnum in enumerate(nearres) for jj,newres in enumerate(candidates)]  #dE[ii,jj]
  for ii in np.argsort(deAI.reshape(-1)): print(results[ii])
  print("\n")
  for ii in np.ravel(np.arange(len(deAI))[:,None]*len(deAI[0]) + np.argsort(deAI)): print(results[ii])

  de2AI = np.full(dim, np.inf)

  mutAI, mutAZ = Molecule(molAI), Molecule(molAZ)

  newres = mAI[1][7].extract_atoms(resnums=nearres[1])  # 170 ASP
  mutAI, eAI, ebmaxAI = mutate_relax(mutAI, nearres[1], newres, ff=ffgbsa)  # host + lig

  newres = mAI[5][3].extract_atoms(resnums=nearres[5])  # 176 THR
  mutAI, eAI, ebmaxAI = mutate_relax(mutAI, nearres[5], newres, ff=ffgbsa)  # host + lig

  newres = mAI[3][15].extract_atoms(resnums=nearres[3])  # 193 TRP
  mutAI, eAI, ebmaxAI = mutate_relax(mutAI, nearres[3], newres, ff=ffgbsa)  # host + lig

  prevdE = 0  #dE0_AI  # - dE0_AZ
  for ii in ressort:
    resnum = nearres[ii]
    if mutAI.residues[resnum].name != 'GLY':
      continue
    for jj in cansort[ii]:
      # compare to deAI for next residue instead of 0?  -4.0 and -2.0 give same result
      if deAI[ii][jj] > -31.0:  # unable to find candidate that does better than GLY, move on to next residue
        break
      newres = mAI[ii][jj].extract_atoms(resnums=[resnum])
      mut1AI, eAI, ebmaxAI = mutate_relax(mutAI, resnum, newres, ff=ffgbsa)  # host + lig
      #mmAZ, eAZ, ebmaxAZ = mutate_relax(mutAZ, resnum, newres, ff=ffgbsa)  # host + explicit waters
      if eAI > 0 or ebmaxAI > 0.05:  # or eAZ[ii,jj] > 0:  # clash
        continue
      dE_AI = dE_binding(mut1AI, ffgbsa, '/A//', '/I//') - dE0_AI

      de2AI[ii,jj] = dE_AI

      #dE_AZ = dE_binding(mmAZ, ffgbsa, '/A//', '/Z//') - dE0_AZ
      #E = dE_AI - dE_AZ
      print("Binding energy %d -> %s: %f kcal/mol (lone: %f)" % (resnum, newres.residues[0].name, dE_AI, deAI[ii][jj]))
      if 0:  ##dE_AI < 1000*prevdE:  ## + 0.5*deAI[ii][jj]:  # adjust param as needed
        rmsd = calc_RMSD(mut1AI, mutAI, 'backbone', align=True)  # extbackbone?
        rmsd0 = calc_RMSD(mut1AI, molAI, 'backbone', align=True)
        print("ACCEPTED %d -> %s: %f kcal/mol; BB RMSD %f to prev (%f to all-GLY)\n" % (resnum, newres.residues[0].name, dE_AI, rmsd, rmsd0))
        prevdE = dE_AI
        mutAI = mut1AI
        break  # accept this candidate and move on to next residue

  np.savez('mutate6.npz', de2AI=de2AI)
  ##write_xyz(mutAI, 'mutate3.xyz')
  for resnum in nearres: print("%d: %s" % (resnum, mutAI.residues[resnum].name))




mutAI, _, _ = mutate_relax(mutAI0, [], [], ff=ffgbsa, maxiter=100, verbose=-1)

# dE_binding = -53.450 kcal/mol (w/ 100 more relaxation iter) ... vs. -60 for wild type!
# 170 ASP, 193 TRP alone give -45.521
dEsep = dE_binding_sep(mutAI, ffgbsa, '/A//', '/I//')  # -32.139 kcal/mol ... -41.606 w/ more 100 more relaxation iter


mutA = mutAI.extract_atoms('/A//')
# -8.874062 H -> -9.365994 H  vs.  -9.794986 H -> -9.889667 H for wild-type (WOW!)
_, mutA.r = moloptim(OpenMM_EandG(mutA, ffgbsa), mutA.r, maxiter=100, raiseonfail=False, verbose=-1)


newres = [mutA.extract_atoms(resnums=res) for res in nearres]
mutAZ, eAZ, ebmaxAZ = mutate_relax(molAZ, nearres, newres, ff=ffgbsa, maxiter=100, verbose=-1)
dE_AZ = dE_binding(mutAZ, ffgbsa, '/A//', '/Z//') - dE0_AZ  # -8.050 kcal/mol


# 6: C, N, CA, O, H, HA
sum([len(molAI0.residues[resnum].atoms) - 6 for resnum in nearres]) # molAI0: 60; mutAI: 100
# heavy atoms
#sum([len([a for a in molAI0.residues[resnum].atoms if molAI0.atoms[a].znuc > 1]) - 4 for resnum in nearres])


mol = load_molecule("A_solv.xyz")
Rs, Es = read_hdf5('md2.h5', 'Rs', 'Es')

# investigate binding energy w/ minimized geometries ... average is -53 kcal/mol, reasonably consistent (vs. -36)
selAI = mol.select('/A,I//')  # protein only
molAI = mol.extract_atoms(selAI)
ctx = openmm_EandG_context(molAI, ffgbsa)
dE = []
for ii in range(0, len(Rs), 20):
  res, r = moloptim(partial(openmm_EandG, ctx=ctx), mol=molAI, r0=Rs[ii][selAI], maxiter=100, raiseonfail=False)
  eAI, eA, eI = [ mmgbsa(mol, np.array([r]), sel, ffgbsa) for sel in ['/A,I//', '/A//', '/I//'] ]
  dE.append(np.mean(eAI - eA - eI))
  print("Binding energy: %f kcal/mol" % (dE[-1]*KCALMOL_PER_HARTREE))


# MM-GBSA binding energy ... -36 kcal/mol
eAI, eA, eI = [ mmgbsa(mol, Rs, sel, ffgbsa) for sel in ['/A,I//', '/A//', '/I//'] ]

dE = np.mean(eAI - eA - eI) # == np.mean(eAI) - np.mean(eA) - np.mean(eI) -- how dE is normally defined!
print("Binding energy: %f kcal/mol" % (dE*KCALMOL_PER_HARTREE))
#plot(eAI - eA - eI, bins='auto', fn='hist')
# no solvent instead of implicit - big difference (-70 kcal/mol none vs. -36 kcal/mol implicit)


# find bound waters resnums [3392, 294, 3968, 5563, 3258, 592]
mol = load_molecule('1MCT_A_solv.xyz')
Rs, Es = read_hdf5('traj1.hd5', 'Rs', 'Es')
od2,og,cg1 = mol.select('/A/189/OD2;/A/190/OG;/A/213/CG1')  #/I/5/NE

rmsf = calc_RMSF(unwrap_pbc(Rs, mol.pbcbox))
#vis = Chemvis(Mol(mol, [ VisGeom(style='lines', sel='protein'), VisGeom(style='licorice', sel='not protein', coloring=scalar_coloring(rmsf, [0,10])) ]), wrap=False).run()

r = np.mean(Rs, axis=0)
nearhoh = sorted([ (norm(r[ii] - r[og]), rmsf[ii], ii, a.resnum) for ii,a in enumerate(mol.atoms) if a.znuc == 8 and mol.atomres(ii).name == 'HOH' and norm(r[ii] - r[og]) < 10 and rmsf[ii] < 2.0 ])
hsel = [ a for t in nearhoh for a in mol.residues[t[-1]].atoms ]
asel = mol.select('/A//')

# for use w/ MM-GBSA
#write_xyz(mol.extract_atoms(asel + sorted(hsel)), 'A_near.xyz')

# MM-PBSA w/ explicit bound waters
eAI, eA, eI = [ mmgbsa(mol, Rs, sel, ffgbsa) for sel in [asel+hsel, asel, hsel] ]
dE = np.mean(eAI - eA - eI)
print("Binding energy: %f kcal/mol" % (dE*KCALMOL_PER_HARTREE))

# checking random explicit waters ... dE ~ +0.2 kcal/mol per explicit HOH - seems reasonable
import random
hohres = [ii for ii,res in enumerate(mol.residues) if res.name == 'HOH']
while 1:
  hres = random.sample(hohres, 5)
  hsel = [ a for resnum in hres for a in mol.residues[resnum].atoms ]
  eAI, eA, eI = [ mmgbsa(mol, Rs, sel, ffgbsa) for sel in [asel+hsel, asel, hsel] ]
  dE = np.mean(eAI - eA - eI)
  print("Binding energy: %f kcal/mol" % (dE*KCALMOL_PER_HARTREE))




# - assuming we probably want single-trajectory binding energy estimation...
#  - we can basically use MM-PBSA w/ a single point or short traj ... of course more samples -> better estimate
#  - while even 100fs of traj might be reasonable, problem is initial equilibriation ... maybe play w/ using minimized geom (as single point)
#  - maybe create simple demo using MM-PBSA inside a design loop (e.g. optimizing host residues given LIG)
#  - for design, shouldn't need very long equilib. time after, e.g., mutating a residue if we start from equilibriated state

# - let's try MM-PBSA with minimized geometries - start w/ random samples from traj

# - what about explicit waters in binding cavity?
#  - run solvation + equilib a few times to confirm stability of waters?
#  - use w/ MM-PBSA to calculate correction to binding energy?
#  ... seems reasonable, but sensitive to choice of explicit waters ... need to analyze traj w/ LIG to choose waters; could include some waters from LIG traj, but seems like this would add noise; just accept the set of bound waters?  was pretty obvious for trypsin, but may not be for other systems!
#  - suppose we could dock LIG for each frame and mark overlapping waters ... would this help?
#  - maybe it's foolish to try to get a quantity that can be combined w/ LIG dE ... maybe better to think of as two separate values that could be optimized separately
#  ... we should try an endpoint calc w/ explicit solvent ... how? LIG bound vs. floating freely in solvent

# we saved molA (and molI) after short MD ... but discarded the velocities ... if we had saved velocities, we could have reasonably equilibriated configs to start from!


## Done: (move to oldcode.py)
# - mutate to wild-type and see if we recover geometry, binding energy of wild type
#  ... looks OK; one sidechain orientation is different, but w/ MD traj instead of single point likely would match well
#  - if so, maybe try sequential mutation and see if increase in binding energy is monotonic ... mostly monotonic (ASP 170 dominates, everything else adds only a few kcal/mol)

# Clashes between residues are an issue
# - try sequential construction again! we can add random noise to dE to get multiple candidates
# - how to limit total number of atoms? maybe clashes will do this automatically? reject candidate if increase in binding energy less than, e.g., half standalone binding energy
# ... this seems to work OK, but stuffs 3 distorted rings into active site (which look bad, although dE_binding_sep is OK) ... so there is probably a large barrier for binding, even if binding energy is OK
# - how could we investigate this barrier (even if we find a way to make a "prettier" active site, we'll still want to check barrier for binding!)
#  - doesn't relaxing after removing lig give us the barrier?

#  - could we use this during design process? i.e., reject if energy decrease when relaxing after removing lig is too big?

#  - what if barrier is entropic instead of energetic? How could we reject system w/o doing a full FEP calc? could we get enough info from a short MD traj? avg RMSD between samples from traj and system?

# - use dE_binding_sep? (too many atoms will distort protein when lig is bound -> large positive binding energy)

# - wild-type shows very small change in host energy when relaxing after removing lig, whereas our bad design shows a  large decrease ... nevermind, additional relaxation fixes everything ... by expanding the protein

# - get best and worst per-residue energies for frequently appearing residues in knapsack combos and adjust energy used to knapsack calc accordingly ... I don't think per-residue energies are reliable
# - any insight from per-residue - LIG non-bonded energies? ... I don't think so
