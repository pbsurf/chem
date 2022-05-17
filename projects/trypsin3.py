from chem.io import *
from chem.io.openmm import *
from chem.model.build import *
from chem.model.prepare import *
from chem.opt.optimize import *
from chem.fep import *
from chem.mm import *

from chem.vis.chemvis import *
from scipy.spatial.ckdtree import cKDTree

# maxiter=100: we need to optimize better than for combined calc!
def dE_binding_sep(mol, ff, host, lig, r=None, maxiter=100):
  if r is None: r = mol.r
  selA, selI = mol.select(host), mol.select(lig)
  selAI = sorted(selA+selI)
  egAI, egA, egI = [ OpenMM_EandG(mol.extract_atoms(sel), ff) for sel in [selAI, selA, selI] ]
  _, rAI = moloptim(egAI, r[selAI], maxiter=maxiter, raiseonfail=False, verbose=-2)
  _, rA = moloptim(egA, r[selA], maxiter=maxiter, raiseonfail=False, verbose=-2)
  _, rI = moloptim(egI, r[selI], maxiter=maxiter, raiseonfail=False, verbose=-2)
  eAI, eA, eI = egAI(rAI, grad=False)[0], egA(rA, grad=False)[0], egI(rI, grad=False)[0]
  return (eAI - eA - eI)*KCALMOL_PER_HARTREE


def atomic_E_bonded(mol, ff=None, ctx=None):
  if ff is not None or ctx is not None:
    openmm_load_params(mol, ctx.getSystem() if ctx else ff, vdw=True, bonded=True)
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


from chem.opt.dlc import HDLC, DLC

# an alternative to choosing nearest rotamer for each sidechain might be to turn off non-bonded interactions
#  between sidechain and rest of host (but not lig) for first stage of relaxation
def mutate_relax(mol, resnums, newress, ff, ligatoms=None, mmargs={}, optargs={}):
  optargs = dict(dict(gtol=1E-3, maxiter=50, raiseonfail=False, verbose=-2), **optargs)
  if np.ndim(resnums) == 0: resnums = [resnums]
  if np.ndim(newress) == 0: newress = [newress]*len(resnums)
  mut = Molecule(mol)
  for resnum, newres in zip(resnums, newress):
    if newres is None: continue
    mutate_residue(mut, resnum, newres)
    if ligatoms is not None and type(newres) is str:
      set_closest_rotamer(mut, resnum, ligatoms)
  ctx = openmm_EandG_context(mut, ff, **mmargs)  #, rigidWater=True)  #constraints=openmm_app.HBonds)
  mutR = mut.r
  coords = HDLC(mut, [Rigid(mutR, res.atoms, pivot=None) if res.name == 'HOH' else
      #DLC(mut, atoms=res.atoms, autoxyzs='all', recalc=1) if resnum in resnums else
      XYZ(mutR, atoms=res.atoms) for resnum,res in enumerate(mut.residues)])
  res, mut.r = moloptim(partial(openmm_EandG, ctx=ctx), mol=mut, coords=coords, **optargs)
  return mut, res.fun


def check_strain(mol, ff, thresh=0.05):
  """ return True if any bonded energy term exceeds thresh (in Hartree) """
  # average MM bonded energy per atom for each residue ... seems less useful than per atom energy (thresh 0.01 or 0.005)
  #Eres = [np.mean([Ebd[a] for a in res.atoms]) for res in mutAZ.residues]
  Ebd = atomic_E_bonded(mol, ff=ff)
  Ebdmax = np.max(Ebd)
  if Ebdmax > thresh:  # bonded energy of any atom > 0.05 Hartree?
    print("Most strained atoms:")
    for ii in np.argsort(Ebd)[-10:]:
      print("%4d %3d (%s) %f" % (ii, mol.atoms[ii].resnum, pdb_repr(mol, ii), Ebd[ii]))
  return Ebdmax > thresh


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


def find_nearres(mol, host, lig, max_dist=4.0):
  """ return indices of residues of `mol` in `host` within `max_dist` of `lig` """
  r = mol.r
  selhost = mol.select(host)
  kd = cKDTree(r[mol.select(lig)])
  dists, locs = kd.query(r[selhost], distance_upper_bound=2*max_dist)
  dsort = np.argsort(dists)
  nearest = np.asarray(selhost)[dsort]
  nearest = nearest[ dists[dsort] < max_dist ]
  return unique(mol.atoms[a].resnum for a in nearest)


# calculate binding energies for single mutations
def dE_mutations(molAI, nearres, candidates, ff, host, lig, mAI=None, skip=None, optargs={}):
  dim = (len(nearres), len(candidates))
  deAI = np.full(dim, np.inf)
  mutAI = filled(dim)
  e0AI, _ = openmm_EandG(molAI, ff=ff, grad=False)
  for ii,resnum in enumerate(nearres):
    for jj,resname in enumerate(candidates):
      # compare to deAI for next residue instead of 0?  -4.0 and -2.0 give same result
      if skip is not None and skip[ii,jj]:
        continue
      newres = mAI[ii][jj].extract_atoms(resnums=[resnum]) if mAI is not None else resname
      mutAI[ii][jj], eAI = mutate_relax(molAI, resnum, newres, ff=ff, ligatoms=lig, optargs=optargs)  # host + lig
      #if (eAI > e0AI + 0.5) != check_strain(mutAI[ii][jj], ff): print("Energy check does not match strain check!")
      if eAI > e0AI + 0.5:  #0 or ebmaxAI > 0.05:  # clash
        print("%d -> %s failed with E = %f" % (resnum, resname, eAI))
        continue
      deAI[ii,jj] = dE_binding(mutAI[ii][jj], ff, host, lig)

  return deAI, mutAI


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


# generate set of theozymes w/ knapsack plus noise
def knapsack_sample(mol, nearres, candidates, deAI, W0, nsamps=20):
  """ generate sets of indices into `candidates` (--w/ -1 = no change--) for residue positions `nearres` (ignoring
    non-GLY residues) using knapsack algorithm with noise, with deAI for values and number of sidechain heavy
    atoms for weights; W0 is target number of sidechain heavy atoms, including non-GLY residues
  """
  nscatoms = [len(PDB_RES()[res].select('znuc > 1 and not extbackbone')) for res in candidates]
  # Wsc: weight consumed by already set (non-GLY) residues
  resnames = [mol.residues[resnum].name for resnum in nearres]
  sel = [ii for ii,resnum in enumerate(nearres) if resnames[ii] == 'GLY']
  Wsc = sum([nscatoms[candidates.index(resname)] for resname in resnames if resname != 'GLY'])
  # add GLY w/ value = 0, weight = 0 ... nevermind
  assert 'GLY' in candidates and np.all(deAI[:,candidates.index('GLY')] == 0), "GLY must be included in candidates"
  vals = -deAI[sel]  #if 'GLY' in candidates else np.c_[np.zeros(len(sel)), -deAI[sel]]
  wgts = np.tile(nscatoms, (len(sel),1))  #if 'GLY' in candidates else np.tile([0] + nscatoms, (len(sel),1))
  # - generating multiple candidates: add random offset to energies (values) ... could also randomize W
  theos0 = [ knapsack(vals + 3.0*(np.random.random(vals.shape)-0.5), wgts, (W0 - Wsc) + np.random.randint(-4, 8)) for ii in range(2*nsamps) ]
  theos0 = np.unique(theos0, axis=0)[:nsamps]  #- 1  # -1 to account for prepended 0,0 for GLY
  theos = np.repeat([[candidates.index(n) for n in resnames]], len(theos0), axis=0)
  theos[:,sel] = theos0
  return theos


def dE_theos(molAI, nearres, candidates, ff, mAI, theos, host, lig):
  mutAI = [None]*len(theos)
  deAI = [np.inf]*len(theos)
  e0AI, _ = openmm_EandG(molAI, ff=ff, grad=False)
  for ii,idxs in enumerate(theos):
    newress = [mAI[jj][idx].extract_atoms(resnums=nearres[jj]) if idx >= 0 else None for jj,idx in enumerate(idxs)]
    mutAI[ii], eAI = mutate_relax(molAI, nearres, newress, ff=ff)  # host + lig

    if (eAI > e0AI + 0.5) != check_strain(mutAI[ii], ff):
      print("Energy check does not match strain check!")

    if eAI > e0AI + 0.5:  #0 or ebmaxAI > 0.05:  # or eAZ[ii,jj] > 0:  # clash
      continue
    deAI[ii] = dE_binding(mutAI[ii], ff, host, lig)

  return deAI, mutAI


# global/non-sequential search
# - given initial set of N theozymes, mutate residues based on heuristic weights, replace worst of N theozymes
#  if better
# - aiming for 50% chance of single res change, 25% two res, etc.; alternative would be explicitly limit to 1 or 2 changes
# Try genetic algorithm w/ cross-over (mating)?
# - choose a subset (best half? probabilistically based on energy?), then partition into pairs
# - use clustering algo on residue pair distance to generate clusters?
# or: choose theo randomly (weighted by energy), then choose partner randomly
# For each pair:
# - choose a residue randomly, then choose which parent to draw from
# - for subsequent residues, choose parent, weighted by choices of nearby residues
# - choose a different candidate (not from either parent) w/ some probability (based on amino acid similarity and deAI)

# Genetic algo refs:
# - https://www.mathworks.com/help/gads/genetic-algorithm.html?s_tid=CRUX_lftnav
# - https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
# - https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
# - https://new.rosettacommons.org/docs/latest/application_documentation/design/mpi-msd

from chem.data.pdb_bonds import GRANTHAM_DIST
pnomr

def dE_search(theos, molAI, molAZ, nearres, candidates, ff, mAI, deAI0, maxiter=200):
  idxhist = list(theos)
  N = len(idxhist)
  e0AI, _ = openmm_EandG(molAI, ff=ff, grad=False)
  e0AZ, _ = openmm_EandG(molAZ, ff=ff, grad=False)
  deAIth, molAIhist = dE_theos(molAI, nearres, candidates, ff, mAI, theos, '/A//', '/I//')
  deAZth, molAZhist = dE_theos(molAZ, nearres, candidates, ff, mAI, theos, '/A//', '/Z//')
  dehist = list(np.asarray(deAIth) - deAZth)
  for i in range(N): print("%r: %f - %f = %f" % (theos[ii], deAIth[ii], deAZth[ii], dehist[ii]))
  results = Bunch(idxs=idxhist, molAI=molAIhist, molAZ=molAZhist, dE=dehist)
  demaxidx = max(range(N), key=lambda kk: dehist[kk])
  chack = ['HIS' if c == 'HIE' else c for c in candidates]
  # range of values is 5 to 215; set weight = 0 for no change case; consider, e.g., **2 to increase effect
  grantham_weights = np.array([ l1normalize([1.0/(GRANTHAM_DIST[a][b] or np.inf) for b in chack]) for a in chack])
  eye = np.eye(len(candidates))
  # favor lighter residues
  wscatoms = np.array([1.0/(1+len(PDB_RES()[res].select('znuc > 1 and not extbackbone'))) for res in candidates])
  try:
    for ii in range(maxiter):
      weights = np.zeros_like(deAI0)  #[0]*len(nearres)
      for jj,_ in enumerate(nearres):
        # favor more common residues for given position
        w0 = l1normalize(np.bincount([theo[jj] for theo in idxhist[:N]], minlength=len(candidates)))
        # ... but also favor candidates which have never been tried for this position
        w1 = l1normalize(np.bincount([theo[jj] for theo in idxhist], minlength=len(candidates)) == 0)
        # favor similar residues
        w2 = grantham_weights[ idxhist[ii%N][jj] ]
        # favor residues w/ better binding energy
        w3 = l1normalize(np.exp(-0.1*deAI0[jj]))  #1 + 0.5*np.clip(-0.1*deAI0[jj], -1, 1))
        # favor no change
        wsame = eye[idxhist[ii%N][jj]]
        # combine terms
        weights[jj] = l1normalize(w0 + w1 + w2 + w3 + 4*len(nearres)*wsame)  #+ 0.5*wscatoms)

      for _ in range(100):
        idxs = [ np.random.choice(range(len(candidates)), p=weights[jj]) for jj,_ in enumerate(idxhist[ii%N]) ]
        if np.any(idxs != idxhist[ii%N]) and not any( np.all(idxs == i) for i in idxhist ): break

      molAI = molAIhist[ii%N]
      # don't touch residue if not changing
      newress = [None if molAI.residues[nearres[jj]].name == candidates[idx] else
          mAI[jj][idx].extract_atoms(resnums=nearres[jj]) for jj,idx in enumerate(idxs)]
      mutAI, eAI = mutate_relax(molAI, nearres, newress, ff=ff, optargs=dict(verbose=-100))  # host + lig
      deAI = np.inf if eAI > e0AI + 0.5 else dE_binding(mutAI, ff, '/A//', '/I//')  #or ebmaxAI > 0.05
      # now with explicit waters
      molAZ = molAZhist[ii%N]
      mutAZ, eAZ = mutate_relax(molAZ, nearres, newress, ff=ff, optargs=dict(verbose=-100))  # host + waters
      deAZ = -np.inf if eAZ > e0AZ + 0.5 else dE_binding(mutAZ, ff, '/A//', '/Z//')  #or ebmaxAZ > 0.05
      dE = deAI - deAZ

      accept = deAI < 0 and deAZ < 0 and dE <= dehist[demaxidx]
      print("%03d: %r -> %r (%d changes): %f - %f = %f%s" % (ii, idxhist[ii%N], idxs,
          np.count_nonzero(idxhist[ii%N] != idxs), deAI, deAZ, dE, ' (accept)' if accept else ''))
      idxhist.append(idxhist[demaxidx] if accept else idxs)
      dehist.append(dehist[demaxidx] if accept else dE)
      if accept:
        idxhist[demaxidx] = idxs
        dehist[demaxidx] = dE
        molAIhist[demaxidx] = mutAI
        molAZhist[demaxidx] = mutAZ
        demaxidx = max(range(N), key=lambda kk: dehist[kk])
  except KeyboardInterrupt as e:  #(, Exception) as e:
    pass

  return results


## Overview
# 1. determine set of active site residues based on proximity to ligand
# 1. set all to GLY initially
# 1. calculate ligand binding energy for mutations of each residue w/ all others remaining GLY
#  - we expect contribution from each residue to be fairly indep.
#  - choose rotamer with atom closest to LIG ... seems to work well
# 1. form and test a set of candidate active sites
#  - first, choose residues with strongest individual binding energy, recalculating individual binding energies
#   for most promising subset of remaining residues and candidates
#  - then use knapsack algorithm w/ noise and limited total heavy sidechain atoms to create set of complete
#   active sites
# 1. global search to optimize difference between binding of ligand vs. explicit waters
#  - idea is that making displacement of water more favorable might be the more important role of the
#   remaining residues (that don't strongly bind ligand)
#  - genetic algo (w/ recombination?) seems like most promising metaheuristic for this problem (vs., e.g., simulated annealing)


# Done:
# - MM-GBSA w/ trajectory for best final geometries ... seems reasonable
# - try some different residues for lig, e.g. GLU, GLN, TYR
#  - GLU ... seems reasonable; explicit waters may have shifted to bad positions
#  - TRP ... this does seem to be selecting more hydrophobic residues, but still picking big residues (e.g. ARG), and not seeing big differences for explicit waters
# - play with 1PSO.pdb (pepsin - aspartic protease)
# - have mutate_relax check strain after e.g. 10 iter and stop if OK; or just use df/f termination threshold
#  ... try gtol ~ 0.05 (0.1 - 0.01)
## Next:
# - visually investigate binding of explicit waters
# - try MM-GBSA w/ different subsets of explicit waters near active site from MD run; try vacuum instead of implicit solvent?
# - what is behavior of (explicit) waters in a hydrophobic pocket? not fixed?
# - does "SA" part of MM-PBSA account for solvent-ligand vdW?  If not then what is implication of vdW inclusion in host-ligand energy?


## Issues
# - how to avoid excessive heavy residues?  backbone RMSD change for mutation?  see if it is a issue for other systems?
#  - is this due to vdW interaction? dE_binding_novdw() looks promising, try using for dE_search
# - barrier to binding! use enhanced sampling MD (e.g. steered, metadynamics, etc.) to get path for binding?
# - strained residues - use indep minimized structures (or indep trajectories) as a sanity check?
# - voids? check with chemvis?


## Notes/observations:
# - mutation (back) to wild-type shows both sequential and single-step approaches to construction can work
# - we can relax after removing lig to check *barrier* for binding, but seems that, with enough relaxation steps, protein expands so binding site is fairly relaxed rather than having a binding barrier
# - backbone looks to have significant interaction (when sidechain is GLY), so just working w/ sidechains is an oversimplification
# - for complete design, we don't necessarily want to maximize binding energy - we'd have a specific target value (and want to keep product binding energy low)
# - note that we should not expect to be using force field params derived from QM calcs in the future, since
#  a single point calculation isn't sufficient, e.g., we'd need to scan dihedrals to get torsion params
# - https://openforcefield.org - SMIRNOFF - force field parameters assigned based on "SMIRKS" string (similar to SMILES) describing environment of part of molecule.  So it can be said that params are assigned directly, instead of first assigning atom types, then params.  This seems like a promising direction.

# - water has much more freedom to arrange itself in active site than ligand ... so non-polar residues in active site are important? ... does MM-PBSA handle this well?
# ... also suggests we might want to ignore HOH binding for initial stage (all GLY but one) since water will have more freedom to arrange vs. final active site
# ... maybe use only deAZ, then explore deAI - deAZ for single mutations from resulting site


# explicit solvent
ff = openmm.app.ForceField('amber99sb.xml', 'tip3p.xml')
# implicit solvent (including tip3p.xml allows HOH in implicit solvent)
#ffgbsa = openmm.app.ForceField('amber99sb.xml', 'tip3p.xml', 'amber99_obc.xml', StringIO(tip3p_gbsa))
ffgbsa = openmm.app.ForceField('amber99sb.xml', 'tip3p.xml', 'implicit/obc1.xml')
# other GBSA models: github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/internal/customgbforces.py


# molAI, molAZ: all nearres (residues near I) mutated to GLY w/ lig/HOH binding energies dE0_AI, dE0_AZ
if os.path.exists('mutate0.zip'):
  nearres, candidates, dE0_AI, dE0_AZ, mol0_AI, molAZ = load_zip('mutate0.zip',
      'nearres', 'candidates', 'dE0_AI', 'dE0_AZ', 'molAI', 'molAZ')
else:
  molAI0 = load_molecule("A_solv.xyz").extract_atoms('/A,I//')  # chain A + chain I (lig)
  molAZ0 = load_molecule('A_near.xyz')  # chain A + waters near active site
  # we depend on system prepared in trypsin2.py
  mol = load_molecule("A_solv.xyz")
  # candidate residues (nearres): side chain proximity to I
  nearres = find_nearres(mol, '/A/~CYS/~C,N,CA,O,H,HA', '/I/5/~C,N,CA,O,H,HA')

  from chem.data.pdb_bonds import PDB_AMINO
  # order candidates by residue mass ... original motivation was to be able to skip bigger residues once
  #  subsitution of lighter ones start failing ... but that concern didn't materialize
  candidates = sorted(list(PDB_AMINO - set(['PRO','GLY', 'MET','CYS'])),  #'TYR','TRP','PHE',
      key=lambda res: np.sum(PDB_RES()[res].mass()))
  #candidates = ['ILE', 'SER', 'HIS', 'GLU', 'ASP', 'THR', 'LYS', 'ALA', 'GLN', 'ARG', 'VAL', 'ASN', 'LEU'])
  candidates[candidates.index('HIS')] = 'HIE'  # HIE is more common than HID

  # mutate all pocket residues to GLY
  molAI, E0_AI = mutate_relax(molAI0, nearres, 'GLY', ff=ffgbsa, optargs=dict(verbose=-1))
  molAZ, E0_AZ = mutate_relax(molAZ0, nearres, 'GLY', ff=ffgbsa, optargs=dict(verbose=-1))
  dE0_AI = dE_binding(molAI, ffgbsa, '/A//', '/I//')
  dE0_AZ = dE_binding(molAZ, ffgbsa, '/A//', '/Z//')

  ## mutate I from ARG to GLU/TRP/etc and recalc dE0_AI
  molAI, _ = mutate_relax(molAI, 224, 'TRP', ff=ffgbsa, optargs=dict(verbose=-1))  # GLU
  dE0_AI = dE_binding(molAI, ffgbsa, '/A//', '/I//')
  #vis = Chemvis(Mol(molAI, [ VisGeom(style='lines', sel='/A//'),  VisGeom(style='spacefill', sel='/I//'), VisGeom(style='licorice', sel=('resnum in %r' % nearres)) ]), fog=True).run()

  save_zip('mutate0.zip',
      nearres=nearres, candidates=candidates, dE0_AI=dE0_AI, dE0_AZ=dE0_AZ, molAI=molAI, molAZ=molAZ)


# calculate dE_binding for each candidate for each residue in nearres (all others in nearres kept as GLY)
# - speeding up this step: we could skip non-polar candidates?  reducing iterations has limited effect because
#  loading MM params and checking for strained bonds is slow
if os.path.exists('mutate2.zip'):
  deAI, mAI = load_zip('mutate2.zip', 'deAI', 'mAI')
  mAI = np.reshape(mAI, deAI.shape)
else:
  deAI, mAI = dE_mutations(mol0_AI, nearres, candidates, ffgbsa, '/A//', '/I//', optargs=dict(maxiter=10))
  deAI -= dE0_AI
  #deVac = [[dE_binding(m, ff, '/A//', '/I//') - dE0_vac for m in mm] for mm in mAI]
  save_zip('mutate2.zip', deAI=deAI, mAI=np.ravel(mAI))
  # visualization check
  vismols = np.ravel(mAI)[ np.argsort(np.ravel(deAI)) ]  #deVac
  vis = Chemvis(Mol(vismols, [ VisGeom(style='lines', sel='/A//'), VisGeom(style='spacefill', sel='/I//'), VisGeom(style='licorice', sel=('resnum in %r' % nearres)) ]), fog=True).run()


# sequential construction w/ recalculation of standalone binding energies (for candidates w/ sufficiently good
#  energy from previous step)
if os.path.exists('twophase2.zip'):
  deAIseq, mutAI, mutAZ = load_zip('twophase2.zip', 'deAIseq', 'mutAI', 'mutAZ')
else:
  mutAI, mutAZ = Molecule(molAI), Molecule(molAZ)
  dEcurr = dE0_AI
  deAIseq = [deAI + dE0_AI]
  skip = np.full(deAI.shape, False)
  while np.min(deAIseq[-1]) - dEcurr < -5.0:
    ridx,cidx = np.unravel_index(np.argmin(deAIseq[-1]), deAIseq[-1].shape)  #deAIseq[-1][ridx,cidx]
    newres = mAI[ridx][cidx].extract_atoms(resnums=nearres[ridx])
    mutAI, eAI = mutate_relax(mutAI, nearres[ridx], newres, ff=ffgbsa)
    mutAZ, eAZ = mutate_relax(mutAZ, nearres[ridx], newres, ff=ffgbsa)
    skip[ridx,:] = True
    deAInew, _ = dE_mutations(mutAI, nearres, candidates, ffgbsa, '/A//', '/I//', mAI, skip=skip|(deAIseq[-1] > -4.0 + dEcurr))
    dEcurr = deAIseq[-1][ridx,cidx]
    deAIseq.append(deAInew)

  save_zip('twophase2.zip', deAIseq=deAIseq, mutAI=mutAI, mutAZ=mutAZ)


# add GLY to candidates
if 'GLY' not in candidates:
  candidates.append('GLY')
  mAI = np.c_[mAI, [molAI]*len(nearres)]
  deAI = np.c_[deAI, np.zeros_like(nearres)]


if os.path.exists('theos4.zip'):
  res = load_zip('theos4.zip')
else:
  # generate candidates, then run global search
  theos = knapsack_sample(mutAI, nearres, candidates, deAI, 28, nsamps=10)
  #mutAZ = align_mol(mutAZ, mutAI, '/A/*/C,CA,N')
  res = dE_search(theos.tolist(), mutAI, mutAZ, nearres, candidates, ffgbsa, mAI, deAI)
  save_zip('theos4.zip', **res)  # save results


ord = np.argsort(res.dE[:len(res.molAI)])
molhist, dehist = [res.molAI[ii] for ii in ord], [res.dE[ii] for ii in ord]
molzhist = [align_mol(res.molAZ[ii], res.molAI[ii], '/A/*/C,CA,N') for ii in ord]

# ... looks reasonable
# - remove two most distant explicit waters and repeat to confirm result qualitatively the same
# - as a sanity check, we should generate equal number of random theozymes and confirm that dE_search does better
# - need to detect (and address?) voids

if os.path.exists('mut2MD.zip'):
  mutAsolv, Rs, Es = load_zip('mut2MD.zip', 'mol', 'Rs', 'Es')
else:
  molAsolv = load_molecule("A_solv.xyz")
  #newress = [molhist[0].extract_atoms(resnums=res) for res in nearres]
  #mutAsolv, eAsolv, ebmaxAsolv = mutate_relax(molAsolv, nearres, newress, ff=ff, mmargs=dict(nonbondedMethod=openmm_app.PME))
  mutAsolv = align_mol(Molecule(molhist[1], pbcbox=molAsolv.pbcbox), molAsolv, '/A/*/C,CA,N')
  mutAsolv.append_atoms(molAsolv.extract_atoms('/Z//'))
  mutAsolv, eAsolv = mutate_relax(mutAsolv, [], [], ff=ff, mmargs=dict(nonbondedMethod=openmm.app.PME), optargs=dict(verbose=-1))
  print("dE_binding: %f ; %f" % (
      dE_binding(molhist[1], ffgbsa, '/A//', '/I//'), dE_binding(mutAsolv, ffgbsa, '/A//', '/I//')))  # -46.664004043000226

  ctx = openmm_MD_context(mutAsolv, ff, T0)
  Rs, Es = openmm_dynamic(ctx, 10000, sampsteps=100)  # plot(Es)
  dE_binding(mutAsolv, ffgbsa, '/A//', '/I//', Rs[30:])  # -32.4582763284179
  save_zip('mut2MD.zip', mol=mutAsolv, Rs=Rs, Es=Es)

# MD w/ GBSA
#ctx = openmm_MD_context(molhist[0], ffgbsa, T0, nonbondedMethod=openmm.app.NoCutoff)
#Rs, Es = openmm_dynamic(ctx, 10000, sampsteps=100)
#dEmd = dE_binding(molAI, ff, '/A//', '/I//', Rs[:30])
#plot(dEmd, bins='auto', fn='hist')  # print(np.mean(dEmd), np.std(dEmd))

#vis = Chemvis(Mol([m for mm in zip(molhist, molzhist) for m in mm], [ VisGeom(style='lines', sel='/A//'), VisGeom(style='spacefill', sel='/~A//'), VisGeom(style='licorice', sel=('resnum in %r' % nearres)) ]), fog=True).run()

# some analysis
if 0:
  dEsep = [dE_binding_sep(m, ffgbsa, '/A//', '/I//') for m in molhist]  # ... nothing terribly unreasonable
  [[m.residues[ii].name for ii in nearres] for m in molhist]
  [len(m.select('resnum in %r and znuc > 1 and not extbackbone' % nearres)) for m in molhist]  # sidechain heavy atoms
  # align entire backbone to molAI - all GLY (use wildtype instead?), then calculate backbone RMSD for nearres ... 0.4 - 0.8 Ang
  malign = Molecule(mutAI)
  [calc_RMSD(align_mol(malign, m, 'backbone'), m, 'resnum in %r and backbone' % nearres) for m in molhist]



## Old

def rotamer_mols(res):
  return [ set_rotamer(PDB_RES(res), 0, angles) for angles in get_rotamers('HIE' if res == 'HIS' else res) ]

# disabling vdW for I vs. for A+I gives same result, as expected
def dE_binding_novdw(mol, ff, host, lig, r=None):
  selA, selI = mol.select(host), mol.select(lig)
  eAI, _ = openmm_EandG(mol.extract_atoms(selA+selI), ff=ff, grad=False, epsilons=[(ii+len(selA),0.0) for ii in range(len(selI))])
  eA, _ = openmm_EandG(mol.extract_atoms(selA), ff=ff, grad=False)
  eI, _ = openmm_EandG(mol.extract_atoms(selI), ff=ff, grad=False, epsilons=[(ii,0.0) for ii in range(len(selI))])
  return (eAI - eA - eI)*KCALMOL_PER_HARTREE


dE_binding_novdw(molhist[0], ffgbsa, '/A//', '/I//')
dE_binding_novdw(molzhist[0], ffgbsa, '/A//', '/Z//')


dim = (len(nearres), len(candidates))
#polarcan = [c for c in candidates if RESIDUE_HYDROPHOBICITY['HIS' if c == 'HIE' else c] > 0.1]  # 10
#nonpcan = [c for c in candidates if c not in polarcan]

# wild-type
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


# residue proximity
if 0:
  rNR = np.array([molAI.atoms[ res_select(molAI, resnum, 'HA3')[0] ].r for resnum in nearres])  # GLY HA3 ~ CB
  drNR = rNR[:,None,:] - rNR[None,:,:]
  dNR = np.sqrt(np.sum(drNR*drNR, axis=2))


if 0:
  deAI = load_npz('mutate4.npz', 'de2AI') - -24.28962706
  newres = mAI[1][7].extract_atoms(resnums=nearres[1])  # 170 ASP
  molAI, eAI, ebmaxAI = mutate_relax(molAI, nearres[1], newres, ff=ffgbsa)  # host + lig


# calculate dE_binding for each candidate for each residue in nearres (all others in nearres kept as GLY)
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


# generate set of theozymes w/ knapsack algorithm (standalone deAI plus noise for value, sidechain atoms for weight)
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


# as expected, hydrophobic residues have weaker effect
if 0:
  hyd = np.array([RESIDUE_HYDROPHOBICITY['HIS' if c == 'HIE' else c] for c in candidates])
  ord = np.argsort(hyd)
  plot(hyd[ord], deAI[:,ord].T)


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
    comp = Bunch()
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
dEsep = dE_binding_sep(mutAI, ffgbsa, '/A//', '/I//')  # -32.139 kcal/mol ... -41.606 w/ 100 more relaxation iter


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


## MM-GBSA w/ MD traj (for wildtype)
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
