# fns for theozyme design

import os
import numpy as np
from .basics import *
from .molecule import *
from .io import load_molecule, copy_residues
from .io.pyscf import pyscf_EandG


# Generating residues: Tinker `protein`, then `xyzpdb` - H2N, COH caps don't work, even with multiple residues
# Could download PDB files http://ligand-expo.rcsb.org/reports/A/ARG/ARG_ideal.pdb etc, but tinker has problems
#  with the non-zwitterionic backbone
def place_residue(resname, align_atoms=None, align_ref=None, r_origin=None):
  """ Place residue specified by resname w/ atoms `align_atoms` aligned with points `align_ref` and w/
    backbone (CA) oriented away from r_origin
  """
  pdbmol = load_molecule(os.path.expandvars('$HOME/qc/2016/common/') + resname + '1.pdb')
  xyzmol = load_molecule(os.path.expandvars('$HOME/qc/2016/common/') + resname + '1.xyz') #, charges='generate')
  mol = copy_residues(xyzmol, pdbmol)
  mol.residues[0].pdb_num = None
  if align_atoms is not None:
    align_subj = mol.r[align_atoms]

    align_cm = np.sum(align_subj, axis=0)/len(align_atoms)
    r_calpha = mol.atoms[1].r  # tinker residues have C alpha as 2nd atom

    v0 = np.sum(align_ref, axis=0)/len(align_atoms) - r_origin
    v1 = r_calpha - align_cm
    calpha_ref = (1 + norm(v1)/norm(v0))*v0 + r_origin

    # use alignment of C alpha w/ small weight (0.1) to orient residue
    align_subj = np.append(align_subj, [r_calpha], axis=0)
    align_ref = np.append(align_ref, [calpha_ref], axis=0)
    weights = [1]*len(align_atoms) + [0.1]
    mol.r = apply_affine(alignment_matrix(align_subj, align_ref, weights), mol.r)

  return mol


# hierarchical clustering
# - seems to work well; any reason to consider alterative approaches?
# - e.g., for each conformation, compare RMSD to accepted conformations and discard if close to one of them,
#  otherwise add to accepted conformations
# - density based clustering (DBSCAN, OPTICS) seems to be the current state-of-the-art; see scikit-learn.org
def cluster_poses(Es, Rs, rslig, thresh=0.9):
  """ cluster geometries `Rs` (w/ energies `Es`) based on RMSD of `rslig` atoms w/ threshold `thresh`,
    choosing representative w/ lowest energy
  """
  from scipy.spatial.distance import squareform
  from scipy.cluster.hierarchy import linkage, fcluster
  # calc pairwise RMSD matrix
  ligRs = Rs[:, rslig, :]
  dRs = ligRs[:,None,:,:] - ligRs[None,:,:,:]
  Ds = np.sqrt(np.einsum('iljk->il', dRs**2)/dRs.shape[-2])
  # perform clustering; squareform converts between square distance matrix and condensed distance matrix
  Z = linkage(squareform(Ds), method='single')
  clust = fcluster(Z, thresh, criterion='distance')  # also possible to specify number of clusters instead
  # pick lowest energy geometry as representative of each cluster
  MCr, MCe = [], []
  for ii in np.unique(clust):
    sel = clust == ii
    MCe.append(np.amin(Es[sel]))
    MCr.append(Rs[sel][np.argmin(Es[sel])])
  esort = np.argsort(MCe)
  return np.asarray(MCe)[esort], np.asarray(MCr)[esort]


## fns for optimizing charges on grid

def pyscf_bq_qgrad(mol, dm, coords):  #, charges):
  # The interaction between QM atoms and MM particles
  # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
  coords = np.asarray(coords)/ANGSTROM_PER_BOHR  # convert to Bohr
  qm_coords = mol.atom_coords(unit='Bohr')
  qm_charges = mol.atom_charges()
  dr = qm_coords[:,None,:] - coords
  r = np.linalg.norm(dr, axis=2)
  g = np.einsum('i,ij', qm_charges, r**-1)  #-qm_charges[:,None]/r

  # The interaction between electron density and MM particles
  for i, rq in enumerate(coords):
    with mol.with_rinv_origin(rq):
      v = mol.intor('int1e_rinv')  # sunqm.github.io/pyscf/_modules/pyscf/gto/moleintor.html
    f = -np.einsum('ij,ji', dm, v)
    g[i] += f
  return g


def bq_grid_init(mol, r_rs, r_ts, r_ps, d=2.5, grid_density=2.0, maxbq=20, minsep=2.5, minG=0):
  """ choose charge positions (up to maxbq, at least d from mol atoms and minsep from each other) with highest
    gradient
  """
  from chem.qmmm.grid import r_grid
  from scipy.spatial.ckdtree import cKDTree
  extents = get_extents(r_ts, pad=(d + 2.0/grid_density))
  grid = r_grid(extents, grid_density)
  kd = cKDTree(np.vstack((r_ts, r_rs, r_ps)))
  dists, locs = kd.query(grid, distance_upper_bound=d)
  grid = grid[dists > d]  # dist = np.inf if no hit

  E_rs, Gr_rs, mf_rs = pyscf_EandG(mol, r_rs)
  Gbq_rs = pyscf_bq_qgrad(mf_rs.base.mol, mf_rs.base.make_rdm1(), grid)
  E_ts, Gr_ts, mf_ts = pyscf_EandG(mol, r_ts)
  Gbq_ts = pyscf_bq_qgrad(mf_ts.base.mol, mf_ts.base.make_rdm1(), grid)
  Gbq = Gbq_ts - Gbq_rs

  bqsort = np.argsort(np.abs(Gbq))[::-1]
  bqs = [ grid[bqsort[0]] ]
  for ii in bqsort[1:]:
    if np.all(np.linalg.norm(bqs - grid[ii], axis=1) >= minsep):
      bqs.append(grid[ii])
      # test pyscf_bq_qgrad
      #E1, Gr, mf = pyscf_EandG(mol, charges=[ Bunch(r=grid[ii], qmq=0.01) ])
      #print("expected dE: %f; actual dE: %f" % (0.01*Gbq[ii], E1 - E))

    if len(bqs) >= maxbq or abs(Gbq[ii]) < minG:
      break

  return bqs


class Charge_QOpt:
  def __init__(self, mol, r_rs, r_ts, r_ps, maxbq=8):
    self.mol, self.r_rs, self.r_ts, self.r_ps = mol, r_rs, r_ts, r_ps
    self.bq_coords = bq_grid_init(mol, r_rs, r_ts, r_ps, maxbq=maxbq)
    self.E_rs0 = pyscf_EandG(mol, r_rs)[0]
    self.E_ts0 = pyscf_EandG(mol, r_ts)[0]
    self.E_ps0 = pyscf_EandG(mol, r_ps)[0]

  def __call__(self, q):
    charges = [ Bunch(r=r, qmq=q) for r,q in zip(self.bq_coords, q) ]
    # consider changing pyscf_EandG to accept charges and positions separately
    E_rs, Gr_rs, mf_rs = pyscf_EandG(self.mol, self.r_rs, charges=charges)
    Gbq_rs = pyscf_bq_qgrad(mf_rs.base.mol, mf_rs.base.make_rdm1(), self.bq_coords)
    E_ts, Gr_ts, mf_ts = pyscf_EandG(self.mol, self.r_ts, charges=charges)
    Gbq_ts = pyscf_bq_qgrad(mf_ts.base.mol, mf_ts.base.make_rdm1(), self.bq_coords)
    E_ps, Gr_ps, mf_ps = pyscf_EandG(self.mol, self.r_ps, charges=charges)
    Gbq_ps = pyscf_bq_qgrad(mf_ps.base.mol, mf_ps.base.make_rdm1(), self.bq_coords)

    # allow, but do not drive increase of E_ps, E_rs; heavily penalize decrease
    dE_ts = E_ts - self.E_ts0
    dE_ps = E_ps - self.E_ps0
    dE_rs = E_rs - self.E_rs0
    # fixed values don't work (optim. fails), but this works well:
    c_ps = -dE_ps/0.001 if dE_ps < 0 else 0  # 0.001 Hartree ~ kB*T
    c_rs = -dE_rs/0.001 if dE_rs < 0 else 0

    print("dE_rs: %f; dE_ts: %f; dE_ps: %f" % (dE_rs, dE_ts, dE_ps))
    # d/dq(dE*dE/0.001) = 2*(dE/0.001)*d(dE)/dq
    return dE_ts - c_ps*dE_ps - c_rs*dE_rs, Gbq_ts - 2*c_ps*Gbq_ps - 2*c_rs*Gbq_rs


## optimizing catres positions
# - should be used w/ rigid residues and vdW interaction only (no charge-charge)

class Theo_Opt2:
  def __init__(self, qmmm, r0, r_rs, r_ts, r_ps, ligatoms=None):
    """ r_rs,ts,ps and E_rs,ts,ps are standalone positions and energies """
    self.qmmm, self.r_rs, self.r_ts, self.r_ps = qmmm, r_rs, r_ts, r_ps
    self.ligatoms = qmmm.qmatoms if ligatoms is None else ligatoms
    #self.r_rs = r_rs[self.ligatoms] if len(r_rs) != len(self.ligatoms) else r_rs
    # reference energies are calculated w/ large separation between LIG and rest of system
    self.E_rs0 = qmmm.EandG(setitem(np.array(r0), self.ligatoms, r_rs + 100.0), dograd=False)
    self.E_ts0 = qmmm.EandG(setitem(np.array(r0), self.ligatoms, r_ts + 100.0), dograd=False)
    self.E_ps0 = qmmm.EandG(setitem(np.array(r0), self.ligatoms, r_ps + 100.0), dograd=False)


  # we expect r to include LIG (i.e. incl all atoms)
  def __call__(self, mol, r=None):
    r = getattr(mol, 'r', mol) if r is None else r
    # role of MM energy is to prevent "collisions" between atoms
    E_rs, G_rs = self.qmmm.EandG(setitem(np.array(r), self.ligatoms, self.r_rs))
    E_ts, G_ts = self.qmmm.EandG(setitem(np.array(r), self.ligatoms, self.r_ts))
    E_ps, G_ps = self.qmmm.EandG(setitem(np.array(r), self.ligatoms, self.r_ps))
    dE_rs, dE_ts, dE_ps = E_rs - self.E_rs0, E_ts - self.E_ts0, E_ps - self.E_ps0
    # allow, but do not drive increase of E_ps, E_rs; heavily penalize decrease
    print("dE_rs: %f; dE_ts: %f; dE_ps: %f" % (dE_rs, dE_ts, dE_ps))
    # beyond this, opt hovers around |proj g| ~ 1E-3 with little improvement but severe distortion of geometry
    #if dE_ts < -0.01:
    #  raise ValueError('Stop')
    # fixed values don't work (optim. fails), but this works well:
    c_ps = -dE_ps/0.001 if dE_ps < 0 else 0  # 0.001 Hartree ~ kB*T
    c_rs = -dE_rs/0.001 if dE_rs < 0 else 0
    # d/dr(dE*dE/0.001) = 2*(dE/0.001)*d(dE)/dr
    E = dE_ts - c_ps*dE_ps - c_rs*dE_rs
    G = G_ts - 2*c_ps*G_ps - 2*c_rs*G_rs
    return E, G


class Theo_Opt:
  def __init__(self, qmmm, r_rs, r_ts, r_ps, E_rs, E_ts, E_ps, ligatoms=None):
    """ r_rs,ts,ps and E_rs,ts,ps are standalone positions and energies """
    self.qmmm, self.r_rs, self.r_ts, self.r_ps = qmmm, r_rs, r_ts, r_ps
    #self.r_rs = r_rs[self.ligatoms] if len(r_rs) != len(self.ligatoms) else r_rs
    #self.E_rs0 = pyscf_EandG(qmmm.mol, r_rs)[0]
    #self.E_ts0 = pyscf_EandG(qmmm.mol, r_ts)[0]
    #self.E_ps0 = pyscf_EandG(qmmm.mol, r_ps)[0]
    self.E_rs0, self.E_ts0, self.E_ps0 = E_rs, E_ts, E_ps
    self.ligatoms = qmmm.qmatoms if ligatoms is None else ligatoms


  # we expect r to include LIG (i.e. incl all atoms)
  def __call__(self, mol, r=None):
    r = getattr(mol, 'r', mol) if r is None else r
    qmmm = self.qmmm
    # role of MM energy is to prevent "collisions" between atoms
    # Note that only the LIG atom positions differ here, so difference in MM energies is just in LIG interaction
    qmmm.EandG(setitem(np.array(r), self.ligatoms, self.r_rs))
    dE_rs = qmmm.Ecomp.Eqm - self.E_rs0
    G_rs = qmmm.Gcomp.Gqm
    E_mm = qmmm.Ecomp.Emm
    G_mm = qmmm.Gcomp.Gmm

    qmmm.EandG(setitem(np.array(r), self.ligatoms, self.r_ts))
    dE_ts = qmmm.Ecomp.Eqm - self.E_ts0
    G_ts = qmmm.Gcomp.Gqm
    E_mm += qmmm.Ecomp.Emm
    G_mm += qmmm.Gcomp.Gmm

    qmmm.EandG(setitem(np.array(r), self.ligatoms, self.r_ps))
    dE_ps = qmmm.Ecomp.Eqm - self.E_ps0
    G_ps = qmmm.Gcomp.Gqm
    E_mm += qmmm.Ecomp.Emm
    G_mm += qmmm.Gcomp.Gmm

    # allow, but do not drive increase of E_ps, E_rs; heavily penalize decrease
    print("dE_rs: %f; dE_ts: %f; dE_ps: %f; E_mm/3: %f" % (dE_rs, dE_ts, dE_ps, E_mm/3))
    # beyond this, opt hovers around |proj g| ~ 1E-3 with little improvement but severe distortion of geometry
    #if dE_ts < -0.01:
    #  raise ValueError('Stop')
    # fixed values don't work (optim. fails), but this works well:
    c_ps = -dE_ps/0.001 if dE_ps < 0 else 0  # 0.001 Hartree ~ kB*T
    c_rs = -dE_rs/0.001 if dE_rs < 0 else 0
    # d/dr(dE*dE/0.001) = 2*(dE/0.001)*d(dE)/dr
    E = E_mm/3 + dE_ts - c_ps*dE_ps - c_rs*dE_rs
    G = G_mm/3 + G_ts - 2*c_ps*G_ps - 2*c_rs*G_rs
    return E, G


## interpolating between molecular geometries
from .opt.dlc import DLC

# this works well for CH2CHOH -> CH3CHO with dlc=DLC(mol, recalc=1) (i.e. bonds, angles, diheds internals!)
def dlc_interp(R, P, nimages, dlc=None, align_sel=None):
  """ Interpolate between geometries R and P in total connection DLC coords (or w/ passed DLC object) """
  if dlc is None:
    # DLC requires Molecule object, but anything with correct number of atoms will work for total connection
    mol = Molecule(r=R, znuc=[1]*len(R))
    dlc = DLC(mol, autobonds='total', autoangles='none', autodiheds='none', recalc=1)  # recalc is essential
  dlc.init(R)
  sR = dlc.active()
  sP = dlc.fromxyz(P)
  ds = (sP - sR)/(nimages - 1)
  s = sR
  images = [R]
  for ii in range(1, nimages-1):
    s += ds
    dlc.update(s)
    images.append(align_atoms(dlc.xyzs(), images[-1], sel=align_sel))
  images.append(P)
  return np.array(images)


## IRC: steepest descent path from TS in mass-weighted Cartesian coordinates
# mass-weighting just accounts for F = ma:  x = x0 + 0.5*a*dt^2 = x0 - step_size*G/m; G = dE/dx
# - 10.1063/1.2841941 argues this is the best way to get path once TS is found via CI-NEB or other method
# - github.com/eljost/pysisyphus implements several integration methods, incl. Hessian based methods which
#  should be faster but are more complex - see refs therein

def irc_integrate(EandG, r, dr, weight=1.0, step_size=0.1, ftol=1E-06, max_steps=100, RK4=True):
  """ use weight = 1/masses to get usual IRC in mass-weighted coordinates """
  Eirc, irc = [], []
  weight = weight if np.isscalar(weight) else weight[:,None]
  for ii in range(max_steps):
    r = r + step_size*dr  # align_atoms(r, r0) -- we may want to align when using mass weighting!
    E,G = EandG(r)
    if Eirc and (E > Eirc[-1] or (E - Eirc[-1])/Eirc[-1] < ftol):
      break
    print("IRC step %d: E = %f" % (ii, E))
    irc.append(r)
    Eirc.append(E)
    # 4th order Runge-Kutta method (RK4) - seems to work well, better than Euler, Adams-Bashforth, trapezoid
    # vs. Euler, RK4 gets much closer to endpoints and eliminates oscillations; couldn't get LQA to work
    if RK4:
      dr1 = -normalize(weight*G)
      E,G = EandG(r + 0.5*step_size*dr1)
      dr2 = -normalize(weight*G)
      E,G = EandG(r + 0.5*step_size*dr2)
      dr3 = -normalize(weight*G)
      E,G = EandG(r + step_size*dr3)
      dr4 = -normalize(weight*G)
      dr = (dr1 + 2*dr2 + 2*dr3 + dr4)/6
    else:
      dr = -normalize(weight*G)  # Euler method

  return Eirc, irc


## misc

# this isn't really useful ... maybe try w/ very small cells and mark all cells within vdW radius of any atom
# anyway, accepted value of protein density is ~1.35 g/cm^3
# 2CHT.pdb (chorismate mutase): max ~0.09 heavy atoms/Ang^3
# water at 1 g/mL: .056 mol/mL -> 0.056*NA/10^24 Ang^3 -> 3.37e22/1e24 -> 3.37e-2 -> 0.03 heavy atoms/Ang^3
def density_counts(mol, a=5.0, numdens=False):
  """ return histogram of density in g/mL, or number of heavy atoms per Ang^3 if numdens == True, for molecule
    `mol`, using cubes of side `a` Ang
  """
  extents = mol.extents()
  range = extents[1] - extents[0]
  nbins = np.array(np.ceil(range/a), dtype=np.uint)
  counts = np.zeros(nbins)
  for atom in mol.atoms:
    counts[ tuple(np.int_((atom.r - extents[0])/a)) ] += int(atom.znuc > 1) if numdens else ELEMENTS[atom.znuc].mass
  return counts


def density_hist(mol, a=5.0, numdens=False):
  counts = density_counts(mol, a, numdens)
  density = counts/(a*a*a) if numdens else counts/AVOGADRO/(a*1E-8)**3
  return np.histogram(np.ravel(density), bins='auto')  #np.max(counts)+1)


def path_ts(energies, coords):
  """ given reaction path energies and coords, return energy, coords, and tangent for highest energy state """
  i_ts = np.argmax(energies)
  dr_ts = coords[i_ts+1] - coords[i_ts-1]
  return energies[i_ts], coords[i_ts], dr_ts/np.linalg.norm(dr_ts)


def totalconn(atoms):
  return [[a0, a1] for ii, a0 in enumerate(atoms) for a1 in atoms[ii+1:]]


def RMSD_rc(r, r_rs, r_ps):
  """ calculate value of RMSD-based reaction coordinate of `r` along path from `r_rs` to `r_ps` """
  return 0.5 + 0.5*(calc_RMSD(r, r_rs) - calc_RMSD(r, r_ps))/calc_RMSD(r_rs, r_ps)
