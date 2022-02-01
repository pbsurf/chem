# In the simplest case, ESP charge fitting is just a linear least squares problem
# ESP at r_j due to point charges q_i at R_i (*in Bohr*):
#  phiq(r_j) = \sum_i A_ji q_i, w/ A_ji = 1/(r_j - R_i)
# Least squares:
#  chi^2 = \sum_j w_j*(phiq(r_j) - phi0(r_j))^2, where phi0(r_j) is input points/grid, w_j is weight
#        = || w*(A*q - phi0) ||^2
# Restraints (quadratic):
#  append to A, phi0; e.g. A_{(jmax + i),k} = delta_ik  phi0_(jmax + i) = 0 to restrain to zero
# Constraints:
#  \sum_i B_ki q_i = c_k
#
# Solution:
# general solution to constraint problem using pseudoinverse:
#  q = B^+ * c + (1 - B^+ * B) * q1, q1 arbitrary
# substitution gives
#  chi^2 = || A * (B^+ * c + P * q1) - phi0 ||^2 w/ P = (1 - B^+ * B)
# this is just the least squares problem for A1*q1 = phi1, w/ soln:
#  q1 = A1^+ * phi1 + (1 - A1^+ * A1) * x w/ A1 = A*P, phi1 = phi0 - A * B^+ * c
# take x = 0 and plug into above expression for q.
#
# Refs:
# - MMTK ChargeFit.py: rank checking, constraint consistency check, random grid generation
# - pDynamo ESPChargeFitting.py: vdW surface grid generation (w/ Lebedev(-Laikov) grids, fancier than I think
#   what Gaussian does), iterative and non-iterative soln
# - github.com/lilyminium/psiresp - RESP, RESP2, multiple conformations
# More:
# - github.com/lpwgroup/respyte - RESP2 (average of gas-phase and solvent RESP charges)
# - github.com/jszopi/restrained-ESP-fit - original Fortran code
# - github.com/cdsgroup/resp
# - GAMESS prplib, prpel
# - Ambertools (<http://ambermd.org/Questions/resp2.txt>)
# - 10.1021/j100142a004 - orig. RESP Paper (1993) ; 10.1021/ja00074a030 (1993) ; 10.1002/jcc.540110404
# - 10.1021/ja00124a002 - orig. AMBER paper (1995)
# Notes:
# - RESP as implemented for AMBER/GAFF: hyperbolic restraints (i.e., constant weight above some charge value,
#  vs. linear weight w/ harmonic restraints) on non-hydrogens (no restraint on H) requiring iterative soln,
#  two stage fit (2nd stage just for -CH3, constraining H atom charges to be equal)
# - Charge assignment/ESP fitting is a huge topic within the even bigger area of force field parameterization,
#  neither of which we should try to reinvent ... so while this code maybe useful for micro-iterative QM/MM,
#  for proper small molecule parameterization use GAFF via Antechamber from AmberTools
# - Saw an interesting aside somewhere that harmonic restraint is equiv. to Gaussian prior distrib. (Bayesian)
# - Chemically equivalent atoms: required to have same MM charge (unless they don't need to move); may be
#  better to average charges after fit instead of constraining fit (esp. for -CH3)
#  -  www.blopig.com/blog/2021/01/calculating-symmeterised-small-molecule-rmsds-using-graph-automorphisms-in-python-with-gemmi-and-networkx/
#  - also github.com/molmod/molmod/blob/master/molmod/graphs.py - equivalent_vertices()
#  - also github.com/choderalab/ambermini/blob/master/antechamber/equatom.c (hard to decipher)

import numpy as np
from ..basics import *
from ..data.elements import ELEMENTS
from .grid import *


def grid_resp(centers, grid, phi0, gridweight=None, restrain=None, netcharge=0.0, equiv=[], verbose=True):
  """ RESP: linear least squares determination of point charge values at specified positions to best
   reproduce electrostatic potential (ESP) at specified grid points subject to total charge constraint and
   optional restraints.  To use multiple conformations for fit, pass lists for centers, grid, and phi0
  Passed:
   centers: point charge positions
   grid: grid points (not necessarily in a regular grid...)
   phi0: ESP at grid points
   gridweight: weights to apply to grid points (None for no weights)
    == 'equal' to weight all grid points equally (weight = 1/phi0)
    otherwise, should be a vector (scalar weight has no effect)
   restrain: weight of harmonic restraint toward q=netcharge/ncenters (scalar or vector); 0.1 is reasonable
   netcharge: net charge (constrained) - for now, this cannot be disabled
    since we need at least one constraint to create B and c matrices
   equiv: pairs of equivalent atoms constrained to have equal charge (needed iff significant motion of
    molecule will be allowed); for H in CH3 and CH2, may be better to average charges after fit instead
  verbose: print goodness-of-fit metrics if True
  Returns:
   q: the point charge values
  """
  # support fitting multiple conformations
  if np.ndim(phi0) > 1:
    dr = np.stack([grid[ii][:,None,:] - r[None,:,:] for ii,r in enumerate(centers)])
    phi0 = np.ravel(phi0)
  else:
    dr = grid[:,None,:] - centers[None,:,:]
  ncenters = len(centers)
  npoints = len(phi0)
  # ESP points (restraints)
  A = ANGSTROM_PER_BOHR/np.sqrt(np.sum(dr*dr, axis=2))  # unbelievable
  # ESP point weighting - weighting is applied to A and phi0 before main calculation; alternatively, we could
  #  form a weight matrix which multiples A and phi0 in main calc.
  if gridweight == 'equal':
    A = 1.0/phi0 * A
    phi0 = np.ones_like(phi0)
  elif gridweight is not None:
    A = gridweight * A
    phi0 = gridweight * phi0
  # charge restraints - idea is to prevent excess charge on poorly determined interior atoms
  if restrain is not None:
    Ar = np.eye(ncenters)*restrain if np.isscalar(restrain) else np.diag(restrain)
    Ar = Ar[~np.all(Ar == 0, axis=1)]  # remove null rows
    A = np.vstack((A, Ar))
    phi0 = np.hstack((phi0, np.ones(len(Ar))*netcharge/ncenters))
  # constraints: B, and c
  B, c = [], []
  #if netcharge is not None:
  B.append( np.ones(ncenters) )
  c.append( netcharge )
  equivpairs = [[a,b] for eqv in equiv for ii,a in enumerate(eqv) for b in eqv[ii+1:]]
  for eqv in equivpairs:
    B.append(setitem(np.zeros(ncenters), eqv, [1,-1]))  #list(eqv) -- list() no longer needed
    c.append(0)
  B, c = np.array(B), np.array(c)
  # alternative calculation: (A.T.*A).*q = A.T.*phi0
  AtA = np.block([[np.dot(A.T, A), B.T], [B, np.zeros((len(B), len(B)))]])
  Atphi = np.block([np.dot(A.T, phi0), c])
  ql = np.dot(np.linalg.inv(AtA), Atphi)  # or: ql = np.linalg.solve(AtA, Atphi)
  q = ql[:ncenters]  # remaining components are the Lagrange multipliers
  # this version isn't satisfying constraints very well (numerical issues?)
  #Binv, Binvcond = MPinv(B);
  #P = np.eye(len(Binv)) - np.dot(Binv, B);
  #A1inv, A1invcond = MPinv(np.dot(A, P));
  #phi1 = phi0 - np.dot(A, np.dot(Binv, c));
  #q1 = np.dot(A1inv, phi1);
  #q = np.dot(Binv, c) + np.dot(P, q1);
  # goodness of fit: reduced chi-squared excluding restraints and relative RMS error
  if verbose:
    eps = phi0[:npoints] - np.dot(A[:npoints,:], q)
    chisq = np.dot(eps.T, eps)
    print("RESP fit: reduced chi-squared = %f; rel. RMS error = %f" % (
        chisq/(npoints - ncenters), np.sqrt(chisq/np.sum(phi0[:npoints]**2)) ))
  return q


def MPinv(A, rcond=1e-15):
  """ calculate Moore-Penrose pseudoinverse of matrix A using SVD. Code is copied from numpy.linalg.pinv, but
   modified to calculate and return condition number (ratio of largest and smallest singular values from SVD)
  """
  u, s, vT = np.linalg.svd(np.conjugate(A), full_matrices=False)
  cond = np.amax(s)/np.amin(s)  # condition number
  cutoff = np.asarray(rcond)[..., None] * np.amax(s, axis=-1, keepdims=True)
  large = s > cutoff
  s = np.divide(1, s, where=large, out=s)
  s[~large] = 0
  res = np.dot(vT.T, np.multiply(s[..., None], u.T))
  return res, cond


# "Merz-Kollman" vdW radii from FMOPRP in GAMESS fmoio.src
MK_VDW_RADII = {1: 1.2, 6: 1.5, 7: 1.5, 8: 1.4, 11: 1.57, 12: 1.36, 15: 1.80, 16: 1.75, 17: 1.70}

# CHELPG eliminates points within VdW radius or outside 2.8 A, but I think I'll use 2*VdW instead
def chelpg_grid(r, znuc, density=5.0, random=False, minr=1.0, maxr=2.0):
  """ generate cubic grid (or randomly placed points if `random` == True) with `density`**3 points per Ang**3
    between `minr` and `maxr` times VdW radii of atoms with atomic number `znuc` at positions `r`
  """
  vdw = [ELEMENTS[z].vdw_radius for z in znuc]  #[MK_VDW_RADII.get(z, ELEMENTS[z].vdw_radius) for z in znuc]
  extents = get_extents(r, pad=maxr*np.max(vdw))  # we could be a little fancier but this is OK I think
  shape = grid_shape(extents, density)
  grid = (extents[1] - extents[0])*np.random.random((np.prod(shape), 3)) + extents[0] if random else \
      r_grid(extents, shape=shape)

  # don't think we can easily use kd-tree since different radius for each element
  d2 = [ np.sum((grid - rr[None,:])**2, axis=1) for rr in r]  #ii, rr in enumerate(r) ]
  gtmin = np.all([dd2 > (minr*vdw[ii])**2 for ii, dd2 in enumerate(d2)], axis=0)
  ltmax = np.any([dd2 <= (maxr*vdw[ii])**2 for ii, dd2 in enumerate(d2)], axis=0)
  return grid[gtmin & ltmax]


## Not sure this is worth keeping when we have cubic grid and random options already
# Merz-(Singh-)Kollman grid scheme: nested shells of grid points 1.4,1.6,1.8,2.0 times approx vdW radius (1.2
#  for H, 1.4 for O, 1.5 for C, N) from nearest atom
# Gaussian seems to default to 0.25 Ang separation between points in a given shell, but shells might be closer
#  together than in original MK scheme
# From GAMESS via github.com/lilyminium/psiresp
def sphere_points(n):
  """ generate n points on surface of unit sphere """
  n_lat = int((np.pi*n)**0.5)
  n_long = int((n_lat/2))
  fi = np.arange(n_long+1)*np.pi/n_long
  z, xy = np.cos(fi), np.sin(fi)
  n_horiz = (xy*n_lat+1e-10).astype(int)
  n_horiz = np.where(n_horiz < 1, 1, n_horiz)
  dots = np.empty((sum(n_horiz), 3))
  dots[:, -1] = np.repeat(z, n_horiz)
  XY = np.repeat(xy, n_horiz)
  fjs = np.concatenate([2*np.pi*np.arange(j)/j for j in n_horiz])
  dots[:, 0] = np.cos(fjs)*XY
  dots[:, 1] = np.sin(fjs)*XY
  return dots[:n]


def connolly_spheres(radii, density=1.0, npts=None):
  rads, inv = np.unique(radii, return_inverse=True)
  npts = ((rads**2)*np.pi*4*density).astype(int) if npts is None else npts
  points = [sphere_points(n)*r for n, r in zip(npts, rads)]
  all_points = [points[ii] for ii in inv]
  return all_points


def connolly_mk_grid(r, znuc, nshells=4, density=1.0):
  vdw = np.array([MK_VDW_RADII.get(z, ELEMENTS[z].vdw_radius) for z in znuc])
  grid = []
  # having different numbers of points in each shell (i.e., same density) makes visual inspection difficult
  #npts = [int((2.55**2)*np.pi*4*density)]*len(vdw)
  for scale in np.linspace(1.4, 2.0, nshells):
    sphs = connolly_spheres(scale*vdw, density)  #npts=npts)
    # can't do r + sphs because spheres don't have same number of points
    shell = np.vstack([rr + sph for rr, sph in zip(r, sphs)])
    gt = [ np.sum((shell - rr[None,:])**2, axis=1) > (0.99*scale*vv)**2 for rr, vv in zip(r, vdw) ]
    grid.append(shell[np.all(gt, axis=0)])

  return np.vstack(grid)


# not sure if we can change orientation of molecule between pyscf HF calc and ESP calc, so orient before:
# masses = np.array([ELEMENTS[z].mass for z in mol.znuc])
# mol.r = np.dot(align_axes(principal_axes(moment_of_inertia(mol.r, masses)), np.eye(3)), mol.r)

# to restrain only heavy atoms (the usual case):  restrain=(mol.znuc > 1)*0.1

def resp(mf, verbose=True, **kwargs):
  """ RESP charge fit based on pyscf QM result `mf` """
  r = mf.mol.atom_coords(unit='Ang')
  grid = chelpg_grid(r, mf.mol.atom_charges())
  esp = pyscf_esp_grid(mf, grid)
  if verbose:
    print("Compare Mulliken charges:\n%r" % mf.mulliken_pop(verbose=0)[1])
  return grid_resp(r, grid, esp, verbose=verbose, **kwargs)


# multiple conformations
def multi_resp(mol, conformations):
  from ..io.pyscf import pyscf_EandG
  for r in conformations:
    _, _, scn = pyscf_EandG(mol, r)
    grids.append(chelpg_grid(r, mol.znuc))
    esps.append(pyscf_esp_grid(scn.base, grids[-1]))

  return grid_resp(conformations, grids, esps)
