import time
import numpy as np
import scipy.optimize
from ..molecule import calc_RMSD
#from .dlc import DLC


# Geom opt/gradient:
# - see my personal notes; next steps are to try better Hessian estimates and full BFGS (vs. L-BFGS)
# - maybe w/ a custom impl we can do things like use full Hessian for QM region but sparse for MM region?
# - if we end up needing restartability, we can just save the last N values, gradients
#  ourselves to feed back to optimizer

# Internal coords: DLC seems to be the accepted approach
# Optimizers (python unless otherwise noted):
# - SciPy, LBFGS: Fortran, not restartable; BFGS: Python
# - github.com/azag0/pyberny - succinct Python impl of Berny algorithm (geom opt from Gaussian)
# - OpenOpt
# - github.com/rpmuller/pistol - BFGS (optimize.py - earlier rev), GDIIS (need to use "controlled")
# - ASE: LBFGS, global opt
# - pDynamo: LBFGS, CG,...
# - tsse: BFGS, CG, SD - very succinct (was at http://theory.cm.utexas.edu/code/ )
# - MMTK: CG, SD
# - PyQuante - SD
# - nlpy: LBFGS, others
# - DL-FIND - Fortran
# TS search (python unless noted): can't use microit?
# - methods: coordinate scan (constrained minimization), chain of states: NEB/string, dimer method,
#  optim. e.g. P-RFO
# - [ASE](https://gitlab.com/ase/ase/): NEB
# - pDynamo: Baker/P-RFO (pCore); reaction paths (pMoleculeScripts)
# - tsse: NEB, dimer method
# Free energy methods
# - Otte (Thiel) thesis: Python code for WHAM (umbrella sampling)


# TODO: can we combine this with dlc.Cartesians? How would we limit DLC, HDLC to subset of atoms?
# TODO: we shouldn't need to pass mol to XYZ() ... although maybe we should if we are passing atoms select
# ... also, it is needed for DLCs, so maybe just pass it for consistency
class XYZ:

  def __init__(self, mol, atoms=None):
    # slice(None) as index selects all elements - note that gradfromxyz and active will return copies!
    self.atoms = atoms or slice(None)
    self.X = np.array(mol.r)

  def init(self, X=None):
    if X is not None:
      self.X[self.atoms] = X[self.atoms]
    return self

  def update(self, S):
    self.X[self.atoms] = np.reshape(S, (-1,3))
    return True

  def gradfromxyz(self, gxyz):
    return np.ravel(gxyz[self.atoms])

  def active(self):
    return np.ravel(self.X[self.atoms])

  def xyzs(self):
    return self.X


def optimize_mon(r, r0, E, G, Gc, timing):
  rmsd = calc_RMSD(r0, r)
  # RMS grad per atom; note that |proj g| printed by minimize is max(abs(grad))
  rmsgrad = np.sqrt(np.sum(G*G)/len(G))
  rmsgradc = np.sqrt(np.sum(Gc*Gc)/len(Gc))
  print("***ENERGY: {:.6f} H, RMS grad: {:.6f} H/Ang, ({:.6f} H/Ang), RMSD: {:.3f} Ang, Coord update: {:.3f}s, E,G calc: {:.3f}s".format(E, rmsgrad, rmsgradc, rmsd, timing['coords'], timing['calc']))


# default gtol, ftol just copied from scipy L-BFGS options
def optimize(mol, fn, fnargs={}, coords=None, r0=None, gtol=1E-05, ftol=2E-09, mon=optimize_mon):
  """ Given `fn` accepting `mol`, r array, and additional args `fnargs` and returning energy and gradient of
    energy, find r which minimizes energy, starting from `mol.r`, or `r0` if given.  Internal coordinates,
    e.g., can use used by passing appropriate object for `coords`
  """
  r0 = mol.r if r0 is None else r0
  coords = XYZ(mol) if coords is None else coords
  coords.init(r0)

  def objfn(S):
    t0 = time.time()
    if coords.update(S):
      r = coords.xyzs()
      t1 = time.time()
      E, G = fn(mol, r, **fnargs)
      t2 = time.time()
      gS = coords.gradfromxyz(G)
      # idea of Gc is to get cartesian grad excluding constraints
      Gc = coords.gradtoxyz(gS) if hasattr(coords, 'gradtoxyz') else np.zeros(1)  # finalize this!
      # call monitor fn, e.g. to print additional info
      mon(r, r0, E, G, Gc, dict(coords=t1-t0, calc=t2-t1, total=t2-t0))
      return E, gS
    else:
      print 'Coord breakdown for S = \n{}'.format(repr(S))
      # optimizer must restart
      raise ValueError('Coord breakdown')

  print("optimize started at %s" % time.asctime())
  while True:
    try:
      res = scipy.optimize.minimize(objfn, coords.active(),
          jac=True, method='L-BFGS-B', options=dict(disp=True, gtol=gtol, ftol=ftol))
      break
    except ValueError as e:
      if e.message != 'Coord breakdown':
        raise
      coords.init()  # reinit with last good cartesians

  if res.success:
    coords.update(res.x)
  return res, coords.xyzs()
