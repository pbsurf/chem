import numpy as np
from pyscf import gto
from ..basics import ANGSTROM_PER_BOHR

# calculating MOs and density on grid
# - also could try https://github.com/orbkit/orbkit

## generic fns

# Note that formatting extents as [[xmin, ymin, zmin], [xmax, ymax, zmax]] is much better for vectorized
#  operations vs. [[xmin, xmax], ...]
def f_grid(f, extents=[(-1, -1, -1), (1, 1, 1)], samples=20, max_eval=None):
  """ evaluate fn `f` on a cubic grid, optionally evaluating `max_eval` z-slices at time to save memory """
  if np.size(samples) == 1:
    samples = np.tile(samples, 3)
  samples = samples.round().astype(np.int).tolist()
  x = np.linspace(extents[0,0], extents[1,0], samples[0])
  y = np.linspace(extents[0,1], extents[1,1], samples[1])
  z = np.linspace(extents[0,2], extents[1,2], samples[2])

  if not max_eval:
    return f(x[:,None,None], y[None,:,None], z[None,None,:])  # almost twice as fast as meshgrid approach

  res = np.empty(samples)
  for start in range(0, samples[2], max_eval):
    print("Evaluating chunks %d - %d of %d..." % (start, min(samples[2], start+max_eval), samples[2]))
    res[:,:,start:start+max_eval] = f(x[:,None,None], y[None,:,None], z[None,None,start:start+max_eval])
  return res


def r_grid(extents, sample_density):
  """ return flattened grid of (x,y,z) values filling `extents` with `sample_density` """
  shape = np.array((extents[1] - extents[0])*sample_density, dtype=np.int32)
  x = np.linspace(extents[0,0], extents[1,0], shape[0])
  y = np.linspace(extents[0,1], extents[1,1], shape[1])
  z = np.linspace(extents[0,2], extents[1,2], shape[2])
  # broadcast_arrays is basically what meshgrid does
  return np.array(np.broadcast_arrays(x[:,None,None], y[None,:,None], z[None,None,:])).reshape(3,-1).T


def esp_grid(q, r, grid):
  """ calculate electrostatic potential at `grid` points from charges `q` at positions `r` """
  return np.sum(q / np.linalg.norm(r - grid[:,None,:], axis=2), axis=1)


# Note this does not give correct valence for transition metals!
def nvalence(znuc):
  for n in [2, 8, 8, 10, 8, 10, 8]:
    if znuc <= n:
      return znuc
    znuc -= n


## pyscf calculation

# use: mol.mo_grid, mol.dens_grid = pyscf_mo_grid(mf, mol.mo_extents, mo.sample_density, density=True)
# - previously, n_calc_mos was last dimension of mo_grid ... calc time was comparable, but access time slower
def pyscf_mo_grid(mf, extents, sample_density=10.0, mo_coeff=None, n_calc=None, density=False, max_memory=2**30):
  """ calculate `n_calc` (default 2*n_occ) MOs and optionally density for pyscf scf object `mf` over grid
    specified by 3D `extents` and `sample_density`.  Default of `mf.mo_coeff` can be overriden with
    `mo_coeff` for LMOs, etc.  Use at most `max_memory` bytes for AO grid (total memory use will be higher)
  """
  homo = np.max(np.nonzero(mf.mo_occ))
  n_calc_mos = n_calc or 2*homo
  max_eval = max_memory/mf.mol.nao_cart()/8  # nao_cart always >= nao_nr
  mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
  if density == 'valence':
    ncore = sum([znuc - nvalence(znuc) for znuc in mf.mol.atom_charges()])
    density = slice(ncore/2, homo+1)
  elif density is True:
    density = slice(0, homo+1)

  shape = [n_calc_mos,0,0,0]
  shape[1:] = np.array((extents[1] - extents[0])*sample_density, dtype=np.int32)
  # pyscf eval_gto requires coordinates in Bohr, and does not convert even if mol.unit == 'angstrom'
  extents = extents/ANGSTROM_PER_BOHR
  x = np.linspace(extents[0,0], extents[1,0], shape[1])
  y = np.linspace(extents[0,1], extents[1,1], shape[2])
  z = np.linspace(extents[0,2], extents[1,2], shape[3])
  # broadcast_arrays is basically what meshgrid does
  grid = np.array(np.broadcast_arrays(x[:,None,None], y[None,:,None], z[None,None,:])).reshape(3,-1).T
  mo_grid = np.empty( (n_calc_mos, len(grid)) )
  ao_type = 'GTOval_cart' if mf.mol.cart else 'GTOval_sph'
  for start in range(0, len(grid), max_eval):
    ao_grid = mf.mol.eval_gto(ao_type, grid[start:start+max_eval])
    # since n_calc_mos will typically be 1/3 to 1/2 of n_mos (== n_aos), better to calculate now and save
    #  rather than saving ao_grid and calculating MOs as needed
    mo_grid[:, start:start+max_eval] = np.dot(ao_grid, mo_coeff[:, :n_calc_mos]).T

  if density:
    occ_mo_grid = np.sqrt(mf.mo_occ[density,None])*mo_grid[density]
    dens_grid = np.sum(occ_mo_grid*occ_mo_grid, axis=0)
    return mo_grid.reshape(shape), dens_grid.reshape(shape[1:])
  return mo_grid.reshape(shape)


# untested; mainly for using pyscf to calculate MOs on grid for visualization
def cclib_to_pyscf(mol_cc):
  """ cclib molecule to pyscf molecule ... assumes GAMESS output for now; returns pyscf molecule and list
    of indices to reorder cclib/GAMESS mocoeffs to match AO order of pyscf molecule  """
  lett_to_num = dict((letter, ii) for ii, letter in enumerate('spdfgh'))
  atom_names = [ ELEMENTS[znuc].symbol + str(ii) for ii, znuc in enumerate(mol_cc.atomnos) ]
  atoms_gto = [ [atom_names[ii], r] for ii, r in enumerate(mol_cc.atomcoords[-1]) ]
  basis_gto = dict( (atom_names[ii], [ [lett_to_num[l.lower()]] + coeffs for l, coeffs in atom_basis]) \
      for ii, atom_basis in enumerate(mol_cc.gbasis) )
  mol_gto = gto.M(atom=atoms_gto, basis=basis_gto, cart=True)
  # we should use mol_gto.ao_labels() to reorder mol_cc.mocoeffs, but we will assume
  #  dxx dxy dxz dyy dyz dzz for pyscf and dxx dyy dzz dxy dxz dyz for GAMESS
  mo_reorder = []
  for ii, atom_basis in enumerate(mol_cc.gbasis):
    for letter, coeffs in atom_basis:
      l_num = lett_to_num[letter.lower()]
      if l_num > 2:
        raise Exception("Only s, p, d orbitals supported")
      reorder = [ [0], [0,1,2], [0,3,4,1,5,2] ][l_num]
      mo_reorder.extend( [ii + len(mo_reorder) for ii in reorder] )

  return mol_gto, mo_reorder


## pyquante2 calculation
# Note that pyquante uses Bohr, not Angstrom!

def getbfs(coords, gbasis):
  """ copied from pyquante2 to change order of D orbitals to match GAMESS """
  from pyquante2.basis.cgbf import cgbf

  sym2powerlist = {
    'S' : [(0,0,0)],
    'P' : [(1,0,0),(0,1,0),(0,0,1)],  # X, Y, Z
    'D' : [(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)]  # XX, YY, ZZ, XY, XZ, YZ
    # verify order of F orbitals matches GAMESS
    #'F' : [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),(0,3,0),(0,2,1),(0,1,2),(0,0,3)]
  }

  bfs = []
  for i, at_coords in enumerate(coords):
    bs = gbasis[i]
    for sym,prims in bs:
      for power in sym2powerlist[sym]:
        bf = cgbf(at_coords,power)
        for expnt,coef in prims:
          bf.add_pgbf(expnt, coef, renormalize=False)  # ... no major effect
        bf.normalize()
        bfs.append(bf)

  return bfs


# cclib.method.volume provides functions to compute orbitals and density on grid, but they use
#  PyQuante v.1, which is extremely slow since it is not vectorized.  We use v.2 which is vectorized
def molecular_orbital(coords, mocoeffs, gbasis):
  """ Return fn to calculate MO wavefn at a point x,y,z given mocoeffs and gbasis as loaded by cclib """
  # Making a closure
  def f(x, y, z, coords=coords, mocoeffs=mocoeffs, gbasis=gbasis):
    return sum(c * bf(x,y,z) for c, bf in zip(mocoeffs, getbfs(coords, gbasis)))
  return f


# TODO: if number of samples exceeds GPU limits, render as multiple volumes (isosurface only)
# mo_vol = np.log(1.0 + mo_vol)
def get_mo_volume(mol_cc, mo_number, extents, sample_density=10, mocoeffs=None):
  mocoeffs = mol_cc.mocoeffs[0] if mocoeffs is None else mocoeffs
  f = molecular_orbital(mol_cc.atomcoords[0]/ANGSTROM_PER_BOHR, mocoeffs[mo_number], mol_cc.gbasis)
  return f_grid(f, extents=extents/ANGSTROM_PER_BOHR, samples=(extents[1] - extents[0])*sample_density)


def electron_density(coords, mocoeffs_list, gbasis):
  bfs = getbfs(coords, gbasis)
  def f(x, y, z):
    aos = [bf(x,y,z) for bf in bfs]
    # 2* accounts for and assumes two electrons per orbital
    return 2*sum(np.square(sum(c * ao for c, ao in zip(mocoeffs, aos))) for mocoeffs in mocoeffs_list)
  return f


# testing: result seems to match Molden visually
# mos can be a slice object (should fix print statement so index array works too) or 'valence', in which case
#  <num core electrons>/2 lowest energy orbitals are excluded
# - what if we exclude core AOs (e.g. first s AO for row 2 elements) by zeroing rows in mocoeffs instead?
def get_dens_volume(mol_cc, extents, sample_density=100, mos=None, max_memory=2**30):
  if mos == 'valence':
    ncore = sum([znuc - nvalence(znuc) for znuc in mol_cc.atomnos])
    mos = slice(ncore/2, mol_cc.homos[0]+1)  # num core electrons/2
  elif mos is None:
    mos = slice(0, mol_cc.homos[0]+1)

  samples = (extents[1] - extents[0])*sample_density
  # all AOs are calculated on grid and saved, so determine number of z-slices of grid can be calculated at
  #  at a time to roughly limit memory needed for AO storage to max_memory
  nao = len(mol_cc.mocoeffs[0][0])
  max_eval = int(max_memory/(8.0*nao*samples[0]*samples[1]))
  if max_eval < 1:
    print("Warning: there may not be sufficient memory to evaluate AOs!")
    max_eval = 1

  print("Calculating electron density for MOs {} to {} inclusive ...".format(mos.start, mos.stop-1))
  f = electron_density(mol_cc.atomcoords[0]/ANGSTROM_PER_BOHR, mol_cc.mocoeffs[0][mos], mol_cc.gbasis)
  return f_grid(f, extents=extents/ANGSTROM_PER_BOHR, samples=samples, max_eval=max_eval)


# according to the "atoms-in-molecules" approach, the Laplacian of the electron density can provide some
#  insight into chemical bonding - see http://www.cmbi.ru.nl/molden/laplacian.html
def laplacian(vol):
  import scipy.ndimage.filters
  return scipy.ndimage.filters.laplace(vol)
