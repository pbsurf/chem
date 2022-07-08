from chem.io import *
from chem.mmtheo import *
from chem.vis.chemvis import *

from chem.qmmm.grid import *
from chem.vis.volume import VisVol

## MM theozyme (for lig binding)

# how would we do w/ a grid?
# - w/ mm.py: coulomb() is straightforward ... what about vdW? hydrophobic?
#  - one grid point ~ 1 atom? or much finer grid? ... assume fine grid for now
#  - calibrate our born_sasa() fn using PDB-Bind database? use Evina()? ... but would we be able to translate these fns to something that works on a fine grid?
#  - could we try multiple configurations (or just MD) with explicit waters?
#  - disallow grid points inside vdW radius of lig ... but what about placing, e.g., waters with fixed grid? vdW-like potential at each grid point (radius larger than grid point spacing, assuming grid points fill covalent radius not vdW radius) ... vdW only for placement, don't include in total energy?  or find way to distribute vdW among grid points to reproduce correct vdW energy?

# store grid complete (positions implied) or sparse (list of positions + properties)?
# - if sparse, use molecule or plain list?
# - if sparse, how to access neighboring points?

# - use a dense grid of possible positions, but do 1 atom = 1 grid point?
#  - how to constrain positions so we get a physically possible configuration?
#  - how to do so for 1 atom = many grid points? main constraint would be on total charge, i.e., sum(abs(charge)) per volume ... naive approach would give approx one constraint eqn per grid point ... just limit max charge per point!!!

# How is optim going to work?
# - maximize diff between lig and water or reactant lig and product lig
# - constraints on total charge, etc. ... "linear programming" problem?

mISJ = load_molecule('ISJ.mol2')
mPRE = load_molecule('PRE.mol2')

# generating close-packed lattices ... not really verified (esp. fcc_grid)

# https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
def hcp_grid(extents, dist=1.0):
  r = dist/2
  dd = extents[1] - extents[0]
  I = np.arange(np.ceil(dd[0]/(2*r)))
  J = np.arange(np.ceil(dd[1]/(np.sqrt(3)*r)))
  K = np.arange(np.ceil(dd[2]/(2*np.sqrt(6)*r/3)))
  x = r * np.array([ (2*i + (j+k)%2) for i in I for j in J for k in K ])
  y = r * np.array([ (np.sqrt(3)*(j + (k%2)/3)) for i in I for j in J for k in K ])
  z = r * np.array([ (2*np.sqrt(6)*k/3) for i in I for j in J for k in K ])
  return np.column_stack([x,y,z]) + extents[0]


def fcc_grid(extents, dist=1.0):
  r = dist/2
  dd = extents[1] - extents[0]
  I = np.arange(np.ceil(dd[0]/(2*r)))
  J = np.arange(np.ceil(dd[1]/(np.sqrt(3)*r)))
  K = np.arange(np.ceil(dd[2]/(2*np.sqrt(6)*r/3)))
  x = r * np.array([ (2*i + (j + k%3)%2) for i in I for j in J for k in K ])  # ???
  y = r * np.array([ (np.sqrt(3)*(j + (k%3)/3)) for i in I for j in J for k in K ])
  z = r * np.array([ (2*np.sqrt(6)*k/3) for i in I for j in J for k in K ])
  return np.column_stack([x,y,z]) + extents[0]


def coulomb_disjoint(rq1, mq1, rq2, mq2, kscale=OPENMM_qqkscale, grad=False):
  Ke = kscale*ANGSTROM_PER_BOHR
  dr = rq1[:,None,:] - rq2[None,:,:]
  dd = np.sqrt(np.sum(dr*dr, axis=2))
  qq = mq1[:,None]*mq2[None,:]
  Eqq = Ke*np.sum(qq/dd)
  return (Eqq, -Ke*np.sum((qq/dd**3)[:,:,None]*dr, axis=1)) if grad else Eqq


def lj_disjoint(r1, rad1, depth1, r2, rad2, depth2, grad=False):
  dr = r1[:,None,:] - r2[None,:,:]
  idr2 = 1.0/np.sum(dr*dr, axis=2)
  # Lorentz-Berthelot combining rules - en.wikipedia.org/wiki/Combining_rules
  r0 = 0.5*(np.ravel(rad1)[:,None] + np.ravel(rad2)[None,:])  # ravel() converts float to 1D array
  depth = np.sqrt(np.ravel(depth1)[:,None]*np.ravel(depth2)[None,:])
  lj2 = (r0*r0)*idr2
  lj6 = lj2*lj2*lj2
  Elj = 0.5*np.sum(depth*(lj6*lj6 - 2*lj6))
  Glj = -12.0*np.sum((depth*(lj6*lj6 - lj6)*idr2)[:,:,None]*dr, axis=1)
  return (Elj, Glj) if grad else Elj


# simplest approach for two molecules (e.g. reactant and product): fixed grid, remove any points colliding with either
def gridtheo2(rgrid, mol1, mol2, r1=None, r2=None, probe=0.0):
  r1 = mol1.r if r1 is None else r1
  r2 = mol2.r if r2 is None else r2
  # exclude grid points within vdW radius of lig atoms
  radii = np.array([ ELEMENTS[z].vdw_radius for z in np.r_[mol1.znuc, mol2.znuc] ]) + probe
  kd = cKDTree(rgrid)
  locs = kd.query_ball_point(np.r_[r1, r2], radii)
  # interaction between grid points and mol
  Cq1 = np.sum( get_coulomb_terms(r1, mol1.mmq, rgrid, np.ones(len(rgrid))), axis=0 )
  Cq2 = np.sum( get_coulomb_terms(r2, mol2.mmq, rgrid, np.ones(len(rgrid))), axis=0 )
  Cq = Cq1 - Cq2
  Cq[ unique(flatten(locs)) ] = 0  #rgrid = np.delete(rgrid, unique(flatten(locs)), axis=0)
  return Cq


def grid_shell(grid):
  """ expects 3D grid """
  shell = np.zeros_like(grid, dtype=bool)
  shell[:-1,:,:] |= grid[1:,:,:]
  shell[1:,:,:] |= grid[:-1,:,:]
  shell[:,:-1,:] |= grid[:,1:,:]
  shell[:,1:,:] |= grid[:,:-1,:]
  shell[:,:,:-1] |= grid[:,:,1:]
  shell[:,:,1:] |= grid[:,:,:-1]
  shell &= ~grid
  return shell


def grid_core(rgrid, r, radii):
  """ expects flattened grid - returns bool array with True for any point within radii of points r """
  locs = cKDTree(rgrid).query_ball_point(r, radii)
  return setitem(np.zeros(len(rgrid), dtype=bool), unique(flatten(locs)), True)


def grid_core_mol(rgrid, mol, r=None, probe=0.0):
  """ expects flattened grid """
  radii = np.array([ ELEMENTS[z].vdw_radius for z in mol.znuc ]) + probe
  locs = cKDTree(rgrid).query_ball_point(mol.r if r is None else r, radii)
  # interaction between grid points and mol
  return setitem(np.zeros(len(rgrid), dtype=bool), unique(flatten(locs)), True)


GRID_PTS_PER_ATOM = 20
def grid_mm(rgrid, qgrid, mol, ljr0=0.25, ljeps=0.1/KCALMOL_PER_HARTREE/GRID_PTS_PER_ATOM, bonded=False):
  molmm = SimpleMM(mol) if bonded else NCMM(mol)
  def EandG(r):
    E00, G00 = molmm(r)
    Eqq01, Gqq01 = coulomb_disjoint(r, mol.mmq, rgrid, qgrid, grad=True)
    Elj01, Glj01 = lj_disjoint(r, mol.lj_r0, mol.lj_eps, rgrid, ljr0, ljeps, grad=True)
    return E00 + Eqq01 + Elj01, G00 + Gqq01 + Glj01
  return EandG


def calc_qshell(mol1, mol2, r1=None, r2=None, hemi=True, calcdE=False, qtot=8.0, density=1/0.5):
  r1 = mol1.r if r1 is None else r1
  r2 = mol2.r if r2 is None else r2
  extents = get_extents(np.vstack([r1, r2]), pad=4.0)
  shape = grid_shape(extents, sample_density=density)
  rgrid = r_grid(extents, shape=shape)

  core1 = grid_core_mol(rgrid, mol1, r1, probe=1.4)
  core2 = grid_core_mol(rgrid, mol2, r2, probe=1.4)
  core = core1 | core2
  shell = np.ravel(grid_shell(np.reshape(core, shape)))
  rshell = rgrid[shell]
  Cq1 = np.sum( get_coulomb_terms(r1, mol1.mmq, rshell, np.ones(len(rshell))), axis=0 )
  Cq2 = np.sum( get_coulomb_terms(r2, mol2.mmq, rshell, np.ones(len(rshell))), axis=0 )
  Cq = Cq2 - Cq1

  # density = 2 -> 8 pts/Ang^3
  qmax = 1.0 * 0.1/density**3  # using ~0.1 heavy atoms/Ang^3 (typical maximum; water is 0.03 heavy/Ang^3)
  #qgrid = np.sign(Cq)*qmax
  #qtot = np.sum(np.abs(qgrid))
  # with fixed qtot:
  numq = int(qtot/qmax)
  setq = np.argsort(np.abs(Cq))[-numq:]
  qshell = np.zeros_like(Cq)
  qshell[setq] = np.sign(Cq[setq])*qmax

  if hemi:
    centroid = np.mean(rshell, axis=0)
    rmaxq = np.mean(rshell*np.abs(qshell)[:,None]/np.sum(np.abs(qshell)), axis=0) - centroid
    keep = np.dot(rshell - centroid, rmaxq) >= 0
    rshell0, qshell0 = rshell, qshell
    rshell, qshell = rshell[keep], qshell[keep]

  if not calcdE:
    return rshell, qshell

  E1 = coulomb_disjoint(r1, mol1.mmq, rshell, qshell)
  E2 = coulomb_disjoint(r2, mol2.mmq, rshell, qshell)
  return rshell, qshell, (E2 - E1)*KCALMOL_PER_HARTREE


## ---

mol1 = mISJ
mol2 = align_mol(mPRE, mISJ)
density=1/0.5

if 0:  # this is now in calc_qshell
  extents = get_extents(np.vstack([mol1.r, mol2.r]), pad=4.0)
  shape = grid_shape(extents, sample_density=density)
  rgrid = r_grid(extents, shape=shape)

  core1 = grid_core_mol(rgrid, mol1, probe=1.4)
  core2 = grid_core_mol(rgrid, mol2, probe=1.4)
  core = core1 | core2
  shell = np.ravel(grid_shell(np.reshape(core, shape)))
  rshell = rgrid[shell]
  Cq1 = np.sum( get_coulomb_terms(mol1.r, mol1.mmq, rshell, np.ones(len(rshell))), axis=0 )
  Cq2 = np.sum( get_coulomb_terms(mol2.r, mol2.mmq, rshell, np.ones(len(rshell))), axis=0 )
  Cq = Cq2 - Cq1

  # density = 2 -> 8 pts/Ang^3
  qmax = 1.0 * 0.1/density**3  # using ~0.1 heavy atoms/Ang^3 (typical maximum; water is 0.03 heavy/Ang^3)
  #qgrid = np.sign(Cq)*qmax
  #qtot = np.sum(np.abs(qgrid))
  # with fixed qtot:
  qtot = 8.0
  numq = int(qtot/qmax)
  setq = np.argsort(np.abs(Cq))[-numq:]
  qshell = np.zeros_like(Cq)
  qshell[setq] = np.sign(Cq[setq])*qmax


ff = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'ISJ.ff.xml', 'PRE.ff.xml', nonbondedMethod=openmm.app.NoCutoff)  #openmm.app.PME)
openmm_load_params(mol1, ff, charges=True, vdw=True, bonded=True)
openmm_load_params(mol2, ff, charges=True, vdw=True, bonded=True)

# start interactive
import pdb; pdb.set_trace()

rshell, qshell, dE0 = calc_qshell(mol1, mol2, mol1.r, mol2.r, hemi=False, calcdE=True)
r1s, dEs = [mol1.r], [dE0]
for ii in range(20):
  EandG = grid_mm(rshell, qshell, mol1, ljr0=0.5, bonded=True)
  res, r1 = moloptim(EandG, r0=r1s[-1], maxiter=100, verbose=-2)
  r1s.append(r1)
  rshell, qshell, dE = calc_qshell(mol1, mol2, r1s[-1], mol2.r, hemi=False, calcdE=True)
  dEs.append(dE)


vmol = Molecule(r=rshell, znuc=[0]*len(rshell))
vmol.mmq = qshell
vis = Chemvis([Mol(mol1, r1s, [VisGeom()]), Mol(vmol, [VisGeom()])]).run()

# mol1 flies off to totally separate shell


# E_lj doesn't become positive until r0 ~ 4.5 Ang; min is around r0 ~ 3.5 Ang (for probe = 1.4) ... 3 Ang is a typical min grid to mol dist
ljr0grid = 0.25
ljepsgrid = 0.1/KCALMOL_PER_HARTREE/GRID_PTS_PER_ATOM
#E_lj = lj_disjoint(mol1.r, mol1.lj_r0, mol1.lj_eps, rshell, 3.5, ljepsgrid)
EandG = grid_mm(rshell, qshell, mol1)
print(EandG(mol1.r))

E1 = coulomb_disjoint(mol1.r, mol1.mmq, rshell, qshell)
E2 = coulomb_disjoint(mol2.r, mol2.mmq, rshell, qshell)
dE = (E2 - E1)*KCALMOL_PER_HARTREE

qgrid = setitem(np.zeros(shape), np.reshape(shell, shape), qshell)
vis = Chemvis(Mol([mol1, mol2], [ VisGeom(), VisVol(qgrid, vis_type='volume', extents=extents) ]), bg_color=Color.white).run()


# find "center of absolute charge" - idea is to find hemisphere w/ least charge in qshell, then remove opposite hemisphere

centroid = np.mean(rshell, axis=0)
rmaxq = np.mean(rshell*np.abs(qshell)[:,None]/np.sum(np.abs(qshell)), axis=0) - centroid
keep = np.dot(rshell - centroid, rmaxq) >= 0
rshell0, qshell0 = rshell, qshell
rshell, qshell = rshell[keep], qshell[keep]

# create something like a funnel

# rotate to align rmaxq with z axis
dr = np.array([0,0,1])
zalign = align_vector(-rmaxq, dr)
r1 = np.dot(mol1.r, zalign)
r2 = np.dot(mol2.r, zalign)

radii12 = np.array([ ELEMENTS[z].vdw_radius for z in np.hstack([mol1.znuc, mol2.znuc]) ])
r12 = np.vstack([r1, r2])

dz = 5  # Angstrom
extents = get_extents(r12, pad=6.0)
extents[1,2] += dz  # extend along +z direction
shape = grid_shape(extents, sample_density=density)
rgrid = r_grid(extents, shape=shape)

core = np.any([grid_core(rgrid, r12 + dr*ii/density, radii12 + 1.4 + 0.2*ii/density) for ii in range(int(dz*density))], axis=0)
shell = np.ravel(grid_shell(np.reshape(core, shape)))
rshell = rgrid[shell]
rshell = rshell[rshell[:,2] < extents[1,2] - 10.0]  # need to figure this out


## NEXT:
# - waters ... how?
# - try changing relative orientation, torsions of reactant, product?
# - try a free energy method w/ grid
#   - get grid working with openmm? openmm_create_system ... I don't think we want implicit solvent with grid ... but explicit solvent outside grid doesn't seem good either

tors23 = lambda mol: set([ (a2,a3) if a3 > a2 else (a3,a2) for a1,a2,a3,a4 in mol.get_internals()[-1] ])

tors23(mol1) - tors23(mol2)

tors23(mol2) - tors23(mol1)
# ... also vary torsions "downstream" from these?

tors1 = mol1.get_internals()[-1]
# get first matching torsion
#tors = next( tors for tors in tors1 if (tors[1],tors[2]) in ??? or (tors[1],tors[2]) in ??? )

mol1.dihedral(tors, newdeg=360*np.random.random())
# - what about moving bond(), angle(), dihedral() out of Molecule class and allowing an r argument?
dEs = [ calc_qshell(mol1, mol2, r1=mol1.r, r2=r2, calcdE=True)[-1] for r2 in r2s ]

# alternatives to brute force search?
# - for fixed qshell, optim to adjust geometry for optimal dE, update qshell, repeat
# - what about optimizing charges on every grid point together with mol geometry? ... this just means reassigning charges?
# ... still want to randomize initial orientation and torsions
# ... we'll need vdW to prevent mol from getting too close to grid (and bonded terms to maintain molecular geom)

## seems like the issue is that we'll just get larger RMSD(mol1, mol2) -> larger dE ... how to constrain RMSD?
# - forget about dE(mol1, mol2) and look at waters
# - don't modify qshell geometry (only charges) - and use full qshell, not hemi!
# ... seems reasonable, but I think we'll still end up w/ max RMSD allowed by constraints (i.e., shell)
# ... if extent of molecule is roughly the same in two direction, shell won't constrain it sufficiently!
# - misalignment penalty? how to calibrate?
# - use tighter shell (w/ smaller vdW radii)?

# - shouldn't we be relaxing mols separately, then computing dE? seems like optimizing just for dE will strain molecule
#  - with fixed product mol, try to minimize energy of reactant mol?



# are we going to do 2 stage grid (find optimal hemisphere and then realign grid?)?
# - single stage setting charges w/ complete shell, then remove hemisphere and calc dE?
# - we could do a scan to get idea of how fast dE will vary with change in orientation, then do monte carlo type search?
# - can we find an analytical expression for d(dE)/dr? ... with automatic differentiation?

# limited random rotation ... for a very oblong molecule, this should depend on rotation vector
randrots = [ np.random.random(3) * [1,1,0.1] for ii in range(20) ]
dEs = [ dE_qshell(mol1, mol2, r1=mol1.r, r2=np.dot(rot, mol2.r), calcdE=True)[-1] for rot in randrots ]


# Note that global search over possible residues isn't necessarily more insightful than ML!
# - with such a large search space, it would be better to have some physical insight on which residues to pick!

vmol = Molecule(r=rshell, znuc=[0]*len(rshell))
vmol.mmq = qshell
vmol.lj_r0 = [ljr0grid]*len(rshell)
vmol.lj_eps = [ljepsgrid]*len(rshell)

gmol1 = Molecule(mol1)
openmm_load_params(gmol1, ff, charges=True, vdw=True, bonded=True)
gmol1.append_atoms(vmol)

openmm_create_system(vmol, freezemass=0)


# for gridtheo2, qgrid will obviously depend heavily on relative orientation of reactant and product ... this is something we can adjust (within a limited range) ... torsions could also be adjusted!
# - scan orientation and torsions? (we'd need to scan torsions of both reactant and product)
# - get_internals(), then scan torsions which differ between reactant and product?

# - allow two separate grids for reactant and product, w/ some cost fn penalizing differences between the grids?  But how does this model movement of residues
# - allow movement of grid points between reactant and product (so one or both no longer a true grid?); cost fn penalizing displacement; parameters for optimization would be charge and displacement of each grid point; require (w/ cost fn) displacement be similar to nearby points (using values from previous iteration)?
# ... perhaps this effect is pretty small and we can ignore it, at least for grid theozyme?  Would have to consider grid-grid energy if we allow rearrangement

# Fitting residues to qgrid:
# - assign charged residues (ARG/LYS,ASP/GLU) to charge concentrations of qgrid ... calc overlap treating MM charges as uniform charge distributions (using covalent radius?) ... this seems a little strange, since electric field is same as a point charge; collapse qgrid regions to point charges instead?
# - filling in the remaining space: estimate how many additional residues can fit using grid ... precalculate number of grid points covered by each residue (sidechain only)?
# - use grid to get some stats about voids/gaps present in real protein?
# ... maybe we should work on providing as many constraints as possible before trying to fit residues to grid (e.g. including waters)

# - could we completely fill out grid, i.e., set vdW params for every point (but with opening to access site)?
#  - place opening based on sector w/ least charge in qgrid? project lig along direction (plus some padding) to remove points

# - can we try FEP, umbrella sampling, etc. w/ grid?  We'll need vdW interaction w/ grid
#  - only use first "shell" of grid around site for vdW?
#  - could we use this shell for everything?  does this complicate fitting residues to the grid? ... qgrid doesn't really look anything like charge dist. from actual residues, so without more constraints to enforce this, perhaps shell grid is OK
# - how to allow shape of grid to vary from exact shell? multiple shell layers, w/ option for grid points to have no charge and no vdW? ... so we're back to regular grid instead of shell!  But we still don't want vdW (or charge?) on buried grid points
# ... how to determine the shape?  random placement of waters?  (use a harmonic well to confine the waters? vdW sufficient?) discard grids w/ lowest water energies?

# ... is there a way to make fine grid more like actual residues?  would we model part of sidechain? whole sidechain? whole residue?  Does this contribute anything except packing?

# - plot dE vs. qtot ... roughly linear for small qtot, eventually asymptotic - as expected
#qtots = [0.5, 1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024]


vmol = Molecule(r=rshell, znuc=[0]*len(rshell))
vmol.mmq = qshell
#vis = Chemvis([Mol([mol1, mol2]), Mol(vmol, [VisGeom(style='spacefill', radius=0.25, coloring=scalar_coloring('mmq', [-qmax,qmax]))])]).run()
vis = Chemvis([Mol([mol1, mol2]), Mol(vmol, [VisGeom(style='licorice', coloring=coloring_opacity(scalar_coloring('mmq', [-qmax,qmax]), 0.5))])]).run()


mol = mISJ
#rgrid = cp_grid(get_extents(mol.r, pad=4.0), dist=0.5)
rgrid = fcc_grid(np.array([[0,0,0], [1.5, 1.5, 1.5]]), dist=0.5)
vis = Chemvis(Mol(Molecule(r=rgrid, znuc=[1]*len(rgrid)), [VisGeom(style='spacefill', radius=np.full(len(rgrid), 0.25))])).run()


# relax close packed lattice with vdW interaction ... stays close packed as expected
hcpgrid = hcp_grid(np.array([[0,0,0], [1.5, 1.5, 1.5]]), dist=0.5)
res, r1 = moloptim(partial(lj_EandG, r0=0.5, depth=0.1/KCALMOL_PER_HARTREE), r0=hcpgrid)

# need a little random perturbation for simple cubic grid to collapse to close packed
scgrid = r_grid(np.array([[0,0,0], [1.5, 1.5, 1.5]]), 2.0)
res, r1 = moloptim(partial(lj_EandG, r0=0.5, depth=0.1/KCALMOL_PER_HARTREE), r0=scgrid + 0.1*np.random.random(scgrid.shape))
# vis = ...






#from chem.qmmm.grid import r_grid
#extents = get_extents(r_ts, pad=(d + 2.0/grid_density))
#grid = r_grid(extents, grid_density)
# vs. storing values as 3D arrays and just storing spacing and coords of grid origin

# include self-interaction of grid points?  certainly not for dense grid!

def gridtheo1(mol, r=None, density=1/0.5):
  rlig = mol.r if r is None else r
  extents = get_extents(rlig, pad=4.0)
  rgrid = r_grid(extents, sample_density=density)
  # exclude grid points within vdW radius of lig atoms
  radii = np.array([ ELEMENTS[z].vdw_radius for z in mol.znuc ])  # + probe
  kd = cKDTree(rgrid)
  locs = kd.query_ball_point(rlig, radii)
  rgrid = np.delete(rgrid, unique(flatten(locs)))
  # interaction between grid points and mol
  Cq = np.sum( get_coulomb_terms(rlig, mol.mmq, rgrid, np.ones(len(rgrid))), axis=1 )
  return Cq
  # maximize/minimize: \sum Cq_i * qgrid_i  w/ abs(qgrid_i) < qmax
  # ... soln is just to put sign(Cq_i)*qmax charge on each point ... so we need more constraints to make problem non-trivial; limitation on total charge (based on total number of grid points) ... we'd then sort grid points by abs(Cq_i) and place qmax on top qtot/qmax grid points; more constraints? minimize total charge (to minimize interaction w/ water)?



# placing waters:
# - how many waters?  initial position?
#  - add a water, relax, repeat until ... vdW energy too big? total interaction energy is worse than N-1 waters?

def fit_waters(rgrid, qgrid):

  # start with upper bound estimate, then do binary search (to find number of waters) ... just do sequential instead of binary search?
  # get extents from rgrid, then use number of missing grid points to get cavity volume

  #for nwat in range(1, 20):
    # how to place waters?

  ncmm = NCMM(hoh)
  ljr0grid = 0.25
  ljepsgrid = 0.1/KCALMOL_PER_HARTREE / GRID_PTS_PER_ATOM  # calculate scale factor from some test systems?
  def EandG(rhoh):
    E00, G00 = ncmm(rhoh)
    Eqq01, Gqq01 = coulomb_disjoint(rhoh, hoh.mmq, rgrid, qgrid, grad=True)
    Elj01, Glj01 = lj_disjoint(rhoh, hoh.lj_r0, hoh.lj_eps, rgrid, ljr0grid, ljepsgrid, grad=True)
    return E00 + Eqq01 + Elj01, G00 + Gqq01 + Glj01

  res, r1 = moloptim(EandG, r0=hoh.r)

  # given a charge grid, find water config that gives the lowest energy (optim?), then add this water config to objective fn; repeat N times; this means we need to calc coulomb and vdW between Molecule and grid
  # - should we keep previous water configs? if not, seems like we'd just oscillate between a few configs ... keep all previous water configs, compute interaction energy of each with current grid, and use the lowest (strongest) energy


  ## let's use reactant/product to start and forget about water!!!

  # how to handle removing overlapping grid points w/ reactant + product?  real protein could rearrange ... could we use separate grids for reactant and product?  How to connect?  Allow points to move while keeping charge (so no longer actually a grid)? Relax grid using just vdW? ... would turn into close packed (hcp/fcc)? start off w/ close-packed grid instead of cubic?
  # - simplest approach: align reactant and product, then remove grid points that collide with either one

  # - for fine grid, vdW radius for grid points = 0 or 0.5*grid spacing; vdW strength scaled by ~ 1/(grid points per atom); we need gradient for optim; just use mask arg to coulomb()/lj_EandG() to prevent grid self-interaction? or create separate fns? ... we still need interaction between waters, so groups would not be completely disjoint ... we could separate: water-water and water-grid ... seems reasonable to provide fns for disjoint coulomb and vdW

# alternative more like QM theo: pick grid points w/ largest Cq_i, discarding points too close to a previously choosen point ... vary min dist based on other points to be more like a molecule: each point given count of number of other points w/ covalent distance ... once this reaches 4, min dist becomes vdW radius




# - w/ openmm: grid of particles w/ charge, vdW


# what if EandG fns returned an object that supported addition (and [] indexing to preserve old behavior)
class EandG_t():
  def __init__(self, E, G):
    self.E, self.G = E,G
  def __getitem__(self, idx):
    if idx < 0 or idx > 1: raise IndexError("index out of range")
    return self.E if idx == 0 else self.G
  def __add__(self, other):
    return EandG_t(self.E + other.E, self.G + other.G)


## OLD

# - vary anchor positions? how?  assuming lig is free to rotate, only relative positions of anchors matters
#  - maybe not necessary if we pack anchor spheres to fill (most of) surrounding space
#  - how to quantify (CA) positions for natural enzymes to guide our placement of anchor points?
#  ... gather from multiple PDB files, searching around non-standard residues (i.e. ligs)

# I detect two main lines of my own thinking (neither of which is necessarily right or wrong)
# 1. incomplete/floppy site, residues/atoms moved by optimization
# 2. complete/densely packed site with some unknown mechanism for choosing and placing residues/atoms
# ... 2 seems preferable IF we can flesh it out

# levels of approx for active site: 1. plane, 2. hemispherical shell (interior) - each residue would take some area on shell (larger radius -> more residues)
# - what is typical radial deviation of CA position from best-fit shell?
# - shell may not be acceptable approximation for ligands not close to spherical

# ... set of points at vdW radius + some offset from nearest lig atom


# - pack spheres around ligand
# CA - CA distances: min ~3.8 Ang; ~4.2Ang for non-consecutive residues
# - construct a web/lattice of anchor positions connected w/ springs setting distance to these values?

# - pack spheres around lig (representing heavy atoms)?
#  - note that a design w/ significant voids (from any approach) is invalid

# - perhaps the surface area from GBSA will be helpful here?
# ... we can calculate exposed SA by subtracting overlap of vdW sphere w/ vdW spheres of nearby atoms
# - or see e.g. 10.1107/S0021889883010985
# ... then what?

# - why not use point charge grid theozyme?  we could still play with water binding with it!
# ... only if we come up with a reasonable way to convert to actual residues

## default option:
## Let's start with this (based on vinyl.py:1138), then maybe try point charges on grid
# - randomly place residues (sidechains only; oriented w/ CA away from center ... maybe include backbone later) and relax?  we could include water too
#  - this is basically what we did for QM theozyme, but with faster MM calc, we can generate more candidates
#  - add restraint fn to keep CAs minimum (and maximum?) dist from lig; use step size limit to prevent things from flying apart
#  - also constrain CAs to be >= 3.8 Ang apart
#  - also restraint to block off some range of solid angle (so lig can get in and out)
# - bias placement based on lig MM charges (favor oppositely charged residues; based on lig charge in a solid angle?)
# - dE_search/genetic algo to refine candidates?
#  - if simple mutation doesn't work well, try alchemical change (w/ soft-core vdW etc)

# - try adding a "compressing" potential to simulate fluid pressure? constant force directed toward COM?

# - generate grid roughly corresponding to solvent accessible surface of lig, use grid points for placing residues (remove points overlapped by placed residue, repeat)

# More:
# - any way to enable continuous (instead of discrete) optimization? ... something like point charge theozyme?
# - how to do from scratch (w/o preexisting pocket)?  Identify charged LIG atoms and try residues w/ oppositely charged atoms (and hydrophobic residues near uncharged LIG atoms?)


## Complete site approach

# - could we start with collection of uniform spheres and have objective fn that can continuously morph them into allowed residues - by changing radius (i.e., element), separation (i.e. bonding), charge (i.e. atom type)


## Incomplete site approach
# - residue side chains on leashes (to CA) from anchor positions (flat bottom harmonic - quartic?)
#  - how to set params? copy from a peptide w/ one of each amino acid?
# - place anchor points randomly some fixed distance from nearest lig atom and some minimum dist from nearest anchor point; set of points subtend some maximum solid angle (... from COM of ligand?)
#  - just generate large number of random points, then eliminate points that don't meet conditions

# - how to prevent nearby oppositely charged residues from associating w/ each other instead of lig
#  - tweak constraints? use only tip of sidechain w/ tighter constraints? disable interaction using NonbondedForce exceptions?

# thoughts: we could basically place charged residues manually - I feel like one of the key roles of automation would be tweaking positions to make displacement of waters more favorable
# ... but we need automated code anyway as a starting point


# Review of previous work (ts_test.py, vinyl.py):
# 1. point charge theozyme:
#  - find N points on grid w/ greatest effect on E_ts - E_rs
#  - place charges at those points and vary charge values to minimize E_ts - E_rs
# 2. manually choose residues and place at point charge positions above; optimize positions
# 3. randomly place additional residues and relax; repeat M times (to find best candidates)

# So what do we want to do differently now?
# - automate selection of residues
# - use just MM for much faster iterations
# - ultimately, have a fully automated process
