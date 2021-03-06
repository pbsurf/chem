import numpy as np
from scipy.spatial.ckdtree import cKDTree
from ..molecule import *
from ..data.pdb_bonds import PDB_PROTEIN


## fns for preparing Molecule for QM/MM calculations

# sidechain charge at pH 7.4
#sidechain_hydrogens = dict(ARG=11, ASP=2, GLU=4, HIS=5, LYS=11)
sidechain_charge = dict(ARG=1.0, ASP=-1.0, GLU=-1.0, HIS=0.0, LYS=1.0)
# sidehchain pKa; source: https://en.wikipedia.org/wiki/Protein_pKa_calculations
sidechain_pKa = dict(ARG=12.0, ASP=3.9, GLU=4.3, HIS=6.08, LYS=10.5)

# see https://en.wikipedia.org/wiki/Protein_pKa_calculations for more advanced analysis
# also https://github.com/jensengroup/propka-3.1
def protonation_check(mol, pH=7.4):
  """ check protonation states of sidechains and report possible errors for manual investigation """
  ok = True
  for ii, res in enumerate(mol.residues):
    if res.name not in PDB_PROTEIN:
      continue
    expect_q = sidechain_charge.get(res.name, 0.0)
    # Note that we checking backbone and sidechain charge separately does not work
    # N terminal end - expect H1, H2, H3 if pH < 9 (+1 charge)
    prev_res = mol.residues[ii-1] if ii > 0 else None
    if not prev_res or res.chain != prev_res.chain or prev_res.name not in PDB_PROTEIN:
      expect_q += 1.0 if pH < 9.0 else 0.0
      if len([jj for jj in res.atoms if mol.atoms[jj].name in ['H', 'H1', 'H2', 'H3']]) != 3:
        print("Unexpected protonation of N-term residue %s %d" % (res.name, ii))

    # C terminal end - no HXT unless pH < 2 (-1 charge)
    next_res = mol.residues[ii+1] if ii + 1 < len(mol.residues) else None
    if not next_res or res.chain != next_res.chain or next_res.name not in PDB_PROTEIN:
      expect_q += -1.0 if pH > 2.0 else 0.0
      if any(mol.atoms[jj].name == 'HXT' for jj in res.atoms):
        print("Unexpected HXT on C-term residue %s %d" % (res.name, ii))

    # pH adjustment
    pKa = sidechain_pKa.get(res.name, None)
    if pKa is not None:
      if pH < pKa and pKa < 7.4:
        expect_q += 1.0
      elif pH > pKa and pKa > 7.4:
        expect_q += -1.0

    res_q = np.sum([mol.atoms[jj].mmq for jj in res.atoms])
    if abs(res_q - expect_q) > 0.01:
      print("Expected %+.1f residue charge, but found %+.3f on residue %s %d" % (expect_q, res_q, res.name, ii))
      ok = False
    elif pKa is not None and abs(pH - pKa) < 1.0:
      print("pH - pKa = %.2f for residue %s %d - consider protonation state" % (pH - pKa, res.name, ii))
      ok = False

  return ok


def geometry_check(mol, r=None, min_angle=np.pi/2):
  """ check molecule for unusual bonds or angles (by default < 90 deg - reasonable threshold for proteins) """
  r = mol.r if r is None else r
  bonds, angles, diheds = mol.get_internals()
  badangles = [a for a in angles if mol.angle(a) < min_angle]
  for a in badangles:
    print("Warning: angle {} = {:.2f} degrees".format(a, 180*mol.angle(a)/np.pi))
  ck = cKDTree(r)
  tooclose = ck.query_pairs(0.75)
  for pair in tooclose:
    print("Warning: distance {} = {:.3f} \u212B".format(pair, mol.bond(pair)))
  guessed_bonds = guess_bonds(r, mol.znuc)
  bonds_set, guessed_set = frozenset(bonds), frozenset(guessed_bonds)
  extra_bonds = bonds_set - guessed_set
  for extra_bond in extra_bonds:
    print("Warning: unexpected bond {}".format(extra_bond))
  missing_bonds = guessed_set - bonds_set
  for b in missing_bonds:
    print("Warning: possibly missing bond {} ({:.2f} \u212B)".format(b, mol.bond(b)))
  return list(set(tooclose) | extra_bonds | missing_bonds), badangles


def antibunch(rcs, ncopies):  #pbcbox=None
  """ return bool array w/ True for `ncopies` points in `rcs` most separated from neighbors """
  # brute force pair distance calc runs out of memory for larger systems (but handles PBC, unlike kd-tree)
  #pairs = np.triu_indices(len(rcs), 1)
  #dr = np.abs((rcs[:,None,:] - rcs[None,:,:])[pairs])  # calc distance w/ PBCs
  #dists = np.linalg.norm(np.min([dr, np.abs(dr - (extents[1] - extents[0]))], axis=0), axis=1)
  #rmpairs = np.transpose(pairs)[np.argsort(dists)]
  kd = cKDTree(rcs)
  dists, locs = kd.query(rcs, k=[2,3,4,5])
  #distance_upper_bound=2*(np.prod(extents[1,:] - extents[0,:])/ncopies)**(1/3))
  pairs = np.reshape(np.dstack([locs.T, [range(len(rcs))]*locs.shape[1]]), (-1,2))
  rmpairs = pairs[np.argsort(np.ravel(dists.T))]
  nkeep = len(rcs)
  keep = np.ones(nkeep, dtype=bool)
  for a,b in rmpairs:
    if keep[a] and keep[b]:
      keep[a] = False
      nkeep -= 1
      if nkeep == ncopies:
        break
  return keep


# Tinker's build solvent box function (xyzedit option 19) places molecules at random positions w/ random
#  orientations; no effort to avoid clashes ... so we can just do it ourselves!
# See tile_mol() in dlc_test.py for an alternative approach (place molecules on grid points, optionally with
#  random offset and rotation)
def solvent_box(mol, ncopies, extents, overfill=1.5):
  """ return Molecule with `ncopies` copies of `mol` in box specified by sides or bounds `extents`
    if `overfill` > 1, overfill*ncopies copies of mol are placed, then (overfill-1)*ncopies copies closest to
    neighbors are removed; pass overfill < 0 to skip removal of extra copies
  """
  extents = np.asarray([extents]*3 if np.isscalar(extents) else extents)
  extents = np.array([-0.5*extents, 0.5*extents]) if np.size(extents) == 3 else extents
  r0 = mol.r - center_of_mass(mol)
  solvent = Molecule()
  rcs = (extents[1] - extents[0])*np.random.rand(int(abs(overfill)*ncopies), 3) + extents[0]  # centers
  if overfill > 1:
    rcs = rcs[ antibunch(rcs, ncopies) ]
  for rc in rcs:
    solvent.append_atoms(mol, r=np.dot(r0, random_rotation()) + rc)

  return solvent


def water_box(sides, mmtypes=None, overfill=1.5):
  from .build import TIP3P_xyz
  from ..io import load_molecule
  tip3p = load_molecule(TIP3P_xyz, center=True, residue='HOH')
  if mmtypes is not None:
    tip3p.mmtype = mmtypes  #[2001, 2002, 2002] -- Tinker Amber types
  # for water
  molar_mass = 18.01488 # amu == g/mol ... just sum of atomic weights
  density = 0.9982 # g/cm^3 at 20C
  sides = np.array([sides]*3 if np.isscalar(sides) else sides)
  vol_cm3 = np.prod(sides)*1E-24  # 1 Ang^3 = 1E-24 cm^3
  nwaters = AVOGADRO*density*vol_cm3/molar_mass
  solvent = solvent_box(tip3p, int(round(nwaters)), sides, overfill=overfill)  #overfill=3.0 might work better
  solvent.pbcbox = sides
  #~water_cube_side = 1E8*(molar_mass/(AVOGADRO*density))**(1/3.0)  # in Ang
  #~# Tinker periodic box seems to be centered at origin
  #~nwaters_side = np.ceil(np.array([sides]*3 if np.isscalar(sides) else sides)/water_cube_side)
  #~solvent = tile_mol(tip3p, water_cube_side, nwaters_side, rand_rot=True, rand_offset=0.2*water_cube_side)
  #~solvent.pbcbox = nwaters_side*water_cube_side
  return solvent


# this basically what Tinker does (but just with nested loops instead of spatial index)
# solvent molecules are assumed to be placed at random independent positions, so that we can just replace
#  first N solvent molecules with ions to neutralize system;  d=2.0 might be more realistic
def solvate(mol, solvent, r_solute=None, d=3.0, ions=[], solute_res=None, solvent_chain=None):
  """ combine molecules `mol` and `solvent`, removing any residues in `solvent` within `d` Ang of `mol` """
  if r_solute is None: r_solute = mol.r
  kd = cKDTree(r_solute)
  # could try a prefiltering step using mol.extents
  dists, locs = kd.query(solvent.r, distance_upper_bound=d)
  remove = set([solvent.atoms[ii].resnum for ii,dist in enumerate(dists) if dist < d])

  solvated = Molecule(pbcbox=solvent.pbcbox)  #, header=mol.header + " in " + solvent.header)
  solvated.append_atoms(mol, r=r_solute, residue=solute_res)
  # replace first N solvent molecules w/ ions
  for ion in ions:
    for ii,res in enumerate(solvent.residues):
      if ii not in remove:
        solvated.append_atoms(ion, r=ion.r + np.mean([solvent.atoms[jj].r for jj in res.atoms], axis=0))
        remove.add(ii)
        break  # inner loop

  retain = [ii for ii in range(len(solvent.residues)) if ii not in remove]
  solvent = solvent.extract_atoms(resnums=retain)
  if solvent_chain is not None:
    assert solvent_chain not in solvated.chains, "Solvent chain already in use!"
    for ii,res in enumerate(solvent.residues):
      res.pdb_num, res.chain = ii+1, solvent_chain
  solvated.append_atoms(solvent)
  return solvated


# we may need to implement max_memory as for pyscf_mo_grid()
# TODO: should make sure chain doesn't already exist
def neutralize(mol, d=3.0, ion_p=None, ion_n=None, grid_density=0.5, chain='X'):
  """ place ions (`ion_p` or `ion_n`) to neutralize net charge of `mol` at grid points (on grid with density
   `grid_density`) with largest electrostatic potential.  Intended for systems in vacuum or implicit
    solvent; should be followed by MM minimization
  """
  from ..qmmm.grid import r_grid, esp_grid
  extents = mol.extents(pad=2.0/grid_density)
  grid = r_grid(extents, grid_density)
  kd = cKDTree(mol.r)
  dists, locs = kd.query(grid, distance_upper_bound=d)
  grid = grid[dists > d]  # dist = np.inf if no hit
  esp = esp_grid(mol.mmq, mol.r, grid)
  net_charge = np.sum(mol.mmq)
  ion = ion_n if net_charge > 0.0 else ion_p
  n_ions = int(round(-net_charge/ion.mmq))
  # small offset from grid point to prevent divide by zero when calculating ESP for added ion
  offset = np.array([0.001, 0.001, 0.001])
  residue = Residue(name=ion.name, chain=chain, het=True)
  for ii in range(n_ions):
    r = ion.r + grid[np.argmax(esp) if net_charge > 0.0 else np.argmin(esp)] + offset
    residue.pdb_num = str(ii)
    mol.append_atoms(ion, r=r, residue=residue)
    # update ESP
    esp = esp + esp_grid([ion.mmq], [r], grid)
  return mol
