import numpy as np
from ..data.pdb_bonds import PDB_PROTEIN
from .grid import r_grid, esp_grid


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


def geometry_check(mol, min_angle=np.pi/2):
  """ check molecule for unusual bonds or angles (by default < 90 deg - reasonable threshold for proteins) """
  bonds, angles, diheds = mol.get_internals()
  for a in angles:
    if mol.angle(a) < min_angle:
      print("Warning: angle %s = %.2f degrees" % (a, 180*mol.angle(a)/np.pi))
  guessed_bonds = guess_bonds(mol.r, mol.znuc)
  bonds_set, guessed_set = frozenset(bonds), frozenset(guessed_bonds)
  for extra_bond in bonds_set - guessed_set:
    print("Warning: unexpected bond %s" % extra_bond)
  for missing_bond in guessed_set - bonds_set:
    print("Warning: possibly missing bond %s" % missing_bond)


# Tinker's build solvent box function (xyzedit option 19) places molecules at random positions w/ random
#  orientations; no effort to avoid clashes ... so we can just do it ourselves!
def solvent_box(mol, ncopies, extents):
  extents = np.asarray([extents]*3 if np.isscalar(extents) else extents)
  extents = np.array([-0.5*extents, 0.5*extents]) if np.size(extents) == 3 else extents
  r0 = mol.r - center_of_mass(mol)
  solvent = Molecule()
  for ii in range(ncopies):
    r = np.dot(random_rotation(), r0) + extents[0] + (extents[1] - extents[0])*np.random.random(3)
    solvent.append_atoms(mol, r)
  return solvent


# this basically what Tinker does (but just with nested loops instead of spatial index)
# solvent molecules are assumed to be placed at random independent positions, so that we can just replace
#  first N solvent molecules with ions to neutralize system (if ion_p and/or ion_n are provided)
# - alternatively, we could just choose solvent molecules for replacement randomly
def solvate(mol, solvent, d=3.0, ion_p=None, ion_n=None):
  from scipy.spatial.ckdtree import cKDTree
  kd = cKDTree(mol.r)
  # could try a prefiltering step using mol.extents
  dists, locs = kd.query(solvent.r, distance_upper_bound=d)
  remove = set([solvent.atoms[ii].resnum for ii,dist in enumerate(dists) if dist < d])

  solvated = Molecule()
  solvated.append_atoms(mol)
  if ion_p or ion_n:
    # solvent assumed to be neutral for now
    net_charge = round(np.sum(mol.mmq))  # + np.sum(solvent.mmq)
    ion = ion_n if net_charge > 0.0 else ion_p
    for ii,res in enumerate(solvent.residues):
      if abs(net_charge) <= 0.5*abs(ion.mmq):
        break
      if ii not in remove:
        r_ii = np.sum([mol.atoms[jj].r for jj in res.atoms], axis=0)/len(res.atoms)
        remove.add(ii)
        solvated.append_atoms(ion, r=ion.r + r_ii)
        net_charge += ion.mmq

  retain = [ii for ii in range(len(solvent.residues)) if ii not in remove]
  solvated.append_atoms(extract_atoms(solvent, resnums=retain))
  return solvated


# we may need to implement max_memory as for pyscf_mo_grid()
# TODO: should make sure chain doesn't already exist
def neutralize(mol, d=3.0, ion_p=None, ion_n=None, grid_density=0.5, chain='X'):
  """ place ions (`ion_p` or `ion_n`) to neutralize net charge of `mol` at grid points (on grid with density
   `grid_density`) with largest electrostatic potential.  Intended for systems in vacuum or implicit
    solvent; should be followed by MM minimization
  """
  from scipy.spatial.ckdtree import cKDTree
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
