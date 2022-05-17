from scipy.spatial.ckdtree import cKDTree
from .basics import *
from .data.elements import ELEMENTS
from .data.pdb_data import PDB_PROTEIN


# also see resp.py, grid.py
def shell_grid(N):
  """ Generate N points on surface of unit sphere using the "golden section spiral algorithm" """
  dl = np.pi*(3 - 5**0.5)
  coords = np.zeros((N, 3))
  for k in range(N):
    z = 1 - (2*k + 1)/N
    r = np.sqrt(1 - z*z)
    coords[k] = [np.cos(k*dl) * r, np.sin(k*dl) * r, z]
  return coords


# Solvent accessible surface area calculation using grid
# reasonably close agreement w/ https://www.ebi.ac.uk/msd-srv/prot_int/cgi-bin/piserver (for 5HSS); refs:
# - https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/src/sasa.cpp
# - https://github.com/biopython/biopython/blob/master/Bio/PDB/SASA.py
def sasa(mol, sel=None, r=None, ngrid=100, probe=1.4):
  """ return solvent accessible surface area per atom in Ang^2 for atoms `sel` (default all) from Molecule
    `mol` using `ngrid` (default 100) grid points per atom and solvent radius `probe` (default 1.4 Ang)
  """
  r = mol.r if r is None else r
  kdgrid = cKDTree(shell_grid(ngrid))
  kdatoms = cKDTree(mol.r if r is None else r)
  radii = np.array([ ELEMENTS[z].vdw_radius + probe for z in mol.znuc ])
  maxr = np.max(radii)
  SA = np.zeros(mol.natoms)
  for ii in mol.select(sel):
    # we translate neighbors and scale radii to avoid reconstructing kdtree for grid points
    scale = 1/radii[ii]
    neighbors = kdatoms.query_ball_point(r[ii], radii[ii] + maxr)
    neighbors.remove(ii)
    rnear = (r[neighbors] - r[ii])*scale
    cov = kdgrid.query_ball_point(rnear, radii[neighbors]*scale)
    # np.flatten doesn't work here; np.hstack does but converts to floats
    SA[ii] = 4*np.pi*radii[ii]**2 * (1 - len(unique(flatten(cov)))/ngrid)
  return SA


## Hydrogen bonds
# For more sophisticated analysis of non-covalent interactions (esp. for MD trajectories), see
# http://marid.bioc.cam.ac.uk/credo / https://bitbucket.org/harryjubb/arpeggio/ , MDAnalysis, MDTraj,
#  https://github.com/ssalentin/plip , https://github.com/akma327/MDContactNetworks ,
#  https://github.com/maxscheurer/pycontact
# We could also use Tinker `analyze` to get electrostatic energies, but `analyze d` generates 100s of MB of
#  output for a protein sized molecule
# Recall that atom to which H is covalently bonded is called the donor; other atom is acceptor
# Default H - acceptor distance cutoff is 3.2 Ang - quite liberal; 2.5 and 3.0 are other common values
# Some codes use donor - acceptor distance instead (often 3 Ang)

def h_bond_energy(mol, h, acceptor, r=None):
  """ DSSP hydrogen bond energy - electrostatic energy between (H + donor) and (acceptor + atoms bonded to
    acceptor); see en.wikipedia.org/wiki/DSSP_(protein)
  """
  r = mol.r if r is None else r  # caller should provide r if calling many times
  # compute electrostatic energy between (H + donor) and (acceptor + atoms bonded to acceptor)
  a1 = [h] + mol.atoms[h].mmconnect  # should just be H and donor
  a2 = [acceptor] + mol.atoms[acceptor].mmconnect
  mq1 = np.array([mol.atoms[ii].mmq for ii in a1])
  mq2 = np.array([mol.atoms[ii].mmq for ii in a2])
  dr = r[a1][:,None,:] - r[a2][None,:,:]
  return ANGSTROM_PER_BOHR*np.sum( (mq1[:,None]*mq2[None,:])/np.sqrt(np.sum(dr*dr, axis=2)) )


def find_h_bonds(mol, r=None, max_dist=3.2, min_angle=120, max_energy=None):
  """ Return list of atom indices of the form (H, acceptor) for hydrogen bonds in `mol` as defined by
    H - acceptor `max_dist` (default 3.2 A), donor - H - acceptor`min_angle` (default 120 deg), and optionally
    `max_energy` (in Hartree), < 0 for attractive interaction.  A list of DSSP-type hydrogen bond energies
    will also be returned iff `max_energy` is not None, in which case atoms must have mmq attrib set.
    Currently, allowed donor and acceptor atoms are hardcoded to N, O, and F
  """
  r = mol.r if r is None else r
  ck = cKDTree(r)
  pairs = ck.query_pairs(max_dist)
  hbonds = []
  energies = []
  for i,j in pairs:
    if mol.atoms[i].znuc == 1 or mol.atoms[j].znuc == 1:
      h, acceptor = (i,j) if mol.atoms[i].znuc == 1 else (j,i)
      # check element of donor and acceptor atoms
      donor = mol.atoms[h].mmconnect[0] if mol.atoms[h].mmconnect else None
      # allowed acceptors and donors hardcoded to N, O, and F ... include S? Cl? use electronegativity instead?
      if donor and mol.atoms[acceptor].znuc in [7,8,9] and mol.atoms[donor].znuc in [7,8,9] and \
          abs(calc_angle(r[[donor, h, acceptor]])) >= np.pi*min_angle/180.0:
        if max_energy is not None:
          E = h_bond_energy(mol, h, acceptor, r=r)
          if E > max_energy:
            continue
          energies.append(E)
        hbonds.append((h, acceptor))

  return (hbonds, np.array(energies)) if max_energy is not None else hbonds


# secondary structure determination - from github.com/boscoh/pyball/blob/master/pyball.py
# - DSSP is standard method for this and uses a specific formula for H-bond detection, see
#  en.wikipedia.org/wiki/DSSP_(hydrogen_bond_estimation_algorithm)
# - we need to enforce a minimum length (i.e. num of residues) for secondary structure features
def secondary_structure(mol):
  """ return list of secondary structure type (DSSP convention) for residues of mol """
  hb = set()
  for h,acc in find_h_bonds(mol):
    if (mol.atoms[h].name == 'H' and mol.atoms[acc].name == 'O' and mol.atomres(h).name in PDB_PROTEIN
        and mol.atomres(acc).name in PDB_PROTEIN and mol.atomres(h).chain == mol.atomres(acc).chain):
      hb.add( (mol.atoms[h].resnum, mol.atoms[acc].resnum) )
      hb.add( (mol.atoms[acc].resnum, mol.atoms[h].resnum) )
  ss = np.array(['']*mol.nresidues)
  for ii in range(mol.nresidues):
    if (ii, ii+4) in hb and (ii+1, ii+5) in hb:
      ss[ii+1:ii+5] = 'H'  # alpha-helix
    if (ii, ii+3) in hb and (ii+1, ii+4) in hb:
      ss[ii+1:ii+4] = 'G'  # 3-10 helix
    for jj in range(ii+6,mol.nresidues):  #list(range(0,ii-5)) + list(range(ii+6,mol.nresidues))
      if (ii,jj) in hb:
        # parallel beta sheet
        if (ii-2, jj-2) in hb:
          ss[ [ii-2, ii-1, ii, jj-2, jj-1, jj] ] = 'E'
        if (ii+2, jj+2) in hb:
          ss[ [ii+2, ii+1, ii, jj+2, jj+1, jj] ] = 'E'
        # anti-parallel beta sheet
        if (ii-2, jj+2) in hb:
          ss[ [ii-2, ii-1, ii, jj+2, jj+1, jj] ] = 'E'
        if (ii+2, jj-2) in hb:
          ss[ [ii+2, ii+1, ii, jj-2, jj-1, jj] ] = 'E'
  return ss  #.tolist()


## molecular geometry/topology - these are only used for DLC, so moved out of molecule.py

def nearest_pairs(r_array, N):
  """ return list of pairs (i,j), i < j for `N` nearest neighbors of points `r_array` """
  if N > len(r_array) - 1:
    print("Warning: requested %d nearest neighbors but only %d points" % (N, len(r_array)))
  ck = cKDTree(r_array)
  dists, locs = ck.query(r_array, min(len(r_array), N+1))  # query returns same point, so query for N+1 points
  return [ (ii, jj) for ii, loc in enumerate(locs) for jj in loc if jj > ii ]


def generate_internals(bonds):  #, impropers=False):
  """ generate angles, dihedrals, and improper torsions (w/ 2nd atom central atom) from a set of bonds """
  max_idx = max([x for bond in bonds for x in bond]) if bonds else -1
  connect = [ [] for ii in range(max_idx + 1) ]
  for bond in bonds:
    connect[bond[0]].append(bond[1])
    connect[bond[1]].append(bond[0])
  angles = [ (ii, jj, kk) for ii in range(len(connect))
    for jj in connect[ii]
    for kk in connect[jj] if kk > ii ]
  diheds = [ (ii, jj, kk, ll) for ii in range(len(connect))
    for jj in connect[ii]
    for kk in connect[jj] if kk != ii
    for ll in connect[kk] if ll != jj and ll > ii ]
  imptor = [ (ii, jj, kk, ll) for ii in range(len(connect))
    for jj in connect[ii]
    for kk in connect[jj] if kk != ii
    for ll in connect[jj] if ll != kk and ll > ii ]
  return angles, diheds, imptor


def mol_fragments(mol, active=None):
  """ return set of unconnected fragments from `active` (default, all) atoms in `mol` """
  remaining = set(active if active is not None else mol.listatoms())
  frags = []
  while remaining:
    frags.append(mol.get_connected(remaining.pop(), active=active))
    remaining -= set(frags[-1])
  return frags


# any reason for option to only connect non-hydrogens?
def fragment_connections(r, frags):
  """ returns set of minimum distance connections needed to fully connect set of fragments specified by
    `frags` - list of lists of indexes into `r` (as returned by mol_fragments())
  """
  r_frags = [ [r[ii] for ii in frag] for frag in frags ]
  kdtrees = [cKDTree(r_frag) for r_frag in r_frags[:-1]]
  connect = []  # list to hold "bonds" needed to connect fragments
  unconnected = set(range(len(frags)-1))
  r_connected = r_frags[-1]
  idx_connected = list(frags[-1])
  while unconnected:
    min_dist = np.inf
    for ii,kd in enumerate(kdtrees):
      if ii in unconnected:
        dists, idxs = kd.query(r_connected)
        min_idx = dists.argmin()
        if dists[min_idx] < min_dist:
          min_dist = dists[min_idx]
          closest_frag = ii
          closest_pair = (idx_connected[min_idx], frags[ii][idxs[min_idx]])
    # connect next fragment
    connect.append(closest_pair)
    unconnected.remove(closest_frag)
    r_connected.extend(r_frags[closest_frag])
    idx_connected.extend(frags[closest_frag])
  return connect
