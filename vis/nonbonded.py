import os, time, subprocess
import numpy as np

from ..basics import *
from ..data.pdb_data import PDB_PROTEIN
from .color import *
from .mol_renderer import LineRenderer
#from ..io.tinker import tinker_contacts, tinker_E_breakdown


## Hydrogen bonds
# - we can move this out of vis/ if we find significant other use for it
# For more sophisticated analysis of non-covalent interactions (esp. for MD trajectories), see
# http://marid.bioc.cam.ac.uk/credo / https://bitbucket.org/harryjubb/arpeggio/ , MDAnalysis, MDTraj,
#  https://github.com/ssalentin/plip , https://github.com/akma327/MDContactNetworks ,
#  https://github.com/maxscheurer/pycontact
# We could also use Tinker `analyze` to get electrostatic energies, but `analyze d` generates 100s of MB of
#  output for a protein sized molecule
# Recall that atom to which H is covalently bonded is called the donor; other atom is acceptor
# Default H - acceptor distance cutoff is 3.2 Ang - quite liberal; 2.5 and 3.0 are other common values
# Some codes use donor - acceptor distance instead (often 3 Ang)

def h_bond_energy(mol, h, acceptor):
  """ DSSP-type hydrogen bond energy - includes H-acceptor, donor-acceptor, and
    (donor,H)-(atoms bonded to acceptor) electrostatic energy; atoms expected to have mmq attributes set
  """
  donor = mol.atoms[h].mmconnect[0]
  # H - acceptor energy
  E = mol.atoms[h].mmq*mol.atoms[acceptor].mmq/mol.dist(h, acceptor)
  # include donor-acceptor electrostatic energy - idea here is to reproduce effect of H bond weakening
  #  as donor-H-acceptor angle decreases from 180 deg
  E += mol.atoms[donor].mmq*mol.atoms[acceptor].mmq/mol.dist(donor, acceptor)
  # also include electrostatic interaction between atoms bonded to acceptor and donor and hydrogen to
  #  reproduce DSSP hydrogen bonding energy - https://en.wikipedia.org/wiki/DSSP_(protein)
  for ii in mol.atoms[acceptor].mmconnect:
    E += mol.atoms[ii].mmq*(mol.atoms[donor].mmq/mol.dist(donor, ii) + mol.atoms[h].mmq/mol.dist(h, ii))
  return E*ANGSTROM_PER_BOHR


def find_h_bonds(mol, max_dist=3.2, min_angle=120, max_energy=None):
  """ Return list of atom indices of the form (H, acceptor) for hydrogen bonds in `mol` as defined by
    H - acceptor `max_dist` (default 3.2 A), donor - H - acceptor`min_angle` (default 120 deg), and optionally
    `max_energy` (in Hartree), < 0 for attractive interaction.  A list of DSSP-type hydrogen bond energies
    will also be returned iff `max_energy` is not None, in which case atoms must have mmq attrib set.
    Currently, allowed donor and acceptor atoms are hardcoded to N, O, and F
  """
  from scipy.spatial.ckdtree import cKDTree
  ck = cKDTree(mol.r)
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
          abs(mol.angle((donor, h, acceptor))) >= np.pi*min_angle/180.0:
        if max_energy is not None:
          E = h_bond_energy(mol, h, acceptor)
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


# H-bond fns for VisContacts

def hbond_contacts(mol):
  if not hasattr(mol, 'hbonds'):
    mol.hbonds, mol.hbond_energies = find_h_bonds(mol, max_energy=-2.0/KJMOL_PER_HARTREE)
  return mol.hbonds

def hbonds_radii_by_energy(mol):
  return 0.0005 + 0.004*mol.hbond_energies/(-100.0/KJMOL_PER_HARTREE)

def hbonds_color_by_energy(mol):
  return color_ramp([Color.blue, Color.red], mol.hbond_energies/(-100.0/KJMOL_PER_HARTREE))


def NCMM_contacts(mol, r=None, sel=None, Ethresh=-10.0/KCALMOL_PER_HARTREE):
  from ..mm import NCMM
  E = {}
  NCMM(mol)(mol, r, components=E)
  sel = range(mol.natoms) if sel is None else mol.select(sel)
  pairs = [ (i,j) for i in range(mol.natoms) for j in range(i+1, mol.natoms)
      if E['Eqq'][i,j] + E['Evdw'][i,j] < Ethresh and i in sel and j in sel ]
  mol.Ebreakdown = { 'charge': {(i,j): E['Eqq'][i,j] for i,j in pairs},
      'vdw-lj': {(i,j): E['Evdw'][i,j] for i,j in pairs} }
  return pairs


def contacts_radii(mol, atoms):
  Etot = mol.Ebreakdown['charge'].get(atoms, 0) + mol.Ebreakdown['vdw-lj'].get(atoms, 0)
  return 0.5 + 4.0*Etot/(-100.0/KCALMOL_PER_HARTREE)


def contacts_color(mol, atoms):
  Etot = mol.Ebreakdown['charge'].get(atoms, 0) + mol.Ebreakdown['vdw-lj'].get(atoms, 0)
  return color_ramp([Color.lime, Color.yellow], mol.Ebreakdown['charge'].get(atoms, 0)/Etot)


class VisContacts:
  """ Class for visualizing non-covalent interactions """

  # in the future, we can add a (dashed) cylinder option (each dash as separate capped cylinder)
  def __init__(self, fn=None, radius=2.0, colors=Color.light_grey, style='lines', dash_len=0.1):
    """ `fn` takes a molecule and returns list of atom pairs (tuples) for which interaction should be shown;
      can also set other attributes of molecule for use by `radii` and `colors`, which can be fns taking
      molecule and atom pair (or just constant values).
    """
    self.contacts_fn = NCMM_contacts if fn is None else fn
    self.radius = radius
    self.colors = colors
    self.line_renderer = LineRenderer(dash_len=dash_len)

  def draw(self, viewer, pass_num):
    if pass_num == 'opaque':
      self.line_renderer.draw(viewer)

  def set_molecule(self, mol, r=None):
    contacts = self.contacts_fn(mol, r=r)
    r = mol.r if r is None else r
    bounds = r[np.asarray(contacts)]
    radii = [self.radius(mol, b) for b in contacts] if callable(self.radius) else [self.radius]*len(bounds)
    colors = [self.colors(mol, b) for b in contacts] if callable(self.colors) else [self.colors]*len(bounds)
    self.line_renderer.set_data(bounds, radii, colors)

  def on_key_press(self, viewer, keycode, key, mods):
    return False
