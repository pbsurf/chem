import os, time, subprocess
import numpy as np

from ..basics import *
from ..molecule import select_atoms
from ..io.tinker import write_tinker_xyz, read_tinker_interactions, TINKER_PATH
from .color import *
from .mol_renderer import LineRenderer


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


# H-bond fns for VisContacts

def hbond_contacts(mol):
  if not hasattr(mol, 'hbonds'):
    mol.hbonds, mol.hbond_energies = find_h_bonds(mol, max_energy=-2.0/KJMOL_PER_HARTREE)
  return mol.hbonds

def hbonds_radii_by_energy(mol):
  return 0.0005 + 0.004*mol.hbond_energies/(-100.0/KJMOL_PER_HARTREE)

def hbonds_color_by_energy(mol):
  return color_ramp([Color.blue, Color.red], mol.hbond_energies/(-100.0/KJMOL_PER_HARTREE))


# Tinker energy breakdown fns for VisContacts

# lots of duplication with tinker_EandG - we can try to refactor later
# - maybe make a context manager so we can do: `with Tempfiles(prefix+".xyz", prefix+".key") as mminp, mmkey:`
def tinker_E_breakdown(mol, key, r=None, sel=None, prefix=None, cutoff=5.0):
  if prefix is None:
    prefix = "mmtmp_%d" % time.time()
    cleanup = True
  mminp = prefix + ".xyz"
  mmkey = prefix + ".key"
  write_tinker_xyz(mol, mminp, r=r)  #title=title)
  with open(mmkey, 'w') as f:
    f.write(key)
    f.write('cutoff %f\n' % cutoff)
    if sel is not None:
      active = select_atoms(mol, sel)
      f.write("\n".join("active  " + " ".join("%d" % x for x in chunk) for chunk in chunks(active, 10)))

  # for some reason Tinker analuze D prints bond and angle terms to stderr instead of stdout
  s = subprocess.check_output([os.path.join(TINKER_PATH, "analyze"), mminp, "D"], stderr=subprocess.STDOUT)
  Eindv = read_tinker_interactions(s)
  if cleanup:
    try:
      os.remove(mminp)
      if key is not None:
        os.remove(mmkey)
    except: pass
  return Eindv

# kB*T is ~ 0.6 kcal/mol
def tinker_contacts(mol, key=None, r=None, sel=None, Ethresh=-10.0/KCALMOL_PER_HARTREE):
  if not hasattr(mol, 'Ebreakdown') or r is not None:
    mol.Ebreakdown = tinker_E_breakdown(mol, key=key, r=r, sel=sel)
  pairs = list(set(mol.Ebreakdown['charge'].keys() + mol.Ebreakdown['vdw-lj'].keys()))
  return [ pair for pair in pairs \
      if mol.Ebreakdown['charge'].get(pair, 0) + mol.Ebreakdown['vdw-lj'].get(pair, 0) < Ethresh ]


def tinker_contacts_radii(mol, atoms):
  Etot = mol.Ebreakdown['charge'].get(atoms, 0) + mol.Ebreakdown['vdw-lj'].get(atoms, 0)
  return 0.5 + 4.0*Etot/(-100.0/KCALMOL_PER_HARTREE)


def tinker_contacts_color(mol, atoms):
  Etot = mol.Ebreakdown['charge'].get(atoms, 0) + mol.Ebreakdown['vdw-lj'].get(atoms, 0)
  return color_ramp([Color.lime, Color.yellow], mol.Ebreakdown['charge'].get(atoms, 0)/Etot)


class VisContacts:
  """ Class for visualizing non-covalent interactions """

  # in the future, we can add a (dashed) cylinder option (each dash as separate capped cylinder)
  def __init__(self, fn, radius=2.0, colors=Color.light_grey, style='lines', dash_len=0.1):
    """ `fn` takes a molecule and returns list of atom pairs (tuples) for which interaction should be shown;
      can also set other attributes of molecule for use by `radii` and `colors`, which can be fns taking
      molecule and atom pair (or just constant values).
    """
    self.contacts_fn = fn
    self.radius = radius
    self.colors = colors
    self.line_renderer = LineRenderer(dash_len=dash_len)

  def draw(self, viewer, pass_num):
    if pass_num == 'opaque':
      self.line_renderer.draw(viewer)

  def set_molecule(self, mol, r=None):
    contacts = self.contacts_fn(mol, r=r)
    r = mol.r if r is None else r
    bounds = np.array(zip(r[ [b[0] for b in contacts] ], r[ [b[1] for b in contacts] ]))
    radii = [self.radius(mol, b) for b in contacts] if callable(self.radius) else [self.radius]*len(bounds)
    colors = [self.colors(mol, b) for b in contacts] if callable(self.colors) else [self.colors]*len(bounds)
    self.line_renderer.set_data(bounds, radii, colors)

  def on_key_press(self, viewer, keycode, key, mods):
    return False
