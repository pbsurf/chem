import os, time, subprocess
import numpy as np

from ..basics import *
from ..analyze import find_h_bonds, secondary_structure
from .color import *
from .mol_renderer import LineRenderer


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
