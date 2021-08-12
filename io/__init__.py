import logging
import numpy as np

from ..basics import *
from ..molecule import Molecule, Residue, guess_bonds, center_of_mass
from .pdb import parse_pdb, copy_residues, write_pdb
from .tinker import parse_xyz, write_xyz, write_tinker_xyz, load_tinker_params, make_tinker_key


def cclib_open(filename):
  from cclib.io import ccopen  # this is what is creating pyquante.log (see PyQuante/settings.py)
  c = ccopen(filename, loglevel=logging.ERROR).parse()
  if hasattr(c, 'errormsg'):
    print("QM log file %s reports error %s" % (filename, c.errormsg))
  return c


def load_molecule(file, postprocess=None, center=False, charges=False, residue=None, **kwargs):
  filename = None if '\n' in file else file
  fileext = filename.split('.')[-1].lower() if filename is not None else ''
  if fileext == 'pdb' or file.startswith('HEADER'):
    mol = parse_pdb(file, **kwargs)
  elif fileext.startswith('xyz') or file[0:file.find('\n')].split()[0].isdigit():
    mol = parse_xyz(file)  #, **kwargs)
    if charges:
      load_tinker_params(mol, file, charges=True)
  else:
    # assume quantum chemistry log file
    mol_cc = cclib_open(file)
    # assume background charges follow atoms
    end_atoms = next((ii for ii,z in enumerate(mol_cc.atomnos) if z < 1), len(mol_cc.atomnos))
    bonds = guess_bonds(mol_cc.atomcoords[0][:end_atoms], mol_cc.atomnos[:end_atoms], tol=0.2)
    mol = Molecule(r=mol_cc.atomcoords[0], znuc=np.fmax(0, mol_cc.atomnos), bonds=bonds)
    mol.cclib = mol_cc
  mol.filename = filename
  if center:
    mol.r = mol.r - center_of_mass(mol)
  if residue:
    mol.set_residue(residue)
  return postprocess(mol) if callable(postprocess) else mol


# cclib is at least talking about doing the sane thing and supporting atomic units: https://github.com/cclib/cclib/issues/89
# I think this should be implemented by removing all inline conversions and instead having log file parser
#  add fields to object specifying units - typically it would only have to indicate length and energy, other
#  other units being inferred - but could specify explicit units for any field.  There would be a parse()
#  option to convert to desired units

def cclib_EandG(mol_cc, grad=True):
  """ return energy (at highest level of theory avail) and optionally gradient of energy from cclib object
    Assumes gradient from cclib is in Hartree/Bohr, as is the case for GAMESS.
  """
  try:
    Eqm = mol_cc.scfenergies[-1]
    Eqm = mol_cc.mpenergies[-1][-1]
    Eqm = mol_cc.ccenergies[-1]
  except: pass
  Eqm /= EV_PER_HARTREE

  return Eqm, (mol_cc.grads[-1]/ANGSTROM_PER_BOHR if grad else None)


# alternatives to hdf5 include Numpy format (.npy, .npz) and netCDF (based on HDF5, built-in support in scipy)
# - h5py was installed for pyscf
def write_or_append_hdf5(filename, mode, **kwargs):
  """ write fields specified by `kwargs` to HDF5 file """
  import h5py
  with h5py.File(filename, mode) as h5f:
    for k,v in kwargs.items():
      if h5f.get(k):
        del h5f[k]
      h5f.create_dataset(k, data=v)  #h5f[k] = v

def write_hdf5(filename, **kwargs):
  write_or_append_hdf5(filename, 'w', **kwargs)

def append_hdf5(filename, **kwargs):
  write_or_append_hdf5(filename, 'a', **kwargs)

def read_hdf5(filename, *args):
  """ return tuple of specified fields from HDF5 file """
  import h5py
  with h5py.File(filename, 'r') as h5f:
    return tuple(getattr(h5f.get(key), 'value', None) for key in args) if len(args) > 1 else getattr(h5f.get(args[0]), 'value', None)
