import logging, io, pickle
import numpy as np

from ..basics import *
from ..molecule import Molecule, Residue, guess_bonds, center_of_mass
from .pdb import parse_pdb, copy_residues, write_pdb, download_pdb
from .xyz import parse_xyz, parse_mol2, write_xyz, write_tinker_xyz


def load_molecule(file, postprocess=None, center=False, charges=False, residue=None, download=False, **kwargs):
  filename = None if '\n' in file else file
  fileext = os.path.splitext(filename)[-1] if filename is not None else ''
  if len(file) == 4 and not fileext:  # PDB ID
    filename = os.path.join(DATA_PATH, 'pdb/', file + '.pdb')
    if not os.path.exists(filename) and download:
      download_pdb(file, dir=os.path.join(DATA_PATH, 'pdb/'))
    mol = parse_pdb(filename, **kwargs)
  elif fileext == 'pdb':
    mol = parse_pdb(file, **kwargs)
  elif file.startswith('HEADER'):
    mol = parse_pdb(file, **kwargs)
  elif fileext == 'mol2' or file.startswith("@<TRIPOS>MOLECULE"):
    mol = parse_mol2(file)
  elif fileext.startswith('xyz') or file[0:file.find('\n')].split()[0].isdigit():
    mol = parse_xyz(file)  #, **kwargs)
    if charges:
      load_tinker_params(mol, file, charges=True)
  else:
    # assume quantum chemistry log file
    mol_cc = cclib_open(file)
    # assume background charges follow atoms
    end_atoms = next((ii for ii,z in enumerate(mol_cc.atomnos) if z < 1), len(mol_cc.atomnos))
    bonds = guess_bonds(mol_cc.atomcoords[0][:end_atoms], mol_cc.atomnos[:end_atoms])  #, tol=0.2)
    mol = Molecule(r=mol_cc.atomcoords[0], znuc=np.fmax(0, mol_cc.atomnos), bonds=bonds)
    mol.cclib = mol_cc
  mol.filename = filename
  if center:
    mol.r = mol.r - center_of_mass(mol)  # or mol.r - np.mean(mol.extents(), axis=0)?
  if residue:
    mol.set_residue(residue)
  return postprocess(mol) if callable(postprocess) else mol


def cclib_open(filename):
  from cclib.io import ccopen  # this is what is creating pyquante.log (see PyQuante/settings.py)
  c = ccopen(filename, loglevel=logging.ERROR).parse()
  if hasattr(c, 'errormsg'):
    print("QM log file %s reports error %s" % (filename, c.errormsg))
  return c


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
  write_or_append_hdf5(filename, 'x', **kwargs)  # fail if file exists

def append_hdf5(filename, **kwargs):
  write_or_append_hdf5(filename, 'a', **kwargs)

def read_hdf5(filename, *args):
  """ return tuple of specified fields from HDF5 file """
  import h5py
  # we write str, h5py reads back bytes ... WTF!?
  def str_fix(dset):
    try: return dset.asstr()[...]
    except: return dset[...]
  with h5py.File(filename, 'r') as h5f:  # .value not longer available :-(
    return tuple(str_fix(h5f.get(key)) for key in args) if len(args) > 1 else str_fix(h5f.get(args[0]))


# zip container: xyz for Molecule, arc for list of Molecules, numpy npy (includes pickle) for everything else
# - why vs. pickling Molecule? human readable, smaller, safe against changes to Molecule class
# - read_zip/write_zip in basics are lower level
# - hopefully this will become our "one true format" and we can eliminate hdf5, etc.

def _load_zip_entry(zf, name, contents):
  if name + '.npy' in contents:
    return np.load(io.BytesIO(zf.read(name + '.npy')))
  if name + '.pkl' in contents:
    return pickle.loads(zf.read(name + '.pkl'))
  if name + '.xyz' in contents:
    return load_molecule(zf.read(name + '.xyz').decode('utf-8'))
  if name + '.arc' in contents:
    f = io.StringIO(zf.read(name + '.arc').decode('utf-8'))
    mols = []
    while 1:  #while m := parse_xyz(f) is not None:
      m = parse_xyz(f)
      if m is None:
        return mols
      mols.append(m)
  print("%s.(npy/pkl/xyz/arc) not found in zip file!" % name)
  return None


# kwargs for custom loaders? ... a,b,c = load_zip('file.zip', a=np.load, b=np.load, c=load_molecule)
def load_zip(filename, *args):
  from zipfile import ZipFile
  with ZipFile(filename, 'r') as zf:
    cts = frozenset(zf.namelist())
    if not args:
      return Bunch([ (f[:-4], _load_zip_entry(zf, f[:-4], cts)) for f in zf.namelist() ])
    if len(args) == 1:
      return _load_zip_entry(zf, args[0], cts)
    return tuple(_load_zip_entry(zf, a, cts) for a in args)


#def write_arc(mols):  return ''.join([write_xyz(m) for m in mols])  # that's it!
def _save_zip(filename, kwargs, mode):
  from zipfile import ZipFile, ZIP_DEFLATED
  with ZipFile(filename, mode, ZIP_DEFLATED) as zf:
    for k,v in kwargs.items():
      if type(v) is Molecule:
        zf.writestr(k + '.xyz', write_xyz(v))
      elif safelen(v) > 0 and type(v[0]) is Molecule:
        # no blank lines in .arc; write_xyz includes trailing \n
        zf.writestr(k + '.arc', ''.join([write_xyz(m) for m in v]))
      elif type(v).__module__ == np.__name__:
        with zf.open(k + '.npy', 'w') as npyf:
          np.save(npyf, v)
      else:  # np.save would either convert to numpy array or pickle anyway
        zf.writestr(k + '.pkl', pickle.dumps(v))

def save_zip(filename, **kwargs):
  _save_zip(filename, kwargs, mode='x')
