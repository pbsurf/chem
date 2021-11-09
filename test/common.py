import os
from ..molecule import *
from ..io import *
from ..io.tinker import tinker_EandG, make_tinker_key
from ..opt.optimize import moloptim, XYZ


def quick_load(pdbfile, xyzfile=None, charges=None):
  pdbfile = pdbfile if pdbfile.endswith('.pdb') or ('\n' in pdbfile) else pdbfile + '.pdb'
  xyzfile = pdbfile[0:-4] + '.xyz' if xyzfile is None else (
      xyzfile if xyzfile.endswith('.xyz') or ('\n' in xyzfile) else xyzfile)
  pdb_mol = load_molecule(pdbfile)
  xyz_mol = load_molecule(xyzfile, charges=charges)
  return copy_residues(xyz_mol, pdb_mol)


def quick_save(mol, pdbfile, xyzfile=None, r=None):
  pdbfile = pdbfile if pdbfile.endswith('.pdb') else pdbfile + '.pdb'
  xyzfile = pdbfile[0:-4] + '.xyz' if xyzfile is None else (xyzfile if xyzfile.endswith('.xyz') else xyzfile)
  write_pdb(mol, pdbfile, r=r)
  write_tinker_xyz(mol, xyzfile, r=r)


# note GBSA = Generalized Born (electrostatic solvent-solute interaction) + Surface Area (hydrophobic effect)
def quick_prep(filename, optargs={}):
  """ load molecule from PDB + XYZ files and relax hydrogens (added by Tinker to XYZ) with implicit solvent """
  mol = quick_load(filename)
  tinker_key = make_tinker_key('amber96') + "\nsolvate GBSA\n"
  tinker_args = Bunch(prefix=filename + "_quick", key=tinker_key)
  #h_and_cl = select_atoms(trypsin, 'znuc in [1, 11, 17]')
  h_atoms = select_atoms(mol, 'znuc == 1')
  res, r_opt = moloptim(tinker_EandG, mol=mol, fnargs=tinker_args, coords=XYZ(mol, h_atoms), **optargs)
  mol.r = r_opt
  write_tinker_xyz(mol, filename + "_quick_opt.xyz")
  return mol


def EandG_sanity(EandG, r0, scale=0.02, dr=None):
  """ generic sanity check for any EandG fn (to help catch typos and dumb mistakes) """
  E0, G0 = EandG(r0)
  dr = scale*(np.random.rand(*np.shape(r0)) - 0.5) if dr is None else dr
  for ii in range(20):
    E1, G1 = EandG(r0 + dr)
    dE0 = np.vdot(G0, dr)
    dE1 = np.vdot(G1, dr)
    # note that rms(r) == rms(norm(r, axis=1))
    print("rms(dr): %.8f; dE_expect: %.8f (G1: %.8f); dE_actual: %.8f" % (rms(dr), dE0, dE1, E1 - E0))
    dr = 0.5*dr


def hess_sanity(fn, r, scale=0.1, steps=4):
  """ sanity check for fn returning energy, grad, and Hessian """
  for ii in range(steps):
    dr = scale*(np.random.random(np.shape(r)) - 0.5)
    E1,G1,H1 = fn(r, hess=True)
    E2,G2,H2 = fn(r+dr, hess=True)
    print("dE expect: %f (%f); dE actual: %f" % (np.vdot(G1, dr), np.vdot(G2, dr), E2 - E1))
    print(G2 - G1)
    print(np.einsum('ijkl,jl->ik', H1, dr))
    scale *= 0.1
