from __future__ import absolute_import
import numpy as np
from pyscf import gto, scf, qmmm, lib
from ..data.elements import ELEMENTS
from ..basics import ANGSTROM_PER_BOHR

# from https://github.com/pyscf/pyscf/blob/master/examples/qmmm/30-force_on_mm_particles.py
# mol: pyscf Mole, dm: make_rdm1() from pyscf calculation, coords: MM charge positions in Ang, charges: MM q
# returns MM charge gradient in Hartree/Bohr
def pyscf_bq_grad(mol, dm, coords, charges):
  # The interaction between QM nuclear charges and MM particles
  # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
  coords = np.asarray(coords)/ANGSTROM_PER_BOHR  # convert to Bohr
  qm_coords = mol.atom_coords(unit='Bohr')
  qm_charges = mol.atom_charges()
  dr = qm_coords[:,None,:] - coords
  r = np.linalg.norm(dr, axis=2)
  g = np.einsum('r,R,rRx,rR->Rx', qm_charges, charges, dr, r**-3)

  # The interaction between electron density and MM particles
  # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
  #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
  for i, q in enumerate(charges):
    with mol.with_rinv_origin(coords[i]):
      v = mol.intor('int1e_iprinv')
    f = (np.einsum('ij,xji->x', dm, v) + np.einsum('ij,xij->x', dm, v.conj())) * -q
    g[i] += f
  return g


# TODO:
# - add prefix arg to support saving log
def pyscf_EandG(mol, r=None, qmatoms=None, caps=[], charges=[], qm_charge=0, scffn=None, basis=None): #, moguess=None)
  """ Hartree-Fock energy and gradient for qmatoms from mol, plus cap atoms, with charges;
    qm_charge is total charge of QM region; basis is default basis for atoms w/o qmbasis attribute
  """
  scffn = scffn if scffn else scf.RHF
  qmatoms = mol.listatoms() if qmatoms is None else qmatoms
  allatoms = [mol.atoms[ii] for ii in qmatoms] + caps
  r = mol.r if r is None else r
  r = np.concatenate([r[qmatoms], [c.r for c in caps]]) if caps else r[qmatoms]
  names = [ ELEMENTS[getattr(a, 'znuc', 1)].symbol + "@%d"%ii for ii, a in enumerate(allatoms) ]
  assert r.ndim == 2 and r.shape[1] == 3, "Shape of coordinates is incorrect"  # pyscf doesn't check this
  atoms = zip(names, r)
  basis = { names[ii]: getattr(a, 'qmbasis', basis) for ii, a in enumerate(allatoms) }
  mol_gto = gto.M(atom=atoms, basis=basis, charge=qm_charge, cart=True, unit='Ang')  # Angstrom should be default
  mol_scf = scffn(mol_gto)  #scf.RHF(mol_gto)
  rq = [c.r for c in charges]
  qq = [c.qmq for c in charges]
  if charges:
    mol_scf = qmmm.add_mm_charges(mol_scf, rq, qq, unit='Ang')
  mol_scf.verbose = lib.logger.WARN  # suppress pyscf's printing of E and G
  mol_scf.chkfile = None  # suppress writing anything to disk
  scn = mol_scf.nuc_grad_method().as_scanner()  # needed to get both E and G - separate calls run SCF twice
  E, G = scn(mol_gto)
  #E = mol_scf.kernel(); G = mol_scf.nuc_grad_method().kernel() ... G calc fails if E calc not run first
  if charges:
    Gq = pyscf_bq_grad(mol_scf.mol, scn.base.make_rdm1(), rq, qq)  # mol_scf.make_rdm1()
    G = np.concatenate([G, Gq])
  return E, G/ANGSTROM_PER_BOHR, scn  # convert gradient from Hartree/Bohr to Hartree/Ang
