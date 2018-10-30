import numpy as np
from chem.basics import *

# TODO:
# - create some self-contained tests
# - hould have an easy way to inspect grad for dipole charges and cap atoms; qmmm.Qcaps - indexes into Gqm of caps? qmmm.Mdipoles - indexes into Gcharge? Gqm? for dipoles?


# For trypsin system:
# For qmatoms = A/195/~C,N,CA,O,H,HA GAMESS and NWChem agree to ~1E-6 Hartree w/ background charges and
#  ~1E-9 Hartree w/o background charges (note NWChem does not include charge-charge energy whereas GAMESS does)
# Cursory check of gradient looks consistent between GAMESS and NWChem
# NWChem QM, MM, and QM/MM gradient sums ~< 1E-04 of rms


# Sanity checks for our QM/MM gradient, esp. for cap atoms and frontier atoms: even if we put aside
#  correctness, gradient must still be *consistent* with energy or optimization won't work right!
# With decreasing step size (dr = 0.5*dr), dE_actual should lie between (or very close to) dE_expect0 and
#  dE_expect1, which should themselves get closer together with each reduction of step size
def sanity_check(qmmm, dratoms=None, r0=None):
  """ given a configured QMMM object `qmmm`, apply random translations to atoms `dratoms` and print energy
    and gradient information
  """
  mol = qmmm.mol
  r0 = mol.r if r0 is None else r0
  qmatoms = qmmm.qmatoms
  M1atoms = mol.get_bonded(qmatoms)
  M2atoms = mol.get_bonded(M1atoms + qmatoms)
  Q1atoms = list(set(qmatoms).intersection(mol.get_bonded(M1atoms)))

  dratoms = qmmm.qmatoms + M1atoms if dratoms is None else dratoms
  drmask = setitem(np.zeros_like(r0), dratoms, 1.0)
  qmmm.components = True
  #qmmm.prefix = tmpfolder + "/sanityQMMM"
  #qmmm.chargeopts['adjust'] = 'dipole'  # set to 'delete' or 'shift' to debug link atoms
  E0, G0 = qmmm.EandG(mol, r0, iternum=0)
  print("E0: %.8f" % E0)
  print("G0(Q1): %s" % G0[Q1atoms])
  print("G0(M1): %s" % G0[M1atoms])
  print("sum(G0): %s" % np.sum(G0, axis=0))
  print("sum(Gmm): %s" % np.sum(qmmm.Gcomp['Gmm'], axis=0))
  print("sum(Gqm): %s" % np.sum(qmmm.Gcomp['Gqm'], axis=0))
  print("sum(Gqm0): %s" % np.sum(qmmm.Gcomp['Gqm0'], axis=0))
  # initial dr for use with dr = 0.5*dr option
  dr = drmask * 0.02*(np.random.rand(*np.shape(r0)) - 0.5)
  sanitycheck = True
  while sanitycheck:
    dr = 0.5*dr
    #dr = drmask * 0.01*(np.random.rand(*np.shape(r0)) - 0.5)
    #dr = drmask * (-0.00001)/G0
    #dr = 0.1*np.ones_like(r0)  # add const to r and verify no change of E
    E1, G1 = qmmm.EandG(mol, r0+dr, iternum=0)
    # G is in Hartree/Ang; dr is in Ang
    dE_expect0 = np.sum(G0*dr)
    dE_expect1 = np.sum(G1*dr)
    print("rms(dr): %.8f; dE_expect: %.8f (G1: %.8f); dE_actual: %.8f" % \
        (rms(dr[dratoms]), dE_expect0, dE_expect1, E1 - E0))
    print("G1(Q1): %s" % G1[Q1atoms])  # all QM atoms?
    print("G1(M1): %s" % G1[M1atoms])
    print("sum(G1): %s" % np.sum(G1, axis=0))
    print("sum(Gmm): %s" % np.sum(qmmm.Gcomp['Gmm'], axis=0))
    print("sum(Gqm): %s" % np.sum(qmmm.Gcomp['Gqm'], axis=0))
    print("sum(Gqm0): %s" % np.sum(qmmm.Gcomp['Gqm0'], axis=0))
