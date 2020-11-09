# transition state search
# play with hcoh -> ch2o, then ch2o -> co + h2
# Refs for hcoh -> ch2o: www.sas.upenn.edu/~mabruder/GAMESS.html

# TS code:
# * github.com/eljost/pysisyphus / pysisyphus.readthedocs.io / 10.1002/qua.26390 - Dimer (multiple approaches), Lanczos, NEB, String; DLC; excited states
# * gitlab.com/ase/ase - NEB, Dimer
# * github.com/pele-python/pele / pele-python.github.io - Dimer, NEB, etc
# * github.com/js850/PyGMIN - predecessor to pele; Dimer, NEB, etc
# * github.com/alexei-matveev/pts - Dimer, String
# * github.com/louisvernon/pesto - NEB, Lanczos
# * theory.cm.utexas.edu - TSSE/TSASE - NEB, Dimer, Lanczos
# * github.com/MFTabriz/vtst - theory.cm.utexas.edu/vtsttools - Fortran - NEB, Dimer, Lanczos
# * github.com/zadorlab/sella - saddle point search
# Misc:
# * 10.1016/j.cpc.2018.06.026 - "PASTA" - Python - NEB
# * github.com/zadorlab/KinBot - automated TS search
# * 10.1002/(SICI)1096-987X(199605)17:7<888::AID-JCC12>3.0.CO;2-7 (Baker, Chan 1996) - commonly used TS test set
#  - see github.com/eljost/pysisyphus/tree/master/xyz_files/baker_ts
# * github.com/ZimmermanGroup/pyGSM - growing string method
# P-RFO/Eigenvector following requires full Hessian I believe, so we won't worry about this for now
# * Baker 1986: 10.1002/jcc.540070402
# * pDynamo: Baker/P-RFO (pCore); reaction paths (pMoleculeScripts)

import os
from chem.molecule import *
from chem.io import load_molecule, read_hdf5, write_hdf5
#from chem.io.tinker import write_tinker_xyz, tinker_EandG
from chem.qmmm.qmmm1 import QMMM
from chem.opt.dlc import *
import chem.opt.optimize as opt  # rename chem.opt.optimize.optimize to ... mol_opt? optimize_mol? dlc_opt?
import chem.opt.lbfgs as lbfgs
import scipy.optimize
from pyscf import scf


ch2o_xyz = """4  molden generated tinker .xyz (mm3 param.)
 1  C     0.000000    0.000000    0.000000      3  2  3  4
 2  O     0.000000    0.000000    1.400000      7  1
 3  H     1.026720    0.000000   -0.362996      5  1
 4  H    -0.513360   -0.889165   -0.363000      5  1
"""

ch2o_opt_xyz = """4  molden generated tinker .xyz (mm3 param.)
 1  C     0.130284   -0.225658    0.155628      3  2  3  4
 2  O    -0.044368    0.076846    1.311490      7  1
 3  H     1.042499    0.108318   -0.396556      5  1
 4  H    -0.615055   -0.848670   -0.396558      5  1
"""

hcoh_xyz = """4  molden generated tinker .xyz (mm3 param.)
 1  C     0.000000    0.000000    0.000000      4  2  3
 2  O     0.000000    0.000000    1.400000      6  1  4
 3  H     1.026720    0.000000   -0.362996    124  1
 4  H    -0.447834    0.775672    1.716667     21  2
"""

co_xyz = """2  molden generated tinker .xyz (mm3 param.)
 1  C     0.000000    0.000000    0.000000      3  2
 2  O     0.000000    0.000000    1.400000      7  1
"""

# these are both MM3 optimized
ch3cho_xyz = """7  molden generated tinker .xyz (mm3 param.)
 1  C     0.216434   -0.239836    0.022589      3  2  3  7
 2  C     0.041247   -0.042250    1.496848      1  1  4  5  6
 3  O     1.273466   -0.143112   -0.555929      7  1
 4  H    -0.670708    0.790679    1.698814      5  2
 5  H    -0.360705   -0.966829    1.971194      5  2
 6  H     1.009413    0.204927    1.989829      5  2
 7  H    -0.702574   -0.492745   -0.564006      5  1
"""

ch2choh_xyz = """7  molden generated tinker .xyz (mm3 param.)
 1  C     0.231311   -0.117468    0.202122      2  2  3  7
 2  C    -0.336100   -0.188389    1.410112      2  1  4  5
 3  O     1.465427   -0.622004   -0.056621      6  1  6
 4  H    -1.343088    0.222649    1.588799      5  2
 5  H     0.192087   -0.661201    2.253726      5  2
 6  H     1.994120    0.037261   -0.538351     73  3
 7  H    -0.308197    0.346592   -0.641855      5  1
"""

#ch3cho = load_molecule(ch3cho_xyz)
#ch2choh = load_molecule(ch2choh_xyz)
#ch2choh.r = align_atoms(ch2choh, ch3cho, sel=[0,1,2])  # align the heavy atoms
#qmmm = QMMM(mol=ch3cho, qmatoms='*', moguess='prev', qm_opts=Bunch(prog='pyscf'), prefix='no_logs', savefiles=False)

ch2o = load_molecule(ch2o_xyz)
hcoh = load_molecule(hcoh_xyz)
co = load_molecule(co_xyz)

qmmm = QMMM(mol=ch2o, qmatoms='*', moguess='prev', qm_opts=Bunch(prog='pyscf'), prefix='no_logs', savefiles=False)

for ii in qmmm.qmatoms:
  qmmm.mol.atoms[ii].qmbasis = '6-31G*' #'6-31++G**'

def print_mos(mf):
  homo = np.max(np.nonzero(mf.mo_occ))
  print(''.join(["%.6f (%d)\n" % (mf.mo_energy[ii], mf.mo_occ[ii]) for ii in range(homo+2)]))

def print_uhf_mos(mf):
  homo = np.max(np.nonzero(mf.mo_occ))
  print(''.join(["%.6f (%d); %.6f (%d)\n" % (mf.mo_energy[0][ii], mf.mo_occ[0][ii], mf.mo_energy[1][ii], mf.mo_occ[1][ii]) for ii in range(homo+2)]))

def path_ts(energies, coords):
  """ given reaction path energies and coords, return energy, coords, and tangent for highest energy state """
  i_ts = np.argmax(energies)
  dr_ts = coords[i_ts+1] - coords[i_ts-1]
  return energies[i_ts], coords[i_ts], dr_ts/np.linalg.norm(dr_ts)


## RC scan
nimages = 10
c1_h4 = (0,3)  # bonds
o2_h4 = (1,3)

ts_test1_h5 = 'ts_test1.h5'
if os.path.exists(ts_test1_h5):
  rcs, RCr, RCe = read_hdf5(ts_test1_h5, 'rcs', 'RCr', 'RCe')
  r_hcoh, r_ch2o = RCr[0], RCr[-1]
else:
  RCr = [0]*nimages
  RCe = [0]*nimages

  res, r_hcoh = opt.optimize(ch2o, r0=hcoh.r, fn=qmmm.EandG, ftol=2E-07)
  RCr[0], RCe[0] = r_hcoh, res.fun

  res, r_ch2o = opt.optimize(ch2o, fn=qmmm.EandG, ftol=2E-07)
  r_ch2o = align_atoms(r_ch2o, r_hcoh, sel=[0,1,2])
  RCr[-1], RCe[-1] = r_ch2o, res.fun

  # why does this fail for intermediate RC values? note that trust param in DLC.toxyzs() doesn't help; can't
  #  find any combination of internal coords that work without adding cartesians; with diheds, failure does seem
  #  related to angle passing through pi, but not clear how and still fails with no diheds, only bends
  # don't worry about this anymore unless it happens again for a non-planar molecule
  #dlc = DLC(ch2o, autobonds='total', autodiheds='impropers', recalc=1)

  dlc = DLC(ch2o, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
  # note that using a single bond length (c1_h4) for RC didn't work - H ends up far from both C and O for large RCs
  dlc.constraint(dlc.bond(o2_h4) - dlc.bond(c1_h4))
  dlc.init()

  rc0 = calc_dist(r_hcoh[1], r_hcoh[3]) - calc_dist(r_hcoh[0], r_hcoh[3])
  rc1 = calc_dist(r_ch2o[1], r_ch2o[3]) - calc_dist(r_ch2o[0], r_ch2o[3])
  rcs = np.linspace(rc0, rc1, 10)
  ch2o.r = RCr[0]

  # could we use gradient at each point to make a smoother graph?
  for ii,rc in enumerate(rcs[1:-1]):
    # reverse bonds and use move_frag=False to keep H4 fixed
    ch2o.bond(o2_h4[::-1], 0.5*(rc - rcs[ii]), rel=True, move_frag=False)
    ch2o.bond(c1_h4[::-1], -0.5*(rc - rcs[ii]), rel=True, move_frag=False)
    #ch2o.bond(c1_h4[::-1], ch2o.bond(o2_h4) - rc, move_frag=False)  # move C1
    #print("rc: %f; O2-H4 - C1-H4: %f" % (rc, ch2o.bond(o2_h4) - ch2o.bond(c1_h4)))
    res, r = opt.optimize(ch2o, fn=qmmm.EandG, coords=dlc, ftol=2E-07)  #, raiseonfail=False)
    RCr[ii+1], RCe[ii+1] = r, res.fun
    ch2o.r = r

  write_hdf5(ts_test1_h5, rcs=rcs, RCr=RCr, RCe=RCe)

  from chem.vis.chemvis import *
  vis = Chemvis(Mol(ch2o, RCr, [ VisGeom(style='lines') ]), fog=True, wrap=False).run()

  import matplotlib.pyplot as plt
  plt.plot(rcs, RCe)
  plt.ylabel("energy (Hartrees)")
  plt.xlabel("C1-H4 (Angstroms)")
  plt.show()


# start interactive
import pdb; pdb.set_trace()


## Dimer, Lanczos
from chem.opt.dimer import *

# highest energy state from RC scan
E_rcts, r_rcts, dr_rcts = path_ts(RCe, RCr)

## Dimer method
if 0:
  dimer = DimerMethod(EandG=qmmm.EandG, R0=r_rcts, mode=dr_rcts)
  #r_dimer = dimer.search()
  #E_dimer = qmmm.EandG(r_dimer, dograd=False)
  DMres = lbfgs.optimize(dimer.EandG, r_rcts, ftol=0, maxiter=100)
  r_dimer, E_dimer = getvalues(DMres, 'x', 'fun')

## Lanczos method
lanczos = LanczosMethod(EandG=qmmm.EandG, tau=dr_rcts)
LCZres = lbfgs.optimize(lanczos.EandG, r_rcts, ftol=0, maxiter=100)  #stepper=lbfgs.LBFGS(H0=1./700),
r_lcz, E_lcz = getvalues(LCZres, 'x', 'fun')

#write_hdf5('hcoh_ts.h5', r_lcz=r_lcz, E_lcz=E_lcz)
#cs, RCr, RCe = read_hdf5(ts_test1_h5, 'rcs', 'RCr', 'RCe')


## try to lower TS energy
from chem.io.pyscf import pyscf_EandG

r_ts = r_lcz
charges = [ Bunch(r=None, qmq=(0.1 if ii%2 else -0.1)) for ii in range(10) ]

# we do not want to include charge-charge interaction - and pyscf doesn't!  Once we flesh this out more, we
#  can integrate into QMMM (which can remove charge-charge interaction for other QM codes)
def charge_opt_EandG(r):
  r = np.reshape(r, (-1,3))  # needed for scipy optimize
  for ii, c in enumerate(charges):
    c.r = r[ii]
  E, G, mf = pyscf_EandG(ch2o, r=r_ts, charges=charges)
  return E, np.ravel(G[-len(charges):])

# using normal distribution gives uniform dist over unit sphere (uniform distribution does not)
# place charges on surface of sphere around mol 1 Ang from furthest atom ... we need a better soln for
#  arbitrarily shaped molecules!
centroid = np.mean(r_ts, axis=0)
maxdist = np.max(norm(r_ts - centroid), axis=0)
r_charges = [(maxdist + 1.0)*normalize(np.random.normal(size=3)) + centroid for c in charges]

qmol = Molecule(atoms=ch2o.atoms + [Atom(name='Q', r=q.r, mmq=q.qmq) for q in charges])

n0, n1 = ch2o.natoms, len(charges)
bonds = [ (a0, a1) for a0 in range(n0 + n1) for a1 in range(max(a0+1, n0), n0 + n1) ]

bounds = [(1.8, 10)]*(n0*n1) + [None]*(len(bonds) - n0*n1)

coords = Redundant(qmol, bonds=bonds, autobonds='none', autoangles='none', autodiheds='none', recalc=1)

opt.optimize(qmol, r0=hcoh.r, fn=charge_opt_EandG, coords=coords, bounds=bounds, ftol=2E-07)

#res = lbfgs.optimize(charge_opt_EandG, r_charges)
res = scipy.optimize.minimize(charge_opt_EandG, np.ravel(r_charges), jac=True, method='L-BFGS-B', options=dict(disp=True))


def pyscf_bq_qgrad(mol, dm, coords, charges):
  # The interaction between QM atoms and MM particles
  # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
  coords = np.asarray(coords)/ANGSTROM_PER_BOHR  # convert to Bohr
  qm_coords = mol.atom_coords(unit='Bohr')
  qm_charges = mol.atom_charges()
  dr = qm_coords[:,None,:] - coords
  r = np.linalg.norm(dr, axis=2)
  g = -qm_charges/r
  #g = np.einsum('r,R,rR->Rx', qm_charges, charges, r**-1)

  # The interaction between electron density and MM particles
  for i, q in enumerate(charges):
    with mol.with_rinv_origin(coords[i]):
      v = mol.intor('int1e_rinv')
    f = -np.einsum('ij,xji->x', dm, v)
    g[i] += f
  return g


# How to keep charges from getting too close to atoms?
# - optimize charge values on fixed grid ... we'd need grad wrt charge, not position!
# - surround mol with charges confined to boxes, each with zero net charge
# - reduandant internals w/ TC, use bounds w/ lbfgsb ... WIP
# - Lagrange multipliers? would need separate one for each charge?
# - add restraining potential (effectively the same as MM force field constraining position of MM charges)
# - keep outside a sphere: use spherical coords and set lower bound on r
#  - could we use ellipse instead?
# At least this gives us some info: negative charge near the moving H atom should lower energy

# To get <> to work: Ctrl+Shift+1 to unfocus Mol 1, Ctrl+Shift+2 to focus Mol 2
mqq = Molecule(atoms=[Atom(name='Q', r=q.r, mmq=q.qmq) for q in charges])
vis = Chemvis([Mol(ch2o, [r_ts], [ VisGeom(style='lines') ]), Mol(mqq, [res.x], VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-0.1,0.1]))) ], fog=True, wrap=False).run()



## NEB
from chem.opt.neb import NEB

NEBe = RCe[:]
NEBr = RCr[:]
r_ch2o_align = align_atoms(r_ch2o, r_hcoh)
neb = NEB(R=r_hcoh, P=r_ch2o_align, EandG=qmmm.EandG, nimages=len(NEBr), k=2.0, climb=False)
NEBres = lbfgs.optimize(neb.EandG, neb.active(), ftol=0)  #gtol=1e-4 ... we can relax gtol unless we need exact path
NEBr[1:-1] = np.reshape(NEBres.x, (-1, ch2o.natoms, 3))
NEBe = [qmmm.EandG(xyz, dograd=False) for xyz in NEBr]
E_nebts, r_nebts, dr_nebts = path_ts(NEBe, NEBr)

# gtol=5e-4: 10 steps, |dr| ~ 0.18 (per step) (optim k probably depends on this |dr|!)
# k = 0.1: 43 iter; k=0.33: 46; k = 1.0: 35, k = 2: 31; k=3.3: 31; k = 10.0: 54
# k = 2, climb=True: 31 iter; energy and |dr| of highest image almost the same as climb=False
# k = 2, starting from RCr: 12 iter
for ii in range(1, len(RCr)): RCr_align[ii] = align_atoms(RCr[ii], RCr[ii-1])



## vibrational freqs / normal modes
from pyscf.hessian import thermo
hess = qmmm.prev_pyscf.base.Hessian().kernel()
thermo_res = thermo.harmonic_analysis(qmmm.prev_pyscf.base.mol, hess)
print(thermo_res['freq_wavenumber'])


# it looks like visualizing electron density is more useful than visualizing ESP
ch2o.r = r_ch2o
qmmm.EandG(ch2o.r, dograd=False)
ch2o.pyscf_mf = qmmm.prev_pyscf.base
from chem.vis.chemvis import *
vis = Chemvis(Mol(ch2o, [ VisGeom(style='lines'), VisVol(pyscf_esp_vol, vis_type='volume', vol_type="ESP", timing=True) ]), bg_color=Color.white).run()


# test our optimizers
res, _ = opt.optimize(ch2o, fn=qmmm.EandG, optimizer=lbfgs.optimize, ftol=2E-07)
r_ch2o_2, E_ch2o_2 = res.x, res.fun

res, _ = opt.optimize(ch2o, r0=hcoh.r, fn=qmmm.EandG, optimizer=lbfgs.optimize, ftol=2E-07)
r_hcoh_2, E_hcoh_2 = res.x, res.fun



# for CH2CHOH -> CH3CHO, this gives path similar to linear interpolation; some literature suggests this may
#  work better in some cases
def dlc_interp(R, P, nimages, align_sel=None):
  """ Interpolate between geometries R and P in total connection DLC coords """
  # DLC requires Molecule object, but anything with correct number of atoms will work for total connection
  mol = Molecule(r_array=R, z_array=[1]*len(R))
  dlc = DLC(mol, autobonds='total', autoangles='none', autodiheds='none', recalc=1)  # recalc is essential
  dlc.init(R)
  sR = dlc.active()
  sP = dlc.fromxyz(P)
  ds = (sP - sR)/(nimages - 1)
  s = sR
  images = [R]
  for ii in range(1, nimages-1):
    s += ds
    dlc.update(s)
    images.append(align_atoms(dlc.xyzs(), images[-1], sel=align_sel))
  images.append(P)
  return images


# C1 - H4 dist will be reaction coordinate
rc = [calc_dist(xyz[0], xyz[3]) for xyz in NEBxyz]
rc = rc - rc[0]  # C1 - H4 dist will decrease - make RC increase from 0


print_mos(qmmm.prev_pyscf.base)
qmmm.qm_opts.scffn = scf.UHF
res, r_hcoh_uhf = optimize(ch2o, r0=hcoh.r, fn=qmmm.EandG, ftol=2E-07)
print_uhf_mos(qmmm.prev_pyscf.base)



qmmm.qm_opts.scffn = scf.UHF
for d in range(1, 10):
  co.bond((0,1), d)
  E, _ = qmmm.EandG(co)
  print("\n%.3f: %.6f Hartree\nMOs:\n" % (d, E))
  print_uhf_mos(qmmm.prev_pyscf.base)


res, r_co = optimize(co, fn=qmmm.EandG, ftol=2E-07)
assert res.success, "CO optimization failed"

qmmm.EandG(co, r=r_co)


