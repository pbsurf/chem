# transition state search
# play with hcoh -> ch2o, then ch2o -> co + h2
# Refs for hcoh -> ch2o: www.sas.upenn.edu/~mabruder/GAMESS.html

# TS code:
# * github.com/eljost/pysisyphus / pysisyphus.readthedocs.io / 10.1002/qua.26390 - Dimer (multiple approaches), Lanczos, NEB, String, IRC; DLC; excited states
# * gitlab.com/ase/ase - NEB, Dimer
# * github.com/pele-python/pele / pele-python.github.io - Dimer, NEB, basin hopping global opt, etc (Wales group)
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
import scipy.optimize
from chem.molecule import *
from chem.io import *
from chem.qmmm.qmmm1 import QMMM
from chem.opt.dlc import *
from chem.opt.lbfgs import *
from chem.opt.optimize import *
from pyscf import scf
from chem.vis.chemvis import *
from chem.test.common import *
from chem.theo import *

from chem.io.pyscf import pyscf_EandG
from chem.io.tinker import tinker_EandG


ch2o_xyz = """4  molden generated tinker .xyz (amber96 param.)
 1  C     0.000000    0.000000    0.000000      342  2  3  4
 2  O     0.000000    0.000000    1.400000      343  1
 3  H     1.026720    0.000000   -0.362996      341  1
 4  H    -0.513360   -0.889165   -0.363000      341  1
"""

ch2o_opt_xyz = """4  molden generated tinker .xyz (mm3 param.)
 1  C     0.130284   -0.225658    0.155628      3  2  3  4
 2  O    -0.044368    0.076846    1.311490      7  1
 3  H     1.042499    0.108318   -0.396556      5  1
 4  H    -0.615055   -0.848670   -0.396558      5  1
"""

hcoh_xyz = """4  molden generated tinker .xyz (amber96 param.)
 1  C     0.000000    0.000000    0.000000    342  2  3
 2  O     0.000000    0.000000    1.400000    343  1  4
 3  H     1.026720    0.000000   -0.362996    341  1
 4  H    -0.447834    0.775672    1.716667    341  2
"""

co_xyz = """2  molden generated tinker .xyz (mm3 param.)
 1  C     0.000000    0.000000    0.000000      3  2
 2  O     0.000000    0.000000    1.400000      7  1
"""

#ch3cho = load_molecule(ch3cho_xyz)
#ch2choh = load_molecule(ch2choh_xyz)
#ch2choh.r = align_atoms(ch2choh, ch3cho, sel=[0,1,2])  # align the heavy atoms
#qmmm = QMMM(mol=ch3cho, qmatoms='*', moguess='prev', qm_opts=Bunch(prog='pyscf'), prefix='no_logs', savefiles=False)

ch2o = load_molecule(ch2o_xyz)
hcoh = load_molecule(hcoh_xyz)
#co = load_molecule(co_xyz)

def print_mos(mf):
  homo = np.max(np.nonzero(mf.mo_occ))
  print(''.join(["%.6f (%d)\n" % (mf.mo_energy[ii], mf.mo_occ[ii]) for ii in range(homo+2)]))

def print_uhf_mos(mf):
  homo = np.max(np.nonzero(mf.mo_occ))
  print(''.join(["%.6f (%d); %.6f (%d)\n" % (mf.mo_energy[0][ii], mf.mo_occ[0][ii], mf.mo_energy[1][ii], mf.mo_occ[1][ii]) for ii in range(homo+2)]))


qmbasis = '6-31G*' #'6-31++G**'
qmatoms = init_qmatoms(ch2o, '*', qmbasis)

## RC scan
nimages = 10
c1_h4 = (0,3)  # bonds
o2_h4 = (1,3)

ts_test1_h5 = 'ts_test1.h5'
if os.path.exists(ts_test1_h5):
  rcs, RCr, RCe = read_hdf5(ts_test1_h5, 'rcs', 'RCr', 'RCe')
  r_hcoh, r_ch2o = RCr[0], RCr[-1]
  E_hcoh, E_ch2o = RCe[0], RCe[-1]
  r_ch2o = align_atoms(r_ch2o, r_hcoh, sel=[0,1,2])
else:
  qmmm = QMMM(mol=ch2o, qmatoms=qmatoms, moguess='prev', qm_opts=Bunch(prog='pyscf'), prefix='no_logs', savefiles=False)

  RCr = [0]*nimages
  RCe = [0]*nimages

  res, r_hcoh = moloptim(qmmm.EandG, mol=ch2o, r0=hcoh.r, ftol=2E-07)
  RCr[0], RCe[0] = r_hcoh, res.fun

  res, r_ch2o = moloptim(qmmm.EandG, mol=ch2o, ftol=2E-07)
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
  rcs = np.linspace(rc0, rc1, nimages)
  ch2o.r = RCr[0]

  # could we use gradient at each point to make a smoother graph?
  for ii,rc in enumerate(rcs[1:-1]):
    # reverse bonds and use move_frag=False to keep H4 fixed
    ch2o.bond(o2_h4[::-1], 0.5*(rc - rcs[ii]), rel=True, move_frag=False)
    ch2o.bond(c1_h4[::-1], -0.5*(rc - rcs[ii]), rel=True, move_frag=False)
    #ch2o.bond(c1_h4[::-1], ch2o.bond(o2_h4) - rc, move_frag=False)  # move C1
    #print("rc: %f; O2-H4 - C1-H4: %f" % (rc, ch2o.bond(o2_h4) - ch2o.bond(c1_h4)))
    res, r = moloptim(ch2o, fn=qmmm.EandG, coords=dlc, ftol=2E-07)  #, raiseonfail=False)
    RCr[ii+1], RCe[ii+1] = r, res.fun
    ch2o.r = r

  write_hdf5(ts_test1_h5, rcs=rcs, RCr=RCr, RCe=RCe)

  vis = Chemvis(Mol(ch2o, RCr, [ VisGeom(style='lines') ]), fog=True, wrap=False).run()

  plot(rcs, RCe, xlabel="C1-H4 (Angstroms)", ylabel="energy (Hartrees)")


# possible alternatives to HDF5: pickle, numpy save, or sqlite
r_ts, bq_coords, bq_qs = read_hdf5(ts_test1_h5, 'r_ts', 'bq_coords', 'bq_qs')
bq_coords = np.asarray(bq_coords)
charges = [ Bunch(r=rbq, qmq=qbq) for rbq, qbq in zip(bq_coords, bq_qs) ]
mqq = Molecule(atoms=[Atom(name='Q', r=q.r, mmq=q.qmq) for q in charges])
r_com = center_of_mass(ch2o)

E_ts0, _, _ = pyscf_EandG(ch2o, r_ts)
E_rs0, E_ps0 = E_hcoh, E_ch2o


## Tinker setup for LIG
TINKER_FF = os.path.abspath(os.path.expandvars("$TINKER_PATH/../params/amber96.prm"))
tinker_qmmm_key = \
"""parameters     %s
digits 8

# even with intra-group interactions disabled, Tinker requires parameters for all atoms
bond          2   34          340.00     1.0900
angle        24    2   34      80.00     126.00
angle        34    2   34      80.00     126.00

# QM charges appended here...
""" % TINKER_FF

## Add polar or charged residues based on charges found by grid optim (moved below)
# ARG (+1; +0.44 on terminal hydrogens)  GLU (-1; -0.8 on each oxygen)
if os.path.exists('theo0.pdb'):
  theo_ch2o = quick_load('theo0.pdb', charges='generate')
else:
  theo_ch2o = Molecule()
  theo_ch2o.append_atoms(place_residue('GLN', [11,18], bq_coords[[1,5]], r_com))  # GLN OE1: 11; HE22 18
  theo_ch2o.append_atoms(place_residue('GLN', [11,18], bq_coords[[4,3]], r_com))
  if 0:
    theo_ch2o.append_atoms(place_residue('GLU', [11,12], bq_coords[[1,4]], r_com)) # OE1, OE2
    # we want ARG (+1 charge) to extend away from GLU (-1 charge) to reduce distortion (when relaxed)
    idx_cd = select_atoms(theo_ch2o, pdb='* GLU CD')[0]
    r_glucd = theo_ch2o.atoms[idx_cd].r
    theo_ch2o.append_atoms(place_residue('ARG', [22,24], bq_coords[[0,3]], r_glucd)) #bq_coords[[3,7]]
    theo_ch2o.append_atoms(place_residue('ARG', [22,24], bq_coords[[5,6]], r_glucd))  # HH11, HH21; ARG Nitrogens: 13,14
    #theo_ch2o.append_atoms(place_residue('GLU', [11,12], bq_coords[[2,5]], r_com))

  quick_save(theo_ch2o, 'theo0.pdb')

# add the substrate
qmatoms = theo_ch2o.append_atoms(ch2o, r=r_ts, residue='LIG')

r_theo0 = theo_ch2o.r
# zero charges on backbone (but not CA or HA):
zero_q = [ (ii, 0) for ii in select_atoms(theo_ch2o, pdb='~LIG N,C,O,H1,H2,H3,OXT') ]  # 'protein and not sidechain'


## (Disabled for now) relax catres w/ spring force constraints holding desired atoms near charge positions
if 1:
  r_theo1 = r_theo0
elif os.path.exists('theo1.xyz'):
  r_theo1 = load_molecule('theo1.xyz').r
else:
  # relax w/ constraint holding selected atoms near background charge positions - catres move away from LIG w/o constraints
  #consatoms = sorted(select_atoms(theo_ch2o, pdb='GLU OE1,OE2; ARG HH11,HH21'))  # sorted shouldn't be necessary
  consatoms = sorted(select_atoms(theo_ch2o, pdb='GLN OE1,HE22'))
  r_cons = r_theo0[consatoms]  #bq_coords[[1,4,0,3,5,6]]

  # k_cons = 0.1 seems good - 0.01 too small, 1.0 doesn't seem much different from 0.1
  def theoEandG(mol, r, k_cons=0.1, printE=True, **kwarg):
    E, G = tinker_EandG(mol, r, **kwarg)
    dr_cons = r[consatoms] - r_cons
    E_cons = k_cons*np.sum(dr_cons**2)  # k_cons units are Hartree/Ang^2
    G_cons = 2*k_cons*dr_cons
    if printE:
      print("MM energy: %f; Constraint energy: %f" % (E, E_cons))
    G[consatoms] += G_cons
    return E + E_cons, G

  xyz_theo = opt.XYZ(r_theo0, select_atoms(theo_ch2o, pdb='~LIG *'))

  # w/o constraints, opt runs for >1000 iterations, so limit iter
  res, r_theo1 = moloptim(theoEandG, mol=theo_ch2o, r0=r_theo0, coords=xyz_theo, maxiter=60, raiseonfail=False, fnargs=dict(key=tinker_qmmm_key, charges=zero_q))
  write_tinker_xyz(theo_ch2o, 'theo1.xyz', r=r_theo1)

  #xyz_theosc = opt.XYZ(r_theo1, select_atoms(theo_ch2o, 'sidechain'))
  #res, r_theo2 = moloptim(tinker_EandG, mol=theo_ch2o, r0=r_theo1, coords=xyz_theosc, vis=vis, fnargs=dict(key=tinker_qmmm_key, charges=zero_q))


## find placement of (rigid) catres for optimal catalysis (minimize E_ts w/o lowering E_rs or E_ps)
# - disabling MM charge-charge interaction (while keeping vdW so catres can't collide) seems to work
def qmmm_opt_EandG(mol, r, qmmm):
  r_mm1 = np.reshape(r, (-1, 3))[:-len(r_ch2o)]
  r_rs1, r_ts1, r_ps1 = np.vstack((r_mm1, r_hcoh)), np.vstack((r_mm1, r_ts)), np.vstack((r_mm1, r_ch2o))

  # role of MM energy is to prevent "collisions" between atoms
  # Note that only the LIG atom positions differ here, so difference in MM energies is just in LIG interaction
  E_rs, G_rs = qmmm.EandG(r_rs1)
  dE_rs = qmmm.Ecomp.Eqm - E_rs0
  G_rs = qmmm.Gcomp.Gqm
  E_mm = qmmm.Ecomp.Emm
  G_mm = qmmm.Gcomp.Gmm

  E_ts, G_ts = qmmm.EandG(r_ts1)
  dE_ts = qmmm.Ecomp.Eqm - E_ts0
  G_ts = qmmm.Gcomp.Gqm
  E_mm += qmmm.Ecomp.Emm
  G_mm += qmmm.Gcomp.Gmm

  E_ps, G_ps = qmmm.EandG(r_ps1)
  dE_ps = qmmm.Ecomp.Eqm - E_ps0
  G_ps = qmmm.Gcomp.Gqm
  E_mm += qmmm.Ecomp.Emm
  G_mm += qmmm.Gcomp.Gmm

  # allow, but do not drive increase of E_ps, E_rs; heavily penalize decrease
  print("dE_rs: %f; dE_ts: %f; dE_ps: %f; E_mm/3: %f" % (dE_rs, dE_ts, dE_ps, E_mm/3))
  # beyond this, opt hovers around |proj g| ~ 1E-3 with little improvement but severe distortion of geometry
  #if dE_ts < -0.0245:
  #  raise ValueError('Stop')
  # fixed values don't work (optim. fails), but this works well:
  c_ps = -dE_ps/0.001 if dE_ps < 0 else 0  # 0.001 Hartree ~ kB*T
  c_rs = -dE_rs/0.001 if dE_rs < 0 else 0
  # d/dr(dE*dE/0.001) = 2*(dE/0.001)*d(dE)/dr
  E = E_mm/3 + dE_ts - c_ps*dE_ps - c_rs*dE_rs
  G = G_mm/3 + G_ts - 2*c_ps*G_ps - 2*c_rs*G_rs
  return E, G


if os.path.exists('theo2.xyz'):
  r_theo2 = load_molecule('theo2.xyz').r
else:
  theo_ch2o.r = r_theo1
  vis = Chemvis([Mol(theo_ch2o, [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()

  theo_qmmm = QMMM(mol=theo_ch2o, qmatoms=qmatoms, qm_opts=Bunch(prog='pyscf'), prefix='theo2', savefiles=False, mm_key=tinker_qmmm_key + "charge-cutoff 0.0\n")  # no charge-charge interactions
  #sanity_check(lambda r: qmmm_opt_EandG(None, r, theo_qmmm), r_theo1)

  rgds = HDLC(theo_ch2o, [XYZ(r_theo1, active=[], atoms=res.atoms) if res.name == 'LIG' else Rigid(r_theo1, res.atoms) for res in theo_ch2o.residues])
  res, r_theo2 = moloptim(qmmm_opt_EandG, fnargs=Bunch(qmmm=theo_qmmm), mol=theo_ch2o, r0=r_theo1, coords=rgds, vis=vis, raiseonfail=False)
  write_tinker_xyz(theo_ch2o, 'theo2.xyz', r=r_theo2)


# start interactive
import pdb; pdb.set_trace()

## place additional non-catalytic residues randomly around active site and relax w/ just LJ potential
if os.path.exists('theo3.pdb'):
  theo_ch2o = quick_load('theo3.pdb')
  load_tinker_params(theo_ch2o, 'theo3.xyz', charges=True, vdw=True)
  r_theo3 = theo_ch2o.r
  qmatoms = init_qmatoms(theo_ch2o, theo_ch2o.select(pdb='LIG *'), qmbasis)
else:
  vis = Chemvis([Mol(theo_ch2o, [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()

  if 1:  #while 1:
    theo_ch2o.remove_atoms('* LEU *')  # clear previous
    # rotate randomly, then place a specified atom at specified position (with some random jitter)
    # One potential sanity check might be comparing radial distribution fn to actual protein
    origin = np.mean(theo_ch2o.r[qmatoms], axis=0)
    dirs = [normalize(np.mean(theo_ch2o.r[res.atoms], axis=0) - origin) for res in theo_ch2o.residues[:-1]]
    ndirs = len(dirs)
    # generate random directions with some minimum angular separation
    while len(dirs) < 6 + ndirs:
      dir = normalize(np.random.randn(3))  # randn: normal (Gaussian) distribution
      if all(np.dot(dir, d) < 0.5 for d in dirs):
        dirs.append(dir)

    for dir in dirs[ndirs:]:
      refpos = origin + 4.0*dir + (np.random.random(3) - 0.5)
      # to support other residues, we place centroid of sidechain at refpos, then rotate to align CA w/ dir
      #mol = place_residue('LEU', [9], [refpos], origin)  # 9 is LEU CG
      mol = place_residue('LEU')
      r = mol.r
      r = r - np.mean(r[mol.select('sidechain')], axis=0)
      r = np.dot(r, align_vector(r[1] - origin, dir).T) + refpos  # [1] is CA for Tinker residues
      theo_ch2o.append_atoms(mol, r=r)

    load_tinker_params(theo_ch2o, charges=True, vdw=True)
    bbone = theo_ch2o.select(pdb='~LIG N,C,O,H1,H2,H3,OXT')
    molLJ = MolLJ(theo_ch2o, repel=[(ii, jj) for ii in bbone for jj in theo_ch2o.listatoms()])

    vis.refresh(repaint=True)

    r_theo3 = theo_ch2o.r
    #rgds = HDLC(theo_ch2o, [Rigid(r_theo3, res.atoms) for res in theo_ch2o.residues])
    rgds = HDLC(theo_ch2o, [XYZ(r_theo3, active=[], atoms=res.atoms) if res.name != 'LEU' else
        Rigid(r_theo3, res.atoms) for res in theo_ch2o.residues])
    res, r_theo3 = moloptim(molLJ, mol=theo_ch2o, r0=r_theo3, coords=rgds, vis=vis, ftol=2E-07)

  quick_save(theo_ch2o, 'theo3.pdb', r=r_theo3)

# In progress
## Adjust non-catres to constrain position of LIG (as close as possible to standalone positions used above)

#load_tinker_params(theo_ch2o, charges=True, vdw=True)
#hess_sanity(partial(coulomb, q=theo_ch2o.mmq[qmatoms]), theo_ch2o.r[qmatoms])
#molLJ = MolLJ(theo_ch2o)
#hess_sanity(molLJ, theo_ch2o.r)

vis = Chemvis([Mol(theo_ch2o, [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()

if 0:  # ... looking like a bust
  molLJ = MolLJ(theo_ch2o)

  # We want to minimize |G|**2 for LIG (while keeping LIG fixed)
  # return value and grad for |grad|**2 given a fn returning energy, grad, and Hessian
  def gradsq(mol, r=None):
    E, G, H = molLJ(mol if r is None else r, hess=True)
    Ga = G[qmatoms]
    Ha = H[:,qmatoms]
    return np.sum(Ga*Ga), 2*np.einsum('ijkl,jk->il', Ha, Ga)


  EandG_sanity(gradsq, r_theo3)

  rgds = HDLC(theo_ch2o, [XYZ(r_theo3, active=[], atoms=res.atoms) if res.name != 'LEU' else
      Rigid(r_theo3, res.atoms) for res in theo_ch2o.residues])
  res, r_theo4 = moloptim(gradsq, mol=theo_ch2o, r0=r_theo3, coords=rgds, vis=vis, gtol=1E-08)
else:
  r_theo4 = r_theo3.copy()

# Next: reaction path w/ full active site; can we do anything to further optimize?
## Reaction path: with full active site

zero_q = [ (ii, 0) for ii in theo_ch2o.select(pdb='~LIG N,C,O,H1,H2,H3,OXT') ]

# reaction path, TS search with r_theo2
qmmm = QMMM(theo_ch2o, qmatoms=qmatoms, mm_opts=Bunch(charges=zero_q), prefix='theo4', savefiles=False, mm_key=tinker_qmmm_key)

c1_h4 = [qmatoms[ii] for ii in (0,3)]
o2_h4 = [qmatoms[ii] for ii in (1,3)]

nimages = 10

if 1:
  RC3r = [0]*nimages
  RC3e = [0]*nimages

  xyz_theo3 = XYZ(r_theo4, active=qmatoms)

  res, r_hcoh_3 = moloptim(qmmm.EandG, mol=theo_ch2o, r0=setitem(r_theo4.copy(), qmatoms, r_hcoh), coords=xyz_theo3, vis=vis, ftol=2E-07)
  RC3r[0], RC3e[0] = r_hcoh_3, res.fun

  res, r_ch2o_3 = moloptim(qmmm.EandG, mol=theo_ch2o, r0=setitem(r_theo4.copy(), qmatoms, r_ch2o), coords=xyz_theo3, vis=vis, ftol=2E-07)
  RC3r[-1], RC3e[-1] = r_ch2o_3, res.fun

#RC3r[0], RC3r[-1] = np.vstack((r_mm3, r_hcoh)), np.vstack((r_mm3, r_ch2o))
#RC3e[0]. RC3e[-1] = qmmm.EandG(RC3r[0]), qmmm.EandG(RC3r[-1])

#rcs, RC3r, RC3e = read_hdf5('ts_test3.h5', 'rcs', 'RC3r', 'RC3e')

dlc_lig = DLC(theo_ch2o, atoms=qmatoms, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
# note that using a single bond length (c1_h4) for RC didn't work - H ends up far from both C and O for large RCs
dlc_lig.constraint(dlc_lig.bond(o2_h4) - dlc_lig.bond(c1_h4))

xyz_prot = opt.XYZ(r_theo4, active=[], atoms=theo_ch2o.listatoms(exclude=qmatoms))
hdlc = HDLC(theo_ch2o, [xyz_prot, dlc_lig])

rc0 = calc_dist(RC3r[0][o2_h4]) - calc_dist(RC3r[0][c1_h4])
rc1 = calc_dist(RC3r[-1][o2_h4]) - calc_dist(RC3r[-1][c1_h4])
rcs = np.linspace(rc0, rc1, nimages)
theo_ch2o.r = RC3r[0]

for ii,rc in enumerate(rcs[1:-1]):
  # reverse bonds and use move_frag=False to keep H4 fixed
  theo_ch2o.bond(o2_h4[::-1], 0.5*(rc - rcs[ii]), rel=True, move_frag=False)
  theo_ch2o.bond(c1_h4[::-1], -0.5*(rc - rcs[ii]), rel=True, move_frag=False)
  #ch2o.bond(c1_h4[::-1], ch2o.bond(o2_h4) - rc, move_frag=False)  # move C1
  #print("rc: %f; O2-H4 - C1-H4: %f" % (rc, ch2o.bond(o2_h4) - ch2o.bond(c1_h4)))
  res, r = moloptim(qmmm.EandG, mol=theo_ch2o, coords=hdlc, ftol=2E-07)  #, raiseonfail=False)
  RC3r[ii+1], RC3e[ii+1] = r, res.fun
  theo_ch2o.r = r

write_hdf5('ts_test3.h5', rcs=rcs, RC3r=RC3r, RC3e=RC3e)

# OK, this looks like a start!
# Next: how to optimize w/ complete active site?
# - minimize E_ts ( - E_rs) ... mostly catres
# - minimize |E_ps - E_ps0| and |E_rs - E_rs0| ... could we minimize (E_ps - E_ps0)**2, etc? (deriv is 2*(E_ps - E_ps0)*dE_ps/dr) ... mostly non-catres
# ... what if we move either LIG or non-catres depending on sign of E_ps - E_ps0?
# - note that we can also compute charge-charge MM interaction in python, so multi-stage optim might be feasible:

# Overall procedure:
# 1. dTS - dRS - dPS optim w/ catres only and fixed LIG
# 2. optim non-catres positions to constrain LIG
# 3. fine tune catres pos (?)

# Step 2:
# - problem is we want E_rs, E_ps for relaxed position, not fixed (soln) pos, so can't get grad
# - w/ analytic Hessian, we could minimize grad**2 (how to avoid maxima and saddles?) ... this may be worth trying if we can't think of anything else

# Fundamentally, how do we optimize other params to put LIG into a minimum?  How do we know LIG is in minimum?  grad == 0, curvature > 0 (all Hessian eigenvals > 0)


# Maybe gradient optim isn't so useful for this step anyway, since we have discrete rotamer, residue params
# - question is how effective would just moving non-catres be at constraining LIG pos (vs. changing residues, rotamers)

# How would we do global optim (no gradient)?
# - for ensemble of (relaxed) active sites, relax RS, PS - pick site(s) w/ energy (and position?) closest to soln case
#  - when relaxing active sites, do initial optim w/ large LJ depth for LIG atoms to penalize voids around LIG



# ? - given a TS geom and closest RS and PS geoms, we'd want to relax RS, PS and make sure energy is not too much lower than unbound state, nor is position significantly changed

# optimizing full active site
# - usual optim dTS - dRS - dPS, moving all residues ... periodically sub-optim RS, PS positions? or allow LIG to move too (rigidly)? ... if all residues moving, no need to allow LIG to move ... unless we want to account for differences in positioning of RS,TS,PS ... how would we eliminate these?
# - w/o MM charge-charge interaction; what will this gain over optim w/ just catres?

# One issue is that catres-only opt showed decrease in barrier, but after RS, PS relaxation, barrier was higher than ref!

# how would we include LIG - catres vdW energy in dTS - dRS - dPS optim? continue to use soln ref?
# - actually OK if RS energy drops, as long as barrier drops more and PS energy doesn't drop
# - we could shift ref for RS energy and use dEts = (Ets - Ers) - (Ets0 - Ers0) instead of Ets - Ets0

# How to avoid configurations where poise of RS,TS,PS differ significantly and lower charge-charge energy is compensated for by higher vdW energy?

# dTS - dRS - dPS optim w/ non-catres didn't do much (no improvement, only a little movement)
# - so the issue really is w/ the optimized position of RS (and PS)

# what if we optim w/ catres and TS fixed, but noncatres, RS, PS moving? but this won't give equilib position for RS, PS!
# So: how do we find poise to hold RS, PS in desired position?
# - looks like non-catres don't significantly affect barrier, so we could move those around independently
# - would moving non-catres to minimize E be sufficient? (due to vdW) (would probably need to include TS too to prevent clash) ... so basically LJ opt averaging over RS,TS,PS as suggested above
# - suppose this isn't good enough - what else could we do?
# - would a larger LIG really save us?  reacting group could still rotate about dihedral(s) to undesired position!
# - in general, how would we pack active site to keep LIG from moving
#  - even more generally, how to drive all components of grad to zero while keeping some inputs fixed (something to do with off-diag elements of Hessian?)
#  work in full cartesians, but prevent some from changing (instead of hiding them from optimizer) ... so would passing bounds to scipy.optimize give different result than hiding the variables?
# - if we had analytic Hessian, we could minimize norm of gradient (instead of E) ... but this could converge to maximum or saddle! ... could we use estimated Hessian (e.g. BFGS) in nested optim procedure?

# issue is that in general, relaxed active site will have voids, and LIG could move around in a void
# - could we have objective fn penalizing voids?
# - what about making vdW depth much larger for LIG atoms? <= THIS
#  - then decrease towards actual value
# - would adding non-catres sequentially (optimizing each separately) change anything?

# 1. trying rotatmers and other residues for non-catres
# 1. what about generating set of random active sites and seeing if there are differences in mobility of RS,PS
# 1. would thinking about the process of crystalization/liquid->solid transition help?
# 1. try larger LIG
# 1. for TRIC coords using min-RMSD alignment to ref, we need dQ/dX for the full procedure, not deriv of rot mat wrt expmap vector components!

# - QM energy of system with catres shouldn't be significantly different than w/ charges
#  - if it is, double check with just the MM charges

# topology check: check for atoms within some threshold dist of each hemisphere (6 in 3D) of atoms we want to constrain
# - bonded atom could be counted for opposite hemisphere as well


from scipy.spatial.ckdtree import cKDTree
kd = cKDTree(r_array)
qidx = qmatoms[0]  #qmatoms[1]
qatom = mol.atoms[qidx]
dists, locs = kd.query(qatom.r, k=6)
for ii in locs:
  r = mol.bond(qidx, ii)
  r_lj = 0.5*(qatom.lj_r0 + mol.atoms[ii].lj_r0)
  print("%s - %s: dist = %f = %f * vdW dist" % (qatom.name, mol.atoms[ii].name, r, r/r_lj))





if 0:
  import pickle
  with open('ts_test3.pickle',"rb") as f:
    rcs = pickle.load(f)
    RC3r = pickle.load(f)
    RC3e = pickle.load(f)

  with open('ts_test3.pickle',"wb") as f:
    pickle.dump(rcs, f)
    pickle.dump(RC3r, f)
    pickle.dump(RC3e, f)


if 0:
  # freeze LIG
  fzn = select_atoms(theo_ch2o, pdb='LIG *')
  # freeze backbone
  #active = select_atoms(theo_ch2o, 'sidechain and name != "HA"')
  #fzn = theo_ch2o.listatoms(exclude=active)  #select_atoms(theo_ch2o, 'not sidechain')
  # freeze tip of sidechain
  #fzn = select_atoms(theo_ch2o, pdb='GLU OE1,OE2; ARG HH11,HH21;LIG *')  # 3rd atom: ARG CZ; GLU CD
  #active = theo_ch2o.listatoms(exclude=fzn)

  hdlc = HDLC(theo_ch2o, autodlc=1, dlcoptions=dict(recalc=1))
  hdlc.constrain([[ii] for ii in fzn])  # frozen atoms
  # also freeze sidechain bond lengths and angles
  for dlc in hdlc.dlcs:
    for ii in range(len(dlc.bonds) + len(dlc.angles)):
      dlc.constraint(setitem(np.zeros(dlc.nQ), ii, 1.0))
    # for this to work, we'd need to change get_internal:
    #dlc.local_cons_idx = True
    #dlc.constrain(dlc.bonds)
    #dlc.constrain(dlc.angles)
  #hdlc.init()

  # this works, but catres are moving too much
  res, r_theo1 = moloptim(tinker_EandG, mol=theo_ch2o, coords=hdlc, fnargs=dict(key=tinker_qmmm_key, charges=zero_q))

vis = Chemvis([Mol(theo_ch2o, [r_theo0, r_theo1], [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()



if 0:
  coords_sc = opt.XYZ(theo_ch2o.r, select_atoms(theo_ch2o, pdb='~LIG ~C,N,CA'))  # sidechains

  # move (non-cat) residues, relax sidechains, repeat
  vis.select('resnum == 4')
  res, r_mm_opt = moloptim(tinker_EandG, mol=theo_ch2o, coords=coords_sc, fnargs=dict(charges=zero_q))
  theo_ch2o.r = r_mm_opt
  vis.refresh()

  # sanity check: total MM energy of theozyme should be less than sum of energies of isolated residues (to ensure no clashes)
  frags = [theo_ch2o.extract_atoms(resnums=ii) for ii in range(len(theo_ch2o.residues))]
  fragE = [tinker_EandG(frag, key=tinker_qmmm_key, charges=[ (ii, 0) for ii in select_atoms(frag, pdb='~LIG N,C,O,H1,H2,H3,OXT') ])[0] for frag in frags]
  np.sum(fragE)



# we need to relax the MM atoms w/o substrate
if 0:
  coords_sc = opt.XYZ(theo_ch2o.r, select_atoms(theo_ch2o, pdb='~C,N,CA'))
  res, r_mm0 = moloptim(tinker_EandG, r0=theo_ch2o.r, mol=theo_ch2o, coords=coords_sc)
  E_mm0 = res.fun
else:
  r_mm0 = r_mm_init
  E_mm0 = tinker_EandG(theo_ch2o)

vis = Chemvis([Mol(theo_ch2o, [r_mm_init, r_mm0], [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()


# add the substrate
qmatoms = theo_ch2o.append_atoms(ch2o, r_ts)

if 0:
  #r_qmmm_rs, r_qmmm_ts, r_qmmm_ps = np.vstack((r_hcoh, r_mm)), np.vstack((r_ts, r_mm)), np.vstack((r_ch2o, r_mm))
  r_rs0, r_ts0, r_ps0 = np.vstack((r_mm0, r_hcoh)), np.vstack((r_mm0, r_ts)), np.vstack((r_mm0, r_ch2o))

  vis = Chemvis([Mol(theo_ch2o, [r_rs0, r_ts0, r_ps0], [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()


theo_qmmm = QMMM(mol=theo_ch2o, qmatoms=qmatoms, moguess='prev', qm_opts=Bunch(prog='pyscf'), prefix='theo1', savefiles=False, mm_key=tinker_qmmm_key)

r_mm1 = r_mm_init
r_rs1, r_ts1, r_ps1 = np.vstack((r_mm1, r_hcoh)), np.vstack((r_mm1, r_ts)), np.vstack((r_mm1, r_ch2o))
theo_qmmm.EandG(r_rs1)[0]

##>>> theo_qmmm.EandG(r_rs1)[0]  -114.61817752285707
##>>> theo_qmmm.EandG(r_ts1)[0]  -114.54888796655254
##>>> theo_qmmm.EandG(r_ps1)[0]  -114.69893572274714
# this actually looks promising ... now we need to constrain catalytic residues

# what should reference states be (only need for rs and ps)?
#E_rs0 = E_hcoh + E_{relaxed empty active site}; E_ps0 = E_ch2o + E_{relaxed empty active site} ? (also, empty = solvated)
E_rs0 = E_hcoh + E_mm0
E_ts0 = E_ts + E_mm0
E_ps0 = E_ch2o + E_mm0


# maybe try opt moving residues
def qmmm_opt_EandG(r_mm):
  r_mm1 = np.reshape(r_mm, (-1, 3))
  r_rs1, r_ts1, r_ps1 = np.vstack((r_mm1, r_hcoh)), np.vstack((r_mm1, r_ts)), np.vstack((r_mm1, r_ch2o))
  E_rs, G_rs = theo_qmmm.EandG(r_rs1)
  E_ts, G_ts = theo_qmmm.EandG(r_ts1)
  E_ps, G_ps = theo_qmmm.EandG(r_ps1)
  # allow, but do not drive increase of E_ps, E_rs; heavily penalize decrease
  dE_rs = E_rs - E_rs0
  dE_ts = E_ts - E_ts0
  dE_ps = E_ps - E_ps0
  print("dE_rs: %f; dE_ts: %f; dE_ps: %f" % (dE_rs, dE_ts, dE_ps))
  # fixed values don't work (optim. fails), but this works well:
  c_ps = -dE_ps/0.001 if dE_ps < 0 else 0  # 0.001 Hartree ~ kB*T
  c_rs = -dE_rs/0.001 if dE_rs < 0 else 0
  # d/dq(dE*dE/0.001) = 2*(dE/0.001)*d(dE)/dq
  return dE_ts - c_ps*dE_ps - c_rs*dE_rs, np.ravel(G_ts - 2*c_ps*G_ps - 2*c_rs*G_rs)[0:len(r_mm)]


res = scipy.optimize.minimize(qmmm_opt_EandG, np.ravel(r_mm0), jac=True, method='L-BFGS-B', options=dict(disp=True))



theo_coords = XYZ(theo_ch2o.r, select_atoms(theo_ch2o, pdb='* * ~C,N,CA'))
res, r_theo = moloptim(qmmm2.EandG, r0=theo_ch2o.r, coords=theo_coords)







## Dimer, Lanczos
from chem.opt.dimer import *

# highest energy state from RC scan
E_rcts, r_rcts, dr_rcts = path_ts(RCe, RCr)

## Dimer method
if 0:
  dimer = DimerMethod(EandG=qmmm.EandG, R0=r_rcts, mode=dr_rcts)
  #r_dimer = dimer.search()
  #E_dimer = qmmm.EandG(r_dimer, dograd=False)
  DMres = gradoptim(dimer.EandG, r_rcts, ftol=0, maxiter=100)
  r_dimer, E_dimer = getvalues(DMres, 'x', 'fun')

## Lanczos method
lanczos = LanczosMethod(EandG=qmmm.EandG, tau=dr_rcts)
LCZres = gradoptim(lanczos.EandG, r_rcts, ftol=0, maxiter=100)  #stepper=lbfgs.LBFGS(H0=1./700),
r_lcz, E_lcz = getvalues(LCZres, 'x', 'fun')

# looks like electron density around moving hydrogen is depleted ... so negative charge better?
#write_hdf5('hcoh_ts.h5', r_lcz=r_lcz, E_lcz=E_lcz)
#cs, RCr, RCe = read_hdf5(ts_test1_h5, 'rcs', 'RCr', 'RCe')


## try to lower TS energy

## position fixed, charge varying

r_ch2o = align_atoms(r_ch2o, r_hcoh, sel=[0,1,2])
r_ts = align_atoms(r_lcz, r_hcoh, sel=[0,1,2])  #r_lcz
#bq_coords = bq_grid_init(ch2o, r_hcoh, r_ts, r_ch2o, maxbq=8)
bq_coords = bq_grid_init(ch2o, r_ch2o, r_ts, r_hcoh, maxbq=8)
charges = [ Bunch(r=rbq, qmq=0) for rbq in bq_coords ]

def charge_qopt_EandG(q):
  for ii, c in enumerate(charges):
    c.qmq = q[ii]
  # consider changing pyscf_EandG to accept charges and positions separately
  E_ts, Gr_ts, mf_ts = pyscf_EandG(ch2o, r_ts, charges=charges)
  Gbq_ts = pyscf_bq_qgrad(mf_ts.base.mol, mf_ts.base.make_rdm1(), bq_coords)

  # we don't want energy of product state (E_ps) to be lowered (relative to solution) ... perhaps we'll need a
  #  non-linear scaling fn to more strongly prevent decrease of E_ps
  E_ps, Gr_ps, mf_ps = pyscf_EandG(ch2o, r_ch2o, charges=charges)
  Gbq_ps = pyscf_bq_qgrad(mf_ps.base.mol, mf_ps.base.make_rdm1(), bq_coords)
  # also don't want energy of reactant state lowered
  E_rs, Gr_rs, mf_rs = pyscf_EandG(ch2o, r_hcoh, charges=charges)
  Gbq_rs = pyscf_bq_qgrad(mf_rs.base.mol, mf_rs.base.make_rdm1(), bq_coords)

  # allow, but do not drive increase of E_ps, E_rs; heavily penalize decrease
  dE_ts = E_ts - E_lcz
  dE_ps = E_ps - E_ch2o
  dE_rs = E_rs - E_hcoh
  # fixed values don't work (optim. fails), but this works well:
  c_ps = -dE_ps/0.001 if dE_ps < 0 else 0  # 0.001 Hartree ~ kB*T
  c_rs = -dE_rs/0.001 if dE_rs < 0 else 0

  print("dE_rs: %f; dE_ts: %f; dE_ps: %f" % (dE_rs, dE_ts, dE_ps))

  # d/dq(dE*dE/0.001) = 2*(dE/0.001)*d(dE)/dq
  return dE_ts - c_ps*dE_ps - c_rs*dE_rs, Gbq_ts - 2*c_ps*Gbq_ps - 2*c_rs*Gbq_rs


res = scipy.optimize.minimize(charge_qopt_EandG, np.zeros(len(charges)), bounds=[(-0.8, 0.5)]*len(charges), jac=True, method='L-BFGS-B', options=dict(disp=True))

append_hdf5(ts_test1_h5, r_ts=r_ts, bq_coords=bq_coords, bq_qs=res.x)

# To get <> to work: Ctrl+Shift+1 to unfocus Mol 1, Ctrl+Shift+2 to focus Mol 2
mqq = Molecule(atoms=[Atom(name='Q', r=q.r, mmq=q.qmq) for q in charges])
vis = Chemvis([Mol(ch2o, [r_hcoh, r_ts, r_ch2o], [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()

# promising result
# dE_rs: -0.000262; dE_ts: -0.020998; dE_ps: -0.000199
# charges = [{'qmq': -0.16712447536992234, 'r': array([-2.13970992,  1.85015931,  2.65632328])}, {'qmq': 0.5, 'r': array([-1.03879899, -0.37551714, -2.82175363])}, {'qmq': 0.5, 'r': array([ 0.06211195,  0.73732108, -2.82175363])}, {'qmq': 0.5, 'r': array([-1.58925445,  0.18090197,  3.20413097])}, {'qmq': -0.5, 'r': array([-3.24062085,  0.73732108,  1.5607079 ])}, {'qmq': -0.25777496846657511, 'r': array([-1.58925445,  1.29374019, -2.82175363])}, {'qmq': 0.5, 'r': array([-0.48834352,  2.40657842,  2.65632328])}, {'qmq': -0.5, 'r': array([-1.58925445,  2.96299753,  1.5607079 ])}]

def ch2o_bq_EandG(r):
  E, G, mf = pyscf_EandG(ch2o, np.vstack((r_ts[0], r)), charges=charges)
  return E, G[1:ch2o.natoms]


lanczos = LanczosMethod(EandG=ch2o_bq_EandG, tau=dr_rcts[1:])
LCZres = gradoptim(lanczos.EandG, r_ts[1:], ftol=0, maxiter=100)
r_ts_bq = LCZres.x

# need to constrain mol for TS refinement w/ charges (other atoms forming enzyme pocket would provide this fn
#  in full simulation)
# - RC scan constraining position of one atom
# - NEB or Lanczos omitting coords of atom to keep fixed ... this seems to work, but how to constrain rotation too?  We could constrain a 2nd atoms in two directions, but rotation about the resulting axis would still be possible


## moving charges

r_ts = r_lcz
charges = [ Bunch(r=None, qmq=(0.1 if ii%2 else -0.1)) for ii in range(10) ]

# we do not want to include charge-charge interaction - and pyscf doesn't!  Once we flesh this out more, we
#  can integrate into QMMM (which can remove charge-charge interaction for other QM codes)
def charge_ropt_EandG(r):
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

moloptim(qmol, r0=hcoh.r, fn=charge_ropt_EandG, coords=coords, bounds=bounds, ftol=2E-07)

#res = gradoptim(charge_opt_EandG, r_charges)
res = scipy.optimize.minimize(charge_ropt_EandG, np.ravel(r_charges), jac=True, method='L-BFGS-B', options=dict(disp=True))


# How to keep charges from getting too close to atoms?
# - optimize charge values on fixed grid ... we'd need grad wrt charge, not position!
# - surround mol with charges confined to boxes, each with zero net charge
# - reduandant internals w/ TC, use bounds w/ lbfgsb ... WIP
# - Lagrange multipliers? would need separate one for each charge?
# - add restraining potential (effectively the same as MM force field constraining position of MM charges)
# - keep outside a sphere: use spherical coords and set lower bound on r
#  - could we use ellipse instead?
# At least this gives us some info: negative charge near the moving H atom should lower energy

# How to place fixed charges around substrate?
# - place on surface some fixed distance from mol?
# - project from 2D grid to this surface ... by projecting line through molecule and choosing two points at some distance from mol
# - 3D grid, keeping only points w/in a certain range of distances to mol
#  - dense grid, keeping only N points w/ largest grad (min dist from mol and other points)
#  - first sort points based on distance to nearest atom, discard points too close, start calculating w/ nearest remaining point
# - randomly choose points in box around molecule, accepting only points some min dist from mol and other points until we have enough points



## NEB
from chem.opt.neb import NEB

NEBe = RCe[:]
NEBr = RCr[:]
r_ch2o_align = align_atoms(r_ch2o, r_hcoh)
neb = NEB(R=r_hcoh, P=r_ch2o_align, EandG=qmmm.EandG, nimages=len(NEBr), k=2.0, climb=False)
NEBres = gradoptim(neb.EandG, neb.active(), ftol=0)  #gtol=1e-4 ... we can relax gtol unless we need exact path
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
vis = Chemvis(Mol(ch2o, [ VisGeom(style='lines'), VisVol(pyscf_esp_vol, vis_type='volume', vol_type="ESP", timing=True) ]), bg_color=Color.white).run()


# test our optimizers
res, _ = moloptim(ch2o, fn=qmmm.EandG, optimizer=gradoptim, ftol=2E-07)
r_ch2o_2, E_ch2o_2 = res.x, res.fun

res, _ = moloptim(ch2o, r0=hcoh.r, fn=qmmm.EandG, optimizer=gradoptim, ftol=2E-07)
r_hcoh_2, E_hcoh_2 = res.x, res.fun


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


# Reundant() doesn't seem to work with cartesians - we could try to create single Redundant obj for entire mol
fznxyz = flatten([[3*ii, 3*ii+1, 3*ii+2] for ii in fzn])
bonds, angles, diheds = theo_ch2o.get_internals(active=active, inclM1=True)
redun = Redundant(theo_ch2o, xyzs=fznxyz, bonds=bonds, angles=angles, diheds=diheds, autobonds='none', autoangles='none', autodiheds='none')
dihedstart = len(bonds) + len(angles)
dihedidx = range(dihedstart, dihedstart + len(diheds))

#redun.constrain([[ii] for ii in fzn])
#redun.constrain(bonds)
#redun.constrain(angles)

redunactive = Active(redun, dihedidx)
res, r_theo2 = moloptim(tinker_EandG, mol=theo_ch2o, coords=redunactive, fnargs=dict(key=tinker_qmmm_key, charges=zero_q))

if 0:
  hdlc = HDLC(theo_ch2o, autodlc=1, dlcoptions=dict(recalc=1))
  hdlc.constrain([[ii] for ii in qmatoms])  # frozen atoms
  # freeze all internals - only translation and rotation possible
  for dlc in hdlc.dlcs:
    for ii in range(len(dlc.bonds) + len(dlc.angles) + len(dlc.diheds)):
      dlc.constraint(setitem(np.zeros(dlc.nQ), ii, 1.0))

  #hdlc = HDLC(theo_ch2o, autodlc=1, dlcoptions=dict(recalc=1))
  #hdlc.constrain([[ii] for ii in qmatoms])  # frozen atoms
  #hdlc.constrain([[ii] for ii in theo_ch2o.select(pdb='GLN *')])
  ## freeze all internals - only translation and rotation possible
  #for dlc in hdlc.dlcs:
  #  for ii in range(len(dlc.bonds) + len(dlc.angles) + len(dlc.diheds)):
  #    dlc.constraint(setitem(np.zeros(dlc.nQ), ii, 1.0))
  #res, r_theo4 = moloptim(growlj_EandG, mol=theo_ch2o, r0=r_theo3, coords=hdlc, vis=vis, raiseonfail=False, gtol=5E-4)


# place non-polar residues to help hold substrate in position
# Manual placement - Ctrl+drag to move
#theo_ch2o.append_atoms(place_residue('LEU'))
#vis.refresh()  #vis.select('resnum == 6') ... just Alt+click to select whole molecule
#tinker_EandG(theo_ch2o, key=tinker_qmmm_key, grad=False)[0]
#...
#write_tinker_xyz(theo_ch2o, 'theo1.xyz')
#write_pdb(theo_ch2o, 'theo1.pdb')

# Improving manual placement: continuous or on-demand calc of total MM energy ... actually doesn't seem to be a major issue as long as we avoid collisions (w/in vdW radii)

# - alternative would be to use other means instead of MM to prevent collision? our own hard-sphere interaction?
# next step would be to figure out backbone positions and other residues to hold catres in desired poise
# - holding desired poise probably impossible w/o additional residues

# Other things to explore: mutate (i.e. try different catres)


if 0:  #os.path.exists('theo4.xyz'):
  r_theo4 = load_molecule('theo4.xyz').r
else:
  # no intra-residue force
  #maskidx = [(a, b) for res in mol.residues for ii,a in enumerate(res.atoms) for b in res.atoms[ii+1:]]

  # repulsion only for backbone atoms
  growlj_EandG = Growlj_EandG(theo_ch2o, repel=[(ii[0], jj) for ii in zero_q for jj in theo_ch2o.listatoms()])

  vis = Chemvis([Mol(theo_ch2o, [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()

  # Rigid() works better than freezing internals with DLC()
  rgds = HDLC(theo_ch2o, [XYZ(r_theo3, active=[], atoms=res.atoms) if res.name != 'LEU' else Rigid(r_theo3, res.atoms) for res in theo_ch2o.residues])
  res, r_theo4 = moloptim(growlj_EandG, mol=theo_ch2o, r0=r_theo3, coords=rgds, vis=vis, ftol=2E-07)
  #write_tinker_xyz(theo_ch2o, 'theo4.xyz', r=r_theo4)
