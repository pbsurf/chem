# vinyl alcohol (CH2CHOH) -> acetaldehyde (CH3CHO)

import os
import scipy.optimize
from chem.molecule import *
from chem.io import *
from chem.qmmm.qmmm1 import *
from chem.opt.dlc import *
from chem.opt.lbfgs import *
from chem.opt.optimize import *
from pyscf import scf
from chem.vis.chemvis import *
from chem.test.common import *
from chem.theo import *
from chem.opt.dimer import *
from chem.opt.neb import *
from chem.mm import MolLJ, NCMM


# Refs:
# - en.wikipedia.org/wiki/Vinyl_alcohol : dH = -42.7 kJ/mol = -10.2 kcal/mol (experimental)
# - sites.psu.edu/dftap/2018/04/27/calculating-the-barrier-of-vinyl-alcohol-tautomerization-to-acetaldehyde/
#  - -11.2 kcal/mol between RS and PS; 51.9 kcal/mol barrier (DFT)
# - unige.ch/sciences/chifi/wesolowski/public_html/wyklad_compchem_2016/exercises/161118.pdf (10.1002/anie.201003530) - double hydrogen shift
#  - 56.6 kcal/mol barrier; w/ CH2CHOH == 0, CH2CHOH + HCOOH: -10.4, TS: +5.6, CH3CHO + HCOOH: -20.3, CH3CHO: -9.5 (high-level post-HF w/ DFT geometries)
# - 10.1002/(SICI)1097-461X(1998)66:1<9::AID-QUA2>3.0.CO;2-Z - CH2CHOH/CH3CHO and CH2CHNH2/CH3CHNH
# - aanda.org/articles/aa/pdf/2020/03/aa37302-19.pdf (10.1051/0004-6361/201937302) - refs

# these are both MM3 optimized
ch3cho_xyz = ("7  CH3CHO w/ Amber96 Tinker types, GAFF names  \n"   # MM3 types:
" 1  c     0.216434   -0.239836    0.022589    342  2  3  7   \n"   # 3
" 2  c3    0.041247   -0.042250    1.496848    340  1  4  5  6\n"   # 1
" 3  o     1.273466   -0.143112   -0.555929    343  1         \n"   # 7
" 4  hc   -0.670708    0.790679    1.698814    341  2         \n"   # 5
" 5  hc   -0.360705   -0.966829    1.971194    341  2         \n"   # 5
" 6  hc    1.009413    0.204927    1.989829    341  2         \n"   # 5
" 7  ha   -0.702574   -0.492745   -0.564006    341  1         \n")  # 5

ch2choh_xyz = ("7  CH2CHOH w/ Amber96 Tinker types, GAFF names\n"    # MM3 types:
" 1  c2    0.231311   -0.117468    0.202122    342  2  3  7   \n"    #  2
" 2  c2   -0.336100   -0.188389    1.410112    340  1  4  5   \n"    #  2
" 3  oh    1.465427   -0.622004   -0.056621    343  1  6      \n"    #  6
" 4  ha   -1.343088    0.222649    1.588799    341  2         \n"    #  5
" 5  ha    0.192087   -0.661201    2.253726    341  2         \n"    #  5
" 6  ho    1.994120    0.037261   -0.538351    341  3         \n"    # 73
" 7  ha   -0.308197    0.346592   -0.641855    341  1         \n")   #  5

tinker_key = make_tinker_key('amber96')


ch3cho = load_molecule(ch3cho_xyz)  #Molecule().append_atoms(load_molecule(ch3cho_xyz), residue='LIG')
ch2choh = load_molecule(ch2choh_xyz)

qmbasis = '6-31G*' #'6-31++G**'
ligatoms = init_qmatoms(ch3cho, '*', qmbasis)
qmatoms = ligatoms[:]

qmmm = QMMM(mol=ch3cho, qmatoms=qmatoms, prefix='no_logs')


## Global optim to find RS, PS
vinyl0_h5 = 'vinyl0.h5'
if os.path.exists(vinyl0_h5):
  ch2choh.r, ch3cho.r = read_hdf5(vinyl0_h5, 'r_rs', 'r_ps')
else:
  dlc = DLC(ch2choh, recalc=1)
  for ii in range(len(dlc.bonds) + len(dlc.angles)):
    dlc.constraint(setitem(np.zeros(dlc.nQ), ii, 1.0))

  # check that full range of dihedrals is being covered
  #mcoptim = MCoptim()
  #Es, Rs = mcoptim(lambda r: 0.0, ch2choh.r, coords=dlc, adjstep=-1, maxiter=100)
  #_, _, diheds = ch2choh.get_internals()
  #tors = [ [calc_dihedral(r[list(dihed)]) for r in Rs] for dihed in diheds ]
  #hist, bins = np.histogram(np.ravel(tors), bins=360/5)
  #plot(bins[:-1], hist)

  mcoptim = MCoptim()
  Es, Rs = mcoptim(qmmm.EandG, ch2choh.r, coords=dlc, hop=True, hopcoords=XYZ(ch2choh.r), maxiter=40)
  #Es, Rs = mcoptim.results()  # if aborted
  #Raligned = [ align_atoms(r, Rs[0], sel=[0,1,2]) for r in Rs]
  #vis = Chemvis(Mol(ch2choh, Raligned, [ VisGeom(style='lines') ]), fog=True, wrap=False).run()
  ch2choh.r = Rs[0]

  dlc = DLC(ch3cho, recalc=1)
  for ii in range(len(dlc.bonds) + len(dlc.angles)):
    dlc.constraint(setitem(np.zeros(dlc.nQ), ii, 1.0))

  mcoptim = MCoptim()
  Es, Rs = mcoptim(qmmm.EandG, ch3cho.r, coords=dlc, hop=True, hopcoords=XYZ(ch3cho.r), maxiter=40)
  #Es, Rs = mcoptim.results()  # if aborted
  # issue here is alignment of the moving atom
  Raligned = [ align_atoms(r, ch2choh.r, sel=[0,1,2]) for r in Rs]
  dh6 = [calc_dist(r[5], ch2choh.atoms[5].r) for r in Raligned]
  ch3cho.r = Rs[argsort(dh6[:6])[0]]  #Rs[0]

  write_hdf5(vinyl0_h5, r_rs=ch2choh.r, r_ps=ch3cho.r)


## RC scan
nimages = 10
c2, h6, o3 = 2-1, 6-1, 3-1
c2_h6 = [c2, h6]  # bonds - must be list, not tuple for indexing
o3_h6 = [o3, h6]

vinyl1_h5 = 'vinyl1.h5'
if os.path.exists(vinyl1_h5):
  rcs, RCr, RCe = read_hdf5(vinyl1_h5, 'rcs', 'RCr', 'RCe')
  r_rs0, r_ps0 = RCr[0], RCr[-1]
  E_rs0, E_ps0 = RCe[0], RCe[-1]
else:
  RCr = [0]*nimages
  RCe = [0]*nimages

  # note that MP2 only changes E_ps - E_rs by 0.00003 Hartree
  res, r_rs0 = moloptim(qmmm.EandG, mol=ch3cho, r0=ch2choh.r)  #, ftol=2E-07)
  RCr[0], RCe[0] = r_rs0, res.fun

  res, r_ps0 = moloptim(qmmm.EandG, mol=ch3cho)  #, ftol=2E-07)
  r_ps0 = align_atoms(r_ps0, r_rs0, sel=[0,1,2])  # align heavy atoms
  RCr[-1], RCe[-1] = r_ps0, qmmm.EandG(r_ps0, dograd=False)  #res.fun ... slight numerical differences risk confusing me

  # incidentally, these coords also give the fewest iters for optimizing RS, PS geom
  dlc = DLC(ch3cho, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
  dlc.constraint(dlc.bond(o3_h6) - dlc.bond(c2_h6))

  rc0 = calc_dist(RCr[0][o3_h6]) - calc_dist(RCr[0][c2_h6])
  rc1 = calc_dist(RCr[-1][o3_h6]) - calc_dist(RCr[-1][c2_h6])
  rcs = np.linspace(rc0, rc1, nimages)
  ch3cho.r = RCr[0]

  # Alternative approach: start from interpolated geometry instead of previous geometry
  #images = dlc_interp(RCr[0], RCr[-1], nimages=10, dlc=DLC(ch2choh, recalc=1), align_sel=[0,1,2])
  #...res, r = moloptim(qmmm.EandG, mol=ch3cho, r0=images[ii], coords=dlc, ftol=2E-07)

  for ii in range(1, nimages-1):
    # reverse bonds and use move_frag=False to keep H6 fixed
    ch3cho.bond(o3_h6[::-1], 0.5*(rcs[ii] - rcs[ii-1]), rel=True, move_frag=False)
    ch3cho.bond(c2_h6[::-1], -0.5*(rcs[ii] - rcs[ii-1]), rel=True, move_frag=False)
    #ch2o.bond(c1_h4[::-1], ch2o.bond(o2_h4) - rc, move_frag=False)  # move C1
    #print("rc: %f; O2-H4 - C1-H4: %f" % (rcs[ii], ch3cho.bond(o3_h6) - ch3cho.bond(c2_h6)))
    res, r = moloptim(qmmm.EandG, mol=ch3cho, coords=dlc, ftol=2E-07)
    r = align_atoms(r, r_rs0, sel=[0,1,2])
    RCr[ii], RCe[ii] = r, res.fun
    ch3cho.r = r

  write_hdf5(vinyl1_h5, rcs=rcs, RCr=RCr, RCe=RCe)

  vis = Chemvis(Mol(ch3cho, RCr, [ VisGeom(style='licorice') ]), fog=True, wrap=False).run()

  plot(rcs, RCe, xlabel="O3-H6 - C2-H6 (Angstroms)", ylabel="energy (Hartrees)")


## thermochemistry
# TODO: check for very low frequency modes (and adjust?) - issue is that harmonic approx may not be valid
if 0:
  from pyscf.hessian import thermo
  _, _, scn = pyscf_EandG(ch3cho, r=r_ps0)
  mf = scn.base
  hess_ps = mf.Hessian().kernel()
  freq_ps = thermo.harmonic_analysis(mf.mol, hess_ps)
  thermo_ps = thermo.thermo(mf, freq_ps['freq_au'], 298.15, 101325)
  Gfe_ps = thermo_ps['G_tot'][0]

  _, _, scn = pyscf_EandG(ch3cho, r=r_rs0)
  mf = scn.base
  hess_rs = mf.Hessian().kernel()
  freq_rs = thermo.harmonic_analysis(mf.mol, hess_rs)
  thermo_rs = thermo.thermo(mf, freq_rs['freq_au'], 298.15, 101325)
  Gfe_rs = thermo_rs['G_tot'][0]

  with open('vinyl0_thermo.pickle',"wb") as f:
    pickle.dump(thermo_ps, f)
    pickle.dump(thermo_rs, f)
  #with open('vinyl0_thermo.pickle',"rb") as f:
  #  thermo_ps = pickle.load(f)
  #  thermo_rs = pickle.load(f)


## NEB
if os.path.exists('vinyl_neb.h5'):
  NEBr, NEBe = read_hdf5('vinyl_neb.h5', 'NEBr', 'NEBe')
else:
  NEBe = RCe[:]
  NEBr = RCr[:]
  neb = NEB(R=RCr[0], P=RCr[-1], EandG=qmmm.EandG, nimages=len(NEBr), k=2.0, climb=False)

  # this gives good results (rotation about CCOH dihedral)
  #neb.images = dlc_interp(RCr[0], RCr[-1], nimages=10, dlc=DLC(ch2choh, recalc=1), align_sel=[0,1,2])
  neb.images = RCr[:]

  NEBres = gradoptim(neb.EandG, neb.active(), ftol=0, gtol=0.1)  # best we can do for gtol
  NEBr[1:-1] = np.reshape(NEBres.x, (-1, ch3cho.natoms, 3))
  NEBe = [qmmm.EandG(xyz, dograd=False) for xyz in NEBr]
  E_nebts, r_nebts, dr_nebts = path_ts(NEBe, NEBr)
  #[calc_dist(r[o3_h6]) - calc_dist(r[c2_h6]) for r in NEBr]

  # - CI-NEB (initialized w/ output of normal NEB) ... seems to work well, TS close to Lanczos
  CINEBr, CINEBe = np.array(NEBr), [0]*nimages
  cineb = NEB(R=RCr[0], P=RCr[-1], EandG=qmmm.EandG, nimages=len(CINEBr), k=2.0, climb=True)
  cineb.images = np.array(NEBr)
  CINEBres = gradoptim(cineb.EandG, cineb.active(), ftol=0, gtol=0.025)
  CINEBr[1:-1] = np.reshape(CINEBres.x, (-1, ch3cho.natoms, 3))
  CINEBe = [qmmm.EandG(xyz, dograd=False) for xyz in CINEBr]

  write_hdf5('vinyl_neb.h5', NEBr=NEBr, NEBe=NEBe, CINEBr=CINEBr, CINEBe=CINEBe)


## Lanczos method to find TS
if os.path.exists('vinyl_lcz2.h5'):
  #r_lcz, E_lcz = read_hdf5('vinyl_lcz.h5', 'r_lcz', 'E_lcz')
  r_lcz, E_lcz, tau_lcz = read_hdf5('vinyl_lcz2.h5', 'r_lcz', 'E_lcz', 'tau_lcz')
else:
  # init w/ highest energy state from NEB ... highest energy state from RC scan doesn't seem to work
  E_rcts, r_rcts, dr_rcts = path_ts(NEBe, NEBr)
  lanczos = LanczosMethod(EandG=qmmm.EandG, tau=dr_rcts)
  LCZres = gradoptim(lanczos.EandG, r_rcts, ftol=0, maxiter=100)
  r_lcz0, E_lcz = getvalues(LCZres, 'x', 'fun')
  align = alignment_matrix(r_lcz0[[0,1,2]], r_rs0[[0,1,2]])
  r_lcz = apply_affine(align, r_lcz0)
  tau_lcz = apply_affine(align, lanczos.tau)
  #r_lcz = align_atoms(r_lcz0, r_rs0, sel=[0,1,2])
  write_hdf5('vinyl_lcz2.h5', r_lcz=r_lcz, E_lcz=E_lcz, tau_lcz=tau_lcz)

r_ts0, E_ts0 = r_lcz, E_lcz

# getting TS from RC path - maxdr needed to prevent bad steps early in opt; in general, limiting step size
#  for TS search seems like a good idea in general since initial geometry should be pretty close
# path_ts() for RC gives RCr[4], in which CH2 has not started rotating, so Lanczos fails to find TS
#LCZres = gradoptim(monitor(lanczos.EandG), np.array(RCr[5]), ftol=0, maxiter=100, maxdr=0.05)


## IRC
# Observations: IRC path matches NEB path fairly well - both have gradual change from CH2 to CH3 geometry
#  In contrast, RC scan has sharp jump between CH2 and CH3 geometry; plotted vs. RC used for RC scan, RC scan
#  energy is below IRC and NEB, as expected.  Fine 10 point RC scan across jump still only has 2 intermediate
#  states, and neither scan direction nor initial states for optim. make a significant difference
# So which path is more correct? gradual CH2 -> CH3 or jump?  Does it matter?
# We mainly need TS, and NEB seems to give better starting point for Lanczos than RC scan; IRC can serve as
#  sanity check on NEB path
if os.path.exists('vinyl_irc.h5'):
  rirc, Eirc = read_hdf5('vinyl_irc.h5', 'rirc', 'Eirc')
else:
  weight = 1.0/np.array([ELEMENTS[z].mass for z in ch3cho.znuc])  # 1.0
  dr = normalize(tau_lcz)
  Efwd, ircfwd = irc_integrate(qmmm.EandG, r_lcz, dr, weight)
  Erev, ircrev = irc_integrate(qmmm.EandG, r_lcz, -dr, weight)

  Eirc = Erev[::-1] + [E_lcz] + Efwd
  rirc = ircrev[::-1] + [r_lcz] + ircfwd

  write_hdf5('vinyl_irc.h5', rirc=rirc, Eirc=Eirc)

  t_irc = [RMSD_rc(r, r_rs0, r_ps0) for r in rirc]
  rc_irc = [calc_dist(r[o3_h6]) - calc_dist(r[c2_h6]) for r in rirc]

  t_rc = [RMSD_rc(r, r_rs0, r_ps0) for r in RCr]
  rc_rc = [calc_dist(r[o3_h6]) - calc_dist(r[c2_h6]) for r in RCr]

  t_neb = [RMSD_rc(r, r_rs0, r_ps0) for r in NEBr]
  rc_neb = [calc_dist(r[o3_h6]) - calc_dist(r[c2_h6]) for r in NEBr]

  # better ways to compare reaction paths?
  plot(t_irc, Eirc, t_rc, RCe, t_neb, NEBe, xlabel="RMSD RC", legend=["IRC", "RC", "NEB"])
  plot(rc_irc, Eirc, rc_rc, RCe, rc_neb, NEBe, xlabel="O3-H6 - C2-H6 (Ang)", legend=["IRC", "RC", "NEB"])


# start interactive
import pdb; pdb.set_trace()


## Concerted reaction with GLH
# HE2 moves from O3 to OE1 and H moves from OE2 to C2
if os.path.exists('cov1.xyz'):
  cov1 = load_molecule('cov1.xyz')
else:
  r_o3, r_h6 = r_rs0[o3], r_rs0[h6]
  cov1 = Molecule(header="CH2CHOH - CH3CHO theozyme 2; amber96.prm")
  cov1.append_atoms(ch3cho, r=r_ps0, residue='LIG')
  cov1.append_atoms(place_residue('GLU', [11], [r_h6 + 2.5*(r_h6 - r_o3)], r_o3))  # GLU 11,12: OE1,OE2

  ah6 = ch2choh.atoms[h6]
  # this is not the correct MM charge for HE2!
  he2 = cov1.append_atoms([Atom('HE2', 1, ah6.r, ah6.mmtype, ah6.mmq, qmbasis=qmbasis)], residue=1)[0]
  #cov1.set_bonds([(he2, c2)], replace=False)  # bond to LIG for MM
  oe1 = cov1.select('* GLU OE1')[0]
  cov1.bond([oe1, he2], 1.0, move_frag=False)
  write_xyz(cov1, 'cov1.xyz')

oe1 = cov1.select('* GLU OE1')[0]
oe2 = cov1.select('* GLU OE2')[0]
he2 = cov1.select('* GLU HE2')[0]

qmatoms = init_qmatoms(cov1, '* LIG *; * GLU OE1,OE2,HE2,CD', qmbasis)
# net charge of the zero_q atoms is -0.125
zero_q = [ (ii, 0) for ii in cov1.select(pdb='~LIG N,C,O,H1,H2,H3,OXT') ]
# set newcharges=False since we don't have correct MM charges for QM region (HE2 in particular)
qmmm = QMMM(cov1, qmatoms=qmatoms, prefix='cov1', mm_key=tinker_key,
    mm_opts=Bunch(charges=zero_q, noconnect=ligatoms, newcharges=False),
    capopts=Bunch(basis=qmbasis, placement='rel', g_CH=0.714),
    chargeopts=Bunch(scaleM1=1.0, adjust='dipole', dipolecenter=0.75, dipolesep=0.5))

if os.path.exists('cov1.h5'):
  r_rs1, r_ps1 = read_hdf5('cov1.h5', 'r_rs1', 'r_ps1')
else:
  xyz_cov1 = XYZ(cov1, active=ligatoms + [oe1, oe2, he2])
  res_ps, r_ps1 = moloptim(qmmm.EandG, mol=cov1, coords=xyz_cov1, ftol=2E-07, vis=vis)
  # Now RS
  cov1.r = r_ps1
  cov1.bond([o3, he2], 0.95, move_frag=False)
  cov1.bond([oe2, h6], 0.95, move_frag=False)
  # DLC doesn't seem to do much better than XYZ here
  res_rs, r_rs1 = moloptim(qmmm.EandG, mol=cov1, coords=xyz_cov1, ftol=2E-07, vis=vis)
  write_hdf5('cov1.h5', r_rs1=r_rs1, r_ps1=r_ps1)


## NEXT
# - explore more and document overall procedure for exploring different RS,PS, and reaction paths
#  - look into issues w/ Lanczos

# using absolute tol for optimization, e.g., 0.1*kB*T: ftol = 0.1*0.001/E0; for E0 ~ 1E2 (QM), ftol ~ 1E-6

# - enthalpy vs. energy?  Assuming no signficant difference ... why doesn't our energy difference between RS
#  and PS match literature?
# - (now or later) look at entropy ... \delta G is the actual physically relevant quantity

# - then, clean up this project and maybe work on displacement of water from active site or finite temp. stuff
#  - can we make some functions to dedup common blocks like RC scan and NEB?  Lanczos/NEB w/ coords?
# - bimolecular reactions?
# - point charges to further stabilize covalent path TS?

# - other residues and rotatmers for non-catres
# ... what quantity are we optimizing? Don't spend too much time, since there are other issues we can't address at this point anyway, such as influence of enzyme flexibility on access to active site from soln
#  - after initial relaxation w/ rigid residues, relax again allowing torsions to move?

# - trying different catres
#  - random placement (like non-catres)?  random mutation?

# Moved from theo.py: TODO: try averaging over RS, TS, PS


# hmmm ... literature says barrier should be more like 0.008 H - let's try RC scan\
# (maybe 0.008 H above separated RS, 0.025 H above minimized close proximity RS)

# with some manual proding of RS and PS, we get down to ~0.035 H
# - starting from TS of a less successful run, how could we have found this improved TS?
# - is, e.g., Lanczos, less sensitive to initial state?  Yes, from RC11r[5], Lanczos gets closer to improved state
# ... but not from RC10r[5]
# ... also unable to improve on RC1r TS

# How to improve on RC10r[5] ?
# - global opt?  How?  Lanczos instead of minimize for basin hopping? (Generate set of candidates, then run Lanczos on them?)  Use (high) temperature corresponding to barrier?
#  - seems like we should focus on the moving atoms (so only sample their positions?)
#  - rigid move, then move reacting atoms?
# - could we try moving uphill along lowest freq mode (isn't this just what Lanczos does?)
# - 2D RC scan, e.g., w/ one H atom constrained to midpoint, try a few positions for other ... seems like this should work, but if we could get random sampling to work that would be more general

# - given separated RS and PS, generate set of random orientations in proximity (just remove collisions w/ LJ or relax completely?), then do NEB?

# optimizing w/ TS vs. trying to optimize RS or PS first?

# RS: generate random orientation (only need to vary orientation of LIG), remove clashes w/ LJ, relax w/ QMMM

# I think we need to penalize distance from GLU; or use LJ for basin hopping? ... LJ for basin hopping mostly seems to pick configurations w/ the two mols stacked like sheets

# old note: scaling r0 and/or depth didn't work well - randomly placed residues can end up locked together

# what about potential to attract the H and O atoms of interest? use qq (w/ vdW to prevent overlap)
# ... this seems promising ... probably need to increase temperature
# - use clustering to dedup similar configurations - with previous global opt above, we were looking for just a single state but here we want to generate a set of states for further testing

# - possible improvements to sample generation?
#  - apply step to randomly chosen previous state instead of always using last state?
#  - use total connection, applying bounded step to redundant coords then calculating valid DLCs
#   - constrain bonded atoms (and M2?) ... how does this compare to using diheds explictly?
#   - project random (w/ atom-specific bounds?) redundant internals step dQ to valid space? ... see Redundant() class notes
# - poling (Leach 9.15): for basin hopping, add something like \sum_{prev minima i} (RMSD(r, r_i))^-N to energy, to force convergence to new state
# - aside: for large molecules, random dihedral search would be a good method

# what now?
# - full RC scan ... slow
#  - what about limited RC scan w/ moving atoms near midpoint
# - can we more quickly find TS? Lanczos on "average" of RS and PS? ... seems OK, let's see if first relaxing w/ some constraints is faster overall
#  - from r_avg, constrain one H atom to be equidistant, relax, then run Lanczos ... didn't seem to help (for h6 constrained, Lanczos looked to be converging to same state; for he2 fixed, relaxed state was quite low in energy and Lanczos wasn't working well)
#  - if we find a useful workflow involving Lanczos, look into getting down from 3 to 2 (or 1?) evals/iter


dr_avg = normalize(RC1r[-1] - RC1r[0])
r_avg = 0.5*(RC1r[-1] + RC1r[0])
#... write_hdf5('lcz_avg0.h5', E=LCZres.fun, r=LCZres.x)

# Given a RS and/or PS, how to sample different reaction paths?
# - Lanczos w/ different/random initial tau vectors? ... no, initial tau seems to have little effect
# - Lanczos w/ random perturbation of initial state? ... might be possible w/ right perturbation size - unclear
# - NEB w/ different initial path
#  - add random offsets to interpolated positions?
#  - for this specific case, what we'd want to try is one H atom moving, then the other ... OK, this works (one path worked better), but how to generalize/automate this?
#  - would randomizing the moving atom positions (along their axes of motion) work?  Seems like we just need to try the 2^n - 2 possible combinations of extreme positions (don't worry, n is small)

# reaction path sampling/transition path sampling:
# - transition path ensemble (Chipot Ch. 7): (using classical mechanics) given initial path between RS and PS, pick random point along path, perturb momenta (conserving total energy and momentum), propagate back to initial time and forward to final time (all paths assumed same duration(?)), accept if initial point, e.g., within some max RMSD of RS and final point within max RMSD of PS
#  - translation of this to, e.g., NEB, might involve perturbing images used to initialize NEB; minimize unnecessary motion in reaction path and focus perturbations on the atoms involved in reaction (in Cartesians? other coords?)
#  - might make sense to explore other reaction path methods besides NEB too
# - could we fit a MM potential for reaction coords to enable sampling paths w/o QM calc ... seems unlikely to work if the whole point is to find novel paths
#  - also, unlike, e.g, protein folding, the parameter space for even the most complex reactions is fairly limited (so not sure there is much point to trying to develop methods using MM for speed)
#  - in general, our ideal balance for this work is go as fast as possible while still being qualitatively correct everywhere (otherwise we could end up with useless methods)

# Lanczos is having difficulty w/ TS from cov1_neb3.h5 - curvature is very low - look at tau direction vs. initial guess and over course of inner loop iterations

#   - linear interp, then adj H atom positions?  can we do in one step so that interpolated positions are equidistant?
#   - adj H atom position, then adjust others to restore equidistance?


# create midpoint interpolated state, then shift h atom positions, then do two separate interpolations
# recall: HE2 moves from O3 to OE1 and H6 moves from OE2 to C2
r_mid = 0.5*(RC1r[0] + RC1r[-1])
cov1.r = r_mid
# option 1 - both H near LIG at midpoint
#cov1.bond([o3, he2], 0.95, move_frag=False)
#cov1.bond([c2, h6], 0.95, move_frag=False)
# option 2 - both H near GLU at midpoint ... this works better
cov1.bond([oe1, he2], 0.95, move_frag=False)
cov1.bond([oe2, h6], 0.95, move_frag=False)

neb = NEB(R=RC1r[0], P=RC1r[-1], EandG=qmmm.EandG, nimages=nimages, k=2.0, climb=False)
img1 = linear_interp(RC1r[0], cov1.r, 6)
img2 = linear_interp(cov1.r, RC1r[-1], 6)
neb.images = np.vstack((img1, img2[1:]))  #img1[:-1], [0.5*(img1[-1] + img2[0])], img2[1:]))
NEBres = gradoptim(neb.EandG, neb.active(), ftol=0, gtol=0.05)  # best we can do for gtol
NEBr = neb.images
NEBe = [qmmm.EandG(r, dograd=0) for r in NEBr]
write_hdf5('cov1_neb3.h5', NEBr=NEBr, NEBe=NEBe)

# cov1_rc2.h5 and cov1_neb3.h5 represent our current best (0.035, 0.03 H barrier)
# separated energies are: RS: -341.647; PS: -341.675



# How to make use of the states generated here?
# - relax to get RS, then shift moving atoms to PS conformation and relax to get PS, then try to find TS with NEB or Lanczos

cov1.r = r_rs1
cov1.set_bonds(guess_bonds(cov1))  # correct bonds needed for vdW (since bonded atoms and neighbors are excluded)
load_tinker_params(cov1, key=tinker_key, charges=False, vdw=True)

cov1.atoms[h6].mmq = 1.0
cov1.atoms[he2].mmq = 1.0
cov1.atoms[c2].mmq = -1.0
cov1.atoms[oe1].mmq = -1.0

rslig = [a for a in ligatoms if a != h6] + [he2]
#pslig = ligatoms
rgds = HDLC(cov1, [XYZ(cov1, active=[], atoms=cov1.listatoms(exclude=rslig)), Rigid(cov1, rslig, pivot=None)])


G0 = 1E-4*np.array([1,1,1, 0.2,0.2,0.2])  # bigger steps for rotation
mcoptim = MCoptim()
Es, Rs = mcoptim(NCMM(cov1), cov1.r, coords=rgds, G0=G0, hop=True, adjstep=10, maxiter=400) #, hopcoords=dlc, hopargs=dict(gtol=2E-04, ftol=5E-07, maxiter=20), G0=2E-5, adjstep=4, maxiter=40)


# hierarchical clustering
# - seems to work well; any reason to consider alterative approaches?
# - e.g., for each conformation, compare RMSD to accepted conformations and discard if close to one of them,
#  otherwise add to accepted conformations
# - density based clustering (DBSCAN, OPTICS) seems to be the current state-of-the-art; see scikit-learn.org
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# calc pairwise RMSD matrix
ligRs = Rs[:, rslig, :]
dRs = ligRs[:,None,:,:] - ligRs[None,:,:,:]
Ds = np.sqrt(np.einsum('iljk->il', dRs**2)/dRs.shape[-2])
# perform clustering; squareform converts between square distance matrix and condensed distance matrix
Z = linkage(squareform(Ds), method='single')
clust = fcluster(Z, 0.9, criterion='distance')  # also possible to specify number of clusters instead
# pick lowest energy geometry as representative of each cluster
MCr, MCe = [], []
for ii in np.unique(clust):
  sel = clust == ii
  MCe.append(np.amin(Es[sel]))
  MCr.append(Rs[sel][np.argmin(Es[sel])])
esort = np.argsort(MCe)
MCe = [MCe[ii] for ii in esort]
MCr = [MCr[ii] for ii in esort]





rc10s, RC10r, RC10e = read_hdf5('cov1_rc0.h5', 'rcs', 'RCr', 'RCe')
rc11s, RC11r, RC11e = read_hdf5('cov1_rc1.h5', 'rcs', 'RCr', 'RCe')
rc1s, RC1r, RC1e = read_hdf5('cov1_rc2.h5', 'rcs', 'RCr', 'RCe')

E_rcts, r_rcts, dr_rcts = path_ts(RC10e[1:], RC10r[1:])
lanczos = LanczosMethod(EandG=qmmm.EandG, tau=dr_rcts)
LCZres = gradoptim(lanczos.EandG, r_rcts, ftol=0, maxiter=100)
r_lcz0, E_lcz = getvalues(LCZres, 'x', 'fun')


xyz_qm = XYZ(r, active=qmatoms, ravel=False)
lanczos = LanczosMethod(EandG=coordwrap(qmmm.EandG, xyz_qm), tau=dr_avg[qmatoms])
LCZres = gradoptim(lanczos.EandG, r[qmatoms], ftol=0, maxiter=100)
r_lcz0, E_lcz = setitem(np.array(r), qmatoms, LCZres.x), LCZres.fun



# - seems there are multiple issues: overall orientation of molecules, as well as details of moving atoms

# - if we can't get closer to 0.01 H, check MM energies, try MP2 energies or better, try w/o MM

# - also: run RC backwards; try HE2 RC

cov1rc_h5 = 'cov1_rc1.h5'
if os.path.exists(cov1rc_h5):
  rc1s, RC1r, RC1e = read_hdf5(cov1rc_h5, 'rcs', 'RCr', 'RCe')
else:
  pass
## I think we can at least make a fn for RC scan for linearly moving atom

RC1r = [0]*nimages
RC1e = [0]*nimages
RC1r[0], RC1r[-1] = r_rs2, r_ps2

rc0 = calc_dist(RC1r[0][[h6, oe2]]) - calc_dist(RC1r[0][[h6, c2]])
rc1 = calc_dist(RC1r[-1][[h6, oe2]]) - calc_dist(RC1r[-1][[h6, c2]])
rc1s = np.linspace(rc0, rc1, nimages)

#rc0 = calc_dist(RC1r[0][[he2, o3]]) - calc_dist(RC1r[0][[he2, oe1]])
#rc1 = calc_dist(RC1r[-1][[he2, o3]]) - calc_dist(RC1r[-1][[he2, oe1]])
#rc1s = np.linspace(rc0, rc1, nimages)

cov1.r = RC1r[0]
dlc = DLC(cov1, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constrain([[ii] for ii in cov1.listatoms(exclude=qmatoms)])
dlc.constraint(dlc.bond([h6, oe2]) - dlc.bond([h6, c2]))
#dlc.constraint(dlc.bond([he2, o3]) - dlc.bond([he2, oe1]))

for ii in range(1, nimages):
  r_c2, r_oe2 = cov1.atoms[c2].r, cov1.atoms[oe2].r
  a = rc1s[ii]/calc_dist(r_c2, r_oe2)
  cov1.atoms[h6].r = 0.5*(1+a)*r_c2 + 0.5*(1-a)*r_oe2
  print("rc: expected: %f; actual: %f" % (rc1s[ii], cov1.bond([h6, oe2]) - cov1.bond([h6, c2])))
  res, r = moloptim(qmmm.EandG, mol=cov1, coords=dlc, ftol=5E-07, vis=vis)
  RC1r[ii], RC1e[ii] = r, res.fun
  cov1.r = r

write_hdf5(cov1rc_h5, rcs=rc1s, RCr=RC1r, RCe=RC1e)


res, r = moloptim(qmmm.EandG, mol=cov1, ftol=2E-07, vis=vis)


if os.path.exists('cov1_neb.h5'):
  NEBr, NEBe = read_hdf5('cov1_neb.h5', 'NEBr', 'NEBe')
else:
  NEBr = [0]*nimages
  neb = NEB(R=r_rs1, P=r_ps1, EandG=qmmm.EandG, nimages=len(NEBr), k=2.0, climb=False)
  NEBr[0], NEBr[-1] = r_rs1, r_ps1

  NEBres = gradoptim(neb.EandG, neb.active(), ftol=0, gtol=0.05)  # best we can do for gtol
  NEBr[1:-1] = np.reshape(NEBres.x, (len(NEBr)-2, -1, 3))
  NEBe = [qmmm.EandG(xyz, dograd=False) for xyz in NEBr]
  E_nebts, r_nebts, dr_nebts = path_ts(NEBe, NEBr)
  #[calc_dist(r[o3_h6]) - calc_dist(r[c2_h6]) for r in NEBr]
  write_hdf5('cov1_neb.h5', NEBr=NEBr, NEBe=NEBe)




# finding candidates for reaction coords
pairs = totalconn([oe1, oe2, he2, h6, c2, o3])
dists = [[calc_dist(r[b]) for r in NEBr] for b in pairs]
sel = [ii for ii,d in enumerate(dists) if d[0] < 2.1 or d[-1] < 2.1]
labels = ["%r" % pairs[ii] for ii in sel]
fdists = np.array([dists[ii] for ii in sel])
fig = plot(range(nimages), fdists.T, legend=labels, picker=True)
#from __future__ import print_function
#fig.canvas.mpl_connect('pick_event', lambda e: print(e.artist))


## uncatalyzed reaction in water
# fixed solvent (relaxed for RS): 0.13 H barrier (i.e., no change)
# - relaxing for TS produces no change in solvent at ftol=2E-08 ... this probably isn't a good model for reaction in soln
# - more realistically, reaction proceeds when thermal motion brings a water close enough to lower barrier significantly
#  - arguably this would be best modeled with just LIG and HOH, no solvent ... one issue is speed of reaction (does HOH stay in place long enough?)
#  - how quickly do we "roll" down hill after crossing barrier?
#  - speed corresponding to kinetic energy needed to cross barrier: v = sqrt(2*E_act/m) ... this is necessarily much greater than mean thermal speed, so relative to thermal motion, reaction should be essentially instantaneous
# - more thinking and reading along this line might help up ... barrier is a potential energy barrier, overcome when relevant part of molecule gains sufficient energy? or is contorted into geometry where barrier is comparable to thermal kinetic energy?

# also, I don't think fully minimizing gives a good model of water at STP (it throws away kinetic energy - T=0!)

from chem.qmmm.prepare import solvate
solvent = parse_xyz("smallbox_min.xyz")  #solvent.xyz_2
solvent.residues = [Residue(name='HOH', atoms=[ii, ii+1, ii+2], het=True) for ii in range(0, solvent.natoms, 3)]
for ii,a in enumerate(solvent.atoms):
  a.resnum = int(ii/3)

solvated = solvate(ch3cho, solvent, d=1.8)

tinker_pbc = tinker_key + ("\na-axis %.3f\nb-axis %.3f\nc-axis %.3f\n" % tuple(solvated.pbcbox))

qmmm = QMMM(solvated, qmatoms=ligatoms, prefix='solv0', mm_key=tinker_pbc, mm_opts = dict(noconnect=ligatoms))

res, r_srs0 = moloptim(qmmm.EandG, mol=solvated, r0=setitem(solvated.r, ligatoms, ch2choh.r))

dlc = DLC(solvated, atoms=ligatoms, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constraint(dlc.bond(o3_h6) - dlc.bond(c2_h6))

hdlc = HDLC(solvated, [XYZ(solvated, active=[], atoms=solvated.listatoms(exclude=ligatoms)), dlc])

RC1r = [0]*nimages
RC1e = [0]*nimages
RC1r[0] = r_srs0
solvated.r = RC1r[0]

# using rcs from standalone case above
for ii in range(1, nimages):
  # reverse bonds and use move_frag=False to keep H6 fixed
  solvated.bond(o3_h6[::-1], 0.5*(rcs[ii] - rcs[ii-1]), rel=True, move_frag=False)
  solvated.bond(c2_h6[::-1], -0.5*(rcs[ii] - rcs[ii-1]), rel=True, move_frag=False)
  print("rc: expected %f; actual: %f" % (rcs[ii], solvated.bond(o3_h6) - solvated.bond(c2_h6)))
  res, r = moloptim(qmmm.EandG, mol=solvated, coords=hdlc, ftol=2E-07)
  RC1r[ii], RC1e[ii] = r, res.fun
  solvated.r = r

write_hdf5('solv0_rc0.h5', rcs=rcs, RCr=RC1r, RCe=RC1e)


E_ts2, r_ts2, _ = path_ts(RC1e, RC1r)
xyz_solv = XYZ(solvated, active=solvated.listatoms(exclude=ligatoms))
res_ts, r_sts0 = moloptim(qmmm.EandG, mol=solvated, r0=r_ts2, coords=xyz_solv)



## GLH w/ intermediate HOH: 0.20 H barrier (!)
# - why?  w/ multiple H atoms moving in concert, TS has multiple H atoms in intermediate, high energy locations ... so a multistep process might have lower energy!  ... but w/ multi-step or better orientation, 0.08 H barrier

from chem.data.test_molecules import water_tip3p_xyz
h2o = load_molecule(water_tip3p_xyz)

if os.path.exists('cov2.xyz'):
  cov2 = load_molecule('cov2.xyz')
else:
  r_o3, r_h6 = r_rs0[o3], r_rs0[h6]
  cov2 = Molecule(header="CH2CHOH - CH3CHO theozyme 2; amber96.prm")
  cov2.append_atoms(ch3cho, r=r_ps0, residue='LIG')
  cov2.append_atoms(place_residue('GLU', [11], [r_h6 + 2.5*(r_h6 - r_o3)], r_o3))  # GLU 11,12: OE1,OE2
  # HE2 (but use empty mmconnect to avoid Tinker complaints)
  ah6 = ch2choh.atoms[h6]
  he2 = cov2.append_atoms([Atom('HE2', 1, ah6.r, ah6.mmtype, ah6.mmq, qmbasis=qmbasis)], residue=1)[0]
  oe1 = cov2.select('* GLU OE1')[0]
  cov2.bond([oe1, he2], 1.0, move_frag=False)
  # add HOH - manual placement in Chemvis seems like the quickest way here
  cov2.append_atoms(h2o, residue='HOH', r=[[-0.83, -1.56, 4.66], [-0.43, -2.39, 4.42], [-0.50, -0.93,  4.03]])
  write_xyz(cov2, 'cov2.xyz')

oe1 = cov2.select('* GLU OE1')[0]
oe2 = cov2.select('* GLU OE2')[0]
he2 = cov2.select('* GLU HE2')[0]
cd = cov2.select('* GLU CD')[0]
ow = cov2.select('* HOH O')[0]
hw1, hw2 = cov2.select('* HOH H')

qmatoms = init_qmatoms(cov2, '* LIG *; * GLU OE1,OE2,HE2,CD; * HOH *', qmbasis)
zero_q = [ (ii, 0) for ii in cov2.select('* GLU N,C,O,H1,H2,H3,OXT') ]
qmmm = QMMM(cov2, qmatoms=qmatoms, prefix='cov2', mm_key=tinker_key,
    mm_opts = dict(charges=zero_q, noconnect=ligatoms),
    capopts=dict(basis=qmbasis, placement='rel', g_CH=0.714),
    chargeopts=dict(scaleM1=1.0, adjust='dipole', dipolecenter=0.75, dipolesep=0.5))

# initial geometries for optimization were created manually, so "else" code isn't tested
if os.path.exists('cov2.h5'):
  r_rs1, r_ps1 = read_hdf5('cov2.h5', 'r_rs1', 'r_ps1')
else:
  xyz_cov2 = XYZ(cov2, active=[a for a in qmatoms if a != cd])
  res_ps, r_ps1 = moloptim(qmmm.EandG, mol=cov2, coords=xyz_cov2, ftol=2E-07, vis=vis)
  # Now RS
  cov2.r = r_ps1
  cov2.bond([o3, he2], 0.95, move_frag=False)
  cov2.bond([oe2, hw1], 0.95, move_frag=False)
  cov2.bond([ow, h6], 0.95, move_frag=False)
  # DLC doesn't seem to do much better than XYZ here
  res_rs, r_rs1 = moloptim(qmmm.EandG, mol=cov2, coords=xyz_cov2, ftol=2E-07, vis=vis)
  write_hdf5('cov2.h5', r_rs1=r_rs1, r_ps1=r_ps1)


## let's try step-wise

vis = Chemvis(Mol(cov2, [ VisGeom(style='licorice') ]), fog=True, wrap=False).run()

RC1r = [0]*nimages
RC1e = [0]*nimages
RC1r[0], RC1r[-1] = r_rs1, r_ps1

dlc = DLC(cov2, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constrain([[ii] for ii in cov2.listatoms(exclude=qmatoms)])
dlc.constraint(dlc.bond([he2, o3]) - dlc.bond([he2, oe1]))

rc0 = calc_dist(RC1r[0][[he2, o3]]) - calc_dist(RC1r[0][[he2, oe1]])
rc1 = calc_dist(RC1r[-1][[he2, o3]]) - calc_dist(RC1r[-1][[he2, oe1]])
rc1s = np.linspace(rc0, rc1, nimages)
cov2.r = RC1r[0]

for ii in range(1, nimages):
  r_oe1, r_o3 = cov2.atoms[oe1].r, cov2.atoms[o3].r
  a = rc1s[ii]/calc_dist(r_oe1, r_o3)
  cov2.atoms[he2].r = 0.5*(1+a)*r_oe1 + 0.5*(1-a)*r_o3
  print("rc: expected: %f; actual: %f" % (rc1s[ii], cov2.bond([he2, o3]) - cov2.bond([he2, oe1])))
  res, r = moloptim(qmmm.EandG, mol=cov2, coords=dlc, ftol=5E-07, vis=vis)
  RC1r[ii], RC1e[ii] = r, res.fun
  cov2.r = r

write_hdf5('cov2_rc1.h5', rcs=rc1s, RCr=RC1r, RCe=RC1e)

## step 2

dlc = DLC(cov2, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constrain([[ii] for ii in cov2.listatoms(exclude=qmatoms)])
cov2.r = RC1r[-1]
cov2.bond([c2, hw2], 1.08, move_frag=False)
res, r_ps2 = moloptim(qmmm.EandG, mol=cov2, coords=dlc, ftol=5E-07, vis=vis)

RC2r = [0]*nimages
RC2e = [0]*nimages
RC2r[0], RC2r[-1] = RC1r[-1], r_ps2

dlc = DLC(cov2, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constrain([[ii] for ii in cov2.listatoms(exclude=qmatoms)])
dlc.constraint(dlc.bond([hw2, ow]) - dlc.bond([hw2, c2]))

rc0 = calc_dist(RC2r[0][[hw2, ow]]) - calc_dist(RC2r[0][[hw2, c2]])
rc1 = calc_dist(RC2r[-1][[hw2, ow]]) - calc_dist(RC2r[-1][[hw2, c2]])
rc2s = np.linspace(rc0, rc1, nimages)
cov2.r = RC2r[0]

for ii in range(1, nimages):
  r_c2, r_ow = cov2.atoms[c2].r, cov2.atoms[ow].r
  a = rc2s[ii]/calc_dist(r_c2, r_ow)
  cov2.atoms[hw2].r = 0.5*(1+a)*r_c2 + 0.5*(1-a)*r_ow
  print("rc: expected: %f; actual: %f" % (rc2s[ii], cov2.bond([hw2, ow]) - cov2.bond([hw2, c2])))
  res, r = moloptim(qmmm.EandG, mol=cov2, coords=dlc, ftol=5E-07, vis=vis)
  RC2r[ii], RC2e[ii] = r, res.fun
  cov2.r = r

write_hdf5('cov2_rc2.h5', rcs=rc2s, RCr=RC2r, RCe=RC2e)

# two step barrier is ~0.08 H
# - should we try other step-wise paths?

# RC2r[1] rotated to a different orientation w/ significantly lower energy
dlc = DLC(cov2, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constrain([[ii] for ii in cov2.listatoms(exclude=qmatoms)])
cov2.r = RC2r[1]
res, r_rs2 = moloptim(qmmm.EandG, mol=cov2, coords=dlc, ftol=5E-07, vis=vis)

# let's try NEB w/ this new orientation
NEBr = [0]*nimages
NEBr[0], NEBr[-1] = r_rs2, r_ps2
neb = NEB(R=NEBr[0], P=NEBr[-1], EandG=qmmm.EandG, nimages=len(NEBr), k=2.0, climb=False)
NEBres = gradoptim(neb.EandG, neb.active(), ftol=0, gtol=0.05)
NEBr[1:-1] = np.reshape(NEBres.x, (len(NEBr)-2, -1, 3))
NEBe = [qmmm.EandG(xyz, dograd=False) for xyz in NEBr]
write_hdf5('cov1_neb2.h5', NEBr=NEBr, NEBe=NEBe)

# this gives barrier of 0.08 H

# how could we have found this other orientation w/o relying on luck?
# - global search with some constraints holding things near TS (e.g., hold H6 equidistant between O3 and OE1)

cov2.r = RC11r[5]  # HE2 approx equidistant
dlc = DLC(cov2, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constrain([[ii] for ii in cov2.listatoms(exclude=qmatoms)])
dlc.constraint(dlc.bond([he2, o3]) - dlc.bond([he2, oe1]))

mcoptim = MCoptim()
Es, Rs = mcoptim(qmmm.EandG, cov2.r, coords=dlc, hop=True, hopcoords=dlc, hopargs=dict(gtol=2E-04, ftol=5E-07, maxiter=20), G0=2E-5, adjstep=4, maxiter=40)



# NEB for GLH + HOH
if os.path.exists('cov2_neb.h5'):
  NEBr, NEBe = read_hdf5('cov2_neb.h5', 'NEBr', 'NEBe')
else:
  NEBr = [0]*nimages
  NEBr[0], NEBr[-1] = r_rs1, r_ps1
  #neb = NEB(R=r_rs1, P=r_ps1, EandG=qmmm.EandG, nimages=len(NEBr), k=2.0, climb=False)

  # NEB w/ DLCs: for now, we can use coordwrap w/ reinit=False
  # - seems good to interpolate in same coords as used for NEB calc so that initial elastic force is 0
  # - if we get good results, we may want to add support for coords to NEB()
  dlc = DLC(cov2, autobonds='total', autoangles='none', autodiheds='none', recalc=1)
  neb = NEB(R=r_rs1, P=r_ps1, EandG=coordwrap(qmmm.EandG, dlc, reinit=False), nimages=len(NEBr), k=2.0, climb=False)
  dlc.init(r_rs1)
  sR = dlc.active()
  sP = dlc.fromxyz(r_ps1)
  neb.images = linear_interp(sR, sP, nimages)

  #neb = NEB(R=NEBr[0], P=NEBr[-1], EandG=qmmm.EandG, nimages=len(NEBr), k=2.0, climb=False)
  # TC interpolates better ... but NEB w/ resulting cartesians doesn't work well
  #neb.images = dlc_interp(r_rs1, r_ps1, dlc=DLC(cov2, autobonds='total', autoangles='none', autodiheds='none', recalc=1), nimages=10)
  # automatically determine missing bonds ... seems to give very similar result to TC
  #neb.images = dlc_interp(r_rs1, r_ps1, dlc=DLC(cov2, connect='auto', recalc=1), nimages=10)

  NEBres = gradoptim(neb.EandG, neb.active(), ftol=0, gtol=0.05)  # best we can do for gtol
  NEBr[1:-1] = np.reshape(NEBres.x, (len(NEBr)-2, -1, 3))
  NEBe = [qmmm.EandG(xyz, dograd=False) for xyz in NEBr]
  E_nebts, r_nebts, dr_nebts = path_ts(NEBe, NEBr)
  #[calc_dist(r[o3_h6]) - calc_dist(r[c2_h6]) for r in NEBr]
  write_hdf5('cov2_neb.h5', NEBr=NEBr, NEBe=NEBe)

  # confirm that Emm isn't varying too much
  Ecomps = [qmmm.EandG(r, dograd=False) and qmmm.Ecomp for r in NEBr]



## Concerted reaction w/ two waters
# NEB path looks good; barrier of 0.07H
# RC scan: OK, except RS energy is -304.93228129726538, so just moving H6 to O8 isn't the right thing

if os.path.exists('con0.xyz'):
  con0 = load_molecule('con0.xyz')
else:
  r_o3, r_h6 = r_rs0[o3], r_rs0[h6]
  con0 = Molecule(header="CH2CHOH - CH3CHO w/ 2x H20; amber96.prm")
  con0.append_atoms(ch3cho, r=r_rs0, residue='LIG')
  #con0.append_atoms(place_residue('GLU', [11], [r_h6 + 2.5*(r_h6 - r_o3)], r_o3))  # GLU 11,12: OE1,OE2
  con0.append_atoms(h2o, r=h2o.r + (r_h6 + 2.5*(r_h6 - r_o3)) - h2o.r[0], residue='HOH')
  con0.append_atoms(h2o, r=h2o.r + (r_rs0[c2] + 2.5*(r_rs0[c2] - r_rs0[0])) - h2o.r[1], residue='HOH')

qmatoms = init_qmatoms(con0, '*', qmbasis)
qmmm = QMMM(con0, qmatoms=qmatoms, prefix='con0', mm_key=tinker_key)

if 0:
  rgds = HDLC(con0, [XYZ(con0, active=[], atoms=res.atoms) if res.name == 'LIG' else
      Rigid(con0, res.atoms, pivot=None) for res in con0.residues])
  res, r_con1 = moloptim(qmmm.EandG, mol=con0, coords=rgds, vis=vis, ftol=2E-07)
  write_xyz(con0, 'con0.xyz', r=r_con1)


vis = Chemvis(Mol(con0, [ VisGeom(style='licorice') ]), fog=True, wrap=False).run()

o8 = 8-1  # water 1
h13 = 13-1  # water 2
if os.path.exists('con_neb1.h5'):
  rc1s, RC1r, RC1e = read_hdf5('con1.h5', 'rc1s', 'RC1r', 'RC1e')
else:
  dlc = DLC(con0, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
  dlc.constraint(dlc.bond([h6, o8]))
  dlc.constraint(dlc.bond([c2, h13]))

  con0.bond([o8, h6], 0.9572, move_frag=False)
  #res, r = moloptim(qmmm.EandG, mol=con0, coords=dlc, ftol=2E-07, vis=vis)

  RC1r = [0]*nimages
  RC1e = [0]*nimages

  rc0 = con0.bond([c2, h13])
  rc1 = con0.bond([c2, 5-1])  # C-H dist
  rc1s = np.linspace(rc0, rc1, nimages)

  for ii in range(1, nimages):
    con0.bond([c2, h13], rc1s[ii], move_frag=False)
    #print("rc: expected: %f; actual: %f" % (rcs[ii], cov0.bond(o3_h6) - cov0.bond(oe1_h6)))
    res, r = moloptim(qmmm.EandG, mol=con0, coords=dlc, ftol=5E-07, vis=vis)
    RC1r[ii], RC1e[ii] = r, res.fun
    con0.r = r

  write_hdf5('con1.h5', rc1s=rc1s, RC1r=RC1r, RC1e=RC1e)


if os.path.exists('con_neb1.h5'):
  NEBr, NEBe = read_hdf5('con_neb1.h5', 'NEBr', 'NEBe')
else:
  NEBr = [0]*nimages
  neb = NEB(R=con0.r, P=RC1r[-1], EandG=qmmm.EandG, nimages=len(NEBr), k=2.0, climb=False)
  NEBres = gradoptim(neb.EandG, neb.active(), ftol=0, gtol=0.05)  # best we can do for gtol
  NEBr[1:-1] = np.reshape(NEBres.x, (len(NEBr)-2, -1, 3))
  NEBe = [qmmm.EandG(xyz, dograd=False) for xyz in NEBr]
  E_nebts, r_nebts, dr_nebts = path_ts(NEBe, NEBr)
  #[calc_dist(r[o3_h6]) - calc_dist(r[c2_h6]) for r in NEBr]
  write_hdf5('con_neb1.h5', NEBr=NEBr, NEBe=NEBe)

  # this gave minimal change vs. NEB TS
  E_rcts, r_rcts, dr_rcts = path_ts(NEBe, NEBr)
  lanczos = LanczosMethod(EandG=qmmm.EandG, tau=dr_rcts)
  LCZres = gradoptim(lanczos.EandG, r_rcts, ftol=0, maxiter=100)
  r_lcz0, E_lcz = getvalues(LCZres, 'x', 'fun')
  r_lcz = align_atoms(r_lcz0, r_rs0, sel=[0,1,2])
  write_hdf5('con_lcz.h5', r_lcz=r_lcz, E_lcz=E_lcz)


## covalent/H-bonding reaction path ... barrier is ~0.10 H for closer GLU
if os.path.exists('cov0.xyz'):
  cov0 = load_molecule('cov0.xyz')
else:
  r_o3, r_h6 = r_rs0[o3_h6[0]], r_rs0[o3_h6[1]]
  cov0 = Molecule(header="CH2CHOH - CH3CHO theozyme 1; amber96.prm")
  cov0.append_atoms(ch3cho, r=r_rs0, residue='LIG')
  cov0.append_atoms(place_residue('GLU', [11], [r_h6 + 2.5*(r_h6 - r_o3)], r_o3))  # GLU 11,12: OE1,OE2

  #r_rs0 = cov0.r
  #cov0.bond(oe1_h6, cov0.bond(o3_h6), move_frag=False)  # move just H6
  #r_ps0 = cov0.r
  write_xyz(cov0, 'cov0.xyz')


oe1 = cov0.select('* GLU OE1')[0]
oe1_h6 = [oe1, h6]

qmatoms = init_qmatoms(cov0, '* LIG *; * GLU OE1,OE2,CD', qmbasis)
zero_q = [ (ii, 0) for ii in cov0.select(pdb='~LIG N,C,O,H1,H2,H3,OXT') ]
qmmm = QMMM(cov0, qmatoms=qmatoms, prefix='cov0', mm_key=tinker_key, qm_charge=-1,
    mm_opts = dict(charges=zero_q, noconnect=ligatoms),
    capopts=dict(basis=qmbasis, placement='rel', g_CH=0.714),
    chargeopts=dict(scaleM1=1.0, adjust='dipole', dipolecenter=0.75, dipolesep=0.5))


# cov2.h5 vs cov3.h5 ... main difference is (initial) distance for GLU from LIG ... initial distance for cov3 was optimized by allowing H6 to move (at fixed reaction coord), along with GLU
cov2_h5 = 'cov0_2.h5'
cov3_h5 = 'cov0_3.h5'
if os.path.exists(cov3_h5):
  rc1s, RC1r, RC1e = read_only(*read_hdf5(cov3_h5, 'rc1s', 'RC1r', 'RC1e'))
  if os.path.exists(cov2_h5):
    rc2s, RC2r, RC2e = read_only(*read_hdf5(cov2_h5, 'rc1s', 'RC1r', 'RC1e'))  # path w/ GLU more distant
else:
  # optimize position of catres at fixed RC (near TS), allowing H6 to move
  #E_ts2, r_ts2, _ = path_ts(RC1e, RC1r)
  _, r_rcts, _ = path_ts(RCe, RCr)
  r_ts2 = setitem(cov0.r, ligatoms, r_rcts)

  dlc = DLC(cov0, atoms=ligatoms, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
  dlc.constrain([ [ii] for ii in ligatoms if ii != h6 ])
  dlc.constraint(dlc.bond(o3_h6) - dlc.bond(c2_h6))

  pivot = np.mean(r_ts2[ligatoms], axis=0)
  rgds = HDLC(cov0, [dlc if res.name == 'LIG' else  #XYZ(r_ts2, active=[], atoms=res.atoms)
      Rigid(r_ts2, res.atoms, pivot=pivot) for res in cov0.residues])

  res, r_ts3 = moloptim(qmmm.EandG, mol=cov0, r0=r_ts2, coords=rgds, vis=vis, maxiter=100, raiseonfail=False)

  # Now optimize RS, PS
  cov0.r = r_ts3
  #xyz_h6 = XYZ(r_rs0, active=[h6])
  xyz_lig = XYZ(cov0, active=ligatoms)
  res_rs, r_rs1 = moloptim(qmmm.EandG, mol=cov0, r0=setitem(cov0.r, ligatoms, r_rs0), coords=xyz_lig, ftol=2E-07, vis=vis)
  res_ps, r_ps1 = moloptim(qmmm.EandG, mol=cov0, r0=setitem(cov0.r, ligatoms, r_ps0), coords=xyz_lig, ftol=2E-07, vis=vis)
  #r_ps1 = align_atoms(r_ps1, r_rs1, sel=[0,1,2])  # align heavy atoms

  RC1r = [0]*nimages
  RC1e = [0]*nimages
  RC1r[0], RC1e[0] = r_rs1, res_rs.fun
  RC1r[-1], RC1e[-1] = r_ps1, res_ps.fun  #qmmm.EandG(r_ps0, dograd=False)

  # incidentally, these coords also give the fewest iters for optimizing RS, PS geom
  dlc = DLC(cov0, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
  dlc.constrain([[ii] for ii in cov0.listatoms(exclude=ligatoms)])
  #dlc.constraint(dlc.bond(o3_h6) - dlc.bond(oe1_h6))
  dlc.constraint(dlc.bond(o3_h6) - dlc.bond(c2_h6))

  rc0 = calc_dist(RC1r[0][o3_h6]) - calc_dist(RC1r[0][c2_h6])
  rc1 = calc_dist(RC1r[-1][o3_h6]) - calc_dist(RC1r[-1][c2_h6])
  rc1s = np.linspace(rc0, rc1, nimages)
  cov0.r = RC1r[0]

  for ii in range(1, nimages-1):
    # reverse bonds and use move_frag=False to keep H6 fixed
    cov0.bond(o3_h6[::-1], 0.5*(rc1s[ii] - rc1s[ii-1]), rel=True, move_frag=False)
    cov0.bond(c2_h6[::-1], -0.5*(rc1s[ii] - rc1s[ii-1]), rel=True, move_frag=False)
    #print("rc: %f; O2-H4 - C1-H4: %f" % (rc1s[ii], cov0.bond(o3_h6) - cov0.bond(c2_h6)))
    res, r = moloptim(qmmm.EandG, mol=cov0, coords=dlc, ftol=2E-07, vis=vis)
    #r = align_atoms(r, r_rs0, sel=[0,1,2])
    RC1r[ii], RC1e[ii] = r, res.fun
    cov0.r = r

  write_hdf5(cov3_h5, rc1s=rc1s, RC1r=RC1r, RC1e=RC1e)


#plot(rc1s, RC1e, xlabel="O3-H6 - C2-H6 (Angstroms)", ylabel="energy (Hartrees)")
#plot(rc2s, RC2e, xlabel="O3-H6 - C2-H6 (Angstroms)", ylabel="energy (Hartrees)")

# reaction path with GLU closer looks very promising ... what next?
# - Lanczos to get TS, maybe try Theo_Opt2
# - then we'd add non-catres

# - this reaction path requires some motion of ligand (but note ligand was free to move, so we don't have to worry about preventing it from flying away
# - is this different from H moving from LIG to GLU, then moving back to LIG?  Try modeling this as a two step process
#  - how far away should LIG be? should we minimize energy w/ RC fixed such that H is equidistant between GLU OE and LIG O?

#internals = cov0.get_internals(active=cov0.listatoms(exclude=[h6]))
#[dlc.constrain(ic) for ic in internals]

vis = Chemvis(Mol(cov0, [ VisGeom(style='licorice') ]), fog=True, wrap=False).run()

r_cov0 = cov0.r

if os.path.exists('cov0_h6opt.xyz'):
  r_h6opt = load_molecule('cov0_h6opt.xyz').r
else:
  dlc = DLC(cov0, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
  dlc.constrain(totalconn([ii for ii in cov0.residues[0].atoms if ii != h6]))
  dlc.constrain(totalconn(cov0.residues[1].atoms))
  dlc.constrain([h6])
  dlc.constraint(dlc.bond(o3_h6) - dlc.bond(oe1_h6))

  cov0.atoms[h6].r = (r_cov0[oe1_h6[0]] + r_cov0[o3_h6[0]])/2
  #cov0.bond(oe1_h6[::-1], cov0.bond(o3_h6)) # move GLU so OE1 and O3 are equidistant from H6

  res, r_h6opt = moloptim(qmmm.EandG, mol=cov0, coords=dlc, ftol=2E-07, vis=vis)

  write_xyz(cov0, 'cov0_h6opt.xyz', r=r_h6opt)


## 2 step: H6 -> OE1, then H6 -> C2 (RC is H6 - C2 dist, starting from step 1 PS)

# step 1: issue is that there are two params: H6 - OE1 dist and O3 - OE1 dist; we've found two different minima for equidistant H6
# let's try: freeze OE1 - O3 dist, scan H6 position

# binary search to find OE1 - O3 dist where H6 stays on OE1
if 0:
  dmin = 2.0
  dmax = 4.0

  for ii in range(10):
    print("dmin: %f, dmax: %f" % (dmin, dmax))
    d = (dmin + dmax)/2.0
    cov0.r = r_h6opt
    cov0.bond([oe1, o3], d)
    cov0.bond(oe1_h6, calc_dist(r_rs0[o3_h6]), move_frag=False)
    # should XYZ.init() also set inactive coords? optionally?
    xyz_oo = XYZ(cov0, active=cov0.listatoms(exclude=[o3, oe1]))
    res_ps, r_ps3 = moloptim(qmmm.EandG, mol=cov0, coords=xyz_oo, ftol=5E-07, vis=vis)
    if calc_dist(r_ps3[o3_h6]) > calc_dist(r_ps3[oe1_h6]):
      dmax = d
    else:
      dmin = d

# result: dmin: 2.480469, dmax: 2.484375

RC3r = [0]*nimages
RC3e = [0]*nimages

d_o3oe1 = 2.5
cov0.r = r_h6opt
cov0.bond([oe1, o3], d_o3oe1)
r_cov0 = cov0.r

# fix just O3 and OE1 - effectively fixes only distance between
xyz_oo = XYZ(cov0, active=cov0.listatoms(exclude=[o3, oe1]))

# rename calc_dist, calc_angle, ... to geom_dist, geom_angle, ... ?
cov0.bond(o3_h6, calc_dist(r_rs0[o3_h6]), move_frag=False)
res_rs, r_rs3 = moloptim(qmmm.EandG, mol=cov0, coords=xyz_oo, ftol=2E-07, vis=vis)
RC3r[0], RC3e[0] = r_rs3, res_rs.fun

cov0.r = r_cov0
cov0.bond(oe1_h6, calc_dist(r_rs0[o3_h6]), move_frag=False)
res_ps, r_ps3 = moloptim(qmmm.EandG, mol=cov0, coords=xyz_oo, ftol=2E-07, vis=vis)
RC3r[-1], RC3e[-1] = r_ps3, res_ps.fun

dlc = DLC(cov0, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constrain([oe1, o3])
dlc.constraint(dlc.bond(o3_h6) - dlc.bond(oe1_h6))

rc0 = calc_dist(RC3r[0][o3_h6]) - calc_dist(RC3r[0][oe1_h6])
rc1 = calc_dist(RC3r[-1][o3_h6]) - calc_dist(RC3r[-1][oe1_h6])
rcs = np.linspace(rc0, rc1, nimages)
cov0.r = RC3r[0]

#d_o3oe1 = calc_dist(RC3r[0][[oe1, o3]])
for ii in range(1, nimages-1):
  cov0.atoms[h6].r = 0.5*(1+rcs[ii]/d_o3oe1)*cov0.atoms[oe1].r + 0.5*(1-rcs[ii]/d_o3oe1)*cov0.atoms[o3].r
  print("rc: expected: %f; actual: %f" % (rcs[ii], cov0.bond(o3_h6) - cov0.bond(oe1_h6)))
  res, r = moloptim(qmmm.EandG, mol=cov0, coords=dlc, ftol=5E-07, vis=vis)
  #r = align_atoms(r, r_rs0, sel=[0,1,2])
  RC3r[ii], RC3e[ii] = r, res.fun
  cov0.r = r

write_hdf5('cov_step1.h5', rcs=rcs, RCr=RC3r, RCe=RC3e)

plot(rcs, RC3e, xlabel="RC (Angstroms)", ylabel="energy (Hartrees)")


# step 2: transfer of H6 to distant C2, vs. close proximity of C2-H-OE1, followed by separation of product?

cov0.r = setitem(RC3r[-1].copy(), ligatoms, r_ps0)
cov0.bond(oe1_h6, 4.0)  #calc_dist(RC3r[-1][oe1_h6]))

xyz_co = XYZ(cov0, active=cov0.listatoms(exclude=[c2, oe1]))  # constrain C2-OE1 dist

res, r_step2 = moloptim(qmmm.EandG, mol=cov0, coords=xyz_co, ftol=5E-07, vis=vis)


RC4r = [0]*nimages
RC4e = [0]*nimages

RC4r[0], RC4e[0] = RC3r[-1], RC3e[-1]

rc0 = calc_dist(RC3r[-1][c2_h6])
rc1 = calc_dist(r_ps0[c2_h6])
rcs = np.linspace(rc0, rc1, nimages)

cov0.r = RC4r[0]

dlc = DLC(cov0, autobonds='total', autoangles='none', autodiheds='none', autoxyzs='all', recalc=1)
dlc.constraint(dlc.bond(c2_h6))
#dlc.constraint(dlc.bond(oe1_h6))

for ii in range(1, nimages):
  cov0.r = RC4r[ii]  #cov0.bond(c2_h6[::-1], rcs[ii])
  #xyz_ch = XYZ(cov0, active=cov0.listatoms(exclude=[h6, c2]))
  #print("rc: expected: %f; actual: %f" % (rcs[ii], cov0.bond(o3_h6) - cov0.bond(oe1_h6)))
  res, r = moloptim(qmmm.EandG, mol=cov0, coords=dlc, ftol=5E-07, vis=vis)
  RC4r[ii], RC4e[ii] = r, res.fun
  cov0.r = r

# RC is OE1-H6 distance
write_hdf5('cov_step2a.h5', rcs=rcs, RCr=RC4r, RCe=RC4e)

# Summary of two step approach: barrier ~0.04 H (vs. ~0.16 H for vacuum, ~0.10 H for single step)
# Step 1 (cov_step1.h5): H to OE1 - Eact ~ -341.086
# - H6 scanned w/ OE1-O3 fixed at 2.5 A (close to minimum value where H6 will stay on OE1)
# Step 2 (cov_step2e.h5): H to C2 - Eact ~ -341.065 (to be compared with -341.00 for single step)
# - created by first scanning C2-H6 w/ OE1-H6 fixed, then 2nd scan initialized from 1st but w/o OE1-H6 fixed
# - significant reorientation of LIG, would probably need to repeat w/ full active site

#theo_opt = Theo_Opt2(qmmm, r_ts2, RC1r[-1][ligatoms], r_ts2[ligatoms], RC1r[0][ligatoms], ligatoms)
#res, r_theo1 = moloptim(theo_opt, mol=cov0, r0=r_ts2, coords=rgds, vis=None, maxiter=50, raiseonfail=False)

# Reference energies should be energies of whole system w/ large separation between LIG and everything else


## point charge theozyme
if os.path.exists('vinyl_bq.h5'):
  bq_coords, bq_q = read_hdf5('vinyl_bq.h5', 'bq_coords', 'bq_q')
  charges = [ Bunch(r=r, qmq=q) for r, q in zip(bq_coords, bq_q) ]
  mqq = Molecule(atoms=[Atom(name='Q', r=q.r, mmq=q.qmq) for q in charges])
else:
  charge_qopt = Charge_QOpt(ch3cho, r_rs0, r_ts0, r_ps0, maxbq=6)
  nq = len(charge_qopt.bq_coords)
  res = scipy.optimize.minimize(charge_qopt, np.zeros(nq), bounds=[(-0.8, 0.5)]*nq, jac=True, method='L-BFGS-B', options=dict(disp=True))

  write_hdf5('vinyl_bq.h5', bq_coords=charge_qopt.bq_coords, bq_q=res.x)

  charges = [ Bunch(r=r, qmq=q) for r, q in zip(charge_qopt.bq_coords, res.x) ]
  mqq = Molecule(atoms=[Atom(name='Q', r=q.r, mmq=q.qmq) for q in charges])
  vis = Chemvis([Mol(ch3cho, [r_ts0], [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()


from zipfile import ZipFile
if os.path.exists('theos0.zip'):
  with ZipFile('theos0.zip', 'r') as zip:
    theos = [load_molecule(zip.read(f)) for f in zip.namelist()]
else:
  charges = sorted(charges, key=lambda c: c.qmq)
  r_com = center_of_mass(ch3cho)
  theos = []
  for ii in range(4):
    theo0 = Molecule()
    theo0.append_atoms(ch3cho, r=r_ts0, residue='LIG')
    theo0.append_atoms(place_residue('GLU', [10], [charges[0 if ii/2 == 0 else 1].r], r_com)) # CD
    theo0.append_atoms(place_residue('LYS', [12], [charges[-1 if ii%2 == 0 else -2].r], r_com)) # NZ
    load_tinker_params(theo0, key=tinker_key, charges=True, vdw=True)

    theo_qmmm = QMMM(mol=theo0, qmatoms=qmatoms, prefix='theo1', mm_opts=dict(prog=NCMM(theo0, qq=False)))
    theo_opt = Theo_Opt(theo_qmmm, r_rs0, r_ts0, r_ps0, E_rs0, E_ts0, E_ps0)
    #EandG_sanity(theo_opt, r_theo0, dr=setitem(2E-4*(np.random.rand(*np.shape(r_theo0)) - 0.5), qmatoms, 0))
    #vis = Chemvis([Mol(theo0, [ VisGeom(style='licorice') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()
    r_theo0 = theo0.r
    pivot = np.mean(r_theo0[qmatoms], axis=0)
    rgds = HDLC(theo0, [XYZ(r_theo0, active=[], atoms=res.atoms) if res.name == 'LIG' else
        Rigid(r_theo0, res.atoms, pivot=pivot) for res in theo0.residues])
    res, r_theo1 = moloptim(theo_opt, mol=theo0, r0=r_theo0, coords=rgds, vis=None, maxiter=50, raiseonfail=False)
    theo0.r = r_theo1
    theos.append(theo0)

  with ZipFile('theos0.zip', 'a') as zip:
    for ii,theo in enumerate(theos):
      zip.writestr('%03d.xyz', write_xyz(theo))


## place non-catres, relax with LJ potential, find RS, TS, and PS for resulting active site
vis = Chemvis([Mol(theos[0], [ VisGeom(style='lines') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()

for ii_theo, theo in enumerate(theos):
  # TODO: skip this theo if catalytic effect too small
  theo3 = Molecule(theo)  #, r=r_theo1)
  qmatoms = init_qmatoms(theo3, '* LIG *', qmbasis)
  for jj_theo in range(4):  #while 1:
    theo3.remove_atoms('* LEU *')  # clear previous
    # rotate randomly, then place a specified atom at specified position (with some random jitter)
    # One potential sanity check might be comparing radial distribution fn to actual protein
    origin = np.mean(theo3.r[qmatoms], axis=0)
    dirs = [normalize(np.mean(theo3.r[res.atoms], axis=0) - origin) for res in theo3.residues if res.name != 'LIG']
    ndirs = len(dirs)
    # generate random directions with some minimum angular separation
    nres = 9  # number of residues to add
    while len(dirs) < nres + ndirs:
      dir = normalize(np.random.randn(3))  # randn: normal (Gaussian) distribution
      if all(np.dot(dir, d) < 0.65 for d in dirs):  # < np.cos(np.sqrt(np.pi*(nres + ndirs)/4/np.pi)/2)
        dirs.append(dir)

    for dir in dirs[ndirs:]:
      refpos = origin + 4.0*dir + (np.random.random(3) - 0.5)
      # to support other residues, we place centroid of sidechain at refpos, then rotate to align CA w/ dir
      #mol = place_residue('LEU', [9], [refpos], origin)  # 9 is LEU CG
      mol = place_residue('LEU')
      r = mol.r
      r = r - np.mean(r[mol.select('sidechain')], axis=0)
      r = np.dot(r, align_vector(r[1] - origin, dir).T) + refpos  # [1] is CA for Tinker residues
      theo3.append_atoms(mol, r=r)

    load_tinker_params(theo3, key=tinker_key, charges=True, vdw=True)
    bbone = theo3.select(pdb='~LIG N,C,O,H1,H2,H3,OXT')
    molLJ = MolLJ(theo3, repel=[(ii, jj) for ii in bbone for jj in theo3.listatoms()])

    vis.children[0].mols = [theo3]
    vis.refresh(repaint=True)

    r_theo3 = theo3.r
    #rgds = HDLC(theo3, [Rigid(r_theo3, res.atoms) for res in theo3.residues])
    rgds = HDLC(theo3, [XYZ(r_theo3, active=[], atoms=res.atoms) if res.name != 'LEU' else
        Rigid(r_theo3, res.atoms, pivot=origin) for res in theo3.residues])
    res, r_theo3 = moloptim(molLJ, mol=theo3, r0=r_theo3, coords=rgds, vis=vis, ftol=2E-07)
    # try again if this is positive (w/ MolLJ(..., rigid=True))?
    print("LJ energy after optim: %f" % molLJ(r_theo3)[0])
    #quick_save(theo3, 'theo3.pdb', r=r_theo3)

    ## relax RS, PS w/ full active site
    zero_q = [ (ii, 0) for ii in theo3.select(pdb='~LIG N,C,O,H1,H2,H3,OXT') ]
    qmmm = QMMM(theo3, qmatoms=qmatoms, prefix='theo3', mm_key=tinker_key,
        mm_opts = dict(charges=zero_q, noconnect=ligatoms))
    xyz_theo3 = XYZ(r_theo3, active=qmatoms, ravel=False)
    # RS and PS
    res_rs, r_rs3 = moloptim(qmmm.EandG, mol=theo3, S0=r_rs0, coords=xyz_theo3, vis=vis, ftol=2E-07)
    res_ps, r_ps3 = moloptim(qmmm.EandG, mol=theo3, S0=r_ps0, coords=xyz_theo3, vis=vis, ftol=2E-07)
    # TS with Lanczos
    E_rcts, r_rcts, dr_rcts = path_ts(NEBe, NEBr)  # need dr_rcts again
    lanczos = LanczosMethod(EandG=coordwrap(qmmm.EandG, xyz_theo3), tau=dr_rcts)
    LCZres = gradoptim(lanczos.EandG, r_ts0, ftol=0, maxiter=100)
    r_lcz, E_lcz = getvalues(LCZres, 'x', 'fun')
    r_ts3 = setitem(r_theo3.copy(), qmatoms, r_lcz)
    E_rs3, E_ts3, E_ps3 = res_rs.fun, E_lcz, res_ps.fun
    print("dE_rs: %f; dE_ts: %f; dE_ps: %f" % (E_rs3 - E_rs0, E_ts3 - E_ts0, E_ps3 - E_ps0))

    write_hdf5('theo_%d_%d.h5' % (ii_theo, jj_theo), mol=write_xyz(theo3),
        r_rs3=r_rs3, r_ts3=r_ts3, r_ps3=r_ps3, E_rs3=E_rs3, E_ts3=E_ts3, E_ps3=E_ps3)



## OLD

from chem.mm import *
# I think the goal for now for should be able to do theozyme E and G w/o calling Tinker; for anything more -
#  full proteins, solvent, PBC, etc. - we should use Tinker
# ... nevermind, our python SimpleMM is 3 - 5x slower than tinker_EandG even for a small molecule!
# - ~0.5 ms NCMM, ~10 ms mmbonded()!
mol = load_molecule("../common/ARG1.xyz")
load_tinker_params(mol, key=tinker_key, charges=True, vdw=True, bonded=True)
test_MM(mol, key=tinker_key, niter=1)


# test subtractive MM calc for QM/MM - need to use system w/ all MM params available
gln1 = place_residue('GLN')
load_tinker_params(gln1, key=tinker_key, charges=True, vdw=True, bonded=True)
qmatoms = init_qmatoms(gln1, '* GLN CD,OE1,NE2,HE21,HE22', qmbasis)
qmmm = QMMM(gln1, qmatoms=qmatoms, prefix='gln1', mm_key=tinker_key,
    mm_opts=Bunch(charges=[], newcharges=False),
    capopts=Bunch(basis=qmbasis, placement='rel', g_CH=0.714),
    chargeopts=Bunch(scaleM1=1.0, adjust='dipole', dipolecenter=0.75, dipolesep=0.5))

qmmm.EandG()
qmmm.mm_opts.subtract=True
qmmm.EandG()
# ... subtractive QM works with Tinker; can't really test w/ SimpleMM because it creates a new Molecule w/o
#  MM params; we can sort this out if and when we have an actual use case


## RESP charges - not actually needed, but added here to document method
vinyl0q_h5 = 'vinyl0q.h5'
if os.path.exists(vinyl0q_h5):
  ch2choh.mmq, ch3cho.mmq = read_hdf5(vinyl0q_h5, 'q_rs', 'q_ps')
else:
  from chem.qmmm.resp import *
  _, _, scn = pyscf_EandG(ch3cho, r=r_ps0)
  ch3cho.mmq = resp(scn.base)
  _, _, scn2 = pyscf_EandG(ch3cho, r=r_rs0)
  ch2choh.mmq = resp(scn2.base)
  write_hdf5(vinyl0q_h5, q_rs=ch2choh.mmq, q_ps=ch3cho.mmq)


## testing RESP charge fit
# Gaussian MK points for comparison
#rmol = load_molecule('floB_hf2.xyz')
#vis = Chemvis(Mol(rmol, [ VisGeom(style='licorice', radius=0.1, coloring=coloring_opacity(color_by_element, 0.5)), VisGeom(style='licorice', sel=range(42)) ]), fog=False, wrap=False).run()

# test grid generation
#flo = load_molecule("flo.xyz")
#flogrid = chelpg_grid(flo.r, flo.znuc)  #connolly_mk_grid(flo.r, flo.znuc)
#gridmol = Molecule(r=flogrid, znuc=[0]*len(flogrid))
#vis = Chemvis([Mol(gridmol, VisGeom(style='licorice', radius=0.1, coloring=coloring_opacity(color_by_element, 0.5))), Mol(flo, VisGeom(style='licorice')) ], fog=False, wrap=False).run()

# test ESP charge fitting
_, _, scn = pyscf_EandG(ch3cho)  #, r)
mf = scn.base  #qmmm.prev_pyscf.base
q_resp = resp(mf)
# restrain non-hydrogens
q_resp2 = resp(mf, restrain=(ch3cho.znuc > 1)*0.1)
# for comparision
mulk1 = mf.mulliken_pop()  #scf.hf.mulliken_pop(mf.mol, mf.make_rdm1())

# charges don't seem too sensitive to details of grid ...
#grid = chelpg_grid(ch3cho.r, ch3cho.znuc, randmom=True)
#grid = chelpg_grid(ch3cho.r, ch3cho.znuc, minr=1.4, maxr=2.0)
#grid = connolly_mk_grid(ch3cho.r, ch3cho.znuc, nshells=11, density=5.0)
#q = grid_resp(ch3cho.r, grid, pyscf_esp_grid(scn.base, grid))

_, _, scn2 = pyscf_EandG(ch3cho, r=ch2choh.r)
q_ch2choh = resp(scn2.base)


## initial catres guess
if os.path.exists('theo0.pdb'):
  theo0 = quick_load('theo0.pdb')
  qmatoms = init_qmatoms(theo0, '* LIG *', qmbasis)
else:
  r_com = center_of_mass(ch3cho)
  theo0 = Molecule()
  theo0.append_atoms(ch3cho, r=r_ts0, residue='LIG')
  theo0.append_atoms(place_residue('GLN', [11,18], bq_coords[[1,0]], r_com))  # GLN OE1: 11; HE22 18
  #theo0.append_atoms(place_residue('GLU', [11,12], bq_coords[[1,4]], r_com)) # OE1, OE2
  theo0.append_atoms(place_residue('GLU', [12], bq_coords[[2]], r_com)) # OE1, OE2
  quick_save(theo0, 'theo0.pdb')


r_theo0 = theo0.r
# zero charges on backbone (but not CA or HA):
zero_q = [ (ii, 0) for ii in select_atoms(theo0, pdb='~LIG N,C,O,H1,H2,H3,OXT') ]  # 'protein and not sidechain'


## find placement of (rigid) catres for optimal catalysis (minimize E_ts w/o lowering E_rs or E_ps)
load_tinker_params(theo0, key=tinker_key, charges=True, vdw=True)

if os.path.exists('theo0.pdb'):
  r_theo1 = load_molecule('theo1.xyz').r
else:
  # no charge-charge interactions
  #theo_qmmm = QMMM(mol=theo0, qmatoms=qmatoms, prefix='theo1', mm_key=tinker_key + "charge-cutoff 0.0\n")
  theo_qmmm = QMMM(mol=theo0, qmatoms=qmatoms, prefix='theo1', mm_opts=dict(prog=NCMM(theo0, qq=False)))
  theo_opt = Theo_Opt(theo_qmmm, r_rs0, r_ts0, r_ps0, E_rs0, E_ts0, E_ps0)
  #EandG_sanity(theo_opt, r_theo0, dr=setitem(2E-4*(np.random.rand(*np.shape(r_theo0)) - 0.5), qmatoms, 0))

  vis = Chemvis([Mol(theo0, [ VisGeom(style='licorice') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()

  pivot = np.mean(r_theo0[qmatoms], axis=0)
  rgds = HDLC(theo0, [XYZ(r_theo0, active=[], atoms=res.atoms) if res.name == 'LIG' else
      Rigid(r_theo0, res.atoms, pivot=pivot) for res in theo0.residues])
  res, r_theo1 = moloptim(theo_opt, mol=theo0, r0=r_theo0, coords=rgds, vis=vis, raiseonfail=False)
  write_tinker_xyz(theo0, 'theo1.xyz', r=r_theo1)


## place additional non-catalytic residues randomly around active site and relax w/ just LJ potential
#if os.path.exists('theo3.pdb'):
#  theo3 = quick_load('theo3.pdb')
#  load_tinker_params(theo3, key=tinker_key, charges=True, vdw=True)
#  r_theo3 = theo3.r
#  qmatoms = init_qmatoms(theo3, theo3.select(pdb='LIG *'), qmbasis)
#else:


RC3r = [0]*nimages
RC3e = [0]*nimages
RC3r[0], RC3e[0] = r_rs3, E_rs3
RC3r[-1], RC3e[-1] = r_ps3, E_ps3


# Once again, barrier is higher in theozyme than soln (0.116 Hartree vs. 0.114), but otherwise things seem to be working OK ... rather in vacuum, not soln!!!


# COM modes:
#E,G,H = MolLJ(theo3)(r_theo3)
#Hcom = np.einsum('ijkl->kl', H[qmatoms][:,qmatoms])
#ev, V = np.linalg.eigh(Hcom)
#vis = Chemvis([Mol(theo3, [ VisGeom(style='lines'), VisVectors([[origin, v] for v in V]) ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))) ], fog=True, wrap=False).run()


## Hydronium
# don't think this will be useful for our purposes, but it would interesting to see if we could
#  get something close to correct pH of water ... how?  Some papers relax, e.g., H11O5+ clusters; would a
#  clever initial arrangement allow a locally stable, e.g., H10O5 cluster containing H3O+ and OH- ?

wat0 = Molecule(header="2x H20; amber96.prm")
wat0.append_atoms(h2o, r=h2o.r - h2o.r[0], residue='HOH')
wat0.append_atoms(h2o, r=h2o.r - h2o.r[0] + 1.8, residue='HOH')

qmatoms = init_qmatoms(wat0, '*', qmbasis)  #'6-31G*'
qmmm = QMMM(wat0, qmatoms=qmatoms, prefix='wat0', mm_key=tinker_key)

vis = Chemvis(Mol(wat0, [ VisGeom(style='licorice') ]), fog=True, wrap=False).run()

rgds = HDLC(wat0, [Rigid(wat0, res.atoms, pivot=None) for res in wat0.residues])

res0, r_wat0 = moloptim(qmmm.EandG, mol=wat0, coords=rgds, vis=vis, ftol=2E-07)
res1, r_wat1 = moloptim(qmmm.EandG, mol=wat0, r=r_wat0, vis=vis, ftol=2E-07)  #-152.03040957818496

r_sep1 = np.array(r_wat1)
r_sep1[wat0.residues[0].atoms] += 20
E_sep1 = qmmm.EandG(r=r_sep1, dograd=False)   # -152.02147257288888

# move H from one HOH to the other
wat0.r = r_wat1
wat0.bond([3, 2], wat0.bond([0, 1]), move_frag=False)
wat0.bond([2, 0], 20)
res2, r_wat2 = moloptim(qmmm.EandG, mol=wat0, vis=vis, ftol=2E-07)  # -151.48391974244419 (sep)

# scf.hf.mulliken_pop(qmmm.prev_pyscf.mol, qmmm.prev_pyscf.base.make_rdm1())

h3o = wat0.extract_atoms([2,3,4,5])
oh = wat0.extract_atoms([0,1])

qmmm_oh = QMMM(oh, qmatoms='*', prefix='oh', qm_charge=-1, mm_key=tinker_key)
qmmm_h3o = QMMM(h3o, qmatoms='*', prefix='h3o', qm_charge=1, mm_key=tinker_key)

res_h3o, r_h3o = moloptim(qmmm_h3o.EandG, mol=h3o, ftol=2E-07)
res_oh, r_oh = moloptim(qmmm_oh.EandG, mol=oh, ftol=2E-07)
res_h3o.fun + res_oh.fun  # -151.61593602892134

wat1 = Molecule(header="2x H20; amber96.prm")
wat1.append_atoms(h3o, r=r_h3o - r_h3o[0], residue='H3O')
wat1.append_atoms(oh, r=r_oh - r_oh[0] + 1.8, residue='OH')

qmmm2 = QMMM(wat1, qmatoms='*', prefix='wat1', mm_key=tinker_key)

rgds = HDLC(wat1, [Rigid(wat1, res.atoms, pivot=None) for res in wat1.residues])

res11, r_wat1 = moloptim(qmmm2.EandG, mol=wat1, coords=rgds, vis=vis, ftol=2E-07)
res11.fun  # -151.92331050815321

E_sep2 = qmmm_oh.EandG(dograd=0) + qmmm_h3o.EandG(dograd=0)  # -151.48879020569711

r_sep2 = np.array(r_wat2)
r_sep2[wat0.residues[0].atoms] += 20
E_sep2 = qmmm.EandG(r=r_sep2, dograd=False)
