import os, subprocess
from pyscf import gto
from chem.molecule import *
from chem.io import load_molecule, cclib_open
from chem.io.pdb import copy_residues, write_pdb
from chem.io.tinker import write_tinker_xyz, tinker_EandG
from chem.opt.dlc import DLC, Cartesian, HDLC
from chem.opt.optimize import moloptim, XYZ
from chem.qmmm.qmmm1 import QMMM
from chem.qmmm.prepare import protonation_check, neutralize

# QM/MM calculation of peptide bond hydrolysis reaction by trypsin, mostly following Ishida & Kato JACS 125,
#  12035 - 10.1021/ja021369m
# Note that the main idea here is to thoroughly document the calculation, not necessarily to automate for
#  reuse with other systems!
# Note that block select can be used to paste multiple indented lines into Python interactive prompt

## TODO:
# - in general, every atom of HIS has different mmtype depending on protonation state, so we can't simply
#  remove H atom and compute MM energy ... ideally, we'd be able to add hydrogens and set mmtypes in python
#  code rather than with Tinker

# - I'm concerned that the neutralizing CL atoms are distorting the structure ... shouldn't we consider the
#  impact of water's high dielectric const (~80)?  How could we account for this in QM calc?  Do we just need
#  point charges from explicit waters?
# - compare crystal atom RMSD and energy w/ and w/o initial optimization of hydrogens?

# - intuitively, QM/MM grad for MM optimized structure should be small and roughly the same for all QM and
#  M1,2,3,... atoms.  For trypsin system, anywhere we cut the I chain backbone will result in large
#  Q1 - M1 or M1 - M2 dipoles.  Even though the various gradient terms mostly cancel, we still end up with
#  relatively large gradient on M1 and M2 atoms which seems mostly due to interaction between HL and the
#  M1 - M2 dipole correction charge.
#  - ideally, we'd just never cut near polar bonds, but it's unavoidable in this case!
#  - if a simpler boundary charge scheme gives smaller gradient, should we just use that?
#  - force on HL is mostly along bond direction - maybe try optimizing bond length?
# - gradient correction for charges: calculated myself, so it's probably wrong; See src/hybrid/adq.c in Chemshell
# - Kato paper doesn't say anything about boundary charge handling, so we should just try to get the best result

# - The DDI program used to run GAMESS adds a three second delay at the end of every run; maybe try rebuilding to use
#  MPI instead of sockets (since NWChem requires OpenMPI anyway)
# - should writepdb write HID/HIE instead of HIS?
# - option to specify default basis for write_gamess_inp?
# - switch to python 3 (main task is reinstalling PySCF) ... first try OpenMM (conda)
#  - note PySCF scf creates python temporary files (tmpXXXXX) in lib.param.TMPDIR (. by default)
# - visualizing polarization: maybe subtract/clear all S orbitals for atom of interest?  note that just
#  subtracting constant from density doesn't work; isosurfaces are probably the best approach!
# - alternatives to current mol + r design?  Dict of r arrays as a member of Molecule?
# - move M1, M2, Q1 to qmmm, make byconnectivity/byproximity top-level attr

## Next:

# - fn optimization notes, code
# - QM/MM opt with full QM region (maybe try chain I alone first); use some of the new code below to explore results, then work on reaction path

# Misc ideas:
# - try adding polarizations fns (6-31G**) to cap atoms?
# - try including M1 (and M2?) in DLC object for optim (see dlcatoms below)
# - what if we optimize w/ just one sidechain at a time as QM region, then final opt w/ full QM region?
# - additional energy term to help restrain MM positions to PDB positions, relaxed (to zero?) eventually, for MM prep?

# - let's use individual sidechains as smaller test systems for debugging
#  - then backbone alone, figuring out how to handle the anomalously large grad there
#  - is the magnitude of the error E(r + dr) - (E(r) + dE(r)*dr) for QM/MM comparable with QM only and MM only?

# - Comparing orbitals and density around QM/MM boundary to uncut equivalent, esp. for chain I backbone
# ... gradient is not tiny ... maybe MM opt of GLY chain with torsions fixed?
# - compare pyscf orbitals to GAMESS orbitals (make sure geometry matches!); maybe try cclib_to_pyscf


## Visualizations
# uncomment this import to use Chemvis!
##from chem.vis.chemvis import *
# - unfortunately, these must be first because we're using pdb to stop execution part way through calculation
vis = None

def view_active_site(c):
  c.set_view(trypsin.atoms[hg_195].r, 40.0, r_up=trypsin.atoms[og_195].r, make_default=True)


def vis_compare_geom(*args):
  vis = Chemvis(Mol(trypsin, list(args), [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='protein'), VisGeom(style='lines', sel='not protein'), VisGeom(style='licorice', radius=0.5, sel=qmatoms) ]), fog=True).run()
  view_active_site(vis)


def vis_nonbonded():
  vis = Chemvis(Mol(trypsin, [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='protein'), VisGeom(style='lines', sel='not protein'), VisGeom(style='licorice', radius=0.5, sel=qmatoms), VisContacts(partial(tinker_contacts, key=tinker_qmmm_key, sel="chain == 'I'", Ethresh=-20.0/KCALMOL_PER_HARTREE), radius=tinker_contacts_radii, colors=tinker_contacts_color) ]), fog=True).run()
  view_active_site(vis)


def vis_qmmm_setup(qmmm):
  #qq = qmmm.get_charges(qmmm.mol, qmmm.qmatoms, **qmmm.chargeopts)
  #cap = qmmm.get_caps(qmmm.mol, qmmm.qmatoms)
  mqq = Molecule(atoms=[Atom(name=q.name, r=q.r, mmq=q.qmq) for q in qmmm.charges if q.qmq != 0])
  mcap = Molecule(atoms=[Atom(name='HL', r=c.r) for c in qmmm.caps])
  vis = Chemvis([Mol(qmmm.mol, [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='extbackbone'), VisGeom(style='lines', sel='sidechain'), VisGeom(style='lines', sel='not protein') ]), Mol(mqq, VisGeom(style='licorice', coloring=scalar_coloring('mmq', [-1,1]))), Mol(mcap, VisGeom(style='lines'))], fog=True).run()
  view_active_site(vis)


def vis_qmmm_grad(qmmm, r=None, reuselogs=True):
  #qm000 = ccopen('test1/tmp/initialQMMM_qm000.log', loglevel=logging.ERROR).parse()
  #Gqm = [qm000.atomcoords[0], qm000.atomcoords[0] + 10*qm000.grads[-1]/ANGSTROM_PER_BOHR]
  E, G = qmmm.EandG(qmmm.mol, r=r, prefix=folder + "/qmtest/initialQMMM", iternum=0, reuselogs=reuselogs, components=True)
  vis = Chemvis(Mol(qmmm.mol, r, [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='protein'), VisGeom(style='lines', sel='not protein'), VisGeom(style='licorice', radius=0.5, sel=qmatoms), VisVectors(10*G, colors=Color.magenta), VisVectors(10*qmmm.Gcomp['Gqmqm'], colors=Color.cyan), VisVectors(10*qmmm.Gcomp['Gcharge'], colors=Color.lime), VisVectors(10*qmmm.Gcomp['Gmm'], colors=Color.yellow) ]), fog=True).run()
  view_active_site(vis)


def vis_qmmm_dens():
  # 3297 is atom C of residue I 6 (M1 atom with current partition)
  #c = Chemvis(Mol(Files(folder + "/qmtest/*qm.log"), [ VisGeom(style='lines'), VisVol(cclib_dens_vol, extents=get_extents([trypsin.atoms[3297].r], pad=4.0)) ])).run()
  trypsin.cclib = cclib_open(folder + '/qmtest/initialQMMM_000qm.log')
  chainI = load_molecule(folder + '/qmtest/bbI_qm.log')
  vis = Chemvis([ Mol([trypsin, chainI], [ VisGeom(style='lines', sel='extbackbone'), VisGeom(style='lines', sel='sidechain'), VisGeom(style='lines', sel='not protein'), VisVol(cclib_dens_vol, extents=get_extents([trypsin.atoms[3295].r], pad=3.0), vis_type='iso', vol_type="MO Volume", timing=True) ]), Mol(trypsin, VisGeom(style='licorice', radius=0.5, sel=qmatoms), focused=False) ], fog=True, bg_color='#CCC').run()
  view_active_site(vis)


## Preparation
# mkdir trypsin1 && cd trypsin1
# bash getpdb.sh 1MCT
# echo parameters $TINKER_PATH/../params/amber96.prm > tinker.key
# $TINKER_PATH/pdbxyz 1MCT.pdb ALL ALL

# 1MCT-trimI.pdb: deleted residues from inhibitor (chain I), leaving just 'pdb_resnum in [3,4,5,6,7,8]';
#  placed OXT of last residue (8) at position of N of deleted residue 9; also deleted waters I 32 and I 35
# Also, changed HIS A 57 to 'HID' and the other three HIS to 'HIE' ... this seems to get Tinker pdbxyz to
#  select correct mmtypes.  It is not necessary to remove the HD1 or HE2 from the PDB: Tinker, copy_residues()
#  will ignore them.  Note that the HE2 position of HIS A 57 receives HG from SER A 195 during reaction
# NOTE: should have removed H from N-terminal CYS on trimmed chain I

# can't have env vars in tinker .key ... could do something like subprocess.check_output("echo $TINKER_PATH...
TINKER_FF = os.path.abspath(os.path.expandvars("$TINKER_PATH/../params/amber96.prm"))
tinker_qmmm_key = \
"""parameters     %s
digits 8

# QM charges appended here...
""" % TINKER_FF

# let's get everything working with GAMESS, then try pyscf (maybe Psi4 and NWchem too)
# - see http://sunqm.github.io/pyscf/qmmm.html
# Note ICHARG=-1, due to ASP residue in QM region; MM charge of QM region is -0.875 ... so can we assume the
#  difference is distributed over link atoms?  What does this mean?
# ISKPRP=1 skips population analysis
gamess_trypsin1_inp = \
""" $CONTRL
   SCFTYP=RHF RUNTYP=GRADIENT ICHARG=-1 MULT=1
   NPRINT=7 ISKPRP=1 MAXIT=100 $END
 $SYSTEM MWORDS=128 $END
"""

# folder for log files
# with qmmm.savefiles = False and a new prefix for each reaction coord (RC) we'll save final logs for each RC
folder = 'test1' #time.strftime('%Y-%m-%d_%H_%M')
tmpfolder = folder + '/tmp'
if not os.path.exists(tmpfolder):
  #os.makedirs(tmpfolder)
  subprocess.check_call("mkdir -p {0} && sudo mount -t tmpfs -o size=64M tmpfs {0}".format(tmpfolder), shell=True)


prepared_xyz_path = folder + "/prepared.xyz"
prepared_pdb_path = folder + "/prepared.pdb"
if not os.path.exists(prepared_xyz_path):
  crystal_pdb = load_molecule('1MCT-trimI.pdb')
  crystal_mol = load_molecule('1MCT-trimI.xyz', charges='generate')
  trypsin = copy_residues(crystal_mol, crystal_pdb)

  # check protonation states ... we'll assume physiological pH of 7.4
  # - at this point, trypsin should have charge of +8
  protonation_check(trypsin, pH=7.4)

  na_plus = Atom(name='NA', znuc=11, r=[0,0,0], mmq=1.0, mmtype=2004)
  cl_minus = Atom(name='CL', znuc=17, r=[0,0,0], mmq=-1.0, mmtype=2012)
  trypsin = neutralize(trypsin, ion_p=na_plus, ion_n=cl_minus)

  # translate COM to origin
  trypsin.r = trypsin.r - center_of_mass(trypsin)

  # save prepared state ... key file required for generate charges
  write_pdb(trypsin, prepared_pdb_path)
  write_tinker_xyz(trypsin, prepared_xyz_path)
  with open(folder + "/prepared.key", 'w') as f:
    f.write(tinker_qmmm_key)
else:
  prepared_pdb = load_molecule(prepared_pdb_path)
  prepared_mol = load_molecule(prepared_xyz_path, charges='generate')
  trypsin = copy_residues(prepared_mol, prepared_pdb)

r_prep = trypsin.r  # save geom

# initial MM minimization of system in cartesians
optimize = moloptim
mm_opt_xyz = folder + "/mm_optimized.xyz"
if not os.path.exists(mm_opt_xyz):
  tinker_args = Bunch(prefix=tmpfolder + "/initialMM", key=tinker_qmmm_key)

  def optim_mon_mm(r, r0, E, G, Gc, timing):
    print("***ENERGY: {:.6f} H, RMS grad: {:.6f} H/Ang, ({:.6f} H/Ang), RMSD: {:.3f} Ang, Coord update: {:.3f}s, E,G calc: {:.3f}s".format(E, rms(G), rms(Gc), calc_RMSD(r0, r), timing['coord'], timing['EandG']))

  # first pass moving only hydrogens and neutralizing ions
  h_and_cl = select_atoms(trypsin, 'znuc in [1, 11, 17]')
  res, r = optimize(trypsin, fn=tinker_EandG, fnargs=tinker_args, coords=XYZ(trypsin, h_and_cl), mon=optim_mon_mm)
  #assert res.success, "Initial hydrogen + ion MM minimization failed!"
  trypsin.r = r

  # second pass moving all atoms
  #hdlc = HDLC(trypsin, autodlc=1, dlcoptions=dict(recalc=1))
  #hdlc = HDLC(trypsin, autodlc=1, dlcoptions=dict(recalc=1, autobonds='total', autoangles='none', autodiheds='none', abseps=1e-14))
  # Note: all-atom MM opt (in Cartesians) was 3161 steps (~0.7s/step)
  res, r = optimize(trypsin, fn=tinker_EandG, fnargs=tinker_args, coords=XYZ(trypsin))
  #assert res.success, "Initial all-atom MM minimization failed!"
  trypsin.r = r

  write_tinker_xyz(trypsin, mm_opt_xyz)

  # also print max displacement?
  crystal_atoms = select_atoms(trypsin, 'znuc not in [1, 11, 17]')
  print("Final energy: %.6f H; crystal atom RMSD after MM opt: %f" \
      % (res.fun, calc_RMSD(r_prepared, trypsin.r, atoms=crystal_atoms)))
else:
  trypsin.r = load_molecule(mm_opt_xyz).r

r_mmopt = trypsin.r  # save geom

## QM/MM

# QM atoms: sidechains of A His57, Asp102, Ser195 plus backbone of I between 5 and 6
# - Kato paper includes CA of Ser195 (but not His57 or Asp102)
fullqmatoms = select_atoms(trypsin, pdb="A 57,102,195 ~C,N,CA,O,H,HA; I 5 CA,C,HA,O; I 6 CA,N,H,HA")

# tiny QM system (incl link atom it's methanol) for testing and debugging
qmatoms = select_atoms(trypsin, pdb="A 195 ~C,N,CA,O,H,HA")
#print("qmatoms net MM charge: %.3f" % sum([trypsin.atoms[ii].mmq for ii in qmatoms]))

qmatoms = fullqmatoms

# other important subsets of atoms ... this should probably be in qmmm
mmatoms = trypsin.listatoms(exclude=qmatoms)
M1atoms = trypsin.get_bonded(qmatoms)
M2atoms = trypsin.get_bonded(M1atoms + qmatoms)
Q1atoms = list(set(qmatoms).intersection(trypsin.get_bonded(M1atoms)))

# reaction coordinate for step 1: (HG_OG - HG_OG_0) - (OG_C - OG_C_0); HG_OG will increase, OG_C will decrease
# From Kato, HG_OG_0 = 0.957, HG_OG_TET = 1.720; OG_C_0 = 2.672, OG_C_TET = 1.513, so RC varies from 0 to
#  1.922 Ang (this seems to match plots in Kato)
hg_195 = select_atoms(trypsin, pdb="A 195 HG")[0]
og_195 = select_atoms(trypsin, pdb="A 195 OG")[0]
c_5 = select_atoms(trypsin, pdb="I 5 C")[0]
hg_og = (hg_195, og_195)
og_c = (og_195, c_5)


# Kato paper states no cutoff for QM-MM interactions (so every MM atom included as background charge?)
# ... but GAMESS won't accept more than 2000 atoms, incl. charges, so we limit charges to a fixed set of MM
#  atoms w/ close to zero net charge; this seems better than passing radius to get_charges in which case net
#  charge could vary between iterations
# - consider coarse-graining distant charges instead of excluding? (by residue? ... most residues neutral!)
charge_atoms = trypsin.get_nearby(qmatoms, 10.05)
print("charge_atoms net charge: %.3f" % sum([trypsin.atoms[ii].mmq for ii in charge_atoms]))

# basis setup:
# - default 6-31G* (incl cap atoms)
# - 6-31+G* ... is wrong in PySCF (< v1.5)!  But any of 6-31++G*, 6-31+G**, or 6-31++G** are the same for znuc > 1
diffuse = select_atoms(trypsin, pdb="A 102 CG,OD1,OD2; A 195 OG; I 5 C; I 6 N")
# - 6-31G** (P func on hydrogens)
pfunc = select_atoms(trypsin, pdb="A 195 HG; A 57 HD1,HD2")
for ii in qmatoms:
  trypsin.atoms[ii].qmbasis = '6-31G**' if ii in pfunc else ('6-31++G*' if ii in diffuse else '6-31G*')

# create QMMM driver
# typically we'll want to use capopts.placement = 'rel' (variable Q1-cap bond length) with M1inactive=True
# NWChem uses g = 0.709; for AMBER FF, g = 0.714 seems a bit closer to equilb C-H/C-C ratio
qmmm = QMMM(mol=trypsin, qmatoms=qmatoms, moguess='prev', M1inactive=True,
    qm_opts=Bunch(charge=-1, inp=gamess_trypsin1_inp),
    capopts=dict(basis='6-31G*', placement='rel', g_CH=0.714),
    chargeopts=dict(charge_atoms=charge_atoms, scaleM1=1.0, adjust='dipole', dipolecenter=0.75, dipolesep=0.5),
    mm_key=tinker_qmmm_key, savefiles=False)

# cap for bonds other than C-C must be manually configured
capI5_N_CA = qmmm.find_cap(select_atoms(trypsin, pdb="I 5 CA,N"))
if capI5_N_CA:
  capI5_N_CA.d0 = 1.09  # or capI5_N_CA.g = 0.752, based on CA-N bond dist of 1.449 from amber96.prm

# visualization of nonbonded interactions (MM level only)
#vis_nonbonded()

## for sanity checks
qmmm.qm_opts.charge = 0
qmmm.qm_opts.inp = \
""" $CONTRL
   SCFTYP=RHF RUNTYP=GRADIENT ICHARG=0 MULT=1
   NPRINT=7 ISKPRP=1 MAXIT=100 $END
 $SYSTEM MWORDS=128 $END
"""

# initial MOs
#gamess_init_log = folder + "/nocharges"
gamess_init_log = folder + "/sanity_noq"
if not os.path.exists(gamess_init_log + "_000qm.log"):
  # run GAMESS with no point charges to generate MOs to be used as initial guess for calculation w/ charges
  qmmm.EandG(trypsin, embed='none', moguess=None, prefix=gamess_init_log, iternum=0)
  # read initial MOs for subsequent calcs
else:
  qmmm.prev_cclib = cclib_open(gamess_init_log + "_000qm.log")

qmmm.qm_opts.inp += " $GUESS GUESS=MOREAD $END\n"

# visualization to check link atoms and background charges
#vis_qmmm_setup(qmmm)
# visualizations to check qmmm grad and electron density
#vis_qmmm_grad(qmmm, reuselogs=False)
#vis_qmmm_dens()

# QMMM E and G sanity checks
#from chem.test.qmmm_test import qmmm_sanity_check
#qmmm_sanity_check(qmmm, dratoms=qmatoms + M1atoms + M2atoms)

# create HDLC object for whole structure w/ DLC object for QM region and cartesians for MM region
# - system has far too many atoms for single DLC object; could try per-residue DLCs instead of Cartesians
dlcatoms = fullqmatoms  #qmatoms # + M1atoms + M2atoms  -- DLC() will fail if we dlcatoms don't include the reaction coord "bond" atoms
xyzatoms = trypsin.listatoms(exclude=dlcatoms)
dlc_qm = DLC(trypsin, atoms=dlcatoms, bonds=[hg_og, og_c], autoxyzs='all', recalc=1)
xyz_mm = Cartesian(trypsin, atoms=xyzatoms)
hdlc = HDLC(trypsin, [dlc_qm, xyz_mm])

def optim_mon_qmmm(r, r0, E, G, Gc, timing):
  print("***ENERGY: {:.6f} H, RMS grad {:.6f} H/Ang {:.6f} (QM), {:.6f} (Q1), {:.6f} (MM), {:.6f} (M1), {:.6f} (M2)  H/Ang, RMSD: {:.3f} time: {:.3f}s".format(E, rms(G), rms(G[qmatoms]), rms(G[Q1atoms]), rms(G[mmatoms]), rms(G[M1atoms]), rms(G[M2atoms]), calc_RMSD(r0, r), timing['total']))

if 0:
  from chem.io import read_hdf5, write_hdf5
  hdlc_opt1_h5 = "hdlc_opt1.h5"
  if not os.path.exists(hdlc_opt1_h5):
    # scipy.optimize.minimize callback is only passed r, so we'll have to stick with intercepting every call of
    #  objective fn, but discard any that increase energy
    E_hist = []
    G_hist = []
    r_hist = []

    def optim_mon_hist(r, r0, E, G, Gc, timing):
      if not E_hist or E < E_hist[-1]:
        E_hist.append(E)
        G_hist.append(G)
        r_hist.append(r)
      print("***ENERGY: {:.6f} H, RMS grad {:.6f} H/Ang, RMSD: {:.6f} time: {:.3f}s".format(E, rms(G), calc_RMSD(r0, r), timing['total']))

    res, r_qmmmopt = optimize(trypsin, fn=qmmm.EandG, coords=hdlc, r0=r_mmopt, mon=optim_mon_hist, ftol=2E-07)
    write_hdf5(hdlc_opt1_h5, r_hist=r_hist, E_hist=E_hist, G_hist=G_hist)
  else:
    r_hist, E_hist, G_hist = read_hdf5(hdlc_opt1_h5, 'r_hist', 'E_hist', 'G_hist')
    r_qmmmopt = r_hist[-1]

  # Arguably, one important check is for significant motion of atoms far from QM region; perhaps we can compute a moment dr*(dist from center of QM site)

  r_avgqm = np.mean(r_qmmmopt[qmatoms], axis=0)
  moment = np.sqrt(np.sum(np.square(r_hist[-1] - r_hist[0]), axis=1))*np.sqrt(np.sum(np.square(r_qmmmopt - r_avgqm), axis=1))
  moment_sort = np.argsort(moment)
  print("\n".join([pdb_repr(trypsin, ii) for ii in moment_sort[-10:]]))

  # this visualization seems more useful than the above list ...
  moment_vec = (r_hist[-1] - r_hist[0])*np.sqrt(np.sum(np.square(r_qmmmopt - r_avgqm), axis=1))[:,None]

  from chem.vis.chemvis import *
  vis = Chemvis(Mol(trypsin, r_hist[0], [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='protein'), VisGeom(style='lines', sel='not protein'), VisGeom(style='licorice', radius=0.5, sel=qmatoms), VisVectors(moment_vec, colors=Color.lime) ]), fog=True).run()


  import matplotlib.pyplot as plt
  ##plt.figure()
  #plt.plot([rms(g) for g in G_hist], label="all")
  #plt.plot([rms(g[qmatoms]) for g in G_hist], label="QM")
  #plt.plot([rms(g[Q1atoms]) for g in G_hist], label="Q1")
  #plt.plot([rms(g[M1atoms]) for g in G_hist], label="M1")
  #plt.plot([rms(g[M2atoms]) for g in G_hist], label="M2")
  #plt.legend()
  #plt.ylabel("rms grad (Hartrees/Ang)")
  #plt.xlabel("step")
  #plt.show()

  plt.plot(E_hist)
  plt.ylabel("energy (Hartrees)")
  plt.xlabel("step")
  plt.show()


# initial QM/MM minimization
qmmm.prefix = tmpfolder + "/initialQMMM"
qmmm_opt_xyz = folder + "/qmmm_optimized.xyz"
qmmm.iternum = 0  # in case reuselogs was used ... need a better soln
if not os.path.exists(qmmm_opt_xyz):
  res, r = optimize(trypsin, fn=qmmm.EandG, coords=hdlc, mon=optim_mon_qmmm, ftol=2E-07)
  #assert res.success, "Initial QM/MM minimization failed!"
  trypsin.r = r
  write_tinker_xyz(trypsin, qmmm_opt_xyz)
else:
  trypsin.r = load_molecule(qmmm_opt_xyz).r

r_qmmmopt = trypsin.r
#vis_compare_geom(r_mmopt, r_qmmmopt)

# constrain reaction coord
dlc_qm.constraint(dlc_qm.bond(hg_og) - dlc_qm.bond(og_c))

## TODO: restarting from checkpoint

# use array of rc values because, in general, we may not have evenly spaced values
rcs = np.linspace(0, 2.0, 9)
prev_rc = None
results = []
res_file = folder + '/results_step_1.py'
if not os.path.exists(res_file):
  with open(res_file, 'w') as f:
    f.write('results = []\n')

# start interactive
import pdb; pdb.set_trace()

for ii, rc in enumerate(rcs):
  # set reaction coord
  # starting from previous, optimized geom, incr HG_OG by delta_rc/2, decr OG_C by delta_rc/2, moving
  #  only HG and C, not any other atoms
  if prev_rc is not None:
    trypsin.bond((og_195, hg_195), 0.5*(rc - prev_rc), rel=True, move_frag=False)
    trypsin.bond((og_195, c_5), -0.5*(rc - prev_rc), rel=True, move_frag=False)
    qmmm.prefix = tmpfolder + "/step_1_rc_%.3f" % rc
    res, r = optimize(trypsin, fn=qmmm.EandG, coords=hdlc, mon=optim_mon_qmmm, ftol=2E-07)
  else:
    r = trypsin.r
  if prev_rc is None or res.success:
    # last call to qmmm.EandG() by optimization may not be for final optimized geometry, but
    #  call to qmmm.energy() here ensures that log files are for final geom
    trypsin.r = r
    E = qmmm.EandG(trypsin, dograd=False)
    # should have some way to get this from qmmm ... maybe just qmmm.gen_filename('mm', 'input')
    mm_inp = "%s_%03dmm.xyz" % (qmmm.prefix, qmmm.iternum)
    respt = (rc, dict(E=E, HG_OG=trypsin.dist(hg_195, og_195), OG_C=trypsin.dist(og_195, c_5), geom=mm_inp))
    results.append(respt)
    with open(res_file, 'a') as f:
      f.write('results.append(%r)\n' % (respt,))
    print "Geom opt found energy %f Hartrees at reaction coord %.3f Ang" % (E, rc)
  else:
    print "Geom opt failed at reaction coord %.3f Ang!" % rc
  prev_rc = rc


# so we can stick inactive code below ... I'd like to avoid indenting script so it can be pasted into interpreter
raise Exception("quit script")

## Next: step 2 of reaction, then try NEB or similar instead of reaction coordinate and compare results
# - also should do something like dimer method that yields transition state directly (to get accurate barrier)
#  and consider what more detailed path is good for ... mostly just providing a qualitative description of mechanism?
# Refs:
# - https://gitlab.com/ase/ase/blob/master/ase/neb.py
# - https://github.com/eljost/pysisyphus
# - growing string method? https://github.com/ZimmermanGroup/molecularGSM
# We'll probably want to make notes on optimization methods ... and write our own optimizer - see optimize.py
# - could we use hessian for lone residues to approximate (sparse/block diag) Hessian?

# - then try polarizable MM force field; see https://github.com/kratman/LICHEM_QMMM and http://cascam.unt.edu/docs/CompChemMeet_07-19-17.pdf

## Misc analysis

# Post-minimization analysis:
# r_aligned = align_atoms(r_prep, r_mmopt, sel=select_atoms(trypsin, "protein and name == 'CA'"))
# RMSD per residue; largest sidechain orientation change (RMSD after CM alignment)
#[calc_RMSD(r_mmopt, r_prep, res.atoms) for res in trypsin.residues if not res.het]
#[calc_RMSD(r_mmopt[res.atoms] - np.mean(r_mmopt[res.atoms], axis=0), r_prep[res.atoms] - np.mean(r_prep[res.atoms], axis=0)) for res in trypsin.residues if not res.het]
# ... nothing stands out

# Sanity check for electron density grid:
# np.sum(mol.e_vol)*np.prod(((mol.mo_extents[1] - mol.mo_extents[0])/ANGSTROM_PER_BOHR)/[x-1 for x in np.shape(mol.e_vol)]) should equal number of electrons

# Inspecting raw and corrected QM grad (incl link atoms and all charges, incl. M1-M2 dipole charges):
# Gqm = qm000.grads[-1]/ANGSTROM_PER_BOHR; Gcorr = np.zeros_like(Gqm)
# Gcorr[q0idx:] =[ Gqm[jj+q0idx] + ANGSTROM_PER_BOHR*a1.qmq*np.sum([ a2.qmq*(a1.r - a2.r)/norm(a1.r - a2.r)**3 for kk, a2 in enumerate(charges) if kk != jj ], axis=0) for jj,a1 in enumerate(charges) ]
# Vqm = [qm000.atomcoords[0], qm000.atomcoords[0] + 10*Gqm]
# Vcorr = [qm000.atomcoords[0], qm000.atomcoords[0] + 10*Gcorr]


## Deferred work

# Note that ASP A 189 is also of interest for substrate binding

## HIS protonation

if 0:
  # Tinker adds both HD1 and HE2 hydrogens to HIS residues, giving +1 charge ... we need to remove one
  # remove HE2 from HIS A 57, since this position receives HG from SER 176 (A 195) during reaction
  trypsin.remove_atoms("pdb_resnum == 57 and name == 'HE2'")

  # for other HIS, remove HD1 or HE2, depending on which results in lower energy
  # - this can be made into a generic fn
  qmmm.prefix = tmpfolder + "/his_H"
  E_h_his = {}
  for resnum, res in enumerate(trypsin.residues):
    if res.name == 'HIS' and res.pdb_num != '57':
      print("Determining optimal protonation for HIS %s %s..." % (res.chain, res.pdb_num))
      E_h_his[resnum] = []
      for ii in res.atoms:
        atom = trypsin.atoms[ii]
        if atom.name in ['HD1', 'HE2']:
          # temporarily clear mmtype and remove from N atom mmconnect
          mmtype, atom.mmtype = atom.mmtype, None
          trypsin.atoms[atom.mmconnect[0]].mmconnect.remove(ii)
          E_h_his[resnum].append( Bunch(atom=ii, E=qmmm.energy(docomp=0)) )
          # restore H atom
          atom.mmtype = mmtype
          trypsin.atoms[atom.mmconnect[0]].mmconnect.append(ii)

  # currently no difference in efficiency between removing one atom at time vs. multiple
  for resnum, E_h in E_h_his.iteritems():
    E_h.sort(key=lambda b: b.E)
    trypsin.remove_atoms([b.atom for b in E_h[1:]])


## Explicit solvation ... not used for now; fairly generic, so maybe move to prepare.py eventually
import chem.data.test_molecules as test_molecules

subprocess.check_call(
    "echo parameters %s > tinker.key && $TINKER_PATH/pdbxyz 1MCT.pdb ALL ALL" % TINKER_FF, shell=True)

tinker_mm_key = \
"""parameters     %s

rattle water
ewald
archive
""" % TINKER_FF

# Solvation
# - explicit solvation seems to be preferred for most cases ... but we could use implicit solvation if
#  performance is an issue
# - how to constrain? periodic boundary conditions? freeze outer waters? spherical trapping potential/wall
#  (Tinker `wall` keyword)? ... for MD, I think periodic BCs are the only real option for explicit solvent
# for more advanced prep see, e.g., https://github.com/mcubeg/packmol
# normally, we should be able to use a preprepared water box, trimming to desired size
# minimizing then tiling a smaller water box is much faster than minimizing large box ... but minimization
#  time is still negligible compared to MD equilibriation time

tip3p = load_molecule(test_molecules.water_tip3p_xyz)
tip3p.residues = [Residue(name='HOH', atoms=[0,1,2])]
# Tinker 7 Amber96 TIP3P atom types
tip3p.mmtype = [2001, 2002, 2002]
# ensure centered
tip3p.r = tip3p.r - center_of_mass(tip3p)

# for water
molar_mass = 18.01488 # g/mol ... just sum of atomic weights
density = 0.9982 # g/cm^3 at 20C
water_cube_side = 1E8*(molar_mass/(AVOGADRO*density))**(1/3.0)  # in Ang
# trypsin is globular (approx 5x5x5 nm), so let's just make box cubic
# Tinker periodic box appear to be centered at origin
mol_extents = trypsin.extents(pad=12.0)
max_extent = np.max(np.abs(mol_extents))
nwaters_side = int(np.ceil(2*max_extent/water_cube_side))
box_axis = nwaters_side*water_cube_side
abc_axes = (box_axis,)*3
extents = np.array([[-0.5*box_axis]*3, [0.5*box_axis]*3])
solvent = tile_mol(tip3p, water_cube_side, [nwaters_side]*3, rand_rot=True, rand_offset=0.2*water_cube_side)

# solvent_box() places molecules completely randomly, which can break minimization due to two atoms ending up
#  very close ... to fix this, first run minimization with Tinker keyword `chargeterm none`
# alternative approach with solvent_box()
#vol_cm3 = np.prod(extents[1] - extents[0])*1E-24  # 1 Ang^3 = 1E-24 cm^3
#nwaters = AVOGADRO*density*vol_cm3/molar_mass
#solvent = solvent_box(tip3p, nwaters, extents)

# we could consider combining write_tinker_xyz, write key, run Tinker, parse_xyz into a fn that takes and
#  returns a Molecule
write_tinker_xyz(solvent, "solvent.xyz", "TIP3P Box")

with open("solvent.key", 'w') as f:
  f.write(tinker_mm_key)
  f.write("a-axis %.3f\nb-axis %.3f\nc-axis %.3f\n" % abc_axes)

# Tinker `dynamic` options:
# - timestep: 2 fs, assuming bond lengths constrained
# - total time: try 50K steps (100ps) ... looks like 1-2 ps/hour :(
# - write snapshots every 1 ps(?)
# - `rattle water` to fix water bond lengths and angle
# - `ewald`?
# - `archive` to write snapshots to single file instead of .xyz_1, .xyz_2, ...
# - Tinker chooses cutoffs of 9 Ang for most interactions for periodic system; cutoffs should be less than
#  half the smallest box dimension and less than 2x (or 1x?) the padding between protein and box edge so that
#  protein doesn't see its periodic image!
# Some refs:
# - http://biomol.bme.utexas.edu/tinkergpu/index.php/Tinkergpu:Tinker-tut
# - http://chembytes.wikidot.com/tinker-s-wiki
# - http://www.bevanlab.biochem.vt.edu/Pages/Personal/justin/gmx-tutorials/index.html (Gromacs)
# - http://ringo.ams.sunysb.edu/index.php/MD_Simulation:_Protein_in_Water (Gromacs)

# Tinker `minimize` converges slower but more reliably, and can handle larger systems than newton` or `optimize`
# Tinker `minimize` arguments: RMS grad per atom for termination (kcal/mol)
# Tinker `dynamic` args (assuming NPT ensemble): <num steps> <step length (fs)> <time between dumps (ps)>
#   <ensemble - 4 for NPT> <temperature (K)> <pressure (atm)>
subprocess.check_call("$TINKER_PATH/minimize solvent.xyz 1.0 && $TINKER_PATH/dynamic solvent.xyz_2 5000 2.0 1 4 298 1.0", shell=True)

equlib_solvent = parse_xyz("solvent.xyz_3")
solvent = setattrs(solvent, r=equilb_solvent.r)

# ~3x3x3 Ang per water in solvent
# - also, maybe run NPT equilb. for a few different values of d and see which produces smallest change in volume
# Use Na+/Cl- to neutralize solvated system; mmtypes are for Tinker 7 Amber96
na_plus = Atom(name='Na+', znuc=11, r=[0,0,0], mmq=1.0, mmtype=2004)
cl_minus = Atom(name='Cl-', znuc=17, r=[0,0,0], mmq=-1.0, mmtype=2012)
solvated = solvate(trypsin, solvent, d=2.0, ion_p=na_plus, ion_n=cl_minus)

write_tinker_xyz(solvated, "solvated.xyz")

# freeze protein and crystal waters
inactive = range(trypsin.natoms)
with open("solvated.key", 'w') as f:
  f.write(tinker_mm_key)
  f.write("a-axis %.3f\nb-axis %.3f\nc-axis %.3f" % abc_axes)
  f.write("\ninactive " + " ".join(["%d" % x for x in inactive]) + "\n\n")

# run NPT equilibriation after solvation
subprocess.check_call("$TINKER_PATH/minimize solvated.xyz 1.0 && $TINKER_PATH/dynamic solvated.xyz_2 5000 2.0 1 4 298 1.0", shell=True)

# need MM charges for QM/MM - should be the same for Amber FF, but in theory could change for a polarizable FF
equilib_solvated = parse_xyz("solvated.xyz_3", charges='generate')
equilib_system = setattrs(solvated, r=equilib_solvated.r, mmq=equilib_solvated.mmq)
