import os, subprocess
from .basics import *

# Refs:
# - github.com/choderalab/pymbar - we should use pymbar for actual work
# - github.com/TinkerTools/tinker/blob/release/source/bar.f
# - arxiv.org/pdf/2008.03067.pdf - good and practical overview
# TODO:
# - enthalpy and entropy (see pymbar or Tinker)


# MM parameters for ligand
# - proper way to do this is w/ AmberTools
# - for this first try, we'll assign GAFF atoms types with Molden, use set_mm_params() to get GAFF parameters,
#  set charges with RESP, then write all params to Tinker .key

# if we install Psi4, try github.com/lilyminium/psiresp or github.com/cdsgroup/resp or github.com/psi4/psi4numpy
def resp_prepare(mol, constr=[], avg=[], netcharge=0.0, maxiter=1000):
  """ run HF/6-31G* ESP charge fitting with restraints toward q = 0 on znuc > 1 atoms, with sets of atoms
    `constr` constrained to have same charge and charges of sets of atoms `avg` made equal by averaging after
    fit; molecule is first relaxed for up to maxiter iterations
  """
  from .molecule import init_qmatoms
  from .opt.optimize import moloptim
  from .qmmm.qmmm1 import QMMM, pyscf_EandG
  from .qmmm.resp import resp
  # RESP charges
  qmbasis = '6-31G*'  # standard basis for RESP
  qmatoms = init_qmatoms(mol, '*', qmbasis)
  if maxiter > 0:
    qmmm = QMMM(mol, qmatoms=qmatoms, qm_charge=netcharge, prefix='no_logs')
    res, r_qm = moloptim(qmmm.EandG, mol=mol, raiseonfail=False, maxiter=maxiter)
    mol.r = r_qm
  _, _, scn = pyscf_EandG(mol, qm_charge=netcharge)
  mmq = resp(scn.base, equiv=constr, restrain=(mol.znuc > 1)*0.1, netcharge=netcharge)
  for a in avg:
    mmq[a] = np.mean(mmq[a])  # average charges of chemically equiv. atoms
  mol.mmq = mmq
  mol.prev_pyscf = scn
  return mol


def gaff_prepare(mol, constr=[], avg=[], mmtype0=801):
  """ load GAFF parameters for mol with atom names as GAFF atom types, setting MM atom types to sequential
    values starting from `mmtype0` (for Tinker) and run resp_prepare()
  """
  from .mm import load_amber_dat, set_mm_params
  # amber parm files can be obtained from, e.g., github.com/choderalab/ambermini
  gaff = load_amber_dat(DATA_PATH + '/amber/gaff.dat')
  mol.mmtype = [a.name.upper() for a in mol.atoms]
  set_mm_params(mol, gaff)
  mol.mmtype = np.arange(mol.natoms) + mmtype0
  return resp_prepare(mol, constr, avg)


def solvate_prepare(mol, ff, T0, pad=6.0, solute_res=None, solvent_chain=None, neutral=False, eqsteps=5000):
  from .molecule import Molecule, Atom
  from .model.prepare import water_box, solvate
  from .io.openmm import openmm, openmm_MD_context, openmm_load_params, UNIT
  # make a cubic water box
  side = np.max(np.diff(mol.extents(pad=pad), axis=0))
  solvent = water_box(side)
  # check radial distribution fn (min dist should be ~1.5 if overfill worked properly
  if 0:
    from chem.analysis import calcRDF
    bins,rdf = calcRDF(solvent.r[solvent.znuc > 6], solvent.pbcbox)
    plot(bins[:-1], rdf)

  na_plus, cl_minus = None, None
  if neutral:
    # Amber names for Na, Cl
    na_plus = Molecule(atoms=[Atom(name='Na+', znuc=11, r=[0,0,0], mmq=1.0)]).set_residue('Na+')
    cl_minus = Molecule(atoms=[Atom(name='Cl-', znuc=17, r=[0,0,0], mmq=-1.0)]).set_residue('Cl-')
    openmm_load_params(mol, ff=ff, charges=True)  # needed
  # any point in doing equilibriation before solute is added?
  mol.r = mol.r - np.mean(mol.extents(), axis=0)
  solvated = solvate(mol, solvent, d=2.0,
      solute_res=solute_res, solvent_chain=solvent_chain, ion_p=na_plus, ion_n=cl_minus)

  # short equilibriation (now using OpenMM instead of Tinker)
  solvated.r = solvated.r + 0.5*side  # OpenMM centers box at side/2 instead of origin
  if eqsteps is not None:
    ctx = openmm_MD_context(solvated, ff, T0)
    openmm.LocalEnergyMinimizer.minimize(ctx, maxIterations=100)
    if eqsteps > 0:
      ctx.getIntegrator().step(eqsteps)
    simstate = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    solvated.r = simstate.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
    solvated.pbcbox = np.diag(simstate.getPeriodicBoxVectors(asNumpy=True).value_in_unit(UNIT.angstrom))
    solvate.md_vel = simstate.getVelocities(asNumpy=True).value_in_unit(UNIT.angstrom/UNIT.picosecond)
  return solvated


# BAR (Bennett acceptance ratio) estimation of free energy difference
# original paper (Bennett 1976) is simple and clear: 10.1016/0021-9991(76)90078-4
def bar(dE0, dE1, dF0=None, maxiter=100, tol=1E-10, calc_std=False):
  """ fwd and rev energy differences dE0 and dE1 and optional initial guess dF0 should be in units of kB*T """
  n0, n1 = float(len(dE0)), float(len(dE1))
  C = dF0 if dF0 is not None else 0.5*(np.mean(dE0) - np.mean(dE1))
  for ii in range(maxiter):
    err = np.log(np.sum(1.0/(1 + np.exp(dE1 + C)))) - np.log(np.sum(1.0/(1 + np.exp(dE0 - C))))
    C = C + err
    if abs(err/C) < tol:
      break
  if not calc_std:
    return C - np.log(n1/n0)
  # estimate error
  f0, f1 = 1.0/(1 + np.exp(dE0 - C)), 1.0/(1 + np.exp(dE1 + C))
  var = np.sum(f0*f0)/np.sum(f0)**2 - 1.0/n0 + np.sum(f1*f1)/np.sum(f1)**2 - 1.0/n1
  return C - np.log(n1/n0), np.sqrt(var)


# bootstrap error estimation: resample data (w/ replacement) multiple times, calculate std over these runs
def bootstrap(fn, samps, niter=100):
  # if samps is always 1-D, we could do fn( np.random.choice(samps, len(samps)) )
  y = np.array([ fn(samps[ np.random.choice(len(samps), len(samps)) ]) for ii in range(niter) ])
  return np.mean(y), np.std(y)  #ddof=1 ?


def fep(dE):
  """ energy differences dE should be in units of kB*T """
  return -np.log( np.sum(np.exp(-dE))/len(dE) )  # Zwanzig formula


def fep_gauss(dE):
  """ lowest order (Gaussian distrib.) approx to FEP; dE should be in units of kB*T """
  return np.mean(dE) - 0.5*np.var(dE)


# enthalpy estimates - entropy is just dS = (dH - dF)/T ; not as reliable as free energy!
# ref: 10.1021/jp103050u , github.com/TinkerTools/tinker/blob/release/source/bar.f (w/ bootstrap error est.)
def fep_enthalpy(E0, E1):
  """ energies E0, E1 (both evaluated for state 0) should be in units of kB*T """
  return np.mean(E1 * np.exp(E0 - E1))/np.mean(np.exp(E0 - E1)) - np.mean(E0)  # eq. 6a


# worried there might still be a bug in here - doesn't agree super well w/ MBAR
def bar_enthalpy(E00, E01, E10, E11):
  """ Exy: potential energy fn x evaluated for ensemble generated w/ potential energy fn y """
  dE0, dE1 = E10 - E00, E11 - E01
  C = bar(dE0, -dE1)  # assumes both ensembles of equal length
  gp0, gm0 = 1.0/(1 + np.exp(dE0 - C)), 1.0/(1 + np.exp(-dE0 + C))
  gp1, gm1 = 1.0/(1 + np.exp(dE1 - C)), 1.0/(1 + np.exp(-dE1 + C))
  a0 = np.mean(gp0*E00) - np.mean(gp0)*np.mean(E00) + np.mean(gp0*gm0*dE0)
  a1 = np.mean(gm1*E11) - np.mean(gm1)*np.mean(E11) - np.mean(gp1*gm1*dE1)
  return (a0 - a1)/(np.mean(gp0*gm0) + np.mean(gp1*gm1))  # eq. 8


# pymbar calls g_ac "statistical inefficiency"; various other authors call g_ac or g_ac/2 the "integrated
#  autocorrelation time" - in any case, the variance of the mean (i.e. squared std error of the mean) is
#  var(x)/(len(x)/g_ac), so we can say we effectively have len(x)/g_ac independent samples
def int_autocorr_time(x):
  from scipy.signal import correlate
  x = np.asarray(x) - np.mean(x)
  acf = correlate(x, x, mode='full')[len(x)-1:]  # np.correlate doesn't use FFT and is thus super-slow
  #acf = np.fft.ifft( np.abs(np.fft.fft( np.hstack([ x, np.zeros(len(x)) ]) ))**2 ).real[:len(x)]
  ixz = np.argmax(acf < 0)  # autocorrelation is just noise by the time it crosses zero (before actually)
  fct = (1 - np.arange(1.0,ixz)/len(x))  # sometimes omitted since ~1 for ixz << len(x)
  t_ac = np.sum(fct * acf[1:ixz]/acf[0])  # this is what pymbar calls the autocorrelation time
  g_ac = 1 + 2*t_ac  # "statistical inefficiency" (pymbar) or "integrated autocorrelation time" (others)
  return g_ac
  # alternative windowing method (ref: Sokal)
  #gs = 1 + 2*np.cumsum(acf[1:]/acf[0])  # note that we don't bother with fct here
  #gidx = np.argmax(5.0*gs < np.arange(len(gs)))  # 4.0 - 10.0 is reasonable range for constant
  #return gs[gidx]


# from pymbar; Tinker uses bootstrap method to estimate FEP error
def fep_err(dE, g=None):
  """ estimate relative(???) error of FEP free energy for energy differences dE in units of kB*T """
  x = np.exp(-dE)  # (-dE - np.max(-dE))
  g = int_autocorr_time(dE) if g is None else g
  return np.std(x)/np.sqrt(len(x)/g)/np.mean(x)


def fep_results(dEup, dEdn, T0):
  """ print free energy change for fwd and rev energy differences dEup, dEdn, passed in units of kB * `T0` """
  beta = 1/(KCALMOL_PER_HARTREE*BOLTZMANN*T0)  # 1/kB/T in our units
  dFs_bar = np.array( [bar(dEup[ii], dEdn[ii+1], calc_std=True) for ii in range(len(dEup)-1)] )/beta
  dF_bar = np.sum(dFs_bar[:,0])
  dF_fepup = np.sum([fep(dEup[ii])/beta for ii in range(len(dEup)-1)])
  dF_fepdn = np.sum([fep(dEdn[ii+1])/beta for ii in range(len(dEdn)-1)])
  print("BAR: %f kcal/mol; FEP: %f kcal/mol; FEP(rev): %f kcal/mol" % (dF_bar, dF_fepup, dF_fepdn))
  print("BAR steps dF (kcal/mol): %s\nBAR steps ddF (kcal/mol): %s" % (dFs_bar[:,0], dFs_bar[:,1]))


# much faster to use Tinker bar option 1 to calculate energies and write to .bar files:
# dEup = E_{i+1}(r_i) - E_i(r_i), dEdn = E_i(r_i) - E_{i-1}(r_i) where i is lambda index
def tinker_fep(nlambda, T0, warmup=2, autocorr=False):
  dEup, dEdn = [], [0]
  beta = 1/(KCALMOL_PER_HARTREE*BOLTZMANN*T0)  # Tinker energies are in kcal/mol
  for ii in range(nlambda-1):
    if not os.path.exists("fe%02d.bar" % ii):
      subprocess.check_output([os.path.join(TINKER_PATH, "bar"),
          "1", "fe%02d.arc" % ii, str(T0), "fe%02d.arc" % (ii+1), str(T0), "N"])
    with open("fe%02d.bar" % ii, 'r') as f:
      dE = np.zeros(int(f.readline().split()[0]))
      for jj in range(len(dE)):
        l = f.readline().split()
        dE[jj] = beta*(float(l[2]) - float(l[1]))
      dEup.append(dE[warmup:])
      dE = np.zeros(int(f.readline().split()[0]))
      for jj in range(len(dE)):
        l = f.readline().split()
        dE[jj] = beta*(float(l[1]) - float(l[2]))
      dEdn.append(dE[warmup:])

  # check for independent samples
  if autocorr:
    for ii in range(nlambda-1):
      print("Integrated autocorr. time (%d): %f (fwd); %f (rev)"
          % (ii, int_autocorr_time(dEup[ii]), int_autocorr_time(dEdn[ii+1])))

  fep_results(dEup, dEdn, T0)  # already removed warmup samples


# calcRDF(mol.r[mol.znuc > 1]) to exclude hydrogens
def calcRDF(mol, pbcbox=None, nbins=200, maxdist=8):
  """ calculate radial pair distribution function, with optional periodic boundary conditions `pbcbox` """
  r = getattr(mol, 'r', mol)
  if pbcbox is not None:
    #dists = [ pbc_dist(r0, r[jj], a) for ii,r0 in enumerate(r) for jj in range(ii+1, len(r)) ]
    dr = (r[:,None,:] - r[None,:,:])[np.triu_indices(len(r), 1)]
    dists = np.linalg.norm(dr - np.round(dr/pbcbox)*pbcbox, axis=1)  # this supports dr > 2*pbcbox
    #dists = np.linalg.norm(np.min([np.abs(dr), np.abs(np.abs(dr) - pbcbox)], axis=0), axis=1)
    vol = np.prod(pbcbox) if np.size(pbcbox) > 1 else pbcbox**3
  else:
    from scipy.spatial.distance import pdist
    dists = pdist(r)
    #dists = [ norm(r0 - r[jj]) for ii,r0 in enumerate(r) for jj in range(ii+1, len(r)) ]
    # exclude bonded atoms?  exclude atoms on same residue (a.resnum != mol.atoms[jj].resnum)
    #[ mol.bond((ii,jj)) for ii,a in enumerate(mol.atoms) for jj in range(ii+1:mol.natoms) if jj not in a.mmconnect ]
    # estimate volume by rounding each position to nearest cell center, then count number of non-empty cells
    cell = 5.0  # Ang
    vol = cell**3 * len(np.unique(np.round(r/cell)*cell, axis=0))

  hist, bins = np.histogram(dists, range=(np.min(dists), maxdist), bins=nbins)  #dists[dists < maxdist]
  binvol = np.diff(4/3.0*np.pi*bins**3)
  # normalization: binvol * vol/N is expected number of atoms in the shell (bin) around a single reference atom;
  #  but list of pair distances includes shells for all N atoms, so we divide by another factor of N - or rather
  #  N/2 since we only count each pair once above; properly normalized RDF should go to 1 for large r
  N = len(r)
  norm = N/2 * N/vol
  return bins, (1.0/norm)*hist/binvol
