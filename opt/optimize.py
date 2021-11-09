import time
import numpy as np
import scipy.optimize
from ..molecule import calc_RMSD
from .coords import *


def moloptim_mon(r, r0, E, G, Gc, timing={}):
  rmsd = calc_RMSD(r0, r)
  # note that |proj g| printed by minimize is max(abs(grad)) ... but won't print if verbose < 0
  rmsG, rmsGc = np.sqrt(np.sum(G*G)/len(G)), np.sqrt(np.sum(Gc*Gc)/len(Gc))  # RMS grad per atom
  maxG, maxGc = np.max(np.abs(G)), np.max(np.abs(Gc))  # largest grad component - for gtol
  df = (timing.get('Eprev', E) - E)/max(abs(E), abs(timing.get('Eprev', E)))  # for ftol
  # \u212B is symbol for Angstrom
  print("[{:3d}] E: {:.6f} H ({:.2E}), G (H/\u212B): rms {:.6f}, max {:.6f} ({:.6f}, {:.6f}), RMSD: {:.3f} \u212B, Time: {:.3f}s ({:.3f}s)".format(timing.get('neval', 0), E, df, rmsG, maxG, rmsGc, maxGc, rmsd, timing.get('total', 0), timing.get('calc', 0)))


def monitor(fn):
  """ generic optimization monitoring wrapper """
  this = Bunch(r0=None, rprev=0)
  def monitored(r, *args, **kwargs):
    E, G = fn(r, *args, **kwargs)
    if this.r0 is None: this.r0 = r
    rmsd = calc_RMSD(this.r0, r)
    # RMS grad per atom; note that |proj g| printed by minimize is max(abs(grad))
    rmsgrad = np.sqrt(np.sum(G*G)/len(G))
    print("***ENERGY: {:.6f} H, RMS grad: {:.6f} H/Ang, (max {:.6f} H/Ang), |step|: {:g}, RMSD: {:.3f} Ang".format(
        E, rmsgrad, np.max(np.abs(G)), norm(r - this.rprev), rmsd))
    this.rprev = r
    return E, G
  return monitored


def coordwrap(fn, coords, reinit=True, **kwargs0):
  """ wrap an EandG `fn` w/ `coords` object """
  def wrapped(S, **kwargs1):
    if not coords.update(S):
      if not reinit:
        raise ValueError('Stop')
      coords.init()  # reinit w/ last good coords
      raise ValueError('Coord breakdown')  # optimizer needs to reset grad
    r = coords.xyzs()
    E, G = fn(r, **(dict(kwargs0, **kwargs1) if kwargs0 else kwargs1))
    gS = coords.gradfromxyz(G)
    return E, gS

  return wrapped


# default gtol, ftol just copied from scipy L-BFGS options
def moloptim(fn, r0=None, S0=None, mol=None, fnargs={}, coords=None, gtol=1E-05, ftol=2E-09, optimizer=None,
    mon=moloptim_mon, vis=None, raiseonfail=True, verbose=True, maxiter=1000):
  """ Given `fn` accepting r array and additional args `fnargs` and returning energy and gradient of energy,
    find r which minimizes energy, starting from `r0`.  Internal coordinates, e.g., can use used by passing
    appropriate object for `coords`; if coords are already inited, initial point can be set by S0 instead of
    r0. `verbose`=1: print every iteration; 0: call `mon` every iter, print optimization status at start and
    finish; -1: only call `mon` (every iteration); -2: only call `mon` at start and finish
  """
  if S0 is not None:
    assert r0 is None, "Cannot pass both r0 and S0"
    coords.update(S0)
    r0 = np.array(coords.xyzs())
  else:
    r0 = mol.r if r0 is None else r0
    coords = XYZ(r0) if coords is None else coords
    coords.init(r0)
  objst = Bunch(updateok=False, neval=0)
  res = None

  def objfn(S):
    t0 = time.time()
    isnewS = np.any(coords.active() != S)
    if coords.update(S):
      if isnewS: objst.updateok = True  # can't assign to outer vars!!! - fixed in python 3 w/ nonlocal keyboard
      r = coords.xyzs()
      t1 = time.time()
      E, G = fn(r, **fnargs) if mol is None else fn(mol, r, **fnargs)
      t2 = time.time()
      gS = coords.gradfromxyz(G)
      # idea of Gc is to get cartesian grad excluding constraints
      Gc = coords.gradtoxyz(gS) if hasattr(coords, 'gradtoxyz') else gS  #np.zeros(1)
      # call monitor fn, e.g. to print additional info
      if mon and (verbose >= -1 or objst.neval == 0):
        mon(r, r0, E, G, Gc, dict(objst, coords=t1-t0, calc=t2-t1, total=t2-t0))
      if vis:
        vis.refresh(r=r, repaint=True)
      objst.neval = objst.neval + 1
      objst.Eprev = E
      return E, np.ravel(gS)
    else:
      print("Coord breakdown for S = \n%r" % S)
      # optimizer must restart
      raise ValueError('Coord breakdown')

  if verbose > 0:
    print("optimize started at %s" % time.asctime())
  while maxiter > 0:
    try:
      if optimizer is not None:
        res = optimizer(objfn, np.ravel(coords.active()), gtol=gtol, ftol=ftol, maxiter=maxiter)
      else:
        res = scipy.optimize.minimize(objfn, np.ravel(coords.active()),
            jac=True, method='L-BFGS-B', options=dict(iprint=int(verbose), gtol=gtol, ftol=ftol, maxiter=maxiter))
      maxiter -= res.nit
      break
    except ValueError as e:
      if e.message == 'Stop':  # allow fn to stop optim based on some internal criteria
        break
      if e.message != 'Coord breakdown':
        raise
      if not objst.updateok:
        print("Coord breakdown on first iteration - unable to continue")
        break
      objst.updateok = False
      coords.init()  # reinit with last good cartesians

  if verbose > 0:
    print("optimize finished at %s" % time.asctime())
  elif verbose < -1 and mon:
    verbose = -1
    objfn(res.x)
  if res and res.success:
    coords.update(res.x)
  elif raiseonfail:
    raise ValueError('optimize() failed')
  return res, np.array(coords.xyzs())  # copy


## Global optimization
# - Also try scipy.optimize.basinhopping and scipy.optimize.anneal
# - Is there any reason to use simulated annealing if we have gradient available?
# - possible termination criteria for hopping: rel change in E < ftol
# - what about avoiding previous minima when hopping?
# - Sniffer algorithm provides another way to utilize grad for global opt: 10.1016/0021-9991(92)90271-Y

# exp( -beta*sqrt(E[dE**2]) ) ~ acceptance
# dE = dR.*grad ; take dR = scale*X/grad  (X = rand - 0.5), so dE = scale*sum(X)
# sqrt(E[dE**2]) = scale * sqrt(E[X**2]) = N*scale/sqrt(12) for uniform or N*scale*sigma for Gaussian
# for Gaussian, scale = 1 -> sigma = -ln(accept)/beta/N

# use a class to allow access to results after Ctrl+C and restart
class MCoptim:

  def __call__(self, fn, r0=None, fnargs={}, coords=None, T0=300, kB=BOLTZMANN, accept=0.5, G0=1E-4, adjstep=10,
      maxiter=1000, nstepped=None, anneal=False, thermal=False, hop=False, hopcoords=None, hopargs={}):
    """ Metropolis Markov Chain Monte Carlo / basin hopping (MCMC with minimize after each move) for global
     optimization and thermal ensemble generation
      fn, r0, fnargs: objective fn, starting point, additional args
      coords: coord object to use for steps
      T0: initial temperature for Boltzmann factor (default 300K)
      kB: sets units via value of Boltzmann constant (default Hartree/K)
      accept: target acceptance rate
      G0: inverse step size scaling factor
      adjstep: number of iterations between step size adjustment - set to False or -1 to disable
      nstepped: number of coordinates to step at once (default: all)
      anneal: decrease temperature to 0 at maxiter
      thermal: return thermal ensemble if set, otherwise all generated points
      hop: enable basin hopping
      (minimize fn after each step - requires gradient)
      hopcoords, hopargs: passed to moloptim for basin hopping
     returns: sorted list of fn values and args
    """
    r0 = self.Rs[-1] if r0 is None else r0
    coords = XYZ(r0) if coords is None else coords
    coords.init(r0)
    hopcoords = coords if hopcoords is None else hopcoords
    hopargs = dict(dict(gtol=2E-04, ftol=2E-07, verbose=False, maxiter=30), **hopargs)
    beta = 1.0/(kB*T0)  # ~1E3 Hartree^-1 at 300K
    scale = 1.0
    res = fn(r0, **fnargs)
    E, G = res if safelen(res) > 1 else (res, 1.0)
    if not hasattr(self, 'Es'):
      self.Es, self.Rs = [], []  #[E], [r0]
    Emin = E
    r = r0
    naccept = 0

    print("mcoptim started at %s" % time.asctime())
    for ii in range(maxiter):
      # adaptive step size - bigger steps if acceptance rate too high, smaller steps if too low
      if adjstep and ii%adjstep == adjstep - 1:
        scale, oldscale = scale*0.8 if naccept/float(ii) < accept else scale*1.25, scale
        print("** step scale changed from %f to %f" % (oldscale, scale))

      if anneal:
        beta = 1.0/(kB*T0 * (1 - ii/float(maxiter))**3)  # many, many other cooling ramp options

      # step: gradient isn't really useful for large steps! scan individual vars separately to get scale for each?
      # G0 = 1E-4 seems OK for both Cartesian and diheds-only DLC (for accept = 0.5, T0 = 300)
      #dr = -scale*np.log(accept)/beta/np.size(r0)/G0 * np.random.normal(size=np.shape(r0))  #*sqrt(12) for uniform
      #dS = coords.gradfromxyz(dr)
      S0 = coords.active()  # or only after init() (when update() fails)? ... not for MCMC!
      # MCMC typically steps one particle at time, since accepting many-particle steps requires tiny step size
      ndS = nstepped if nstepped else np.size(S0)
      idx0 = nstepped*int(np.random.rand()*len(S0)/nstepped) if nstepped else None
      dS0 = -scale*np.log(accept)/beta/G0/ndS * np.random.normal(size=ndS)
      dS = setitem(np.zeros_like(S0), slice(idx0,idx0+nstepped), dS0) if nstepped else dS0
      if not coords.update(S0 + dS):
        coords.init()
        S0 = coords.active()
        if not coords.update(S0 + dS):
          print("Iteration %d: Coord update failed" % ii)
          continue
      r1 = coords.xyzs()

      if hop:
        res, r1 = moloptim(fn, r1, fnargs=fnargs, coords=hopcoords, raiseonfail=False, **hopargs)
        E1, G1 = res.fun, res.jac
      else:
        res = fn(r1, **fnargs)
        E1, G1 = res if safelen(res) > 1 else (res, 1.0)

      # accept always if E < Eprev or with probability e^{-beta*(E-Eprev)} otherwise
      if E1 <= E or np.exp(-beta*(E1 - E)) > np.random.rand():
        E, G, r = E1, G1, r1
        Emin = min(E, Emin)
        naccept += 1
      elif not coords.update(S0):  # newX = r
        coords.init(r)  #print("Iteration %d: coords reinitialzed" % ii)
      # record accepted point (new or prev) point to generate thermal ensemble or new point otherwise (for hop)
      self.Es.append(E if thermal else E1)
      self.Rs.append(r if thermal else r1)
      print("iteration %d: f = %f; fmin = %f; acceptance = %f" % (ii, E1, Emin, naccept/float(ii+1)))

    return self.results()


  def results(self):
    Eorder = argsort(self.Es)
    return np.array(self.Es)[Eorder], np.array(self.Rs)[Eorder]
