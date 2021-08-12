# L-BFGS optimizer
# I didn't want to do this (really!), but for NEB, dimer method w/ L-BFGS, etc., we need optimizer that can
#  work w/o function value, only gradient ... maxls=0 (or 1) w/ scipy.optimize L-BFGS doesn't work
# If fn value is available, no reason not to use scipy LBFGS, and no indication I can improve on that code
# See test/opt_test.py for tests and misc code (e.g. update of Hessian instead of inverse)

# LBFGS history length defaults:
#  pytorch, UT SDLBFGS 100; scipy, pDynamo 10; NLPy 5, ? num DOF
# First step (to get initial curvature estimate along steepest descent direction):
#  tsase: 1E-5 Ang; tsase, ase: H0 = 1/70; pele: 0.1; pDynamo: H0 = 1/g.*g; many others: H0 = 1
# gtol:
#  Tinker default: 0.01 kcal/mol/Ang = 1.6E-5 Hartree/Ang; GAMESS 1E-4 Hartree/Bohr = 5E-5 Hartree/Ang
# ftol:
#  Chemical accuracy: ~1kcal/mol ~ 0.001 Ha
# maxstep/trust radius:
#  GAMESS: 0.15 - 0.25 Ang

# Refs:
# - gitlab.com/ase/ase/-/blob/master/ase/optimize - BFGS, LBFGS, FIRE (SD + inertia), etc.; More-Thuente line search
#  - TSSE/TSASE - theory.cm.utexas.edu/svn/tsase/tsase/optimize - succinct code, similar to ASE
# - github.com/pytorch/pytorch/blob/master/torch/optim/lbfgs.py - cubic line search
# - github.com/mattbernst/pDynamo-mirror/tree/master/pCore-1.9.0/pCore - LBFGS, More-Thuente, etc.; doesn't use numpy
# - gist.github.com/jiahao/1561144 - LBFGS, BFGS, line search, etc.
# - github.com/rpmuller/pistol/blob/d411d0876f339b470f0754a4817b5a6a7853ed62/Pistol/optimize.py - BFGS, quadratic + cubic line search, GDIIS
# - optbench.org/ - comparision of optimization methods for minima, TS, and reaction paths from UT group
#  - paper: 10.1021/ct5008718
#  - their SDLBFGS (no line search!) wins for minimization; larger L-BFGS history (100) and larger max step
#   seem to do better; github.com/zadorlab/sella/ wins for TS
# More-Thuente line search uses quadratic and cubic interpolation along step direction until point satisfying
#  Wolfe conditions is found; Hager-Zhang is similar to More-Thuente
# Standard notation (e.g. Nocedal, Wright), w/ independent variables r, gradient g:
#  B: Hessian, H: inverse Hessian; s_k = r_k - r_{k-1} (step); y_k = g_k - g_{k-1}; Newton step is s = -H.*g
# Others:
# - SciPy, LBFGS: Fortran (fancy Nocedal code), not restartable; BFGS: Python
# - github.com/azag0/pyberny - succinct Python impl of Berny algorithm (geom opt from Gaussian)
# - MMTK: CG, SD
# - PyQuante: SD
# - nlpy: LBFGS, others - https://sourceforge.net/p/nlpy
# - DL-FIND: Fortran
# - OpenOpt: abandoned, mostly wrappers around scipy, etc
# See test/ts_test.py for TS optimization refs

# Misc ideas:
# - ASE BFGS updates B (instead of H) and doesn't enforce it be positive-definite; instead get eigenvalues
#  and eigenvectors every iteration and reverses sign of negative eigenvalues

from collections import deque
import numpy as np
from ..basics import Bunch  # or just use dict() instead


# BFGS update of inv. Hessian
class BFGS:
  def __init__(self, H0=1/70.0):
    self.H0 = H0
    self.H = H0
    self.g0 = None

  def reset(self, H0=None, g0=None):
    self.H = self.H0 if H0 is None else H0
    self.g0 = g0

  def step(self, dr, g):
    if self.g0 is not None:
      s, y = dr, g - self.g0  # standard notation
      invrho = np.dot(s,y)
      if invrho > 0:
        if np.ndim(self.H) == 0:
          self.H = invrho/np.dot(y,y)  # replace H0 (if scalar) based on curvature from first point
        rho = 1/invrho
        I = np.eye(len(s))
        A1 = I - np.outer(s,y)*rho
        A2 = I - np.outer(y,s)*rho
        self.H = np.dot(A1, np.dot(self.H, A2)) + np.outer(s,s)*rho

    self.g0 = g
    return -np.dot(self.H,g)


# symmetric-rank-one update of inv. Hessian - does not guarantee positive definite Hessian
#s, y = dr, g - self.g0  # standard notation
#z = s - np.dot(self.H, y)
#self.H = self.H + np.outer(z, z)/np.dot(z, y)


def lbfgs_mult(g, s, y, rho):
  """ calculate H .* g where H is inverse Hessian estimate from s, y, rho """
  m = len(s)
  a = np.empty(m)
  q = np.array(g)  # copy!!!
  for i in range(m-1, -1, -1):
    a[i] = rho[i] * np.dot(s[i], q)
    q -= a[i] * y[i]

  # H0 = 1/C, where C approximates curvature at last step - see Nocedal, Wright eq. 7.20
  # note that if we use first step [0] instead of last [-1], LBFGS should match BFGS for first maxhist iters
  H0 = 1/(rho[-1]*np.dot(y[-1], y[-1]))  #np.vdot(s[-1], y[-1])/np.vdot(y[-1], y[-1])
  z = H0 * q
  for i in range(m):
    b = rho[i] * np.dot(y[i], z)
    z += s[i] * (a[i] - b)

  return z


# L-BFGS update of inv. Hessian
class LBFGS:
  def __init__(self, H0=1/70.0, maxhist=10):
    self.H0 = H0
    self.g0 = None
    # deque automatically pops leftmost element ([0]) when len exceeds max
    # CPython deque uses linked list of 64 element blocks, so access is only really O(N) for very large N
    self.s = deque([], maxhist)  # steps (dr)
    self.y = deque([], maxhist)  # gradient differences (g_k - g_{k-1})
    self.rho = deque([], maxhist)  # 1/(y .* s)

  def reset(self, H0=None, g0=None):
    self.s.clear()
    self.y.clear()
    self.rho.clear()
    self.g0 = g0
    self.H0 = self.H0 if H0 is None else H0

  def step(self, dr, g):
    if self.g0 is not None:
      s = np.array(dr)  # copy
      y = g - self.g0
      invrho = np.dot(s,y)
      # do not store updates with negative curvature to keep H positive definite
      if invrho > 0:
        self.s.append(s)
        self.y.append(y)
        self.rho.append(1/invrho)
    self.g0 = g
    return -lbfgs_mult(g, self.s, self.y, self.rho) if self.s else -np.dot(self.H0, g)  # allow matrix or scalar H0


def gradoptim(fn, x0, stepper=None, gtol=1E-05, ftol=2E-09, wolfe1=1e-4, wolfe2=0.9, maxiter=1000, maxdr=1.0):
  """ Find stationary point of objective function fn in negative gradient direction
    fn: objective function returning scalar value and gradient
    x0: initial guess
    stepper: object with method step(dr, gradient) -> next dr; defaults to LBFGS Hessian update
    gtol: optimization terminated if max(abs(G)) < gtol
    ftol: optimization terminated if abs(delta f)/f < ftol (if ftol > 0)
    wolfe1: constant for first Wolfe condition: f(r0 + dr) < wolfe1*g0.*step (ignored unless ftol > 0)
    wolfe2: constant for second Wolfe condition: abs(g.*step) < wolfe2*abs(g0.*step)
    maxiter: maximum number of iterations (== maximum number of fn evaluations)
    maxdr: largest allowed step for any single component
  """
  stepper = LBFGS() if stepper is None else stepper
  rinshape = np.shape(x0) if np.ndim(x0) > 1 else None
  r = np.ravel(x0)
  dr, f0, g0 = 0, None, 0

  for iter in range(maxiter):
    # note that reshaping here usually won't require any copying of data
    rin = np.reshape(r, rinshape) if rinshape else r
    try:
      fout, gout = fn(rin)
    except ValueError as e:
      if e.message == 'Stop':  # allow fn to stop optim based on some internal criteria
        return Bunch(x=prev_rin, nextx=rin, fun=fout, grad=gout, jac=gout, success=False, nit=iter)
      if e.message != 'Coord breakdown':
        raise
      stepper.reset()  # also r -= dr to go back to last known working coords?
      continue
    except KeyboardInterrupt:
      print("Optimization stopped by KeyboardInterrupt; returning state at iteration %d" % iter)
      return Bunch(x=prev_rin, nextx=rin, fun=fout, grad=gout, jac=gout, success=False, nit=iter)
    g = np.ravel(gout) if np.ndim(gout) > 1 else np.asarray(gout)
    prev_rin = rin

    # check for termination
    gterm = np.max(np.abs(g)) < gtol  # other options: np.linalg.norm(g) < gtol*np.size(r)
    fterm = f0 and ftol > 0 and abs(fout - f0)/max(abs(fout), abs(f0)) < ftol
    if gterm or fterm:
      print("Optimization completed:%s%s" % (
          (" max |g| < %g" % gtol) if gterm else "", (" df/f < %g" % ftol) if fterm else "" ))
      return Bunch(x=rin, fun=fout, grad=gout, jac=gout, success=True, nit=iter)

    # check first Wolfe condition (aka Armijo condition) - sufficient fn decrease
    if iter > 0 and ftol > 0 and fout > f0 + wolfe1*np.dot(g0, dr):
      print("Bad step: delta f = %f" % (fout - f0))
      # failure of this condition usually accompanied by failure of 2nd condition, so don't do anything here
      #stepper.reset(g);  g = g0;  dr = -dr

    # check second (strong) Wolfe condition
    sg0 = np.dot(dr,g0)
    sg1 = np.dot(dr,g)
    if iter > 0 and np.abs(sg1) > wolfe2*np.abs(sg0):
      print("Curvature condition not satisfied: s.g0: %f, s.g: %f" % (sg0, sg1))
      if sg0 < 0 and sg1 > 0:  # sg1 > sg0:
        # quadratic interpolation to estimate stationary point between r and prev r; reseting with the step
        #  from r to interpolated point seems to work better than the smaller step from prev r
        stepper.reset(g0=g)
        dr = -sg1*dr/(sg1 - sg0)  # for step from prev r: dr = -dr*sg0/(sg1-sg0)
        r = r + dr  # step is always smaller than previous, so no need to check maxdr
        f0, g0 = fout, g
        continue

    # check for negative curvature; any advantage to doing a line search here?
    if iter > 0 and sg1 < sg0:  #np.dot(dr,y) <= 0:  # s.y = s.g - s.g0 = sg1 - sg0
      print "Reseting due to negative curvature"
      y = g - g0
      stepper.reset(H0=-np.dot(dr,y)/np.dot(y,y))  # let's reverse sign of curvature

    f0, g0 = fout, g
    s = stepper.step(dr, g)
    dr = s if np.max(np.abs(s)) < maxdr else s*maxdr/np.max(np.abs(s))
    # this should never happen as long as we keep H positive definite
    #if np.vdot(dr, g) > 0: print("Step direction >90 deg from steepest descent direction")
    r = r + dr

  return Bunch(x=rin, fun=fout, grad=gout, jac=gout, success=False, nit=maxiter)
