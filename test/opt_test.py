from chem.basics import *
from chem.opt.lbfgs import *
import scipy.optimize

# Move BFGS2 to lbfgs.py if we find a use for it (cursory testing shows inferior performance)

# BFGS updating Hessian instead of inverse
class BFGS2:
  def __init__(self, H0=1/70.0):
    self.B0 =  np.linalg.inv(H0) if np.ndim(H0) > 0 else 1.0/H0
    self.B = self.B0
    self.g0 = None

  def reset(self, H0=None, g0=None):
    self.B = self.B0 if H0 is None else (np.linalg.inv(H0) if np.ndim(H0) > 0 else 1.0/H0)
    self.g0 = g0

  def step(self, dr, g):
    if np.ndim(self.B) == 0:
      self.B = self.B * np.eye(len(g))
    if self.g0 is None:
      self.g0 = g
      return -np.dot(np.linalg.inv(self.B), g)
    s, y = dr, g - self.g0  # standard notation
    dg = np.dot(self.B, s)
    self.B = self.B + np.outer(y, y)/np.dot(s, y) - np.outer(dg, dg)/np.dot(s, dg)
    self.g0 = g
    #return -np.dot(np.linalg.inv(self.B), g)
    # RFO from pyberny
    rfo = np.vstack((np.hstack((self.B, g[:, None])), np.hstack((g, 0))[None, :]))
    D, V = np.linalg.eigh((rfo + rfo.T)/2)  # not necessary - B and rfo already symmetric by construction
    return V[:-1, 0]/V[-1, 0]
    # try removing small eigenvals (for translation/rotation)
    # ... if hess is created from grad, does this actually happen?

# For full Newton's method w/ analytic Hessian, likely necessary to remove small Hessian eigenvalues, esp.
#  corresponding to translation and rotation; Typical LJ and Coulomb systems seem to have only 3 very small
#  eigenvalues (<1E-12), all others >~1E-5 ... remove very small eigenvalues and limit step size, or remove more?
# This does not appear to be an issue for quasi-Newton Hessian estimates
#omega, V = np.linalg.eigh(self.B)
#nz = np.abs(omega) > 1E-12  # threshold should probably be configurable
#omega, V = omega[nz], (V.T)[nz]
#return -np.dot(V.T, np.dot(V, g) / np.fabs(omega))


def monitor(fn, printr=False):
  self = Bunch(r0=0, g0=None)
  def monitored(r):
    f, g = fn(r)
    #print("f(%s): %f, RMS grad: %f, step RMS: %f" % (r if printr else 'x', f, rms(g), rms(r - self.r0)))
    dr = r - self.r0
    s = dr/np.linalg.norm(dr)
    if self.g0 is None: self.g0 = g
    print("f(%s): %f, s.g0: %f, s.g: %f, step RMS: %f" % (r if printr else 'x', f, np.dot(s, self.g0), np.dot(s, g), rms(dr)))
    self.g0 = np.array(g)
    self.r0 = np.array(r)
    return f, g
  return monitored


def fn1(r):
  dr = r - 1.0  # put minimum at (1,1,1)
  dr2 = np.dot(dr,dr)
  return dr2**2, 4*dr*dr2


# en.wikipedia.org/wiki/Himmelblau%27s_function (also en.wikipedia.org/wiki/Test_functions_for_optimization )
# 4 minima: f(3,2)==0, f(-2.805118,3.131312)==0, f(-3.779310,-3.283186)==0, f(3.584428,-1.848126)==0
# 4 saddle points
def himmelblau(r):
  x,y = r
  return ( (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
      np.array([4*x*(x**2 + y - 11) + 2*x + 2*y**2 - 14, 2*x**2 + 4*y*(x + y**2 - 7) + 2*y - 22]) ) # via sympy


# test fn included in scipy; any number of dimensions supported; minima f(1,1,...1)==0
# en.wikipedia.org/wiki/Rosenbrock_function
def rosenbrock(r):
  return scipy.optimize.rosen(r), scipy.optimize.rosen_der(r)


if __name__ == '__main__':

  assert 0, "exit"
  #import pdb; pdb.set_trace()

  # find minima from "outside"
  hres1 = [ optimize(monitor(himmelblau, printr=1), [x0, y0], stepper=LBFGS(H0=1./700), maxiter=100) for x0 in [-10, 10] for y0 in [-10, 10] ]
  assert all(res.success for res in hres1), "Himmelblau test 1 failed"

  # find minima from negative curvature region
  hres2 = [ optimize(monitor(himmelblau, printr=1), r0 stepper=LBFGS(H0=1./700), maxiter=100) for r0 in [[0.7, -0.7], [0.3, 0.3], [-0.7, -0.7], [-0.3, 0.3]] ]
  assert all(res.success for res in hres2), "Himmelblau test 2 failed"

  rres1 = [ optimize(monitor(rosenbrock, printr=1), [-2,3], stepper=LBFGS(H0=1./700), maxiter=100, maxdr=1) for r0 in [[-2,3], [2,3], [2,-3], [-2,-3]] ]
  assert all(res.success for res in rres1), "Rosenbrock test failed"

    x0 = np.random.rand(3)
  # scipy.optimize.minimize(monitor(fn1), x0, jac=True, method='L-BFGS-B', options=dict(disp=True))
  res = optimize(monitor(fn1), x0, stepper=LBFGS(), maxiter=20)
  assert res.success, "LBFGS test failed"

  res = optimize(monitor(fn1), x0, stepper=BFGS(), maxiter=20)
  assert res.success, "BFGS test failed"
