# Dimer method (transition state search)

import math
import numpy as np
from chem.basics import norm, normalize
from chem.molecule import calc_RMSD

# Refs:
# * TSSE dimer.py
# * Kastner, Sherwood (2008): 10.1063/1.2815812 (DL-FIND)
# * Heyden, Bell, Keil (2005): 10.1063/1.2104507 - fancier rotation (better explained by DL-FIND paper)

class DimerMethod:

  def __init__(self, EandG, R0, mode=None,
      maxStep=0.2, dT=0.1, dR=0.005, dTheta=0.01, rotWin=(1, 4), forceWin=(0.0, 1.0)):
    """
    Parameters:
    EandG - fn to return energy and gradient given atomic coordinates
    R0 - initial atomic coordinates
    mode - initial mode (will be randomized if one is not provided)
    maxStep - longest distance dimer can move in a single iteration
    dT - quickmin timestep
    dR - finite difference step size for translation - 0.01 Bohr suggested by Heyden, Bell, Keil for SCF level calc
    dTheta - finite difference step size for rotation
    rotWin - (min, max) rotations per translational step
    forceWin - (min, max) forces, < min, no rotation, > min < max, rotate once, > max, rotate up to rotWin max
    """
    self.steps = 0
    self.dTheta = dTheta
    self.dT = dT
    self.dR = dR
    self.imgEandG = EandG
    self.R0 = np.array(R0)
    self.initialR = np.array(R0)
    self.E = 0
    self.N = normalize(np.random.rand(*np.shape(R0)) if mode is None else mode)
    self.maxStep = maxStep
    self.FCross = None
    self.forceCalls = 0
    self.V = np.array(R0) * 0.0
    self.rotWin = rotWin
    self.forceWin = forceWin
    self.maxForce = 1000


  def EandFperp(self, r1, r2, n):
    E1, G1 = self.imgEandG(r1)
    # estimate G2 from G0 and G1 so we only need one EandG call per rotation step
    # note that this gives Fperp = (F1perp - F0perp)/self.dR
    #E2, G2 = self.imgEandG(r2)
    E2, G2 = 2*self.E0 - E1, 2*self.G0 - G1
    self.forceCalls += 1  #2
    self.F1, self.F2 = -G1, -G2
    F1perp = self.F1 - np.vdot(self.F1, n)*n
    F2perp = self.F2 - np.vdot(self.F2, n)*n
    return (E1 + E2)/2, (F1perp - F2perp)/(2*self.dR)


  def rotate(self):
    iteration = 0
    self.E0, self.G0 = self.imgEandG(self.R0)
    R1 = self.R0 + self.dR * self.N
    R2 = self.R0 - self.dR * self.N
    self.E, FPerp = self.EandFperp(R1, R2, self.N)
    self.Fnorm = norm(FPerp)
    BigTheta = FPerp/self.Fnorm
    while (self.Fnorm > self.forceWin[1] and iteration < self.rotWin[1]) \
        or (self.Fnorm > self.forceWin[0] and iteration < self.rotWin[0]) or iteration == 0:
      r1 = self.R0 + normalize(self.N * math.cos(self.dTheta) + BigTheta * math.sin(self.dTheta)) * self.dR
      n = normalize(r1 - self.R0)
      r2 = self.R0 - (n * self.dR)
      _, fperp = self.EandFperp(r1, r2, n)
      bigtheta = normalize(-self.N * math.sin(self.dTheta) + BigTheta * math.cos(self.dTheta))
      Fmag = np.vdot(FPerp, BigTheta)
      fmag = np.vdot(fperp, bigtheta)
      FMAG = (fmag + Fmag)/2
      FmagPrime = (fmag - Fmag)/self.dTheta
      deltaTheta = -0.5*math.atan(2*FMAG/FmagPrime) + self.dTheta/2
      if FmagPrime > 0:
        deltaTheta += np.pi/2
      # perform rotation and update R1, R2
      R1 = self.R0 + (self.N * math.cos(deltaTheta) + BigTheta * math.sin(deltaTheta)) * self.dR
      self.N = normalize(R1 - self.R0)
      R2 = self.R0 - self.dR * self.N
      self.E, FPerp = self.EandFperp(R1, R2, self.N)
      self.Fnorm = norm(FPerp)
      BigTheta = FPerp/self.Fnorm
      iteration += 1


  def step(self):
    self.steps += 1
    self.rotate()
    self.FR = (self.F1 + self.F2)/2
    self.FParallel = np.vdot(self.FR, self.N)*self.N
    self.C = np.vdot((self.F2 - self.F1), self.N)/(2*self.dR)
    self.FCross = -self.FParallel if self.C > 0 else (self.FR - 2*self.FParallel)
    dV = self.FCross * self.dT
    self.V = dV*(1.0 + np.vdot(dV, self.V)/np.vdot(dV, dV)) if np.vdot(self.V, self.FCross) > 0 else dV
    step = self.V * self.dT
    self.R0 += step if norm(step) < self.maxStep else self.maxStep * normalize(step)
    self.maxForce = np.max(np.linalg.norm(self.FCross, axis=1))


  def search(self, minForce=0.01, quiet=False, maxForceCalls=100000, movie=None):
    if movie: movie.append(self.R0)
    while self.maxForce > minForce and self.forceCalls < maxForceCalls:
      self.step()
      if movie: movie.append(self.R0)
      if not quiet:
        print("max force: %f; RMSD from start: %f; curvature: %f; rotational force: %f; force calls: %d" %
            (self.maxForce, calc_RMSD(self.R0, self.initialR), self.C, self.Fnorm, self.forceCalls))
    self.converged = self.maxForce <= minForce
    return self.R0


  # for use w/ external optimizer (gradient only)
  def EandG(self, r):
    self.R0 = r
    self.rotate()
    self.FR = (self.F1 + self.F2)/2
    self.FParallel = np.vdot(self.FR, self.N)*self.N
    self.C = np.vdot((self.F2 - self.F1), self.N)/(2*self.dR)
    # if curvature along min mode is negative, move directly uphill to get away from region
    self.FCross = -self.FParallel if self.C > 0 else (self.FR - 2*self.FParallel)
    print("max force: %f; curvature: %f; EandG calls: %d" % (np.max(np.abs(self.FCross)), self.C, self.forceCalls))
    return self.E0, -self.FCross


# Lanczos method - 10.1063/1.4862410 (Zeng, Xiao, Henkelman 2014) shows that dimer method is just an
#  approximation to Lanczos method for eigenvalue/vector estimation (and theoretically cannot outperform it)
# Refs:
# * github.com/eljost/pysisyphus/blob/master/pysisyphus/modefollow/lanczos.py
# * github.com/eljost/pysisyphus/blob/master/pysisyphus/tsoptimizers/dimer.py
# * TSASE dimer/lanczos.py
# * en.wikipedia.org/wiki/Lanczos_algorithm

# Kastner, Sherwood eq. 5
def minmodeangle(dG, tau):
  """ Given tau and dG = (g(r0 + dx*tau) - g(r0))/dx, estimate angle to rotate tau to minimize curvature """
  g_perp = dG - np.dot(dG,tau)*tau
  theta = -g_perp/np.linalg.norm(g_perp)  # steepest descent estimate of theta direction
  C = np.dot(dG, tau)
  dC = 2*np.dot(dG, theta)  # dC/dphi
  return -0.5*np.arctan2(dC, 2*abs(C))


class LanczosMethod:

  def __init__(self, EandG, tau=None):
    self.imgEandG = EandG
    self.tau = tau  # minimum eigenvector direction
    self.ncalls = 0


  def lanczos(self, r0, G0, u0, dx=5e-3, eigtol=1e-2, maxiter=25, reortho=True):
    """ Use Lanczos method to estimate smallest eigenvalue/vector of Hessian, using only gradient information
    dx: step size for finite differencing (default 0.005 Ang ~ 0.01 Bohr, appropriate for SCF level calc)
    """
    r0shape = np.shape(r0)
    r0, G0 = np.ravel(r0), np.ravel(G0)
    u = np.ravel(u0) if u0 is not None else np.random.rand(len(r0)) - 0.5
    Q = np.zeros((len(r0), maxiter))
    T = np.zeros((maxiter, maxiter))  # tri-diagonal matrix that will be diagonalized
    for i in range(maxiter):
      beta = np.linalg.norm(u)
      q = u/beta
      Q[:, i] = q
      E1, G1 = self.imgEandG(np.reshape(r0 + dx*q, r0shape))
      self.ncalls += 1
      u = (np.ravel(G1) - G0)/dx
      # after the first few outer steps (lanczos + translate), Gperp becomes small for u0 and eigenvalue
      #  converges after just 2 iterations, so we could maybe skip the second iteration by checking this:
      #if i == 0 and minmodeangle(u, q) < rad_tol: break  # 0.5 deg seems good  #Gperp = u - np.dot(u, q)*q
      if i > 0:
        u = u - beta * Q[:, i-1]
      alpha = np.dot(q, u)
      u = u - alpha * q
      # re-orthogonalize
      if reortho:
        u = u - np.dot(Q, np.dot(Q.T, u))
      # extend T matrix
      T[i, i] = alpha
      if i > 0:
         T[i-1, i] = beta
         T[i, i-1] = beta

      w, v = np.linalg.eigh(T[:i+1, :i+1])
      w_min = w[0]  # we assume eigh() returns eigenvalues in ascending order (LAPACK routine it uses does)
      # Check eigenvalue convergence
      if i > 0 and abs((w_min - w_min0)/w_min0) < eigtol:
        Q = Q[:, :i+1]
        break
      w_min0 = w_min
      # TSASE code checks angle between eigenvector and prev estimate instead
      #v_min = v[:,0]
      #eigv = np.dot(Q[:, :i+1], v_min)
      #eigv = eigv/np.linalg.norm(eigv)
      #dphi = np.arccos(np.dot(eigv, eigv0))
      #if dphi > np.pi/2.0: dphi = np.pi - dphi
      #if dphi < dphi_tol: break
      #eigv0 = np.array(eigv)

    v_min = v[:,0]#[:, 0]
    eigv = np.dot(Q, v_min)  #np.dot(Q[:, :i+1], v_min)
    eigv = eigv/np.linalg.norm(eigv)
    return w_min, np.reshape(eigv, r0shape)  # w_min gives curvature along eigv direction


  # for use w/ gradient-only optimizer
  def EandG(self, r):
    E0, G0 = self.imgEandG(r)  # corresponds to dimer midpoint
    self.ncalls += 1
    w, self.tau = self.lanczos(r, G0, self.tau)  # use last tau direction as initial guess
    Gtau = np.vdot(G0, self.tau)*self.tau
    # reverse gradient along tau direction (if curvature < 0 - otherwise move directly uphill)
    Geff = G0 - 2*Gtau if w < 0 else -Gtau
    print("max force: %f; curvature: %f; EandG calls: %d" % (np.max(np.abs(Geff)), w, self.ncalls))
    return E0, Geff
