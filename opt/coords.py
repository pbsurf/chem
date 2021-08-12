import numpy as np
from ..basics import *


# coord object for Cartesians
class XYZ:
  def __init__(self, X, active=None, atoms=None, ravel=True):
    # slice(None) as index selects all elements - note that gradfromxyz and active will return copies!
    self.actv = active if active is not None else slice(None)
    self.atoms = atoms  # for use w/ HDLC
    self.ravel = ravel
    self.X = X.r if hasattr(X, 'atoms') else np.array(X)  #np.asarray(getattr(X, 'r', X)) - asarray doesn't copy!
    if atoms and len(atoms) != len(self.X):
      self.X = self.X[atoms]

  def __repr__(self):
    return "XYZ(atoms: %d, active atoms: %d)" % (len(self.X), np.size(self.active())/3)

  def init(self, X=None):
    if X is not None:
      self.X[self.actv] = X[self.actv]

    if X is not None and np.any(X != self.X):
      assert 0, "Inactive atom positions don't match!!!"  # self.X = X would better match DLC class!

    return self

  def update(self, S, newX=None):
    self.X[self.actv] = self.toxyz(S)
    return True

  def toxyz(self, S):
    return np.reshape(S, (-1,3))

  def gradfromxyz(self, gxyz):
    return np.ravel(gxyz[self.actv]) if self.ravel else gxyz[self.actv]

  def active(self):
    return np.ravel(self.X[self.actv]) if self.ravel else self.X[self.actv]

  def xyzs(self):
    return self.X


# coord object for treating molecule as rigid body
class Rigid:
  def __init__(self, X, atoms=None, pivot=None):
    self.atoms = atoms  # for use w/ HDLC
    self.pivot = pivot
    self.X = X.r if hasattr(X, 'atoms') else np.array(X)
    if atoms and len(atoms) != len(self.X):
      self.X = self.X[atoms]

  def __repr__(self):
    return "Rigid(atoms: %d)" % len(self.X)

  def init(self, X=None):
    if X is not None:
      self.X = X
    com = np.mean(self.X, axis=0) if self.pivot is None else self.pivot
    self.X0 = self.X - com
    self.S = np.hstack((com, [0, 0, 0]))
    return self

  def update(self, S, newX=None):
    self.S = S
    self.X = self.toxyz(S)
    return True

  # using Euler angles
  #def toxyz(self, S):
  #  A = rotation_matrix([1,0,0], S[3])
  #  B = rotation_matrix([0,1,0], S[4])
  #  C = rotation_matrix([0,0,1], S[5])
  #  return np.dot(self.X0, C.dot(B).dot(A)) + S[0:3]
  #
  #def gradfromxyz(self, gxyz):
  #  A, dA = rotation_matrix([1,0,0], self.S[3], grad=True)
  #  B, dB = rotation_matrix([0,1,0], self.S[4], grad=True)
  #  C, dC = rotation_matrix([0,0,1], self.S[5], grad=True)
  #  da = np.vdot(self.X0.dot(C.dot(B).dot(dA)), gxyz)
  #  db = np.vdot(self.X0.dot(C.dot(dB).dot(A)), gxyz)
  #  dc = np.vdot(self.X0.dot(dC.dot(B).dot(A)), gxyz)
  #  return np.hstack((np.sum(gxyz, axis=0), [da, db, dc]))

  # using exponential map (rotation vector)
  def toxyz(self, S):
    return np.dot(self.X0, expmap_rot(S[3:6])) + S[0:3]  # should be R.T (and dRii.T below)!

  def gradfromxyz(self, gxyz):
    R, dR = expmap_rot(self.S[3:6], grad=True)
    return np.hstack((np.sum(gxyz, axis=0), [np.vdot(self.X0.dot(dRii), gxyz) for dRii in dR]))

  def active(self):
    return self.S

  def xyzs(self):
    return self.X


def coord_sanity(EandG, coords, r0=None, scale=0.02):
  if r0 is None:
    r0 = coords.xyzs()
  else:
    coords.init(r0)
  S = coords.active()
  dr = scale*(np.random.rand(*np.shape(r0)) - 0.5)
  dS = coords.gradfromxyz(dr)
  E0, G0xyz = EandG(r0)
  G0 = coords.gradfromxyz(G0xyz)
  for ii in range(20):
    coords.update(S + dS)
    E1, G1xyz = EandG(coords.xyzs())
    G1 = coords.gradfromxyz(G1xyz)
    dE0 = np.vdot(G0, dS)
    dE1 = np.vdot(G1, dS)
    # note that rms(r) == rms(norm(r, axis=1))
    print("rms(dS): %.8f; dE_expect: %.8f (G1: %.8f); dE_actual: %.8f" % (rms(dS), dE0, dE1, E1 - E0))
    dS = 0.5*dS
