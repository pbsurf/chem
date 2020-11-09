import math, collections
import numpy as np


## general fns

#~ class Bunch:
#~   def __init__(self, **kwargs):
#~     self.__dict__.update(kwargs)
# version that supports dict methods as well by inheriting from dict; see also namedtuple()
class Bunch(dict):
  def __init__(self, *args, **kwargs):
    super(Bunch, self).__init__(*args, **kwargs)
    self.__dict__ = self


def setattrs(obj, **kwargs):
  """ set one or more attributes of object `obj` from `kwargs` and return `obj` """
  for k,v in kwargs.iteritems():
    setattr(obj, k, v)
  return obj


def setitem(obj, idx, val):
  """ obj[idx] = val; return obj """
  obj[idx] = val
  return obj


def getvalues(d, *args):
  """ get multiple values from dict, e.g., x, y = getvalues(dictionary, 'x', 'y') """
  return tuple(d.get(k, None) for k in args)


def getattrs(obj, *args):
  """ get multiple values from object, e.g., x, y = getattrs(obj, 'x', 'y') """
  return tuple(getattr(obj, k, None) for k in args)
  #return tuple(getattr(obj, k, v) for k,v in kwargs.items())  -- kwargs order only guaranteed in python 3.6+


def argsort(seq):
  return sorted(range(len(seq)), key=seq.__getitem__)


def flattengen(t):
  for item in t:
    if isinstance(item, collections.Iterable) and not isinstance(item, basestring):
      for subitem in flattengen(item):
        yield subitem
    else:
      yield item


def flatten(t):
  return list(flattengen(t))


def chunks(t, size):
  return (t[ii:ii+size] for ii in xrange(0, len(t), size))


def partial(f, *args, **kwargs):
  def newf(*fargs, **fkwargs):
    kwargs2 = kwargs.copy()
    kwargs2.update(fkwargs)
    return f(*(args + fargs), **kwargs2)
  return newf


# Not sure safelen() and islist() should be retained; note that len(x) > 0 doesn't assure that we can use
#  numeric indexing

# test if list and get length in one call
def safelen(x):
  try: return len(x)
  except: return -1

def islist(x):
  return safelen(x) >= 0


def skiptoline(f, line, extra=0):
  l = f.readline()
  while l and l[:len(line)] != line:
    l = f.readline()
  for _ in range(extra):
    l = f.readline()
  return l


## math

def norm(v):
  return math.sqrt(np.vdot(v,v))  # faster than np.linalg.norm; vdot flattens to 1D

def normalize(v):
  v = np.asarray(v)
  n = norm(v)
  if n == 0.0:
    raise Exception("Cannot normalize() null vector")
  return v/n

def rms(v):
  return np.sqrt(np.sum(v*v)/len(v))  # np.sqrt(np.mean(v*v, axis=axis)

# affine transformations
# these are transposed from rotation_matrix() and translation_matrix() in glutils ... would be nice to unify

def rotation_matrix(direction, angle):
  """ Create a rotation matrix for rotation around an axis `direction` (d) by `angle` (a) radians:
   R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)
  """
  d = normalize(direction)
  ddt = np.outer(d, d)
  skew = np.array([[    0,  d[2], -d[1]],
                   [-d[2],     0,  d[0]],
                   [ d[1], -d[0],    0]])
  # ---
  return ddt + np.cos(angle) * (np.eye(3) - ddt) + np.sin(angle) * skew


def translation_matrix(dr):
  return np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [dr[0],dr[1],dr[2],1]], dtype=np.float64)


def to4x4(m):
  M = np.identity(4);  M[:3,:3] = m; return M


# any way to avoid duplication with vis.glutils?
#~ def rotation_about(direction, angle, point):
#~   """ return 4x4 matrix for rotation about a point """
#~   R = np.zeros((4,4))
#~   R0 = rotation_matrix(direction, angle)
#~   R[:3, :3] = R0
#~   R[:3, 3] = point - np.dot(R0, point)
#~   return R


def apply_affine(M, v):
  return np.dot(v, M[:3, :3]) + M[3, :3]


def random_rotation(rand=None):
  """ ref: https://math.stackexchange.com/questions/442418 """
  rand = np.random.random(3) if rand is None else rand
  cos_theta = 2*rand[0] - 1
  sin_theta = np.sin(np.arccos(cos_theta))
  phi = 2*np.pi*rand[1]
  direction = [sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta]
  return rotation_matrix(direction, 2*np.pi*rand[2])


## geometry related fns

def calc_dist(v1, v2=None, grad=False):
  """ v1,v2: numpy 3-vectors; returns bond length (l) and, if grad=True, gradient as:
    [[dl/dx1, dl/dy1, dl/dz1], [dl/dx2, dl/dy2, dl/dz2]]
  """
  v1, v2 = (v1,v2) if v2 is not None else v1
  l = norm(v1 - v2)
  if not grad:
    return l
  dl1 = (v1 - v2)/l
  return l, [dl1, -dl1]


def calc_angle(v1, v2=None, v3=None, grad=False):
  """ v1,v2,v3: numpy 3-vectors; returns interior angle (a) specified by input vectors and if grad=True,
    gradient as: [[da/dx1, da/dy1, da/dz1], [da/dx2, da/dy2, da/dz2], [da/dx3, da/dy3, da/dz3]]
  """
  v1, v2, v3 = (v1,v2,v3) if v3 is not None else v1
  v12 = v1 - v2
  v23 = v2 - v3
  # result is in radians (pi - converts to interior bond angle)
  deg = np.pi - np.arccos(np.dot(v12, v23)/norm(v12)/norm(v23))
  if not grad:
    return deg
  # from Mathematica
  d12sq = np.dot(v12, v12)
  d23sq = np.dot(v23, v23)
  dot1223 = np.dot(v12, v23)
  denom = np.sqrt(d12sq*d23sq - dot1223**2)  # |v12|*|v23|*sin(angle)
  # for small angle, numerator ~ angle^2, denominator ~ angle so grad -> 0
  ## TODO: write tests for this code and then cleanup the math to use unit vectors 12 and 23
  if denom/norm(v12)/norm(v23) < 1E-20:
    return deg, [0.0, 0.0, 0.0]
  da1 = (v23 - v12*dot1223/d12sq)/denom
  da3 = (-v12 + v23*dot1223/d23sq)/denom
  # translation of one atom (e.g. middle atom) by delta is equiv to translation of other two atoms by -delta
  da2 = -da1 - da3
  return deg, [da1, da2, da3]


# TODO: need to handle v123 or v234 null, although not sure we'll encounter this in practice
# Note: all the signs in the gradient calculation were determined by trial and error, not careful algebra
def calc_dihedral(v1, v2=None, v3=None, v4=None, grad=False):
  """ return dihedral angle - angle between planes defined by points 1,2,3 and 2,3,4 - and optionally gradient
    refs: Wilson, Mol. Vibrations (1955), ch. 4; JCC 17, 1132 (via Wikipedia)
  """
  v1, v2, v3, v4 = (v1,v2,v3,v4) if v4 is not None else v1
  # vab = va - vb is vector from b to a
  v12 = v1 - v2
  v32 = v3 - v2
  v43 = v4 - v3
  # normal vectors for planes (1,2,3) and (2,3,4)
  v123 = np.cross(v12, v32)
  v234 = np.cross(v43, v32)
  # arccos expr has numerical instability near 0 and pi which causes problems with DLC convergence
  #dihed = <sign> * np.arccos(np.dot(v123, v234)/(norm(v123)*norm(v234)))
  dihed = np.arctan2(np.dot(np.cross(v123, v234), v32/norm(v32)), np.dot(v123, v234))
  if not grad:
    return dihed
  # gradient formulas from Wilson, p. 61
  # dd1 determined from geometric considerations - direct calculation of gradient was way too messy
  d32 = norm(v32)
  dd1 = d32*v123/np.sum(v123**2)
  dd4 = -d32*v234/np.sum(v234**2)  # 1<->4, 2<->3
  dd2 = ( -(d32**2 - np.dot(v12,v32))*v123/np.sum(v123**2) - np.dot(v43,v32)*v234/np.sum(v234**2) )/d32
  # alternative calc - useful for consistency check since grad must satisfy np.sum(grad) == 0
  #dd2 = (np.dot(v12,v32)/d32**2 - 1)*dd1 + np.dot(v43,v32)*(dd4/d32**2)
  #dd3 = (-np.dot(v43,v32)/d32**2 - 1)*dd4 - np.dot(v12,v32)*(dd1/d32**2)
  # translation of one atom by delta is equiv to translation of all others by -delta
  dd3 = -(dd1 + dd2 + dd4)
  # note that we do not reverse sign of the gradient when the the sign of dihed is reversed
  return dihed, [dd1, dd2, dd3, dd4]


# this was created to try using instead of calc_dihedral for DLCs ... but gradient -> 0 at 0 and 180 deg
def cos_dihedral(v1, v2=None, v3=None, v4=None, grad=False):
  """ return value and gradient of the cosine of a dihedral angle
    ref: https://salilab.org/modeller/9v6/manual/node436.html
  """
  v1, v2, v3, v4 = (v1,v2,v3,v4) if v4 is not None else v1
  v12 = v1 - v2
  v32 = v3 - v2
  v34 = v3 - v4
  # normal vectors for planes (1,2,3) and (2,3,4)
  v123 = np.cross(v12, v32)
  v234 = np.cross(v32, v34)
  # use clip to handle y a tiny bit out of range due to FP rounding
  cosdihed = np.clip(np.dot(v123, v234)/(norm(v123)*norm(v234)), -1.0, 1.0)
  if not grad:
    return cosdihed
  # alternatively, we could calculate the gradient as in calc_dihedral then multiply each term by -sin(dihed)
  a = (v234/norm(v234) - cosdihed*v123/norm(v123))/norm(v123)
  b = (v123/norm(v123) - cosdihed*v234/norm(v234))/norm(v234)
  dd1 = np.cross(v32, a)
  dd4 = np.cross(v32, b)  # typo in reference!
  dd2 = np.cross(v1 - v3, a) - np.cross(v34, b)
  dd3 = np.cross(v2 - v4, b) - np.cross(v12, a)
  return cosdihed, [dd1, dd2, dd3, dd4]


## constants

# Units - atomic units, except lengths are in Angstroms by default
# length: Angstroms
# energy: Hartree; but we will often print values in kJ/mol
# charge: electron charge (q_e)
# Coulomb energy (Hartree): q_1*q_2/norm(r_2 - r1) - where r is in *Bohr* NOT Angstrom

# https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
ANGSTROM_PER_BOHR = 0.52917721067
KJMOL_PER_HARTREE = 2625.499638
KCALMOL_PER_HARTREE = 627.509391
EV_PER_HARTREE = 27.2113845
AVOGADRO = 6.02214076E23  # exact value as of Nov 2018
# recall that kB*T ~ 2.5 kJ/mol at 300K


# inc() and dec() were introduced to translate between 1-based atom indexing of most comp. chem. software
#  and 0-based indexing of python ... but for real work, it seems we're rarely specifying atoms explicitly
#  by index, so perhaps these should be removed

#~ def dec(*args):
#~   """ convert from 1-based to 0-based indexing for internal use by molecule object, etc """
#~   return [_dec(x) for x in args]

#~ def _dec(x):
#~   try: return [_dec(y) for y in x]
#~   except: return x - 1


#~ def inc(*args):
#~   """ convert from 0-based to 1-based indexing for printing """
#~   return [_inc(x) for x in args]

#~ def _inc(x):
#~   try: return [_inc(y) for y in x]
#~   except: return x + 1
