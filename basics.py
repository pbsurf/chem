import os, math, collections, time
import numpy as np


DATA_PATH = os.path.expandvars("$HOME/qc/data")

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
  for k,v in kwargs.items():
    setattr(obj, k, v)
  return obj


def setitem(obj, idx, val):
  """ obj[idx] = val; return obj """
  obj[idx] = val
  return obj


def getitem(obj, idx, dflt=None):
  try: return obj[idx]
  except: return dflt


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
    if isinstance(item, collections.Iterable) and not isinstance(item, ''.__class__):
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


# sorting out python lists/numpy lists/python scalars/numpy lists is a huge mess
# - `x is int` (float): fails for numpy scalar types (e.g. numpy.int64)
# - np.isscalar()/np.ndim(): would be preferable to have a general soln that doesn't require numpy
# - safelen/islist(): is this the best way?  should we even keep these?  len(x) > 0 doesn't mean x[0] will work

# test if list and get length in one call
def safelen(x):
  try: return len(x)
  except: return -1

def islist(x):
  return safelen(x) >= 0


def skiptoline(f, line, extra=0, strip=False):
  l = f.readline()
  while l and not (l.strip() if strip else l).startswith(line):  #l[:len(line)] != line:
    l = f.readline()
  for _ in range(extra):
    l = f.readline()
  return l


def read_only(*args):
  """ make passed numpy arrays read-only """
  for a in args:
    a.flags.writeable = False
  return args


def unique(x):
  """ get unique items from iterable, preserving order """
  #np.unique(x, axis=0).tolist()
  return list(dict.fromkeys(x))


#all_cycles = [list(dfs_path(mol.mmconnect, ii, ii)) for ii in mol.listatoms()]
def dfs_path(graph, start, end):
  """ generator yielding paths from start to end in graph; graph[i] is list of nodes with edge from node i """
  fringe = [[start]]
  while fringe:
    path = fringe.pop()
    for next_state in graph[path[-1]]:
      if (start != end or len(path) > 2) and next_state == end:
        yield path  #path + [end]
      elif next_state not in path:
        fringe.append(path+[next_state])


# clock() doesn't seem to include time of any external processes run
def benchmark(fn):
  """ simple benchmark for single run of a slow function - use timeit for all other cases """
  c0, t0 = time.perf_counter(), time.time()  # clock -> perf_counter
  res = fn()
  print("clock(): %fs; time(): %fs" % (time.perf_counter() - c0, time.time() - t0))
  return res


def read_file(filename, mode='r'):
  with open(filename, mode) as f:
    return f.read()


def write_file(filename, contents, mode='x'):
  with open(filename, mode if type(contents) is str else mode+'b') as f:
    f.write(contents)


# storing anything but numeric numpy arrays in hdf5 is a mess
def write_pickle(filename, **kwargs):
  import pickle
  write_file(filename, pickle.dumps(kwargs))


def read_pickle(filename, *args):
  import pickle
  return getvalues(pickle.loads(read_file(filename, 'rb')), *args)


def _write_zip(filename, kwargs, mode):
  from zipfile import ZipFile, ZIP_DEFLATED
  with ZipFile(filename, mode, ZIP_DEFLATED) as zf:
    for k,v in kwargs.items():
      zf.writestr(k, v)

def write_zip(filename, **kwargs):
  _write_zip(filename, kwargs, mode='x')

def append_zip(filename, **kwargs):
  _write_zip(filename, kwargs, mode='a')


def read_zip(filename, *args):
  from zipfile import ZipFile
  with ZipFile(filename, 'r') as zf:
    return zf.read(args[0]) if len(args) == 1 else tuple(zf.read(a) for a in args)


# object returned by np.load("*.npz") must be closed, so we need this fn to do one-liners
def load_npz(filename, *args):
  with np.load(filename) as npz:
    return Bunch(npz) if len(args) == 0 else npz[args[0]] if len(args) == 1 else tuple(npz[a] for a in args)


# not sure if we'll actually use this
# use: write_zip('mutate3.zip', dE=npy_bytes(dE));  dE = npy_bytes(read_zip('mutate3.zip', 'dE'))
def npy_bytes(x):
  """ convert between numpy array and npy serialized data in bytes object """
  from io import BytesIO
  if type(x) is bytes:
    return np.load(BytesIO(x))
  f = BytesIO()
  np.save(f, x)
  return f.getvalue()


## plotting
# `echo "backend: Qt4Agg" >> ~/.config/matplotlib/matplotlibrc` OR do matplotlib.use('Qt4Agg') to prevent
#  crashes with default TkAgg backend when using with Chemvis
# Ctrl+C interference: happens with both TkAgg and Qt4Agg (but not WebAgg); w/ and w/o plt.ion() (interactive
#  mode), even when run in different thread or process (both fork and spawn) ... looks like no easy fix -
#  bug is here: https://bugs.python.org/issue23237 - but not sure why it still happens w/ separate process
# Solution is to run import gc; gc.collect() after closing all figure windows!

# can't have keyword args after *args in Python 2!
def plot(*args, **kwargs):
  """ Pass x,y pairs, then optional keyword args: xlabel, ylabel, title, legend. For subplots, pass shape as
    subplots=(rows, cols) and subplot=index (ordered left to right, top to bottom, from 0)
  """
  import matplotlib.pyplot as plt
  subplot = kwargs.pop('subplot', None)
  subplots = kwargs.pop('subplots', None)
  if not subplot:  # None or 0 (first subplot)
    plt.ion()  # interactive mode - make plot window non-blocking
    plt.figure()
  if subplots is not None:
    plt.subplot(subplots[0], subplots[1], subplot+1)
  if 'xlabel' in kwargs: plt.xlabel(kwargs.pop('xlabel'))
  if 'ylabel' in kwargs: plt.ylabel(kwargs.pop('ylabel'))
  if 'title' in kwargs: plt.title(kwargs.pop('title'))
  if 'legend' in kwargs: plt.legend(kwargs.pop('legend'))
  plotfn = getattr(plt, kwargs.pop('fn', 'plot'))
  plotfn(*args, **kwargs)  #picker=kwargs.get('picker', None))
  if not subplot:
    plt.show()  # try block=False here instead of ion()?
  return plt.gcf()


## math

def norm(v):
  return math.sqrt(np.vdot(v,v))  # faster than np.linalg.norm; vdot flattens to 1D

def normalize(v):
  v = np.asarray(v)
  n = norm(v)
  if n == 0.0:
    raise Exception("Cannot normalize() null vector")
  return v/n

# l1normalize() might be a more precise name
def l1normalize(v):
  """ normalize array of probabilities (to sum to 1) """
  return np.asarray(v)/(np.sum(v) or 1.0)

# note that this gives us rms(norm(r, axis=1)) for an array of vectors ... rename to rmsnorm()?
def rms(v):
  return np.sqrt(np.sum(v*v)/len(v))  # not equivalent: np.sqrt(np.mean(v*v, axis=axis)

def get_extents(r, pad=0.0):
  """ return extents of box containing all points r, with optional padding pad """
  return np.array([np.amin(r, axis=0) - pad, np.amax(r, axis=0) + pad])

# affine transformations
# these are transposed from rotation_matrix() and translation_matrix() in glutils ... would be nice to unify

def skew_matrix(a):
  """ Given 3-vector a, returns [a]_\cross, the matrix such that [a]_\cross . b = a \cross b for any b """
  return np.array([[    0, -a[2],  a[1]],
                   [ a[2],     0, -a[0]],
                   [-a[1],  a[0],    0]])


# Using rotation matrices: R.dot(x) for single (column) vector x; x.dot(R.T) for array of vectors x (since R
#  is unitary, R.T == R^-1)

# we'll keep this separate from expmap_rot for now since derivative is wrt angle instead of rot vector
def rotation_matrix(direction, angle, grad=False):
  """ Create a rotation matrix for rotation around an axis `direction` (d) by `angle` (a) radians:
   R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)
  """
  d = normalize(direction)
  ddt = np.outer(d, d)
  skew = skew_matrix(d)  # -d ... whoops, afraid to change it now
  R = ddt + np.cos(angle)*(np.eye(3) - ddt) + np.sin(angle)*skew
  return R if not grad else (R, -np.sin(angle)*(np.eye(3) - ddt) + np.cos(angle)*skew)


# Exponential map representation of a rotation is axis-angle rep. w/ the length of vector equal to the angle
#  instead of one.  For an infinitesimal rotation d_ang about unit vector d, R = I + skew(d)*d_ang, so for
#  a general rotation, R = exp(skew(d)*ang).  Then with ang = norm(v) and d = v/norm(v),
# R = exp(skew(v)) = I + sin(ang)*skew(d) + (1 - cos(ang))*skew(d)**2
#   = d.dt + cos(ang)*(I - d.dt) + sin(ang)*skew(d)  (note d.dt = np.outer(d,d))
# This is equivalent to the Euler-Rodrigues rotation formula
# Refs:
#  1. arxiv.org/pdf/1312.0788.pdf (Gallego, Yezzi)
#  2. rotations.berkeley.edu/other-representations-of-a-rotation
#  3. en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
def expmap_rot(v, grad=False):
  """ return rotation matrix R for rotation by norm(v) around axis v and, if grad==True, derivatives of R wrt
    components of v
  """
  I = np.eye(3)
  angle = norm(v)
  if angle == 0:
    return (I, [skew_matrix(ei) for ei in I]) if grad else I  # Ref 1.: sec. 3.3
  d = v/angle
  ddt = np.outer(d, d)
  R = ddt + np.cos(angle)*(I - ddt) + np.sin(angle)*skew_matrix(d)
  if not grad:
    return R
  dR = np.array([  # note we also use I[i] to get basis vectors; Ref. 1, eq. 9
      (v[i]*skew_matrix(v) + skew_matrix(np.cross(v, (I - R).dot(I[i])))).dot(R)/angle**2 for i in [0,1,2] ])
  return R, dR


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


def random_direction(rand=None):
  #return normalize(np.random.randn(3))  # randn: normal (Gaussian) distribution
  rand = np.random.random(2) if rand is None else rand
  cos_theta = 2*rand[0] - 1
  sin_theta = np.sin(np.arccos(cos_theta))  #np.sqrt(1 - cos_theta**2)
  phi = 2*np.pi*rand[1]
  return [sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta]


def random_rotation(rand=None):
  """ ref: https://math.stackexchange.com/questions/442418 """
  rand = np.random.random(3) if rand is None else rand
  return rotation_matrix(random_direction(rand[0:2]), 2*np.pi*rand[2])


def align_vector(subject, ref):
  """ return rotation matrix to align subject vector to ref vector """
  v = np.cross(subject, ref)
  if norm(v) == 0:
    v = [ref[1], -ref[0], 0] if abs(ref[2]) < abs(ref[0]) else [0, -ref[2], ref[1]]  # orthogonal vector
  return rotation_matrix(v, np.arccos(np.dot(subject, ref)/norm(subject)/norm(ref)))


def align_axes(subject, ref):
  """ return rotation matrix to align two orthogonal subject vectors to two orthogonal ref vectors """
  R0 = align_vector(subject[0], ref[0])
  R1 = align_vector(np.dot(R0, subject[1]), ref[1])
  return np.dot(R1, R0)


# moment of inertia
def moment_inertia(r, m):
  """ calculate moment of inertia tensor (about center of mass) for masses m at positions r """
  r = r - np.einsum('z,zr->r', m, r)/np.sum(m)
  return np.eye(3)*np.einsum('z,zr,zr', m, r, r) - np.einsum('z,zr,zs->rs', m, r, r)


def principal_axes(I, moments=False):
  """ diagonalize moment of inertia tensor I to return principle axes and moments """
  # as usual, we assume eigh eigenvalues are sorted despite documentation
  ev, evec = np.linalg.eigh(I)
  evec = evec.T
  if np.dot(np.cross(evec[0], evec[1]), evec[2]) < 0:  # ensure right-handed axes
    evec *= -1
  return (ev, evec) if moments else evec


## geometry related fns

def calc_dist(v1, v2=None, grad=False):
  """ v1,v2: numpy 3-vectors; returns bond length (l) and, if grad=True, gradient as:
    [[dl/dx1, dl/dy1, dl/dz1], [dl/dx2, dl/dy2, dl/dz2]]
  """
  if v2 is None:
    v1, v2 = v1
  l = norm(v1 - v2)
  if not grad:
    return l
  dl1 = (v1 - v2)/l
  return l, np.array([dl1, -dl1])


def calc_angle(v1, v2=None, v3=None, grad=False):
  """ v1,v2,v3: numpy 3-vectors; returns interior angle in radians (a) specified by input vectors and, if
    grad=True, gradient as: [[da/dx1, da/dy1, da/dz1], [da/dx2, da/dy2, da/dz2], [da/dx3, da/dy3, da/dz3]]
  """
  if v3 is None:
    v1, v2, v3 = v1
  v12 = v1 - v2
  v23 = v2 - v3
  # result is in radians (pi - converts to interior bond angle)
  ang = np.pi - np.arccos(np.dot(v12, v23)/norm(v12)/norm(v23))
  if not grad:
    return ang
  # from Mathematica
  d12sq = np.dot(v12, v12)
  d23sq = np.dot(v23, v23)
  dot1223 = np.dot(v12, v23)
  denom = np.sqrt(d12sq*d23sq - dot1223**2)  # |v12|*|v23|*sin(angle)
  # for small angle, numerator ~ angle^2, denominator ~ angle so grad -> 0
  ## TODO: write tests for this code and then cleanup the math to use unit vectors 12 and 23
  if denom/norm(v12)/norm(v23) < 1E-20:
    return ang, [0.0, 0.0, 0.0]
  da1 = (v23 - v12*dot1223/d12sq)/denom
  da3 = (-v12 + v23*dot1223/d23sq)/denom
  # translation of one atom (e.g. middle atom) by delta is equiv to translation of other two atoms by -delta
  da2 = -da1 - da3
  return ang, np.array([da1, da2, da3])


# TODO: need to handle v123 or v234 null, although not sure we'll encounter this in practice
# Note: all the signs in the gradient calculation were determined by trial and error, not careful algebra
# - can be checked with EandG_sanity(lambda x: calc_dihedral(x, grad=True), np.random.rand(4,3))
def calc_dihedral(v1, v2=None, v3=None, v4=None, grad=False):
  """ return dihedral angle in radians - angle between planes defined by points 1,2,3 and 2,3,4 - and
    optionally gradient; refs: Wilson, Mol. Vibrations (1955), ch. 4; JCC 17, 1132 (via Wikipedia)
  """
  if v4 is None:
    v1, v2, v3, v4 = v1
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
  return dihed, np.array([dd1, dd2, dd3, dd4])


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
  return cosdihed, np.array([dd1, dd2, dd3, dd4])


## constants

# Units - atomic units, except lengths are in Angstroms by default
# length: Angstroms
# energy: Hartree; but we will often print values in kJ/mol
# charge: electron charge (q_e)
# Coulomb energy (Hartree): q_1*q_2/norm(r_2 - r1) - where r is in *Bohr* NOT Angstrom

# https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
ANGSTROM_PER_BOHR = 0.52917721067  # 2018 value is 0.529177210903
KJMOL_PER_HARTREE = 2625.499639
KCALMOL_PER_HARTREE = 627.5094740631  # updated CODATA 2018; was 627.509391
EV_PER_HARTREE = 27.2113845
AVOGADRO = 6.02214076E23  # exact value as of Nov 2018
BOLTZMANN = 1.380649e-23/4.3597447222071e-18  # J/K / J/Hartree
# recall that kB*T ~ 2.5 kJ/mol at 300K
