
# In the simplest case, ESP charge fitting is just a linear least squares problem
# ESP at r_j due to point charges q_i at R_i:
#  phiq(r_j) = \sum_i A_ji q_i, w/ A_ji = 1/(r_j - R_i)
# Least squares:
#  chi^2 = \sum_j w_j*(phiq(r_j) - phi0(r_j))^2, where phi0(r_j) is input points/grid, w_j is weight
#        = || w*(A*q - phi0) ||^2
# Restraints (quadratic):
#  append to A, phi0; e.g. A_{(jmax + i),k} = delta_ik  phi0_(jmax + i) = 0 to restrain to zero
# Constraints:
#  \sum_i B_ki q_i = c_k
#
# Solution:
# general solution to constraint problem using pseudoinverse:
#  q = B^+ * c + (1 - B^+ * B) * q1
# substitution gives
#  chi^2 = || A * (B^+ * c + P * q1) - phi0 ||^2 w/ P = (1 - B^+ * B)
# this is just the least squares problem for A1*q1 = phi1, w/ soln:
#  q1 = A1^+ * phi1 + (1 - A1^+ * A1) * x w/ A1 = A*P, phi1 = phi0 - A * B^+ * c
# take x = 0 and plug into above expression for q.
#
# Refs:
#  MMTK ChargeFit.py: rank checking, constraint consistency check, random gird generation
#  pDynamo ESPChargeFitting.py: vdW surface grid generation, iterative and non-iterative soln
# Other refs:
# - Ambertools (<http://ambermd.org/Questions/resp2.txt>)
# - MMTK
# - pDynamo/pMoleculeScripts

def resp(centers, grid, phi0, gridweight=None, restrain=0, netcharge=0.0):
  """ RESP: linear least squares determination of point charge values at
  specified positions to best reproduce ESP at specified grid points
  Weighting is applied to A and phi0 before main calculation; alternatively,
   we could form a weight matrix which multiples A and phi0 in main calc.
  Passed:
   centers: point charge positions
   grid: grid points (not necessarily in a regular grid...)
   phi0: ESP at grid points
   gridweight: weights to apply to grid points (None for no weights)
    == 'equal' to weight all grid points equally (weight = 1/phi0)
    otherwise, should be a vector (scalar weight has no effect)
   restrain: weight to apply to restraint toward q=netcharge/ncenters
   netcharge: net charge (constrained) - for now, this cannot be disabled
    since we need at least one constraint to create B and c matrices
  Returns:
   q: the point charge values
  """
  ncenters = len(centers);
  npoints = len(phi0);
  # ESP points (restraints)
  A = 1.0/(np.tile(grid, ) - np.tile(centers, ));
  # ESP point weighting
  if gridweight == 'equal':
    A = 1.0/phi0 * A;
    phi0 = np.ones_like(phi0);
  elif gridweight is not None:
    A = gridweight * A;
    phi0 = gridweight * phi0;
  # charge restraints
  if restrain > 0:
    A.append( np.eye(ncenters)*restrain );
    phi0.append( np.ones(ncenters)*netcharge/ncenters );
  # constraints: B, and c
  B, c = np.array([]), np.array([]);
  #if netcharge is not None:
  B.append( np.ones(ncenters) );
  c.append( netcharge );
  # the central calculation
  Binv, Binvcond = MPinv(B);
  P = np.eye( ) - np.dot(Binv, B);
  A1inv, A1invcond = MPinv(np.dot(A, P));
  phi1 = phi0 - np.dot(A, np.dot(Binv, c));
  q1 = np.dot(A1inv, phi1);
  q = np.dot(Binv, c) + np.dot(P, q1);
  # TODO: calculate chi^2, relative chi^2 or other goodness of fit info
  return q;


def MPinv(A, rcond=1e-15):
  """ calculate Moore-Penrose pseudoinverse of matrix A using SVD
  Code is copied from numpy.linalg.pinv, but modified to calculate
   and return condition number (ratio of largest and smallest
   singular values from SVD)
  """
  u, s, vT = np.linalg.svd(A.conjugate(), 0);
  cond = np.amax(s)/np.amin(s);  # condition number
  cutoff = rcond*maximum.reduce(s);
  for i in range(min(vT.shape[1], u.shape[0])):
    if s[i] > cutoff:
      s[i] = 1.0/s[i];
    else:
      s[i] = 0.0;
  res = np.dot(T(vT), np.multiply(s[:, np.newaxis], T(u)));
  return res, cond;
