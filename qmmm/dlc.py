# (Hybrid) Delocalized Internal Coordinates - (H)DLCs
# - DLC and related seems to be most popular coords for geom opt
#
# 1. Given set of Cartesians X...
# 1. Create set, Q = Q(X), of redundant (aka primitive) internals
#  - stretches, bends, torsions or total connection or ...
# 1. Create B matrix: B_ij = dQ_i/dX_j (depends on geom due to nonlinearity)
# 1. DLCs are S = U'*Q, where U is set of eigenvectors of G = B*B' w/ non-zero eigenvalues
#  - ' denotes transpose
#  - take 3N-6 largest eigenvalues (i.e., should be nonredundant)
#  - Note that S are linear combs of Qs (also note S = U' in the Q-basis where Q = I)
#  - Even though B changes w/ geom, DL-FIND only diagonalizes G at beginning (i.e., reuses U) and
#   in case of "breakdown", indicated by poor convergence of S_new to X_new conversion (see below)
# 1. To convert energy gradient to DLCs, g_DLC = A*g_Cart, where g_DLC = dE/dQ, d_Cart = dE/dX,
#  and A = (B_active')^{-1} = (B_active*B_active')^{-1}*B_active w/ B_active = U'*B.
#  - Hessian can also be converted, but for quasi-Newton methods, usually H is simply
#   maintained in DLCs instead of Cartesians.
#  - note that A is the (Moore-Penrose) pseudoinverse of B_active', assuming columns of B_active' are
#   linearly indep.  g_DLC = A*g_Cart is the linear least-squares optimal solution
#   to B_active'*g_DLC = g_Cart
# 1. As Q is a nonlinear fn of X, we must use an iterative procedure to recover X_new from S_new:
#  X_{k+1} = X_k + A_k [S_new - S(X_k)], where A_k = A(X_k), since A = A(B) = A(B(Q(X))). Also note
#  that S(X) = S(Q(X)) (this is the comparatively trivial transformation from Cartesians to DLCs).
#  Convergence is achieved when the term in brackets becomes small; we take X_new = X_{k_final}.
#  Convergence is still possible, but slower, if A is not updated.
# 1. Constraints: internal coord constraints are implemented by projecting them out of S
#  - project vectors C_i representing internal coord constraints onto active space:
#   Cp_i = \sum_j <U_j|C_i> U_j, then apply Gram-Schmidt to [Cp_1 ... Cp_Nc U] to yield 3N-6-Nc active
#   coordinates Ua; then we use U = [Ua C] as usual.  It appears any linear combination of internals
#   can be constrained.
#  - atoms outside active region should simply be excluded, not constrained
#  - freezing of other atoms: if cartesians are included in set of primitives, as with HDLC,
#   might be able to constrain in the same way as internals!
#  - more complicated constraints, e.g., on CM, can be implemented with Lagrange multipliers, or
#   fudged as restraints added to energy gradient (usually g_Cart, before conversion to g_DLC)
#
# JCP 110, 4986 introduces two complicated techniques yielding significant speedup for large systems
#  DL-FIND uses neither, instead relying on small fragment size to defeat N^3 scaling of matrix ops
#
# Constraining arbitrary linear combinations of internals was the primary motivation for writing this
#  code - if this was supported by DL-FIND, I'd just have written a python wrapper for DL-FIND.

# Handling disconnected geometry (i.e. multiple molecule fragments):
# - use total connection, use autoxyzs, pass connect=[] with sufficient number of bonds, or pass
#  connect='auto' to automatically connect fragments (note GAMESS $ZMAT NONVDW option basically does this)
# - it appears that diagonalization (calc_active()) will detect incompleteness of internals, due e.g. to
#  disconnected fragments, even if there are >3N-6 redundant internals
# NH_3, for example, won't produce enough DLCs w/ default parameters; one solution is to pass
#  diheds='impropers' to include dihedrals like 1,0,2,3 (where 1,2,3 are bonded exclusively to 0)

# With previous impl of calc_dihedral() using arccos, dihedrals near 0 or 180 deg caused convergence problems
#  (even with biasing of angles to removing jumps), but using arctan2 instead seems to fix the problem
# - tried using cosine of dihedral instead, but gradient vanishes at 0,180, which fails for some systems,
#  although it did work fine with HDLC for entire trypsin system

# Variations:
# * Hybrid DLC - DLCs created separately for each fragment (residue).
#  - Cartesians for each atom in fragment are included in Q (simply easier than including
#   CM coords and Euler angles)
#  - note that we need cartesians for 3 non-colinear atoms to fully specify position and orientation of
#   fragment - a single atom leaves 3 DOF of rotation, two atoms leaves one DOF of rotation about their axis
#  - poorly behaved fragments (those triggering many breakdowns) are switched from DLCs to Cartesians
# * Redundant internals (used in G03): every bond, angle, dihedral (based on guessed connectivity)
# * Natural internals: individual bonds and linear combinations of angles and dihedrals; difficult to
#  create

# Refs:
# - ../../dl-find/dlf_hdlc_hdlclib.f90
# - http://www.cse.scitech.ac.uk/ccg/software/chemshell/manual/hdlc.html
# - http://www.q-chem.com/qchem-website/manual/qchem50_manual/sect0041.html

# TODO:
# - need better approach for bias_angles()
# - consider lumping together all internals instead of separate lists for each type; this would also allow
#  unification with Cartesians class (cartesian internally represented by length 1 tuple or bare int)
# - support including bonds directly (GAMESS) - should we exclude from eigenvalue calc or project out after?

import numpy as np
from ..basics import *
from ..molecule import mol_fragments, fragment_connections, generate_internals, nearest_pairs

class DLC:

  # tolerance for iterative DLC to Cartesian conversion
  #  1e-10 seems like a reasonable value for distances in Ang and angles in rad
  eps = 1E-10
  abseps = 1E-15  # absolute convergence threshold (to handle DLCs near zero)
  diffeps = 0.0  # converge threshold of RMS( (dS_i - dS_{i-1})/S ) - for redundant internals (UT = 1)
  maxit = 60
  recalc = 0

  def __init__(self, mol, atoms=None, bonds=[], angles=[], diheds=[], xyzs=[], coord_fns=[], connect=[],
      autobonds='conn', autoangles='conn', autodiheds='conn', autoxyzs='none', constrain=[], **kargs):
    """ read geom from molecule object
    atoms: only include specified atoms from molecule
    bonds/angles/diheds: explicit coordinates to include, typically for use in constraints
    xyzs: cartesian coordinate index, not atom number, i.e., atomnum*3 + 0, 1, or 2
    coord_fns: list of fns `fn(r, grad=False)` returning scalar or list of internals and grads
    connect: additional bonds to be included in molecule connectivity, e.g. to join fragments (incl use for
      generation of angles and dihedrals); 'auto' to automatically connect fragments
    autobonds/angles/diheds: 'conn' - internals determined from connectivity,
     'none' - use only explicitly specified bonds/angles/diheds
    autobonds: 'total': include set of distances between all pairs of atoms
     for total connection scheme (no other internals used), set autoangles/diheds='none'
     or autobonds=N to connect only N nearest neighbors
    autodiheds: 'impropers' to include improper dihedrals
    autoxyzs: 'all' to include all cartesians for all atoms; default is 'none'
    Note that including cartesians will increase number of DLCs from 3*N-6 to (up to) 3*N
    This is of course desired for HDLC where total system energy is sensitive to
     translation and rotation of residue.
    TODO: center of mass as an internal
    connectivity information is obtained from molecule object on creation; molecule object
     is not needed for any subsequent operations
    """
    # atom numbers in use
    self.atoms = atoms or mol.listatoms()
    # number of atoms
    self.N = len(self.atoms)
    assert self.N > 1, "DLC() requires two or more atoms; use Cartesian() for single atom"
    # core quantities - key rule is to maintain consistency among these
    self.X = mol.r[atoms] if atoms else np.array(mol.r)  # cartesians - from molecule object by default
    self.S = None    # delocalized internal coordinates (DLCs)
    self.B = None    # Jacobian (Wilson B-matrix)
    self.A = None    # generalized inverse (pseudoinverse) of Jacobian
    self.Q = None    # redundant internals - not really needed, but always calculated with B
    self.UT = None   # active space eigenvectors
    self.C = []      # list of constraint vectors

    # default - autobonds/angles/diheds = 'conn'
    if connect == 'auto':
      frags = mol_fragments(mol)
      connect = fragment_connections(mol.r, frags) if len(frags) > 1 else []
    bonds0 = mol.get_bonds(active=atoms) + connect
    angles0, diheds0 = generate_internals(bonds0, impropers=(autodiheds == 'impropers'))
    if autobonds == 'total':
      # total connection
      bonds0 = [ (a0, a1) for ii,a0 in enumerate(self.atoms) for a1 in self.atoms[ii+1:] ]
    elif type(autobonds) is int:
      # this is a bit silly since indices will be converted back to indices into self.X
      bonds0 = [ (self.atoms[a0], self.atoms[a1]) for a0,a1 in nearest_pairs(self.X, autobonds) ]
    xyzs0 = range(3*self.N) if autoxyzs == 'all' else []
    if xyzs and atoms:
      xyz_global = [ 3*ii + jj for ii in atoms for jj in [0,1,2] ]
      xyzs = [xyz_global.index(xyz) for xyz in xyzs]

    # specification of redundant internals
    #  order (in Q vector, etc) is bonds, angles, dihedrals, cartesians, with explicit coords first
    self.bonds = self.init_coord(bonds, bonds0 if autobonds != 'none' else [])
    self.angles = self.init_coord(angles, angles0 if autoangles != 'none' else [])
    self.diheds = self.init_coord(diheds, diheds0 if autodiheds != 'none' else [])
    self.cartesians = xyzs + [ c for c in xyzs0 if c not in xyzs ]
    # custom coordinate fns
    self.coord_fns = []
    for coord in coord_fns:
      try:
        coord = coord()  # handle class constructor
      except TypeError: pass
      if hasattr(coord, 'init'):
        coord.init(self.X)  # init if necessary, as we need to call to get number of coordinates returned
      self.coord_fns.append(coord)

    # number of redundant internals
    self.nQ = len(self.bonds) + len(self.angles) + len(self.diheds) + len(self.cartesians) \
        + sum(len(coord(self.X)) for coord in self.coord_fns)

    # remaining args (e.g. eps, maxit) and constraints
    for key, val in kargs.iteritems():
      if not hasattr(self, key):
        raise AttributeError, key
      setattr(self, key, val)
    if constrain:
      self.constrain(constrain)


  def __repr__(self):
    return "DLC(atoms: %d, bonds: %d, angles: %d, diheds: %d, cartesians: %d, constraints: %d, DLCs: %d)" \
        % (self.N, len(self.bonds), len(self.angles), len(self.diheds), len(self.cartesians), len(self.C),
        len(self.S) if self.S is not None else 0)


  # rename to 'start', 'restart', or 'reset'?
  def init(self, X=None):
    """ Given cartesians, calculate new set of DLCs, i.e., diagonalize B*B^T.
    This is not done in __init__ since user may want to set constraints first.
    May be called at any point if a new set of DLCs is desired (e.g. due to breakdown).
    """
    if X is not None:
      self.X = np.array(X)
    for coord in self.coord_fns:
      if hasattr(coord, 'init'):
        coord.init(self.X)
    self.Q, self.B = self.get_QandB(self.X)
    self.UT = self.calc_active()  # make this optional?
    self.S = np.dot(self.UT, self.Q)  #self.calc_S(xyz)
    self.A = None
    return self


  def update(self, Sactive, newX=None):
    """store new DLCs, calculate corresponding cartesians and new B
    if toxyz() has already been called separately, newX can be passed explicitly
    """
    S = np.concatenate( (self.S[:len(self.C)], Sactive) )
    if newX is None:
      newX = self.toxyz(S)
    if newX is None:
      print "DLC update rejected: unable to generate Cartesians"
      return False
    self.S = S
    self.X = newX
    self.Q, self.B = self.get_QandB(self.X)
    # don't recalc unless and until needed for gradient:
    self.A = None
    return True


  # We do not handle restart (recalc of active space) here because optimizer must also restart (i.e. discard
  #  all gradient history).
  # So, the calling code should perform the restart when update() fails: this is done by simply calling
  #  init() - active space and new DLCs will be recalculated using last good set of Cartesians
  # Adaptive step size: used in DL-FIND, tried here at rev. 27 - if norm(dS) increases, reject step and
  #  retry newX = last_good_X + trust*dS*A with trust *= 0.5; if norm(dS) decr and trust < 1, trust *= 1.25
  # examining (Q - prevQ)/np.dot(B, np.ravel(dX)) can be insightful for debugging convergence problems ...
  #  negative value in this array indicates update moved coord in opposite direction expected from gradient
  def toxyz(self, S):
    """ toxyz: iterative transformation from DLCs to Cartesians
    will accept all DLCs or just active DLCs (in which case constraints are prepended)
    intended only as a helper fn for update() and for testing
    recalc = 0 or False to disable recalculation of A in loop;
     = 1 to recalculate every iteration; not sure if recalc interval > 1 is useful
    """
    if len(S) < len(self.S):
      S = np.concatenate( (self.S[:len(self.C)], S) )
    B = self.B
    dS = np.inf
    X = np.array(self.X)  # copy self.X
    for ii in range(self.maxit):
      if ii == 0:
        Q = self.Q
        # A may have already been calculated by gradfromxyz
        A = self.A if self.A is not None else self.calc_A(B)
      elif self.recalc and ii % self.recalc == 0:
        Q, B = self.get_QandB(X)
        ## TODO: remove this if not needed for TRIC
        try:
          A = self.calc_A(B)
        except np.linalg.linalg.LinAlgError:  # singular B matrix
          break
      else:
        Q = self.get_Q(X)
      # this is S(Q(X_ii))
      Sii = np.dot(self.UT, Q)
      prev_dS = dS
      dS = S - Sii
      # convergence test: for DLCs, we use largest element of dS, with adjustment by abseps to handle very
      #  small values of S - e.g., with total connection, most components of S are very close to zero for
      #  initial geometry; to handle redundant internals (UT = 1), for which dS will not in general approach
      #  zero, we can also check for small change in dS (diffeps) ... for DLCs diffeps = 0 to disable
      diff_dS = np.sqrt(np.mean(np.square( (dS - prev_dS)/S ))) if self.diffeps > 0 else 0
      max_dS = np.max(np.abs( dS/(np.abs(S)+np.abs(Sii)+(self.abseps/self.eps)) ))
      if max_dS < self.eps or diff_dS < self.diffeps:
        return X
      # using += is a nightmare for debugging
      dX = np.reshape(np.dot(dS, A), (self.N, 3))  # dS * A = A^T * dS
      X = X + dX
      ##print "(Q_n - Q_{n-1})/dQ:\n", (self.get_Q(X) - Q)/np.dot(B, np.ravel(dX))
    # ---
    # failure
    #print "(Q_n - Q_{n-1})/dQ:\n", (self.get_Q(X) - Q)/np.dot(B, np.ravel(dX))
    print("Warning: DLC to Cartesian conversion failed to converge; max_dS: %G, diff_dS: %G" \
        % ( max_dS, np.sqrt(np.mean(np.square( (dS - prev_dS)/S ))) ))
    return None


  # TODO: can we think of a better name? r()? R()? .r via __getattr__?
  def xyzs(self):
    return self.X


  def active(self):
    """ return active DLCs, excluding constrained DLCs """
    # constraint vectors appear first
    return self.S[len(self.C):]


  def nactive(self):
    """ return number of active DLCs """
    return len(self.S) - len(self.C)


  def gradfromxyz(self, gxyz):
    self.A = self.A if self.A is not None else self.calc_A(self.B)
    gS = np.dot(self.A, np.ravel(gxyz))
    return gS[len(self.C):]


  ## TODO: remove if not used
  def gradtoxyz(self, gdlc):
    gdlc = np.concatenate( (np.zeros(len(self.C)), gdlc) ) if len(gdlc) < len(self.S) else gdlc
    return np.dot(gdlc, np.dot(self.UT, self.B))


  # TODO: handle constraints!
  def hessfromxyz(self, hxyz):
    # note this omits the grad_q * dB/dxyz term (see Baker eq. 5b and discussion following)
    self.A = self.A if self.A is not None else self.calc_A(self.B)
    hS = np.linalg.multi_dot((self.A, hxyz, self.A.T))
    return hS  #[len(self.C):]


  ## constraints

  # fn for freezing bonds, angles, dihedrals (determined by length of atom list)
  # consider removing this and requiring user to do dlc.constraint([dlc.bond(...), dlc.bond(...), ...])
  def constrain(self, constraints):
    clist = [constraints] if type(constraints[0]) is int else constraints
    for c in clist:
      if len(c) == 2: self.constraint(self.bond(c))
      elif len(c) == 3: self.constraint(self.angle(c))
      elif len(c) == 4: self.constraint(self.dihed(c))
      else: raise ValueError('Atom list must have length 2, 3, or 4')


  # these fns can be used to create more complex constraints, e.g. linear combinations
  # - how to handle if constraint is already present? raise error? ignore/replace?
  # - any reason not to normalize constraint vector?
  def constraint(self, c):
    """ add a new constraint """
    if not np.any(c):
      raise ValueError('Constraint vector is null!')
    self.C.append(normalize(c))


  def bond(self, a0, a1=None):
    """ get redundant internal unit vector corresponding to bond """
    return self.get_internal(tuple(a0) if a1 is None else tuple(a0, a1), self.bonds, 0)

  def angle(self, a0, a1=None, a2=None):
    """ get redundant internal unit vector corresponding to bend angle """
    return self.get_internal(tuple(a0) if a2 is None else tuple(a0, a1, a2), self.angles, len(self.bonds))

  def dihed(self, a0, a1=None, a2=None, a3=None):
    """ get redundant internal unit vector corresponding to dihedral """
    atoms = tuple(a0) if a3 is None else tuple(a0, a1, a2, a3)
    return self.get_internal(atoms, self.diheds, len(self.bonds) + len(self.angles))


  ## private methods

  def get_internal(self, atoms, internals, offset):
    """ helper function for bond(), angle(), dihed() """
    Qunit = np.zeros(self.nQ)
    for ii, q_local in enumerate(internals):
      q_global = tuple(self.atoms[jj] for jj in q_local)
      if q_global == atoms or q_global == atoms[::-1]:
        Qunit[ii + offset] = 1.0
        break
    return Qunit


  def init_coord(self, c1, c0):
    """ for bonds, angle, or dihedrals, append to c1 the coords in set c0 that are not already in c1,
    then convert atom numbers to indicies into self.atoms
    """
    c = c1 + [ b for b in c0 if b not in c1 and b[::-1] not in c1 ]
    return [ tuple([ self.atoms.index(a) for a in b ]) for b in c ]


  def get_QandB(self, r):
    """ get redundant internals and Jacobian (B) """
    Q = np.zeros(self.nQ)
    B = np.zeros((self.nQ, 3*self.N))
    ii = 0
    # bonds
    for (a0, a1) in self.bonds:
      Q[ii], grad = calc_dist(r[a0], r[a1], grad=True)
      B[ii, 3*a0:3*(a0+1)] = grad[0]
      B[ii, 3*a1:3*(a1+1)] = grad[1]
      ii += 1
    # angles
    for (a0, a1, a2) in self.angles:
      Q[ii], grad = calc_angle(r[a0], r[a1], r[a2], grad=True)
      B[ii, 3*a0:3*(a0+1)] = grad[0]
      B[ii, 3*a1:3*(a1+1)] = grad[1]
      B[ii, 3*a2:3*(a2+1)] = grad[2]
      ii += 1
    # dihedrals
    for (a0, a1, a2, a3) in self.diheds:
      #Q[ii], grad = self.dihed_fn(r[a0], r[a1], r[a2], r[a3], ref=self.Q[ii], grad=True)
      Q[ii], grad = calc_dihedral(r[a0], r[a1], r[a2], r[a3], grad=True)
      B[ii, 3*a0:3*(a0+1)] = grad[0]
      B[ii, 3*a1:3*(a1+1)] = grad[1]
      B[ii, 3*a2:3*(a2+1)] = grad[2]
      B[ii, 3*a3:3*(a3+1)] = grad[3]
      ii += 1
    rflat = np.ravel(r)
    for a0 in self.cartesians:
      Q[ii] = rflat[a0]
      B[ii, a0] = 1.0
      ii += 1
    for coord in self.coord_fns:
      q, b = coord(r, grad=True)
      Q[ii:ii+len(q)], B[ii:ii+len(q),:] = q, np.reshape(b, (len(q), 3*len(r)))  # flatten gradient x,y,z
      ii += len(q)
    # ---
    return self.bias_angles(Q), B


  def get_Q(self, r):
    """ get just Q, without Jacobian """
    Qb = [ calc_dist(r[a0], r[a1]) for (a0, a1) in self.bonds ]
    Qa = [ calc_angle(r[a0], r[a1], r[a2]) for (a0, a1, a2) in self.angles ]
    #dihed_offset = len(self.bonds) + len(self.angles)
    #Qd = [ self.dihed_fn(r[a0], r[a1], r[a2], r[a3], ref=self.Q[dihed_offset + ii]) for ii, (a0, a1, a2, a3) in enumerate(self.diheds) ]
    Qd = [ calc_dihedral(r[a0], r[a1], r[a2], r[a3]) for (a0, a1, a2, a3) in self.diheds ]
    rflat = np.ravel(r)
    Qx = [ rflat[a0] for a0 in self.cartesians ]
    Qfn = flatten([ coord(r) for coord in self.coord_fns ])
    return self.bias_angles(np.array(Qb + Qa + Qd + Qx + Qfn))


  # jumps in dihedral angles prevent convergence in toxyz()
  # TODO: cleaner way to do this? pass bias to calc_dihed?
  # TODO: bend angles too?  if angle is out of range (what range?), we need to do a restart
  #return q - 2*np.pi if q - ref > np.pi else (q + 2*np.pi if q - ref < -np.pi else q)
  def bias_angles(self, Q):
    """ adjust dihedral angles in Q to be within pi of angles in self.Q """
    if self.Q is not None:
      ii0 = len(self.bonds) + len(self.angles)
      for ii in range(ii0, ii0 + len(self.diheds)):
        if Q[ii] - self.Q[ii] > np.pi: Q[ii] -= 2*np.pi
        elif Q[ii] - self.Q[ii] < -np.pi: Q[ii] += 2*np.pi
    return Q


  def calc_A(self, B):
    """ calculates pseudoinverse of Ba', assuming Ba' has full column rank (all columns linearly indep.)
    Ba (B_active) is B matrix for active DLCs (B is B matrix for primitive internals)
    in general, SVD (np.linalg.pinv) must be used to find pseudoinverse
    """
    Ba = np.dot(self.UT, B)
    return np.dot( np.linalg.inv(np.dot( Ba, Ba.T )), Ba )


  def calc_active(self):
    """ diagonalize G = B*B' to determine active space """
    G = np.dot(self.B, self.B.T)
    eigval, eigvec = np.linalg.eigh(G)  # B*B' is symmetric
    # The LAPACK routine used by numpy's eigh() always returns eigenvalues in ascending order, although the
    #  numpy doc says eigenvalues may be unordered.  We will assume the eigenvalues are ordered.
    nS = np.sum(np.abs(eigval) > 1e-10)
    if not( 3*self.N - 6 <= nS <= 3*self.N ):
      print "Warning: invalid number of pre-constraint DLCs (%d DLCs, %d atoms)" % (nS, self.N)
    UT = eigvec.T[-nS:]
    if self.C:
      # project out constraints
      # project constraints onto active space
      Cps = np.array([ np.sum([np.dot(Uj, Ci) * Uj for Uj in UT], 0) for Ci in self.C ]).T
      # use QR decomp to perform Gram-Schmidt
      q,r = np.linalg.qr(np.hstack( (Cps, UT.T) ))
      # form [constraints, active space]
      # TODO: verify that T(q) here is correct!
      ###UT = np.vstack( (self.C, T(q)[-(nS - len(self.C)):]) )
      UT = np.vstack( (self.C, q.T[len(self.C):nS]) )
    # ---
    return UT


# Using identity matrix (actually just scalar 1.0) for UT matrix in DLCs gives redundant internal coordinates

# In general, we can't expect tight convergence in DLC.toxyz() with redundant internals (X will converge
#  to least squares optimal solution for given Q), so convergence is detected by small change in dS (=dQ)
#  between iterations (threshold is diffeps)
class Redundant(DLC):

  def __init__(self, *args, **kwargs):
    kwargs['diffeps'] = kwargs.get('diffeps', 1E-15)
    # note that super() only works for new style classes (which must inherit `object`)
    DLC.__init__(self, *args, **kwargs)

  def __repr__(self):
    return "Redundant(atoms: %d, bonds: %d, angles: %d, diheds: %d, cartesians: %d, constraints: %d, internal coords: %d)" \
        % (self.N, len(self.bonds), len(self.angles), len(self.diheds), len(self.cartesians), len(self.C),
        len(self.S) if self.S is not None else 0)

  def calc_A(self, B):
    """ calculates pseudoinverse of `B`, w/o assuming it has full column rank, using SVD (np.linalg.pinv) """
    Ba = np.dot(self.UT, B)
    return np.dot( np.linalg.pinv(np.dot( Ba, Ba.T )), Ba )

  def calc_active(self):
    UT = 1.0
    if self.C:
      UT = np.eye(self.nQ)
      # project out constraints
      # project constraints onto active space
      Cps = np.array([ np.sum([np.dot(Uj, Ci) * Uj for Uj in UT], 0) for Ci in self.C ]).T
      # use QR decomp to perform Gram-Schmidt
      q,r = np.linalg.qr(np.hstack( (Cps, UT.T) ))
      # form [constraints, active space]
      UT = np.vstack( (self.C, q.T[len(self.C):self.nQ]) )
    # ---
    return UT


class Cartesian:
  """ emulate DLC object for cartesian coordinates; to use cartesians with some residues for HDLC """

  def __init__(self, mol, atoms=None):
    # note that ravel() returns a view if possible, so must explicitly copy array if a copy is needed
    self.X = mol.r[atoms] if atoms is not None else np.copy(mol.r)
    self.atoms = atoms if atoms is not None else mol.listatoms()

  def __repr__(self):
    return "Cartesian(atoms: %d)" % len(self.atoms)

  def init(self, X=None):
    if X is not None:
      self.X = np.copy(X)
    return self

  # note that here newX will equal Sactive, so it's ignored
  def update(self, Sactive, newX=None):
    self.X = np.reshape(Sactive, (-1,3))

  def toxyz(self, S):
    return np.reshape(S, (-1,3))

  def xyzs(self):
    return self.X

  def active(self):
    return np.ravel(self.X)

  def nactive(self):
    return np.size(self.X)

  def gradfromxyz(self, gxyz):
    return np.ravel(gxyz)


# TODO: I think we can eliminate self.groups and just use self.dlcs[].atoms
class HDLC:
  # partitioning into dlcs:
  # - fully automatic: use residue info from molecule object; number of residues per dlc can be specified
  # - of course, user can specify complete assignment of atoms to dlcs
  # - what to do when user specifies only assignment of some atoms?  No choice but to use residue info from
  #  molecule object; as in fully automatic case, but atoms from residues including user specified atoms
  #  are added to "user" dlcs

  def __init__(self, mol, groups=[], autodlc=0, dlcoptions={}):
    """ groups: list of dlc objects or lists of atom numbers
    autodlc: number of residues per DLC or 0 (or False) to disable automatic
      creation of DLCs
    """
    # TODO: using only a subset of atoms in mol
    self.N = mol.natoms
    # partition according to residues before examining user specified groups
    resgroups = [ [ a for res in mol.residues[ii:ii+autodlc] for a in res.atoms ]
        for ii in range(0, mol.nresidues, autodlc) ] if autodlc else []
    # expand user groups and remove overlapping groups from resgroups
    # note the inclusion of cartesians with internals to give 3N instead of
    #  3N-6 DLCs for each group
    self.groups = []
    self.dlcs = []
    dlcoptions['autoxyzs'] = dlcoptions.get('autoxyzs', 'all')
    for ii, group in enumerate(groups):
      if hasattr(group, 'atoms'):
        self.dlcs.append(group)
        group = group.atoms
      elif len(group) == 1:
        self.dlcs.append( Cartesian(mol, atoms=group) )
      else:
        self.dlcs.append( DLC(mol, atoms=group, **dlcoptions) )
      self.groups.append(group)
      for jj, resgroup in enumerate(resgroups):
        # remove all atoms in user group from resgroups
        resgroups[jj] = difference(resgroup, group)

    # add remaining residue groups after removing empties
    if autodlc:
      resgroups = [x for x in resgroups if x]
      self.groups.extend(resgroups)
      for resgroup in resgroups:
        if len(resgroup) == 1:
          self.dlcs.append(Cartesian(mol, atoms=resgroup))
        else:
          self.dlcs.append(DLC(mol, atoms=resgroup, **dlcoptions))

      # make sure all atoms assigned
      assigned = sorted([a for group in self.groups for a in group])
      assert assigned == range(self.N), \
          "Atoms %r not assigned to HDLC groups!" % list(set(range(self.N)) - set(assigned))


  def __repr__(self):
    return "HDLC(atoms: %d, residues: %d)" % (self.N, len(self.dlcs))


  def find_dlc(self, atoms):
    """ find DLC object for atom(s); all atoms must be in same DLC object """
    atoms = set([atoms] if type(atoms) is int else atoms)
    for ii, group in enumerate(self.groups):
      if len(atoms - set(group)) == 0:
        return self.dlcs[ii]
    return None


  def init(self, X=None):
    for ii, dlc in enumerate(self.dlcs):
      dlc.init( X[self.groups[ii]] if X is not None else None )
    return self


  def update(self, Sactive):
    """ We do not update any dlc objects until we verify that all can be updated successfully.  Otherwise,
      failing dlc objects are re-inited and we return False to indicate that optimizer must be restarted (new
      cartesians gradients can still be used, but gradfromxyz must be called again and result applied to new
      set of actives)
    """
    failed = np.zeros(len(self.dlcs), dtype=np.uint)
    Xs = []
    S = []
    jj = 0
    for ii, dlc in enumerate(self.dlcs):
      nactive = dlc.nactive()
      S.append(Sactive[jj:jj+nactive])
      jj += nactive
      Xs.append( dlc.toxyz(S[ii]) )
      if Xs[-1] is None:
        failed[ii] = 1
        dlc.init()
    if np.any(failed):
      self.failed = getattr(self, 'failed', 0) + failed
      print("%d of %d DLCs reinitialized due to update failure" % (np.count_nonzero(failed), len(self.dlcs)))
      return False
    for ii, dlc in enumerate(self.dlcs):
      dlc.update(Sactive=S[ii], newX=Xs[ii])
    return True
    # failed can be added to internal array of counters
    #  to track number of failures for each residue


  def active(self):
    """ return active HDLCs """
    return np.hstack( [dlc.active() for dlc in self.dlcs] )


  def xyzs(self):
    """ return cartesians """
    xyzs = np.zeros((self.N, 3))
    for ii, dlc in enumerate(self.dlcs):
      xyzs[self.groups[ii]] = dlc.X
    return xyzs


  def gradfromxyz(self, grad):
    return np.hstack([ dlc.gradfromxyz( grad[self.groups[ii]] ) for ii, dlc in enumerate(self.dlcs) ])


  def constrain(self, constraints):
    clist = [constraints] if type(constraints[0]) is int else constraints
    for c in clist:
      dlc = self.find_dlc(c)
      if dlc is not None:
        dlc.constrain(c)
