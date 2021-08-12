from chem.molecule import *
from chem.opt.dlc import *
from chem.io import load_molecule
from chem.data.test_molecules import water, ethanol, C2H3F

## Notes

# NEXT:
# - learn about optimization, make notes, (probably) implement custom optimizer
# - plot E,G vs. optim step; try forcing reinit of optimizer and/or DLCs
# - work a little more on TRIC-type coords
# - test constraints with redundant internals (?)

# Observation: great progress often made after reinitializing DLCs ... maybe force reinit when gradient falls below threshold or RMSD to last geometry exceeds threshold?
# - first verify this is really the case by plotting E and RMS grad vs. optim step
# - also try forcing restart of optimizer w/o reinit of DLCs

# TRIC-type coordinates (HDLC w/ center of mass and orientation instead of all cartesians)
# - as implemented in geomeTRIC, too complicated for now ...
# - centroid, MeanRotVector: seems MeanRotVector is the main problem, although even just centroid seems to
#  make dQ_actual/dQ_expected messy
# ... recalc=1 improves performance - note that recalc has no effect on derivs of centroid or MeanRotVector
# - what about using x,y,z skew instead of rotation?  or some linear combination of points?
# ... rotation is arguably unique in that it has no effect on energy (for isolated residue)
# - does DLC make off-diagonal elements of the hessian smaller? maybe not ... need to learn more about optimization!
#  - in any case, let's reproduce some Cartesian vs DLC result to make sure our impl is correct; PDB 1L2Y?
#  - I suspect that there might not be a significant performance diff between DLCs and Cartesians for a globular system
#  - also, for MM only opt, DLC calculations can be slower than MM calc, so cartesians will be faster even if 2x more iterations (do some quick profiling!)
#  - for DLC opt, we should probably set very low tolerances and handle termination ourselves based on RMS cartesian grad
#  - can we use a quantity with units of distance for angle and dihedral coordinates?

# Refs:
# - https://github.com/leeping/geomeTRIC
# - 10.1063/1.4952956

# - Lagrange multipliers to handle constraints not initially satisfied?  See http://www.q-chem.com/qchem-website/manual/qchem50_manual/sect0043.html


# try rms_cartesian_grad_tol = 0.1 / KCALMOL_PER_HARTREE w/ DLC.gradtoxyz()
# optimization usually seems to terminate on ftol ... so should we not worry much about gtol?

# gradtoxyz(): seems reasonable, but not sure if correct in general or useful
# - geomeTRIC seems to use full B matrix (i.e. internals not DLCs), but I'm not sure that will work if constraining linear combinations of internals)
# - doesn't recover grad as A*g_xyz is least squares optimal soln to Ba*g_dlc

# NOTE: HDLC with total connection (trypsin) - some failures returning to initial coords w/ abseps = 1E-15; 1E-14 is OK (consider making that the new default)

# Should be able to prevent rotations while allowing translations by constraining, e.g., r_C - r_CA and r_N - r_CA (as vectors, not bond lengths!)

# Old:

# note Psi4 has option to use linear combinations of coords, or centroid + points along 1st and 2nd principle
#  axes ... not surprisingly, derivative computation is not implemented for the latter option

# ... cos_dihedral doesn't work for angles near 180 because gradient goes to zero!  ... so why did it work
#  for HDLC(trypsin)?  Maybe by simply removing those angles from the active space ... since cartesians
#  included, still enough coords for complete active space

# - normal mode coordinates? ... Baker paper (ref Pulay) says any two coord systems related by linear
#  transformation are equivalent for gradient-optimization methods;
#  someone tried it any way: J Chem Phys 117, 4126 ... no consistent advantage over redundant internals


def dlc_test_1(mol, constraints, dlc=None, **kwargs):
  """ Test: convert to DLC with constraint, add random perturbation, confirm constraint satisfied, then undo """
  clist = [constraints] if type(constraints[0]) is int else constraints
  mol = load_molecule(mol) if type(mol) is str else mol
  dlc = DLC(mol, **kwargs) if dlc is None else dlc
  r0 = np.array(mol.r)  # copy
  b0 = [mol.coord(c) for c in clist]
  dlc.constrain(clist)
  dlc.init(mol.r)
  S0 = dlc.active()

  # compute random cartesian displacement and use gradfromxyz to convert to perturbation of S
  dr = 0.1*(np.random.rand(*np.shape(mol.r)) - 0.5)
  dS = dlc.gradfromxyz(dr)
  dlc.update(S0 + dS)

  # random perturbation
  #dlc.update( S0*(0.975 + 0.05*np.random.rand(*np.shape(S0))) )
  mol.r = dlc.xyzs()
  b1 = [mol.coord(c) for c in clist]
  assert np.allclose(b1, b0, rtol=1E-05, atol=1E-08), "Constraint(s) not satisified: {} != {}".format(b1, b0)
  dlc.update(S0)
  mol.r = dlc.xyzs()
  assert np.allclose(mol.r, r0, rtol=1E-05, atol=1E-08), "Test failed:\n{}\n!=\n{}".format(mol.r, r0)
  return dlc


def dlc_test_2(mol, dlc=None, **kwargs):
  """ Test: generate random perturbation to cartesians, convert to DLCs, verify that cartesians are recovered """
  mol = load_molecule(mol) if type(mol) is str else mol
  dlc = DLC(mol, **kwargs) if dlc is None else dlc
  dlc.init(mol.r)
  Q0 = dlc.Q  # TODO: should have a get_internal_vals fn in molecule
  # random displacement of cartesians
  dr = 0.1*(np.random.rand(*np.shape(mol.r)) - 0.5)
  r1 = mol.r + dr
  Q1 = dlc.Q + np.dot(dlc.B, np.ravel(dr)).T
  #Q1 = dlc.get_Q(r1)
  S1 = np.dot(dlc.UT, Q1)
  dlc.update(S1)
  print "Relative error for internals:\n", (dlc.Q - Q1)/Q0
  # for use w/ Q1 = dlc.get_Q(r1)
  mol.r = dlc.xyzs()
  print "Relative error for cartesians:\n", (r1 - align_atoms(dlc.xyzs(), r1))/r1
  #assert np.allclose(mol.r, r1), "Test failed:\n{}\n!=\n{}".format(mol.r, r1)


# if these prove useful, we can put in dlc.py (any other options?)  Otherwise maybe just leave in dlc_test

def centroid(r, grad=False):
  centroid = np.mean(r, axis=0)
  ones = np.tile(1.0/len(r), len(r))
  zeros = np.zeros(len(r))
  G = [np.column_stack((ones,zeros,zeros)), np.column_stack((zeros,ones,zeros)), np.column_stack((zeros,zeros,ones))]
  return (centroid, G) if grad else centroid


# exponential map (axis-angle) rotation
class ExpMapRot:

  def init(self, r):
    self.r0 = r

  def __call__(self, r, grad=False):
    # Rotation matrix to axis-angle (expmap) ref: https://arxiv.org/pdf/1312.0788.pdf
    M = alignment_matrix(r, self.r0)
    R = M[:3,:3]
    ang = np.arccos((np.trace(R) - 1)/2)
    v = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])*ang/(2*np.sin(ang))
    #v = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])/np.sqrt(2 - (np.trace(R) - 1)**2)
    return expmap_rot(v, grad)


# Rotation matrix: (a1,a2,a3 assumed to be small)
# eye(3) + [[  0,  a3, -a2],
#           [-a3,   0,  a1],
#           [ a2, -a1,   0]]
# ... rather, this is the average of axis-angle rotation vectors v = cross(r,r0)/|r|/|r0| ~ cross(r,r0)/|r0|^2
#  where angle is arcsin(|v|)
# ... but this will match actual rotation vector only for planar system in plane normal to rotation axis - so
#  what about projecting system onto xy, xz, yz planes and calculating average rot angle for each?  Actually,
#  the only change necessary for this is to project the r0^2 factor
# Note: sum(da1) = 0 (also da2, da3), which feels right ... global translation leaves value unchanged
class MeanRotVector:

  def init(self, r):
    r0 = r - np.mean(r, axis=0)
    self.r0 = r0
    C = 1/(len(r0) * np.sum(r0*r0, axis=1))
    D0 = r0*C[:,None]
    self.D0 = D0
    # term accounting for subtraction of centroid of r (subtraction of centroid of r0 does not affect grad)
    Dc = np.sum(D0, axis=0)/len(r0)
    zeros = np.zeros(len(r0))
    self.da1 = np.column_stack((zeros, D0[:,2] - Dc[2], -D0[:,1] + Dc[1]))
    self.da2 = np.column_stack((-D0[:,2] + Dc[2], zeros, D0[:,0] - Dc[0]))
    self.da3 = np.column_stack((D0[:,1] - Dc[1], -D0[:,0] + Dc[0], zeros))

  def __call__(self, r, grad=False):
    r = r - np.mean(r, axis=0)
    a1 = np.sum(self.D0[:,2]*r[:,1] - self.D0[:,1]*r[:,2])
    a2 = np.sum(self.D0[:,0]*r[:,2] - self.D0[:,2]*r[:,0])
    a3 = np.sum(self.D0[:,1]*r[:,0] - self.D0[:,0]*r[:,1])
    return ([a1, a2, a3], [self.da1, self.da2, self.da3]) if grad else [a1, a2, a3]


def test_rigid_rot(scale=0.1, offset=0.0, angle=None, dir=None):
  mol = load_molecule(ethanol)
  r0 = mol.r - np.mean(mol.r, axis=0) + offset*(np.random.random(3) - 0.5)
  coord = SmallRigidRotation()
  coord.init(r0)
  q0, grad = coord(r0, grad=True)
  dir = normalize(np.random.random(3) - 0.5) if dir is None else dir
  angle = 0.1*(np.random.random(1)[0] - 0.5) if angle is None else angle
  R = rotation_matrix(dir, angle)
  centroid = np.mean(r0, axis=0)
  r1 = np.dot(r0 - centroid, R) + centroid
  q1 = coord(r1)
  dq = [np.sum(g * (r1 - r0)) for g in grad]
  print("rot, q1-q0, dq:", dir*np.sin(angle), np.array(q1) - np.array(q0), np.array(dq))


# mol = load_molecule(ethanol)
# dlc = DLC(mol, coord_fns=[centroid, MeanRotVector]).init()
# hdlc = HDLC(mol, autodlc=1, dlcoptions=dict(coord_fns=[centroid, MeanRotVector], autoxyzs='none', recalc=0)).init()


# - we'll assume .hes file has already been generated
from chem.io.tinker import read_tinker_hess, write_tinker_key, tinker_EandG

# test_hessian("/home/mwhite/qc/2016/GLY_GLY")
# ... seems DLCs don't make Hessian much more diagonal, only RICs
def test_hessian(prefix, **kwargs):
  mol = load_molecule(prefix + ".xyz")
  hess = read_tinker_hess(prefix + ".hes")

  dlc = DLC(mol, **kwargs)
  dlc.init()
  H = dlc.hessfromxyz(hess)
  #H = hess

  D = np.diag(H)
  offD = H - np.diag(D)

  print("Norm of diagonal: ", np.linalg.norm(D))
  print("Off-diag Frobenius norm: ", np.linalg.norm(offD))  # Frobenius norm
  # operator 2-norm - largest singular value == sqrt(largest eigenvalue of dot(M.T,M)
  print("Off-diag operator 2-norm: ", np.linalg.norm(offD, ord=2))
  print("Diagonal/row sums: ", np.abs(D)/np.sum(np.abs(offD), axis=1)) # diagonal dominance
  #np.sqrt(np.sum(H*H))
  return H


from chem.io.pdb import copy_residues
def load_xyz_pdb(prefix='test1/prepared'):
  trypsin_pdb = load_molecule(prefix + '.pdb')
  trypsin = load_molecule(prefix + '.xyz') #, charges='generate')
  return copy_residues(trypsin, trypsin_pdb)

from chem.opt.optimize import optimize, XYZ
def test_optim():
  mol = load_xyz_pdb("/home/mwhite/qc/2016/GLY_GLY")

  #coords = XYZ(mol) # F = -0.17117385259976131 (bad) 153 iter, even with ftol=1E-12
  coords = DLC(mol)  # F = -0.21489224214654024 - 216 iter, 2 breakdowns
  #coords = DLC(mol, recalc=1)  # reaches invalid geom at step 54 and Tinker gradient calc fails
  #coords = DLC(mol, autobonds='total', autoangles='none', autodiheds='none')  # F = -0.21489240150670511 - 385 iter, 2 breakdowns; recalc=1 gives same result, but more iterations
  #coords = HDLC(mol, autodlc=1, dlcoptions=dict(recalc=0))  # F = -0.21489144534571591 - 229 iter, no breakdowns; basically the same with recalc=1
  #coords = HDLC(mol, autodlc=1, dlcoptions=dict(autobonds='total', autoangles='none', autodiheds='none', recalc=0)) # F = -0.21489208278637537 - 224 iter, 1 breakdown
  #coords = Redundant(mol)  # F = -0.21489080790505649 - 204 iter, 4 breakdowns, incl at iter 0; basically the same with recalc=1
  # Note that all runs terminated on ftol, not gtol, so magnitude of coordinates shouldn't matter
  # F = -0.21489... = good structure

  #coords = HDLC(mol, autodlc=1, dlcoptions=dict(coord_fns=[centroid, MeanRotVector], autoxyzs='none', recalc=1))
  # ... manages ~60 iter, but eventually reaches geometry where first iter always fails

  #coords.constrain((9,10,11))  # DLC, no constraint, 0.000114 H/Ang final RMS cartesian grad; 0.0097 w/ constraint
  # grad projecting out constraint: 0.000067  (0.000066 w/o constraint)

  tinker_args = Bunch(prefix="/home/mwhite/qc/2016/trypsin1/test1/tmp/GLY_GLY")
  tinker_key = "parameters  /home/mwhite/qc/tinker/params/oplsaa.prm"
  write_tinker_key(tinker_key, tinker_args.prefix + ".key")
  res, r = optimize(mol, fn=tinker_EandG, fnargs=tinker_args, coords=coords)
  assert res.success, "Minimization failed!"
  return mol, r

#c = Chemvis(Mol(mol, [r], [ VisGeom(style='lines') ])).run()


def test_diheds(dlc, scale=0.1):
  S0 = dlc.active()
  dr = scale*(np.random.rand(dlc.N, 3) - 0.5)
  dS = dlc.gradfromxyz(dr)
  ok = dlc.update(S0 + dS)
  print "update to S0 + dS: ", ok
  if not ok:
    import pdb; pdb.set_trace()
  ok = dlc.update(S0)
  print "update to S0: ", ok
  return dlc


def test_1():
  dlc_test_2(C2H3F)
  dlc_test_1(water, (0,2))
  dlc_test_1(water, [(0,1), (0,2)])
  dlc_test_1(water, (1,0,2))
  dlc_test_1(C2H3F, (0,2))


def test_planar():
  print("Testing DLC for planar molecule...")
  nh3 = Molecule(r=[[0,0,0], [1,0,0], [0,1,0], [-1/np.sqrt(2),-1/np.sqrt(2),0]], znuc=[7,1,1,1])
  nh3.set_bonds(guess_bonds(nh3.r, nh3.znuc))
  print("This should fail for planar molecule:")
  dlc = DLC(nh3).init()
  #assert dlc is None, "DLC init did not return None for planar molecule with default internal coords"
  print("This should work:")
  dlc = DLC(nh3, autodiheds='impropers').init()
  #assert dlc is not None, "DLC init failed for planar molecule despite autodiheds='impropers'"
  dlc_test_1(nh3, dlc=dlc)


def test_mol_box():
  mol = load_molecule(ethanol, center=True)
  mol.residues = [Residue(name='EtOH', atoms=mol.listatoms())]
  molbox = tile_mol(mol, 10.0, [3, 3, 3], rand_rot=True, rand_offset=1.0)

  dlc1 = DLC(molbox, connect='auto')
  print("Testing DLC(connect='auto')...")
  dlc_test_1(molbox, dlc=dlc1)

  hdlc1 = HDLC(molbox, autodlc=1, dlcoptions=dict(recalc=1))
  print("Testing HDLC...")
  dlc_test_1(molbox, dlc=hdlc1)

  hdlc2 = HDLC(molbox, autodlc=1, dlcoptions=dict(recalc=1, autobonds='total', autoangles='none', autodiheds='none', abseps=1e-14))
  print("Testing HDLC with autobonds='total'...")
  dlc_test_1(molbox, dlc=hdlc2)


def all_tests():
  test_1()
  test_planar()
  test_mol_box()


def dlc_test_3(dlc, nit=100, scale=0.1, mask=None):
  S0 = dlc.active()
  r0 = dlc.xyzs()
  for ii in range(nit):
    # compute random cartesian displacement and use gradfromxyz to convert to perturbation of S
    dr = scale*(np.random.rand(*np.shape(r0)) - 0.5)
    dS = dlc.gradfromxyz(dr)
    if mask:
      dS = setitem(np.zeros_like(dS), mask, dS[mask])
    if not dlc.update(S0 + dS):
      print("DLC update (S0 + dS) failed (iteration %d)" % ii)
      break
    #if not np.allclose(dlc.xyzs(), r0 + dr, rtol=1E-05, atol=1E-08):  -- always fails
    if not dlc.update(S0):
      print("DLC update (S0) failed (iteration %d)" % ii)
      break


# How to move only diheds?
# - Redundant doesn't work if Cartesians included (but seems to otherwise)
# - DLC (incl Cartesians) + constraints on fixed atoms, bonds, and angles seems to work
# - other option is non-redundant internals (Z-matrix)
from chem.test.common import quick_load
if 0:
  arg1 = quick_load('ARG1.pdb')
  active = select_atoms(arg1, 'sidechain and name != "HA"')
  fzn = arg1.listatoms(exclude=active)
  fznxyz = flatten([[3*ii, 3*ii+1, 3*ii+2] for ii in fzn])
  bonds, angles, diheds = arg1.get_internals(active=active, inclM1=True)
  dlc = DLC(arg1, xyzs=fznxyz, bonds=bonds, angles=angles, diheds=diheds, autobonds='none', autoangles='none', autodiheds='none', recalc=1)
  dlc.constrain([[ii] for ii in fzn])
  dlc.constrain(bonds)
  #indepangles = [(1, 8, 9), (1, 8, 15), (1, 8, 16), (7, 1, 0), (7, 1, 2), (7, 1, 8), (8, 1, 0), (8, 1, 2), (8, 9, 10), (8, 9, 17), (8, 9, 18), (9, 8, 15), (9, 8, 16), (9, 10, 11), (9, 10, 19), (9, 10, 20), (10, 9, 17), (10, 9, 18), (10, 11, 12), (10, 11, 21), (11, 10, 19), (11, 10, 20), (11, 12, 13), (11, 12, 14), (12, 11, 21), (12, 13, 22), (12, 13, 23), (12, 14, 24), (12, 14, 25), (22, 13, 23), (24, 14, 25)]
  dlc.constrain(angles)
  #pdb.run('dlc.init()')
  import pdb; pdb.set_trace()
  dlc.init()
  Q0 = dlc.Q
  S0 = dlc.active()
  dr = 0.1*(np.random.rand(*np.shape(arg1.r)) - 0.5)
  dS = dlc.gradfromxyz(dr)
  dlc.update(S0 + dS)
  Q1 = dlc.Q


if 1:
  arg1 = quick_load('ARG1.pdb')
  active = select_atoms(arg1, 'sidechain and name != "HA"')
  fzn = arg1.listatoms(exclude=active)
  fznxyz = flatten([[3*ii, 3*ii+1, 3*ii+2] for ii in fzn])
  bonds, angles, diheds = arg1.get_internals(active=active, inclM1=True)
  redun = Redundant(arg1, xyzs=fznxyz, bonds=bonds, angles=angles, diheds=diheds, autobonds='none', autoangles='none', autodiheds='none', diffeps=1E-13, recalc=1)

  #redun = Redundant(arg1, diffeps=1E-13, recalc=1)
  dihedstart = len(redun.bonds) + len(redun.angles)
  dihedidx = range(dihedstart, dihedstart + len(redun.diheds))
  redun.init()
  import pdb; pdb.set_trace()
  dlc_test_3(redun, mask=dihedidx)

  S0 = redun.active()
  dr = 0.1*(np.random.rand(*np.shape(arg1.r)) - 0.5)
  dS = redun.gradfromxyz(dr)
  #A = redun.calc_A(redun.B)
  #proj = np.dot(redun.B, A.T)
  #dS = np.dot(proj, dS0)
  #dSdihed = setitem(np.zeros_like(S0), dihedidx, dS[dihedidx])
  redun.update(S0 + dS)




if 0:  #__name__ == '__main__':
  #np.set_printoptions(suppress=True)
  #np.set_printoptions(suppress=True, precision=4, linewidth=150)

  if 0:
    hdlc = HDLC(mol).init()
    S0 = hdlc.active()

    diheds0 = [calc_dihedral([dlc.X[ii] for ii in dihed]) for dlc in hdlc.dlcs for dihed in dlc.diheds]

    dr = 0.1*np.random.rand(*np.shape(mol.r))
    dS = hdlc.gradfromxyz(dr)
    hdlc.update(S0 + dS)

    diheds1 = [calc_dihedral([dlc.X[ii] for ii in dihed]) for dlc in hdlc.dlcs for dihed in dlc.diheds]

    maxdiff = np.max( np.abs( (np.array(diheds0) - np.array(diheds1) + np.pi) % (2*np.pi) - np.pi ) )

    if maxdiff > np.pi/2:
      print "Dihedral flipped!!!"


  mol = molecule(xyz=water)

  if 0:
    dlc = DLC(mol, autobonds='total', autoxyzs='all')
    dlc.init()

    dlc2 = DLC(mol, autobonds='total')
    dlc2.constraint(dlc2.angle(1, 0, 2))
    dlc2.init()

    np.random.seed(137)
    rnd = np.random.rand(*np.shape(dlc.active()))
    rnd2 = np.random.rand(*np.shape(dlc2.active()))


  if 0:
    r = mol.r
    # 9 - 3 - 2 - 1 dihedral
    dihed0, dd = calc_dihedral(r[8], r[2], r[1], r[0], grad=1)

    dr1 = np.array( [0.019, -0.023, 0.034] )  # random
    dihed1 = calc_dihedral(r[8], r[2], r[1]+dr1, r[0])
    print "direct: ", dihed1 - dihed0
    print "grad: ", np.dot(dd[2], dr1)


  if 0:
    dlc = DLC(mol)
    dlc.init()
    # add random perturbation to cartesians
    xyz0 = dlc.X
    #np.random.seed(137)
    xyz = xyz0 + 0.02*(np.random.rand(*np.shape(xyz0)) - 0.5)
    # compare Q(xyz) to Q(xyz0) + B*(xyz-xyz0)
    Q0 = np.array(dlc.Q);  # copy
    Qgrad = Q0 + np.dot(dlc.B, np.ravel(xyz-xyz0))
    dlc.init(xyz)
    Qdirect = dlc.Q
    print "Internals: ", dlc.bonds, dlc.angles, dlc.diheds
    print "\nQdirect: ", Qdirect
    print "\nQdirect - Q0: ", Qdirect - Q0
    print "\nQgrad - Q0: ", Qgrad - Q0
    print "\ndifference: ", Qdirect - Qgrad
    print "\nrelative diff: ", (Qdirect - Qgrad)/Qdirect


  if 0:
    #mol = molecule(xyz=C2H3F)
    # simplest possible test - convert to DLC and back
    print mol.r
    print "1-3 bond: ", mol.bond(dec(1,3))
    print "1-2 bond: ", mol.bond(dec(1,2))
    dlc = DLC(mol)
    dlc.constraint(dlc.bond(dec(1,3)))
    dlc.init()
    oldS = dlc.active()
    # random perturbation - to ensure constrained internals don't change
    dlc.update( oldS*(0.975 + 0.05*np.random.rand(*np.shape(oldS))) )
    # TODO: option for DLC object to hold ref to molecule object and update mol.r in update()
    mol.r = dlc.xyzs()
    print mol.r
    print "1-3 bond: ", mol.bond(dec(1,3))
    print "1-2 bond: ", mol.bond(dec(1,2))
    dlc.update(oldS)
    mol.r = dlc.xyzs()
    print mol.r
    print "1-3 bond: ", mol.bond(dec(1,3))
    print "1-2 bond: ", mol.bond(dec(1,2))

  mol = molecule(xyz=C2H3F)

  if 0:
    print "1-2 bond: ", mol.bond(dec(1,2))
    print "4-1-3 angle: ", mol.angle(dec(4,1,3))*180/np.pi
    print "5-2-6 angle: ", mol.angle(dec(5,2,6))*180/np.pi

    # HDLCs
    hdlc = HDLC( mol, groups=dec((1,3,4), (2,5,6)) )
    # we can add more automated handling of constraints if the need arises
    hdlc.constrain(dec(4,1,3))
    hdlc.constrain(dec(5,2,6))
    import pdb; pdb.set_trace()
    hdlc.init(mol.r)

    oldS = hdlc.active()
    hdlc.update( oldS*(0.975 + 0.05*np.random.rand(*np.shape(oldS))) )
    mol.r = hdlc.xyzs()
    print "1-2 bond: ", mol.bond(dec(1,2))
    print "4-1-3 angle: ", mol.angle(dec(4,1,3))*180/np.pi
    print "5-2-6 angle: ", mol.angle(dec(5,2,6))*180/np.pi

  if 0:
    # HDLCs
    dlc1 = DLC(mol, dec(1,3,4))
    dlc2 = DLC(mol, dec(2,5,6))
    dlc1.constrain(dec(4,1,3))
    dlc2.constrain(dec(5,2,6))
    hdlc = HDLC( mol, groups=[dlc1, dlc2] )
    hdlc.init(mol.r)

  if 0:
    # reproduce Baker, 1996 (orig. DLC paper)
    bonds = dec( (2,1), (3,1), (4,1), (5,2), (6,2) )
    angles = dec( (2,1,3), (2,1,4), (3,1,4), (5,2,1), (6,2,1), (5,2,6) )
    diheds = dec( (5,2,1,3), (5,2,1,4), (6,2,1,3), (6,2,1,4) )

    dlc = DLC( mol, None, bonds, angles, diheds ) #,
    dlc.init()

    dlc2 = DLC(mol, autobonds='total', autoangles='none', autodiheds='none')
    dlc2.init()
    S2 = dlc2.S

    dlc3 = DLC(mol, autobonds='total', autoangles='conn', autodiheds='none')
    dlc3.init()
    S3 = dlc3.S

    dlc4 = DLC(mol, autobonds='total', autoangles='none', autodiheds='conn')
    dlc4.init()
    S4 = dlc4.S

    dlc5 = DLC(mol)
    dlc5.init()
    S5 = dlc5.S

    np.random.seed(137)
    rnd = np.random.rand(*np.shape(S2))

    # failure of large coord changes may be due to Ga = Ba*Ba^T becoming singular
    #  only option may be to restart
