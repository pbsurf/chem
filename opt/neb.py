# (Yet another) nudged elastic band implementation
#  main goal here is to help build idea of interface to DLC and optimizer code

# Refs: tsse neb.py - note that force(obj) updates obj.f with force (i.e., gradient)
#  ase neb1.py + neb.py + bfgs.py + optimize.py
# - 10.1021/ct060032y - succinct overview of NEB (w/ applications to gas-phase reactions)
# - 10.1063/1.1329672 - climbing-image NEB (orig paper)
# - 10.1063/1.2841941 - shows optimizing all images together (allowing cross-image Hessian elements) is better
#  - also argues that problems with quasi-Newton methods from pathological Hessian due to NEB modification
#  of forces as discussed in 10.1063/1.1627754 are not an issue in practice

# NEB procedure
# 1. given optimized reactant and product states (R and P) defining ends of reaction path...
# 1. create initial states ("images") along band by interpolation
# 1. run optimizer on collection of geom variables for all images on band (excluding R and P)
#  using NEB force expr (collection of gradients for each image, with NEB modification)

# NEB sits between optimizer and energy/gradient code - variables are simply collection
#  of geom vars (DLCs or cartesians) for all images on "band"; key computation of NEB code
#  is to adjust gradients

# Aligning each image to previous position (eliminating spurious translation and rotation) may be helpful
#  since spring force depends on |dr|, but would need to be done in optimizer - we can't do it in EandG
#newimgs = [align_atoms(newimg, self.images[ii+1]) for ii,newimg in enumerate(newimgs)]

# Double nudged elastic band (Ref: pele, TSSE) - might help w/ initial optim, but must be turned off to get true path
#fk = self.k*(tangent2 - tangent1); fkperp = fk - np.dot(fk, tandir)*tandir
#gperp = normalize(grads[ii] - gt*tandir); fdneb = fkperp - np.dot(fkperp,gperp)*gperp; fneb += fdneb

# String methods: spring force replaced by manual redistribution of images along path (component of grad along
#  path is still removed however); growing string: images initially placed only near ends of path; new images
#  added along tangents at ends as optimization proceeds

# Does NEB using DLCs make sense? - should be as simple as NEB( ... EandG=dlc_wrap_EandG() ... )
# Consider option to init with full set of images (to support external interpolation or results from RC scan)
# If climb causes trouble, consider option to only turn it on when gradient exceeds threshold (or just do
#  two optimization runs, first with climb=False?)


import numpy as np
from numpy.linalg import norm

# If this has difficulty with complex system or reaction, try dlc_interp() (in theo.py)
# - DL-FIND suggests DLC-TC for interp, cartesian for optimization
# This is only needed for numpy < v1.16 (after which np.linspace can be used)
def linear_interp(R, P, nimages):
  """ Linear interpolation between geometries R and P """
  dr = (P - R)/(nimages - 1)
  return R[None,...] + dr[None,...]*np.arange(nimages).reshape((-1,) + (1,)*np.ndim(dr))


class NEB:

  def __init__(self, R, P, EandG, nimages=8, k=1.0, climb=False, verbose=True, hist=None):
    """ R: reactant state, P: product state - coordinates
    EandG: fn to return energy and gradient for a single image when passed coordinates
    nImages: number of points (images) along band, including R and P
    k: spring constant (the elastic part)
    climb: use climbing-image NEB - highest energy image is pushed uphill w/o spring force
    """
    self.imgEandG = EandG
    self.k = k  # if k else (EandG(P) - EandG(R))/norm(P - R)
    self.climb = climb
    self.verbose = verbose
    self.hist = hist
    # create initial images along band via linear interpolation between R and P
    #  I think this is the same as least linear motion (LLM) path
    self.images = linear_interp(R, P, nimages)
    # check for bad image (maybe due to R and P not being well-aligned)
    from scipy.spatial.distance import pdist
    for img in self.images:
      mindist = np.min(pdist(img))
      if mindist < 0.75:  # H2 bond length (shortest) is 0.74 Ang
        print("Warning: small distance in interpolated reaction path: %f Ang" % mindist)


  def active(self):
    """ return vector of coordinates for all active images - i.e., all but first and last (R and P) images.
    generally only used once to get initial (interpolated) guess
    """
    return np.ravel(self.images[1:-1])


  def EandG(self, xyzs):
    """ Given new coordinates for active images, return energy of highest image and gradient """
    nimages = len(self.images)
    energies = [0]*nimages
    grads = [0]*nimages
    # optionally save previous images
    if self.hist is not None:
      self.hist.append(np.array(self.images))
    # partition new coordinate vector into images
    self.images[1:-1] = np.reshape(xyzs, np.shape(self.images[1:-1]))  #(self.nimages - 2, -1, 3))
    for ii in range(1, nimages - 1):
      energies[ii], grads[ii] = self.imgEandG(self.images[ii])

    imax = np.argmax(energies[1:-1]) + 1
    self.emax = energies[imax]

    tangent1 = self.images[1] - self.images[0]
    for ii in range(1, nimages-1):
      tangent2 = self.images[ii+1] - self.images[ii]
      # choose "uphill" tangent ... should we check energies[ii+/-1] to handle path w/ >1 maxima?
      tangent = (tangent1 if ii >= imax else 0) + (tangent2 if ii <= imax else 0)
      tandir = tangent/norm(tangent)
      gt = np.vdot(grads[ii], tandir)
      #fneb = np.vdot(tangent2 - tangent1, tandir)*self.k  # ASE does np.vdot(tangent2 - tangent1, tangent)
      fneb = (norm(tangent2) - norm(tangent1))*self.k  # I think this is better
      if self.verbose:
        print("i: %d, E: %f, |g_perp|: %f, |f_NEB|: %f, |dr1|: %f, |dr2|: %f" %
            (ii, energies[ii], norm(grads[ii] - gt*tandir), fneb, norm(tangent1), norm(tangent2)))
      if ii == imax and self.climb:
        grads[ii] -= 2*gt*tandir
      else:
        grads[ii] -= (gt + fneb)*tandir

      tangent1 = tangent2

    print("*** Emax: %f; max |g|: %f\n" % (self.emax, np.max(np.abs(grads[1:-1]))))
    return self.emax, np.ravel(grads[1:-1])
