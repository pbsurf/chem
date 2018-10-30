# (Yet another) nudged elastic band implementation
#  main goal here is to help build idea of interface to DLC and optimizer code

# Refs: tsse neb.py - note that force(obj) updates obj.f with force (i.e., gradient)
#  ase neb1.py + neb.py + bfgs.py + optimize.py

# NEB procedure
# 1. given optimized reactant and product states (R and P) defining ends of reaction path...
# 1. create initial states ("images") along band by interpolation
# 1. run optimizer on collection of geom variables for all images on band (excluding R and P)
#  using NEB force expr (collection of gradients for each image, with NEB modification)

# NEB sits between optimizer and energy/gradient code - variables are simply collection
#  of geom vars (DLCs or cartesians) for all images on "band"; key computation of NEB code
#  is to adjust gradients

# TODO: NEB using DLCs - should be as simple as NEB( ... EandG=dlc_wrap_EandG() ... )!
# TODO: allow different coords for optim steps vs. interpolation to obtain initial states
#  (DL-FIND suggests DLC-TC for interp, cartesian for opt)


# From ASE
class NEB:

  def __init__(self, R, P, EandG, nimages=8, k=0.1, climb=False):
    """ R: reactant state, P: product state - coordinates
    EandG: fn to return energy and gradient for a single image when
      passed coordinates
    nImages: number of points (images) along band, including R and P
    k: spring constant (the elastic part)
    """
    self.imageEandG = EandG;
    self.k = k;
    self.climb = climb;
    self.nimages = nimages;
    # create initial images along band via linear interpolation between R and P
    #  I think this is the same as least linear motion (LLM) path
    # TODO: molecule objects (so we need P.r, R.r) or just coords?
    dr = (P - R) / (nimages - 1.0);
    self.images = [R];
    for ii in range(1, nimages - 1):
      self.images.append(R + ii*dr);
    self.images.append(P);


  def active(self):
    """ return vector of coordinates for all active images - i.e.,
    all but first and last (R and P) images.
    generally only used once to get initial (interpolated) guess
    """
    return np.ravel(self.images[1:-1]);


  def EandG(self, xyzs):
    """ Given new coordinates for active images, return
    gradient (and energy of highest image)
    """
    forces = np.empty(self.nimages - 2);
    energies = np.empty(self.nimages - 2);
    # partition new coordinate vector into images
    xyzs = np.reshape(xyzs, (self.nimages - 2, -1));
    for ii in range(1, self.nimages - 1):
      self.images[ii] = xyzs[ii];
      energies[ii - 1], forces[ii - 1] = self.imageEandG(self.images[ii]);

    imax = 1 + np.argsort(energies)[-1]
    self.emax = energies[imax - 1]

    tangent1 = self.images[1] - self.images[0];
    for i in range(1, self.nimages - 1):
      tangent2 = self.images[i + 1] - self.images[i];
      if i < imax:
        tangent = tangent2
      elif i > imax:
        tangent = tangent1
      else:
        tangent = tangent1 + tangent2

      tt = np.vdot(tangent, tangent)
      f = forces[i - 1]
      ft = np.vdot(f, tangent)
      if i == imax and self.climb:
        f -= 2 * ft / tt * tangent
      else:
        f -= ft / tt * tangent
        f -= (np.vdot(tangent1 - tangent2, tangent) * self.k / tt * tangent)

      tangent1 = tangent2

    # return E and G for NEB
    return self.emax, np.ravel(forces);




