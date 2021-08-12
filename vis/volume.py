import numpy as np
from itertools import cycle
from ..external.glfw import GLFW_KEY_BACKSPACE
from ..qmmm.grid import *
from .volume_renderer import VolumeRenderer


## Volume Visualization

# Volume functions for use with VisVol
# For now, VisVol will query number of volumes and initial volume index (e.g. HOMO) by passing mo_number=None

def pyscf_mo_vol(mol, mo_number, extents, sample_density):
  if mo_number is None:
    homo = np.max(np.nonzero(mol.pyscf_mf.mo_occ))
    return 2*homo, homo
  if not hasattr(mol._vis, 'mo_vols'):
    mol._vis.mo_vols = pyscf_mo_grid(mol.pyscf_mf, extents, sample_density)
  return mol._vis.mo_vols[mo_number]


def pyscf_dens_vol(mol, mo_number, extents, sample_density, mos=True):
  if mo_number is None:
    return 1, 0
  if not hasattr(mol._vis, 'e_vol'):
    homo = np.max(np.nonzero(mol.pyscf_mf.mo_occ))
    # discard mo_vols to save memory
    discard, mol._vis.e_vol = \
        pyscf_mo_grid(mol.pyscf_mf, extents, sample_density, n_calc=homo+1, density=mos)
  return mol._vis.e_vol


def pyscf_esp_vol(mol, mo_number, extents, sample_density):
  if mo_number is None:
    return 1, 0
  if not hasattr(mol._vis, 'esp_vol'):
    shape = grid_shape(extents, sample_density)
    grid = r_grid(extents, shape=shape)
    mol._vis.esp_vol = np.reshape(pyscf_esp_grid(mol.pyscf_mf, grid), shape)
  return mol._vis.esp_vol


def cclib_mo_vol(mol, mo_number, extents, sample_density):
  if mo_number is None:
    return mol.cclib.nmo, mol.cclib.homos[0]
  if not hasattr(mol._vis, 'mo_vols'):
    mol._vis.mo_vols = [None]*mol.cclib.nmo
  if mol._vis.mo_vols[mo_number] is None:
    mol._vis.mo_vols[mo_number] = get_mo_volume(mol.cclib, mo_number, extents, sample_density)
  return mol._vis.mo_vols[mo_number]


def cclib_lmo_vol(mol, mo_number, extents, sample_density):
  if mo_number is None:
    return len(mol.cclib.lmocoeffs[0]), 0
  if not hasattr(mol._vis, 'lmo_vols'):
    mol._vis.lmo_vols = [None]*len(mol.cclib.lmocoeffs[0])
  if mol._vis.lmo_vols[mo_number] is None:
    mol.lmo_vols[mo_number] = \
        get_mo_volume(mol.cclib, mo_number, extents, sample_density, mocoeffs=mol.cclib.lmocoeffs[0])
  return mol._vis.lmo_vols[mo_number]


def cclib_dens_vol(mol, mo_number, extents, sample_density, mos=None):
  if mo_number is None:
    return 1, 0
  if not hasattr(mol._vis, 'e_vol'):
    mol._vis.e_vol = get_dens_volume(mol.cclib, extents, sample_density, mos=mos)
  return mol._vis.e_vol


class VisVol:
  def __init__(self, vol_fn, vis_type='iso', vol_type=None, extents=None,
      colormap=[(0.,0.,1.,0.75), (1.,0.,0.,0.75)], iso_step=0.01, postprocess=None, timing=False):
    """ see, e.g., cclib_mo_vol for example of `vol_fn`; vis_type is 'iso' or 'volume'; vol_type is title
      used when printing information; extents can be passed to manually specify bounds for volume; iso_step
      is step size used when changing iso level; volume returned by vol_fn can be modified by `postprocess` fn
      For isosurface, colormap[0] is used for negative surface and colormap[-1] for positive surface
      NOTE: colormap takes color in GL format (0 - 1), not 0 - 255 (change this?)
      If `timing` is True, timing information will be printed
    """
    self.vol_fn = vol_fn
    self.vol_methods = cycle(['iso', 'volume'])
    self.vol_rend = VolumeRenderer(method=vis_type)
    self.colormap = np.asarray(colormap)
    self.vol_obj = None
    self.vol_type = "Volume" if vol_type is None else vol_type
    self.iso_step = iso_step  # could we determine this automatically from volume values?
    self.extents = extents
    self.postprocess = postprocess
    self.timing = timing
    self.nmos = 0


  def __repr__(self):
    return "VisVol(type='%s')" % self.vol_type


  def help(self):
    print("--- %s" % self)
    print(
""";': next/prev MO
[]: change absorptivity for volume rendering by 0.8^(+/-1); Shift for 0.8^(+/-10) step
 \: reset absorptivity
-=: change isosurface threshold by 0.025; Shift for 0.25 step
 Bksp: reset isosurface threshold
 V: change rendering method (volume, isosurface)
""")


  # currently volume is cached before postprocess fn; we could consider passing postprocess fn vol_fn so
  #  that final processed volume can be cached instead
  def update_volume(self, vol_obj, mol, mo_number):
    if self.nmos < 1:
      return
    t0 = time.time() if self.timing else 0
    vol = self.vol_fn(mol, mo_number, mol._vis.mo_extents, mol._vis.sample_density)
    vol_obj.set_data((self.postprocess(vol) if self.postprocess is not None else vol), mol._vis.mo_extents)
    print_time = " - returned in %d ms" % int(1000*(time.time() - t0)) if self.timing else ""
    print("Showing %s %d%s" % (self.vol_type, mo_number, print_time))


  def draw(self, viewer, pass_num):
    if pass_num == 'volume':
      self.vol_rend.draw(viewer)


  def set_molecule(self, mol, r=None):
    assert r is None, "VisVol does not support r parameter"
    self.mol = mol
    # mol._vis is used to cache volume data - reset if geometry changes
    if not hasattr(mol, '_vis') or np.any(mol._vis.r != mol.r):
      mol._vis = Bunch(r=mol.r)  # each use of mol.r constructs a new r
    if not hasattr(mol._vis, 'mo_extents'):
      mol._vis.mo_extents = mol.extents(pad=2.0) if self.extents is None else self.extents
      # set sample density for volume generation to ensure no more than 2**24 points
      mol._vis.sample_density = min(20.0, (np.prod(mol._vis.mo_extents[1] - mol._vis.mo_extents[0])/(2**24))**(-1/3.0))
    # query vol_fn for number of volumes and initial index
    self.nmos, self.mo_number = self.vol_fn(mol, None, None, None)
    if self.vol_obj is None:
      self.vol_obj = self.vol_rend.make_volume_obj(colormap=self.colormap, isolevel=0.2, isolevel_n=-0.2)
    self.update_volume(self.vol_obj, self.mol, self.mo_number)


  def on_key_press(self, viewer, keycode, key, mods):
    if key in ";'":
      if key == "'" and self.mo_number+1 < self.nmos:
        self.mo_number += 1
      elif key == ';' and self.mo_number > 0:
        self.mo_number -= 1
      else:
        print("No more volumes to show!")
      self.update_volume(self.vol_obj, self.mol, self.mo_number)
    elif key in '-=':
      if 'Ctrl' in mods:
        self.vol_rend.ray_steps *= (1.25 if key == '=' else 0.8)
        print("Volume ray marching steps: {}".format(self.vol_rend.ray_steps))
      else:
        s = self.iso_step * (10 if 'Shift' in mods else 1) * (-1 if key == '-' else 1)
        if self.vol_obj.isolevel + s > 0:
          self.vol_obj.isolevel += s
          self.vol_obj.isolevel_n -= s
        print("Isosurface threshold: +/-%0.3f" % self.vol_obj.isolevel)
    elif keycode == GLFW_KEY_BACKSPACE:
      self.vol_obj.isolevel = 0.2
      self.vol_obj.isolevel_n = -0.2
      print("Isosurface threshold reset to: +/-%0.3f" % self.vol_obj.isolevel)
    elif key in '[]':
      s = np.power((1.25 if key == ']' else 0.8), (10 if 'Shift' in mods else 1))
      self.vol_obj.absorption *= s
      print("Volume absorptivity: %0.3f" % self.vol_obj.absorption)
    elif key == '\\':
      self.vol_obj.absorption = 10
      print("Volume absorptivity reset to: %0.3f" % self.vol_obj.absorption)
    elif key == 'V':
      self.vol_rend.method = next(self.vol_methods)
      print("Volume render method: %s" % self.vol_rend.method)
    else:
      return False
    return True


# molecular surfaces:
# - for each atom, find grid point and fill in distances within some radius (if less than existing distance)
# if you need EDT, try scipy.ndimage.morphology.distance_transform_edt
# possible refs:
# - http://webglmol.osdn.jp/surface.html
# - https://github.com/boscoh/pdbremix/blob/master/pdbremix/asa.py
# - https://github.com/arose/ngl/blob/master/src/surface/edt-surface.js
# - http://zhanglab.ccmb.med.umich.edu/EDTSurf/EDTSurf.pdf

def molecular_surface(mol, extents, sample_density=100):

  #r_flip = mol.r_array[:, [2, 1, 0]]
  #extents = np.stack((np.amin(r_flip, 0) - 0.2, np.amax(r_flip, 0) + 0.2), -1)

  vol_min, vol_max = extents
  samples = ((vol_max - vol_min)*sample_density).astype(np.int)
  vol = np.full(samples, 1e6)

  print("Total samples: {} => {}".format(np.shape(vol), np.size(vol)))

  def dist_fn(x,y,z):
    r = np.array([x,y,z])
    return np.sqrt(np.dot(r,r))
  # precalculate distance box
  dist_extents = np.array([(-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)])
  dist_box = f_grid(dist_fn, extents=dist_extents, samples=(dist_extents[1] - dist_extents[0])*sample_density)
  dx, dy, dz = np.shape(dist_box)

  for ii, r in enumerate(mol.r):
    vdw = ELEMENTS[mol.atomnos[ii]].vdw_radius
    gx, gy, gz = ((r - vol_min)*sample_density + 0.5).astype(np.int) - [dx/2,dy/2,dz/2]
    curr = vol[gx:gx+dx, gy:gy+dy, gz:gz+dz]
    try:
      ## TODO: need to make dist box slightly smaller or vol extents slightly larger
      vol[gx:gx+dx, gy:gy+dy, gz:gz+dz] = np.fmin(curr, dist_box - vdw)
    except:
      print("gx,gy,gz: {}, {}, {}; dx,dy,dz: {}, {}, {}".format(gx,gy,gz, dx,dy,dz))
    #curr_type = type_vol[gx:gx+dx, gy:gy+dy, gz:gz+dz]
    #type_vol[gx:gx+dx, gy:gy+dy, gz:gz+dz] = np.where((dist_box - vdw) < curr, mol.atomnos[ii], curr_type)

  return vol
