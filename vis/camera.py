import numpy as np
from .glutils import *


# Arcball camera class taken from Chemlab
class Camera:
  """ "Arcball" camera supporting mouse manipulation and providing view matrix and projection matrix
  Attributes:
  * position: position of camera in world (model) space; default (0,0,5)
  * a,b,c: vectors pointing to right, upward, and forward (view direction) from camera, respectively
    - default (1,0,0), (0,1,0), (0,0,-1) (so initial view is in -z direction, looking at origin, w/ y axis up)
  * pivot: point about which camera will orbit (always along c vector from position); default (0,0,0)
  * z_near, z_far: near and far clipping planes; set by passing extents of scene to set_extents()
  * fov: field of view for projection matrix, in degrees (default: 45 deg)
  * aspectratio: aspect ratio (width/height) of window - should be set whenever window size changes
  """

  def __init__(self, perspective=True, lock_orientation=False):
    """ if `perspective` == False, orthogonal projection is used; if `lock_orientation` == True y-axis is
      prevented from tilting left or right with x,y orbit (keeps north up when looking a globe)
    """
    self.perspective = perspective
    self.lock_orientation = lock_orientation
    self.position = np.array([0.0, 0.0, 5.0])  # Position in world coordinates
    self.pivot = np.array([0.0, 0.0, 0.0])
    self.fov = 45.0
    self.aspectratio = 1.0
    self.min_z_near = 0.01
    self.z_near = 0.5
    self.z_far = 500.0
    self.a = np.array([1.0, 0.0, 0.0])  # direction to right of camera
    self.b = np.array([0.0, 1.0, 0.0])  # direction up from camera
    self.c = np.array([0.0, 0.0, -1.0])  # direction from camera to pivot


  def rotate_abc(self, angle_a, angle_b, angle_c):
    """ rotation about a,b,c vectors, in that order; mostly for small angles (as when dragging with mouse) """
    rot = rotation_matrix(-angle_a, self.a)
    bvec = [0, -1.0 if self.b[1] < 0 else 1.0, 0] if self.lock_orientation else self.b
    rot = np.dot(rotation_matrix(-angle_b, bvec), rot)
    rot = np.dot(rotation_matrix(-angle_c, self.c), rot)[:3,:3]
    self.position = self.pivot + np.dot(rot, self.position - self.pivot)

    self.a = np.dot(rot, self.a)
    self.b = np.dot(rot, self.b)
    self.c = np.dot(rot, self.c)


  def mouse_rotate(self, dx, dy, dz=0):
    """ Convenience function for mouse rotation """
    f = 1.5
    self.rotate_abc(f*dy, f*dx, dz)


  def rotate_to(self, r, dist_to_r=None, r_up=None):
    """ rotate camera about pivot to place point `r` on line between position and pivot, optionally setting
      distance to `r` at `dist` and rotating so that point `r_up` is above `r`
    """
    r_pivot = normalize(r - self.pivot)
    pos_pivot = normalize(self.position - self.pivot)
    crossp = np.cross(r_pivot, pos_pivot)
    dotp = np.dot(r_pivot, pos_pivot)
    if norm(crossp) < 0.001:
      # handle case of r_pivot and pos_pivot (anti-)parallel
      crossp = self.b
      dotp = 1.0 if dotp > 0.0 else dotp  # may still need to handle dist_to_r and r_up for parallel case
    rot = rotation_matrix(-np.arccos(dotp), normalize(crossp))[:3, :3]
    # if dist provided, adjust position to set r - position distance to dist
    s = (norm(r - self.pivot) + dist_to_r)/norm(self.position - self.pivot) if dist_to_r is not None else 1.0
    self.position = self.pivot + s*np.dot(rot, self.position - self.pivot)
    self.a = np.dot(rot, self.a)
    self.b = np.dot(rot, self.b)
    self.c = np.dot(rot, self.c)
    # in theory this could be combined w/ above rotation
    if r_up is not None:
      r_up = r_up - r
      up_ab = normalize(r_up - np.dot(r_up, self.c)*self.c)
      self.rotate_abc(0, 0, np.arccos(np.dot(self.b, up_ab)))


  def mouse_translate(self, dx, dy, dz=0):
    # translate by 1/20 of FOV at pivot point
    sz = norm(self.position - self.pivot)
    sy = np.tan(self.fov*np.pi/180.0)*sz
    sx = self.aspectratio * sy
    dr = 0.05*(-self.a*sx*dx + self.b*sy*dy + self.c*sz*dz)
    self.position += dr
    self.pivot += dr


  def mouse_zoom(self, inc):
    """ implement zoom by moving camera along c - note that we scale distance """
    self.position = (self.position - self.pivot)*pow(1.1, -inc) + self.pivot


  def set_extents(self, extents):
    """ should be called before each frame with extents defining bounding cube in world space of everything
    to be rendered.  This will be used to set z range just large enough to accommodate, maximizing depth
    buffer precision
    """
    world_corners = np.array([ [extents[i,0], extents[j,1], extents[k,2], 1.0] for i in [0,1] for j in [0,1] for k in [0,1] ])
    view_corners = np.dot(world_corners, self.view_matrix().T)
    z_near = -np.amax(view_corners, 0)[2]
    z_far = -np.amin(view_corners, 0)[2]
    # maximum allowed range
    self.z_near = max(self.min_z_near, z_near)
    self.z_far = min(self.z_near + 1e3, max(self.z_near + 0.01, z_far))  # must have z_far > z_near


  def proj_matrix(self):
    """ projection (perspective) matrix - convert from homogeneous 3D coordinates to 2D coordinates """
    z = self.z_near if self.perspective else norm(self.position - self.pivot)
    fov = self.fov*np.pi/180.0
    top = np.tan(fov/2)*z
    right = self.aspectratio * top
    if self.perspective:
      return perspective_proj(-right, right, -top, top, self.z_near, self.z_far)
    else:
      return ortho_proj(-right, right, -top, top, self.z_near, self.z_far)


  def view_matrix(self):
    """ mapping from world coords to camera coords (in which camera is looking down -z axis toward origin) """
    rot = self._get_rotation_matrix()
    tra = self._get_translation_matrix()
    return np.dot(rot, tra)


  def _get_translation_matrix(self):
    return translation_matrix(-self.position)


  def _get_rotation_matrix(self):
    # Rotate the system to bring it to coincide with 0, 0, -1
    a, b, c = self.a, self.b, self.c
    a0 = np.array([1.0, 0.0, 0.0])
    b0 = np.array([0.0, 1.0, 0.0])
    c0 = np.array([0.0, 0.0, -1.0])
    mfinal = np.array([a0, b0, c0]).T
    morig = np.array([a, b, c]).T
    mrot = np.dot(mfinal, morig.T)
    ret = np.eye(4)
    ret[:3,:3] = mrot
    return ret


  def unproject(self, x, y, z=-1.0):
    """ Convert normalized (-1 to 1) screen coordinates x,y,z to world coordinates """
    source = np.array([x,y,z,1.0])
    # Invert the combined matrix
    matrix = np.dot(self.proj_matrix(), self.view_matrix())
    IM = np.linalg.inv(matrix)
    res = np.dot(IM, source)
    return res[0:3]/res[3]


  def autozoom(self, points, min_dist=0):
    """Fit the current view to the correct zoom level to display all points in `points`
    The camera viewing direction and rotation pivot match the
    geometric center of the points and the distance from that
    point is calculated in order for all points to be in the field
    of view.
    """
    points = np.asarray(points)
    extraoff = 0.01
    old_geom_center = points.sum(axis=0)/len(points)
    # Translate points
    points = points.copy() + self.position

    # Translate position to geometric_center along directions a and b
    geom_center = points.sum(axis=0)/len(points)
    self.position += self.a * np.dot(geom_center, self.a)
    self.position += self.b * np.dot(geom_center, self.b)

    # Translate pivot to the geometric center
    self.pivot = old_geom_center

    # Get the bounding sphere radius by searching for the most distant point
    bound_radius = np.sqrt(((points-geom_center) * (points-geom_center)).sum(axis=1).max())

    # Calculate the distance in order to have the most distant point in our field of view (top/bottom)
    fov_topbottom = self.fov*np.pi/180.0
    dist_fullview = (bound_radius + self.z_near)/np.tan(fov_topbottom * 0.5)
    # dot product of c with each point
    dist_to_points = np.einsum('ij,j', points, self.c)
    dist = max((1.0 + extraoff)*dist_fullview, -min(dist_to_points) + min_dist)
    # set position based on calculated distance
    self.position = self.pivot - dist*self.c


  def state(self):
    """ Return the current camera state as a dictionary, it can be restored with `Camera.restore`. """
    return dict(a=self.a.tolist(), b=self.b.tolist(), c=self.c.tolist(),
          pivot=self.pivot.tolist(), position=self.position.tolist(), fov=self.fov)


  def restore(self, state):
    """ Restore the camera state, passed as a *state* dictionary. You can obtain a previous state from the
      method `Camera.state`.
    """
    self.a = np.array(state['a']).copy()
    self.b = np.array(state['b']).copy()
    self.c = np.array(state['c']).copy()
    self.pivot = np.array(state['pivot']).copy()
    self.position = np.array(state['position']).copy()
    self.fov = state['fov']
