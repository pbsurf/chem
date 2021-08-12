import numpy as np
from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO
from .glutils import *
from .shading import *
from .color import *
from ..data.elements import ELEMENTS
from ..molecule import select_atoms

# Refs:
# - pymol (w/ caps): https://sourceforge.net/p/pymol/code/HEAD/tree/trunk/pymol/data/shaders/cylinder.fs
# - https://github.com/wwwtyro/speck/blob/gh-pages/src/shaders/bonds.glsl
# - https://github.com/arose/ngl/
# - https://github.com/3dmol/3Dmol.js/blob/master/3Dmol/WebGL/shaders.js
# - https://github.com/cmbruns/cinemol

# Interesting: https://github.com/cmbruns/swcimposters - cone imposters

# Transparency: try depth peeling!
# * Shouldn't we turn off face culling for transparency?  Do imposters render properly on back face?
# * I think drawing back-to-front even with depth write and depth test enabled still works because OpenGL
#  does depth test to decide whether to store fragment; once it's stored, a subsequent fragment that is closer
#  (i.e., passes depth test) will be blended instead of just replacing it.  A subsequent fragment that is
#  farther (fails depth test) will be discarded
# * If we need to do better, look into depth peeling - multiple render passes, each with access to depth texture
#   from previous pass to output Nth fragment from front, to be combined back to front in a final pass. Separate
#   cylinder and sphere rendering would be a complication - maybe alternative cylinder and sphere passes? We
#   can also set a minimum depth delta to eliminate artifacts from imposter overlap
#  * Depth peeling example: https://github.com/McNopper/OpenGL/tree/master/Example35
#  * Notes other techniques: https://github.com/vispy/vispy/wiki/Tech.-Transparency ; Google: Order-independent transparency
#  * Possibly interesting approach: http://www.alecjacobson.com/weblog/?p=2750


CYL_VERT_SHADER = """
uniform mat4 model_view_mat;
uniform mat4 model_view_projection_mat;
uniform mat4 projection_mat;

attribute vec3 position;
attribute vec4 color;
attribute vec3 cylinder_axis;
attribute float cylinder_radius;
attribute vec3 vert_local_coordinate; // Those coordinates are between 0,1

varying vec4 vertex_viewspace;
varying float cylinder_radiusv;
varying float cylinder_lengthv;
varying vec3 U, V, H;
varying vec4 cylinder_origin;
varying vec3 local_coords;

mat3 alignVector(vec3 a, vec3 b) {
  vec3 v = cross(a, b);
  mat3 vx = mat3(
      0, v.z, -v.y,
      -v.z, 0, v.x,
      v.y, -v.x, 0
  );
  return mat3(1.0) + vx + vx * vx * ((1.0 - dot(a, b)) / dot(v, v));
}

void main()
{
  float cylinder_length = length(cylinder_axis);

  cylinder_lengthv = cylinder_length;
  cylinder_radiusv = cylinder_radius;

  // We compute the bounding box

  // We receive 8 points, and we should place this 8 points
  // at their bounding box position
  vec4 cylinder_base = vec4(position, 1.0);
  cylinder_origin = model_view_mat * cylinder_base;  //cylinder_origin /= cylinder_origin.w;

  // We receive from the program the origin that is the cylinder start
  // point. To this guy we have to add the local coordinates.


  //* vec3 pos = (vert_local_coordinate*2.0 - 1.0 + vec3(0, 0, 1)) * vec3(cylinder_radius, cylinder_radius, cylinder_length/2.0);
  //* vec3 a = vec3(0, 0, 1); //normalize(vec3(-0.000001, 0.000001, 1.000001));
  //* vec3 b = normalize(cylinder_axis);
  //* mat3 R = alignVector(a, b);
  //* pos = R * pos + cylinder_base.xyz;
  //* gl_Position = model_view_projection_mat * vec4(pos, 1);


  // Local vectors, u, v, h
  vec3 u, h, v;

  h = normalize(cylinder_axis);
  u = cross(h, vec3(1.0, 0.0, 0.0));
  if (length(u) < 0.001){
    u = cross(h, vec3(0.0, 0.0, 1.0));
  }
  u = normalize(u);
  v = normalize(cross(u, h));

  // We do the addition in object space
  vec4 vertex = cylinder_base;
  vertex.xyz += u * (vert_local_coordinate.x*2.0 - 1.0) * cylinder_radius;
  vertex.xyz += v * (vert_local_coordinate.y*2.0 - 1.0) * cylinder_radius;
  vertex.xyz += h * vert_local_coordinate.z * cylinder_length;

  // Vertex in view space
  vertex_viewspace = model_view_mat * vertex; //vertex_viewspace /= vertex_viewspace.w;

  // Base vectors of cylinder in view space
  U = normalize((model_view_mat * vec4(u, 0)).xyz);
  V = normalize((model_view_mat * vec4(v, 0)).xyz);
  H = normalize((model_view_mat * vec4(h, 0)).xyz);

  // To reconstruct the current fragment position, I pass the local coordinates
  local_coords = vert_local_coordinate;

  // Projecting
  gl_Position = model_view_projection_mat * vertex;
  gl_FrontColor = color;
}
"""

CYL_FRAG_SHADER = """
uniform mat4 projection_mat;

varying vec4 vertex_viewspace; // this guy should be the surface point.

varying vec3 U, V, H;
varying float cylinder_radiusv;
varying float cylinder_lengthv;

varying vec4 cylinder_origin;
varying vec3 local_coords;

void main()
{
  // First of all, I need the correct point that we're pointing at
  vec3 surface_point = cylinder_origin.xyz;
  surface_point += U * (local_coords.x*2.0 - 1.0) * cylinder_radiusv;
  surface_point += V * (local_coords.y*2.0 - 1.0) * cylinder_radiusv;
  surface_point += H * (local_coords.z * cylinder_lengthv);

  // We need to raytrace the cylinder (!)
  // we can do the intersection in impostor space

  // Calculate the ray direction in viewspace
  vec3 ray_origin = vec3(0.0, 0.0, 0.0);
  vec3 ray_target = surface_point;
  vec3 ray_direction = normalize(ray_target - ray_origin);

  // basis = local coordinate system of cylinder
  mat3 basis = transpose(mat3(U, V, H));

  vec3 base = cylinder_origin.xyz;
  vec3 end_cyl = cylinder_origin.xyz + H * cylinder_lengthv;

  // Origin of the ray in cylinder space
  vec3 P = - cylinder_origin.xyz;
  P = basis * P;

  // Direction of the ray in cylinder space
  vec3 D = basis * ray_direction;

  // Now the intersection is between z-axis aligned cylinder and a ray
  float c = P.x*P.x + P.y*P.y - cylinder_radiusv*cylinder_radiusv;
  float b = 2.0*(P.x*D.x + P.y*D.y);
  float a = D.x*D.x + D.y*D.y;

  float d = b*b - 4*a*c;

  if (d < 0.0)
    discard;

  float t = (-b - sqrt(d))/(2*a);
  vec3 new_point = ray_origin + t * ray_direction;
  // Discarding points outside cylinder
  float outside_top = dot(new_point - end_cyl, -H);
  if (outside_top < 0.0) {
    discard;
  }
  float outside_bottom = dot(new_point - base, H);
  if (outside_bottom < 0.0) {
    discard;
  }

  vec3 tmp_point = new_point - cylinder_origin.xyz;
  vec3 normal = normalize(tmp_point - H * dot(tmp_point, H));

  // calculate depth buffer value for point; assumes default gl_DepthRange of [0,1]
  vec4 projected_point = projection_mat * vec4(new_point, 1.0);
  gl_FragDepth = 0.5*(1.0 + projected_point.z/projected_point.w);

  vec3 color = shading(new_point, normal, gl_Color.rgb);
  gl_FragData[0] = vec4(color, gl_Color.a);
  gl_FragData[1] = vec4(normal * 0.5 + 0.5, 1.0);
}
"""


class StickRenderer:

  def __init__(self, ordered_draw=False, name="StickRenderer"):
    self.ordered_draw = ordered_draw
    self.name = name
    self.shader = None
    self.vao = None


  def set_data(self, bounds, radii, colors):
    """ passed list of [start, end] points (i.e., [x,y,z]) for each cylinder to draw as 3D numpy array in
     `bounds`, along with corresponding `radii` and `colors`
    """
    if self.shader is None:
      self.modules = [RendererConfig.header, RendererConfig.shading]
      vs = compileShader([m.vs_code() for m in self.modules] + [CYL_VERT_SHADER], GL_VERTEX_SHADER)
      fs = compileShader([m.fs_code() for m in self.modules] + [CYL_FRAG_SHADER], GL_FRAGMENT_SHADER)
      self.shader = compileProgram(vs, fs)

    self.n_cylinders = len(bounds)
    if self.n_cylinders == 0:
      return

    # get vertices and indices for a (0,0,0) - (1,1,1) cube
    local, self.indices = cube_triangles()
    # We pass the starting position 8 times, and each of these has
    # a mapping to the bounding box corner.
    self.bounds = np.array(bounds, dtype=np.float32)
    vertices = np.repeat(self.bounds[:, 0], 8, axis=0).astype(np.float32)
    directions = np.repeat(self.bounds[:, 1] - self.bounds[:, 0], 8, axis=0).astype(np.float32)
    prim_radii = np.repeat(radii, 8, axis=0).astype(np.float32)
    prim_colors = np.repeat(colors, 8, axis=0).astype(np.uint8)
    local = np.tile(local, self.n_cylinders)

    if self.vao is None:
      self.vao = glGenVertexArrays(1)
      glBindVertexArray(self.vao)
      self._verts_vbo = bind_attrib(self.shader, 'position', vertices, 3, GL_FLOAT)
      self._color_vbo = bind_attrib(self.shader, 'color', prim_colors, 4, GL_UNSIGNED_BYTE, GL_TRUE)
      self._local_vbo = bind_attrib(self.shader, 'vert_local_coordinate', local, 3, GL_FLOAT)
      self._directions_vbo = bind_attrib(self.shader, 'cylinder_axis', directions, 3, GL_FLOAT)
      self._radii_vbo = bind_attrib(self.shader, 'cylinder_radius', prim_radii, 1, GL_FLOAT)
      glBindVertexArray(0)

      gl_indices = np.repeat([0], 36 * self.n_cylinders)
      self._elem_vbo = VBO(gl_indices, target=GL_ELEMENT_ARRAY_BUFFER)
    else:
      # just update existing VBO for subsequent calls
      update_vbo(self._verts_vbo, vertices)
      update_vbo(self._directions_vbo, directions)
      update_vbo(self._local_vbo, local)
      update_vbo(self._color_vbo, prim_colors)
      update_vbo(self._radii_vbo, prim_radii)


  def draw(self, viewer):
    if self.n_cylinders == 0:
      return
    # GL state
    #glDepthMask(GL_FALSE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # uniforms
    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)
    view_mat, proj_mat = viewer.view_matrix(), viewer.proj_matrix(),
    set_uniform(self.shader, 'model_view_mat', 'mat4fv', view_mat)
    set_uniform(self.shader, 'projection_mat', 'mat4fv', proj_mat)
    set_uniform(self.shader, 'model_view_projection_mat', 'mat4fv', np.dot(proj_mat, view_mat))

    # VAO
    glBindVertexArray(self.vao)

    if self.ordered_draw:
      # sort cylinders from back to front if transparent
      dir_to_camera = 0.5*(self.bounds[:, 0] + self.bounds[:, 1]) - camera.position
      dist_to_camera_sq = np.sum(dir_to_camera*dir_to_camera, 1)
      # reverse order so indicies are given (and elements drawn) from far to near
      sort_order = np.argsort(dist_to_camera_sq)[::-1]
    else:
      sort_order = range(self.n_cylinders)

    gl_indices = np.repeat(sort_order, 36)*8 + np.tile(self.indices, len(sort_order))
    self._elem_vbo.set_array(gl_indices.astype(np.uint32))
    self._elem_vbo.bind()
    glDrawElements(GL_TRIANGLES, len(gl_indices), GL_UNSIGNED_INT, None)  #self._elem_vbo)
    #self._elem_vbo.unbind()
    glBindVertexArray(0)
    glUseProgram(0)

    #glDepthMask(GL_TRUE)
    glDisable(GL_CULL_FACE)


# Sphere imposters
# Ref: https://paroj.github.io/gltut/Illumination/Tut13%20Correct%20Chicanery.html

SPHERE_VERT_SHADER = """
uniform mat4 camera_mat;
uniform mat4 projection_mat;
uniform float scalefac;

attribute vec3 position;
attribute vec4 color;
attribute vec2 at_mapping;
attribute float at_sphere_radius;

varying vec2 mapping;
varying vec3 sphere_center;
varying float sphere_radius;

void main()
{
  vec4 actual_vertex = vec4(position, 1.0);
  sphere_center = (camera_mat*actual_vertex).xyz;
  mapping = at_mapping;
  sphere_radius = at_sphere_radius;
  actual_vertex = camera_mat*actual_vertex + vec4(mapping*sphere_radius*scalefac, 0.0, 0.0);
  gl_Position = projection_mat * actual_vertex;
  gl_FrontColor = color;
}
"""

SPHERE_FRAG_SHADER = """
uniform mat4 projection_mat;
uniform float scalefac;

// sphere-related
varying vec2 mapping;
varying vec3 sphere_center;
varying float sphere_radius;

void impostor_point_normal(out vec3 point, out vec3 normal)
{
  bool ortho = projection_mat[3][3] == 1.0;  // 1.0 for orthographic prog, 0.0 for perspective

  // Get point P on the square
  vec3 P_map = vec3(sphere_radius*mapping.xy*scalefac, 0) + sphere_center;

  // From P, calculate the ray to be traced
  vec3 ray_origin = ortho ? vec3(P_map.xy, 0.0) : vec3(0.0, 0.0, 0.0);
  vec3 ray_direction = ortho ? vec3(0.0, 0.0, -1.0) : normalize(P_map - ray_origin);

  // Calculation of the intersection: second-order equation solution
  float b = 2.0*dot(ray_direction, ray_origin - sphere_center);
  float c = dot(ray_origin - sphere_center, ray_origin - sphere_center) - sphere_radius*sphere_radius;
  float determinant = b*b - 4.0*c;
  if(determinant >= 0.0) {
    // Calculation of the normal - we'll take the closest intersection value
    float x1 = (- b - sqrt(determinant))/2.0;
    vec3 sph_intersection = ray_origin + x1*ray_direction;
    point = sph_intersection;
    normal = normalize(sph_intersection - sphere_center);
  }
  else
    discard;
}

void main()
{
  vec3 point, normal;
  impostor_point_normal(point, normal);

  // calculate depth buffer value for point; assumes default gl_DepthRange of [0,1]
  vec4 point_clipspace = projection_mat * vec4(point, 1.0);
  gl_FragDepth = 0.5*(1.0 + point_clipspace.z/point_clipspace.w);

  vec3 color = shading(point, normal, gl_Color.rgb);
  gl_FragData[0] = vec4(color, gl_Color.a);
  // Must write normal with 4th component as 1.0 since blend fn affects all color attachments!
  gl_FragData[1] = vec4(normal * 0.5 + 0.5, 1.0);
}
"""


class BallRenderer:

  def __init__(self, name="BallRenderer"):
    self.name = name
    self.shader = None
    self.vao = None


  def set_data(self, positions, radii, colors):
    if self.shader is None:
      self.modules = [RendererConfig.header, RendererConfig.shading]
      vs = compileShader([m.vs_code() for m in self.modules] + [SPHERE_VERT_SHADER], GL_VERTEX_SHADER)
      fs = compileShader([m.fs_code() for m in self.modules] + [SPHERE_FRAG_SHADER], GL_FRAGMENT_SHADER)
      self.shader = compileProgram(vs, fs)

    #self.positions = positions
    self.n_spheres = len(positions)
    vertices = np.repeat(positions, 6, axis=0).astype(np.float32)
    radii = np.repeat(radii, 6, axis=0).astype(np.float32)
    colors = np.repeat(colors, 6, axis=0).astype(np.uint8)
    # two triangles for a square
    mapping = np.tile([
        1.0,1.0, -1.0,1.0, -1.0,-1.0,  # CCW triangle 1
        1.0,1.0, -1.0,-1.0, 1.0,-1.0   # CCW triangle 2
      ], self.n_spheres).astype(np.float32)

    if self.vao is None:
      self.vao = glGenVertexArrays(1)
      glBindVertexArray(self.vao)
      self._verts_vbo = bind_attrib(self.shader, 'position', vertices, 3, GL_FLOAT)
      self._color_vbo = bind_attrib(self.shader, 'color', colors, 4, GL_UNSIGNED_BYTE, GL_TRUE)
      self._mapping_vbo = bind_attrib(self.shader, 'at_mapping', mapping, 2, GL_FLOAT)
      self._radius_vbo = bind_attrib(self.shader, 'at_sphere_radius', radii, 1, GL_FLOAT)
      glBindVertexArray(0)

      ##gl_indices = np.repeat([0], 6 * self.n_spheres)
      ##self._elem_vbo = VBO(gl_indices, target=GL_ELEMENT_ARRAY_BUFFER)

    else:
      # just update existing VBO for subsequent calls
      update_vbo(self._verts_vbo, vertices)
      update_vbo(self._color_vbo, colors)
      update_vbo(self._mapping_vbo, mapping)
      update_vbo(self._radius_vbo, radii)


  def draw(self, viewer):
    # We do not want GL_CULL_FACE as we only have a single square (vs. a whole cube for cylinder imposters)
    #glDepthMask(GL_FALSE)

    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)
    set_uniform(self.shader, 'camera_mat', 'mat4fv', viewer.view_matrix())
    set_uniform(self.shader, 'projection_mat', 'mat4fv', viewer.proj_matrix())
    set_uniform(self.shader, 'scalefac', '1f', 1.5)

    glBindVertexArray(self.vao)


    ##if True: ##self.ordered_draw:
    ##  # sort balls from front to back if transparent
    ##  dir_to_camera = self.positions - camera.position
    ##  dist_to_camera_sq = np.sum(dir_to_camera*dir_to_camera, 1)
    ##  # reverse order so indicies are given (and elements drawn) from far to near
    ##  sort_order = np.argsort(dist_to_camera_sq)  ##[::-1]
    ##else:
    ##  sort_order = range(self.n_spheres)
    ##
    ##gl_indices = np.repeat(sort_order, 6)*6 + np.tile(range(6), len(sort_order))
    ##self._elem_vbo.set_array(gl_indices.astype(np.uint32))
    ##self._elem_vbo.bind()
    ##glDrawElements(GL_TRIANGLES, len(gl_indices), GL_UNSIGNED_INT, None)  #self._elem_vbo)


    glDrawArrays(GL_TRIANGLES, 0, 6*self.n_spheres)

    glBindVertexArray(0)
    glUseProgram(0)
    #glDepthMask(GL_TRUE)



LINE_VERT_SHADER = """
uniform mat4 mvp_matrix;
uniform mat4 inv_p_matrix;
uniform vec2 viewport;

attribute vec3 position;
attribute vec3 direction;
attribute vec2 mapping;
attribute float line_radius;
attribute vec4 color_in;

varying vec4 color;
varying vec2 v_mapping;
varying float v_line_radius;
varying vec4 pos_view;

void main()
{
  color = color_in;
  v_line_radius = line_radius;
  v_mapping = vec2(mapping[0]*length(direction), mapping[1]*line_radius);
  vec4 pos0_clip = mvp_matrix*vec4(position + mapping[0]*direction, 1.0);
  // direction in screen space
  vec2 dir_screen = normalize(viewport*(mvp_matrix*vec4(direction, 0.0)).xy);
  // pos0_clip.w is distance to camera (of point mapped0) - scales width to give fixed screen space width
  vec2 width_vec = 2.0*pos0_clip.w*v_mapping[1]*vec2(-dir_screen.y, dir_screen.x)/viewport;
  vec4 pos_clip = pos0_clip + vec4(width_vec, 0.0, 0.0);
  pos_view = inv_p_matrix*pos_clip;
  gl_Position = pos_clip;
}
"""

LINE_FRAG_SHADER = """
uniform float dash_len;

varying vec4 color;
varying vec2 v_mapping;
varying float v_line_radius;
varying vec4 pos_view;

void main()
{
  // we could set alpha = 0 instead, but let's use discard so we don't write gl_FragDepth!
  float alpha = clamp(v_line_radius - abs(v_mapping[1]), 0.0, 1.0);
  if(alpha < 0.05 || (dash_len > 0 && mod(v_mapping[0]/dash_len, 2) > 1.0))
    discard;

  // this gives some improvement when drawing volume with lines (volumes are drawn after lines)
  //gl_FragDepth = alpha > 0.5 ? gl_FragCoord.z : (0.9999 + 0.0001*gl_FragCoord.z)
  // we take normal to always point toward camera (+z direction)
  vec3 final_color = shading_effects(pos_view.xyz, vec3(0,0,1), color.rgb);
  gl_FragData[0] = vec4(final_color, alpha*color.a);
}
"""

# - add flag to disable line antialiasing for supersampling?
# - note that because of our fake antialiasing, line renderer should run after other geometry renderers,
#  since geometry behind our semi-transparent pixels won't render due to depth test
# - also, minor imperfection - wide line appears to rotate when passing through horizontal plane
# https://github.com/uglymol/uglymol/blob/master/src/lines.js has a shader that approximates joining lines
#  with something like a miter join; not sure if that would be useful at all here
# For wide lines, probably better to just draw as licorice with shading='none'
class LineRenderer:

  def __init__(self, dash_len=0, ordered_draw=False, name="LineRenderer"):
    self.dash_len = dash_len
    self.ordered_draw = ordered_draw
    self.name = name
    self.shader = None
    self.vao = None


  def set_data(self, bounds, radii, colors):
    """ Set/update geometry: passed pairs of points defining lines (`bounds`), line `radii` (in pixels), and
      `colors`
    """
    if self.shader is None:
      self.modules = [RendererConfig.header, RendererConfig.shading]
      vs = compileShader([m.vs_code() for m in self.modules] + [LINE_VERT_SHADER], GL_VERTEX_SHADER)
      fs = compileShader([m.fs_code() for m in self.modules] + [LINE_FRAG_SHADER], GL_FRAGMENT_SHADER)
      self.shader = compileProgram(vs, fs)

    self.n_lines = len(bounds)
    if self.n_lines == 0:
      return

    # two triangles for a square
    mapping = np.tile([
        1.0,1.0, 0.0,1.0, 0.0,-1.0,  # CCW triangle 1
        1.0,1.0, 0.0,-1.0, 1.0,-1.0   # CCW triangle 2
      ], self.n_lines).astype(np.float32)

    # 6 vertices for each line:
    self.bounds = np.array(bounds, dtype=np.float32)
    vertices = np.repeat(self.bounds[:, 0], 6, axis=0).astype(np.float32)
    directions = np.repeat(self.bounds[:, 1] - self.bounds[:, 0], 6, axis=0).astype(np.float32)
    prim_radii = np.repeat(radii, 6, axis=0).astype(np.float32)
    prim_colors = np.repeat(colors, 6, axis=0).astype(np.uint8)

    if self.vao is None:
      self.vao = glGenVertexArrays(1)
      glBindVertexArray(self.vao)
      self._verts_vbo = bind_attrib(self.shader, 'position', vertices, 3, GL_FLOAT)
      self._local_vbo = bind_attrib(self.shader, 'mapping', mapping, 2, GL_FLOAT)
      self._directions_vbo = bind_attrib(self.shader, 'direction', directions, 3, GL_FLOAT)
      self._radii_vbo = bind_attrib(self.shader, 'line_radius', prim_radii, 1, GL_FLOAT)
      self._color_vbo = bind_attrib(self.shader, 'color_in', prim_colors, 4, GL_UNSIGNED_BYTE, GL_TRUE)
      glBindVertexArray(0)
    else:
      # just update existing VBO for subsequent calls
      update_vbo(self._verts_vbo, vertices)
      update_vbo(self._local_vbo, mapping)
      update_vbo(self._directions_vbo, directions)
      update_vbo(self._radii_vbo, prim_radii)
      update_vbo(self._color_vbo, prim_colors)


  def draw(self, viewer):
    if self.n_lines == 0:
      return
    # We do not want GL_CULL_FACE as we only have a single square (vs. a whole cube for cylinder imposters)
    # uniforms
    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)
    set_uniform(self.shader, 'mvp_matrix', 'mat4fv', np.dot(viewer.proj_matrix(), viewer.view_matrix()))
    set_uniform(self.shader, 'inv_p_matrix', 'mat4fv', np.linalg.inv(viewer.proj_matrix()))
    set_uniform(self.shader, 'viewport', '2f', [viewer.width, viewer.height])
    set_uniform(self.shader, 'dash_len', '1f', self.dash_len)

    # draw
    glBindVertexArray(self.vao)
    glDrawArrays(GL_TRIANGLES, 0, 6*self.n_lines)
    glBindVertexArray(0)
    glUseProgram(0)


# MolRender should probably be in separate module from geom renderers (geom_renderers.py?)
# ball-and-stick?  separate radii for balls and sticks?
class VisGeom:

  def __init__(self, style='licorice', sel=None, radius=1.0, coloring=color_by_element, colors=None):
    """ render atoms selected by `sel` with `style` ('licorice', 'lines', 'spacefill'), radius scale factor
      `radius`, and `coloring`/`colors`
    """
    self.radius = radius
    self.style = style
    self.mol_styles = ['licorice', 'spacefill', 'lines']  # too hard to set initial style w/ itertools.cycles
    self.style_idx = self.mol_styles.index(style) if style in self.mol_styles else 0
    self.coloring = (lambda *args: coloring(*args, colors=colors)) if colors is not None else coloring
    self.sel = sel
    self.stick_renderer = StickRenderer() #ordered_draw=(opacity < 1.0))
    self.ball_renderer = BallRenderer()
    self.line_renderer = LineRenderer()


  def __repr__(self):
    return "VisGeom(style='%s', sel='%s')" % (self.style, self.sel)


  def help(self):
    print("--- %s" % self)
    print(" M: toggle molecule rendering method")
    print("ER: scale radius")


  def cylinder_data(self, mol, bonds, r_array, atomnos, dbl_bonds, radius):
    def cyl_bounds(r_array, bonds):
      if len(bonds) == 0:
        return np.empty((0,2,3))  # proper shape needed for concatenation with lone_bounds

      # b0, b1 = map(list, zip(*bonds)) -- slower
      # bonds = np.array(bonds); b0, b1 = bonds[:,0], bonds[:,1] -- even slower due to np.array() creation
      starts = r_array[[b[0] for b in bonds]]
      ends = r_array[[b[1] for b in bonds]]
      middle = (starts + ends)/2

      bounds = np.empty((2*len(bonds), 2, 3))
      bounds[0::2, 0, :] = starts
      bounds[0::2, 1, :] = middle
      bounds[1::2, 0, :] = middle
      bounds[1::2, 1, :] = ends
      return bounds

    bounds = cyl_bounds(r_array, bonds)
    radii = [radius] * len(bounds)
    colors = []
    for i, j in bonds:
      colors.append(self.coloring(mol, i))  #self.atom_color(atomnos[i]))
      colors.append(self.coloring(mol, j))  #self.atom_color(atomnos[j]))

    # double bonds - inspired by ngl, but I mostly did this to provide a reason for avoiding cylinder caps
    ## TODO: need to use other bonds for atom to set orientation (dr)
    if dbl_bonds:
      dbl_scale = 0.375
      dbl_bounds = []
      dbl_colors = []
      for i,j in dbl_bonds:
        r0, r1 = r_array[i], r_array[j]
        dr = np.cross(r1 - r0, [0, 0, 1])
        dr = (1 - dbl_scale) * radius * dr/norm(dr)
        dbl_bounds.append(cyl_bounds(np.array([r0+dr, r1+dr, r0-dr, r1-dr]), np.array([[0,1], [2,3]])))

        c0 = self.coloring(mol, i)
        c1 = self.coloring(mol, j)
        dbl_colors.extend([c0, c1, c0, c1])

      dbl_bounds = np.concatenate(dbl_bounds)
      dbl_radii = [radius*dbl_scale] * len(dbl_bounds)

      bounds = np.concatenate((bounds, dbl_bounds))
      radii.extend(dbl_radii)
      colors.extend(dbl_colors)

    return bounds, radii, colors


  def draw(self, viewer, pass_num):
    if pass_num == 'opaque' and self.style == 'licorice':
      self.stick_renderer.draw(viewer)
    if pass_num == 'opaque' and self.style in ['spacefill', 'licorice']:
      self.ball_renderer.draw(viewer)
    if pass_num == 'transparent' and self.style == 'lines':
      self.line_renderer.draw(viewer)


  # is there any advantage to supporting passing bonds + r + z to MolRenderer instead of just mol?
  def set_molecule(self, mol, r=None, active=None):
    """ Set current molecule to `mol`. If `active` is nonempty, only active atoms and bonds containing only
      active atoms will be rendered
    """
    self.mol = mol
    self.r = mol.r if r is None else r
    atomnos, bonds, dbl_bonds = mol.znuc, mol.bonds, [] #mol.dbl_bonds()
    # we're no longer caching result of decode_atom_sel(sel) so sel can be updated
    self.active = select_atoms(mol, self.sel) if self.sel and active is None else active
    # line radius is in pixels, all others in Ang
    radius = self.radius * (1.75 if self.style == 'lines' else 0.2)
    if self.style in ['licorice', 'lines']:
      if self.active is not None:
        activeset = frozenset(self.active)  # critical for performance
        bonds = [b for b in bonds if b[0] in activeset and b[1] in activeset]
        dbl_bonds = [b for b in dbl_bonds if b[0] in activeset and b[1] in activeset]
      cyl_bounds, cyl_radii, cyl_colors = self.cylinder_data(mol, bonds, self.r, atomnos, dbl_bonds, radius)
      if self.style == 'licorice':
        self.stick_renderer.set_data(cyl_bounds, cyl_radii, cyl_colors)
      elif self.style == 'lines':
        # draw a 3D cross (axis-aligned) for any lone atoms in lines representation; make this optional?
        bonded_atoms = frozenset([a for bond in bonds for a in bond])
        lone_bounds = []
        for ii, r in enumerate(self.r):
          if ii not in bonded_atoms and (self.active is None or ii in activeset):
            cr = ELEMENTS[atomnos[ii]].cov_radius if atomnos[ii] > 0 else 0.5
            lone_bounds.append(np.array([[r + [cr, 0, 0], r - [cr, 0, 0]]]))
            lone_bounds.append(np.array([[r + [0, cr, 0], r - [0, cr, 0]]]))
            lone_bounds.append(np.array([[r + [0, 0, cr], r - [0, 0, cr]]]))
            cyl_radii.extend([radius]*3)
            cyl_colors.extend([self.coloring(mol, ii)]*3)
        cyl_bounds = np.concatenate((cyl_bounds, np.concatenate(lone_bounds))) if lone_bounds else cyl_bounds
        self.line_renderer.set_data(cyl_bounds, cyl_radii, cyl_colors)
    if self.style in ['spacefill', 'licorice']:
      active = range(len(self.r)) if self.active is None else self.active
      ball_colors = [self.coloring(mol, ii) for ii in active]
      if self.style == 'spacefill':
        ball_radii = [self.radius*ELEMENTS[mol.atoms[ii].znuc].vdw_radius for ii in active]
      else:
        ball_radii = [radius]*len(active)
      self.ball_renderer.set_data(self.r[active], ball_radii, ball_colors)
    return self


  def on_key_press(self, viewer, keycode, key, mods):
    if key == 'M':
      # toggle molecule rendering method
      self.style_idx = (self.style_idx + 1) % len(self.mol_styles)
      self.style = self.mol_styles[self.style_idx]
      self.set_molecule(self.mol, self.r, self.active)  # force refresh
      print("Molecule style: %s" % self.style)
    elif key in 'ER':
      self.radius *= np.power((1.25 if key == 'R' else 0.8), (10 if 'Shift' in mods else 1))
      # TODO: radius scale should be a shader uniform so we don't have to refresh!
      self.set_molecule(self.mol, self.r, self.active)
      print("Radius (scale): %0.3f" % self.radius)
    else:
      return False
    return True
