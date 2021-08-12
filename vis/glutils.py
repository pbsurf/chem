import math
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO


class StringShaderModule:
  def __init__(self, vs, fs):
    self.vs = vs
    self.fs = fs

  def vs_code(self): return self.vs
  def fs_code(self): return self.fs
  def setup_shader(self, shader, viewer): pass


# Class (not instance) variables serve as globals to provide modules for shaders; convention (other than header)
#  is that name of variable, e.g. 'shading', should correspond to top level fn provided by module
# Note that previous approach of passing a list of modules to Renderer constructor isn't really appropriate,
#  as shader code can only make use of fns that it calls explicitly (and will fail if any are missing)
class RendererConfig:
  # I think this is OK for now - we may need things in here that depend on platform instead of renderers, e.g.
  #  "#define gl_FragDepth gl_FragDepthEXT" for GLES
  header = StringShaderModule("#version 120\n", "#version 120\n")
  header130 = StringShaderModule("#version 130\n", "#version 130\n")
  shading = None  # must be specified


## utility fns

def gl_color(c):
  return (c[0]/255.0, c[1]/255.0, c[2]/255.0, c[3]/255.0)


def create_texture(width, height, int_fmt, fmt, data_type, data=None,
    mag_filter=GL_NEAREST, min_filter=GL_LINEAR, wrap=GL_CLAMP_TO_EDGE, border_value=(0.,0.,0.,0.)):
  tex = glGenTextures(1)
  glBindTexture(GL_TEXTURE_2D, tex)
  glTexImage2D(GL_TEXTURE_2D, 0, int_fmt, width, height, 0, fmt, data_type, data)
  glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
  glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)
  glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
  glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap);
  if wrap == GL_CLAMP_TO_BORDER:
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_value);
  return tex


def set_uniform(prog, uni, typ, value):
  location = glGetUniformLocation(prog, uni.encode('utf-8'))
  if location == -1:
    pass  #print('glGetUniformLocation failed for ' + uni)  # not fatal - may fail for uniforms optimized out!
  elif typ == '1f':
    glUniform1f(location, value)
  elif typ == '2f':
    glUniform2f(location, *value)
  elif typ == '3f':
    glUniform3f(location, *value)
  elif typ == '4f':
    glUniform4f(location, *value)
  elif typ == 'mat4fv':
    value = value.copy() # That was an AWFUL BUG
    glUniformMatrix4fv(location, 1, GL_TRUE, value.astype(np.float32))
  elif typ == '1i':
    glUniform1i(location, value)
  elif typ == '1fv':
    glUniform1fv(location, len(value), value)
  elif typ == '2fv':
    glUniform2fv(location, len(value), value)
  else:
    raise Exception('Unknown type function')


def set_sampler(prog, uni, id, tex, tex_type=GL_TEXTURE_2D):
  glActiveTexture(GL_TEXTURE0+id)
  glBindTexture(tex_type, tex)
  set_uniform(prog, uni, '1i', id)


def unbind_texture(id, tex_type=GL_TEXTURE_2D):
  glActiveTexture(GL_TEXTURE0+id)
  glBindTexture(tex_type, 0)


# PyOpenGL VBO object only copies data if it is bound explicitly, so doesn't play nice with VAO
def update_vbo(vbo, data):
  vbo.set_array(data)
  vbo.bind()


# bind_attrib assumes use of Vertex Array Objects (VAOs) and hence does not return attribute location as there
#  is no need for individual attributes to be enabled/disabled because they are encapsulated by VAO

# if normalized is true, OpenGL will scale integer values to [-1,1] or [0,1] range
# - should be set if passing color as 4 GL_UNSIGNED_BYTES
def bind_attrib(shader, name, vbo_or_array, size, type, normalized=GL_FALSE, stride=0, offset=0):
  attr = glGetAttribLocation(shader, name)
  glEnableVertexAttribArray(attr)
  vbo = vbo_or_array if isinstance(vbo_or_array, VBO) else VBO(vbo_or_array)
  vbo.bind()
  glVertexAttribPointer(attr, size, type, normalized, stride, vbo+offset)
  #vbo.unbind()
  return vbo


# Note: only use glDisableVertexAttribArray() if not using a VAO (unless you want to unbind the attrib from the VAO itself)
def unbind_attrib(attr):
  glBindBuffer(GL_ARRAY_BUFFER, 0)
  glDisableVertexAttribArray(attr)


def unbind_all(attrs):
  glBindBuffer(GL_ARRAY_BUFFER, 0)
  for attr in attrs:
    glDisableVertexAttribArray(attr)


def compileShader(source, shaderType):
  """Compile shader source of given type

  source -- GLSL source-code for the shader
  shaderType -- GLenum GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc,

  returns GLuint compiled shader reference
  raises RuntimeError when a compilation failure occurs
  """
  if isinstance(source, str):
    source = [source]
  elif isinstance(source, bytes):
    source = [source.decode('utf-8')]

  shader = glCreateShader(shaderType)
  glShaderSource(shader, source)
  glCompileShader(shader)
  result = glGetShaderiv(shader, GL_COMPILE_STATUS)

  if not(result):
    # on compile error, print shader source with line numbers to aid debugging
    srclines = ''.join(source).splitlines()
    for ii,line in enumerate(srclines):
      print("{}: {}".format(ii+1, line))
    print("\nShader compile failure (code {}); line(column): {}".format(result, glGetShaderInfoLog(shader)))
    raise RuntimeError("Shader compile failure", shaderType)
  return shader


# Can't use PyOpenGL.compileProgram because it can fail on glValidateProgram, which actually should only be
#  called after the full OpenGL state needed by the program is set up, not at compile time
def compileProgram(vertex_shader, fragment_shader):
  program = glCreateProgram()
  glAttachShader(program, vertex_shader)
  glAttachShader(program, fragment_shader)
  glLinkProgram(program)
  # check linking error
  result = glGetProgramiv(program, GL_LINK_STATUS)
  if not(result):
    raise RuntimeError(glGetProgramInfoLog(program))
  return program

## Transformations
# ref: https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
#  but projections changed to match http://www.songho.ca/opengl/gl_projectionmatrix.html

def translation_matrix(direction):
  """ Return matrix to translate by vector `direction` vector """
  M = np.identity(4)
  M[:3, 3] = direction[:3]
  return M

def rotation_matrix(angle, direction, point=None):
  """ Return matrix to rotate about axis defined by point and direction """
  sina = math.sin(angle)
  cosa = math.cos(angle)
  direction = normalize(direction[:3])
  # rotation matrix around unit vector
  R = np.diag([cosa, cosa, cosa])
  R += np.outer(direction, direction) * (1.0 - cosa)
  direction *= sina
  R += np.array([[ 0.0,         -direction[2],  direction[1]],
                 [ direction[2], 0.0,          -direction[0]],
                 [-direction[1], direction[0],  0.0]])
  M = np.identity(4)
  M[:3, :3] = R
  if point is not None:
    # rotation not around origin
    point = np.array(point[:3], dtype=np.float64, copy=False)
    M[:3, 3] = point - np.dot(R, point)
  return M

def ortho_proj(left, right, bottom, top, near, far):
  if left >= right or bottom >= top or near >= far:
    raise ValueError("invalid frustum")
  M = [[2.0/(right-left), 0.0, 0.0, (right+left)/(left-right)],
       [0.0, 2.0/(top-bottom), 0.0, (top+bottom)/(bottom-top)],
       [0.0, 0.0, -2.0/(far-near), (far+near)/(near-far)],
       [0.0, 0.0, 0.0, 1.0]]
  return np.asarray(M)

def perspective_proj(left, right, bottom, top, near, far):
  if left >= right or bottom >= top or near >= far or near <= 0:
    raise ValueError("invalid frustum")
  t = -2.0 * near
  M = [[t/(left-right), 0.0, (right+left)/(right-left), 0.0],
       [0.0, t/(bottom-top), (top+bottom)/(top-bottom), 0.0],
       [0.0, 0.0, (far+near)/(near-far), t*far/(far-near)],
       [0.0, 0.0, -1.0, 0.0]]
  return np.asarray(M)


## Geometry

# Use indexed drawing to reuse cube vertices; handy illustration adapted from vispy:
#           6-------7
#          /|      /|
#         4-------5 |
# z  y    | |     | |
# | /     | 2-----|-3
# |/      |/      |/
# +---x   0-------1
# Previously we were using array of 32 explicit vertices from speck
def cube_triangles(flat=True):
  """ vertices and indices for glDrawElements for cube for use with imposters, volume rendering, etc. """
  verts = np.array(
      [[0, 0, 0],  # corner 0
       [1, 0, 0],
       [0, 1, 0],
       [1, 1, 0],
       [0, 0, 1],
       [1, 0, 1],
       [0, 1, 1],
       [1, 1, 1]], dtype=np.float32)  # corner 7
  indices = np.array(
      [[0, 2, 1], [2, 3, 1],  # triangle 0, triangle 1
       [1, 4, 0], [1, 5, 4],
       [3, 5, 1], [3, 7, 5],
       [2, 7, 3], [2, 6, 7],
       [0, 6, 2], [0, 4, 6],
       [5, 6, 4], [5, 7, 6]], dtype=np.uint32)
  # another option - triangle strip with primitive restart (append 2^32-1)
  #self.indices = np.array([2, 6, 0, 4, 5, 6, 7, 2, 3, 0, 1, 5, 3, 7], dtype=np.uint32)
  return (verts.flatten(), indices.flatten()) if flat else (verts, indices)


def cube_separate(flat=True):
  """ vertices, normals, indices for cube, with separate vertices for each face for actually drawing cubes """
  n0 = np.array(
      [[ 0, 0,-1],  # 0123 face (see ASCII art above)
       [ 0,-1, 0],  # 0145 face
       [ 1, 0, 0],  # 1357 face
       [ 0, 1, 0],  # 2367 face
       [-1, 0, 0],  # 0246 face
       [ 0, 0, 1]], dtype=np.float32)  # 4567 face
  v0, i0 = cube_triangles(flat=False)
  vertices = v0[i0.flatten()]
  normals = n0.repeat(6, axis=0)
  indices = np.arange(len(vertices), dtype=np.uint32)
  return (vertices.flatten(), normals.flatten(), indices) if flat else (vertices, normals, indices)


## test fns

# 2x2 black and white checkerboard texture for debugging
def checkerboard_texture():
  test_data = np.array([0.0, 0.0, 0.0,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  0.0, 0.0, 0.0], dtype=np.float32)
  return create_texture(2, 2, GL_RGB, GL_RGB, GL_FLOAT, test_data)

## useful fns

def norm(v):
  return math.sqrt(np.dot(v,v))  # faster than np.linalg.norm

def normalize(v):
  v = np.asarray(v)
  n = norm(v)
  if n == 0.0:
    raise Exception("Cannot normalize() null vector - use normalize_or_null()!")
  return v/n

def normalize_or_null(v):
  v = np.asarray(v)
  n = norm(v)
  return v/n if n > 0.0 else v


def cardinal_spline(t, p0, p1, p2, p3, tension=1.0):
  """ Returns a point at fraction t between p1 and p2 on the cardinal spline defined by p0,p1,p2,p3. Default
    `tension` of 1.0 gives a Catmull-Rom spline.  See https://en.wikipedia.org/wiki/Cubic_Hermite_spline
  """
  v0 = 0.5*tension*(p2 - p0)
  v1 = 0.5*tension*(p3 - p1)
  t2 = t*t
  t3 = t*t2
  #return (2*p1 - 2*p2 + v0 + v1)*t3 + (-3*p1 + 3*p2 - 2*v0 - v1)*t2 + v0*t + p1
  return (2*t3 - 3*t2 + 1)*p1 + (-2*t3 + 3*t2)*p2 + (t3 - 2*t2 + t)*v0 + (t3 - t2)*v1


# From chemlab pickers.py
def ray_spheres_intersection(origin, direction, centers, radii):
  """Calculate the intersection points between a ray and multiple spheres.
  Returns intersections, distances in order of increasing dist from origin
  """
  b_v = 2.0 * ((origin - centers) * direction).sum(axis=1)
  c_v = ((origin - centers)**2).sum(axis=1) - radii ** 2
  det_v = b_v * b_v - 4.0 * c_v

  inters_mask = det_v >= 0
  intersections = (inters_mask).nonzero()[0]
  distances = (-b_v[inters_mask] - np.sqrt(det_v[inters_mask])) / 2.0

  # We need only the thing in front of us, that corresponts to
  # positive distances.
  dist_mask = distances > 0.0

  # We correct this aspect to get an absolute distance
  distances = distances[dist_mask]
  intersections = intersections[dist_mask].tolist()

  if intersections:
    distances, intersections = zip(*sorted(zip(distances, intersections)))
    return list(intersections), list(distances)
  else:
    return [], []
