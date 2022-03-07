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
  sRGB = True


## utility fns

# from stb_image_resize.h
sRGBtoLinear = np.asarray([
  0.000000, 0.000304, 0.000607, 0.000911, 0.001214, 0.001518, 0.001821, 0.002125, 0.002428, 0.002732, 0.003035,
  0.003347, 0.003677, 0.004025, 0.004391, 0.004777, 0.005182, 0.005605, 0.006049, 0.006512, 0.006995, 0.007499,
  0.008023, 0.008568, 0.009134, 0.009721, 0.010330, 0.010960, 0.011612, 0.012286, 0.012983, 0.013702, 0.014444,
  0.015209, 0.015996, 0.016807, 0.017642, 0.018500, 0.019382, 0.020289, 0.021219, 0.022174, 0.023153, 0.024158,
  0.025187, 0.026241, 0.027321, 0.028426, 0.029557, 0.030713, 0.031896, 0.033105, 0.034340, 0.035601, 0.036889,
  0.038204, 0.039546, 0.040915, 0.042311, 0.043735, 0.045186, 0.046665, 0.048172, 0.049707, 0.051269, 0.052861,
  0.054480, 0.056128, 0.057805, 0.059511, 0.061246, 0.063010, 0.064803, 0.066626, 0.068478, 0.070360, 0.072272,
  0.074214, 0.076185, 0.078187, 0.080220, 0.082283, 0.084376, 0.086500, 0.088656, 0.090842, 0.093059, 0.095307,
  0.097587, 0.099899, 0.102242, 0.104616, 0.107023, 0.109462, 0.111932, 0.114435, 0.116971, 0.119538, 0.122139,
  0.124772, 0.127438, 0.130136, 0.132868, 0.135633, 0.138432, 0.141263, 0.144128, 0.147027, 0.149960, 0.152926,
  0.155926, 0.158961, 0.162029, 0.165132, 0.168269, 0.171441, 0.174647, 0.177888, 0.181164, 0.184475, 0.187821,
  0.191202, 0.194618, 0.198069, 0.201556, 0.205079, 0.208637, 0.212231, 0.215861, 0.219526, 0.223228, 0.226966,
  0.230740, 0.234551, 0.238398, 0.242281, 0.246201, 0.250158, 0.254152, 0.258183, 0.262251, 0.266356, 0.270498,
  0.274677, 0.278894, 0.283149, 0.287441, 0.291771, 0.296138, 0.300544, 0.304987, 0.309469, 0.313989, 0.318547,
  0.323143, 0.327778, 0.332452, 0.337164, 0.341914, 0.346704, 0.351533, 0.356400, 0.361307, 0.366253, 0.371238,
  0.376262, 0.381326, 0.386430, 0.391573, 0.396755, 0.401978, 0.407240, 0.412543, 0.417885, 0.423268, 0.428691,
  0.434154, 0.439657, 0.445201, 0.450786, 0.456411, 0.462077, 0.467784, 0.473532, 0.479320, 0.485150, 0.491021,
  0.496933, 0.502887, 0.508881, 0.514918, 0.520996, 0.527115, 0.533276, 0.539480, 0.545725, 0.552011, 0.558340,
  0.564712, 0.571125, 0.577581, 0.584078, 0.590619, 0.597202, 0.603827, 0.610496, 0.617207, 0.623960, 0.630757,
  0.637597, 0.644480, 0.651406, 0.658375, 0.665387, 0.672443, 0.679543, 0.686685, 0.693872, 0.701102, 0.708376,
  0.715694, 0.723055, 0.730461, 0.737911, 0.745404, 0.752942, 0.760525, 0.768151, 0.775822, 0.783538, 0.791298,
  0.799103, 0.806952, 0.814847, 0.822786, 0.830770, 0.838799, 0.846873, 0.854993, 0.863157, 0.871367, 0.879622,
  0.887923, 0.896269, 0.904661, 0.913099, 0.921582, 0.930111, 0.938686, 0.947307, 0.955974, 0.964686, 0.973445,
  0.982251, 0.991102, 1.0
], dtype=np.float32)

def gl_color(c):
  return tuple(sRGBtoLinear[list(c)]) if RendererConfig.sRGB else (c[0]/255.0, c[1]/255.0, c[2]/255.0, c[3]/255.0)

# note color attribs are 4x f32 for sRGB, but 4x u8 otherwise (whereas color uniforms are always f32)
def gl_colors(c):
  c = np.asarray(c, dtype=np.uint8)  # convert to int and clamp
  return sRGBtoLinear[np.ravel(c)] if RendererConfig.sRGB else c


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
# - should be GL_TRUE if passing color as 4 GL_UNSIGNED_BYTES
def bind_attrib(shader, name, vbo_or_array, size, type=None, normalized=None, stride=0, offset=0, divisor=0):
  attr = glGetAttribLocation(shader, name)
  glEnableVertexAttribArray(attr)
  vbo = vbo_or_array if isinstance(vbo_or_array, VBO) else VBO(vbo_or_array)
  vbo.bind()
  type = dict(float32=GL_FLOAT, uint8=GL_UNSIGNED_BYTE)[vbo_or_array.dtype.name] if type is None else type
  normalized = (GL_TRUE if type == GL_UNSIGNED_BYTE else GL_FALSE) if normalized is None else normalized
  glVertexAttribPointer(attr, size, type, normalized, stride, vbo+offset)
  if divisor:
    glVertexAttribDivisor(attr, divisor)   # for instanced drawing
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
