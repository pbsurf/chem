## Classes for post processing/final output pass
import numpy as np
from OpenGL.GL import *
from .glutils import *
from .color import *

## TODO

## AO
# - could we use molecular surface (distance field?) volume texture for AO?
# - try original SSAO, since biggest problem with SAO is that noise is in screen space!  Also, I think
#  random points in camera/world space might be a better fit for spacefilling geom - which is very different
#  from typical 3D graphics geometry - than screen space.  Implement depth-aware blur pass if necessary
# - try setting extents only when molecule changes ... maybe less SAO noise when rotating
# See:
# - https://github.com/glumpy/glumpy/blob/master/examples/protein-ssao.py
# - https://github.com/Twinklebear/ssao/blob/master/res/shaders/blur_frag.glsl
# - https://github.com/PeterTh/gedosato/blob/master/pack/assets/dx9/martinsh_ssao.fx
# - https://github.com/frederikaalund/sfj/blob/master/black_label/libraries/rendering/shaders/starcraft2_ambient_occlusion.fragment.glsl

## - supersampling and maybe postprocess antialiasing (FXAA)
# - I don't think we want to render to screen for every pass, since this may limit speed
# ... but it would be nice to be able to abort supersampling
# - since in general we swap color textures for postprocessing, we need a separate color texture to accumulate samples
# - also, we can't use screen as persistent buffer due to double buffering
# In viewer: need to create accumulator texture in on_resize(); need to bind it instead of 0 in
#  bind_framebuffer() and set blend fn to additive; in render(), need to call draw repeatedly, jittering
#  camera ... how to do gl_FragColor = gl_FragColor * scale; in last pass?  PostprocessHost checks flag in viewer?
# Refs:
# - https://github.com/arose/ngl/blob/master/src/viewer/viewer.js#L1045
# - https://github.com/mrdoob/three.js/blob/dev/examples/js/postprocessing/SSAARenderPass.js
# - https://github.com/mrdoob/three.js/blob/dev/examples/js/shaders/CopyShader.js

## - Antialiasing:
#  - MSAA doesn't work with imposters (only edges of real polygons); could try supersampling (render to 4x larger image, downsample, possibly repeat for 8x, etc); pv does 4x supersampling with OK results
#  - for smooth surfaces, normal is parallel to plane of screen at edges - if we find the normal is close to meeting this condition in FS, we could run a loop to calculate multiple samples
# - google for "FXAA 3.11"
#  - see https://github.com/vispy/vispy/wiki/Tech.-Antialiasing and https://github.com/vispy/experimental/tree/master/fsaa
# - see if we can antialias our imposters easily - http://stackoverflow.com/questions/23385033/antialiased-glsl-impostors
# - note that aliasing is much less noticible with black background


POSTPROCESS_VERT_SHADER = """
attribute vec2 position;

varying vec2 texcoord;

void main()
{
  texcoord = 0.5*(position + 1.0);
  gl_Position = vec4(position, 0.0, 1.0);
}
"""

POSTPROCESS_FRAG_SHADER = """
varying vec2 texcoord;

uniform sampler2D color_tex;
uniform sampler2D depth_tex;
uniform sampler2D normal_tex;

uniform mat4 p_matrix;
uniform vec2 inv_viewport;

float getEyeZ(vec2 texcoord)
{
  float ndcZ = 2.0*texture2D(depth_tex, texcoord).x - 1.0;  // assuming depth range of [0,1]
  return -p_matrix[3][2]/(ndcZ + p_matrix[2][2]);  // assuming p_matrix[2][3] = -1
}

"""

class PostprocessHost:

  def __init__(self, modules):
    """ `modules` should be list of postprocessing shader modules - see examples below.  Empty `modules`
      results in a pass-thru shader
    """
    self.postprocess_modules = modules
    self.shader = None


  def gl_init(self):
    code, calls, reqs = [], [], []
    for module in self.postprocess_modules:
      code.append(module.fs_code())
      calls.append("  " + module.fs_call() + ";")
      reqs.extend(module.reqs())

    code.append("\nvoid main()\n{")
    code.append("  gl_FragColor = texture2D(color_tex, texcoord);")
    if 'eye_z' in reqs:
      code.append("  float eyeZ = getEyeZ(texcoord);")
    ##if 'normal' in reqs: ...
    code.extend(calls)
    code.append("}")
    fs_code = POSTPROCESS_FRAG_SHADER + "\n".join(code)

    self.modules = [RendererConfig.header]
    vs = compileShader([m.vs_code() for m in self.modules] + [POSTPROCESS_VERT_SHADER], GL_VERTEX_SHADER)
    fs = compileShader([m.fs_code() for m in self.modules] + [fs_code], GL_FRAGMENT_SHADER)
    self.shader = compileProgram(vs, fs)
    self.modules.extend(self.postprocess_modules)
    # VAO for quad vertices
    vertices = np.array([
      1.0,1.0, -1.0,1.0, -1.0,-1.0,  # CCW triangle 1
      1.0,1.0, -1.0,-1.0, 1.0,-1.0   # CCW triangle 2
    ], dtype=np.float32)
    self.vao = glGenVertexArrays(1)
    glBindVertexArray(self.vao)
    self._verts_vbo = bind_attrib(self.shader, 'position', vertices, 2, GL_FLOAT)
    glBindVertexArray(0)


  def draw(self, viewer):
    if self.shader is None:
      self.gl_init()

    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)

    # bind gbuffer textures
    set_sampler(self.shader, 'color_tex', 0, viewer.fb_textures['color'])
    set_sampler(self.shader, 'depth_tex', 1, viewer.fb_textures['depth'])
    set_sampler(self.shader, 'normal_tex', 2, viewer.fb_textures['normal'])

    set_uniform(self.shader, 'p_matrix', 'mat4fv', viewer.camera.proj_matrix())
    set_uniform(self.shader, 'inv_viewport', '2f', [1.0/viewer.width, 1.0/viewer.height])

    # bind vertices and draw
    glBindVertexArray(self.vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glBindVertexArray(0)
    unbind_texture(0)
    unbind_texture(1)
    unbind_texture(2)
    glUseProgram(0)


  def on_key_press(self, viewer, keycode, key, mods):
    return any(m.on_key_press(viewer, keycode, key, mods) for m in self.modules if hasattr(m, 'on_key_press'))


# gamma correction - usually should be last effect
class GammaCorrEffect:

  def __init__(self, gamma=2.2):
    self.inv_gamma = 1.0/gamma

  #def vs_code(self): raise Exception("No vertex shader code for postprocess modules!")
  def fs_code(self): return """
uniform float inv_gamma;

vec4 gamma_corr(vec4 color)
{
  return vec4(pow(color.rgb, inv_gamma), color.a);
}
  """

  def fs_call(self): return "gl_FragColor = gamma_corr(gl_FragColor)"

  def reqs(self): return [] # ['depth', 'normal', 'eye_z']

  def setup_shader(self, shader, viewer):
    set_uniform(shader, 'inv_gamma', '1f', self.inv_gamma)


# Fog/depth cueing (mix in more background color the greater the fragment's depth, eventually fading out
#  completely).  Apply in postprocessing instead doesn't work quite right with transparency but I will
#  keep this as a postprocessing module for now rather than complicate line and volume shaders.
# TODO: don't touch pixels with depth = 1 (i.e. no geometry)
FOG_FRAG_MODULE = """
uniform vec4 fog_color;
uniform float fog_start;
uniform float fog_density;

vec3 calc_fog(vec3 color, float z)
{
  // fixed fn OpenGL provided linear, exponential, and e^2 fog
  // to match fixed fn OpenGL, set fog_start = 0 for exponential fog
  float fog_dist = -z - fog_start;
  // standard exponential fog
  float fog = exp(-fog_dist * fog_density);

  // linear fog
  // fog_end corresponds to the value where exponential fog factor ~ 1/256
  //float fog_end = fog_start + 5.5/fog_density;
  //float fog = (fog_end - fog_dist)/(fog_end - fog_start);

  // e^2 fog
  //float fdfd = fog_dist * fog_density;
  //float fog = exp(-fd*fd);

  return mix(fog_color.rgb, color, clamp(fog, 0.0, 1.0));
}
"""

class FogEffect:

  def __init__(self, density=0.08, start=None, color=None):
    self.fog_density = density
    self.fog_start = start
    self.fog_color = color

  def fs_code(self): return FOG_FRAG_MODULE

  def fs_call(self): return "gl_FragColor.rgb = calc_fog(gl_FragColor.rgb, eyeZ)"

  def reqs(self): return ['eye_z']

  def setup_shader(self, shader, viewer):
    if self.fog_color is None:
      self.fog_color = viewer.bg_color
    if self.fog_start is None:
      self.fog_start = viewer.camera.z_near + min(25, 0.5*(viewer.camera.z_far - viewer.camera.z_near))
    set_uniform(shader, 'fog_color', '4f', gl_color(self.fog_color))
    set_uniform(shader, 'fog_start', '1f', self.fog_start)
    set_uniform(shader, 'fog_density', '1f', self.fog_density)

  def on_key_press(self, viewer, keycode, key, mods):
    if key in 'FG':
      if 'Ctrl' in mods:
        s = (1 if key == 'G' else -1) * (0.1 if 'Shift' in mods else 0.01)
        self.fog_start += s*(viewer.camera.z_far - viewer.camera.z_near)
        print("Fog start: %.2f" % self.fog_start)
      else:
        s = np.power((1.25 if key == 'G' else 0.8), (10 if 'Shift' in mods else 1))
        self.fog_density *= s
        print("Fog density: %.3f" % self.fog_density)
      return True
    return False


OUTLINE_FRAG_MODULE = """
uniform float outline_strength;
uniform float outline_radius;
uniform vec4 outline_color;

vec3 do_outline(vec3 color)
{
  float dx = inv_viewport.x * outline_radius;
  float dy = inv_viewport.y * outline_radius;
  float n[9];
  n[0] = getEyeZ(texcoord + vec2(-dx, -dy));
  n[1] = getEyeZ(texcoord + vec2(0.0, -dy));
  n[2] = getEyeZ(texcoord + vec2( dx, -dy));
  n[3] = getEyeZ(texcoord + vec2(-dx, 0.0));
  n[4] = 0.0; // not used!
  n[5] = getEyeZ(texcoord + vec2( dx, 0.0));
  n[6] = getEyeZ(texcoord + vec2(-dx, dy));
  n[7] = getEyeZ(texcoord + vec2(0.0, dy));
  n[8] = getEyeZ(texcoord + vec2( dx, dy));

  float sobel_h = n[2] + (2.0*n[5]) + n[8] - (n[0] + (2.0*n[3]) + n[6]);
  float sobel_v = n[0] + (2.0*n[1]) + n[2] - (n[6] + (2.0*n[7]) + n[8]);
  // (approximation to) gradient magnitude
  float sobel = sqrt((sobel_h*sobel_h) + (sobel_v*sobel_v));
  float outline = clamp(sobel*outline_strength, 0.0, 1.0);

  return mix(color.rgb, outline_color.rgb, outline);
}
"""

# Outline radius: artifacts start becoming noticible with radius > 2, but radius <= 2 seems sufficient for
#  most cases.  To support radius > 2 w/o artifacts, we either need a larger kernel - see
#  https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size (scipy.signal.convolve2d) -
#  or multiple passes to blur.  We could try mipmapping the depth texture, but it must be converted to a color
#  texture first (OpenGL doesn't support mipmapping depth textures) - maybe use a pass to calc eye-space z
#  and write to color texture which can then be mipmapped (glBindTexture then glGenerateMipmap)
# A blur pass could also let us experiment with a depth halo effect
class OutlineEffect:

  def __init__(self, color=Color.black, strength=10.0, radius=1.0):
    self.color = color
    self.strength = strength
    self.radius = radius

  def fs_code(self): return OUTLINE_FRAG_MODULE

  def fs_call(self): return "gl_FragColor.rgb = do_outline(gl_FragColor.rgb)"

  def reqs(self): return ['eye_z']

  def setup_shader(self, shader, viewer):
    # divide strength by maximum gradient magnitude from Sobel operator
    outline_strength = self.strength/(8.0*np.sqrt(2)*abs(viewer.camera.z_far - viewer.camera.z_near))
    set_uniform(shader, 'outline_strength', '1f', outline_strength)
    set_uniform(shader, 'outline_radius', '1f', self.radius)
    set_uniform(shader, 'outline_color', '4f', gl_color(self.color))

  def on_key_press(self, viewer, keycode, key, mods):
    if key in '-=':
      s = np.power((1.25 if key == '=' else 0.8), (10 if 'Shift' in mods else 1))
      self.outline_strength *= s
      print("Outline strength: %0.3f" % self.strength)
    #elif key in '[]':
    #  s = (1 if key == ']' else -1) * (0.5 if 'Shift' in mods else 0.1)
    #  self.radius = max(0.1, self.radius + s)
    #  print("Outline radius: %0.3f" % self.radius)
    else:
      return False
    return True


## Ambient occlusion and depth-of-field are WIP, so we won't convert to modules yet

## Ambient Occulsion effect, using "Scalable Ambient Occlusion" technique (optimized version of Alchemy AO)
# Primary ref: http://casual-effects.com/g3d/G3D10/data-files/shader/AmbientOcclusion/AmbientOcclusion_AO.pix
# Additional refs:
#  SAO paper: http://research.nvidia.com/sites/default/files/publications/McGuire12SAO.pdf
#  Alchemy AO paper: http://research.nvidia.com/sites/default/files/pubs/2011-08_The-Alchemy-Screen-space/paper.pdf
#  three.js SAO PR: https://github.com/mrdoob/three.js/pull/11458/files
#  Comprehensive comparision of AO techniques with code:
#   http://frederikaalund.com/a-comparative-study-of-screen-space-ambient-occlusion-methods/  (open the PDF)
#   https://github.com/frederikaalund/sfj/tree/master/black_label/libraries/rendering/shaders
#  Another AO comparison w/ code: https://upcommons.upc.edu/bitstream/handle/2117/82591/114624.pdf

# Ambient occulsion approximates one effect of global illumination and greatly improves appearance of
#  spacefilling models of molecules
# Screen-space techniques use depth buffer and sometimes surface normals to recover some info about 3D
#  geometry
# SSAO - original screen-space AO technique; samples geometry at order of 10 random *camera space* points in
#  hemisphere around fragment by comparing depth of random point to depth buffer to check for occluding geom
# - requires additional randomization of sample points and 2nd blur pass to suppress artifacts
# HBAO (horizon-based AO): march along a few directions in camera space to find occluding geometry around pt
# Alchemy AO/SAO (scalable AO): generate random points in screen space around fragment, then determine the
#  camera space position of the geometry at each point and calculate contribution to occlusion based on
#  relative position vs. current fragment's camera space position
# - generally superior to SSAO since every point contributes to occlusion, whereas with SSAO only points
#  hitting geometry that is closer to camera can contribute
# Bent normals: when sampling in camera space, we can use the median unoccluded direction as the normal
#  for diffuse lighting instead of the actual surface normal

# More advanced AO and even better approximations to global illumination (real shadows, etc.) can be achieved
#  using coarse 3D voxel or distance field map of world
# AO techniques are called voxel AO (VXAO) and distance-field AO (DFAO)
# For molecular geometry, we could even try using only ray casting for rendering

# - since our geometry is static, precalculating AO could be an option
# - using low frequency noise (function of world space position) for sampling pattern rotation angle
#  (randAngle) yields "trippy" patterns, and doesn't appear to be useful
# - for now, we are relying on clamping to border = 1 (max depth) for depth texture rather than discarding
#  off-screen samples
# - in light of comment in SAO paper that normal recovered from depth is more accurate than RGB888 normal,
#  we could try writing eye-space depth instead of normal in geom pass
# - using dFdx(V),dFdy(V) to apply 2x2 box filter to doesn't have a significant effect

# use of texelFetch requires GLSL 1.3; maybe try texture2D instead (GLSL 1.2)
SAO_FRAG_SHADER = """
#define HIGH_QUALITY 1
#define NORMAL_TEXTURE 1

#define TWO_PI 6.28318530718

uniform sampler2D color_tex;
uniform sampler2D depth_tex;
#ifdef NORMAL_TEXTURE
uniform sampler2D normal_tex;
#endif

// The height in pixels of a 1m object if viewed from 1m away.
uniform float projScale;

// World-space AO radius in scene units (r).  e.g., 1.0m
uniform float radius;
// Bias to avoid AO in smooth corners, e.g., 0.01m
uniform float bias;
// constant controlling effect strength - default value is 1.0
uniform float intensity;

// may want to make these compile-time constants
uniform int nSamples;
uniform int nTurns;  // number of complete turns for spiral sample pattern

uniform mat4 p_matrix;
uniform vec4 ssToCsPrecalc;

uniform mat4 inv_view_matrix;


vec3 screenSpaceToCameraSpace(ivec2 ssP)
{
  float ndcZ = 2.0*texelFetch(depth_tex, ssP, 0).x - 1.0;  // assuming depth range of [0,1]
  float eyeZ = -p_matrix[3][2]/(ndcZ + p_matrix[2][2]);  // assuming p_matrix[2][3] = -1
  vec2 ssPf = vec2(ssP) + vec2(0.5);
  return vec3(eyeZ*(ssPf*ssToCsPrecalc.xy + ssToCsPrecalc.zw), eyeZ);
}

float radius2, invRadius2;

float fallOffFunction(vec3 C, vec3 n_C, vec3 Q)
{
  const float epsilon = 0.001;  // to prevent divide by zero is vv is 0
  vec3 v = Q - C;
  float vv = dot(v, v);
  float vn = dot(v, n_C);

  // A: From the HPG12 paper
  // Note large epsilon to avoid overdarkening within cracks
  //  Assumes the desired result is intensity/radius^6 in main()
  //return float(vv < radius2) * max((vn - bias) / (epsilon + vv), 0.0) * radius2 * 0.6;

  // B: Smoother transition to zero (lowers contrast, smoothing out corners). [Recommended]
#ifdef HIGH_QUALITY
  // Epsilon inside the sqrt for inv sqrt operation
  float f = max(1.0 - vv * invRadius2, 0.0);
  return f * max((vn - bias) * inversesqrt(epsilon + vv), 0.0);
#else
  // Avoid the square root from above.
  //  Assumes the desired result is intensity/radius^6 in main()
  float f = max(radius2 - vv, 0.0);
  return f * f * f * max((vn - bias) / (epsilon + vv), 0.0);
#endif

  // C: Medium contrast (which looks better at high radii), no division.  Note that the
  // contribution still falls off with radius^2, but we've adjusted the rate in a way that is
  // more computationally efficient and happens to be aesthetically pleasing.  Assumes
  // division by radius^6 in main()
  //return 4.0 * max(1.0 - vv * invRadius2, 0.0) * max(vn - bias, 0.0);

  // D: Low contrast, no division operation
  //return 2.0 * float(vv < radius2) * max(vn - bias, 0.0);
}

/** Compute the occlusion due to sample with index \a i about the pixel at \a ssC that corresponds
  to camera-space point \a C with unit normal \a n_C, using maximum screen-space sampling radius \a ssDiskRadius

  Note that units of H() in the HPG12 paper are meters, not unitless.  The whole falloff/sampling function
  is therefore unitless.  In this implementation, we factor out (9 / radius).
*/
float invNSamples;

float sampleAO(ivec2 ssC, vec3 C, vec3 n_C, float ssDiskRadius, int tapIndex, float randAngle)
{
  // Radius relative to ssR
  float ssR = float(tapIndex + 0.5) * invNSamples;
  float angle = ssR * (nTurns * TWO_PI) + randAngle;
  // Offset on the unit disk, spun for this pixel
  vec2 unitOffset = vec2(cos(angle), sin(angle));
  // Ensure that the taps are at least 1 pixel away
  ssR = max(0.75, ssR * ssDiskRadius);
  // The occluding point in camera space
  vec3 Q = screenSpaceToCameraSpace(ivec2(ssR * unitOffset) + ssC);

  // Without the angular adjustment term, surfaces seen head on have less AO
  return fallOffFunction(C, n_C, Q) * mix(1.0, max(0.0, 1.5 * n_C.z), 0.35);
}

const float MIN_RADIUS = 3.0; // pixels

// interestingly, collapsing to a single expression introduces artifacts even when not using GLES2
highp float rand(const vec2 uv)
{
  const highp float a = 12.9898, b = 78.233, c = 43758.5453;
  highp float dt = dot(uv.xy, vec2(a, b)), sn = mod(dt, 3.14);
  return fract(sin(sn) * c);
}

highp float rand3(const vec3 r)
{
  const highp float a = 12.9898, b = 78.233, c = 54.335, d = 43758.5453;
  highp float dt = dot(r, vec3(a, b, c)), sn = mod(dt, 3.14);
  return fract(sin(sn) * d);
}

float getVisibility(ivec2 ssC)
{
  // don't run if no geometry
  float ssZ = texelFetch(depth_tex, ssC, 0).x;
  if(ssZ >= 1.0)
    return 1.0;

  // precompute some values
  invNSamples = 1.0/nSamples;
  radius2 = radius*radius;
  invRadius2 = 1.0/radius2;

  // camera space coords of geometry at current pixel
  vec3 C = screenSpaceToCameraSpace(ssC);

#ifdef NORMAL_TEXTURE
  // normal is stored in normal_tex as 0.5*n + 0.5
  vec3 n_C = 2.0*(texelFetch(normal_tex, ssC, 0).xyz - 0.5);
#else
  // Reconstruct normals from positions - since n_C is computed from the cross product of camera-space edge
  //  vectors from points at adjacent pixels, its magnitude will be proportional to the square of distance
  //  from the camera, except at depth discontinuities, where calculation is not reliable
  // SAO paper claims this is more accurate than RGB888 normal buffer
  vec3 n_C = cross(dFdy(C), dFdx(C));
  // threshold too large -> black dots where due to bad normal at edges; threshold too small -> white dots
  if(dot(n_C, n_C) > pow(C.z * C.z * 0.00006, 2))
    return 1.0;
  else
    n_C = normalize(n_C);
#endif

  // Hash function used in the HPG12 AlchemyAO paper; return mod(randAngle / TWO_PI, 1.0) to debug
  //float randAngle = (((3 * ssC.x) ^ (ssC.y + ssC.x * ssC.y))) * 10;  // hyperbole-like artifacts
  //float randAngle = (30u * ssC.x) ^ (ssC.y + 10u * ssC.x * ssC.y);
  //~float randAngle = mod( (((3 * ssC.x) ^ (ssC.y + ssC.x * ssC.y))) * 10, TWO_PI ); // no major artifacts
  //float randAngle = TWO_PI*fract(sin(mod(dot(ssC.xy, vec2(12.9898, 78.233)), TWO_PI) ) * 43758.5453);
  //float randAngle = TWO_PI*rand(ssC);


  vec4 wsC = inv_view_matrix*vec4(C, 1.0);
  //float randAngle = TWO_PI*rand3(floor(100*wsC.xyz));
  ivec3 iwsC = ivec3(40*wsC.xyz);
  float randAngle = mod( (((3 * iwsC.x + iwsC.x * iwsC.z) ^ (iwsC.y + iwsC.x * iwsC.y))) * 10, TWO_PI );


  // Choose the screen-space sample radius proportional to the projected area of the sphere
  float ssDiskRadius = -projScale * radius / C.z;
  if(ssDiskRadius <= MIN_RADIUS)
    return 1.0;

  float sum = 0.0;
  for (int i = 0; i < nSamples; ++i) {
    sum += sampleAO(ssC, C, n_C, ssDiskRadius, i, randAngle);
  }

#ifdef HIGH_QUALITY
  float A = pow(max(0.0, 1.0 - sqrt(sum * (3.0*invNSamples))), intensity);
#else
  float intensityDivR6 = intensity/pow(radius, 6);
  float A = max(0.0, 1.0 - sum * intensityDivR6 * (5.0*invNSamples));
  // Anti-tone map to reduce contrast and drag dark region farther; (x^0.2 + 1.2 * x^4)/2.2
  A = (pow(A, 0.2) + 1.2 * A*A*A*A) / 2.2;
#endif

  // Fade in as the radius reaches 2 pixels
  return mix(1.0, A, clamp(ssDiskRadius - MIN_RADIUS, 0.0, 1.0));
}

void main()
{
  ivec2 ssC = ivec2(gl_FragCoord.xy);  // screen space coords of current pixel
  float V = getVisibility(ssC);
  vec4 color = texelFetch(color_tex, ssC, 0);
  gl_FragColor = vec4(color.rgb*V, color.a);
  //gl_FragColor = vec4(V, V, V, 1.0);
}
"""

# obviously need to factor out common code with PassThruEffect
# fallOffFunction option B: radius of 5 to 10 seems reasonable; need >=20 to darken pockets though
# rel_bias doesn't seem to have noticable effect until >0.05; ~>1 turns off AO
# fallOffFunction A & C: need to increase intensity when increasing radius - fix this!
class AOEffect:
  def __init__(self, intensity=1.0, radius=8.0, nsamples=19):
    self.intensity = intensity
    self.radius = radius
    self.rel_bias = 0.01
    self.nsamples = nsamples
    self.modules = [RendererConfig.header130]
    self.shader = None


  def gl_init(self):
    vs = compileShader([m.vs_code() for m in self.modules] + [POSTPROCESS_VERT_SHADER], GL_VERTEX_SHADER)
    fs = compileShader([m.fs_code() for m in self.modules] + [SAO_FRAG_SHADER], GL_FRAGMENT_SHADER)
    self.shader = compileProgram(vs, fs)
    # VAO for quad vertices
    vertices = np.array([
      1.0,1.0, -1.0,1.0, -1.0,-1.0,  # CCW triangle 1
      1.0,1.0, -1.0,-1.0, 1.0,-1.0   # CCW triangle 2
    ], dtype=np.float32)
    self.vao = glGenVertexArrays(1)
    glBindVertexArray(self.vao)
    self._verts_vbo = bind_attrib(self.shader, 'position', vertices, 2, GL_FLOAT)
    glBindVertexArray(0)


  # from http://g3d.cs.williams.edu/g3d/G3D10/G3D-app.lib/source/AmbientOcclusionSettings.cpp
  #  for >100 samples, we just use some large prime
  #   0   1   2   3   4   5   6   7   8   9
  NUM_TURNS_FOR_SAMPLES = [
      1,  1,  1,  2,  3,  2,  5,  2,  3,  2,
      3,  3,  5,  5,  3,  4,  7,  5,  5,  7,
      9,  8,  5,  5,  7,  7,  7,  8,  5,  8,
     11, 12,  7, 10, 13,  8, 11,  8,  7, 14,
     11, 11, 13, 12, 13, 19, 17, 13, 11, 18,
     19, 11, 11, 14, 17, 21, 15, 16, 17, 18,
     13, 17, 11, 17, 19, 18, 25, 18, 19, 19,
     29, 21, 19, 27, 31, 29, 21, 18, 17, 29,
     31, 31, 23, 18, 25, 26, 25, 23, 19, 34,
     19, 27, 21, 25, 39, 29, 17, 21, 27, 29]

  def draw(self, viewer):
    if self.shader is None:
      self.gl_init()

    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)

    # bind gbuffer textures
    set_sampler(self.shader, 'color_tex', 0, viewer.fb_textures['color'])
    set_sampler(self.shader, 'depth_tex', 1, viewer.fb_textures['depth'])
    set_sampler(self.shader, 'normal_tex', 2, viewer.fb_textures['normal'])

    # uniforms
    P = viewer.proj_matrix()
    set_uniform(self.shader, 'p_matrix', 'mat4fv', P)
    # precalculated values to eliminate divides from screen space to camera space calculation
    ss_to_cs = [-2.0/(viewer.width*P[0][0]), -2.0/(viewer.height*P[1][1]), 1.0/P[0][0], 1.0/P[1][1]]
    set_uniform(self.shader, 'ssToCsPrecalc', '4f', ss_to_cs)


    set_uniform(self.shader, 'inv_view_matrix', 'mat4fv', np.linalg.inv(viewer.view_matrix()))


    set_uniform(self.shader, 'nSamples', '1i', self.nsamples)
    try: nturns = AOEffect.NUM_TURNS_FOR_SAMPLES[self.nsamples]
    except: nturns = 5779
    set_uniform(self.shader, 'nTurns', '1i', nturns)

    # AO sampling region radius in world space units (here, Angstroms)
    set_uniform(self.shader, 'radius', '1f', self.radius)
    # value in world space units, small relative to radius, to avoid AO in smooth corners
    set_uniform(self.shader, 'bias', '1f', self.radius*self.rel_bias)
    # constant controlling effect strength
    set_uniform(self.shader, 'intensity', '1f', self.intensity)

    # move this to camera.py - gives pixels per camera space unit at image plane
    scale = abs(2.0*np.tan(0.5*viewer.camera.fov*np.pi/180.0))
    # camera.fov is for vertical direction
    set_uniform(self.shader, 'projScale', '1f', viewer.height/scale)

    # bind vertices and draw
    glBindVertexArray(self.vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glBindVertexArray(0)
    glUseProgram(0)


  def on_key_press(self, viewer, keycode, key, mods):
    if key in '-=':
      s = np.power((1.25 if key == '=' else 0.8), (10 if 'Shift' in mods else 1))
      self.intensity *= s
      print("AO intensity: %0.3f" % self.intensity)
    elif key in '[]':
      s = np.power((1.25 if key == ']' else 0.8), (10 if 'Shift' in mods else 1))
      self.radius *= s
      print("AO radius: %0.3f" % self.radius)
    #~elif key in ";'":
    #~  s = np.power((1.25 if key == "'" else 0.8), (10 if 'Shift' in mods else 1))
    #~  self.effect.rel_bias *= s
    #~  print("AO rel_bias: %0.3f" % self.rel_bias)
    #~elif key in 'GH':
    #~  s = np.power((1.25 if key == 'H' else 0.8), (10 if 'Shift' in mods else 1))
    #~  self.effect.rel_scale *= s
    #~  print("AO rel_scale: %0.3f" % self.rel_scale)
    else:
      return False
    return True


# two-pass Gaussian blur shader
DOF_FRAG_SHADER = """
varying vec2 texcoord;

uniform sampler2D color_tex;
uniform sampler2D depth_tex;

// parameters for calculating circle of confusion radius directly from screen-space depth
uniform float coc_scale;
uniform float coc_offset;
uniform vec2 direction;  // (0,1/height) or (1/width,0)

// #define KERNEL_RADIUS <n> should be prepended to this chunk
// assumes a symmetric filter, so weight[-ii] = weight[ii] is implicit; also, weights assumed to be normalized
uniform float weights[KERNEL_RADIUS + 1];

uniform mat4 p_matrix;

float getEyeZ(vec2 texcoord)
{
  float ndcZ = 2.0*texture2D(depth_tex, texcoord).x - 1.0;  // assuming depth range of [0,1]
  return -p_matrix[3][2]/(ndcZ + p_matrix[2][2]);  // assuming p_matrix[2][3] = -1
}

void main()
{
  float ssZ = texture2D(depth_tex, texcoord).x;
  // circle of confusion radius
  //float radius = abs(ssZ*coc_scale + coc_offset);

  float radius = coc_scale*abs(getEyeZ(texcoord) - coc_offset);

  vec2 dxy = (radius/KERNEL_RADIUS)*direction;
  //vec4 sum = weights[0]*texture2D(color_tex, texcoord);
  float tot = weights[0]*abs(getEyeZ(texcoord) - coc_offset);

  vec4 sum = tot*texture2D(color_tex, texcoord);

  for(int ii = 1; ii <= KERNEL_RADIUS; ++ii) {
    //sum += weights[ii]*(texture2D(color_tex, texcoord + dxy*ii) + texture2D(color_tex, texcoord - dxy*ii));


    // this prevents bluring in-focus geometry onto background
    float s0 = weights[ii]*abs(getEyeZ(texcoord + dxy*ii) - coc_offset);
    float s1 = weights[ii]*abs(getEyeZ(texcoord - dxy*ii) - coc_offset);
    sum += s0*texture2D(color_tex, texcoord + dxy*ii);
    sum += s1*texture2D(color_tex, texcoord - dxy*ii);
    tot += s0 + s1;
  }
  gl_FragColor = sum/tot;  // no divide since weights are normalized
}
"""


SPECK_DOF_FRAG_SHADER = """
varying vec2 texcoord;

uniform sampler2D color_tex;
uniform sampler2D depth_tex;

uniform vec2 inv_viewport;
uniform float coc_offset;
uniform float coc_scale;

// unused
uniform vec2 direction;  // (0,1/height) or (1/width,0)
uniform float weights[KERNEL_RADIUS + 1];
uniform mat4 p_matrix;

//uniform vec2 usamples[64];


float getEyeZ(vec2 texcoord)
{
  float ndcZ = 2.0*texture2D(depth_tex, texcoord).x - 1.0;  // assuming depth range of [0,1]
  return -p_matrix[3][2]/(ndcZ + p_matrix[2][2]);  // assuming p_matrix[2][3] = -1
}

void main()
{
  // these points are uniformly (not purely random) distributed over unit circle (radius = 1)
  vec2 samples[64];
  samples[0] = vec2(0.857612, 0.019885);
  samples[1] = vec2(0.563809, -0.028071);
  samples[2] = vec2(0.825599, -0.346856);
  samples[3] = vec2(0.126584, -0.380959);
  samples[4] = vec2(0.782948, 0.594322);
  samples[5] = vec2(0.292148, -0.543265);
  samples[6] = vec2(0.130700, 0.330220);
  samples[7] = vec2(0.236088, 0.159604);
  samples[8] = vec2(-0.305259, 0.810505);
  samples[9] = vec2(0.269616, 0.923026);
  samples[10] = vec2(0.484486, 0.371845);
  samples[11] = vec2(-0.638057, 0.080447);
  samples[12] = vec2(0.199629, 0.667280);
  samples[13] = vec2(-0.861043, -0.370583);
  samples[14] = vec2(-0.040652, -0.996174);
  samples[15] = vec2(0.330458, -0.282111);
  samples[16] = vec2(0.647795, -0.214354);
  samples[17] = vec2(0.030422, -0.189908);
  samples[18] = vec2(0.177430, -0.721124);
  samples[19] = vec2(-0.461163, -0.327434);
  samples[20] = vec2(-0.410012, -0.734504);
  samples[21] = vec2(-0.616334, -0.626069);
  samples[22] = vec2(0.590759, -0.726479);
  samples[23] = vec2(-0.590794, 0.805365);
  samples[24] = vec2(-0.924561, -0.163739);
  samples[25] = vec2(-0.323028, 0.526960);
  samples[26] = vec2(0.642128, 0.752577);
  samples[27] = vec2(0.173625, -0.952386);
  samples[28] = vec2(0.759014, 0.330311);
  samples[29] = vec2(-0.360526, -0.032013);
  samples[30] = vec2(-0.035320, 0.968156);
  samples[31] = vec2(0.585478, -0.431068);
  samples[32] = vec2(-0.244766, -0.906947);
  samples[33] = vec2(-0.853096, 0.184615);
  samples[34] = vec2(-0.089061, 0.104648);
  samples[35] = vec2(-0.437613, 0.285308);
  samples[36] = vec2(-0.654098, 0.379841);
  samples[37] = vec2(-0.128663, 0.456572);
  samples[38] = vec2(0.015980, -0.568170);
  samples[39] = vec2(-0.043966, -0.771940);
  samples[40] = vec2(0.346512, -0.071238);
  samples[41] = vec2(-0.207921, -0.209121);
  samples[42] = vec2(-0.624075, -0.189224);
  samples[43] = vec2(-0.120618, 0.689339);
  samples[44] = vec2(-0.664679, -0.410200);
  samples[45] = vec2(0.371945, -0.880573);
  samples[46] = vec2(-0.743251, 0.629998);
  samples[47] = vec2(-0.191926, -0.413946);
  samples[48] = vec2(0.449574, 0.833373);
  samples[49] = vec2(0.299587, 0.449113);
  samples[50] = vec2(-0.900432, 0.399319);
  samples[51] = vec2(0.762613, -0.544796);
  samples[52] = vec2(0.606462, 0.174233);
  samples[53] = vec2(0.962185, -0.167019);
  samples[54] = vec2(0.960990, 0.249552);
  samples[55] = vec2(0.570397, 0.559146);
  samples[56] = vec2(-0.537514, 0.555019);
  samples[57] = vec2(0.108491, -0.003232);
  samples[58] = vec2(-0.237693, -0.615428);
  samples[59] = vec2(-0.217313, 0.261084);
  samples[60] = vec2(-0.998966, 0.025692);
  samples[61] = vec2(-0.418554, -0.527508);
  samples[62] = vec2(-0.822629, -0.567797);
  samples[63] = vec2(0.061945, 0.522105);

  //float depth = getEyeZ(texcoord);

  float depth = 100.0*texture2D(depth_tex, texcoord).r;
  vec2 scale = (64.0 * coc_scale * abs(coc_offset - depth)) * inv_viewport;

  vec4 sample = texture2D(color_tex, texcoord);
  float count = 1.0;
  for(int i = 0; i < 64; i++) {
    vec2 p = texcoord + scale * samples[i];
    //float d = getEyeZ(p);
    float d = 100.0*texture2D(depth_tex, p).r;
    float s = abs(coc_offset - d);
    sample += texture2D(color_tex, p) * s;
    count += s;
  }

  gl_FragColor = sample/count;
}
"""

# Depth of field/blur
# The Gaussian is the only separable, circularly symmetric (radial) fn in Real^n, where
#  separable means that f(x1, x2, ...) = f1(x1)f2(x2)..., so that as a discrete 2D filter, the kernel can be
#  written as the outer product of two 1D vectors and thus implemented as two passes (one horizontal, one
#  vertical), each reading only n pixels instead of one pass reading n^2 pixels (where n is the kernel width)
# A blur simulating an out-of-focus camera, called a bokeh blur, has a kernel more like a uniform disk, not a
#  Gaussian

# Observations:
# Gaussian blur: can look OK with sufficiently large kernel radius (~18) and logic to prevent bluring
#  in-focus geometry onto background pixels
# Disk blur from Speck:
# - two passes of Speck shader produces nice result (found accidentally)
# - per-pixel random rotation of pattern doesn't seem to do much
# - uniform pattern seems superior to purely random
# - see https://en.wikipedia.org/wiki/Low-discrepancy_sequence for more!
# It is possible to approximate a disk blur as a separable filter (i.e. two 1D passes) using complex numbers:
#    www.shadertoy.com/view/lsBBWy , www.shadertoy.com/view/Xd2BWc , http://yehar.com/blog/?p=1495
#  but since we want to consider to depth of every pixel contributing, I don't think we can use this approach
# One issue (with Gaussian and disk) is that foreground geometry doesn't blur properly over in-focus region
# Ultimately, DOF doesn't seem super-useful for exploratory work, so we shouldn't spend too much time on it


class DOFEffect:

  def __init__(self, z_focus=50.0, aperture=10.0, focal_len=10.0, kernel_radius=6):
    self.z_focus = z_focus
    self.aperture = aperture
    self.focal_len = focal_len
    # std dev should be somewhere between 1/2 and 1/3 of kernel radius; actual filter radius is determined by
    #  by radius used for texture sampling in shader
    sigma = kernel_radius/2.0
    weights = np.exp(-0.5*np.square(np.arange(kernel_radius + 1.0)/sigma))
    self.weights = weights/(weights[0] + 2*np.sum(weights[1:]))
    self.modules = [RendererConfig.header130]
    self.shader = None


  def gl_init(self):
    k_rad = "#define KERNEL_RADIUS %d\n" % (len(self.weights)-1)
    vs = compileShader([m.vs_code() for m in self.modules] + [POSTPROCESS_VERT_SHADER], GL_VERTEX_SHADER)
    fs = compileShader([m.fs_code() for m in self.modules] + [k_rad, SPECK_DOF_FRAG_SHADER], GL_FRAGMENT_SHADER)
    self.shader = compileProgram(vs, fs)
    # VAO for quad vertices
    vertices = np.array([
      1.0,1.0, -1.0,1.0, -1.0,-1.0,  # CCW triangle 1
      1.0,1.0, -1.0,-1.0, 1.0,-1.0   # CCW triangle 2
    ], dtype=np.float32)
    self.vao = glGenVertexArrays(1)
    glBindVertexArray(self.vao)
    self._verts_vbo = bind_attrib(self.shader, 'position', vertices, 2, GL_FLOAT)
    glBindVertexArray(0)


  def draw(self, viewer, to_screen):
    z_far, z_near = viewer.camera.z_far, viewer.camera.z_near
    if self.shader is None:
      self.gl_init()
      # initialize parameters to something reasonable
      self.z_focus = -0.5*(z_far + z_near)

    ## HUGE MESS
    # use framebuffer since we make two passes
#~     color_in = viewer.swap_color_tex()

    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)

    # bind gbuffer textures
#~     set_sampler(self.shader, 'color_tex', 0, color_in)
    set_sampler(self.shader, 'depth_tex', 1, viewer.fb_textures['depth_2'])
    # vertical blur to framebuffer
    set_uniform(self.shader, 'direction', '2f', [0.0, 1.0/viewer.height])
    set_uniform(self.shader, 'weights', '1fv', self.weights)

    # ref: https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems/gpugems_ch23.html
    #A = self.aperture*self.focal_len
    #coc_scale = A*self.z_focus*(z_far - z_near)/((self.z_focus - self.focal_len)*z_near*z_far)
    #coc_offset = A*(z_near - self.z_focus)/(self.z_focus*self.focal_len*z_near)

    set_uniform(self.shader, 'p_matrix', 'mat4fv', viewer.camera.proj_matrix())

    set_uniform(self.shader, 'inv_viewport', '2f', [1.0/viewer.width, 1.0/viewer.height])

    #n_samples = 64
    #angles = 2.0*np.pi*np.random.rand(n_samples)
    #radii = np.sqrt(np.random.rand(n_samples))
    #samples = radii[:,None]*np.stack((np.sin(angles), np.cos(angles)), axis=1)
    #print samples
    #set_uniform(self.shader, 'usamples', '2fv', samples)


    coc_offset = self.z_focus
    coc_scale = self.aperture/np.sqrt(viewer.width*viewer.height)

    set_uniform(self.shader, 'coc_scale', '1f', coc_scale)
    set_uniform(self.shader, 'coc_offset', '1f', coc_offset)

    # bind vertices and draw first pass to framebuffer
    glBindVertexArray(self.vao)
#~     glDrawArrays(GL_TRIANGLES, 0, 6)

    # second pass: horizontal blur
    if to_screen:
      viewer.bind_framebuffer(0)
      color_in = viewer.fb_textures['color']
    else:
      color_in = viewer.swap_color_tex()
    set_sampler(self.shader, 'color_tex', 0, color_in)
    set_uniform(self.shader, 'direction', '2f', [1.0/viewer.width, 0.0])
    glDrawArrays(GL_TRIANGLES, 0, 6)

    glBindVertexArray(0)
    glUseProgram(0)


  def on_key_press(self, viewer, keycode, key, mods):
    if key in '-=':
      s = (1 if key == '=' else -1) * (0.1 if 'Shift' in mods else 0.01)
      self.z_focus += s*(viewer.camera.z_far - viewer.camera.z_near)
      print("DOF z focus: %0.3f" % self.z_focus)
    elif key in '[]':
      s = np.power((1.25 if key == ']' else 0.8), (10 if 'Shift' in mods else 1))
      self.aperture *= s
      print("DOF aperture: %0.3f" % self.aperture)
    elif key in ";'":
      s = np.power((1.25 if key == "'" else 0.8), (10 if 'Shift' in mods else 1))
      self.focal_len *= s
      print("DOF focal length: %0.3f" % self.focal_len)
    else:
      return False
    return True

