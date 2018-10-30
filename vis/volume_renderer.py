# Volume renderer copied from pyvol

import ctypes
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO
from glutils import *


# Refs (C+S+o to open!)
# - visvis/wobjects/textures.py, visvis/core/shaders_3.py - does do depth calculation, looks same as vispy
# - vispy/visuals/volume.py - higher GLSL version, but no depth calculation

# More refs; 2 pass unless noted
# - http://prideout.net/blog/?p=64 and http://devlog-martinsh.blogspot.com/2011/06/glsl-raycasting-clipping-plane.html (single pass)
# - http://www.voreen.org/ ; https://github.com/bilgili/Voreen - lots of shaders!
#  - see mod_compositing.frag, compositing.frag, and rc_singlevolume.frag
# - http://www.mccauslandcenter.sc.edu/mricrogl/source ; http://www.cabiatl.com/mricro/raycast/
# - http://www.lebarba.com/

# Shader based isosurface visualization (not to mention other volume visualization methods) seems mostly
#  superior to isosurface precalculation, although FPS will be lower ... if you do need precalculation, see
#  vispy/geometry/isosurface.py or PyMCubes (faster C++ impl) for marching cubes calculation of isosurface
# There are also GPU techniques for marching cubes, using geom shader or "histopyramids", e.g. https://github.com/sintefmath/hpmc

# volume rendering approaches:
# src = color from new voxel; dest = accumulated color from previous voxels
# 1. blending: scale current voxel color based on absorption and blend with dest
# 2. accumulate total absorption (1-T) and set alpha = 1-T, set color = colormap(maxvalue/scale)
# 3. vispy "translucent" approach - great for MRI image, not good for MO
# 4. set color from max value (aka MIP: maximum intensity projection), no variation of transparency except from colormap
# - let's limit ourselves to two interactive parameters (i.e. brightness and contrast; even that might be one too many!)
# - consider taking log() of high-dynamic range data before passing to VolumeRenderer

# Clipping for volume rendering
# - allows proper rendering of fog and transparent isosurfaces over opaque geometry
# - we accomplish by passing depth buffer for opaque geometry to volume rendering shader as a texture.  Frag
#  shader recovers position of opaque geometry (see below) and terminates ray there

# alpha blending (Porter-Duff): dest is buffer, src is new fragment
# src over (for back to front rendering): dest.rgb = src.a*src.rgb + (1-src.a)*dest.rgb
# dest over (for front to back rendering): starting with dest.a = 0,
#  dest.rgb = src.a*(1-dest.a)*src.rgb + dest.rgb, dest.a = (1-dest.a)*src.a + dest.a

# Calculation of depth buffer value (i.e. window space depth) from camera space point:
# Usual camera space to clip space transform: r_clip = projection_mat * r_camera
# Usual clip space to NDC transform: r_ndc = r_clip.xyz/r_clip.w
# NDC to screen (z): gl_FragDepth =
#   r_ndc.z*(gl_DepthRange.far - gl_DepthRange.near)/2 + (gl_DepthRange.far + gl_DepthRange.near)/2,
# where gl_DepthRange can set by calling glDepthRange() - but usually the defaults of near = 0, far = 1 are
# best, since [0,1] is the largest range possible, in which case
#   gl_FragDepth = (r_ndc.z + 1)/2 = (r_clip.z/r_clip.w + 1)/2

# Calculation of camera (or model) space point from window space depth:
# - we want to invert the above calculation, but without access to w component
# window space to NDC: r_ndc = 2*r_window - 1,
#  assuming default gl_DepthRange of [0,1] (otherwise, see above for z calculation)
# Assuming that only P[2][2], P[2][3], and P[3][2] are non-zero in the lower half of the projection matrix:
#   r_clip.w = P[3][2]/(r_ndc.z - P[2][2]/P[2][3])
#   r_clip.xyz = r_ndc.xyz * r_clip.w
#   r_camera = inv_P*r_clip
# We can also bypass r_clip and go directly from NDC z to camera space z:
#   r_camera.z = P[3][2]/(P[2][3]*r_ndc.z - P[2][2])
# Given r_camera.z and the view direction in camera space, view_camera, we can recover the full r_camera:
#   r_camera = (r_camera.z/view_camera.z)*view_camera
# Note view_camera = mv_matrix*vec4(view_model, 0).xyz, where view_model is the view direction in model space
# Alternatively, since r_clip.w = P[2][3]*r_camera.z = -r_camera.z and assuming symmetric viewport:
#   r_camera.xy = -r_camera.z*r_ndc.xy/vec2(P[0][0], P[1][1])
# Another alternative calculation, which seems wrong in general (note the divide after multiplying by
#  inv_P), but gives the correct result here, I suspect because of the form of P:
#   r_camera = inv_P*r_ndc;
#   r_camera = r_camera/r_camera.w
# Finally, if we need the model space position: r_model = inv_mv_matrix*r_camera
# Refs:
# - https://www.khronos.org/opengl/wiki/Compute_eye_space_from_window_space
# - https://stackoverflow.com/questions/11277501

# Note that in any raymarching loop, we must have a foolproof compare like i < 1200 in case anything goes wrong
#  computing nsteps - otherwise shader will hang and vmware will crash (bug was introduced between hg r14 and r15)

# TODO:
# - add a pass with full screen quad to copy depth values (if anything outside vol object bounds)
#  - shouldn't need extra pass, just a flag for full-screen quad!
# - try removing explicit MAX_STEPS checks!
# - try PyMol-type volume rendering: render multiple planes slicing through 3D texture; each frag on a given
#  plane samples texture just once; normal alpha blending combines frags; see pymolwiki.org/index.php/Volume
#  and https://sourceforge.net/p/pymol/code/HEAD/tree/trunk/pymol/layer2/ObjectVolume.cpp
#  PyMol's point-and-click editor for arbitrary mapping from density to colors is also interesting
# - see http://graphicsrunner.blogspot.com/search?updated-max=2010-04-10T01:38:00-04:00&max-results=10 for
#   more on density -> color mapping plus misc optimizations (gradient precalc, coarse-graining to skip empty)
# - A couple more refs to investigate:
#  - https://github.com/nopjia/WebGL-Volumetric/blob/master/shaders/vol-fs.glsl
#  - https://github.com/maartenbreddels/ipyvolume/blob/master/js/glsl/volr-fragment.glsl
# - make sure ray_steps ~ sample_density works for sample_density < 20


VERT_SHADER = """
uniform mat4 mvp_matrix;

attribute vec3 texcoord;

varying vec3 v_position;

void main()
{
  gl_Position = mvp_matrix * vec4(texcoord, 1.0);
  v_position = texcoord;
}
"""

FRAG_SHADER_MAIN = """
#define MAX_STEPS 1200

varying vec3 v_position;

uniform mat4 mv_matrix;
uniform mat4 inv_mv_matrix;
uniform mat4 mvp_matrix;
uniform vec3 vol_shape;
uniform float ray_steps;

uniform sampler3D texture_3d;
uniform sampler2D depth_tex;
uniform sampler1D colormap;

uniform float isolevel;
uniform vec4 color;
uniform float isolevel_n;
uniform vec4 color_n;
uniform float absorption;
uniform vec2 viewport;

// FRAG_SHADER_MAIN will come before definition of renderVol(), so declare here
void renderVol(vec3 pos, vec3 rayStop, vec3 step, int nsteps, out vec4 final_color, out vec3 final_pos);

void main()
{
  vec3 viewOrigin = inv_mv_matrix[3].xyz;  // = inv_mv_matrix * vec4(0,0,0,1)
  vec3 viewDir = normalize(v_position - viewOrigin);
  vec3 invR = 1.0/viewDir;

  // calculate where view ray hits background geometry - direct calculation for general proj matrix
  // see above comment about camera space point calculation from window space depth for other methods
  float bg_depth = texture2D(depth_tex, gl_FragCoord.xy/viewport).x;
  float bg_ndc_z = 2.0*bg_depth - 1.0;
  vec4 r0 = mvp_matrix*vec4(viewOrigin, 1.0);
  vec4 d = mvp_matrix*vec4(viewDir, 0.0);
  float t_bg = (bg_ndc_z * r0.w - r0.z)/(d.z - bg_ndc_z * d.w);

  // Calculate intersections between view ray (viewOrigin + t*viewDir) and AABB (cube) corresponding to the
  //  volume, in the volume's (model/object) texture space, where x,y,z all range from 0 to 1
  vec3 tbot = invR*(vec3(0.0) - viewOrigin);  // cube is defined by (0,0,0) and (1,1,1)
  vec3 ttop = invR*(vec3(1.0) - viewOrigin);
  vec3 tmin = min(ttop, tbot);
  vec3 tmax = max(ttop, tbot);
  // ray is inside box after crossing all three near planes
  float tnear = max(0.0, max(tmin.x, max(tmin.y, tmin.z)));
  // ray is outside box after crossing first far plane; or stop if background geom hit
  float tfar = min(t_bg, min(tmax.x, min(tmax.y, tmax.z)));

  vec4 final_color = vec4(0.0);
  vec3 final_pos;  // model space position for calculating depth
  if(tfar > tnear) {
    vec3 rayStart = viewOrigin + viewDir * tnear;
    vec3 rayStop = viewOrigin + viewDir * tfar;
    vec4 view_stop = mv_matrix*vec4(rayStop, 1.0);
    vec4 view_start = mv_matrix*vec4(rayStart, 1.0);
    int nsteps = clamp(int(length(view_stop.xyz - view_start.xyz)*ray_steps) + 1, 1, MAX_STEPS);
    vec3 step = (rayStop - rayStart)/nsteps;
    renderVol(rayStart, rayStop, step, nsteps, final_color, final_pos);
  }

  // We cannot discard fragment because we must write depth (if only to copy input value), so we must write
  //  color too - we assume that blend mode is such that alpha = 0 will discard color
  if(final_color.a == 0.0)
    gl_FragDepth = bg_depth;
  else {
    // calculate depth buffer value for final_pos; assumes default gl_DepthRange of [0,1]
    vec4 clip_space_pos = mvp_matrix*vec4(final_pos, 1.0);
    gl_FragDepth = 0.5*(1.0 + clip_space_pos.z/clip_space_pos.w);
  }
  gl_FragData[0] = final_color;
}
"""

# vispy's translucent approach - best for MRI image but not useful for MO
# - visvis's ray approach is identical with slightly clearer math
TRANSLUCENT_VOL_FRAG_FN = """
void renderVol(vec3 pos, vec3 rayStop, vec3 step, int nsteps, out vec4 final_color, out vec3 final_pos)
{
  float stepSize = length(step);
  final_color = vec4(0.0);
  for(int i = 0; i < nsteps && i < MAX_STEPS; ++i, pos += step) {
    float density = texture3D(texture_3d, pos).x;
    vec4 color = texture1D(colormap, density/isolevel);
    float a1 = final_color.a;
    float a2 = color.a * (1.0 - a1);
    float alpha = max(a1 + a2, 0.001);
    final_color = (final_color * (a1 / alpha)) + (color * (a2 / alpha));
    final_color.a = alpha;
  }
  final_pos = pos;
}
"""

# blending approach ... is there a good way to make this work with colormap?
BLENDING_VOL_FRAG_FN = """
void renderVol(vec3 pos, vec3 rayStop, vec3 step, int nsteps, out vec4 final_color, out vec3 final_pos)
{
  float stepSize = length(step);
  final_color = vec4(0.0);
  for(int i = 0; i < nsteps && i < MAX_STEPS; ++i, pos += step) {
    float density = texture3D(texture_3d, pos).x;
    // - this is dest over blending (front to back) - try reversing ray direction and using src over
    vec4 src = density > 0 ? color : color_n;
    src *= abs(density)*stepSize*absorption;
    final_color += clamp(1.0 - final_color.a, 0.0, 1.0) * src;
  }
  final_pos = pos;
}
"""


# blending approach with lighting:
# TODO: test this!
LIT_BLENDING_VOL_FRAG_FN = """
void renderVol(vec3 pos, vec3 rayStop, vec3 step, int nsteps, out vec4 final_color, out vec3 final_pos)
{
  float stepSize = length(step);
  final_color = vec4(0.0);
  for(int i = 0; i < nsteps && i < MAX_STEPS; ++i, pos += step) {
    float density = texture3D(texture_3d, pos).x;

    float T = 1.0;
    // take fixed size steps until pos is out of unit cube? or calculate intersection exactly?
    vec3 lpos = pos + light_dir;
    for(int j = 0; j < 32; ++j) {
      float ld = texture3D(texture_3d, lpos).x;
      T *= max(1.0 - abs(ld)*stepSize*absorption, 0.0);
      lpos += light_dir;
    }

    // - this is dest over blending (front to back) - try reversing ray direction and using src over
    vec4 src = density > 0 ? color : color_n;
    src *= abs(density)*stepSize*absorption;
    final_color += clamp(1.0 - final_color.a, 0.0, 1.0) * src;
  }
  final_pos = pos;
}
"""

# colormap indexed based on total absorption encounter by ray; so colormap should have incr alpha
# commented: alpha = absorption approach; crank up absorption to turn into maximum intensity projection (MIP)
ABSORPTION_VOL_FRAG_FN = """
void renderVol(vec3 pos, vec3 rayStop, vec3 step, int nsteps, out vec4 final_color, out vec3 final_pos)
{
  float stepSize = length(step);
  float T = 1.0;  // transmission through volume
  //float max_density = -1e6;
  //float min_density = 1e6;
  for(int i = 0; i < nsteps && i < MAX_STEPS; ++i, pos += step) {
    float density = texture3D(texture_3d, pos).x;
    // HEY! since we using multiplicative absorption, proper way to accomodate varying step size would be
    //  something like T *= pow(1.0 - abs(density)*absorption, stepSize)
    T *= max(1.0 - abs(density)*stepSize*absorption, 0.0);
    //max_density = max(density, max_density);
    //min_density = min(density, min_density);
  }
  final_pos = pos;
  final_color = texture1D(colormap, 1.0 - T);
  //final_color = texture1D(colormap, max_density/isolevel);
  //final_color.a = 1.0 - T;
}
"""

ISOSURFACE_FRAG_FN = """
// requires: shading()

// Better gradient calc: http://www.aravind.ca/cs788h_Final_Project/gradient_estimators.htm
// - no visible difference between one-sided and central difference
vec3 normal_calc(vec3 p, float u)
{
  // NOTE: magnitude of dstep affects how smooth isosurface appears
  vec3 dstep = 1.5/vol_shape;
  float dx = texture3D(texture_3d, p + vec3(dstep.x,0,0)).x - u;
  float dy = texture3D(texture_3d, p + vec3(0,dstep.y,0)).x - u;
  float dz = texture3D(texture_3d, p + vec3(0,0,dstep.z)).x - u;
  return vec3(dx, dy, dz);
}

void renderVol(vec3 pos, vec3 rayStop, vec3 step, int nsteps, out vec4 final_color, out vec3 final_pos)
{
  // trace transparent volumes back to front
  if(color.a < 1.0) {
    pos = rayStop;
    step = -step;
  }
  final_color = vec4(0.0);
  vec3 normal, last_pos, hit_pos;
  float sample;
  float last_sample = texture3D(texture_3d, pos).x;
  bool was_in_vol = last_sample > isolevel || last_sample < isolevel_n;
  vec4 curr_color = last_sample > isolevel ? color : color_n;  // assume we are in vol; won't be used if not
  pos += step;

  for(int i = 1; i < nsteps && i < MAX_STEPS; ++i, pos += step) {
    sample = texture3D(texture_3d, pos).x;
    bool in_vol = sample > isolevel || sample < isolevel_n;
    if(in_vol != was_in_vol) {
      if(in_vol)
        curr_color = sample > isolevel ? color : color_n;
      // linear interpolation to refine isosurface position
      float level = curr_color == color ? isolevel : isolevel_n;
      hit_pos = pos + step*(level - sample)/(sample - last_sample);

      // calculate isosurface normal
      vec3 obj_normal = normal_calc(hit_pos, texture3D(texture_3d, hit_pos).x);
      normal = normalize((mv_matrix * vec4(obj_normal, 0.0)).xyz);
      // view space position for lighting calc
      vec3 view_hit_pos = (mv_matrix * vec4(hit_pos, 1.0)).xyz;
      // reverse normal so lighting is calculated from surface facing user
      if(in_vol && color.a < 1.0)
        normal = -normal;
      if(curr_color == color)
        normal = -normal;
      vec4 new_color = vec4(shading(view_hit_pos, normal, curr_color.rgb), curr_color.a);

#ifdef ISOSURFACE_MESH
      // mesh lines
      obj_normal = normalize(obj_normal);
      if(any(lessThan( mod(hit_pos*20, 1), 0.1*sqrt(1.0 - obj_normal*obj_normal) ))) {
        final_color = mix(final_color, new_color, new_color.a);
        last_pos = hit_pos;
      }
#else
      final_color = mix(new_color, final_color, final_color.a);
      last_pos = hit_pos;
#endif
      if(final_color.a > 0.99)
        break;
      was_in_vol = in_vol;
    }
    last_sample = sample;
  }
  final_pos = last_pos;
}
"""

# row-major (C): right-most index varies the fastest, ie. a[0][0], a[0][1], ... in memory
# column-major (Fortran): left-most index varies fastest: ie. a[0][0], a[1][0], ... in memory
# 3D texture array should be in column-major format (or x and z dimensions must be swapped)

class VolumeObject:

  def __init__(self, colormap, isolevel=0.5, isolevel_n=-np.inf, colormap_interp='linear'):
    self.active = False
    self.stack_texture = None
    self.shape = (-1,-1,-1)
    self.vertices, self.indices = cube_triangles()

    self.absorption = 10
    self.isolevel = isolevel
    self.isolevel_n = isolevel_n
    # if colormap only has a single color, duplicate it
    self.colormap = np.tile(color, (2,1)) if len(colormap) == 1 else np.asarray(colormap)
    self.cmap_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_1D, self.cmap_texture)
    interp = GL_NEAREST if colormap_interp == 'nearest' else GL_LINEAR
    glTexParameter(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, interp)
    glTexParameter(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, interp)
    glTexParameter(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    # set texture data
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, len(self.colormap), 0, GL_RGBA, GL_FLOAT,
        self.colormap.astype(np.float32).flatten())
    # attribute(s) will be bound to VAO by VolumeRenderer
    self.vao = glGenVertexArrays(1)
    self.elem_vbo = VBO(self.indices, target=GL_ELEMENT_ARRAY_BUFFER)


  def set_data(self, stack, extents):
    self.active = True
    s = np.array(stack, dtype=np.float32, order='F')  # order='F' to get column-major order
    w, h, d = s.shape

    if not np.array_equal(s.shape, self.shape):
      if self.stack_texture is not None:
        glDeleteTextures([self.stack_texture])
      self.stack_texture = glGenTextures(1)
      glActiveTexture(GL_TEXTURE0)
      glBindTexture(GL_TEXTURE_3D, self.stack_texture)
      glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
      glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
      #glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # not needed as default alignment of 4 bytes always works for float data
      glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
      glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
      glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
      # set texture data - must use ctypes.data_as to preserve column-major order (I guess implicit conversion
      #  by passing s directly reverts to row-major ordering)
      # alternative is to pass s.transpose((2,1,0)) but that is 10x slower
      glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, w, h, d, 0, GL_RED, GL_FLOAT, s.ctypes.data_as(ctypes.c_void_p))
      self.shape = s.shape
    else:
      # reuse existing texture
      glActiveTexture(GL_TEXTURE0)
      glBindTexture(GL_TEXTURE_3D, self.stack_texture)
      glTexSubImage3D(
          GL_TEXTURE_3D,  # Target
          0,              # Level for mipmapping (aka LOD) - not used by us
          0, 0, 0,        # xoffset, yoffset, zoffset
          w, h, d,        # width, height, depth
          GL_RED,         # Format of the pixel data
          GL_FLOAT,       # Type of the pixel data
          s.ctypes.data_as(ctypes.c_void_p))  # Data

    tl = extents[1] - extents[0]
    tr = extents[0]
    # model space (unit cube w/ corner at origin) to world space transform
    self.transform = np.array(
        [[tl[0], 0.0, 0.0, tr[0]],
         [0.0, tl[1], 0.0, tr[1]],
         [0.0, 0.0, tl[2], tr[2]],
         [0.0, 0.0,   0.0,   1.0]])


class VolumeRenderer:
  """ Volumetric rendering with shaders """

  def __init__(self, method='volume', use_geom_depth=True):
    """ Allowed methods: 'blended', 'absorption', 'iso', or 'volume' (chooses 'blended') """
    self.volume_objects = []
    self.ray_steps = 1.0
    self.method = method
    self.use_geom_depth = use_geom_depth
    self.shader = None


  def gl_init(self):
    # setup shaders
    self.modules = [RendererConfig.header, RendererConfig.shading]
    vs = compileShader([m.vs_code() for m in self.modules] + [VERT_SHADER], GL_VERTEX_SHADER)
    fs_modules = [m.fs_code() for m in self.modules] + [FRAG_SHADER_MAIN]
    iso_fs = compileShader(fs_modules + [ISOSURFACE_FRAG_FN], GL_FRAGMENT_SHADER)
    fog_fs_frag = ABSORPTION_VOL_FRAG_FN if self.method == 'absorption' else BLENDING_VOL_FRAG_FN
    fog_fs = compileShader(fs_modules + [fog_fs_frag], GL_FRAGMENT_SHADER)
    self.iso_shader = compileProgram(vs, iso_fs)
    self.fog_shader = compileProgram(vs, fog_fs)
    self.shader = self.iso_shader if self.method == 'iso' else self.fog_shader
    if not self.use_geom_depth:
      d = np.array([1], dtype=np.float32)
      self.default_depth = create_texture(1, 1, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, d)


  def _render_volume_obj(self, vol_obj, viewer):
    depth_in = viewer.swap_depth_tex() if self.use_geom_depth else self.default_depth
    # cull front face (render back face) so that we can move camera into volume without clipping
    glEnable(GL_CULL_FACE)
    glCullFace(GL_FRONT)

    self.shader = self.iso_shader if self.method == 'iso' else self.fog_shader
    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)

    # compute (geometric) mean 1D sample density; previously ray_steps was hardcoded to max sample density, 20
    ray_steps = self.ray_steps*(np.prod(vol_obj.shape)/np.prod(np.diag(vol_obj.transform)[:3]))**(1/3.0)

    mv_matrix = np.dot(viewer.view_matrix(), vol_obj.transform)
    set_uniform(self.shader, 'mv_matrix', 'mat4fv', mv_matrix)
    set_uniform(self.shader, 'inv_mv_matrix', 'mat4fv', np.linalg.inv(mv_matrix))
    set_uniform(self.shader, 'mvp_matrix', 'mat4fv', np.dot(viewer.proj_matrix(), mv_matrix))
    set_uniform(self.shader, 'vol_shape', '3f', vol_obj.shape[::-1])
    set_uniform(self.shader, 'isolevel', '1f', vol_obj.isolevel)
    set_uniform(self.shader, 'isolevel_n', '1f', vol_obj.isolevel_n)
    set_uniform(self.shader, 'color', '4f', vol_obj.colormap[-1])
    set_uniform(self.shader, 'color_n', '4f', vol_obj.colormap[0])
    # number of ray marching steps
    set_uniform(self.shader, 'ray_steps', '1f', ray_steps)
    set_uniform(self.shader, 'viewport', '2f', [viewer.width, viewer.height])

    # bind volume texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_3D, vol_obj.stack_texture)
    set_uniform(self.shader, 'texture_3d', '1i', 0)

    # bind input depth texture
    glActiveTexture(GL_TEXTURE0+1)
    glBindTexture(GL_TEXTURE_2D, depth_in)
    set_uniform(self.shader, 'depth_tex', '1i', 1)

    if self.method != 'iso':
      # bind colormap texture
      glActiveTexture(GL_TEXTURE0+2)
      glBindTexture(GL_TEXTURE_1D, vol_obj.cmap_texture)
      set_uniform(self.shader, 'colormap', '1i', 2)
      set_uniform(self.shader, 'absorption', '1f', vol_obj.absorption)

    glBindVertexArray(vol_obj.vao)
    glDrawElements(GL_TRIANGLES, len(vol_obj.indices), GL_UNSIGNED_INT, None)
    # should we unbind textures? glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_3D, 0)
    glDisable(GL_CULL_FACE)  # return to default, disabled state
    glBindVertexArray(0)
    glUseProgram(0)


  def draw(self, viewer):
    for vol_obj in self.volume_objects:
      if vol_obj.active:
        self._render_volume_obj(vol_obj, viewer)


  def make_volume_obj(self, *args, **kwargs):
    if self.shader is None:
      self.gl_init()
    vol_obj = VolumeObject(*args, **kwargs)
    glBindVertexArray(vol_obj.vao)
    vol_obj._verts_vbo = bind_attrib(self.shader, 'texcoord', vol_obj.vertices, 3, GL_FLOAT)
    vol_obj.elem_vbo.bind()
    glBindVertexArray(0)
    self.volume_objects.append(vol_obj)
    return vol_obj
