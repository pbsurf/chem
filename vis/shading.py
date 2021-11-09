import numpy as np
from .glutils import *


# Shadow mapping:
# refs:
# - http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
# - http://www.sunandblackcat.com/tipFullView.php?l=eng&topicid=35
# - https://en.wikipedia.org/wiki/Shadow_mapping - many links
# - Eisemann E., et. al. Real-Time Shadows - libgen.io
# Notes:
# - support for positional lights (perspective projection instead of orthographic)
# - try PCSS for fancy soft-shadows

OLD_VALUE_NOISE = """
#define TWO_PI 6.28318530718

// 3D value noise - see https://www.shadertoy.com/view/4sfGzS for ref and other types of noise
// - notice that the basic idea is to assign random values to grid points (floor(x) + (0/1, 0/1, 0/1)) and
//  then interpolate between them
float hash(vec3 p)
{
  p = 17.0*fract( p*0.3183099 + .1 );
  return fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
}

//vec3 hash( vec3 p ) // replace this by something better
//{
//	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
//			  dot(p,vec3(269.5,183.3,246.1)),
//			  dot(p,vec3(113.5,271.9,124.6)));
//
//	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
//}

float noise(vec3 x)
{
  vec3 p = floor(x);
  vec3 f = fract(x);
  f = f*f*(3.0-2.0*f);

  return mix(mix(mix( hash(p+vec3(0,0,0)), hash(p+vec3(1,0,0)),f.x),
                 mix( hash(p+vec3(0,1,0)), hash(p+vec3(1,1,0)),f.x),f.y),
             mix(mix( hash(p+vec3(0,0,1)), hash(p+vec3(1,0,1)),f.x),
                 mix( hash(p+vec3(0,1,1)), hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

float fbm(vec3 pos)
{
  const mat3 m = mat3( 0.00,  0.80,  0.60,
                      -0.80,  0.36, -0.48,
                      -0.60, -0.48,  0.64 );
  vec3 q = 2.0*pos;
  float f = 0.5000*noise( q ); q = m*q*2.01;
  f += 0.2500*noise( q ); q = m*q*2.02;
  f += 0.1250*noise( q ); q = m*q*2.03;
  f += 0.0625*noise( q );
  return f;  // sqrt(1.2*f);
}

uniform mat4 inv_view_matrix;
  // in shadow():

  // random rotation of sampling points ... produces pixel-scale noise, doesn't look good!
  //vec4 world_pos = inv_view_matrix*vec4(pos, 1.0);
  //float angle = TWO_PI*fract(sin(mod(dot(world_pos.xyz, vec3(12.9898, 78.233, 50.351)), TWO_PI))*43758.5453);
  //ivec2 seed = ivec2(floor(1000*(world_pos.xy + 2*worldpos.z)));
  //float angle = mod( (((3 * seed.x) ^ (seed.y + seed.x * seed.y))) * 10, TWO_PI );
  //
  //float f = fbm(world_pos.xyz);
  //
  //float angle = TWO_PI*f;
  //float c = cos(angle);
  //float s = sin(angle);
  //mat2 rot = mat2(c, -s, s, c);
"""

SHD_FRAG_SHADER = """
uniform int shading_type;
uniform vec3 light_dir;
uniform float ambient;
uniform vec4 fog_color;
uniform float fog_start;
uniform float fog_density;
uniform float outline_strength;
uniform vec4 outline_color;

#ifdef NO_SHADOW
float shadow(vec3 pos) { return 1.0; }
#else
// shadow sampler samples from 4 closest pixels and does depth comparison for each
uniform sampler2DShadow shadow_tex;
uniform float shadow_strength;
uniform mat4 view_to_shadow_matrix;
uniform vec2 inv_viewport;

const vec2 poissonDisk[16] = vec2[](
   vec2( -0.94201624, -0.39906216 ),
   vec2( 0.94558609, -0.76890725 ),
   vec2( -0.094184101, -0.92938870 ),
   vec2( 0.34495938, 0.29387760 ),
   vec2( -0.91588581, 0.45771432 ),
   vec2( -0.81544232, -0.87912464 ),
   vec2( -0.38277543, 0.27676845 ),
   vec2( 0.97484398, 0.75648379 ),
   vec2( 0.44323325, -0.97511554 ),
   vec2( 0.53742981, -0.47373420 ),
   vec2( -0.26496911, -0.41893023 ),
   vec2( 0.79197514, 0.19090188 ),
   vec2( -0.24188840, 0.99706507 ),
   vec2( -0.81409955, 0.91437590 ),
   vec2( 0.19984126, 0.78641367 ),
   vec2( 0.14383161, -0.14100790 )
);

float shadow(vec3 pos)
{
  if(shadow_strength <= 0.0)
    return 1.0;

  vec4 clip_shadow = view_to_shadow_matrix*vec4(pos, 1.0);
  // note that the NDC to texcoord transform, 0.5*ndc + 0.5, could be rolled into the view_to_shadow_matrix
  vec3 texcoord_shadow = 0.5*clip_shadow.xyz/clip_shadow.w + 0.5;

  float illum = 0.0;
  //if(texture2D(shadow_tex, texcoord_shadow.xy).x < texcoord_shadow.z - 0.005)
  //  illum = 0.5;
  //illum = 0.5 + 0.5*shadow2D(shadow_tex, vec3(texcoord_shadow.xy, texcoord_shadow.z - 0.005)).x;

  // bias can be a fn of curvature to better handle shadow acne; note acos(diffuse) = angle(light_dir, normal)
  // another approach for bias is including a +z translation in shadow view space in view_to_shadow_matrix
  //float bias = clamp(0.005*tan(acos(diffuse)), 0, 0.01);
  const float bias = 0.005;
  const int shadow_samples = 4;
  for (int ii = 0; ii < shadow_samples; ii++) {
    vec2 uv = texcoord_shadow.xy + 2.0*inv_viewport*poissonDisk[ii];
    illum += shadow2D(shadow_tex, vec3(uv, texcoord_shadow.z - bias)).x;
  }
  illum = (1.0 - shadow_strength) + shadow_strength*(illum/shadow_samples);
  return illum;
}
#endif // !NO_SHADOW

vec3 blinn_phong(vec3 pos, vec3 normal, vec3 color)
{
  vec3 view_dir = normalize(-pos);  // vector from point to camera (i.e. origin) in camera space
  vec3 halfvector = normalize(light_dir + view_dir);
  float NdotHV = max(dot(halfvector, normal), 0.0);
  float specular = 0.3*pow(NdotHV, 110.0);
  float diffuse = max(dot(light_dir, normal), 0.0);
  float illum = shadow(pos);
  return color * (ambient + diffuse*illum) + specular*illum;
}

// antialiasing color boundaries: http://prideout.net/blog/?p=22
vec3 toon_shading(vec3 pos, vec3 normal, vec3 color)
{
  float NdotL = dot(light_dir, normal);
  vec3 ret;

  if(NdotL <= 0.4)
    ret = vec3(0.0);
  else if(NdotL <= 0.6)
    ret = 0.75 * color;
  else
    ret = color;

  float illum = shadow(pos);
  return ret*illum;
}

vec3 shading_fog(vec3 pos, vec3 color)
{
  // fog (depth cueing); fixed fn OpenGL provided linear, exponential, and e^2 fog
  float fog_dist = -pos.z - fog_start;  // to match fixed fn OpenGL, set fog_start = 0 for exponential fog
  //float fog_dist = distance(pos, pivot) - fog_start; if(fog < 0.01) discard;  // pivot=[0,0,dist(pos,pivot)]
  // standard exponential fog
  float fog = exp(-fog_dist * fog_density);

  // linear fog
  // fog_end corresponds to the value where exponential fog factor ~ 1/256
  //float fog_end = fog_start + 5.5/fog_density;
  //float fog = (fog_end - fog_dist)/(fog_end - fog_start);

  // e^2 fog
  //float fog = exp(-pow(fog_dist * fog_density, 2.0));

  return mix(fog_color.rgb, color, clamp(fog, 0.0, 1.0));
}

float shading_outline(vec3 pos, vec3 normal)
{
  vec3 view_dir = normalize(-pos);  // direction to camera
  float edge = dot(view_dir, normal);  // 1 for surface facing camera, 0 for perpendicular surface
  return clamp(outline_strength*(0.5 - edge), 0.0, 1.0);  // 0 for no outline, 1 for full outlinr
}

vec3 shading_effects(vec3 pos, vec3 normal, vec3 color)
{
  // effects
  if(outline_strength > 0.0)
    color = mix(color, outline_color.rgb, shading_outline(pos, normal));
  if(fog_density > 0.0)
    color = shading_fog(pos, color);
  return color;
}

// inputs: pos and normal should be in camera space
vec3 shading(vec3 pos, vec3 normal, vec3 color_in)
{
  vec3 color;
  if(shading_type == 0)
    color = blinn_phong(pos, normal, color_in);
  else if(shading_type == 1)
    color = toon_shading(pos, normal, color_in);
  else if(shading_type == 2)
    color = color_in;

  color = shading_effects(pos, normal, color);
  return color;
}
    """

class LightingShaderModule:
  def __init__(self, light_dir=[0.0, 0.0, 1.0], shading='phong', ambient=0.05, shadow_strength=0.0,
      fog_density=0.0, fog_start=0.0, fog_color=(0,0,0,255), fog_start_pivot=True,
      outline_strength=0.0, outline_color=(0,0,0,255)):
    """ light_dir should be direction in view_space - so a fixed direction will fix light relative to camera
     if `fog_start_pivot`, fog_start is fixed relative to camera pivot, not camera position
    """
    self.light_dir = light_dir
    self.ambient = ambient
    self.fog_density = fog_density
    self.fog_start = fog_start
    self.fog_color = fog_color
    self.fog_start_pivot = fog_start_pivot
    self.outline_strength = outline_strength
    self.outline_color = outline_color
    self.shadow_strength = shadow_strength
    self.shading_type = shading

  def vs_code(self): return ""

  # shader seems to break on some platforms if shadow2D() is used w/o an actual texture (not just 0) bound
  def fs_code(self): return ("" if self.shadow_strength > 0 else "#define NO_SHADOW 1") + SHD_FRAG_SHADER

  def setup_shader(self, shader, viewer):
    """ assumes glUseProgram() has been called with a program containing our shader code """
    fog_start = self.fog_start
    if self.fog_start_pivot:
      fog_start = self.fog_start + norm(viewer.camera.position - viewer.camera.pivot)

    # we'll support runtime switching of shading method as it's not uncommon to provide a UI control for it
    shd = {'phong': 0, 'toon': 1, 'none': 2}['none' if viewer.curr_pass =='shadow' else self.shading_type]
    set_uniform(shader, 'shading_type', '1i', shd)

    set_uniform(shader, 'light_dir', '3f', normalize(self.light_dir))
    set_uniform(shader, 'ambient', '1f', self.ambient)
    set_uniform(shader, 'fog_color', '4f', gl_color(self.fog_color))
    set_uniform(shader, 'fog_start', '1f', fog_start)
    set_uniform(shader, 'fog_density', '1f', self.fog_density)
    set_uniform(shader, 'outline_strength', '1f', self.outline_strength)
    set_uniform(shader, 'outline_color', '4f', gl_color(self.outline_color))
    if self.shadow_strength > 0:
      set_uniform(shader, 'shadow_strength', '1f', self.shadow_strength)
      set_uniform(shader, 'shadow_tex', '1i', viewer.shadow_tex_id)
      #set_uniform(shader, 'inv_view_matrix', 'mat4fv', np.linalg.inv(viewer.view_matrix()))
      set_uniform(shader, 'view_to_shadow_matrix', 'mat4fv', np.dot(viewer.shadow_proj_matrix(),
          np.dot(viewer.shadow_view_matrix(), np.linalg.inv(viewer.view_matrix()))))
      set_uniform(shader, 'inv_viewport', '2f', [1.0/viewer.width, 1.0/viewer.height])


  def rotate_light(self, dx, dy):
    f = 1.5
    rot = rotation_matrix(f*dx, [0.0, 1.0, 0.0])
    rot = np.dot(rotation_matrix(f*dy, [1.0, 0.0, 0.0]), rot)
    self.light_dir = np.dot(rot[:3,:3], self.light_dir)


  def on_key_press(self, viewer, keycode, key, mods):
    if key in 'FG':
      if 'Alt' in mods:
        self.fog_start_pivot = not self.fog_start_pivot
        print("Fog start relative to %s" % ('pivot' if self.fog_start_pivot else 'camera'))
      elif 'Ctrl' in mods:
        # decr fog_start with 'G' so that C and Ctrl+G both reduce visibility
        s = (1 if key == 'F' else -1) * (0.1 if 'Shift' in mods else 0.01)
        self.fog_start += s*(viewer.camera.z_far - viewer.camera.z_near)
        print("Fog start: %.2f" % self.fog_start)
      else:
        if self.fog_density < 0.001:
          self.fog_density = 0.001 if key == 'G' else 0.0
        s = np.power((1.25 if key == 'G' else 0.8), (10 if 'Shift' in mods else 1))
        self.fog_density *= s
        print("Fog density: %.3f" % self.fog_density)
    elif key in 'KL':
      s = (1 if key == 'L' else -1) * (1 if 'Shift' in mods else 0.2)
      self.outline_strength += s
      print("Outline strength: %0.3f" % self.outline_strength)
    else:
      return False
    return True
