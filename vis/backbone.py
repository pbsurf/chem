import numpy as np
from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO
from .glutils import *
from .shading import *
from .color import *
from .mol_renderer import StickRenderer, LineRenderer
from ..molecule import decode_atom_sel
from ..data.pdb_bonds import *


TRIANGLE_VERT_SHADER = """
uniform mat4 mv_matrix;
uniform mat4 p_matrix;

attribute vec3 position;
attribute vec3 normal_in;
attribute vec4 color_in;

varying vec4 pos_view;
varying vec3 normal;
varying vec4 color;
void main()
{
  color = color_in;
  normal = (mv_matrix*vec4(normal_in, 0.0)).xyz;
  pos_view = mv_matrix*vec4(position, 1.0);
  gl_Position = p_matrix*pos_view;
}
"""

TRIANGLE_FRAG_SHADER = """
varying vec4 pos_view;
varying vec3 normal;
varying vec4 color;

void main()
{
  // interpolated normal will not be normalized in general!
  gl_FragData[0] = vec4(shading(pos_view.xyz, normalize(normal), color.rgb), color.a);
  gl_FragData[1] = vec4(0.5*normal + 0.5, 1.0);
}
"""

# Generic geometry renderer
# we can move this and imposter renderers in mol_renderer.py to something like renderers.py
class TriangleRenderer:

  def __init__(self):
    self.n_indices = 0
    self.shader = None
    self.vao = None


  def set_data(self, vertices, normals, colors, indices):
    if self.shader is None:
      self.modules = [RendererConfig.header, RendererConfig.shading]
      vs = compileShader([m.vs_code() for m in self.modules] + [TRIANGLE_VERT_SHADER], GL_VERTEX_SHADER)
      fs = compileShader([m.fs_code() for m in self.modules] + [TRIANGLE_FRAG_SHADER], GL_FRAGMENT_SHADER)
      self.shader = compileProgram(vs, fs)

    vertices = np.asarray(vertices, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    colors = gl_colors(colors)  #np.asarray(colors, dtype=np.uint8)
    indices = np.asarray(indices, dtype=np.uint32)
    self.n_indices = np.size(indices)

    if self.vao is None:
      self.vao = glGenVertexArrays(1)
      glBindVertexArray(self.vao)
      self._verts_vbo = bind_attrib(self.shader, 'position', vertices, 3, GL_FLOAT)
      self._norms_vbo = bind_attrib(self.shader, 'normal_in', normals, 3, GL_FLOAT)
      self._color_vbo = bind_attrib(self.shader, 'color_in', colors, 4)  #, GL_UNSIGNED_BYTE, GL_TRUE)
      self._elem_vbo = VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)
      self._elem_vbo.bind()
      glBindVertexArray(0)
    else:
      # just update existing VBO for subsequent calls
      update_vbo(self._verts_vbo, vertices)
      update_vbo(self._norms_vbo, normals)
      update_vbo(self._color_vbo, colors)
      update_vbo(self._elem_vbo, indices)


  def draw(self, viewer):
    if self.n_indices < 1:
      return
    # GL state
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # uniforms
    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)
    set_uniform(self.shader, 'mv_matrix', 'mat4fv', viewer.view_matrix())
    set_uniform(self.shader, 'p_matrix', 'mat4fv', viewer.proj_matrix())

    # draw
    glBindVertexArray(self.vao)
    glDrawElements(GL_TRIANGLES, self.n_indices, GL_UNSIGNED_INT, None)

    glBindVertexArray(0)
    glUseProgram(0)
    glDisable(GL_CULL_FACE)


# "mol" should be object w/ attributes: r (vertices), normals, colors, indices
class VisTriangles:

  def __init__(self):
    self.renderer = TriangleRenderer()

  def __repr__(self):
    return "VisTriangles()"

  def draw(self, viewer, pass_num):
    if pass_num == 'opaque':  #'transparent'
      self.renderer.draw(viewer)

  def set_molecule(self, mol, r=None):
    vertices, normals, colors, indices = mol.r, mol.normals, mol.colors, mol.indices
    colors = np.repeat(colors, (np.size(vertices)/3)/(np.size(colors)/4), axis=0)
    self.renderer.set_data(vertices, normals, colors, indices)

  def on_key_press(self, viewer, keycode, key, mods):
    return False


class TubeMeshRenderer(TriangleRenderer):

  def __init__(self, aspect_ratio=0.125, radial_segments=20):
    TriangleRenderer.__init__(self)
    self.n_radial = radial_segments
    self.aspect_ratio = aspect_ratio


  def set_data(self, centers, axes, radii, orientations, colors, breaks):
    """ Set data for a tube mesh through `centers` with radius and orientation of major axis (assuming aspect
      ratio < 1) at each center given by `radii` and `orientations`, and color by `colors`.  Multiple tubes
      are separated by giving start of each tube in `breaks`, which should end with len(centers)
    """
    if len(centers) < 2:
      self.n_indices = 0
      return

    nr = self.n_radial
    centers = np.asarray(centers)
    axes = np.asarray(axes)
    # we accept axes as arg instead of calculating because of diff. between chain breaks and color step breaks
    #dc = centers[1:] - centers[:-1]
    #norm_axes = dc/(np.sqrt(np.sum(dc*dc, axis=-1))[:,None])
    #avg_axes = 0.5*(norm_axes[1:] + norm_axes[:-1])
    #axes = np.concatenate(([norm_axes[0]], avg_axes, [norm_axes[-1]]))

    radii = np.asarray(radii)
    aspect_ratio = (radii[:,1]/radii[:,0])[:,None,None] if np.ndim(radii) == 2 else self.aspect_ratio
    radii = radii = radii[:,0] if np.ndim(radii) == 2 else radii
    majors = np.asarray(orientations)
    minors = np.cross(majors, axes)
    minors = minors/(np.sqrt(np.sum(minors*minors, axis=-1))[:,None])
    ## TODO: need to concentrate points in regions w/ greatest curvature
    # don't include 2*pi since we close hoop with facet between last and first point
    angles = np.linspace(0, 2*np.pi, nr, endpoint=False)
    # equivalent to tiling angles by n_axial and repeating everything else by n_radial
    cosmajors = np.cos(angles)[:,None]*(radii[:,None]*majors)[:,None]
    sinminors = np.sin(angles)[:,None]*(radii[:,None]*minors)[:,None]
    vertices = centers[:,None] + cosmajors + aspect_ratio*sinminors
    # these are vertex normals, which will get interpolated across faces and give smooth shading
    normals = aspect_ratio*cosmajors + sinminors
    normals = normals/(np.sqrt(np.sum(normals*normals, axis=-1))[:,:,None])
    # colors
    rep_colors = np.repeat(colors, nr, axis=0)  #.astype(np.uint8)
    # indices to create triangle strip around each axial segment
    facet = np.array([0, nr, nr + 1, 0, nr + 1, 1])  # two triangles for quad for single facet
    hoop = facet + np.arange(nr)[:,None]  # note use of broadcasting
    hoop[-1] = np.array([nr - 1, 2*nr - 1, nr, nr - 1, nr, 0])
    chain_indices = nr*np.concatenate([np.arange(a, b-1) for a,b in zip(breaks, breaks[1:])])
    indices = hoop + chain_indices[:,None,None]  # note use of broadcasting

    TriangleRenderer.set_data(self, vertices, normals, rep_colors, indices)


LINE_RIBBON_VERT_SHADER = """
uniform mat4 mv_matrix;
uniform mat4 p_matrix;
uniform vec2 viewport;

attribute vec3 position;
attribute vec3 offset_dir;
attribute float ribbon_radius;
attribute float mapping;
attribute vec4 color_in;

varying vec4 color;
varying vec4 pos_view;
varying float v_mapping;
varying float ribbon_width_screen;

void main()
{
  color = color_in;
  v_mapping = mapping;
  vec3 dr = mapping*ribbon_radius*offset_dir;
  pos_view = mv_matrix*vec4(position + dr, 1.0);
  vec4 clip_pos_p = p_matrix*pos_view;
  vec4 clip_pos_n = p_matrix*mv_matrix*vec4(position - dr, 1.0);
  vec2 ribbon_dir_ndc = clip_pos_p.xy/clip_pos_p.w - clip_pos_n.xy/clip_pos_n.w;
  ribbon_width_screen = 0.5*length(ribbon_dir_ndc*viewport);
  gl_Position = clip_pos_p;
}
"""

LINE_RIBBON_FRAG_SHADER = """
varying vec4 color;
varying vec4 pos_view;
varying float v_mapping;
varying float ribbon_width_screen;

uniform float num_lines;
uniform float line_radius;

void main()
{
  /*// add contribution from every line to try to reduce aliasing - only a minor improvement
  float line_alpha = 0.0;
  float ribbon_mapping = 0.5*ribbon_width_screen*((v_mapping + 1.0) - 1.0/num_lines);
  for(int ii = 0; ii < int(num_lines); ++ii) {
    float a = clamp(line_radius - abs(ribbon_mapping), 0.0, 1.0);
    line_alpha = line_alpha + a - line_alpha*a;
    ribbon_mapping -= ribbon_width_screen/num_lines;
  }*/

  float line_alpha = 1.0;
  if(num_lines > 0) {
    float line_mapping = ribbon_width_screen*(fract(num_lines*0.5*(v_mapping + 1.0)) - 0.5)/num_lines;
    line_alpha = clamp(line_radius - abs(line_mapping), 0.0, 1.0);
  }
  // ensure edge of ribbon is antialiased at small scales
  float ribbon_radius = 0.5*ribbon_width_screen;
  float ribbon_alpha = clamp(ribbon_radius - abs(ribbon_radius*v_mapping), 0.0, 1.0);
  float alpha = min(line_alpha, ribbon_alpha);
  if(alpha < 0.05)
    discard;  // so we don't write depth for nearly transparent pixels
  else {
    vec3 final_color = shading_effects(pos_view.xyz, vec3(0,0,1), color.rgb);
    gl_FragData[0] = vec4(final_color, alpha*color.a);
  }
}
"""

# Aliasing is quite bad with this approach, but drawing separate lines with GL_LINES (as in uglymol) doesn't
#  completely eliminate aliasing; hopefully supersampling/postprocess-AA will help. Tried various tricks like
#  converting to solid ribbon below some width, but nothing worked too well.
# The flashing effect on foreground lines when panning happens in uglymol and xtal.js too
class LineRibbonRenderer:

  def __init__(self, n_lines=9, line_width=2.5):
    """ render a 2D ribbon with world space width `width`, split width-wise into `n_lines` lines of width
      `line_width` in pixels
    """
    self.n_lines = n_lines
    self.line_radius = 0.5*line_width
    self.shader = None
    self.vao = None


  #def set_data(self, bounds, radii, offset_dirs, colors):
  def set_data(self, centers, axes, radii, orientations, colors, breaks):
    """ Set/update geometry: passed pairs of points defining lines (bounds), line radii, and colors """
    if self.shader is None:
      self.modules = [RendererConfig.header, RendererConfig.shading]
      vs = compileShader([m.vs_code() for m in self.modules] + [LINE_RIBBON_VERT_SHADER], GL_VERTEX_SHADER)
      fs = compileShader([m.fs_code() for m in self.modules] + [LINE_RIBBON_FRAG_SHADER], GL_FRAGMENT_SHADER)
      self.shader = compileProgram(vs, fs)

    if len(centers) < 2:
      self.n_indices = 0
      return

    mapping = np.tile([-1.0, 1.0], len(centers)).astype(np.float32)
    vertices = np.repeat(centers, 2, axis=0).astype(np.float32)
    prim_offsets = np.repeat(orientations, 2, axis=0).astype(np.float32)
    prim_radii = np.repeat(radii, 2, axis=0).astype(np.float32)
    prim_colors = gl_colors(np.repeat(colors, 2, axis=0))  #.astype(np.uint8)
    # indices for triangle strip for each chain
    chain_indices = 2*np.concatenate([np.arange(a, b-1) for a,b in zip(breaks, breaks[1:])])
    indices = np.array([0,1,2,2,1,3]) + chain_indices[:,None,None]  # note use of broadcasting
    self.n_indices = np.size(indices)
    #import pdb; pdb.set_trace()

    if self.vao is None:
      self.vao = glGenVertexArrays(1)
      glBindVertexArray(self.vao)
      self._verts_vbo = bind_attrib(self.shader, 'position', vertices, 3, GL_FLOAT)
      self._local_vbo = bind_attrib(self.shader, 'mapping', mapping, 1, GL_FLOAT)
      self._radii_vbo = bind_attrib(self.shader, 'ribbon_radius', prim_radii, 1, GL_FLOAT)
      self._offsets_vbo = bind_attrib(self.shader, 'offset_dir', prim_offsets, 3, GL_FLOAT)
      self._color_vbo = bind_attrib(self.shader, 'color_in', prim_colors, 4)  #, GL_UNSIGNED_BYTE, GL_TRUE)
      self._elem_vbo = VBO(indices.astype(np.uint32), target=GL_ELEMENT_ARRAY_BUFFER)
      self._elem_vbo.bind()
      glBindVertexArray(0)
    else:
      # just update existing VBO for subsequent calls
      update_vbo(self._verts_vbo, vertices)
      update_vbo(self._local_vbo, mapping)
      update_vbo(self._radii_vbo, prim_radii)
      update_vbo(self._offsets_vbo, prim_offsets)
      update_vbo(self._color_vbo, prim_colors)
      update_vbo(self._elem_vbo, indices.astype(np.uint32))


  def draw(self, viewer):
    if self.n_indices < 1:
      return
    # We do not want GL_CULL_FACE as we only have a single square
    # uniforms
    glUseProgram(self.shader)
    for m in self.modules:
      m.setup_shader(self.shader, viewer)
    set_uniform(self.shader, 'mv_matrix', 'mat4fv', viewer.view_matrix())
    set_uniform(self.shader, 'p_matrix', 'mat4fv', viewer.proj_matrix())
    set_uniform(self.shader, 'viewport', '2f', [viewer.width, viewer.height])
    set_uniform(self.shader, 'num_lines', '1f', self.n_lines)
    set_uniform(self.shader, 'line_radius', '1f', self.line_radius) # line radius in pixels

    # draw
    glBindVertexArray(self.vao)
    glDrawElements(GL_TRIANGLES, self.n_indices, GL_UNSIGNED_INT, None)

    glBindVertexArray(0)
    glUseProgram(0)


# peptide backbone is N,CA,C,...; often we render trace through just CA, as trace including N and C is wobbly
# See https://github.com/boscoh/pyball and https://github.com/boscoh/pdbremix for cartoon representation,
#  incl. simple determination of secondary structure
# Arguably disulfide bonds should be rendered with a separate renderer, but it seems like they will very
#  often want to be rendered whenever backbone is rendered
# - Any way to use just `sel` and `sort` fns instead of bbatoms and orientatoms?
# - Consider removing the 'tube' style option
class VisBackbone:
  """ class for visualizing backbone of protein (or other polymer) """
  def __init__(self, style='tubemesh', sel='polymer', atoms=None, orientatoms=None, disulfides='line',
      untwist=True, aspect_ratio=0.125, max_degrees=5,
      coloring=color_by_resnum, colors=None, color_interp='ramp', radius=None, radius_fn=None):
    """ init trace of type `style` ('tubemesh', 'tube', or 'ribbon'), with `radius` and `coloring`, passing
      through `atoms` of each residue (autodetection for protein, DNA, and RNA chains; default ['CA'] for
      protein).  `atoms` should be in order (e.g., N,CA,C for polypeptide).  If `untwist` is true, residue
      orientation will be reversed if >90 deg from previous residue (common in beta sheets).  Disulfide bonds
      will be drawn with style specified by `disulfides` ('line', 'tube', or None)
    """
    self.bbatoms = atoms
    self.orientatoms = orientatoms
    self.radius = (0.2 if style == 'tube' else 0.5) if radius is None else radius
    if radius_fn == 'b_factor':
      self.radius_fn = lambda mol, idx, r: r * getattr(mol.atoms[idx], 'b_factor', 10.0)/10.0
    else:
      self.radius_fn = radius_fn if callable(radius_fn) else lambda mol, idx, r: r
    self.aspect_ratio = aspect_ratio
    self.untwist = untwist
    self.coloring = color_by(coloring, colors)
    # color interpolation along trace
    self.color_interp = color_interp
    if color_interp == 'step':
      self.color_fn = lambda c0, c1, t: c0 if t <= 0.5 else c1
    else:  # 'ramp'
      self.color_fn = lambda c0, c1, t: [(1.0 - t)*cc0 + t*cc1 for cc0, cc1 in zip(c0, c1)]
    # parameters for converting trace to straight segments
    self.max_cosine_axis = math.cos(math.pi*max_degrees/180.0)
    self.max_cosine_orient = self.max_cosine_axis
    self.min_dt = 0.1
    self.sel_fn = decode_atom_sel(sel)
    # renderer
    self.style = style
    if style == 'tubemesh':
      self.base_renderer = TubeMeshRenderer(aspect_ratio=aspect_ratio)
    elif style == 'tube':
      self.base_renderer = StickRenderer()
    elif style == 'ribbon':
      self.base_renderer = LineRibbonRenderer()
    else:
      raise Exception("Invalid backbone style: " + style)
    self.disulfides = disulfides
    if disulfides == 'tube':
      self.disulfide_renderer = StickRenderer()
    elif disulfides: # make 'line' the default if disulfides == True
      self.disulfide_renderer = LineRenderer()

  def __repr__(self):
    return "VisBackbone(style='%s', atoms='%s')" % (self.style, self.bbatoms)

  def on_key_press(self, viewer, keycode, key, mods):
    return False

  def draw(self, viewer, pass_num):
    if pass_num == ('transparent' if self.style == 'ribbon' else 'opaque'):
      self.base_renderer.draw(viewer)
    if self.disulfides and pass_num == ('opaque' if self.disulfides == 'tube' else 'transparent'):
      self.disulfide_renderer.draw(viewer)

  def set_molecule(self, mol, r=None):
    r = mol.r if r is None else r
    # extract backbone trace
    trace, orientation, ds_bounds, chain_breaks = [], [], [], []
    bbatoms, orientatoms = self.bbatoms, self.orientatoms
    prev_chain = False  # can't use None since residue.chain will be None if chain is not set
    for resnum, residue in enumerate(mol.residues):
      # attempt to detect chain type based on first residue
      # ref: https://github.com/arose/ngl/blob/master/src/structure/structure-constants.js
      if not self.bbatoms and residue.chain != prev_chain:
        if residue.name in PDB_PROTEIN:
          bbatoms, orientatoms = ['CA'], ('C', 'O')
        elif residue.name in PDB_DNA:
          bbatoms, orientatoms = ["C3'"], ("C2'", "O4'")
        elif residue.name in PDB_RNA:
          bbatoms, orientatoms = ["C4'"], ("C1'", "C3'")
        else:
          continue  # try next residue
      # build table of atom names for residue
      name_to_idx = dict( (mol.atoms[jj].name, jj) for jj in residue.atoms )
      if all(a in name_to_idx and self.sel_fn(mol.atoms[name_to_idx[a]], mol, name_to_idx[a]) for a in bbatoms):
        if residue.chain != prev_chain:
          chain_breaks.append(len(trace))
        prev_chain = residue.chain
        trace.extend([name_to_idx[a] for a in bbatoms])
        dir = normalize(r[name_to_idx[orientatoms[0]]] - r[name_to_idx[orientatoms[1]]])
        # reverse orientation vector if > 90 degree change from previous residue (if `untwist`)
        if self.untwist and orientation and np.dot(orientation[-1], dir) < 0:
          dir = -dir
        orientation.extend([dir]*len(bbatoms))
        # nest this here since we only want to draw disulfide bonds if both residues are selected
        if self.disulfides and residue.name == 'CYS':
          # many possible ways for things to go wrong, so just use try/except
          try:
            sg2 = next(jj for jj in mol.atoms[name_to_idx['SG']].mmconnect if mol.atoms[jj].name == 'SG')
            cys2_atoms = mol.residues[mol.atoms[sg2].resnum].atoms
            ca2 = next(jj for jj in cys2_atoms if mol.atoms[jj].name == 'CA')
            # resnum check is to prevent double counting
            if mol.atoms[sg2].resnum > resnum and self.sel_fn(mol.atoms[ca2], mol, ca2):
              ds_bounds.append([r[name_to_idx['CA']], r[ca2]])
          except: pass
      else:
        prev_chain = False

    chain_breaks.append(len(trace))
    color = self.coloring(mol, trace)
    radii = np.array([self.radius_fn(mol, ii, self.radius) for ii in trace])
    if self.style == 'tube':
      self.set_tube_data(r[trace], radii, color)
    else:
      self.set_ribbon_data(r[trace], radii, orientation, color, chain_breaks)

    if self.disulfides:
      # TODO: we obviously need to make radii and colors configurable
      ds_radii = [1.75 if self.disulfides == 'line' else 0.2]*len(ds_bounds)
      ds_colors = [CPK_COLORS[16]]*len(ds_bounds)
      self.disulfide_renderer.set_data(ds_bounds, ds_radii, ds_colors)


  def set_ribbon_data(self, r_array, radii, orient, colors, chain_breaks):
    rib_bounds, rib_axes, rib_radii, rib_dirs, rib_colors, rib_breaks = [], [], [], [], [], [0]
    chain_start = chain_breaks[0]
    for chain_end in chain_breaks[1:]:
      rib_bounds.append(r_array[chain_start])
      rib_radii.append(radii[chain_start])
      rib_dirs.append(orient[chain_start])
      rib_colors.append(colors[chain_start])
      r1 = r_array[chain_start]
      axis = None
      prev_axis = np.zeros(3)

      for ii in range(chain_start, chain_end - 1):
        t_start, t_end = 0,0
        while t_end < 1:
          r0 = r1
          # ensure at least two segments between each point (esp. for step color)
          t_start, t_end = t_end, (0.5 if t_end < 0.5 else 1.0)
          while True:
            iim1, iip2 = max(chain_start, ii - 1), min(chain_end - 1, ii + 2)
            r1 = cardinal_spline(t_end, r_array[iim1], r_array[ii], r_array[ii+1], r_array[iip2])
            axis = normalize(r1 - r0)
            rib_dir = normalize_or_null((1.0 - t_end)*orient[ii] + t_end*orient[ii+1])
            if t_end - t_start < self.min_dt or (np.dot(axis, prev_axis) > self.max_cosine_axis
                and np.dot(rib_dir, rib_dirs[-1]) > self.max_cosine_orient):
              break
            t_end = (t_start + t_end)/2.0

          # duplicate previous point with other color
          if self.color_interp == 'step' and t_start == 0.5:
            # adding break is not necessary since we supply axes; saw some small gaps w/ break ... why!?
            #rib_breaks.append(len(rib_bounds))
            rib_bounds.append(rib_bounds[-1])
            rib_axes.append(rib_axes[-1])
            rib_dirs.append(rib_dirs[-1])
            rib_radii.append(rib_radii[-1])
            rib_colors.append(colors[ii+1])
          # next segment; note that rib_axis is one step behind
          rib_bounds.append(r1)
          rib_axes.append(normalize(prev_axis + axis))
          prev_axis = axis
          # just linear interpolation of direction and radius
          rib_dirs.append(rib_dir)
          rib_radii.append((1.0 - t_end)*radii[ii] + t_end*radii[ii+1])
          rib_colors.append(self.color_fn(colors[ii], colors[ii+1], t_end))

      rib_axes.append(axis)
      rib_breaks.append(len(rib_bounds))
      chain_start = chain_end

    self.base_renderer.set_data(rib_bounds, rib_axes, rib_radii, rib_dirs, rib_colors, rib_breaks)


  def set_tube_data(self, r_array, radii, colors):
    cyl_bounds, cyl_radii, cyl_colors = [],[],[]
    r1 = r_array[0]
    dir = None
    prev_dir = normalize(r_array[1] - r_array[0])
    r_array = np.concatenate( ([r_array[0]], r_array, [r_array[-1]]) )
    for ii in range(len(r_array)-3):
      t_start, t_end = 0,0
      while t_end < 1:
        r0 = r1
        # ensure at least two segments between each point (esp. for step color)
        t_start, t_end = t_end, (0.5 if t_end < 0.5 else 1.0)
        while True:
          r1 = cardinal_spline(t_end, r_array[ii], r_array[ii+1], r_array[ii+2], r_array[ii+3])
          dir = normalize(r1 - r0)
          if t_end - t_start < self.min_dt or np.dot(dir, prev_dir) > self.max_cosine_axis:
            break
          t_end = (t_start + t_end)/2.0

        radius_at_gap = (1-t_start)*radii[ii] + t_start*radii[ii+1]
        h = np.arccos(np.dot(dir, prev_dir)) * radius_at_gap  # assumes small angle
        # extend end of prev cylinder and start of new cylinder to close gap at joint
        if cyl_bounds:
          cyl_bounds[-1][-1] = cyl_bounds[-1][-1] + 0.5*h*prev_dir
          r0 = r0 - 0.5*h*dir
        # alternative approach: extra cylinder to fill gap; either approach produces minor artifacts, but
        #  better than using spheres at joints
        #if h > 0:
        #  gap_dir = normalize(dir + prev_dir)
        #  cyl_bounds.append([r0 - 0.5*h*gap_dir, r0 + 0.5*h*gap_dir])
        #  cyl_radii.append(radius_at_gap)
        #  cyl_colors.append(self.color_interp(colors[ii], colors[ii+1], t_start))
        prev_dir = dir
        # next cylinder
        cyl_bounds.append([r0, r1])
        # get color and radius at *center* of cylinder
        t = (t_start + t_end)/2.0
        cyl_radii.append((1-t)*radii[ii] + t*radii[ii+1])
        cyl_colors.append(self.color_fn(colors[ii], colors[ii+1], t))

    self.base_renderer.set_data(np.array(cyl_bounds), cyl_radii, cyl_colors)
