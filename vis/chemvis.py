import numpy as np
import sys, os, time, glob, threading, logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  #logging.DEBUG

from ..data.elements import ELEMENTS
from ..molecule import *
from ..io import load_molecule
from ..model.build import add_hydrogens
from .viewer import GLFWViewer
from ..external.glfw import GLFW_KEY_BACKSPACE, GLFW_KEY_F1, GLFW_KEY_HOME
from .camera import Camera
from .glutils import *
from .color import *
from .shading import LightingShaderModule
from .mol_renderer import VisGeom, LineRenderer, BallRenderer, StickRenderer
from .backbone import VisBackbone, VisTriangles
from .nonbonded import *
from .postprocess import GammaCorrEffect, AOEffect, OutlineEffect, DOFEffect, FogEffect, PostprocessHost

## Chemvis - top level class
# Notes and example moved to ../test/vis_test.py

class Chemvis:

  def __init__(self, children, effects=[], shading='phong', shadows=False, fog=False, bg_color=Color.black,
      wrap=False, timing=False, threaded=True, step_singles=False, drag_atoms=False, lock_orientation=False,
      refresh_sel=False, verbose=False):
    """ create Chemvis instance, with list of Mol objects in `children`
      Options:
      - `wrap`: wrap to beginning of file lists when stepping
      - `bg_color`: window background color,
      - `timing`: print frame render times
      - `threaded`: run viewer on separate thread so python prompt is available (disable for pdb debugging)
      - `step_singles`: include `Mol`s with single file (or r) for stepping (so stepping will be prevented
        unless they are hidden or unfocused)
      - `lock_orientation`: keep "north-south" axis vertical when orbiting camera
      - `refresh_sel`: update selection to follow atom positions
    """
    self.children = children if type(children) is list else [children]
    self.wrap = wrap
    self.print_timing = timing
    self.threaded = threaded
    self.step_singles = step_singles
    self.drag_atoms = drag_atoms
    self.refresh_sel = refresh_sel
    self.verbose = verbose
    self.animation_period = 500 # ms
    self.animating = False
    self.n_mols = 0
    self.mol_number = 0
    bg_color = decode_color(bg_color)
    # camera and viewer
    self.camera = Camera(lock_orientation=lock_orientation)
    self.initial_view = None
    self.viewer = GLFWViewer(self.camera, bg_color=bg_color)
    self.viewer.user_on_key = self.on_key_press
    self.viewer.user_on_click = self.on_click
    self.viewer.user_on_drag = self.on_drag
    self.viewer.user_draw = self.draw
    self.viewer.user_finish = self.on_finish

    # lighting config shared by geometry renderers
    if type(shading) is str:
      RendererConfig.shading = LightingShaderModule(light_dir=[0.0, 0.0, 1.0], shading=shading,
          shadow_strength=(0.8 if shadows else 0), fog_color=bg_color, fog_density=0.1 if fog else 0.0)
    else:
      RendererConfig.shading = shading
    self.shadows = shadows
    # atom selection
    self.selection_renderer = BallRenderer()
    self.selection = []
    self.sel_color = (0, 255, 0, 128)
    # wrap postprocess effects that are modules with PostprocessHost; this should probably be moved elsewhere
    self.effects = []
    modules = []
    for effect in effects:
      if hasattr(effect, 'draw') and callable(effect.draw):
        if modules:
          self.effects.append(PostprocessHost(modules))
          modules = []
        self.effects.append(effect)
      else:
        modules.append(effect)
    if modules:  # or not effects:  # we can use blit_framebuffer now if no effects
      self.effects.append(PostprocessHost(modules))


  def help(self):
    print(
""" F1: print help
 Mouse drag: left button to orbit camera around origin, Shift+LMB to rotate about view axis (Z), right button to translate; Alt+LMB to orbit light
 Mouse click: select atom; Shift+click to center view on atom, Ctrl+click for multi-select
WS: translate camera (z); Ctrl+WS to zoom in/out
AD: translate camera (x); Ctrl+AD to change field of view
 0: reset camera position; Shift+0 to save current position as default; Ctrl+0 to autozoom current molecule
,.: next/prev molecule
 P: start/stop animation
IO: slower/faster animation
 `: list renderers
 1-9: toggle visibility of individual renderers
 Shift+(1-9): toggle focus of molecules of individual renderers
""")
    for child in self.children:
      if child.focused and hasattr(child, 'help'):
        child.help()


  def select(self, sel=None, pdb=None, mol=0, color=None, opacity=0.5, clear=True, refresh=False, print_dim=False, radius=0.3):
    """ programmatically select atoms based on atom selection string or function `sel` from `mol`, by default
      the .mol of the first Mol() object, drawing w/ color or color function `color` and `opacity`, by
      default the global selection color.  Previous selection is replaced unless `clear` is false
    """
    if clear:
      self.selection = []
    if refresh:
      for hit in self.selection:
        hit.r = hit.vismol.curr_r[hit.idx] if hit.vismol else hit.r
    if sel or pdb:
      child = self.children[mol] if type(mol) is int else None
      mol = child.mol if child else mol
      r_array = child.curr_r if child else mol.r
      color_fn = coloring_opacity(color, opacity) if color else lambda mol, ii: self.sel_color
      selected = select_atoms(mol, sel, pdb=pdb)
      self.selection.extend([Bunch(mol=mol, r=r_array[ii], idx=ii,
          radius=(0.1 + ELEMENTS[mol.atoms[ii].znuc].vdw_radius) if radius == 'vdw' else radius,
          color=color_fn(mol,ii), vismol=child) for ii in selected])
    # print bond/angle/dihed info for 2/3/4 atom selection
    if print_dim:
      if len(self.selection) == 2:
        dist = calc_dist([h.r for h in self.selection])
        print("distance: %.2f Ang" % dist)
      elif len(self.selection) == 3:
        angle = calc_angle([h.r for h in self.selection])
        print("angle: %.2f degrees" % (180*angle/np.pi))
      elif len(self.selection) == 4:
        angle = calc_dihedral([h.r for h in self.selection])
        print("dihedral: %.2f degrees" % (180*angle/np.pi))
    # update renderer
    if len(self.selection) > 0:
      radii = [h.radius for h in self.selection]
      colors = [h.color for h in self.selection]
      self.selection_renderer.set_data([h.r for h in self.selection], radii, colors)
    self.viewer.repaint()
    return self  # for chaining


  def set_view(self, r=None, dist=None, r_up=None, center=False, make_default=False):
    """ rotate view to center `r` (or centroid of the current selection if None), optionally at `dist` from
      camera and with orientation defined by `r_up`; `r` is made camera pivot if `center` is True
    """
    if r is None and self.selection:
      r = np.mean([h.r for h in self.selection], axis=0)
    if r is not None:
      self.camera.rotate_to(r, dist, r_up)
    if center:
      dr = r - self.camera.pivot
      self.camera.pivot += dr
      self.camera.position += dr
    if make_default:
      self.initial_view = self.camera.state()
    self.viewer.repaint(wake=True)


  def get_n_mols(self):
    n_mols = [child.n_mols() for child in self.children
        if child.focused and child.visible and (child.n_mols() > 1 or self.step_singles)]
    return min(n_mols) if n_mols else 0  # min(..., default=0) in python 3


  # for efficiency, we do not load molecule for invisible mols or renderers until a mol or renderer is toggled
  #  to visible, at which point molecule is loaded for everything, visible and invisible
  def refresh(self, r=None, load_invisible=False, repaint=False, refresh_sel=False):
    """ (re)load molecule selected by self.mol_number for visible and focused children """
    for ii, child in enumerate(self.children):
      if child.focused and (child.visible or load_invisible):
        mol_number = self.mol_number if child.n_mols() > 1 else 0
        child.load_mol(mol_number, print_timing=self.print_timing, load_invisible=load_invisible, r=r)
        if self.verbose or self.print_timing:  #not self.animating:
          molname = getattr(child.mol, 'caption', getattr(child.mol, 'filename', ("Mol %d" % ii)))
          print("Loaded %s (%d)" % (molname, mol_number))
    self.invisible_loaded = load_invisible
    if refresh_sel:
      self.select(clear=False, refresh=True, print_dim=self.verbose)
    if repaint:
      self.viewer.repaint(wake=True)


  def run(self):
    # loading a molecule, and thus renderer data, will trigger shader compilation, so must create window first
    self.viewer.show_window()
    self.n_mols = self.get_n_mols()
    try:
      for ii,child in enumerate(self.children):
        if child.n_mols() > 0:
          child.load_mol(0, print_timing=self.print_timing, load_invisible=True)
          if not self.animating and hasattr(child.mol, 'filename'):
            print("Loaded %s" % child.mol.filename)
        else:
          print("No molecules (yet) for Mol %d" % ii)
    except:
      # if this works well, also add to viewer, exp. around render(); maybe also wrap callbacks!
      self.viewer.terminate()
      raise  # reraise

    self.invisible_loaded = True
    # set terminal title
    print("\x1B]0;%s (%s)\x07" % ('Python - Chemvis', os.getcwd()))
    # TODO: better way to get autozoom r_array; also, crashes if no molecules yet!
    if self.initial_view is None:
      self.camera.autozoom(self.children[0].mol.r, min_dist=20)
      self.camera.mouse_zoom(1.0)  # a little extra zoom
      self.initial_view = self.camera.state()
    if self.threaded:
      # run viewer event loop on separate thread so interpreter remains available
      t = threading.Thread(target=self.viewer.run)
      # TODO: find a way to set viewer.run_loop = False instead of using Thread.daemon
      # if getattr(sys, 'ps1', sys.flags.interactive):
      t.daemon = True  # end thread when interpreter exits
      t.start()
    else:
      self.viewer.run()
    # return self so user can easily access this object in interpreter
    return self


  def exit(self):
    self.viewer.run_loop = False
    self.viewer.repaint(wake=True)


  def on_finish(self):
    """ clear references to allow for GC (to prevent undesired printing when exiting python) """
    self.children = None
    self.viewer = None


  def draw(self, viewer):
    # callback to draw geometry for shadow and geometry passes
    def draw_geom(viewer):
      for pass_num in ['opaque', 'transparent', 'volume']:
        for ii, child in enumerate(self.children):
          if self.print_timing:
            molname = getattr(child.mol, 'filename', ("Mol %d" % ii))
            print("%s pass render times for %s:" % (pass_num, molname))
          child.draw(viewer, pass_num, self.print_timing)
        if viewer.curr_pass == 'geom' and pass_num == 'transparent' and len(self.selection) > 0:
          self.selection_renderer.draw(viewer)

    geom_extents = [get_extents(child.curr_r, pad=2.0) for child in self.children]
    mo_extents = [child.mol._vis.mo_extents for child in self.children
        if hasattr(child.mol, '_vis') and hasattr(child.mol._vis, 'mo_extents')]
    extents = np.array(geom_extents + mo_extents)
    view_extents = np.array([np.amin(extents[:,0], axis=0) - 0.01, np.amax(extents[:,1], axis=0) + 0.01])
    self.viewer.set_extents(view_extents)
    if self.shadows:
      if self.print_timing:
        print("Shadow pass:")
      viewer.shadow_pass(draw_geom)
    if self.print_timing:
      print("Geometry passes:")
    viewer.geom_pass(draw_geom)
    if self.print_timing:
      print("Postprocess passes:")
    for effect in self.effects:
      t0 = time.time() if self.print_timing else 0
      viewer.postprocess_pass(effect.draw, to_screen=(effect == self.effects[-1]))
      if self.print_timing:
        effect_name = getattr(effect, 'name', effect.__class__.__name__)
        print("  %s: %.2f ms" % (effect_name, (time.time() - t0)*1000.0))
    if not self.effects:
      viewer.blit_framebuffer()

    return False  # no extra redraw needed


  def on_click(self, viewer, x, y, mods):
    x,y = viewer.screen_to_ndc(x, y)
    # unproject() reverses perspective and view transforms to calculate world position given screen position
    origin = self.camera.unproject(x, y, -1.0)
    direction = normalize(self.camera.unproject(x, y, 0.0) - origin)

    hits = []
    for child in self.children:
      if child.visible and child.focused:
        for vis in child.renderers:
          if isinstance(vis, VisGeom) and getattr(vis, 'visible', True) and getattr(vis, 'focused', True):
            r_array = child.curr_r[vis.active] if vis.active is not None else child.curr_r
            selr = 1.0 if vis.style == 'spacefill' else 0.2  # H vdW radius is 1.1
            idxs, dists = ray_spheres_intersection(origin, direction, r_array, selr)
            if len(idxs) > 0:
              atomidx = vis.active[idxs[0]] if vis.active is not None else idxs[0]
              # TODO: clean this up after cleaning up VisGeom/MolRenderer; need to support for select() too!
              radius = 0.3
              if vis.style == 'spacefill':
                radius = 0.1 + ELEMENTS[child.mol.atoms[atomidx].znuc].vdw_radius
              hits.append(Bunch(mol=child.mol, r=r_array[idxs[0]], idx=atomidx, radius=radius, vismol=child,
                  color=self.sel_color, dist=dists[0]))

    if len(hits) > 0:
      hit = min(hits, key=lambda h: h.dist)  # pick closest
      mol = hit.mol
      atom = mol.atoms[hit.idx]
      res_str = ""
      if atom.resnum is not None and atom.resnum < len(mol.residues):
        res = mol.residues[atom.resnum]
        res_str = (" (%s %d)" % (res.name, atom.resnum))
        res_str += " (PDB %s %s)" % (res.chain, res.pdb_num) if res.pdb_num else ''
      name_str = atom.name or (ELEMENTS[atom.znuc].symbol if atom.znuc > 0 else 'X')
      print("%s (%d): %s%s" % (getattr(mol, 'filename', "???"), hit.idx, name_str, res_str))
      if 'Ctrl' in mods and self.selection:
        curridx = [ii for ii, h in enumerate(self.selection) if h.mol == hit.mol and h.idx == hit.idx]
        if curridx:
          # deselect if already selected
          del self.selection[curridx[0]]
        else:
          self.selection.append(hit)
      else:
        self.selection = [hit]
        if 'Ctrl' in mods:
          # print extra info if single atom selected with Ctrl
          print(atom)
      if 'Shift' in mods:
        # center view on clicked atom
        dr = hit.r - self.camera.pivot
        self.camera.pivot += dr
        self.camera.position += dr
        print("Camera centered on selected atom")
      if 'Alt' in mods:  # select entire molecule
        self.selection.extend([ Bunch(mol=mol, r=hit.vismol.curr_r[ii], idx=ii, radius=0.3,
            vismol=hit.vismol, color=self.sel_color) for ii in mol.get_connected(hit.idx) if ii != hit.idx ])
      # use select() to redraw selection and print dimensions
      self.select(clear=False, print_dim=True)
    elif len(self.selection) > 0 and 'Ctrl' not in mods:
      # Note that we don't clear selection if user was trying to extend
      self.selection = []
      viewer.repaint()


  def on_key_press(self, viewer, keycode, key, mods):
    if keycode == GLFW_KEY_F1:
      self.help()
      return
    if key == '0' or keycode == GLFW_KEY_HOME:
      if 'Shift' in mods:
        self.initial_view = self.camera.state()
        print("Camera default set: %s" % self.initial_view)
      elif 'Ctrl' in mods:
        # autozoom on first molecule - perhaps we should translate molecule COM to origin (persistent)?
        self.camera.autozoom(self.children[0].mol.r, min_dist=20)
        self.camera.mouse_zoom(1.0)
      elif 'Alt' in mods:
        # reset light direction
        RendererConfig.shading.light_dir = [0.0, 0.0, 1.0]
      else:
        self.camera.restore(self.initial_view)
    elif key in ',.':
      # ,/. steps through molecules
      # refresh file list before wrapping ... assumes new files are only added at end of sort order
      if self.animating and 'Auto' not in mods:  # pause animation
        self.animating = False
        viewer.animate(0)
      new_mol_number = self.mol_number + (1 if key == '.' else -1)
      if 'Shift' in mods or new_mol_number < 0 or new_mol_number >= self.n_mols:
        self.n_mols = self.get_n_mols()
        if 'Shift' in mods:
          new_mol_number = (self.n_mols - 1) if key == '.' else 0
      if self.wrap or not (new_mol_number < 0 or new_mol_number >= self.n_mols):
        self.mol_number = new_mol_number % self.n_mols
        self.refresh(refresh_sel=self.refresh_sel)
      elif self.verbose:  #not self.animating:
        print("Reached %s of file list" % ("beginning" if self.mol_number == 0 else "end"))
    elif key == 'P':
      # start/stop animation
      self.animating = not self.animating
      if self.animating:
        viewer.animate(self.animation_period, lambda: self.on_key_press(viewer, None, '.', ['Auto']))
      else:
        viewer.animate(0)
    elif key in 'IO':
      # adjust animation speed
      if self.animating:
        self.animation_period *= np.power((1.25 if key == 'I' else 0.8), (10 if 'Shift' in mods else 1))
        viewer.animate(self.animation_period)
    elif key == 'R' and 'Ctrl' in mods:  # and/or keycode==F5?
      if 'Shift' in mods:
        self.refresh_sel = not self.refresh_sel
      else:
        self.select(clear=False, refresh=True, print_dim=True)  # refresh selection
    elif key == 'U':
      self.wrap = not self.wrap  # 'W' is taken by viewer
    elif key == '`':
      ii = 1
      for jj, child in enumerate(self.children):
        mol_name = getattr(child.mol, 'filename', ("Mol %d" % jj))
        print("%s (Ctrl+%d) focused=%s visible=%s:" % (mol_name, jj, child.focused, child.visible))
        for vis in child.renderers:
          print("  %d: %s focused=%s visible=%s" % (ii, vis,
              getattr(vis, 'focused', True), getattr(vis, 'visible', True)))
          ii += 1
    elif key in '123456789':
      idx = ord(key) - ord('1')
      if 'Ctrl' in mods:
        # Ctrl+0-9 toggles entire molecule
        if idx >= len(self.children):
          return
        child = self.children[idx]
        if 'Shift' in mods:
          child.focused = not child.focused
        else:
          child.visible = not child.visible
          if child.visible and not self.invisible_loaded:
            self.refresh(load_invisible=True)
        mol_name = getattr(child.mol, 'filename', ("Mol %d" % idx))
        print("%s focused=%s visible=%s:" % (mol_name, child.focused, child.visible))
        # recompute n_mols
        self.n_mols = self.get_n_mols()
        if self.mol_number >= self.n_mols:
          # hack to reset focused molecules to mol 0
          self.on_key_press(viewer, keycode, ',', ['Shift'])
          return
      else:
        vs = [vis for child in self.children for vis in child.renderers]
        if idx >= len(vs):
          return
        # renderers don't have focused and visible attributes by default
        if 'Shift' in mods:
          vs[idx].focused = not getattr(vs[idx], 'focused', True)
        else:
          vs[idx].visible = not getattr(vs[idx], 'visible', True)
          if vs[idx].visible and not self.invisible_loaded:
            self.refresh(load_invisible=True)
        print("  %d: %s focused=%s visible=%s" % (idx+1, vs[idx],
            getattr(vs[idx], 'focused', True), getattr(vs[idx], 'visible', True)))
    else:
      if not any(child.on_key_press(viewer, keycode, key, mods) \
          for child in self.children if child.focused and child.visible):
        if not RendererConfig.shading.on_key_press(viewer, keycode, key, mods):
          if not any(effect.on_key_press(viewer, keycode, key, mods) for effect in self.effects):
            return
    viewer.repaint()


  def on_drag(self, dx, dy, mode):
    if not self.drag_atoms:
      return
    if mode == 'user_l':
      dr = self.camera.a*dx - self.camera.b*dy
      for hit in self.selection:
        hit.mol.atoms[hit.idx].r += 4*dr
    else:
      com = np.mean([h.r for h in self.selection], axis=0)
      if mode == 'user_shift_r':
        rot = rotation_matrix(self.camera.c, -1.5*dx)  # rotate about camera axis
      else:
        rot = rotation_matrix(self.camera.a, -1.5*dy)
        rot = np.dot(rotation_matrix(self.camera.b, -1.5*dx), rot)
      for hit in self.selection:
        hit.mol.atoms[hit.idx].r = com + np.dot(rot, hit.mol.atoms[hit.idx].r - com)
    # update renderers and selection
    for vismol in set([h.vismol for h in self.selection]):
      assert vismol.r is None, "Dragging not supported for Mol with separate r"
      vismol.load_mol(self.mol_number if vismol.n_mols() > 1 else 0)
    self.select(clear=False, refresh=True)


## Mol objects

class Mol:
  def __init__(self, mols, r=None, renderers=None, focused=True, visible=True, update_mol=False):
    """ Note that if r is passed, mols.r is not included automatically; if `update_mol` is true, current r
      will be assigned to mol.r
    """
    self.r = ([r] if np.ndim(r) == 2 else r) if renderers is not None and r is not None else None
    renderers = r if renderers is None else renderers
    self.mols = [mols] if hasattr(mols, 'r') else mols  # support anything with a .r attribute; len(dict) >= 0
    self.renderers = renderers if type(renderers) is list else [renderers]
    self.focused = focused
    self.visible = visible
    self.update_mol = update_mol
    assert self.r is None or np.shape(self.r)[1:] == np.shape(self.mols[0].r), "Mol(): r values do not match molecule!"

  def help(self):
    for child in self.renderers:
      child.help()

  def draw(self, viewer, pass_num, print_timing=False):
    for child in self.renderers:
      if getattr(child, 'visible', True):
        t0 = time.time() if print_timing else 0
        child.draw(viewer, pass_num)
        if print_timing:
          child_name = getattr(child, 'name', child.__class__.__name__)
          print("  %s: %.2f ms" % (child_name, (time.time() - t0)*1000.0))

  def n_mols(self):
    return len(self.mols) if self.r is None else len(self.r)

  def load_mol(self, mol_number, print_timing=False, load_invisible=False, r=None):
    self.mol = self.mols[mol_number] if self.r is None else self.mols[0]
    self.curr_r = r if r is not None else self.r[mol_number] if self.r is not None else self.mol.r
    if self.update_mol and (r is not None or self.r is not None):
      self.mol.r = self.curr_r
    for child in self.renderers:
      if getattr(child, 'visible', True) or load_invisible:
        t0 = time.time() if print_timing else 0
        child.set_molecule(self.mol, r=self.curr_r)
        if print_timing:
          child_name = getattr(child, 'name', child.__class__.__name__)
          print("  %s: %.2f ms" % (child_name, (time.time() - t0)*1000.0))

  def on_key_press(self, viewer, keycode, key, mods):
    for child in self.renderers:
      if getattr(child, 'focused', True) and getattr(child, 'visible', True):
        if child.on_key_press(viewer, keycode, key, mods):
          return True
    return False


class Files:
  def __init__(self, files, hydrogens=False, **kwargs):
    self.hydrogens, self.load_mol_kwargs = hydrogens, kwargs
    # molecule list
    if type(files) is list:
      self.mol_files = files
      self.file_glob = None
    else:
      self.mol_files = []
      self.file_glob = files
    self.mols = {}

  def __len__(self):
    if self.file_glob:
      self.mol_files = sorted(glob.glob(self.file_glob))
    return len(self.mol_files)

  def __getitem__(self, mol_number):
    mol_file = self.mol_files[mol_number]
    if mol_file not in self.mols:
      self.mols[mol_file] = load_molecule(mol_file, **self.load_mol_kwargs)
      if self.hydrogens: self.mols[mol_file] = add_hydrogens(self.mols[mol_file])
    return self.mols[mol_file]


## Misc visualization classes

class VisDLC(VisGeom):
  def __init__(self, coord_idx=0, **kwarg):
    super(VisDLC, self).__init__(**kwarg)

  def help(self):
    super(VisDLC, self).help()
    print(
""";': next/prev DLC coordinate
-=: step coordinate; Shift for 10x step
 Bksp: reset coordinate
""")

  def set_molecule(self, mol, r=None):
    super(VisDLC, self).set_molecule(mol, r)
    self.dlc = DLC(self.mol, recalc=1)
    self.dlc.init(r)
    self.S = dlc.active()
    self.initialS = self.S
    self.coord_idx = 0

  def on_key_press(self, viewer, keycode, key, mods):
    if key in ";'":
      self.coord_idx = (self.coord_idx + (-1 if key == ';' else 1)) % len(self.S)
      print("DLC coordinate %d selected (= %0.3f)" % (self.coord_idx, self.S[self.coord_idx]))
      return True
    if key in '-=':
      #if vol_rend.method in ['iso']:
      s = (0.25 if 'Shift' in mods else 0.025) * (-1 if key == '-' else 1)
      self.S[self.coord_idx] += s
      print("DLC coordinate %d set to %0.3f" % (self.coord_idx, self.S[self.coord_idx]))
    elif keycode == GLFW_KEY_BACKSPACE:
      self.S[self.coord_idx] = self.initialS[self.coord_idx]
      print("DLC coordinate %d reset to %0.3f" % (self.coord_idx, self.S[self.coord_idx]))
    else:
      return False
    # note that we do not reinit on breakdown, as that would change all DLCs,
    #  and so that user can step back in other direction
    if not dlc.update(S):
      print("DLC breakdown: geometry not updated!")
    # Update display without modifying molecule object (which is cached, not reloaded from log file!)
    self.sel_mol(mol, r_array=dlc.xyzs())
    return True


# should we support scale factor? log scale option?  remove vectors below some length threshold?
class VisVectors:

  def __init__(self, vec_fn=1.0, style='stick', radius=None, colors=None, origin=None):
    """ vec_fn can be a list of (vec_start, vec_end) pairs (or the transpose of this), or a single list of
      vectors, which will be assumed to originate from points mol.r, or a function returning one the above,
      given a Molecule; if vec_fn is scalar, it gives the length of x,y,z vectors drawn from origin
    """
    if np.isscalar(vec_fn):
      origin = np.zeros(3) if origin is None else np.array(origin)
      vec_fn = [(origin, vec_fn*np.eye(3)[ii]) for ii in [0,1,2]]
      colors = (lambda x: [Color.red, Color.green, Color.blue]) if colors is None else colors
    self.vec_fn = vec_fn
    self.radius = 1.5 if style == 'lines' else 0.05
    self.colors = Color.light_grey if colors is None else colors
    self.style = style
    self.renderer = LineRenderer() if style == 'lines' else StickRenderer()

  def __repr__(self):
    return "VisVectors()"

  def draw(self, viewer, pass_num):
    if pass_num == ('transparent' if self.style == 'lines' else 'opaque'):
      self.renderer.draw(viewer)

  def set_molecule(self, mol, r=None):
    vec = self.vec_fn(mol) if callable(self.vec_fn) else self.vec_fn
    if np.ndim(vec) == 2 and np.shape(vec)[1] == 3:
      r = mol.r if r is None else r
      bounds = np.stack((r, r + vec), axis=1)
    else:
      bounds = np.transpose(vec, axes=(1,0,2)) if len(vec) == 2 else np.asarray(vec)
    radii = self.radius(mol) if callable(self.radius) else [self.radius]*len(bounds)
    colors = self.colors(mol) if callable(self.colors) else [self.colors]*len(bounds)
    self.renderer.set_data(bounds, radii, colors)

  def on_key_press(self, viewer, keycode, key, mods):
    return False


# show lat, long when we have text support?
class VisGlobe:

  def __init__(self):
    # github.com/bgolus/EquirectangularSeamCorrection/blob/main/Assets/8081_earthmap10k.jpg
    self.renderer = BallRenderer(texture=DATA_PATH + "/earth1K.jpg", inset=True)

  def __repr__(self):
    return "VisGlobe()"

  def draw(self, viewer, pass_num):
    if pass_num == 'opaque':
      self.renderer.draw(viewer)

  def set_molecule(self, mol, r=None):
    self.renderer.set_data([(0,0,0)], radii=[3], colors=None)

  def on_key_press(self, viewer, keycode, key, mods):
    return False
