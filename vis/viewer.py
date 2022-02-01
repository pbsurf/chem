""" GLFW OpenGL application: main loop and basic camera movement """

import time, math, threading  # threading for animation timer
import numpy as np
import OpenGL
OpenGL.ERROR_CHECKING = False  # if our error check at end of frame fires, disable this to track down error
#OpenGL.FULL_LOGGING = True  # set log level to DEBUG in chemvis.py to trace GL calls
#OpenGL.ERROR_ON_COPY = True  -- requires we manage lifetime of everything passed to GL
#OpenGL.STORE_POINTERS = False
from OpenGL.GL import *
from ..external.glfw import *
from ..external.glfw import _glfw  # symbols starting with underscore not imported by *
from .glutils import *

# dict_keys() printed on exit after using chemvis come from PyOpenGL arrays/arraydatatype.py:55, called from
#  doBufferDeletion() at arrays/vbo.py:107 set by arrays/vbo.py:286 (for some reason this doesn't happen when
#  threading=False); soln for now is to use user_finish callback to clear children in chemvis for GC

class GLFWViewer:
  inited = False

  def __init__(self, camera, bg_color=(255,255,255,255)):
    """ callbacks to be set by caller after creating GLFWViewer object: user_on_key, user_on_click,
      user_on_drag, user_draw
    """
    self.camera = camera
    self.bg_color = bg_color
    self.animate_event = False
    self.fbo = None
    self.window = None
    self.animation_period = 0
    # texture unit for reading shadow map texture; anyway to avoid hardcoding value?
    self.shadow_tex_id = 7
    self.curr_pass = ''
    self.mouse_dist_sq = 0
    self.auto_orbit = (0,0)


  def show_window(self):
    if self.window is not None:
      glfwShowWindow(self.window)
      return
    # Shaders will not compile without a current GL context
    if not GLFWViewer.inited:
      glfwInit()
      GLFWViewer.inited = True
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    # this disables deprecated functionality
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)

    window = glfwCreateWindow(640, 480, str.encode("Chemvis"), None, None)
    self.window = window
    glfwMakeContextCurrent(window)
    #glfwSwapInterval(1)  # vsync

    # setup callbacks
    glfwSetWindowSizeCallback(window, self.on_resize)
    glfwSetKeyCallback(window, self.on_key)
    glfwSetMouseButtonCallback(window, self.on_mouse_button)
    glfwSetCursorPosCallback(window, self.on_mouse_move)
    glfwSetScrollCallback(window, self.on_mouse_wheel)
    glfwSetWindowRefreshCallback(window, self.repaint)


  def run(self):
    window = self.window
    # we may be called on different thread than create_window()
    glfwMakeContextCurrent(window)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_MULTISAMPLE)
    glClearColor(*gl_color(self.bg_color))
    self.on_resize(window, *glfwGetWindowSize(window))

    self.dragging = None
    # main loop
    self.run_loop = True
    self.should_repaint = True
    while self.run_loop and not glfwWindowShouldClose(window):
      if self.should_repaint:
        t = time.time()
        self.should_repaint = self.render()
        err = glGetError()
        if err != GL_NO_ERROR:
          print("OpenGL error 0x%08x - enable OpenGL.ERROR_CHECKING!\n" % err)
        glfwSwapBuffers(window)
        # glDraw* fns return before rendering is complete - need to wait until glfwSwapBuffers returns instead
        #print "Render time: {:.3f} ms".format(1000*(time.time() - t))
      glfwWaitEvents()
      # see animate() method below
      if self.animate_event and self.user_animate:
        self.animate_event = False
        self.user_animate()
        self.should_repaint = True
    glfwHideWindow(window)
    #glDeleteTextures(list(self.fb_textures.values()))
    self.terminate()
    print("GLFWViewer event loop stopped")


  def terminate(self):
    if self.window is not None:
      # GLFW docs warn against destroying window when context is current on another thread
      #glfwMakeContextCurrent(self.window)
      #glfwTerminate() -- only if used for final exit, not per window as currently!
      #from OpenGL import contextdata;  contextdata.cleanupContext()  #contextdata.getContext()
      self.user_finish()
      glfwDestroyWindow(self.window)
      self.window = None


  def render(self):
    # user_draw should return True to request another repaint
    return self.user_draw(self)


  def on_resize(self, window, width, height):
    glViewport(0, 0, width, height)
    self.camera.aspectratio = width/float(height)
    print("Viewport resized to {} x {}".format(width, height))
    self.width, self.height = width, height
    if self.fbo is None:
      self.fbo = glGenFramebuffers(1)
    else:
      glDeleteTextures(list(self.fb_textures.values()))  # values() is a dictionary view object in Python 3
    # create textures
    color_tex = create_texture(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)
    # GL_RGB, i.e., three-bytes written with gl_FragData[1].xyz = ... doesn't seem to work in VMware
    # - and with GL_RGBA for 0 and GL_RGB for 1, writing to gl_FragColor crashes VMware
    normal_tex = create_texture(width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)
    depth_tex = create_texture(width, height, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT,
        wrap=GL_CLAMP_TO_BORDER, border_value=(1.,1.,1.,1.))
    self.fb_textures = {'color': color_tex, 'normal': normal_tex, 'depth': depth_tex, 'shadow': 0}
    # bind to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, normal_tex, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0)
    # Note that writing to gl_FragColor will write to every color attachment; also, by default, glBlendFunc
    #  applies to every color attachment, so a=1 should be used when writing normal; alternatively,
    #  glEnablei(GL_BLEND, <draw buffer index>) (GL 3.0) can be used to disable blending selectively
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
      raise Exception("Error: glCheckFramebufferStatus returned %d" % status)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)  # unnecessary?
    self.repaint()


  def swap_depth_tex(self):
    if 'depth_2' not in self.fb_textures:
      self.fb_textures['depth_2'] = create_texture(self.width, self.height, GL_DEPTH_COMPONENT24,
          GL_DEPTH_COMPONENT, GL_FLOAT, wrap=GL_CLAMP_TO_BORDER, border_value=(1.,1.,1.,1.))
    # swap depth textures
    self.fb_textures['depth'], self.fb_textures['depth_2'] = \
        self.fb_textures['depth_2'], self.fb_textures['depth']
    # bind new depth texture to fb
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.fb_textures['depth'], 0)
    glClear(GL_DEPTH_BUFFER_BIT)
    return self.fb_textures['depth_2']


  # workaround for a crazy bug that breaks volume rendering w/ updated vmware and/or host graphics drivers
  def swap_color_tex(self):
    if 'color_2' not in self.fb_textures:
      self.fb_textures['color_2'] = create_texture(self.width, self.height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)
    self.fb_textures['color'], self.fb_textures['color_2'] = \
        self.fb_textures['color_2'], self.fb_textures['color']
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fb_textures['color'], 0)


  def blit_framebuffer(self, destFBO=0):
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, destFBO)
    glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, GL_COLOR_BUFFER_BIT, GL_NEAREST)
    glBindFramebuffer(GL_FRAMEBUFFER, destFBO)


  # For now, we will implement multipass rendering by having user pass a callback to viewer methods *_pass()
  # - callback approach seems best because it allows viewer to run code before and after user draw code
  # - seems switching whole FBO is recommended over changing attachements, and binding both color
  #  textures to FBO and using glDrawBuffers to switch is even faster
  # - a key sanity check for shadows is that with light and camera pointing in same direction, e.g., -z axis,
  #  no shadows should be visible with orthographic projection

  def shadow_pass(self, callback):
    glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
    if self.fb_textures['shadow'] == 0:
      self.fb_textures['shadow'] = create_texture(self.width, self.height,
          GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, mag_filter=GL_LINEAR, min_filter=GL_LINEAR,
          wrap=GL_CLAMP_TO_BORDER, border_value=(1.,1.,1.,1.))
      # setup for sampler2DShadow
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.fb_textures['shadow'], 0)
    # previously we tried glBindTexture(..., 0) for shadow texture, which is not read during shadow pass, but
    #  this doesn't work for some drivers, nor does binding depth texture (GL_TEXTURE_COMPARE may need to be
    #  setup properly for texture bound to sampler2DShadow), so we'll bind the shadow texture
    #  which works for now, but seems like it could also fail for some drivers
    # ideal soln would be to use a separate shader for shadow pass that doesn't reference shadow texture
    glActiveTexture(GL_TEXTURE0+self.shadow_tex_id)
    glBindTexture(GL_TEXTURE_2D, self.fb_textures['shadow'])
    glDrawBuffer(GL_NONE)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    self.curr_pass = 'shadow'
    callback(self)


  def geom_pass(self, callback, to_screen=False):
    glBindFramebuffer(GL_FRAMEBUFFER, 0 if to_screen else self.fbo)
    if not to_screen:
      self.swap_color_tex()  # workaround bug breaking volume rendering - see TODO
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.fb_textures['depth'], 0)
      glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT3])
    # just bind shadow texture once, don't rebind for every shader
    glActiveTexture(GL_TEXTURE0+self.shadow_tex_id)
    glBindTexture(GL_TEXTURE_2D, self.fb_textures['shadow'])
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    self.curr_pass = 'geom'
    callback(self)
    glDisable(GL_BLEND)


  def postprocess_pass(self, callback, to_screen=False):
    glBindFramebuffer(GL_FRAMEBUFFER, 0 if to_screen else self.fbo)
    if not to_screen:
      if 'color_2' not in self.fb_textures:
        self.fb_textures['color_2'] = create_texture(self.width, self.height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self.fb_textures['color_2'], 0)
      if self.curr_pass != 'postprocess':
        # no depth write or depth test for postprocess
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        self.curr_color_attachment = 1
      else:
        # read texture is always fb_textures['color']
        self.fb_textures['color'], self.fb_textures['color_2'] = \
            self.fb_textures['color_2'], self.fb_textures['color']
        self.curr_color_attachment = 1 - self.curr_color_attachment  # toggle between 0 and 1
      glDrawBuffer(GL_COLOR_ATTACHMENT0+self.curr_color_attachment)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    self.curr_pass = 'postprocess'
    callback(self)
    if to_screen:
      # reenable depth test and write after last postprocess pass
      glEnable(GL_DEPTH_TEST)
      glDepthMask(GL_TRUE)
      self.curr_pass = ''


  def set_extents(self, extents):
    def get_rot_mat(dir):
      """ calculate a matrix to rotate vector dir to (0, 0, -1) """
      a0 = np.array([1.0, 0.0, 0.0])
      b0 = np.array([0.0, 1.0, 0.0])
      c0 = np.array([0.0, 0.0, -1.0])
      a = normalize(np.cross(a0, dir) if abs(np.dot(a0, dir)) < abs(np.dot(b0, dir)) else np.cross(b0, dir))
      b = normalize(np.cross(a, dir))
      c = dir
      mfinal = np.array([a0, b0, c0]).T
      morig = np.array([a, b, c]).T
      mrot = np.eye(4)
      mrot[:3,:3] = np.dot(mfinal, morig.T)
      return mrot

    self.camera.set_extents(extents)
    world_corners = np.array([ [extents[i,0], extents[j,1], extents[k,2], 1.0] for i in [0,1] for j in [0,1] for k in [0,1] ])
    ##self.camera._get_rotation_matrix(self)
    center = np.array([0.0, 0.0, 0.0, 1.0])
    center[:3] = 0.5*(extents[0,:] + extents[1,:])
    ## RendererConfig.shading.light_dir is direction TO light ... maybe should change this!
    shadow_rot = np.dot(np.dot(get_rot_mat(-np.asarray(RendererConfig.shading.light_dir)),
        translation_matrix(-np.dot(self.camera.view_matrix(), center)[:3])), self.camera.view_matrix())
    # calculate visible volume with view rotation
    view_corners = np.dot(world_corners, shadow_rot.T)
    view_max = np.amax(view_corners, 0)
    self._shadow_view_matrix = np.dot(translation_matrix(np.array([0.0, 0.0, -view_max[2]])), shadow_rot)
    # calculate visible volume with final view matrix
    view_corners = np.dot(world_corners, self._shadow_view_matrix.T)
    view_max = np.amax(view_corners, 0)
    view_min = np.amin(view_corners, 0)
    right = max(abs(view_max[0]), abs(view_min[0]))
    top = max(abs(view_max[1]), abs(view_min[1]))
    self._shadow_proj_matrix = ortho_proj(-right, right, -top, top, -view_max[2], -view_min[2])


  def shadow_view_matrix(self):
    return self._shadow_view_matrix

  def shadow_proj_matrix(self):
    return self._shadow_proj_matrix

  def view_matrix(self):
    return self.shadow_view_matrix() if self.curr_pass == 'shadow' else self.camera.view_matrix()

  def proj_matrix(self):
    return self.shadow_proj_matrix() if self.curr_pass == 'shadow' else self.camera.proj_matrix()


  # note that window argument is passed when used as glfwSetWindowRefreshCallback
  def repaint(self, window=None, wake=False):
    self.should_repaint = True
    if wake:
      _glfw.glfwPostEmptyEvent()


  # Do we need any locking for member access here?
  def animation_thread(self):
    while self.animation_period > 0:
      self.animate_event = True
      _glfw.glfwPostEmptyEvent()  # this causes glfwWaitEvents() to return
      time.sleep(self.animation_period/1000.0)

  # call callback at the specified period; pass period_ms = 0 to stop
  def animate(self, period_ms, callback=None):
    create_thread = self.animation_period <= 0 and period_ms > 0
    self.animation_period = period_ms
    if callback:
      self.user_animate = callback
    if create_thread:
      t = threading.Thread(target=self.animation_thread)
      t.daemon = True  # so thread doesn't prevent program from exiting
      t.start()


  # should this be in glutils instead?
  def screen_to_ndc(self, x, y):
    return 2.0*float(x)/self.width - 1.0, 1.0 - 2.0*float(y)/self.height


  def decode_mods(self, mods):
    mods_dict = dict(Shift=GLFW_MOD_SHIFT, Ctrl=GLFW_MOD_CONTROL, Alt=GLFW_MOD_ALT)
    return [k for k,v in mods_dict.items() if mods & v]


  # Note that key is a GLFW keycode - a bare letter key will actually correspond to the capital letter
  # - if you want actual key, accounting for modifiers automatically, use glfwSetCharCallback
  def on_key(self, window, key, scancode, action, mods):
    if action == GLFW_RELEASE:  # accept GLFW_PRESS and GLFW_REPEAT
      return
    # ignore modifier key press (wait for 2nd key)
    if key >= GLFW_KEY_LEFT_SHIFT and key <= GLFW_KEY_RIGHT_SUPER:
      return
    char = chr(key) if key >= 0 and key < 256 else chr(0)
    if key == GLFW_KEY_ESCAPE or key == GLFW_KEY_F10:
      self.run_loop = False  #glfwSetWindowShouldClose(window, 1)
    elif char in 'WASD':
      mag = 10 if mods & GLFW_MOD_SHIFT else 1
      delta = mag * (-1 if char in 'SA' else 1)
      dh, dv = delta if char in 'AD' else 0, delta if char in 'WS' else 0
      if mods & GLFW_MOD_CONTROL:
        if char in 'WS':
          self.camera.min_z_near = max(0.01, self.camera.min_z_near + delta)  #self.camera.mouse_zoom(0.2*delta)
          print("Near clipping plane: %f" % self.camera.min_z_near)
        else:
          self.camera.fov *= pow(1.1, delta)
          print("Camera FOV: %f degrees" % self.camera.fov)
      elif mods & GLFW_MOD_ALT:
        self.auto_orbit = (self.auto_orbit[0] - 0.002*dh, self.auto_orbit[1] - 0.002*dv)
        self.animate(50, lambda: self.camera.mouse_rotate(self.auto_orbit[0], self.auto_orbit[1]))
      else:
        self.camera.mouse_translate(dh, 0, dv)
      self.repaint()
    elif self.user_on_key is not None:
      self.user_on_key(self, key, char, self.decode_mods(mods))


  # motion: left button - arcball orbit, right button - translate x,y, wheel: zoom
  # shift+left button: rotate about view axis
  def on_mouse_button(self, window, button, action, mods):
    if action == GLFW_RELEASE:
      self.dragging = None
      # with touch support in the future, we'll need to have more complex click detection
      if self.mouse_dist_sq == 0 and self.user_on_click:
        self.user_on_click(self, self.last_mouse_x, self.last_mouse_y, self.decode_mods(mods))
    elif action == GLFW_PRESS:
      if any(self.auto_orbit):
        self.auto_orbit = (0,0)
        self.animate(0)
      self.mouse_dist_sq = 0
      if mods == 0:
        self.dragging = 'zoom' if button == GLFW_MOUSE_BUTTON_RIGHT else 'orbit'
      elif mods & GLFW_MOD_CONTROL and self.user_on_drag:
        self.dragging = 'user' + ('_shift' if mods & GLFW_MOD_SHIFT else '') \
            + ('_r' if button == GLFW_MOUSE_BUTTON_RIGHT else '_l')
      elif mods & GLFW_MOD_SHIFT:
        self.dragging = 'pan' if button == GLFW_MOUSE_BUTTON_RIGHT else 'spin'
      elif mods & GLFW_MOD_ALT:
        self.dragging = 'light'
      self.last_mouse_x, self.last_mouse_y = glfwGetCursorPos(self.window)


  def on_mouse_move(self, window, x, y):
    if self.dragging is None:
      return
    dx = 2.0*(x - self.last_mouse_x)/self.width
    dy = 2.0*(y - self.last_mouse_y)/self.height
    self.mouse_dist_sq += dx*dx + dy*dy
    if self.dragging == 'orbit':
      self.camera.mouse_rotate(dx, dy)
    elif self.dragging == 'spin':
      angle = math.atan2(y - 0.5*self.height, x - 0.5*self.width)
      last_angle = math.atan2(self.last_mouse_y - 0.5*self.height, self.last_mouse_x - 0.5*self.width)
      self.camera.mouse_rotate(0, 0, angle - last_angle)
    elif self.dragging == 'pan':
      self.camera.mouse_translate(dx, dy)
    elif self.dragging == 'zoom':
      self.camera.mouse_zoom(0.02*self.height*dy)
    elif self.dragging == 'light':
      ## TODO: obviously, we need to do something better than accessing a global class variable
      RendererConfig.shading.rotate_light(dx, dy)
    elif self.dragging[0:4] == 'user':
      self.user_on_drag(dx, dy, self.dragging)
    self.last_mouse_x, self.last_mouse_y = x,y
    self.repaint()


  def on_mouse_wheel(self, window, x, y):
    if glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS:
      self.camera.mouse_translate(0, 0, 0.1*y)
    else:
      self.camera.mouse_zoom(0.2*y)
    self.repaint()
