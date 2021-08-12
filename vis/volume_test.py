import numpy as np
from itertools import cycle

from glfw import GLFW_MOD_SHIFT
from viewer import GLFWViewer
from camera import Camera
from glutils import fire_colormap, colorramp
from volume_renderer import VolumeRenderer
from mol_renderer import MolRenderer, LightingShaderModule, StringShaderModule

# see /home/mwhite/vispy/examples/basics/scene/volume.py

vol1 = np.load('/home/mwhite/chemlab/examples/stent.npz')['arr_0']
#~ vol2 = np.load('./mri.npz')['data']
#~ vol2 = np.flipud(np.rollaxis(vol2, 1))
vol1 = vol1.astype(np.float32)/np.max(vol1)

s = np.shape(vol1)
vol1_max = np.array(s)/float(np.max(s))
vol1_extents = np.transpose([-vol1_max, vol1_max])
print "Volume extents: {}".format(vol1_extents)


camera = Camera()
v = GLFWViewer(camera, bg_color=(0,0,0,1))

lighting_module = LightingShaderModule(camera, light_dir=[0.0, 0.0, 10.0], shading='phong')
header_module = StringShaderModule("#version 120", "#version 120")

methods = cycle(['iso', 'volume'])
vol_rend = VolumeRenderer(camera, method=next(methods), modules=[header_module, lighting_module])
#colormap = colorramp((0.,0.,0.,0.), (1.,1.,1.,0.5))
colormap = fire_colormap(np.linspace(0, 1, 256))
vol_obj = vol_rend.make_volume_obj(vol1, vol1_extents,
    color=(1.,0.,0.,0.75), isolevel=0.2, color_n=(0.,0.,1.,0.75), isolevel_n=-0.2, colormap=colormap)
v.renderers.append(vol_rend)

#camera.autozoom(np.transpose(vol1_extents))
# Give some extra zoom
camera.mouse_zoom(1.0)
initial_view = camera.state()

def on_key_press(vis, keycode, key, mods):
  if key == '0':  # or Home key?
    camera.restore(initial_view)
  elif key in '-=':
    #if vol_rend.method in ['iso']:
    s = (0.25 if mods & GLFW_MOD_SHIFT else 0.025) * (-1 if key == '-' else 1)
    if vol_obj.isolevel + s >= 0:
      vol_obj.isolevel += s
      vol_obj.isolevel_n -= s
    print("Isosurface threshold: +/-%0.3f" % vol_obj.isolevel)
  elif key in '[]':
    s = np.power((1.25 if key == ']' else 0.8), (10 if mods & GLFW_MOD_SHIFT else 1))
    vol_obj.absorption *= s
    print("Volume absorbtivity: +/-%0.3f" % vol_obj.absorption)
  elif key == 'V':
    vol_rend.set_method(next(methods))
    print("Volume render method: %s" % vol_rend.method)
  else:
    return
  vis.repaint()

v.user_on_key = on_key_press

v.run()
