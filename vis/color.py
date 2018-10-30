import numpy as np
from ..data.elements import ELEMENTS
from ..molecule import residue_chain


## Color basics

def hex_to_rgba(h):
  h = h[1:] if h[0] == '#' else h
  h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2] if len(h) == 3 else h
  return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)

# using a class to provide a namespace for colors; Color instead of Colors saves 1 char, no other reason
class Color:
  white = hex_to_rgba('#FFFFFF')
  black = hex_to_rgba('#000000')
  red = hex_to_rgba('#FF0000')
  green = hex_to_rgba('#008000')
  lime = hex_to_rgba('#00FF00')
  blue = hex_to_rgba('#0000FF')
  yellow = hex_to_rgba('#FFFF00')
  magenta = hex_to_rgba('#FF00FF')
  cyan = hex_to_rgba('#00FFFF')
  grey = hex_to_rgba('#808080')
  light_grey = hex_to_rgba('#D3D3D3')
  dark_grey = hex_to_rgba('#A9A9A9')

# alternative approach: set static attributes of Color from a dict of colors
#   for k,v in named_colors: setattr(Color, k, hex_to_rgba(v))

# rename to Color.decode()?
def decode_color(c):
  if type(c) is str:
    return getattr(Color, c) if hasattr(Color, c) else hex_to_rgba(c)
  return c


# Note that we don't need many points for linear gradient color map loaded as texture, since OpenGL will
#  interpolate texture
def color_ramp(colors, t):
  """ linear interpolation of a list of `colors` based on `t`, scalar or array of values between 0 and 1
    alternatively, if `t` is an integer > 1, an array of length `t` will be returned for the color ramp
  """
  colors = np.array(colors)
  if type(t) is int and t > 1:
    t = np.reshape(np.linspace(0, 1, t), (-1,1))
  else:
    t = np.clip(np.array([t]) if np.isscalar(t) else np.asarray(t), 0.0, 1.0)
  i0 = np.floor(t * (len(colors) - 1)).astype(np.int)
  i1 = np.fmin(len(colors) - 1, i0 + 1)
  t = t*(len(colors) - 1) - i0
  return colors[i0]*(1 - t[:, None]) + colors[i1]*t[:, None]

# colormaps
# see https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/

# colormap fn can either be passed array of values between 0 and 1, or number of points desired
def fire_colormap(t):
  t = np.linspace(0, 1, t) if type(t) is int and t > 1 else t
  return np.clip(np.transpose([np.power(t, 0.5), t, t*t, t*1.05 - 0.05]), 0.0, 1.0)

# https://github.com/dhmunro/yorick/blob/master/g/heat.gp
def heat_colormap(t):
  t = np.linspace(0, 1, t) if type(t) is int and t > 1 else t
  return np.clip(np.transpose([1.5*t, 2.0*(t - 0.5), 4.0*(t - 0.75), t]), 0.0, 1.0)


# predefined colorings for atoms and residues
# following CSV is generated as described in elements.py, with
#  ... select atomic_number, symbol cpk_color, jmol_color, molcas_gv_color from ...
# Note that Speck uses jmol colors, QuteMol uses CPK colors
_ELEMENT_COLORS = \
""" 0,X ,#a0a0a0,#a0a0a0,#a0a0a0
 1,H,#ffffff,#ffffff,#f2f2f2
 2,He,#ffc0cb,#d9ffff,#d9ffff
 3,Li,#b22222,#cc80ff,#cc80ff
 4,Be,#ff1493,#c2ff00,#c2ff00
 5,B ,#00ff00,#ffb5b5,#ffb5b5
 6,C ,#c8c8c8,#909090,#555555
 7,N ,#8f8fff,#3050f8,#3753bb
 8,O ,#f00000,#ff0d0d,#f32e42
 9,F ,#daa520,#90e050,#7fd03b
10,Ne,#ff1493,#b3e3f5,#b3e3f5
11,Na,#0000ff,#ab5cf2,#ab5cf2
12,Mg,#228b22,#8aff00,#8aff00
13,Al,#808090,#bfa6a6,#bfa6a6
14,Si,#daa520,#f0c8a0,#f0c8a0
15,P ,#ffa500,#ff8000,#ff8000
16,S ,#ffc832,#ffff30,#fff529
17,Cl,#00ff00,#1ff01f,#38b538
18,Ar,#ff1493,#80d1e3,#80d1e3
19,K ,#ff1493,#8f40d4,#8f40d4
20,Ca,#808090,#3dff00,#3dff00
21,Sc,#ff1493,#e6e6e6,#e6e6e6
22,Ti,#808090,#bfc2c7,#bfc2c7
23,V ,#ff1493,#a6a6ab,#a6a6ab
24,Cr,#808090,#8a99c7,#8a99c7
25,Mn,#808090,#9c7ac7,#9c7ac7
26,Fe,#ffa500,#e06633,#e06633
27,Co,#ff1493,#f090a0,#f090a0
28,Ni,#a52a2a,#50d050,#50d050
29,Cu,#a52a2a,#c88033,#c88033
30,Zn,#a52a2a,#7d80b0,#7d80b0
31,Ga,#ff1493,#c28f8f,#c28f8f
32,Ge,#ff1493,#668f8f,#668f8f
33,As,#ff1493,#bd80e3,#bd80e3
34,Se,#ff1493,#ffa100,#ffa100
35,Br,#a52a2a,#a62929,#a62929
36,Kr,#ff1493,#5cb8d1,#5cb8d1
37,Rb,#ff1493,#702eb0,#702eb0
38,Sr,#ff1493,#00ff00,#00ff00
39,Y ,#ff1493,#94ffff,#94ffff
40,Zr,#ff1493,#94e0e0,#94e0e0
41,Nb,#ff1493,#73c2c9,#73c2c9
42,Mo,#ff1493,#54b5b5,#54b5b5
43,Tc,#ff1493,#3b9e9e,#3b9e9e
44,Ru,#ff1493,#248f8f,#248f8f
45,Rh,#ff1493,#0a7d8c,#0a7d8c
46,Pd,#ff1493,#006985,#006985
47,Ag,#808090,#c0c0c0,#c0c0c0
48,Cd,#ff1493,#ffd98f,#ffd98f
49,In,#ff1493,#a67573,#a67573
50,Sn,#ff1493,#668080,#668080
51,Sb,#ff1493,#9e63b5,#9e63b5
52,Te,#ff1493,#d47a00,#d47a00
53,I ,#a020f0,#940094,#940094
54,Xe,#ff1493,#429eb0,#429eb0
55,Cs,#ff1493,#57178f,#57178f
56,Ba,#ffa500,#00c900,#00c900
57,La,#ff1493,#70d4ff,#70d4ff
58,Ce,#ff1493,#ffffc7,#ffffc7
59,Pr,#ff1493,#d9ffc7,#d9ffc7
60,Nd,#ff1493,#c7ffc7,#c7ffc7
61,Pm,#ff1493,#a3ffc7,#a3ffc7
62,Sm,#ff1493,#8fffc7,#8fffc7
63,Eu,#ff1493,#61ffc7,#61ffc7
64,Gd,#ff1493,#45ffc7,#45ffc7
65,Tb,#ff1493,#30ffc7,#30ffc7
66,Dy,#ff1493,#1fffc7,#1fffc7
67,Ho,#ff1493,#00ff9c,#00ff9c
68,Er,#ff1493,#00e675,#00e675
69,Tm,#ff1493,#00d452,#00d452
70,Yb,#ff1493,#00bf38,#00bf38
71,Lu,#ff1493,#00ab24,#00ab24
72,Hf,#ff1493,#4dc2ff,#4dc2ff
73,Ta,#ff1493,#4da6ff,#4da6ff
74,W ,#ff1493,#2194d6,#2194d6
75,Re,#ff1493,#267dab,#267dab
76,Os,#ff1493,#266696,#266696
77,Ir,#ff1493,#175487,#175487
78,Pt,#ff1493,#d0d0e0,#d0d0e0
79,Au,#daa520,#ffd123,#ffd123
80,Hg,#ff1493,#b8b8d0,#b8b8d0
81,Tl,#ff1493,#a6544d,#a6544d
82,Pb,#ff1493,#575961,#575961
83,Bi,#ff1493,#9e4fb5,#9e4fb5
84,Po,#ff1493,#ab5c00,#ab5c00
85,At,#ff1493,#754f45,#754f45
86,Rn,#ffffff,#428296,#428296
87,Fr,#ffffff,#420066,#420066
88,Ra,#ffffff,#007d00,#007d00
89,Ac,#ffffff,#70abfa,#70abfa
90,Th,#ff1493,#00baff,#00baff
91,Pa,#ffffff,#00a1ff,#00a1ff
92,U ,#ff1493,#008fff,#008fff
93,Np,#ffffff,#0080ff,#0080ff
94,Pu,#ffffff,#006bff,#006bff
95,Am,#ffffff,#545cf2,#545cf2"""

_element_colors = [ [hex_to_rgba(c) for c in line.split(',')[-3:]] for line in _ELEMENT_COLORS.splitlines() ]
CPK_COLORS, JMOL_COLORS, MOLCAS_COLORS = zip(*_element_colors)


# I got used to these colors!
CHEMLAB_COLORS = {
    0: hex_to_rgba('#A0A0A0'), # for, e.g., background charges
  'H': hex_to_rgba('#FFFFFF'), #(255, 255, 255, 255),
  'C': hex_to_rgba('#808080'), #(128, 128, 128, 255),
  'N': hex_to_rgba('#ADD8E6'), #(173, 216, 230, 255),
  'O': hex_to_rgba('#FF0000'), #(255,   0,   0, 255),
  'S': hex_to_rgba('#FFD700')  #(255, 215,   0, 255)
}

# from Jmol via ngl
RESIDUE_COLORS = {
  'ALA': '#8CFF8C',
  'ARG': '#00007C',
  'ASN': '#FF7C70',
  'ASP': '#A00042',
  'CYS': '#FFFF70',
  'GLN': '#FF4C4C',
  'GLU': '#660000',
  'GLY': '#FFFFFF',
  'HIS': '#7070FF',
  'ILE': '#004C00',
  'LEU': '#455E45',
  'LYS': '#4747B8',
  'MET': '#B8A042',
  'PHE': '#534C52',
  'PRO': '#525252',
  'SER': '#FF7042',
  'THR': '#B84C00',
  'TRP': '#4F4600',
  'TYR': '#8C704C',
  'VAL': '#FF8CFF',

  'ASX': '#FF00FF',
  'GLX': '#FF00FF',
  'ASH': '#FF00FF',
  'GLH': '#FF00FF',

  'A': '#A0A0FF',
  'G': '#FF7070',
  'I': '#80FFFF',
  'C': '#FF8C4B',
  'T': '#A0FFA0',
  'U': '#FF8080',

  'DA': '#A0A0FF',
  'DG': '#FF7070',
  'DI': '#80FFFF',
  'DC': '#FF8C4B',
  'DT': '#A0FFA0',
  'DU': '#FF8080'
}

# this should probably be moved somewhere in chem/data/
# ref: http://blanco.biomol.uci.edu/Whole_residue_HFscales.txt
# columns: \delta G for water to bilayer interface; \delta G for water to octanol; difference
# Note that larger value means more hydrophilic
RESIDUE_HYDROPHOBICITY = {
  'ALA' : 0.17 , # 0.50  0.33
  'ARG' : 0.81 , # 1.81  1.00  # ARG+
  'ASN' : 0.42 , # 0.85  0.43
  'ASP' : 1.23 , # 3.64  2.41  # ASP-
  #'ASP0' : -0.07, # 0.43  0.50  # aka ASH
  'CYS' : -0.24, # -0.02 0.22
  'GLN' : 0.58 , # 0.77  0.19
  'GLU': 2.02 , # 3.63  1.61  # GLU-
  #'GLU0' : -0.01, # 0.11  0.12  # aka GLH
  'GLY' : 0.01 , # 1.15  1.14
  #'HIS+': 0.96 , # 2.33  1.37
  'HIS' : 0.17 , # 0.11  -0.06  # HIS0
  'ILE' : -0.31, # -1.12 -0.81
  'LEU' : -0.56, # -1.25 -0.69
  'LYS' : 0.99 , # 2.80  1.81  # LYS+
  'MET' : -0.23, # -0.67 -0.44
  'PHE' : -1.13, # -1.71 -0.58
  'PRO' : 0.45 , # 0.14  -0.31
  'SER' : 0.13 , # 0.46  0.33
  'THR' : 0.14 , # 0.25  0.11
  'TRP' : -1.85, # -2.09 -0.24
  'TYR' : -0.94, # -0.71 0.23
  'VAL' : 0.07   # -0.46 -0.53
}

## Coloring methods - passed molecule object and atom index; return color to be assigned to the atom

def color_by_constant(mol, idx, colors):
  return colors

def color_by_element(mol, idx, colors=CHEMLAB_COLORS):
  znuc = mol.atoms[idx].znuc
  return colors.get(znuc) or colors.get(ELEMENTS[znuc].symbol) or CPK_COLORS[znuc]

def color_by_chain(mol, idx, colors):
  resnum = mol.atoms[idx].resnum
  chain, chain_idx, chain_start, chain_stop = residue_chain(mol, resnum)
  return colors[chain_idx % len(colors)]

def color_by_residue(mol, idx, colors={}):
  res = mol.residues[mol.atoms[idx].resnum].name
  return colors.get(res) or hex_to_rgba(RESIDUE_COLORS[res])

def color_by_resnum(mol, idx, colors=(Color.blue, Color.green, Color.red)):
  resnum = mol.atoms[idx].resnum
  chain, chain_idx, chain_start, chain_stop = residue_chain(mol, resnum)
  t = (float(resnum) - chain_start)/(chain_stop - chain_start)  # chain_stop -= 1 for inclusive range
  return color_ramp(colors, t)[0]

# use scalar_coloring?
# TODO: handle ASP0, GLU0, HIS+
def color_by_hydrophobicity(mol, idx, colors=(Color.blue, Color.white, Color.red)):
  range = (-2.0, 2.0)
  try:
    val = RESIDUE_HYDROPHOBICITY[mol.residues[mol.atoms[idx].resnum].name]
    return color_ramp(colors, (val - range[0])/(range[1] - range[0]))[0]
  except:
    return Color.dark_grey

# color based on scalar attribute of atom such as `mmq`, or by fn returning scalar
def scalar_coloring(attr, range, default=0.0):
  def color_by_scalar(mol, idx, colors=None):
    # for signed values, use white for 0
    if colors is None:
      colors = (Color.blue, Color.white, Color.red) if range[0] == -range[1] \
          else (Color.blue, Color.green, Color.red)
    val = attr(mol, idx) if callable(attr) else getattr(mol.atoms[idx], attr, default)
    return color_ramp(colors, (val - range[0])/float(range[1] - range[0]))[0]
  return color_by_scalar


## Coloring modifiers

def coloring_opacity(coloring, opacity):
  """ wrap a coloring fn to change opacity """
  def wrapper(*args, **kwargs):
    c = coloring(*args, **kwargs) if callable(coloring) else coloring
    return (c[0], c[1], c[2], c[3]*opacity)
  return wrapper

def coloring_mix(coloring1, coloring2, a):
  """ mix two colorings (functions or single colors) together with factor `a`; alpha is taken from first  """
  def wrapper(*args, **kwargs):
    c1 = coloring1(*args, **kwargs) if callable(coloring1) else coloring1
    c2 = coloring2(*args, **kwargs) if callable(coloring2) else coloring2
    return (c1[0]*(1-a) + c2[0]*a, c1[1]*(1-a) + c2[1]*a, c1[2]*(1-a) + c2[2]*a, c1[3])
  return wrapper

# convert color to a shade of `base` color based on its luminosity (for molecule of the month style)
# `base` can be single color or a function returning a color
def coloring_monotone(coloring, base):
  def wrapper(*args, **kwargs):
    c = coloring(*args, **kwargs) if callable(coloring) else coloring
    b = base(*args, **kwargs) if callable(base) else base
    Y = (0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2])/255.0  # luminosity
    return (b[0]*Y, b[1]*Y, b[2]*Y, c[3])
  return wrapper
