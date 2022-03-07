import numpy as np
from ..basics import partial
from ..data.elements import ELEMENTS
from ..data.pdb_bonds import RESIDUE_HYDROPHOBICITY


## Color basics

def hex_to_rgba(h):
  h = h[1:] if h[0] == '#' else h
  h, a = (h[2:], int(h[:2], 16)) if len(h) == 8 else (h, 255)
  h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2] if len(h) == 3 else h
  return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)

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


# we can convert this to apply a generic function in HSV space as the need arises
# example usage: VisGeom(..., colors=pastel_colors(CHEMLAB_COLORS))
def pastel_colors(colors, s=[1.0, 0.5, 1.5, 1.0]):
  """ shift list `colors` to pastel shades """
  import colorsys
  hsv = [colorsys.rgb_to_hsv(c[0]/255.0, c[1]/255.0, c[2]/255.0) + (c[3]/255.0,) for c in colors]
  pastel = np.clip(hsv * np.asarray(s), 0, 1)
  return 255*np.array([colorsys.hsv_to_rgb(c[0], c[1], c[2]) + (c[3],) for c in pastel])


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
# - http://jmol.sourceforge.net/jscolors/ - colors for atoms, residues, chains, etc.
# following CSV is generated as described in elements.py, with
#  ... select atomic_number, symbol cpk_color, jmol_color, molcas_gv_color from ...
# Note that Speck uses jmol colors, QuteMol uses CPK colors; Chemlab colors same as CPK except C,N,O,S
_ELEMENT_COLORS = \
""" 0,X ,#a0a0a0,#a0a0a0,#a0a0a0,#a0a0a0
 1,H ,#ffffff,#ffffff,#f2f2f2,#ffffff
 2,He,#ffc0cb,#d9ffff,#d9ffff,#ffc0cb
 3,Li,#b22222,#cc80ff,#cc80ff,#b22222
 4,Be,#ff1493,#c2ff00,#c2ff00,#ff1493
 5,B ,#00ff00,#ffb5b5,#ffb5b5,#00ff00
 6,C ,#c8c8c8,#909090,#555555,#808080
 7,N ,#8f8fff,#3050f8,#3753bb,#ADD8E6
 8,O ,#f00000,#ff0d0d,#f32e42,#FF0000
 9,F ,#daa520,#90e050,#7fd03b,#daa520
10,Ne,#ff1493,#b3e3f5,#b3e3f5,#ff1493
11,Na,#0000ff,#ab5cf2,#ab5cf2,#0000ff
12,Mg,#228b22,#8aff00,#8aff00,#228b22
13,Al,#808090,#bfa6a6,#bfa6a6,#808090
14,Si,#daa520,#f0c8a0,#f0c8a0,#daa520
15,P ,#ffa500,#ff8000,#ff8000,#ffa500
16,S ,#ffc832,#ffff30,#fff529,#FFD700
17,Cl,#00ff00,#1ff01f,#38b538,#00ff00
18,Ar,#ff1493,#80d1e3,#80d1e3,#ff1493
19,K ,#ff1493,#8f40d4,#8f40d4,#ff1493
20,Ca,#808090,#3dff00,#3dff00,#808090
21,Sc,#ff1493,#e6e6e6,#e6e6e6,#ff1493
22,Ti,#808090,#bfc2c7,#bfc2c7,#808090
23,V ,#ff1493,#a6a6ab,#a6a6ab,#ff1493
24,Cr,#808090,#8a99c7,#8a99c7,#808090
25,Mn,#808090,#9c7ac7,#9c7ac7,#808090
26,Fe,#ffa500,#e06633,#e06633,#ffa500
27,Co,#ff1493,#f090a0,#f090a0,#ff1493
28,Ni,#a52a2a,#50d050,#50d050,#a52a2a
29,Cu,#a52a2a,#c88033,#c88033,#a52a2a
30,Zn,#a52a2a,#7d80b0,#7d80b0,#a52a2a
31,Ga,#ff1493,#c28f8f,#c28f8f,#ff1493
32,Ge,#ff1493,#668f8f,#668f8f,#ff1493
33,As,#ff1493,#bd80e3,#bd80e3,#ff1493
34,Se,#ff1493,#ffa100,#ffa100,#ff1493
35,Br,#a52a2a,#a62929,#a62929,#a52a2a
36,Kr,#ff1493,#5cb8d1,#5cb8d1,#ff1493
37,Rb,#ff1493,#702eb0,#702eb0,#ff1493
38,Sr,#ff1493,#00ff00,#00ff00,#ff1493
39,Y ,#ff1493,#94ffff,#94ffff,#ff1493
40,Zr,#ff1493,#94e0e0,#94e0e0,#ff1493
41,Nb,#ff1493,#73c2c9,#73c2c9,#ff1493
42,Mo,#ff1493,#54b5b5,#54b5b5,#ff1493
43,Tc,#ff1493,#3b9e9e,#3b9e9e,#ff1493
44,Ru,#ff1493,#248f8f,#248f8f,#ff1493
45,Rh,#ff1493,#0a7d8c,#0a7d8c,#ff1493
46,Pd,#ff1493,#006985,#006985,#ff1493
47,Ag,#808090,#c0c0c0,#c0c0c0,#808090
48,Cd,#ff1493,#ffd98f,#ffd98f,#ff1493
49,In,#ff1493,#a67573,#a67573,#ff1493
50,Sn,#ff1493,#668080,#668080,#ff1493
51,Sb,#ff1493,#9e63b5,#9e63b5,#ff1493
52,Te,#ff1493,#d47a00,#d47a00,#ff1493
53,I ,#a020f0,#940094,#940094,#a020f0
54,Xe,#ff1493,#429eb0,#429eb0,#ff1493
55,Cs,#ff1493,#57178f,#57178f,#ff1493
56,Ba,#ffa500,#00c900,#00c900,#ffa500
57,La,#ff1493,#70d4ff,#70d4ff,#ff1493
58,Ce,#ff1493,#ffffc7,#ffffc7,#ff1493
59,Pr,#ff1493,#d9ffc7,#d9ffc7,#ff1493
60,Nd,#ff1493,#c7ffc7,#c7ffc7,#ff1493
61,Pm,#ff1493,#a3ffc7,#a3ffc7,#ff1493
62,Sm,#ff1493,#8fffc7,#8fffc7,#ff1493
63,Eu,#ff1493,#61ffc7,#61ffc7,#ff1493
64,Gd,#ff1493,#45ffc7,#45ffc7,#ff1493
65,Tb,#ff1493,#30ffc7,#30ffc7,#ff1493
66,Dy,#ff1493,#1fffc7,#1fffc7,#ff1493
67,Ho,#ff1493,#00ff9c,#00ff9c,#ff1493
68,Er,#ff1493,#00e675,#00e675,#ff1493
69,Tm,#ff1493,#00d452,#00d452,#ff1493
70,Yb,#ff1493,#00bf38,#00bf38,#ff1493
71,Lu,#ff1493,#00ab24,#00ab24,#ff1493
72,Hf,#ff1493,#4dc2ff,#4dc2ff,#ff1493
73,Ta,#ff1493,#4da6ff,#4da6ff,#ff1493
74,W ,#ff1493,#2194d6,#2194d6,#ff1493
75,Re,#ff1493,#267dab,#267dab,#ff1493
76,Os,#ff1493,#266696,#266696,#ff1493
77,Ir,#ff1493,#175487,#175487,#ff1493
78,Pt,#ff1493,#d0d0e0,#d0d0e0,#ff1493
79,Au,#daa520,#ffd123,#ffd123,#daa520
80,Hg,#ff1493,#b8b8d0,#b8b8d0,#ff1493
81,Tl,#ff1493,#a6544d,#a6544d,#ff1493
82,Pb,#ff1493,#575961,#575961,#ff1493
83,Bi,#ff1493,#9e4fb5,#9e4fb5,#ff1493
84,Po,#ff1493,#ab5c00,#ab5c00,#ff1493
85,At,#ff1493,#754f45,#754f45,#ff1493
86,Rn,#ffffff,#428296,#428296,#ffffff
87,Fr,#ffffff,#420066,#420066,#ffffff
88,Ra,#ffffff,#007d00,#007d00,#ffffff
89,Ac,#ffffff,#70abfa,#70abfa,#ffffff
90,Th,#ff1493,#00baff,#00baff,#ff1493
91,Pa,#ffffff,#00a1ff,#00a1ff,#ffffff
92,U ,#ff1493,#008fff,#008fff,#ff1493
93,Np,#ffffff,#0080ff,#0080ff,#ffffff
94,Pu,#ffffff,#006bff,#006bff,#ffffff
95,Am,#ffffff,#545cf2,#545cf2,#ffffff"""

_element_colors = np.array([ [hex_to_rgba(c) for c in line.split(',')[-4:]]
    for line in _ELEMENT_COLORS.splitlines() ], np.uint8)
CPK_COLORS, JMOL_COLORS, MOLCAS_COLORS, CHEMLAB_COLORS = [_element_colors[:,ii] for ii in range(4)]  #zip(*_element_colors)


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

# from https://github.com/molstar/molstar/blob/master/src/mol-util/color/lists.ts
_CHAIN_COLORS =  ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666',
  '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999',
  '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

CHAIN_COLORS = [ hex_to_rgba(c) for c in _CHAIN_COLORS ]


def get_chains(mol):
  """ for molecule `mol`, return a dict: chain -> (start resnum, stop resnum + 1) """
  chains = {}
  for resnum,res in enumerate(mol.residues):
    lim = chains.setdefault(res.chain, [resnum, resnum+1])
    lim[0], lim[1] = min(lim[0], resnum), max(lim[1], resnum+1)
  return chains

## Coloring methods - passed molecule object and atom index; return color to be assigned to the atom
# previous approach of separate call for each atom was a bit slow, so we now take array of atom indexes

def color_by_constant(mol, idxs, colors):
  return [colors]*len(idxs)

def color_by_element(mol, idxs, colors=CHEMLAB_COLORS):
  znucs = [mol.atoms[idx].znuc for idx in idxs]
  return colors[znucs]  #return colors.get(znuc) or colors.get(ELEMENTS[znuc].symbol) or CPK_COLORS[znuc]

def color_by_chain(mol, idxs, colors=CHAIN_COLORS, offset=0):
  resnums = [mol.atoms[idx].resnum for idx in idxs]
  chains = mol.chains
  return [colors[ (chains.index(mol.residues[ii].chain) + offset) % len(colors) ] for ii in resnums]

def color_by_residue(mol, idxs, colors={}):
  ress = [mol.residues[mol.atoms[idx].resnum].name for idx in idxs]
  return [colors.get(res) or hex_to_rgba(RESIDUE_COLORS[res]) for res in ress]

def color_by_resnum(mol, idxs, colors=(Color.blue, Color.green, Color.red)):
  resnums = [mol.atoms[idx].resnum for idx in idxs]
  #chain, chain_idx, chain_start, chain_stop = residue_chain(mol, resnum)
  #t = (float(resnum) - chain_start)/(chain_stop - chain_start)  # chain_stop -= 1 for inclusive range
  chains = get_chains(mol)
  tfn = lambda resnum, lims: (float(resnum) - lims[0])/(lims[1] - lims[0])
  t = [ tfn(resnum, chains[mol.residues[resnum].chain]) for resnum in resnums ]
  return color_ramp(colors, t)

# use scalar_coloring?
# TODO: handle ASP0, GLU0, HIS+
def color_by_hydrophobicity(mol, idxs, colors=(Color.blue, Color.white, Color.red)):
  range = (-2.0, 2.0)
  #try:
  vals = [ RESIDUE_HYDROPHOBICITY.get(mol.residues[mol.atoms[idx].resnum].name, 0) for idx in idxs ]
  return color_ramp(colors, (vals - range[0])/(range[1] - range[0]))
  #except: return Color.dark_grey

# color based on scalar attribute of atom such as `mmq`, or by fn returning scalar
def scalar_coloring(attr, range=None, default=0.0):
  range = [np.min(attr), np.max(attr)] if range is None else range
  def color_by_scalar(mol, idxs, colors=None):
    # for signed values, use white for 0
    if colors is None:
      colors = (Color.blue, Color.white, Color.red) if range[0] == -range[1] \
          else (Color.blue, Color.green, Color.red)
    vals = [attr(mol, idx) for idx in idxs] if callable(attr) \
        else [getattr(mol.atoms[idx], attr, default) for idx in idxs] if type(attr) is str else attr[idxs]
    return color_ramp(colors, (vals - range[0])/float(range[1] - range[0]))
  return color_by_scalar


## Coloring modifiers

def coloring_opacity(coloring, opacity):
  """ wrap a coloring fn to change opacity """
  def wrapper(*args, **kwargs):
    c = coloring(*args, **kwargs) if callable(coloring) else coloring
    return c*np.array([1, 1, 1, opacity])
  return wrapper

def coloring_mix(coloring1, coloring2, a):
  """ mix two colorings (functions or single colors) together with factor `a`; alpha is taken from first  """
  def wrapper(*args, **kwargs):
    c1 = coloring1(*args, **kwargs) if callable(coloring1) else coloring1
    c2 = coloring2(*args, **kwargs) if callable(coloring2) else coloring2
    aa = np.outer(a(*args, **kwargs) if callable(a) else a, [1,1,1,0])
    return c1*(1-aa) + c2*aa
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


## Common colorings

COLORING_FNS = dict(
  element=color_by_element,
  chain=color_by_chain,
  residue=color_by_residue,
  resnum=color_by_resnum,
  carbonchain=coloring_mix(color_by_element, color_by_chain, lambda mol,idxs: 0.4*(mol.znuc[idxs] == 6)),
  resnumchain=coloring_mix(partial(color_by_resnum, colors=(Color.black, Color.white)), color_by_chain, 0.5),
  motm=coloring_mix(color_by_element, color_by_chain, 0.85),  # molecule-of-the-month type coloring
  mmq=scalar_coloring('mmq', [-1,1])
)

def color_by(coloring, colors=None):
  coloring = COLORING_FNS[coloring] if type(coloring) is str else coloring
  return partial(coloring, colors=colors) if colors is not None else coloring
