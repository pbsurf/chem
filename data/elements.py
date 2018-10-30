# Element data obtained from https://bitbucket.org/lukaszmentel/mendeleev/ project
# Would have stuck with the excellent https://www.lfd.uci.edu/~gohlke/code/elements.py.html but needed colors
#  for rendering elements!  See https://en.wikipedia.org/wiki/CPK_coloring
# Creating ELEMENTS_CSV:
#   wget https://bitbucket.org/lukaszmentel/mendeleev/raw/e3e43817e710c09541bca0d12d520a52fc169d5f/mendeleev/elements.db
#   sqlite3 elements.db
#   .mode csv
#   select atomic_number, symbol, atomic_weight, vdw_radius/100, covalent_radius_pyykko/100
#     from elements where atomic_number < 96;

#atomic_number, symbol, atomic_weight, vdw_radius, covalent_radius
ELEMENTS_CSV = \
"""1,H,1.008,1.1,0.32
2,He,4.002602,1.4,0.46
3,Li,6.94,1.82,1.33
4,Be,9.0121831,1.53,1.02
5,B,10.81,1.92,0.85
6,C,12.011,1.7,0.75
7,N,14.007,1.55,0.71
8,O,15.999,1.52,0.63
9,F,18.998403163,1.47,0.64
10,Ne,20.1797,1.54,0.67
11,Na,22.98976928,2.27,1.55
12,Mg,24.305,1.73,1.39
13,Al,26.9815385,1.84,1.26
14,Si,28.085,2.1,1.16
15,P,30.973761998,1.8,1.11
16,S,32.06,1.8,1.03
17,Cl,35.45,1.75,0.99
18,Ar,39.948,1.88,0.96
19,K,39.0983,2.75,1.96
20,Ca,40.078,2.31,1.71
21,Sc,44.955908,2.15,1.48
22,Ti,47.867,2.11,1.36
23,V,50.9415,2.07,1.34
24,Cr,51.9961,2.06,1.22
25,Mn,54.938044,2.05,1.19
26,Fe,55.845,2.04,1.16
27,Co,58.933194,2.0,1.11
28,Ni,58.6934,1.97,1.1
29,Cu,63.546,1.96,1.12
30,Zn,65.38,2.01,1.18
31,Ga,69.723,1.87,1.24
32,Ge,72.63,2.11,1.21
33,As,74.921595,1.85,1.21
34,Se,78.971,1.9,1.16
35,Br,79.904,1.85,1.14
36,Kr,83.798,2.02,1.17
37,Rb,85.4678,3.03,2.1
38,Sr,87.62,2.49,1.85
39,Y,88.90584,2.32,1.63
40,Zr,91.224,2.23,1.54
41,Nb,92.90637,2.18,1.47
42,Mo,95.95,2.17,1.38
43,Tc,97.90721,2.16,1.28
44,Ru,101.07,2.13,1.25
45,Rh,102.9055,2.1,1.25
46,Pd,106.42,2.1,1.2
47,Ag,107.8682,2.11,1.28
48,Cd,112.414,2.18,1.36
49,In,114.818,1.93,1.42
50,Sn,118.71,2.17,1.4
51,Sb,121.76,2.06,1.4
52,Te,127.6,2.06,1.36
53,I,126.90447,1.98,1.33
54,Xe,131.293,2.16,1.31
55,Cs,132.90545196,3.43,2.32
56,Ba,137.327,2.68,1.96
57,La,138.90547,2.43,1.8
58,Ce,140.116,2.42,1.63
59,Pr,140.90766,2.4,1.76
60,Nd,144.242,2.39,1.74
61,Pm,144.91276,2.38,1.73
62,Sm,150.36,2.36,1.72
63,Eu,151.964,2.35,1.68
64,Gd,157.25,2.34,1.69
65,Tb,158.92535,2.33,1.68
66,Dy,162.5,2.31,1.67
67,Ho,164.93033,2.3,1.66
68,Er,167.259,2.29,1.65
69,Tm,168.93422,2.27,1.64
70,Yb,173.045,2.26,1.7
71,Lu,174.9668,2.24,1.62
72,Hf,178.49,2.23,1.52
73,Ta,180.94788,2.22,1.46
74,W,183.84,2.18,1.37
75,Re,186.207,2.16,1.31
76,Os,190.23,2.16,1.29
77,Ir,192.217,2.13,1.22
78,Pt,195.084,2.13,1.23
79,Au,196.966569,2.14,1.24
80,Hg,200.592,2.23,1.33
81,Tl,204.38,1.96,1.44
82,Pb,207.2,2.02,1.44
83,Bi,208.9804,2.07,1.51
84,Po,209.0,1.97,1.45
85,At,210.0,2.02,1.47
86,Rn,222.0,2.2,1.42
87,Fr,223.0,3.48,2.23
88,Ra,226.0,2.83,2.01
89,Ac,227.0,2.47,1.86
90,Th,232.0377,2.45,1.75
91,Pa,231.03588,2.43,1.69
92,U,238.02891,2.41,1.7
93,Np,237.0,2.39,1.71
94,Pu,244.0,2.43,1.72
95,Am,243.0,2.44,1.66"""

class Element:
  def __init__(self, **kwargs):
    for key,val in kwargs.iteritems():
      setattr(self, key, val)

  def __repr__(self):
    return "Element(symbol='%s', number=%d, mass=%f, vdw_radius=%f, cov_radius=%f)" % \
        (self.symbol, self.number, self.mass, self.vdw_radius, self.cov_radius)


ELEMENTS = {}
for line in ELEMENTS_CSV.splitlines():
  l = line.split(',')
  e = Element(
    number=int(l[0]),
    symbol=l[1],
    mass=float(l[2]),
    vdw_radius=float(l[3]),
    cov_radius=float(l[4]),
  )
  ELEMENTS[e.number] = e
  ELEMENTS[e.symbol] = e
