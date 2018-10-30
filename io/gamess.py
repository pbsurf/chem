import os, time
from . import cclib_open
from ..basics import *
from ..data.elements import ELEMENTS


GAMESS_PATH = os.getenv('GAMESS_PATH', '')

# alternative is to convert text from nwchem format without parsing values - see r. 158 for untested impl
def to_gamess_basis(basis, znuc=None):
  """ write basis set in cclib or pyscf format to GAMESS format for single atom; since support for named basis
    is so limited in this format, string - which must be accompanied by `znuc` - is assumed to be a standard
    basis name and passed to pyscf to obtain coefficients """
  if type(basis) is str:
    #return "  " + str + "\n"
    from pyscf import gto
    basis = gto.basis.load(basis, ELEMENTS[znuc].symbol)

  res = []
  # need to print shared exponent L basis fns to ensure associated optimizations are used
  Ldict = {}
  for ao in basis:
    l, coeffs = ('SPDFGHIK'[ao[0]], ao[1:]) if type(ao[0]) is int else (ao[0], ao[1])
    if l == 'P':
      Ldict[tuple(coeff[0] for coeff in coeffs)] = [coeff[1] for coeff in coeffs]
  for ao in basis:
    # cclib uses S,P,D,... whereas pyscf uses 0,1,2,..
    l, coeffs = ('SPDFGHIK'[ao[0]], ao[1:]) if type(ao[0]) is int else (ao[0], ao[1])
    key = tuple(coeff[0] for coeff in coeffs)
    if l == 'S' and key in Ldict:
      pcoeffs = Ldict[key]
      res.append("  L  %d" % len(coeffs))
      res.extend("    %4d %22.12f   % .12f   % .12f" % (ii+1, coeff[0], coeff[1], pcoeffs[ii]) \
          for ii, coeff in enumerate(coeffs))
      Ldict[key] = 'done'
    elif l == 'P' and Ldict[key] == 'done':
      pass
    else:
      res.append("  %s  %d" % (l, len(coeffs)))
      res.extend("    %4d %22.12f   % .12f" % (ii+1, coeff[0], coeff[1]) for ii, coeff in enumerate(coeffs))
  res.append('')  # blank line to end basis
  return "\n".join(res)


# Background charges (called "sparkles" internally in GAMESS) must follow atoms in GAMESS $DATA and must
#  have abs(q) - int(abs(q)) > 1E-05 (otherwise, interpreted as QM atom)
# - see comments and code in GAMESS source/inputa.src
# znuc field is optional for caps; assumed to be 1 (hydrogen) if not present
def write_gamess_inp(mol, filename, header,
    r=None, qmatoms=None, caps=[], charges=[], moguess=None, title=None, write_basis=True):
  """ generate GAMESS input file with `qmatoms` from `mol`, plus point charges `charges` and link atoms
    `caps`.  Initial guess MOs `moguess` are written to $VEC section if provided
  """
  def adjust_q(q):
    fract = abs(q) - int(abs(q))
    sign = -1 if q < 0 else 1
    if fract > 1E-05:
      return q
    return int(q) + sign*1.1E-05 if fract > 0.5E-05 else int(q) - sign*1E-06

  strs = [header, (" $DATA\n%s\nC1" % title)]
  # QM atoms
  qmatoms = mol.listatoms() if qmatoms is None else qmatoms
  r = mol.r if r is None else r
  for ii in qmatoms:
    a = mol.atoms[ii]
    strs.append("%-6s %3d.0         %13.8f %13.8f %13.8f" % (a.name, a.znuc, r[ii][0], r[ii][1], r[ii][2]))
    if write_basis:
      strs.append(to_gamess_basis(a.qmbasis, a.znuc) if getattr(a, 'qmbasis', None) else '')
  # link/cap atoms
  for cap in caps:
    znuc = getattr(cap, 'znuc', 1)
    strs.append("%-6s %3d.0         %13.8f %13.8f %13.8f" % ("HL", znuc, cap.r[0], cap.r[1], cap.r[2]))
    if write_basis:
      strs.append(to_gamess_basis(cap.qmbasis, znuc) if getattr(cap, 'qmbasis', None) else '')
  # point charges
  for q in charges:
    assert abs(q.qmq) > 0.5E-06, "Charge must be larger than +/-0.5E-6"  # caller must remove small charges!
    strs.append("Q%-5s %10.6f    %13.8f %13.8f %13.8f" % (q.name, adjust_q(q.qmq), q.r[0], q.r[1], q.r[2]))
    if write_basis:
      strs.append('')
  # end $DATA
  strs.append("$END")
  # write a $VEC section in format of PUNCH file if moguess provided
  if moguess is not None:
    moguess = moguess[0] if len(moguess) == 1 else moguess
    vec = [ ("%2d" % ((ii+1)%100))[-2:] + " " + ("%2d" % ((jj/5+1)%100))[-2:]  \
        + "".join("% .8E" % c for c in coeffs_ii[jj:jj+5])  \
        for ii, coeffs_ii in enumerate(moguess) for jj in range(0, len(coeffs_ii), 5) ]
    strs.append(" $VEC")
    strs.extend(vec)
    strs.append("$END")
  # write everything to .inp file
  with open(filename, 'w') as f:
    f.write("\n".join(strs))


def gamess_cclib(mol, r=None, prefix=None, cleanup=False, **kwargs):
  """ generate GAMESS inp, run GAMESS, and return resulting log parsed by cclib """
  if prefix is None:
    prefix = "qmtmp_%d" % time.time()
    cleanup = True

  # rungms must be run from folder containing inp file, so split path from prefix
  splitfix = prefix.rsplit('/', 1)
  qmdir, prefix = (splitfix[0] + '/', splitfix[1]) if len(splitfix) > 1 else ('', prefix)
  qminp = prefix + ".inp"
  qmout = prefix + ".log"

  write_gamess_inp(mol, qmdir + qminp, r=r, **kwargs)
  # changed rungms to clear scratch files instead of complaining
  # SCR_ROOT must be absolute path, hence the use of pwd
  qmcmd = "cd ./%s && SCR_ROOT=`pwd` && %s %s > %s 2> qm_stderr.out" % (qmdir, os.path.join(GAMESS_PATH, 'rungms'), qminp, qmout)
  assert os.system(qmcmd) == 0, "Error running QM program: " + qmcmd
  mol_cc = cclib_open(qmdir + qmout)

  if cleanup:
    try:
      os.remove(qmdir + qminp)
      os.remove(qmdir + qmout)
    except: pass

  return mol_cc


def gamess_header(basis='6-31G**', grad=True, cart=None, charge=0, local='none', guess='huckel'):
  """ generate a GAMESS header based on options """
  if basis[1] == '-':
    cart = True if cart is None else cart
    opts = ['NGAUSS=' + basis[0]]
    if basis[-1] == '*':
      opts.append('NPFUNC=1')
      basis = basis[:-1]
    if basis[-1] == '*':
      opts.append('NDFUNC=1')
      basis = basis[:-1]
    assert basis[-1].upper() == 'G', "Unknown Pople basis set"
    basis = basis[:-1]
    if basis[-1] == '+':
      opts.append('DIFFSP=1')
      basis = basis[:-1]
    if basis[-1] == '+':
      opts.append('DIFFS=1')
      basis = basis[:-1]
    gbasis = ('N%s ' % basis[2:]) + ' '.join(opts)
  else:
    assert cart != True, "Basis set may require ISPHER=1"
    gbasis = basis

  # GAMESS doesn't accept lines longer than ~80 chars; ISKPRP=1 skips population analysis
  header = """ $CONTRL
   SCFTYP=RHF RUNTYP={runtyp}
   ICHARG={charge} MULT=1
   ISPHER={ispher} LOCAL={local}
   NPRINT=7 ISKPRP=1 MAXIT=100
 $END
 $SYSTEM MWORDS=128 $END
 $BASIS GBASIS={gbasis} $END
 $GUESS GUESS={guess} $END
"""
  return header.format(runtyp=("GRADIENT" if grad else "ENERGY"),
      charge=charge, local=local.upper(), ispher=(-1 if cart else 1), gbasis=gbasis, guess=guess.upper())
