## Source: https://github.com/BioGeek/pairwise-alignment-in-python/blob/master/alignment.py

#This software is a free software. Thus, it is licensed under GNU General Public License.
#Python implementation to Smith-Waterman Algorithm for Homework 1 of Bioinformatics class.
#Forrest Bao, Sept. 26 <http://fsbao.net> <forrest.bao aT gmail.com>
import math
from ..molecule import get_chain, Atom, norm


# Sequence alignment
# - many, many packages, many of which are available in debian repos such as MAFFT, MUSCLE, T-Coffee, Clustal
# - protein sequence alignment should take into account similarity of amino acids (e.g. ASP closer to GLU
#  than LYS); see https://en.wikipedia.org/wiki/Substitution_matrix
# - for proteins, structure should usually be considered in addition to sequence, since sequence actually
#  changes faster than structure.  One package w/ support is T-Coffee - many other packages
#  are mainly for DNA alignment and thus don't support including structural information

if 0:
  mol1 = load_molecule('./2016/trypsin/1CE5.pdb',  hydrogens=lambda s: s[:-4] + '-tinker.pdb')
  mol2 = load_molecule('./2016/trypsin/1GDU.pdb',  hydrogens=lambda s: s[:-4] + '-tinker.pdb')
  mol2 = align_mol(mol2, mol1, sel='pdb_resnum in [57, 102, 189, 195] and sidechain and atom.znuc > 1', sort='pdb_resnum, atom.name')
  # align3d = 20 significantly decreases CA RMSD (1.5 vs. 4) with minimal decrease in identity (47% to 42%)
  mol_seq_align(mol1, mol2, align3d=20.0)


ANSI_COLORS = dict(reset=0, bold=1, disable=2, italic=3, underline=4, bright=5, reverse=7,
  fg=dict(black=30, red=31, green=32, yellow=33, blue=34, purple=35, cyan=36, lightgrey=37,
    darkgrey=90, lightred=91, lightgreen=92, lightyellow=93, lightblue=94, pink=95, lightcyan=96, white=97),
  bg=dict(black=40, red=41, green=42, yellow=43, blue=44, purple=45, cyan=46, lightgrey=47))

def ansi_color(s, style='', fg='', bg=''):
  # note \033 (octal) = \x1b (hex)
  if style or fg or bg:
    codes = [ANSI_COLORS.get(style, style), ANSI_COLORS['fg'].get(fg, fg), ANSI_COLORS['bg'].get(bg, bg)]
    return '\033[' + ';'.join([str(c) for c in codes if c]) + 'm' + s + '\033[0m'
  return s


def named_atom(mol, resnum, name, default=None):
  idxs = [ii for ii in mol.resatoms[resnum] if mol.atoms[ii].name == name]
  return mol.atoms[idxs[0]] if idxs else default


def mol_seq_align(mol1, mol2, chain1=0, chain2=0,
    align3d=False, match_score=10, gap_score=-5, mismatch_score=-5, line_length=120):
  seq1 = range(*get_chain(mol1, chain1))
  seq2 = range(*get_chain(mol2, chain2))
  gap = None

  align3d = 5.0 if align3d is True else align3d
  if align3d:
    # start res_pos arrays from residue 0 even if chain does not start from 0 so that we don't have to mess
    #  deal with converting residue number to index
    default_atom = Atom(r=[0, 0, 0])
    res_pos1 = [named_atom(mol1, ii, 'CA', default_atom).r for ii in range(seq1[-1] + 1)]
    res_pos2 = [named_atom(mol2, ii, 'CA', default_atom).r for ii in range(seq2[-1] + 1)]
    # scoring fn
    def score_fn(a,b):
      if a == gap or b == gap:
        return gap_score
      s3d = align3d/(1.0 + norm(res_pos1[a] - res_pos2[b]))
      return match_score + s3d if mol1.residues[a] == mol2.residues[b] else mismatch_score + s3d
  else:
    def score_fn(a,b):
      return gap_score if a == gap or b == gap else (
          match_score if mol1.residues[a] == mol2.residues[b] else mismatch_score)

  align1, align2 = needleman_wunsch(seq1, seq2, scoring=score_fn, gap=gap)

  # print result
  n_lines = 3 if align3d else 2
  # calculate identity, score and aligned sequences
  lines = [[] for ii in range(n_lines)]
  print1 = []
  sqdists = []
  score, identity = 0, 0
  gap_str = '-'*len(mol1.residues[align1[0]])
  for ii, a1 in enumerate(align1):
    a2 = align2[ii]
    score += score_fn(a1, a2)
    res1 = gap_str if a1 == gap else mol1.residues[a1]
    res2 = gap_str if a2 == gap else mol2.residues[a2]
    if a1 == gap or a2 == gap:
      color = '30;43'  # black on yellow
    elif res1 == res2:
      color = '30;42'  # black on green
      identity = identity + 1
    else:
      color = '41'  # red bg
    # fixed line line printing
    lines[0].append(ansi_color(res1, color))
    lines[1].append(ansi_color(res2, color))
    if align3d:
      if a1 == gap or a2 == gap:
        lines[2].append(' '*len(gap_str))
      else:
        dist = norm(res_pos1[a1] - res_pos2[a2])
        sqdists.append(dist*dist)
        lines[2].append(str(dist)[:len(gap_str)])
    if (len(lines[0]) + 1) * (len(gap_str) + 1) > line_length or ii == len(align1) - 1:
      print1.extend([' '.join(l) for l in lines])
      print1.append('')
      lines = [[] for ii in range(n_lines)]

  identity = float(identity) / len(align1) * 100
  print("Identity = %3.3f%%; Score = %d" % (identity, score))
  if align3d:
    print("Residue RMSD: %.3f" % math.sqrt((sum(sqdists)/len(sqdists))))
  print('\n'.join(print1))


def needleman_wunsch(seq1, seq2, scoring, gap=None):
  m, n = len(seq1), len(seq2) # length of two sequences
  gap_penalty = scoring(seq1[0], gap)

  # Calculate DP table; python list of lists is faster than 2D numpy array
  # note [[0]*(n+1)]*(m+1) doesn't work because list reference is repeated instead of making m+1 new lists
  score = [ [0]*(n+1) for ii in range(m+1) ]
  for i in range(0, m + 1):
    score[i][0] = gap_penalty * i
  for j in range(0, n + 1):
    score[0][j] = gap_penalty * j
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      match = score[i - 1][j - 1] + scoring(seq1[i-1], seq2[j-1])
      delete = score[i - 1][j] + gap_penalty
      insert = score[i][j - 1] + gap_penalty
      score[i][j] = max(match, delete, insert)

  # Traceback and compute the alignment
  align1, align2 = [], []
  i,j = m,n # start from the bottom right cell
  while i > 0 and j > 0: # end toching the top or the left edge
    score_current = score[i][j]
    score_diagonal = score[i-1][j-1]
    score_up = score[i][j-1]
    score_left = score[i-1][j]

    if score_current == score_diagonal + scoring(seq1[i-1], seq2[j-1]):
      align1.append(seq1[i-1])
      align2.append(seq2[j-1])
      i -= 1
      j -= 1
    elif score_current == score_left + gap_penalty:
      align1.append(seq1[i-1])
      align2.append(gap)
      i -= 1
    elif score_current == score_up + gap_penalty:
      align1.append(gap)
      align2.append(seq2[j-1])
      j -= 1

  # Finish tracing up to the top left cell
  while i > 0:
    align1.append(seq1[i-1])
    align2.append(gap)
    i -= 1
  while j > 0:
    align1.append(gap)
    align2.append(seq2[j-1])
    j -= 1

  return align1[::-1], align2[::-1]


def smith_waterman(seq1, seq2, scoring, gap=None):
  m, n = len(seq1), len(seq2) # length of two sequences
  gap_penalty = scoring(seq1[0], gap)

  # Generate DP table and traceback path pointer matrix
  score = [ [0]*(n+1) for ii in range(m+1) ]  # the DP table
  pointer = [ [0]*(n+1) for ii in range(m+1) ]  # to store the traceback path

  max_score = 0    # initial maximum score in DP table
  # Calculate DP table and mark pointers
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      score_diagonal = score[i-1][j-1] + scoring(seq1[i-1], seq2[j-1])
      score_up = score[i][j-1] + gap_penalty
      score_left = score[i-1][j] + gap_penalty
      score[i][j] = max(0,score_left, score_up, score_diagonal)
      if score[i][j] == 0:
        pointer[i][j] = 0 # 0 means end of the path
      if score[i][j] == score_left:
        pointer[i][j] = 1 # 1 means trace up
      if score[i][j] == score_up:
        pointer[i][j] = 2 # 2 means trace left
      if score[i][j] == score_diagonal:
        pointer[i][j] = 3 # 3 means trace diagonal
      if score[i][j] >= max_score:
        max_i = i
        max_j = j
        max_score = score[i][j];

  align1, align2 = [], []  # initial sequences

  i,j = max_i,max_j  # indices of path starting point

  #traceback, follow pointers
  while pointer[i][j] != 0:
    if pointer[i][j] == 3:
      align1.append(seq1[i-1])
      align2.append(seq2[j-1])
      i -= 1
      j -= 1
    elif pointer[i][j] == 2:
      align1.append(gap)
      align2.append(seq2[j-1])
      j -= 1
    elif pointer[i][j] == 1:
      align1.append(seq1[i-1])
      align2.append(gap)
      i -= 1

  return align1[::-1], align2[::-1]


if __name__ == '__main__':
  seq1 = ['[A0]', '[C0]', '[A1]', '[B1]']
  seq2 = ['[A0]', '[A1]', '[B1]', '[C1]']

  print "Needleman-Wunsch"
  needle(seq1, seq2)
  print
  print "Smith-Waterman"
  water(seq1, seq2)


