from chem.io import *
from chem.mmtheo import *
from chem.vis.chemvis import *


# very quick example to debug things for system w/ 2 reactants, 2 products: I22 + G3H -> F6R + E4P
# mechanism (?): E4P breaks off from I22, G3H binds to remainder (left bound to LYS), giving F6R
# - https://en.wikipedia.org/wiki/Transaldolase

if not os.path.exists("E4P.mol2"):
  mI22 = load_compound('I22').set_residue('I22')
  mG3H = load_compound('G3H').set_residue('G3H')
  mF6R = load_compound('F6R').set_residue('F6R')
  mE4P = load_compound('E4P').set_residue('E4P')
  # remove H from phosphate groups
  mI22.remove_atoms('* * HO10,HO8')
  mG3H.remove_atoms('* * HOP3,HOP4')
  mF6R.remove_atoms('* * HO1,HO2')
  mE4P.remove_atoms('* * HOP2,HOP3')

  # alternative: use reactmol ... issues w/ geometry, possible atom name conflict (not in this instance)
  # ... is there any reason to worry about atom order outside of unimolecular case?
  if 0:
    mRct = Molecule(mI22)
    mRct.append_atoms(mG3H, r=mG3H.r + (0,8,0))
    C3,C4,O4,HO4 = res_select(mRct, 0, 'C3,C4,O4,HO4')
    C1,O1 = res_select(mRct, 1, 'C1,O1')
    mProd = reactmol(mRct, breakbonds=[(C3,C4), (O4,HO4)], makebonds=[(C1,C3), (O1,HO4)])

    mF6R = mProd.extract_atoms(mProd.get_connected(40)).set_residue('F6R')  #mF6R.residues[0].name = 'F6R'
    mE4P = mProd.extract_atoms(mProd.get_connected(1)).set_residue('E4P')  #mE4P.residues[0].name = 'E4P'

    # straighten products
    bb = res_select(mF6R, 0, 'P,O1P,C4,C5,C6,C3,C2,C1')
    for ii in range(len(bb) - 3):
      mol.dihedral(bb[ii:ii+4], newdeg=-180)  ## match angles of I22!

    #bb = res_select(mE4P, 0, 'P,O1P,C4,C5,C6,C3,C2,C1')

  antechamber_prepare(mI22, 'I22', netcharge=-2)
  antechamber_prepare(mG3H, 'G3H', netcharge=-2)
  antechamber_prepare(mF6R, 'F6R', netcharge=-2)
  antechamber_prepare(mE4P, 'E4P', netcharge=-2)


T0 = 300
host, ligin, ligout, ligall = '/A//', '/I//', '/O//', '/I,O//'

ff = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'I22.ff.xml', 'G3H.ff.xml', 'F6R.ff.xml', 'E4P.ff.xml')
ffgbsa = OpenMM_ForceField('amber99sb.xml', 'tip3p.xml', 'I22.ff.xml', 'G3H.ff.xml', 'F6R.ff.xml', 'E4P.ff.xml',
    'implicit/obc1.xml', nonbondedMethod=openmm.app.CutoffNonPeriodic)
ffp = OpenMM_ForceField(ff.ff, ignoreExternalBonds=True)
ffpgbsa = OpenMM_ForceField(ffgbsa.ff, ignoreExternalBonds=True)
ffpgbsa10 = OpenMM_ForceField(ffgbsa.ff, nonbondedMethod=openmm.app.CutoffNonPeriodic, ignoreExternalBonds=True)

if os.path.exists("TALwt.zip"):
  molwt_AI, molwt_AO = load_zip("TALwt.zip", 'molwt_AI', 'molwt_AO')
else:
  tal_pdb = load_molecule('3TK7', bonds=False, download=False)  #=True
  mol = tal_pdb.extract_atoms("protein and chain == 'A'")  # active sites are at interface of chains
  #mol.remove_atoms('*/HIS/HD1')
  mol = add_hydrogens(mol)

  # P, C6 (closest to P), C1 (furthest from P)
  rref = tal_pdb.r[ res_select(tal_pdb, '/A/501/*', 'P,C6,C4') ]

  mI22 = load_molecule('I22.mol2').set_residue('I22', chain='I')
  mI22 = align_mol(mI22, rref, sel=res_select(mI22, 0, 'P1,C3,C1'), sort=None)
  molwt_AI = Molecule(mol)
  molwt_AI.append_atoms(mI22)
  molwt_AI.r = molwt_AI.r - np.mean(molwt_AI.extents(), axis=0)  # move to origin - after appending lig aligned to PDB pos!
  res, molwt_AI.r = moloptim(OpenMM_EandG(molwt_AI, ffgbsa), molwt_AI, maxiter=100)

  mF6R = load_molecule('F6R.mol2').set_residue('F6R', chain='O')
  mF6R = align_mol(mF6R, rref, sel=res_select(mF6R, 0, 'P,C6,C4'), sort=None)  #P1 for reactmol() version
  molwt_AO = Molecule(mol)
  molwt_AO.append_atoms(mF6R)
  molwt_AO.r = molwt_AO.r - np.mean(molwt_AO.extents(), axis=0)  # move to origin - after appending lig aligned to PDB pos!
  res, molwt_AO.r = moloptim(OpenMM_EandG(molwt_AO, ffgbsa), molwt_AO, maxiter=100)

  save_zip('TALwt.zip', molwt_AI=molwt_AI, molwt_AO=molwt_AO)

  openmm_load_params(molwt_AI, ff=ff)
  protonation_check(molwt_AI)
  badpairs, badangles = geometry_check(molwt_AI)




## setup LYS bound to fragment of LIG remaining after first step

# parameterize the non-standard residue (excluding backbone)
if not os.path.exists("IMD1.mol2"):
  mLYS = PDB_RES('LYS')
  mLYS.remove_atoms("not sidechain or name in ['HA', 'HZ2', 'HZ3']")
  CA = res_select(mLYS, 0, 'CA')[0]
  mLYS.atoms[CA].znuc = 1
  mLYS.atoms[CA].name = 'HCA'

  mol = Molecule(mI22)
  mol.append_atoms(mLYS)
  C2,C3,C4,O2 = res_select(mol, 0, 'C2,C3,C4,O2')
  NZ = res_select(mol, 1, 'NZ')[0]
  C3frag, C4frag = mol.partition(C3, C4)
  mol.add_bonds((NZ,C2))  #mol = reactmol(mol, breakbonds=[(C3,C4)], makebonds=[(NZ,C2)])
  mol.remove_atoms(C4frag + [O2])
  mol = reactmol(mol, [], [])
  mol.set_residue('IMD')

  antechamber_prepare(mol, 'IMD1', netcharge=0)


if not os.path.exists("IMD2.xyz"):
  # combine with backbone
  tmpff = OpenMM_ForceField('IMD1.ff.xml')
  imd0 = load_molecule('IMD1.mol2').set_residue('IMD')  #, chain='I')
  openmm_load_params(imd0, tmpff)

  # hack to get ExternalBond fields
  mol = add_residue(Molecule(), 'GLY', -180, 135)
  mol = add_residue(mol, 'LYS', 180, 180)
  mol = add_residue(mol, 'GLY', -135, -180)

  # get mmtypes
  tmpffp = OpenMM_ForceField(DATA_PATH + '/ff99SB.xml')  #, ignoreExternalBonds=True)
  top = openmm_top(mol)
  sysdata = tmpffp.ff._SystemData(top)
  tmpffp.ff._matchAllResiduesToTemplates(sysdata, top, {}, False)  #True)
  for ii,a in enumerate(sysdata.atoms):
    mol.atoms[ii].mmtype = sysdata.atomType[a]
  openmm_load_params(mol, tmpffp)

  CA,CB = res_select(mol, 1, 'CA,CB')
  CAfrag,CBfrag = mol.partition(CA, CB)
  # align to sidechain before replacing
  molsc = res_select(mol, 1, 'CA,CB,CG')
  imdsc = res_select(imd0, 0, 'HCA,CB,CG')
  imd0.r = apply_affine(alignment_matrix(imd0.r[imdsc], mol.r[molsc]), imd0.r)

  mol.append_atoms(imd0, residue=1)
  mol.residues[1].name = 'IMD'
  HCA = mol.select("name == 'HCA'")[0]
  mol.add_bonds((CA, mol.atoms[HCA].mmconnect[0]))
  mol.remove_atoms(CBfrag + [HCA])
  #assert len(mol.name) == len(unique(mol.name)), "Atom name conflict!"

  #mol.r = reactmol(Molecule(mol), [], [], relax=100).r  -- we now align before replacing instead
  # make sure we didn't change it to D-amino acid
  rC,rCA,rN,rHA = mol.r[ res_select(mol, 1, 'C,CA,N,HA') ]
  # > 0 for L-amino acid (normal), < 0 for D-amino acid
  assert np.dot(np.cross(rC - rCA, rN - rCA), rHA - rCA) > 0, "We need L-amino acid, not D-amino acid!"

  #print(openmm_resff(mol, [1], gennames=False, extbonds=True))
  write_file('IMD2.ff.xml', openmm_resff(mol, [1], gennames=False, extbonds=True))
  write_xyz(mol.extract_atoms(resnums=1), 'IMD2.xyz')

imd = load_molecule('IMD2.xyz')
#mol = Molecule(molwt_AI)
#mol.remove_atoms(ligall)
#mol = mutate_residue(mol, '/A/135/*', imd, align=True)
ff2 = OpenMM_ForceField(DATA_PATH + '/ff99SB.xml', DATA_PATH + '/gaff.xml', 'tip3p.xml', 'IMD2.ff.xml')

mol = add_residue(Molecule(), 'GLY', -180, 135)
mol = add_residue(mol, imd, 180, 180)
mol = add_residue(mol, 'GLY', -135, -180)

openmm_load_params(mol, ff2, charges=True, vdw=True, bonded=True)


# start interactive
import pdb; pdb.set_trace()

mm_stretch = frozenset( tuple(b[0]) for b in mol.mm_stretch )
mm_bend = frozenset( tuple(b[0]) for b in mol.mm_bend )
mm_torsion = frozenset( tuple(b[0]) for b in mol.mm_torsion )

bonds, angles, diheds = mol.get_internals()
missing_bonds = [b for b in bonds if b not in mm_stretch]
missing_angles = [b for b in angles if b not in mm_bend]
missing_diheds = [b for b in diheds if b not in mm_torsion]

AMBER_TO_GAFF = dict(s.strip().split(',') for s in AMBER_TO_GAFF_STR.split(';'))
# ... seems like the easiest approach would be to modify the generators in openmm ForceField to try using this
#  if no match found
# ... or rather than creating .ff.xml for the modified residue, try ForceField.registerTemplateGenerator()!

#ctx = openmm_MD_context(mol, ff2, T0)


## ---

#dEwt_AI = dE_binding(molwt_AI, ffgbsa, host, ligin)  # -9.83327764269541 ... obviously not bound well
#dEwt_AO = dE_binding(molwt_AO, ffgbsa, host, ligout)  # -63.66996450891943

ligreswt = molwt_AI.atomsres(ligin)[0]
nearres7 = molwt_AI.atomsres(molwt_AI.get_nearby(ligin, 7.0, active='protein'))
pcenter = None

mol0 = molwt_AI
molp0 = mol0.extract_atoms(resnums=nearres7 + [ligreswt])
nearres4p = molp0.atomsres(molp0.get_nearby(ligin, 4.0, active='sidechain'))  # 4 Ang
freeze = 'protein and (extbackbone or resnum not in %r)' % nearres4p
mol = prep_solvate(molp0, ffp, pad=6.0, center=pcenter, solvent_chain='Z', neutral=True)
mol = prep_relax(mol, ffp, T0, freeze=freeze, eqsteps=5000)
ctx = openmm_MD_context(mol, ffp, T0, v0=mol.md_vel, freeze=freeze)
Rs, Es = openmm_dynamic(ctx, 10000, 100)
dEs = dE_binding(mol, ffpgbsa, host, ligin, Rs)  # -38.8(3.5) for molwt_AI ... much better

save_zip('TALwt_solv.zip', mol=mol)


NZ, HZ2, HZ3, C2, O2, C3, C4 = mol.select('/A/135/NZ;/A/135/HZ2;/A/135/HZ3;/I/*/C2;/I/*/O2;/I/*/C3;/I/*/C4')  # LYS NZ,HZ2,HZ3;I22 C2 (C=O)
mol.remove_atoms([HZ2, HZ3, O2])  # HOH eliminated
mProd = reactmol(mol, breakbonds=[(C3,C4)], makebonds=[(NZ,C2)])

# Use GAFF for bound LYS? ... convieniently isolated, so this might work ... but we'd want to truncate ... replace CA with H? with CH3?
# ... should be done with full protein and allowed to relax!

# better to treat LYS + fragment as a residue

# using openmm_create_system:
# - given plain protein and bound residue as two separate Molecules, w/ MM params ...
#  - CA replaced w/ CH3 in bound residue
# - create mapping between plain residue atoms and bound residue atoms
# - iterate over MM terms containing overlapping atoms: if two MM terms present, pick one
# - add the bound residue atoms

# using openmm_resff: how would we handle interface between GAFF and Amber?
# - ideal would be params for GAFF - AMBER interface ... which direction should we map?  Multiple AMBER types might map to a single GAFF type. If we use GAFF params, we'd iterate over GAFF terms and output one for each equiv AMBER type

# we'd be using full residue, with mixed GAFF and AMBER types



#~ nearres11 = sorted(set(nearres7)
#~     | set(molwt_AI.atomsres(molwt_AI.get_nearby(molwt_AI.resatoms(nearres7), 4.0, active='protein'))))

#~ molpwt_AI = molwt_AI.extract_atoms(resnums=nearres11 + [ligreswt])

#~ nearres4p = molpwt_AI.atomsres(molpwt_AI.get_nearby(ligin, 4.0, active='sidechain'))  # 4 Ang
#~ nearres7p = molpwt_AI.atomsres(molpwt_AI.get_nearby(ligin, 7.0, active='protein'))

#~ active = 'not extbackbone or resnum in %r' % nearres7p


## Autodock Vina scoring fn; refs:
# * github.com/ttjoseph/mmdevel/blob/master/VinaScore/vinascore.py
# * github.com/ccsb-scripps/AutoDock-Vina/
# * www.ncbi.nlm.nih.gov/pmc/articles/PMC3041641/
def Evina(mol, selA=None, selB=None, r=None):
  """ Compute Autodock Vina scoring fn for `mol` (w/ optional `r`); only between sets of atoms `selA`, `selB`
    if specified
  """
  VINA_VDW = {6: 1.9, 7: 1.8, 8: 1.7, 9: 1.5, 15: 2.1, 16: 2.0, 17: 1.8}

  # only heavy atoms
  znuc = mol.znuc
  heavy = znuc > 1
  znuc = znuc[heavy]
  r = mol.r[heavy] if r is None else r[heavy]
  dr = r[:,None,:] - r[None,:,:]
  dd = np.sqrt(np.sum(dr*dr, axis=2))
  np.fill_diagonal(dd, np.inf)
  dd[dd > 8.0] = np.inf
  vdw = np.array([VINA_VDW[z] for z in znuc])
  dd = dd - vdw[:,None] - vdw[None,:]

  if selA is not None and selB is not None:
    maskA = setitem(np.zeros(mol.natoms), mol.select(selA), 1.0)[heavy]
    maskB = setitem(np.zeros(mol.natoms), mol.select(selB), 1.0)[heavy]
    assert not np.any(maskA*maskB), "selA, selB must be disjoint"
    dd[ maskA[:,None]*maskB[None,:] + maskA[None,:]*maskB[:,None] == 0 ] = np.inf  # make symmetric
  else:
    bonds, angles, diheds = mol.get_internals()
    pairs = np.array(bonds + [(a[0], a[2]) for a in angles] + [(a[0], a[3]) for a in diheds])
    mask = np.full((mol.natoms, mol.natoms), False)
    mask[pairs[:,0], pairs[:,1]] = True
    mask[pairs[:,1], pairs[:,0]] = True
    dd[ mask[heavy,:][:,heavy] ] = np.inf

  hphob = (znuc == 6) | (znuc == 9) | (znuc == 17)  # C,F,Cl,Br,I
  hdonor = np.array([a.znuc in [7,8] and any(mol.atoms[jj].znuc == 1 for jj in a.mmconnect) for a in mol.atoms])[heavy]
  haccept = ((znuc == 7) | (znuc == 8)) & ~hdonor
  hbond = (hdonor[:,None]*haccept[None,:]) | (hdonor[None,:]*haccept[:,None])  # make symmetric

  # terms
  Egauss1 = -0.0356*np.sum( np.exp(-(dd/0.5)**2) )
  Egauss2 = -0.00516*np.sum( np.exp(-((dd-3.0)/2.0)**2) )
  Erepuls = 0.840*np.sum( np.clip(dd, None, 0)**2 )
  Ehphob = -0.0351*np.sum( hphob[:,None]*hphob[None,:]*np.clip((1.5-dd), 0.0, 1.0) )
  Ehbond = -0.587*np.sum( hbond*np.clip(-dd/0.7, 0.0, 1.0) )

  return 0.5*(Egauss1 + Egauss2 + Erepuls + Ehphob + Ehbond)


mol = load_molecule("/home/mwhite/qc/work/2016/vinyl/cov1.xyz")
mol.set_bonds(guess_bonds(mol))
Evina(mol)
#Evina(mol, '* LIG *', '* GLU *')
