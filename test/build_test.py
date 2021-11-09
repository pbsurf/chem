from chem.model.build import *
from chem.vis.chemvis import *
from chem.io.openmm import *
from chem.model.prepare import *
from chem.opt.optimize import *
from chem.mm import *

# Use http://ligand-expo.rcsb.org/ld-search.html to search for ligands
ff = openmm_app.ForceField('amber99sb.xml', 'tip3p.xml')

# start interactive
import pdb; pdb.set_trace()

if 1:
  # compare params loaded from openmm vs. params loaded from AMBER data files
  mol = add_residue(Molecule(), 'GLY_LFZW', -180, 135)
  mol = add_residue(mol, 'TRP_LFZW', 180, 180)
  mol = add_residue(mol, 'GLY_LFZW', -135, -180)
  mol2 = Molecule(mol)

  openmm_load_params(mol2, ff=ff, vdw=True, bonded=True)

  DAT = DATA_PATH + '/amber/'
  parm = load_amber_dat(DAT + 'parm99.dat')
  p99sb = load_amber_frcmod(DAT + 'frcmod.ff99SB')
  parm.torsion = dict(parm.torsion, **p99sb.torsion)  # in general we'd need to merge stretch, bend, etc. too
  reslib = load_amber_reslib(DAT + 'all_amino94.lib', DAT + 'all_aminont94.lib', DAT + 'all_aminoct94.lib')
  set_mm_params(mol, parm, reslib)


if 0:
  # compare openmm and SimpleMM
  ommcomp = {}
  Eomm, Gomm = openmm_EandG(mol, ff=ff, components=ommcomp)

  openmm_load_params(mol, ff=ff, vdw=True, bonded=True)
  comp = {}
  Emm, Gmm = SimpleMM(mol, ncmm=NCMM(mol, qqkscale=OPENMM_qqkscale))(components=comp)

  # nothing helpful here
  Enb = 0.5*np.sum(comp['Eqq'], axis=1) + 0.5*np.sum(comp['Evdw'], axis=1)
  for ii in np.argsort(Enb):
    print("%4d %3d (%s) %f" % (ii, mol.atoms[ii].resnum, pdb_repr(mol, ii), Enb[ii]))


if 1:
  mol = load_molecule('1HPI.pdb', bonds=False)
  mol = add_hydrogens(mol)

  #mol.r = mol.r - np.mean(mol.r, axis=0)
  #vis = Chemvis(Mol(mol, [ VisVectors(10.0), VisGlobe() ]), lock_orientation=True).run()

  vis = Chemvis(Mol(mol, [ VisGeom(style='licorice', sel='protein'), VisGeom(style='spacefill', sel='znuc == 26 or znuc == 16'), VisBackbone(style='tubemesh', coloring=color_by_resnum, color_interp='ramp'), VisGlobe() ]), lock_orientation=True).run()

  mol.remove_atoms('* SF4,HOH *')
  openmm_load_params(mol, ff=ff, charges=True, vdw=True)


if 0:
  mol = load_molecule('3AU1.pdb', bonds=False)
  mol = add_hydrogens(mol)
  # fixes
  mol.remove_atoms('* * O1,HO1; A 20,42,165 HD22')  # manually fix up glycans
  # I think this is cleaner than trying to check in add_hydrogens()
  badvalence = lambda a: (a.znuc == 8 and len(a.mmconnect) > 2) or (a.znuc == 6 and len(a.mmconnect) > 4)
  extraH = [ [ii for ii in a.mmconnect if mol.atoms[ii].znuc == 1] for a in mol.atoms if badvalence(a) ]
  mol.remove_atoms(set(flatten(extraH)))

  vis = Chemvis(Mol(mol, [ VisGeom(style='licorice', sel='/{}/*/*'.format(c)) for c in sorted(list(set(mol.chain))) ] + [ VisBackbone(style='tubemesh', coloring=color_by_resnum, color_interp='ramp') ]), wrap=False).run()

  badpairs, badangles = geometry_check(mol)
  for pair in badpairs:
    vis.select(pair).set_view(center=True)  #dist=10
    input("")  # wait for Enter key


if 1:
  mol = load_molecule('../../2016/trypsin1/1MCT.pdb', bonds=False)
  mol = add_hydrogens(sort_chains(mol))  # openmm requires residues on chain be continguous
  #vis = Chemvis(Mol(mol, [ VisBackbone(style='tubemesh', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='spacefill') ]), wrap=False).run()

  mol.remove_atoms('not protein')
  ctx = openmm_EandG_context(mol, ff)
  h_atoms = mol.select('znuc == 1')
  res, r = moloptim(openmm_EandG, mol=mol, fnargs=Bunch(ctx=ctx), coords=XYZ(mol, h_atoms))
  badpairs, badangles = geometry_check(mol)
  protonation_check(mol)

  openmm_load_params(mol, ff=ff, charges=True, vdw=True)
  #vis = Chemvis(Mol(mol, [ VisGeom(style='spacefill', coloring=scalar_coloring('mmq', [-1,1])) ])).run()
  vis = Chemvis(Mol(mol, [ VisBackbone(style='tubemesh', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='licorice'), VisContacts(partial(NCMM_contacts, Ethresh=-40.0/KCALMOL_PER_HARTREE), style='lines', dash_len=0.1, colors=Color.light_grey) ]), wrap=False).run()



# examine all AA variants for, e.g., ALA
if 0:
  pdbres = PDB_RES().keys()
  for k in pdbres:
    PDB_RES()[k].filename = PDB_RES()[k].header
  alares = [res for res in pdbres if res.startswith('GLU')]
  vis = Chemvis(Mol([PDB_RES()[res] for res in alares], [ VisGeom(style='licorice') ]), wrap=False, verbose=True).run()

if 0:
  mol = PDB_RES('ALA_LFZW')  # zwitterionic version of residue
  mol = add_residue(mol, 'ALA_LFZW', -57, -47)
  mol = add_residue(mol, 'ALA_LFZW', -57, -47)
  mutate_residue(mol, 1, 'GLU_LFZW_DHE2')

  mol2 = PDB_RES('ALA_LFZW')  # zwitterionic version of residue
  mol2 = add_residue(mol2, 'GLU_LFZW_DHE2', -57, -47)
  mol2 = add_residue(mol2, 'ALA_LFZW', -57, -47)

  vis = Chemvis(Mol([mol, mol2], [ VisGeom(style='licorice') ]), wrap=False, verbose=True).run()

if 0:
  mol = PDB_RES('ALA_LFZW')  # zwitterionic version of residue
  for ii in range(9):
    mol = add_residue(mol, 'ALA_LFZW', -57, -47)

  print(np.array(get_bb_angles(mol))/np.pi*180)

if 0:
  mol = PDB_RES('ALA_LFZW')  # zwitterionic version of residue
  #mol = add_residue(mol, 'ALA_LFZW', -57, -47)
  mol = add_residue(mol, 'NME', -57, -47) # this will give +1 next charge; use GLY_LFZW to get zero charge

  vis = Chemvis(Mol(mol, [ VisBackbone(style='tubemesh', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='licorice') ]), wrap=False).run()

  openmm_load_params(mol, ff=ff, charges=True, vdw=True)


if 0:
  # test set_rotamer()
  trp = add_residue(Molecule(), 'GLY_LFZW', -180, 135)
  trp = add_residue(trp, 'TRP_LFZW', 180, 180)
  trp = add_residue(trp, 'GLY_LFZW', -135, -180)

  rot0 = align_mol(set_rotamer(Molecule(trp), 1, [52.81, 271.06]), tink0, 'backbone')
  rot8 = align_mol(set_rotamer(Molecule(trp), 1, [293.22, 99.9]), tink8, 'backbone')


if 0:
  # compare bonds in PDB_RES to PDB_BONDS
  def get_pdb_bonds(res, atom):
    mol = PDB_RES()[res]
    atomobj = next((a for a in mol.atoms if a.name == atom), None)
    return [] if atomobj is None else [mol.atoms[b].name for b in atomobj.mmconnect]

  # compare PDB_BONDS and PDB_RES
  from chem.data.pdb_bonds import PDB_BONDS

  for res in PDB_BONDS.keys():
    if res in PDB_RES():
      for a in PDB_BONDS[res]:
        old, new = sorted(PDB_BONDS[res][a]), sorted(get_pdb_bonds(res + '_LFZW', a))
        if old != new:
          print("%s %s: %s != %s" % (res, a, old, new))
    else:
      print("%s not in PDB_RES" % res)


# might be helpful to make our code work w/ other molecule objects, not just our own ...
class OpemMM_Mol:
  def __init__(self, ctx, top):
    self.ctx, self.top = ctx, top

  def __getattr__(self, attr):
    if attr == 'r':
      simstate = self.ctx.getState(getPositions=True, enforcePeriodicBox=True)
      return simstate.getPositions(asNumpy=True).value_in_unit(UNIT.angstrom)
    if attr == 'znuc':
      return np.array([a.element.atomic_number for a in self.top.atoms()])
    if attr == 'natoms':
      return self.top.getNumAtoms()
    if attr == 'nresidues':
      return self.top.getNumResidues()
    if attr == 'bonds':
      return [(b.atom1.index, b.atom2.index) for b in self.top.bonds()]
    if attr == 'atoms':
      pass #??? ... make mol.znuc, .r, etc. efficient and remove use of .atoms wherever possible?
      # return object which can access r, znuc, etc?
    raise AttributeError(attr)
