from chem.vis.chemvis import *

## Goals
# - render molecules with controllable transparency and greyscale (to show current and previous geom)
# - provide some basic selection functions so we can render backbone, substrates, QM regions, etc. differently
# - play movie of geom opt steps
# - render MO or combined MOs (e.g. total electron density) as volume or isosurfaces
# Notes:
# * color format is (R,G,B,A); 0-255; conversion to OpenGL format (0.0-1.0) done when necessary
# * objects from external libraries such as cclib and pyscf will be stored in attributes of Molecule object


## Cleanup tasks
# - shadows: orthographic projection for cylinder imposters
# - rename Chemvis.children to .mols (?)
# - should shaders be class variables instead of instance variables? - we'll have to get rid of self.modules stuff!
# - we should probably pass colors to a fn to generate color_by_ rather combining in Vis* constructors!
# - poisson disk as uniform array; move to glutils.py


## Performance
# - try `pip install pyopengl-accelerate`
# - biggest remaining issue in mol_renderer is collecting znuc in color_by_element() (and mol.znuc in set_molecule)

# Ideas:
# - accessing mol.r, .znuc, etc, converts Molecule to use these and hides .atoms (._atoms will be the actual array); accessing .atoms deletes them
#  - mark as read-only (numpy) and save ids to check if they've been replaced (in which case, write back to atoms)
# - Atom is object which accesses .r, .znuc, etc.

# - VisBackbone.set_ribbon_data() is slow (200 ms per load for trypsin) - try line_profiler; see if we can get same quality with larger max_cosine_axis/_orient; would fixed step size allow us to vectorize?
# - more complete caching in mol_renderer so we only need to update positions VBO (when just changing r)?


## Todo
# - text rendering ... stb_truetype can now generate SDF directly!
#  - add easy way to show and persist a distance or angle measurement
# - simple GUI? or even just printing some info about key bindings?
# - supersampling?
# - molecular surfaces? ... could try hyperball shader from ngl?  depth peeled sphere rendering?
# - need a better alternative to RenderConfig.shading global
# - use mouse wheel to control fog?
# - animation for set_view()
# - freezing a molecule from list: wrap renderer list with lambda so duplicate set can be created?
# - transparency: opacity param for Mol() and global alpha in shading.py?  See notes in mol_renderer.py
# - option to draw wireframe for (non-imposter) geometry? (barycentric approach)
# - method of Mol or Chemvis to add or replace a renderer and/or single fn to change attributes and refresh
# - decide upon and apply a uniform naming convention for GLSL variables!
# - switch from GLFW to SDL2 to support touch input (sadly not quite working on linux with vmware) - https://github.com/marcusva/py-sdl2
# - any better way to handle drawing selection spheres to improve visibility with volumes?
# - option for gradient background
# - try shading LineRenderer like StickRenderer
# - progressive enhancement for volume rendering?  maybe try adaptive step size (for fog) first!
# - idea: save and restore views (positions 0-9 vs. undo stack?)
# - capped cylinder renderer (for e.g. disulfides in backbone)
# - fancier shading: geometry has "material" with lighting properties, instead of just a color
# - try bump mapping with noise in shading module
# - how might we visualize results of seq+structure alignment ... color residues green, yellow, or red?
# - better visualization of proline with extbackbone + sidechain?
# - camera: rewrite; eliminate c vector and compute from pivot and position?
# - maybe cycling through views should be user defined, i.e., we pass a list of Vis* objects in place of a single one
# - attach notes/annotations to parts of molecule (we could save as PDB REMARKs) (credit to jolecule.com); can
#  also serve as links to saved views of molecule
# - any reason to try GL_DEPTH_COMPONENT32F and/or inverted depth buffer?

# How to handle selection with list of molecules?
#  - option to run selection fn for each molecule?
#  - store selection per-molecule?  In Molecule or make Chemvis.selection a dict?

# playing with 1Y0L: color each chain with gradient of different color?  easy way to only draw atoms within some radius of selection or other point?

# Logic for input too varied to do something like
#viewer.register_var("Volume absorptivity: %0.3f", self.vol_obj, 'absorption', keys, scale=1.25, reset=10.0)
# so instead maybe try this:
#keys = dict(mo_number=";'", ray_steps='Ctrl+-=', isolevel='-=|Bksp', absorption='[]|\\', method='V')
#def on_input(self, viewer, var, step):  # step = +/- 1 or 10, or 'reset'


## Examples
# Chemvis accepts a list of molecule objects, which optionally can be loaded on demand from a glob using a
#  Files object.  Visualization objects (Vis*) can cache computed values in the molecule objects
if 0:
  Chemvis(Mol(Files('*-tinker.pdb'), [
    VisGeom(style='spacefill', coloring=coloring_opacity(color_by_element, 0.5)) ]) ).run()

  Chemvis(Mol(Files('*-tinker.pdb'), [
    VisGeom(style='licorice', sel='backbone'), VisGeom(style='lines', sel='sidechain'),
    # select residues having any atom within 10 Ang of atom 3250
    VisGeom(style='spacefill', sel='any(mol.dist(a, 3250) < 10 for a in resatoms)') ]) ).run()

  Chemvis(Mol(Files('../QMMM_1/pureQM_qm*.log'), [
    VisGeom(style='licorice'),
    VisVol(cclib_mo_vol, vis_type='iso', vol_type="MO Volume"),
    VisVol(cclib_lmo_vol, vis_type='iso', vol_type="LMO Volume") ]) ).run()

  # laplacian of electron density - "atoms in molecules"
  Chemvis(Mol(Files('../QMMM_1/pureQM_qm010.log'), [
    VisGeom(style='licorice'),
    VisVol(cclib_dens_vol, vis_type='volume', vol_type="Electron Density", postprocess=laplacian, iso_step=0.001)
  ]), bg_color=Color.white).run()

  # show only valence electron density ... I'm not sure this is useful or interesting
  # absorption style + heat_colormap works well with black BG, unlike blending + red/blue
  Chemvis(Mol(Files('../QMMM_1/pureQM_qm010.log'), [
    VisGeom(style='licorice'),
    VisVol(partial(cclib_dens_vol, mos='valence'), vis_type='absorption', colormap=heat_colormap(256), vol_type="Density Volume", timing=True)
  ]), bg_color=Color.black).run()

  # two QM/MM
  Chemvis([
    Mol(Files('ethanol/*qm.log'), [ VisVol(cclib_mo_vol, vol_type="MO Volume", vis_type='iso') ]),
    Mol(Files('ethanol/*mm.xyz'), [ VisGeom(style='licorice') ]),
    Mol(Files('ethanol/*qm2.log'), [ VisVol(cclib_mo_vol, vol_type="MO Volume", vis_type='iso') ]),
    Mol(Files('ethanol/*mm2.xyz'), [ VisGeom(style='licorice') ])
  ]).run()

  # Files(['1CE5.xyz', '1CE5-min.xyz'], charges='generate')
  vis = Chemvis([
    Mol([load_molecule('1CE5.xyz', charges='generate')], [
      VisGeom(style='lines'),
      VisContacts(style='lines', dash_len=0.1, colors=Color.light_grey) ]),
    Mol([load_molecule('1CE5-tinker.pdb')], [
      VisBackbone(style='tube', atoms=['CA'], coloring=color_by_residue, color_interp='step') ]) ],
    bg_color=Color.black).run()

  # ball and stick
  vis = Chemvis(Mol(Files('1CE5', hydrogens=True), [VisGeom(style='licorice', sel='protein', stick_radius=0.35, coloring='carbonchain') ]), fog=True).run()

  # ball and stick, solid color sticks, no shading
  vis = Chemvis(Mol(Files('1CE5', hydrogens=True), [VisGeom(style='ballstick', sel='protein', coloring='carbonchain') ]), shading=LightingShaderModule(shading='none', outline_strength=8.0), bg_color=Color.white).run()

  # cartoon shading/"molecule-of-the-month" - e.g. https://pdb101.rcsb.org/motm/213 (typically no hydrogens)
  vis = Chemvis(Mol(Files('1CE5'), [VisGeom(style='spacefill', sel='protein and znuc > 1', coloring='motm') ]), shading=LightingShaderModule(shading='none', outline_strength=4.0), effects=[AOEffect(nsamples=70)], bg_color=Color.white).run()

  # ambient occlusion
  Chemvis(Mol(Files('1CE5.pdb', hydrogens=True), [ VisGeom(style='spacefill') ]),
    effects=[AOEffect(nsamples=70)], shadows=True).run()

  # Cutinase - esterase w/ classic SER, HIS, ASP catalytic triad
  vis = Chemvis(Mol(Files('1CEX', hydrogens=True), [ VisBackbone(style='tubemesh', disulfides='line', coloring=color_by_resnum, colors=None, color_interp='ramp'), VisGeom(style='lines', sel='extbackbone'), VisGeom(style='lines', sel='sidechain') ]), fog=True).run()
  vis.select("/A/120,175,188/~C,N,CA,O,H,HA")  # catalytic triad sidechains

  # pseudo-cartoon
  # see github.com/boscoh/pyball/blob/master/pyball.py to get started w/ real cartoon
  mol = add_hydrogens(load_molecule('2CHT.pdb').extract_atoms('/A,B,C//'))
  ss = secondary_structure(mol)
  ssrad = {'': [0.2, 0.2], 'H': [0.5, 0.0625], 'G': [0.5, 0.0625], 'E': [0.8, 0.1]}
  radfn = lambda mol, idx, r: np.array(ssrad[ss[mol.atoms[idx].resnum]])
  vis = Chemvis(Mol(mol, [VisBackbone(style='tubemesh', disulfides='line', coloring='resnumchain', radius_fn=radfn), VisGeom(sel='protein', coloring='carbonchain'), VisGeom(style='spacefill', sel=ligin)]), fog=True).run()

  # align each molecule to a reference
def test_trypsin_1():
  ref_mol = load_molecule('1CE5.pdb', hydrogens=True)
  align_fn = lambda mol: align_mol(mol, ref_mol, sel='pdb_resnum in [57, 102, 189, 195] and sidechain and atom.znuc > 1', sort='pdb_resnum, atom.name')
  return Chemvis(Mol(Files('????.pdb', hydrogens=True, postprocess=align_fn), [ VisBackbone(style='tubemesh', coloring=color_by_resnum, color_interp='ramp'), VisGeom(style='lines', sel='extbackbone'), VisGeom(style='lines', sel='sidechain'), VisGeom(style='lines', sel='pdb_resnum in [57, 102, 189, 195]')]), bg_color=Color.black).run()

def test_pyscf_1(mol=None):
  # pyscf MOs
  from pyscf import gto, scf
  import chem.data.test_molecules as test_molecules

  mol = test_molecules.ethanol_old_oplsaa_xyz if mol is None else mol
  mol = load_molecule(mol) if type(mol) is str else mol
  mol_gto = gto.M(atom=[ [ELEMENTS[a.znuc].symbol, a.r] for a in mol.atoms ], basis='6-311g**', cart=True) #ccpvdz')
  mol.pyscf_mf = scf.RHF(mol_gto).run()
  return Chemvis(Mol([mol], [ VisGeom(style='licorice'), VisVol(pyscf_dens_vol, vis_type='volume', vol_type="Electron density", timing=True) ]), bg_color=Color.white).run()

## Tests

if __name__ == "__main__":
  from test_molecules import *

  mols = [water_tip3p_xyz, water_dimer_tip3p_xyz, C2H3F_noFF_xyz, ethanol_old_oplsaa_xyz]
  Chemvis([ Mol([load_molecule(m) for m in mols], [ VisGeom(style='licorice') ]) ]).run()
