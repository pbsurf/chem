chem: collection of mostly python code for molecular visualization, QM/MM, FEP, etc.  Very long-term goal is something like enzyme design.  Intended for interactive use from standard python prompt.

Major features:
- 3D visualization of molecular geometry, orbitals, ESP, etc.
  - fast OpenGL volume and isosurface rendering
  - oriented toward analysis and debugging of calculations rather than presentation (for that, try [molstar](https://github.com/molstar/molstar))
- QM/MM driver supporting electrostatic embedding with various charge shifting schemes
- DLC/HDLC/Redundant internal coordinates
- transition state search (Dimer method, Lanczos method), reaction path optimization (NEB)
- read/write .pdb, TINKER, GAMESS and NWChem files
- model creation and setup: build polypeptides, add hydrogens and bonds, mutate residues, solvate

How to use this code:
1. If an example in projects/ or test/ looks promising, try using that as a starting point ... all the examples are outdated or work-in-progress currently.  Open an issue describing what you're interested in doing and I'll help with getting started.
- for example, see the "Preparation" section in projects/trypsin1.py; then from the folder containing 1MCT-trimI.pdb, at a Python prompt, run `execfile('<path to this repo>/projects/trypsin1.py')`
- add `import pdb; pdb.set_trace()` to step line by line
- see "Examples" in test/vis_test.py for standalone visualization examples
or
2. Cut and paste what you need: code is kept as self-contained as possible
 - opt/dlc.py: delocalized internal coordinates (DLC) and hybrid DLC; useful for geometry opt. w/ constraints
 - opt/lbfgs.py: gradient-only BFGS and L-BFGS optimizers (i.e., no line search)
 - opt/neb.py: nudged elastic band (reaction path optimization)
 - opt/dimer.py: Dimer and Lanczos methods (transition state search)
 - qmmm/resp.py: RESP/CHELPG charge fitting (only harmonic restraints currently)
 - mm.py: slow but simple MM energy and gradient for AMBER-type force field (and Hessian for Coulomb and LJ)
 - fep.py: simple FEP, BAR free energy calculations
 - model/build.py: build polypeptides, add hydrogens and bonds, mutate residues

Requirements:
- Python 3 w/ scipy and numpy (should mostly still work with Python 2.7)
- [OpenMM](https://openmm.org/) - molecular mechanics calculations
- [PySCF](https://github.com/sunqm/pyscf) - quantum chemistry calculations

See misc/chem-inst.sh for setup on Debian/Ubuntu

Optionally:
- [openmmtools](https://github.com/choderalab/openmmtools) - free energy calculations
- [AmberTools](https://ambermd.org/AmberTools.php) - small molecule parameterization (GAFF)
- [TINKER](https://dasher.wustl.edu/tinker/) - molecular mechanics calculations
- GAMESS (US) or NWChem - quantum chemistry calculations
- https://github.com/cclib/cclib - for reading GAMESS and NWChem output

Credit to [chemlab](https://github.com/chemlab/chemlab/) (3D camera, some shaders) and [speck](https://github.com/wwwtyro/speck) (some shaders), among others.

License:
Any published results obtained using this software should be accompanied by all code needed to replicate.

Screenshot: 1MCT.pdb shown with backbone ribbon, MM atoms as lines, QM atoms as sticks, and components of QM/MM force on each atom as yellow, cyan, magenta cylinders.

![Screenshot](misc/screenshot.png)
