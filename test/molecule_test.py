from chem.molecule import *


if __name__ == "__main__":
#~   mol = molecule("water");
#~   bonds, angles, diheds = mol.get_internals();
#~   for b in bonds:
#~     print b, mol.bond(b);
#~   for b in angles:
#~     print b, mol.angle(b);
#~   for b in diheds:
#~     print b, mol.dihedral(b);

  # check gradient calculations
  r = np.random.rand(2,3);
  bond0, grad = calc_dist(r, grad=True);
  dr = np.random.rand(2,3)*0.01;
  bond1 = calc_dist(r + dr);
  print "bond0: {}; bond1: {}; err vs. dr*grad: {}".format(bond0, bond1, bond1 - (bond0 + np.sum(grad*dr)));

  r = np.random.rand(3,3);
  angle0, grad = calc_angle(r, grad=True);
  dr = np.random.rand(3,3)*0.01;
  angle1 = calc_angle(r + dr);
  print "angle0: {}; angle1: {}; err vs. dr*grad: {}".format(angle0, angle1,
      angle1 - (angle0 + np.sum(grad*dr)));

  r = np.random.rand(4,3);
  dihed0, grad = calc_dihedral(r, grad=True);
  dr = np.random.rand(4,3)*0.01;
  dihed1 = calc_dihedral(r + dr);
  print "dihed0: {}; dihed1: {}; err vs. dr*grad: {}".format(dihed0, dihed1,
      dihed1 - (dihed0 + np.sum(grad*dr)));

  r = np.random.rand(4,3);
  dihed0, grad = cos_dihedral(r, grad=True);
  dr = np.random.rand(4,3)*0.01;
  dihed1 = cos_dihedral(r + dr);
  print "cosdihed0: {}; cosdihed1: {}; err vs. dr*grad: {}; sum(grad): {}".format(dihed0, dihed1,
      dihed1 - (dihed0 + np.sum(grad*dr)), np.sum(grad));
