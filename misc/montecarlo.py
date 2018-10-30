#!/usr/bin/env python

# Monte Carlo simulation of finite temperature classical SHO
# Ref: http://people.chem.ucsb.edu/kahn/kalju/MonteCarlo_3.html
# Step size should be choosen to yield 25% - 50% acceptance rate (...could implement automatic step size)
# Burn-in: to eliminate bias due to a poor starting point, we can discard the first N samples
# ... how to determine N?!?

import sys
from math import exp      # Math exp is much faster than SciPy exp
from random import random   # Mersienne Twister as the core generator

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Check if all the input is provided
if len(sys.argv) != 7:
  sys.exit ( "Usage: MC_Harmonic.py k r_eq temperature num_points r_initial max_stepsize" )

#
# Monte Carlo run parameters come from the command line
#
k       = float(sys.argv.pop(1))      # Harmonic force constant
r_eq    = float(sys.argv.pop(1))      # Potential minimum distance
temp    = float(sys.argv.pop(1))      # Temperature
points  = int(sys.argv.pop(1))        # Number of MC attempts
r_ini   = float(sys.argv.pop(1))      # Initial distance in MC
step    = float(sys.argv.pop(1))      # Step size in MC

R      = 1.9872156e-3    # Gas constant  kcal/mol/degree
beta   = 1 / (R * temp)
print '################################################'
print '#            HARMONIC POTENTIAL                #'
print '################################################'
print ' '
print 'Harmonic force constant k: ' , k,    'kcal/mol/A**2'
print 'Potential minimum:         ' , r_eq, 'Ang'
print 'Thermal energy RT/2:       ' , 0.5*R*temp, 'kcal/mol'
print ' '
#
# Hamiltonian: Harmonic potential
#
def hamiltonian(r):
  return 0.5 * k * (r - r_eq)**2
#
# Monte Carlo code
#
acc = 0
mc_dst = 0.0
mc_dst2 = 0.0
mc_ener = 0.0

r = r_ini
accept_dist = []

for point in range(points):
  dr = step * ( random() - 0.5 ) * 2.0
  r_new = r + dr
  v     = hamiltonian(r)
  v_new = hamiltonian(r_new)
  if v_new < v:        # Downhill move always accepted
    r = r_new
    v = v_new
    acc = acc + 1
  else:          # Uphill move accepted with luck
    A_mov = exp(-beta * (v_new - v))
    if A_mov > random():
      r = r_new
      v = v_new
      acc = acc + 1
  # Update regardless of acceptance!
  accept_dist += [ r ]
  mc_dst  = mc_dst + r
  mc_dst2 = mc_dst2 + r*r
  mc_ener = mc_ener + v

#
# Print out averages
#
mc_dst_av  = mc_dst / float(points)
mc_dst2_av = mc_dst2 / float(points)
mc_ener_av = mc_ener / float(points)

print 'Acceptance is:', 100 * acc / float(points), '%'
print ' '
print 'Monte Carlo Averages:'
print 'Mean distance, Ang:                 ', mc_dst_av
print 'Mean squared distance, Ang**2       ', mc_dst2_av
print 'Mean MC square displacement, Ang**2 ', mc_dst2_av - mc_dst_av*mc_dst_av
print 'Mean MC potential energy, kcal/mol: ', mc_ener_av

#print 'Mean dr: ', np.sum(accept_dist)/float(points)

# histogram of positions
n, bins, patches = plt.hist(accept_dist, 500, normed=True, facecolor='green', alpha=0.75)

plt.xlabel('Distance (A)')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Metropolis positions}$j')
plt.axis([-10, 10, 0, 2])
plt.grid(True)

plt.show()
