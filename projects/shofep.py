import numpy as np
from chem.basics import *
from chem.mm import *
from chem.fep import *


## Toy free energy calculations for ensemble of harmonic oscillators
# - calculate free energy change for small change of frequency
# - we'll use amu (Dalton) ~ g/mol, ps, nm, kJ/mol = amu*nm^2/ps^2 to avoid constants

# k = m * omega**2
def EandG(r, k):
  return 0.5*k * np.sum(r*r), k * r


# analytical free energy: F = kB*T*ln(hbar*w/kB/T)
kB = BOLTZMANN*KJMOL_PER_HARTREE  # kJ/mol/K
hbar = 1.054571817E-34 * 1E-3 * AVOGADRO * 1E12 # in kJ/mol*ps
def shoF(omega, T0, kB):
  return kB*T0*np.log(hbar*omega/(kB*T0))  # = E - T*S = kB*T - T * kB*[1 - ln(hbar*w/kB/T)]


# free energy perturbation
def shoFEP(R, k0, k1, T0, kB):
  dE = 0.5*(k1 - k0) * np.sum(R*R, axis=1)  # energy differences
  return -kB*T0 * np.log( np.sum(np.exp(-dE/(kB*T0)))/len(dE) )  # Zwanzig formula


# BAR (Bennett acceptance ratio)
#def bar(dE0, dE1, T0, kB, dF0=None, maxiter=100, tol=1E-10):
#  bdE0, bdE1 = dE0/(kB*T0), dE1/(kB*T0)
#  bC = dF0/(kB*T0) if dF0 is not None else 0.5*(np.mean(bdE0) - np.mean(bdE1))
#  for ii in range(maxiter):
#    err = np.log(np.sum(1.0/(1 + np.exp(bdE1 + bC)))) - np.log(np.sum(1.0/(1 + np.exp(bdE0 - bC))))
#    bC = bC + err
#    if abs(err/bC) < tol:
#      break
#  return kB*T0*(bC - np.log(len(dE1)/len(dE0)))


# alternative calculation of BAR that might work better numerically (we only do exp() on negative numbers)
def bar2(dE0, dE1, T0, kB, dF0=None, maxiter=100, tol=1E-10):
  from scipy.special import logsumexp
  bdE0, bdE1 = dE0/(kB*T0), dE1/(kB*T0)
  max0, max1 = np.amax(bdE0), np.amax(bdE1)
  e0 = np.exp(bdE0 - max0)
  e1 = np.exp(bdE1 - max1)
  bC = dF0/(kB*T0) if dF0 is not None else 0.5*(np.mean(bdE0) - np.mean(bdE1))
  for ii in range(maxiter):
    f0 = -(max0 - bC) - np.log(np.exp(-(max0 - bC)) + e0)
    f1 = -(max1 + bC) - np.log(np.exp(-(max1 + bC)) + e1)
    err = logsumexp(f1) - logsumexp(f0)
    bC = bC + err
    if abs(err/bC) < tol:
      break
  return kB*T0*(bC - np.log(len(dE1)/len(dE0)))


# we'll try to use scale typical of molecular ro-vibrational modes
N = 1000  # number of 1D SHOs
T0 = 300  # corresponds to 2.5 kJ/mol
m = 12.0  # amu
w0 = 2*np.pi * 30  # typical ro-vibrational freq = 1000 cm^-1 ~ 30 THz = 30 ps^-1
w1 = 2*np.pi * 30.1
k0 = m*w0**2
k1 = m*w1**2
mddt = 0.2/w0  # time step for MD in ps


# analytical:
# 30 THz -> 30.1 THz gives us dF_exact = 8.301 kJ/mol
dF_exact = N*(shoF(w1, T0, kB) - shoF(w0, T0, kB))


# online MC FEP
class MCFEP:
  def __init__(self, r0, k0, k1, T0, kB, scale=1.0):
    self.r = np.array(r0)
    self.beta = 1.0/(kB*T0)
    self.k0, self.k1 = k0, k1
    self.scale = scale


  def run(self, nsamps=1000000, nprint=10000, adjstep=100, accept=0.5, verbose=False):
    naccept = 0
    dF = 0
    self.Es = np.zeros(nsamps)
    E = 0.5*self.k0 * np.sum(self.r*self.r)
    # for uniform SHOs, we could calculate dEfeps from E, but not the case in general so track separately
    self.dEfeps = np.zeros(nsamps)
    dEfep = 0.5*(self.k1 - self.k0) * np.sum(self.r*self.r)
    for ii in range(1, nsamps+1):
      idx = int(np.random.rand()*len(self.r))
      x0 = self.r[idx]
      x1 = x0 + self.scale*np.random.normal()
      dE = 0.5*self.k0*(x1*x1 - x0*x0)  # difference in energy for candidate change for system being simulated
      if dE < 0 or np.exp(-self.beta*dE) > np.random.rand():
        naccept += 1
        self.r[idx] = x1
        # difference in energy between system being simulated and second system for the current state
        dEfep += dE*(self.k1 - self.k0)/self.k0  # = 0.5*(k1 - k0)*(x1*x1 - x0*x0)
        E += dE

      self.dEfeps[ii-1] = dEfep
      self.Es[ii-1] = E

      if adjstep and ii%adjstep == 0:
        a = naccept/float(ii)
        self.scale *= np.clip(a/accept, 0.5, 2.0)
        if verbose:
          print("step %d: E = %f; accept = %f; scale = %f" % (ii, 0.5*self.k0*np.sum(self.r*self.r), a, self.scale))

      if (nprint and ii%nprint == 0) or ii == nsamps:
        dF = -(1/self.beta) * np.log( np.sum(np.exp(-self.beta*self.dEfeps[:ii]))/ii )  # Zwanzig formula
        #fluctE = np.std(self.Es[:ii])/np.mean(self.Es[:ii])
        if nprint:
          kap2 = np.var(self.dEfeps[:ii])/(2*kB*T0)  # dF = <dE> - var(dE)/2kBT + ... from cumulant expansion
          print("step %d: var(E)/2kBT = %f; <dE> = %f; dF = %f" % (ii, kap2, np.sum(self.dEfeps[:ii])/ii, dF))

    return dF


# Thermodynamic integration: E = 0.5*[k0 + (k1 - k0)*l]*x^2; dE/dl = 0.5*(k1-k0)*x^2 ... since dE/dl is indep.
#  of l, it comes out of integral and we basically just recover FEP


# start interactive
import pdb; pdb.set_trace()

# Open issues:
# - fluctuations of potential energy (i.e. since that's all we have for MC)
# - Liouville formalism / symplectic integrators ... how to determine if an integrator is symplectic? (RK is not, Verlet is); www.unige.ch/~hairer/poly_geoint/week1.pdf ... week2 etc
# - interesting derivation of ideal gas density of states: en.wikipedia.org/wiki/Thermal_fluctuations


## Monte Carlo
if 0:
  for c in [1.02]:  #[1.001, 1.01, 1.1, 1.5]:
    w1 = c*w0
    k1 = m*w1**2
    dF_exact = N*(shoF(w1, T0, kB) - shoF(w0, T0, kB))

    # std(E)/E should be sqrt(2)/sqrt(N) (sqrt(2) since E only includes potential energy)
    mc0 = MCFEP(np.zeros(N), k0, k1, T0, kB, scale=0.005)
    mc0.run(nsamps=50000, nprint=0)  # equilibration and scale determination
    dF_mc0 = mc0.run(adjstep=10000, nprint=0)  # production

    mc1 = MCFEP(np.zeros(N), k1, k0, T0, kB, scale=0.005)
    mc1.run(nsamps=50000, nprint=0)  # equilibration and scale determination
    dF_mc1 = mc1.run(adjstep=10000, nprint=0)  # production

    h0, b0 = np.histogram(mc0.dEfeps, bins=1000)
    h1, b1 = np.histogram(-mc1.dEfeps, bins=1000)
    plot(b0[:-1], h0, b1[:-1], h1, legend=['fwd(0)', 'rev(1)'])
    # exp(dE/kBT) should shift histograms to have the same peak (compensating for difference in distribution
    #  of states; note that higher k -> smaller <dx^2> -> smaller <dE>) - only works if sufficient overlap!
    # exp(-np.mean(b1)/kBT) factor is just a scalar to keep y-range approx the same
    #plot(b0[:-1], h0, b1[:-1], h1*np.exp((b1[:-1] - np.mean(b1[:-1]))/(kB*T0)), legend=['fwd(0)', 'shifted rev(1)'])
    g0 = int_autocorr_time(mc0.dEfeps)
    g1 = int_autocorr_time(mc1.dEfeps)

    #ddF0 = fep_err(mc0.dEfeps/(kB*T0), g0)
    #ddF1 = fep_err(mc1.dEfeps/(kB*T0), g1)
    #dF_bar, ddF_bar = [kB*T0*x for x in bar(mc0.dEfeps[::g0]/(kB*T0), mc1.dEfeps[::g0]/(kB*T0), calc_std=True)]

    # BAR calculation
    dF_bar, ddF_bar = [kB*T0*x for x in bar(mc0.dEfeps/(kB*T0), mc1.dEfeps/(kB*T0), calc_std=True)]
    print("w1/w0: %f; exact: %g; FEP: %g; FEP (rev): %g; BAR: %g (std: %g)" % (c, dF_exact, dF_mc0, dF_mc1, dF_bar, ddF_bar))


if 0:
  # more steps ... no improvement, looks like 200K samps isn't enough, need more like 1M
  nsteps = 10
  dk = (k1 - k0)/nsteps
  k = k0
  dF = 0
  for ii in range(10):
    mc = MCFEP(np.zeros(N), k, k + dk, T0, kB, scale=0.005)
    mc.run(nsamps=20000, nprint=0)
    dF += mc.run(nsamps=200000, adjstep=10000)
    k += dk


## Dynamics (MD)
def md_mon(md, r, v, E):
  KE = 0.5*np.sum(md.m*(v*v))
  T = KE/(0.5*md.kB*md.Ndim)
  print("step %d: KE = %f; PE = %f; E = %f; T = %f" % (md.step, KE, E, KE + E, T))

if 1:
  md0 = SimpleMD(lambda r: EandG(r, k0), m, r0=np.zeros(N), kB=kB, T0=T0, dt=mddt)
  _ = md0.run(10000, mon=md_mon)
  mdR0 = md0.run(100000, nsave=200)  # 150 - 200 for uncorrelated samples (at least for wide freq range)

  w1 = 1.02*w0
  k1 = m*w1**2
  md1 = SimpleMD(lambda r: EandG(r, k1), m, r0=np.zeros(N), kB=kB, T0=T0, dt=mddt)
  _ = md1.run(10000, mon=md_mon)
  mdR1 = md1.run(100000, nsave=200)

  dE0 = 0.5*(k1 - k0) * np.sum(mdR0*mdR0, axis=1)
  dF_md0 = kB*T0 * -np.log( np.sum(np.exp(-dE0/(kB*T0)))/len(dE0) )  #shoFEP(np.array(mdR0), k0, k1, T0, kB)

  dE1 = 0.5*(k0 - k1) * np.sum(mdR1*mdR1, axis=1)
  dF_md1 = kB*T0 * -np.log( np.sum(np.exp(-dE1/(kB*T0)))/len(dE1) )

  dF_mdbar, ddF_mdbar = [kB*T0*x for x in bar(dE0/(kB*T0), dE1/(kB*T0), calc_std=True)]

  dF_exact = N*(shoF(w1, T0, kB) - shoF(w0, T0, kB))
  print("w1/w0: %f; exact: %g; FEP: %g; FEP (rev): %g; BAR: %g (std: %g)" % (w1/w0, dF_exact, dF_md0, dF_md1, dF_mdbar, ddF_mdbar))


# detecting equilibriation by finding offset that maximizes number of uncorrelated samples (again from pymbar)
if 0:
  K = k0*(np.random.rand(N))  # using narrow range of frequencies results in AC fn going negative too soon
  EandG2 = lambda r: (0.5*np.sum(K*(r*r)), K*r)
  md2 = SimpleMD(EandG2, m, r0=np.zeros(N), kB=kB, T0=T0, dt=mddt)
  mdR2 = np.asarray(md2.run(40000, nsave=1))
  dE2 = np.sum(0.5*K * mdR2*mdR2, axis=1)
  offsets = [len(dE2)/2**n for n in range(1, 9)]
  effsamps = [(len(dE2) - offset)/int_autocorr_time(dE2[offset:]) for offset in offsets]
  print("optimal offset: %d, %f effective samples" % (offsets[argmax(effsamps)], max(effsamps)))


if 0:
  # incr k -> energy transfered to bath; need big change to overcome noise
  md0 = SimpleMD(lambda r: EandG(r, k0), m, r0=np.zeros(N), kB=kB, T0=T0, dt=mddt)
  _ = md0.run(10000, mon=md_mon)
  md0.dEbath = 0
  md0.EandG = lambda r: EandG(r, k0 + k0*min(1, md0.step/50000.0))
  mdR0 = md0.run(100000, nsave=5000, mon=md_mon)


if 0:
  # nsave=1: int_autocorr_time = 146; nsave=10: int_autocorr_time=18.8; nsave=100: int_autocorr_time=2.27
  K = k0*(np.random.rand(N))  # using narrow range of frequencies results in AC fn going negative too soon
  EandG2 = lambda r: (0.5*np.sum(K*(r*r)), K*r)
  md2 = SimpleMD(EandG2, m, r0=np.zeros(N), kB=kB, T0=T0, dt=mddt)
  md2.run(2000)
  for nsave in [1,10,30,100,500]:
    mdR2 = md2.run(100000, nsave=nsave)
    dE2 = np.sum(0.5*K * mdR2*mdR2, axis=1)  #np.sum(0.5*(k1 - k0)*K/k0 * mdR2*mdR2, axis=1)
    print("nsave: %d; nsave * autocorr_steps: %f" % (nsave, nsave*int_autocorr_time(dE2)))


if 0:
  # test energy conservation
  K = k0*(0.9 + 0.2*np.random.rand(N))
  EandG2 = lambda r: (0.5*np.sum(K*(r*r)), K*r)
  md2 = SimpleMD(EandG2, m, r0=np.zeros(N), kB=kB, T0=T0, dt=mddt, therm_steps=0)
  _ = md2.run(2000)
  _ = md2.run(10000, mon=md_mon)


if 0:
  # fluctuations: C_V = dE/dT = N*kB; <dE^2> = N kB^2 T^2; <dT^2> = T^2/N
  Es, Us, Ks, Ts = [], [], [], []
  def md_rec(md, r, v, E):
    KE = 0.5*np.sum(md.m*(v*v))
    T = KE/(0.5*md.kB*md.Ndim)
    #Us.append(E); Ks.append(KE)
    Es.append(KE + E); Ts.append(T)

  md3 = SimpleMD(EandG2, m, r0=np.zeros(N), kB=kB, T0=T0, dt=mddt, therm_steps=100)
  _ = md3.run(2000)
  _ = md3.run(100000, nsave=1, mon=md_rec)
  #print("std(E): %f (%f); std(T): %f (%f)" % (np.std(Es), np.sqrt(N)*kB*T0, np.std(Ts), T0/np.sqrt(N)))
  print("std(E)/E: %f; std(T)/T0: %f; N^0.5: %f" % (np.std(Es)/np.mean(Es), np.std(Ts)/np.mean(Ts), 1/np.sqrt(N)))
