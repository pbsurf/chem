#! /usr/bin/python
import sys, os
import numpy as np
import matplotlib.pyplot as plt

# if __name__ == "__main__" and : execfile("../qmmm/plotter.py");

def plot_Eparts(skey, comps=False):
  """ One trace for each component in Ecomps for a given key in Ecomps """
  # convert from list of dicts to dict of lists
  # this is a generalized transpose - seems very useful
  ET = listdict_T(Ecomps[skey]);
  plt.figure();
  comps = comps or ET.keys();
  for key in comps:
    E = ET[key];
    plt.plot(diheds, np.array(E) - E[0], label=key);
  plt.legend();
  plt.title(skey);
  plt.xlabel(xlabel);
  plt.ylabel("Energy (Hartrees)");
  plt.show();


def plot_Etot(traces=None):
  """ one trace for each key on Etots dict """
  # for loop is only way to set separate labels for each trace
  plt.figure();
  traces = traces or Etots.keys();
  for key in traces:
    E = Etots[key];
    plt.plot(diheds, np.array(E) - E[0], label=key);
  plt.legend();
  plt.title("Total Energy (shifted)");
  plt.xlabel(xlabel);
  plt.ylabel("Energy (Hartrees)");
  plt.show();


def listdict_T(listofdict):
  ET = {};
  for Ecomp in listofdict:
    for key, E in Ecomp.iteritems():
      if key not in ET:
        ET[key] = [];
      ET[key].append(E);
  return ET;

def writeres(resultfile, **kargs):
  os.system("mv %s %s.old" % (resultfile, resultfile));  # backup previous
  f = open(resultfile, 'w');
  f.write( ";\n".join([ "%s = %r" % (key, val) for key, val in kargs.iteritems() ]) );
  f.close();


for f in sys.argv[1:]:
  execfile(f);
# interactive mode (makes figure display nonblocking
#  so we can interactively create multiple figs)
plt.ion();
