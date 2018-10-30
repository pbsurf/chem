# Quantum chemistry VM setup for Debian Linux

APTINST="sudo apt-get --no-install-recommends -y install "
GATEWAY=`ip addr | sed -nE 's/^\s*inet ([0-9]+\.[0-9]+\.[0-9]+).*eth0$/\1.1/p'`
SERVER="http://$GATEWAY:8080"

SITE_PKGS=$HOME/.local/lib/python2.7/site-packages

# root directory for installation
CHEM=$HOME/qc

# abort script on error
set -e

echo "This script is totally untested!  Cut and paste sections instead of running directly."
exit 1

# Make sure download links point to recent versions
read -p "Have you updated download links in this script to latest versions? " -n 1 -r
echo    # blank line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
  echo "Please update download links!"
  exit 1
fi

# put everything in $CHEM directory; we assume .bashrc includes a line like 'source $HOME/.localrc'
mkdir -p $CHEM
cd $CHEM
echo "export CHEM=$CHEM" >> $HOME/.localrc

# Python
$APTINST libopenblas-dev python-numpy python-scipy python-sympy #python-matplotlib

#$APTINST nwchem
#$APTINST pymol
# openbabel converts between many different comp chem file formats ... but too general to be useful
#$APTINST openbabel

# chem repo
git clone https://bitbucket.org/mattrwhite/chem
ln -s $CHEM/chem $SITE_PKGS/chem

# PyQuante v1
#$APTINST subversion
#svn checkout svn://svn.code.sf.net/p/pyquante/code/trunk pyquante
wget https://sourceforge.net/code-snapshots/svn/p/py/pyquante/code/pyquante-code-219-trunk.zip
unzip pyquante-code-219-trunk.zip
mv pyquante-code-219-trunk pyquante
rm pyquante-code-219-trunk.zip
(cd pyquante && python setup.py build)
ln -s $CHEM/pyquante/build/lib.linux-x86_64-2.7/PyQuante $SITE_PKGS/PyQuante

# PyQuante v2
git clone https://github.com/rpmuller/pyquante2.git
(cd pyquante2 && python setup.py build)
ln -s $CHEM/pyquante2/build/lib.linux-x86_64-2.7/pyquante2 $SITE_PKGS/pyquante2

# cclib
git clone https://github.com/cclib/cclib.git
# I don't think build is actually needed, as cclib is python only
(cd cclib && python setup.py build) # && python setup.py install --user)
ln -s $CHEM/cclib/src/cclib $SITE_PKGS/cclib

# PySCF
$APTINST cmake
git clone https://github.com/sunqm/pyscf
pushd pyscf/pyscf/lib
(mkdir build && cd build && cmake .. && make)
popd
ln -s $CHEM/pyscf/pyscf $SITE_PKGS/pyscf

# optimized integral library for PySCF - qcint
pushd pyscf/pyscy/lib/build/deps/src
git clone https://github.com/sunqm/qcint
(mkdir build && cd build && cmake .. && make VERBOSE=1)
cd ../../../deps/lib
mv libcint.so.3.0 old_libcint.so.3.0
ln -s ../../build/deps/src/qcint/build/libcint.so.3.0 libcint.so.3.0
popd
echo "Replaced libcint with qcint for PySCF - may need to rerun link command with `-lmvec -lm` prepended to list of libraries"

# Molden
wget ftp://ftp.cmbi.ru.nl/pub/molgraph/molden/bin/Linux/molden5.7.full.ubuntu.64.tar.gz
tar xaf molden5.7.full.ubuntu.64.tar.gz
rm molden5.7.full.ubuntu.64.tar.gz

# VMD
## TODO: install to $HOME instead of system-wide!
wget $SERVER/vmd-1.9.3beta3.bin.LINUXAMD64-OptiX.opengl.tar.gz
tar xaf vmd-1.9.3beta3.bin.LINUXAMD64-OptiX.opengl.tar.gz
rm vmd-1.9.3beta3.bin.LINUXAMD64-OptiX.opengl.tar.gz
(cd vmd-1.9.3beta3 && ./configure && sudo make install)
rm -r vmd-1.9.3beta3

# NAMD - likely will need to login to site first
mkdir namd
pushd namd
wget https://www.ks.uiuc.edu/Research/namd/2.12/download/832164/NAMD_2.12_Linux-x86_64-multicore.tar.gz -O namd.tar.gz
tar xaf namd.tar.gz --strip-components 1
rm namd.tar.gz
# CHARMM format FF parameters - http://mackerell.umaryland.edu/charmm_ff.shtml
wget "http://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/toppar_c36_jul18.tgz" -O toppar.tar.gz
tar xaf toppar.tar.gz
# stream format not used by NAMD
#rm -r toppar/stream
rm toppar.tar.gz
popd
echo 'export NAMD_PATH=$CHEM/namd' >> $HOME/.localrc

# Tinker
mkdir tinker
pushd tinker
# common files
wget http://dasher.wustl.edu/tinker/downloads/tinker-7.1.2.tar.gz
tar xaf tinker-7.1.2.tar.gz
# linux exe
wget http://dasher.wustl.edu/tinker/downloads/bin-linux64-7.1.2.tar.gz
tar xaf bin-linux64-7.1.2.tar.gz
rm *-7.1.2.tar.gz
popd
echo 'export TINKER_PATH=$CHEM/tinker/bin-linux64' >> $HOME/.localrc

# GAMESS
wget $SERVER/gamess-us-18aug2016R1.tar.gz
tar xaf gamess-us-18aug2016R1.tar.gz
rm gamess-us-18aug2016R1.tar.gz
(cd gamess && sh ../chem/misc/inst-gamess.sh)
echo 'export GAMESS_PATH=$CHEM/gamess' >> $HOME/.localrc
