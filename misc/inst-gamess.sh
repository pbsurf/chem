# GAMESS compile script - debian, gfortran, openblas
# for subsequent rebuilds (e.g. after changing compiler), just run
#  ./compall 2>&1 | tee compall.log && ./lked gamess 00 2>&1 | tee lked.log

# exit on error
set -e
CWD=`pwd`

# openblas seems to be 2x faster than atlas (on single core VM), competitive with intel MKL
# tests fail with gfortran-6 unless you use -O0
sudo apt-get install -y tcsh gfortran-5 libopenblas-dev
# libopenblas-dev requires gfortran-6, so we must force gfortran to point to gfortran-5
sudo ln -sf /usr/bin/gfortran-5 /usr/bin/gfortran

#gfortran -dumpversion
echo Choose defaults in config script, except: linux64, gfortran, gfortran version 5.3 (later not recognized), sockets
echo Choose pgiblas for math library but enter openblas path; this works because both provide a libblas.a
./config
ddi/compddi && mv ddi/ddikick.x .
./compall 2>&1 | tee compall.log
# use openblas lapack and include pthread to fix linking
# use '#' as delimiter for sed so we don't have to escape /
sed -i -E "s#set LAPACK='[^']+'#set LAPACK='/usr/lib/openblas-base/liblapack.a -lpthread'#g" lked
./lked gamess 00 2>&1 | tee lked.log

# configure rungms; in c-shell, :r removes extension, so rm will work if we include .inp in gamess input
sed -i -E "s#^set SCR=.*#if (! $?SCR_ROOT) set SCR_ROOT=/tmp\nset SCR=$SCR_ROOT/scr#g" rungms
sed -i -E "s#^set USERSCR=.*#set USERSCR=$SCR_ROOT/userscr#g" rungms
sed -i -E "s#^set GMSPATH=.*#set GMSPATH=$CWD\nmkdir -p \$SCR \$USERSCR\nrm -f \$USERSCR/\$1* \$USERSCR/\$1:r*#g" rungms

# run tests - use time as a rough benchmark
echo 'Running tests; you may Ctrl-C now and run later'
time ./runall 00 && tests/standard/checktst

