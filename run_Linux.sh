wget https://github.com/intel/mkl-dnn/releases/download/v0.18/mklml_lnx_2019.0.3.20190220.tgz
tar -xvzf mklml_lnx_2019.0.3.20190220.tgz
mv -f mklml_lnx_2019.0.3.20190220 mkldnn
rm *.tgz
mkdir build
cd build
rm -rf * && cmake .. && make
cd ..
cd mkldnn
cd lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`
cd ..
cd ..
cd build
./my_pair_dist 10000 20000 3