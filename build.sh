echo "Configuring and building Thirdparty/SPHORB ..."

cd Thirdparty/
tar -xvf SPHORB.tar.gz && cd SPHORB
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j


echo "Configuring and building DEMO ..."
cd ../../../
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
