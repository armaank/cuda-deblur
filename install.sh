# installs dependencies for cuda-deblur project

# first, install libpng
sudo apt-get install libpng-dev

# install libpng++
wget download.savannah.nongnu.org/releases/pngpp/png++-0.2.9.tar.gz
sudo tar -zxf png++-0.2.9.tar.gz -C /usr/src
cd /usr/src/png++00.2.9
make
# test installation - note that some will likely fail, ignore that
make test
# complete installation
make install

