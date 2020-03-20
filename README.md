# cuda-deblur
GPU accelerated deblurring using CUDA C++.

This project implements a novel blind deblurring algorithm on a variety of blurred images.
It has seperate CPU and GPU implementations, allowing for a runtime comparison between the two.

## Set up
For the C libpng library and its C++ wrapper extension, png++, to work as intended, they first need to be installed:
```sh
sudo apt-get install libpng-dev

wget download.savannah.nongnu.org/releases/pngpp/png++-0.2.9.tar.gz
sudo tar -zxf png++-0.2.9.tar.gz -C /usr/src
cd /usr/src/png++00.2.9
make
make install
```

## Running
Now you can run the CPU and/or GPU implementation. 
To do so, go to the CPU and/or GPU directories, and call $make$ followed by the executable. 
The executable takes in 3 arguments: the blurry image, the image output file name, and the original image. 
This all looks like the following (when inside either the CPU or GPU directories):
```sh
make
./gpuLucyRich BLURRED_IMAGE OUTPUT_IMAGE_NAME ORIGINAL_IMAGE
```