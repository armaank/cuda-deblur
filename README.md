# cuda-deblur
GPU accelerated image deblurring using CUDA C++.

This project implements a novel blind deblurring algorithm on a variety of blurred images.
It has seperate CPU and GPU implementations, allowing for a runtime comparison between the two.

## Set up
For the C libpng library and its C++ wrapper extension, png++, to work as intended, they first need to be installed. These should already be installed on the Jetson Nano, but you can also install the libraries manually as follows:
```sh
sudo apt-get install libpng-dev

wget download.savannah.nongnu.org/releases/pngpp/png++-0.2.9.tar.gz
sudo tar -zxf png++-0.2.9.tar.gz -C /usr/src
cd /usr/src/png++00.2.9
make
make install
```
`lib_png` is not available in the repos provided on the machine used to cross compile, since the operating system version is outdated. Thus, in order to run the program, you must `make` it directly on the Jetson Nano. 

## Installation
After setup, to install the project, clone the repo into your directory of choice and go into the project directory
```sh
git clone https://github.com/armaank/cuda-deblur.git
cd cuda-deblur
```

## Usage
Now you can run the CPU and/or GPU implementation. 
To do so, go to the CPU and/or GPU directories, and call `make` followed by the executable. 
The executable takes in 3 arguments: the blurry image, the image output file name, and the original image. 
For example, to make the GPU code:
```sh 
cd gpu
make
./gpu_deblur.out BLURRED_IMAGE OUTPUT_IMAGE_NAME ORIGINAL_IMAGE
```
Likewise, to make the CPU code:
```sh 
cd cpu
make
./cpu_deblur.out BLURRED_IMAGE OUTPUT_IMAGE_NAME ORIGINAL_IMAGE
```
In order to compute and store logs of the results for the entire image dataset, you can use one of our provided `benchmark` scripts. 
For example, to benchmark the CPU code:
```sh 
cd cpu
make
sh cpu_benchmark.sh
```
Likewise, to benchmark the GPU code:
```sh 
cd gpu
make
sh gpu_benchmark.sh
```
These will store statistics and output images into `*pu/output/benchmarks`
