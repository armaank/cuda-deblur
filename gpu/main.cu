

#include <vector>
#include <iostream>

#include "gpuLucyRichardson.cu"
#include "../benchmarks/metrics.hpp" 
#include "../benchmarks/gputime.cu"
#include "../utils/pngConnector.hpp"
#include "../utils/ops.hpp" 

void deblurImage(double *filter_ptr, double *filter_mirror_ptr, double *image_ptr, double *output_ptr,
    const Image &target_image, const std::string &output_file, int filter_width, int filter_height, double *s_filter_ptr, int s_filter_width, int s_filter_height);

int main(int argc, char **argv)
{
    if (argc <= 3)
    {
        std::cerr << "error: specify input and output files" << std::endl;
        return -1;
    }
    std::string input_file = argv[1];  // blurry image
    std::string output_file = argv[2]; // deblurred image
    std::string target_file = argv[3]; // target image 'ground truth';


    std::cout << "Loading image from" << input_file << std::endl;
    Image image = loadImage(input_file);
    Image target_image = loadImage(target_file);

    double *image_ptr  = image2ptr(image);
    double *output_ptr = new (std::nothrow) double[3*image[0].size()*image[0][0].size()];



    /////////////////////////////////////////////////////////////////////////

    // Kernel: gaussian 3x3
    Matrix filter = gaussian(3, 3, 1);
    int filter_width  = filter[0].size(); 
    int filter_height = filter.size(); 
    Matrix filter_m = createMatrix(filter_height, filter_width);
    for (int i = 0; i < filter_height; i++)
        for (int j = 0; j < filter_width; j++)
            filter_m[i][j] = filter[j][i];
    
    Matrix s_filter = sharpen(3,3);
    int s_filter_width = s_filter[0].size();
    int s_filter_height = s_filter.size();
    

    double *filter_ptr = matrix2ptr(filter);    
    double *filter_mirror_ptr = matrix2ptr(filter_m);
    double *s_filter_ptr = matrix2ptr(s_filter);

    deblurImage(filter_ptr, filter_mirror_ptr, image_ptr, output_ptr, target_image, output_file + "_gaussKernel3"+".png", filter_width, filter_height, s_filter_ptr, s_filter_width, s_filter_height);

    
    std::cout << "Done!" << std::endl;
    
    /////////////////////////////////////////////////////////////////////////


    return 0;
}



void deblurImage(double *filter_ptr, double *filter_mirror_ptr, double *image_ptr, double *output_ptr,
    const Image &target_image, const std::string &output_file, int filter_width, int filter_height, double *s_filter_ptr, int s_filter_width, int s_filter_height)
{
    /* initalize gpu timers */
    GpuTimer gputime_gpu;
    gputime_gpu.start();
    int height = target_image[0].size();
    int width = target_image[0][0].size();
    int filter_size = filter_width*filter_height*sizeof(double);
    int element_count = 3*height*width;
    int size = element_count*sizeof(double);
    int s_filter_size = s_filter_width*s_filter_height*sizeof(double);

    cudaError_t err = cudaSuccess;  // Error code to check return values for CUDA calls

    // Allocate the device output vector f
    double *d_f = NULL;
    err = cudaMalloc((void **)&d_f, size);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device vector f (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector g
    double *d_g = NULL;
    err = cudaMalloc((void **)&d_g, size);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device vector g (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

	
    // Allocate the device input vector g_m
    double *d_g_m = NULL;
    err = cudaMalloc((void **)&d_g_m, size);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device vector g_m (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate device inout vector s (sharpening filter)
    double *d_s = NULL;
    err = cudaMalloc((void **)&d_s, size);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device vector c (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    
    // Allocate the device output vector c
    double *d_c = NULL;
    err = cudaMalloc((void **)&d_c, size);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device vector c (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors f, g, and c in host memory to the device input vectors in device memory
    std::cout << "Copy input data from the host memory to the CUDA device." << std::endl;
    err = cudaMemcpy(d_g, filter_ptr, filter_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy vector g from host to device (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_g_m, filter_mirror_ptr, filter_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy vector g_m from host to device (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_f, image_ptr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector f from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_s, s_filter_ptr, s_filter_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector f from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	
    err = cudaMemcpy(d_c, image_ptr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector c from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
   
    // allocate the temporary gpu memory
    double *d_tmp1 = NULL;
    err = cudaMalloc((void **)&d_tmp1, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector tmp1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    double *d_tmp2 = NULL;
    err = cudaMalloc((void **)&d_tmp2, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector tmp2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    double *d_tmp3 = NULL;
    err = cudaMalloc((void **)&d_tmp3, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector tmp2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    std::cout << "running lucy iterations... ";
    for (int i=0; i<NUM_ITERATIONS; ++i)
    {
        std::cout << i+1 << ", " << std::flush;
        updateUnderlyingImg(d_c, d_g, d_g_m, d_f, d_tmp1, d_tmp2, d_tmp3, width, height, filter_width, filter_height, d_s, s_filter_width, s_filter_height);
        //updateUnderlyingImg_old(d_c, d_g, d_g_m, d_f, d_tmp1, d_tmp2, width, height, filter_width, filter_height);
	
    }
    std::cout << std::endl;
    
    // Copy the device result vector in device memory to the host result vector in host memory.
    std::cout << "Copy output data from the CUDA device to the host memory" << std::endl;
    err = cudaMemcpy(output_ptr, d_f, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy vector f from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_f);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to free device vector f (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_c);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to free device vector c (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_g);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to free device vector g (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_g_m);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to free device vector g_m (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(d_tmp1);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to free device vector tmp1 (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_tmp2);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to free device vector tmp2 (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_tmp3);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to free device vector tmp3 (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        exit(EXIT_FAILURE);
    }


    Image output = ptr2image(output_ptr, width, height);
    gputime_gpu.stop();

    std::cout << "Total time Elapsed - GPU: " << gputime_gpu.elapsed_time() << " ms" << std::endl;

    std::cout << "BaselinePSNR: " << psnr(ptr2image(image_ptr, width, height), target_image) << std::endl;
    std::cout << "PSNR: " << psnr(output, target_image) << std::endl;

    saveImage(output, output_file);
    std::cout << "Image saved to: " << output_file << std::endl;

}


