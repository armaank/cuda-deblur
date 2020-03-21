/**
 * cpuFunctions.cpp - Functions used by the GPU implementation
 * that are still run by the host (the CPU).
 */


#include <iostream>

#include "kernelFunctions.cu"
#include "../utils/ops.hpp"
#include "../benchmarks/gputime.cu"
#include "../benchmarks/metrics.hpp"
#include "../utils/pngConnector.hpp"


/**
* gpuDeblur - Executes a Lucy-Richardson iteration to get an updated
* underlying image (updates the values of f) and applies a sharpening filter to the result
*   c - The blurred image.
*   g - The PSF.
*   f - The underlying image we are trying to recover.
*   H - Height of the image.
*   W - Width of the image.
*/
void gpuDeblur(const double *c, 
			const double *   g, 
			const double *   g_m, 
			double *         f, 
			double *         tmp1,
			double *         tmp2, 
			double *         tmp3,
			const unsigned   W,
			const unsigned   H,
			const unsigned   filter_W, 
			const unsigned   filter_H,
			const double *   s, 
			const unsigned   s_filter_W,
			const unsigned   s_filter_H)
{
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

	int threadsPerBlock = 1024;
	int blocksPerGrid   = 2*65535;

	kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(g, f, tmp1, W, H, filter_W, filter_H);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	kernel_elementWiseDivision<<<blocksPerGrid, threadsPerBlock>>>(c, tmp1, tmp2, W, H);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch kernel_elementWiseDivision kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(g_m, tmp2, tmp1, W, H, filter_W, filter_H);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	kernel_elementWiseMultiplication<<<blocksPerGrid, threadsPerBlock>>>(tmp1, f, tmp2, W, H);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch kernel_elementWiseMultiplication kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(s, tmp2, tmp3, W, H, s_filter_W, s_filter_H);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}


	kernel_internalMemcpy<<<blocksPerGrid, threadsPerBlock>>>(f, tmp3, W, H);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch kernel_internalMemcpy kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
}



void deblurImage(double *       filter_ptr,
				double *        filter_mirror_ptr, 
				unsigned        filter_width, 
				unsigned        filter_height, 
				double *        image_ptr, 
				double *        output_ptr,
				const Image &   target_image, 
				double *        s_filter_ptr, 
				unsigned        s_filter_width, 
				unsigned        s_filter_height,
				const std::string &output_file)
{
	GpuTimer gputime_gpu;
	gputime_gpu.start();

	unsigned height = target_image[0].size();
	unsigned width  = target_image[0][0].size();
	unsigned filter_size   = filter_width*filter_height*sizeof(double);
	unsigned element_count = 3*height*width;
	unsigned size          = element_count*sizeof(double);
	unsigned s_filter_size = s_filter_width*s_filter_height*sizeof(double);

	cudaError_t err = cudaSuccess;  // Error code to check return values for CUDA calls

	//// ALLOCATE MEMORY ON GPU
	double *d_f = NULL;
	err = cudaMalloc((void **)&d_f, size);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector f (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	double *d_g = NULL;
	err = cudaMalloc((void **)&d_g, size);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector g (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	double *d_g_m = NULL;
	err = cudaMalloc((void **)&d_g_m, size);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector g_m (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	double *d_s = NULL;
	err = cudaMalloc((void **)&d_s, size);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector c (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	double *d_c = NULL;
	err = cudaMalloc((void **)&d_c, size);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector c (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
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
	
	//// COPY MEMORY CPU -> GPU
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

	//// EXECUTING ALGORITHM
	gpuDeblur(d_c, d_g, d_g_m, d_f, d_tmp1, d_tmp2, d_tmp3, width, height, filter_width, filter_height, d_s, s_filter_width, s_filter_height);

	//// COPY MEMORY GPU -> CPU
	err = cudaMemcpy(output_ptr, d_f, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy vector f from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	Image output_image = ptr2image(output_ptr, width, height);

	//// FREE GPU MEMORY
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

	//// EVALUATE OUTPUT
	gputime_gpu.stop();

	std::cout << "Total time Elapsed - GPU: " << gputime_gpu.elapsed_time() << " ms" << std::endl;

	std::cout << "Baseline PSNR: " << psnr(ptr2image(image_ptr, width, height), target_image) << std::endl;
	std::cout << "Our algorithm's PSNR: " << psnr(output_image, target_image) << std::endl;

	saveImage(output_image, output_file);
	std::cout << "Image saved to: " << output_file << std::endl;
}


