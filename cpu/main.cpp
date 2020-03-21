#include <iostream>
#include <vector>
#include <png++/png.hpp>
#include <chrono>

#include "cpuDeblur.hpp"
#include "../utils/ops.hpp"
#include "../utils/pngConnector.hpp"
#include "../benchmarks/metrics.hpp"

void deblurImage(const Matrix &kernel, const Image &blurry_image, const Image &target_image, const std::string &output_file);

int main(int argc, char **argv)
{
	if (argc <= 3)
	{
		std::cerr << "error: specify input and output files" << std::endl;
		return -1;
	}
	std::string input_file  = argv[1];  // blurry image
	std::string output_file = argv[2];  // deblurred image
	std::string target_file = argv[3];  // target image 'ground truth'

	std::cout << "Loading image from " << input_file << std::endl;
	Image image = loadImage(input_file);
	Image target_image = loadImage(target_file);
	
        /////////////////////////////////////////////////////////////////////////

	// baseline psnr calculation
	std::cout << "Blurry Image (baseline) PSNR: " << psnr(image, target_image) << std::endl;
	
	// deblur w/ a 3x3 gaussian kernel
	Matrix filter = gaussian(3, 3, 1);
	deblurImage(filter, image, target_image, output_file + "_gaussKernel3" + ".png");

        /////////////////////////////////////////////////////////////////////////

	return 0;
}

void deblurImage(const Matrix &kernel, const Image &blurry_image, const Image &target_image, const std::string &output_file)
{
	auto start      = std::chrono::high_resolution_clock::now();
	Image new_image = cpuDeblur(blurry_image, kernel, 1);
	auto end        = std::chrono::high_resolution_clock::now();

	std::cout << "Execution time: " 
			<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
			<< " microseconds" << std::endl;

	std::cout << "PSNR: " << psnr(new_image, target_image) << std::endl;

	saveImage(new_image, output_file);
	std::cout << "Image saved to: " << output_file << std::endl;
}

