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
    std::string input_file = argv[1];  // blurry image
    std::string output_file = argv[2]; // deblurred image
    std::string target_file = argv[3]; // target image 'ground truth'
    if (argc <= 3)
    {
        std::cerr << "error: specify input and output files" << std::endl;
        return -1;
    }
    std::cout << "Loading image from " << input_file << std::endl;
    Image image = loadImage(input_file);
    Image target_image = loadImage(target_file);
    /////////////////////////////////////////////////////////////////////////

    // Kernel: gaussian 3x3
    Matrix filter = gaussian(3, 3, 1);
    deblurImage(filter, image, target_image, output_file + "_gaussKernel3" + ".png");

    // Kernel: gaussian 7x7
    filter = gaussian(7, 7, 1);
    deblurImage(filter, image, target_image, output_file + "_gaussKernel7" + ".png");

    // Kernel: sharpening filter 3x3
    filter = sharpen(3, 3);
    //simple_filter(filter, image, target_image, output_file + "_sharpen3" + ".png");

    // baseline psnr calculation
    double baseline_psnr = psnr(image, target_image);
    std::cout << "Blurry Image PSNR: " << baseline_psnr << std::endl;
    /////////////////////////////////////////////////////////////////////////

    return 0;
}

void deblurImage(const Matrix &kernel, const Image &blurry_image, const Image &target_image, const std::string &output_file)
{
    auto start = std::chrono::high_resolution_clock::now();
    Image newImage = cpuDeblur(blurry_image, kernel);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;

    std::cout << "PSNR: " << psnr(newImage, target_image) << std::endl;

    saveImage(newImage, output_file);
    std::cout << "Image saved to: " << output_file << std::endl;
}

