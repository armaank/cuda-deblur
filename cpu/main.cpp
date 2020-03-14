#include <iostream>
#include <vector>
#include <png++/png.hpp>
#include <chrono>

#include "cpuLucyRichardson.hpp"
#include "../benchmarks/metrics.hpp"


#define NUM_ITERATIONS 5


Image loadImage(const std::string &filename);
void  saveImage(Image &image, const std::string &filename);
void runLucyRichardson(const Matrix &kernel, const Image &blurry_image, const std::string &output_file);


int main(int argc, char **argv)
{
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    if (argc <= 2)
    {
        std::cerr << "error: specify input and output files" << std::endl;
        return -1;
    }
    std::cout << "Loading image from" << input_file << std::endl;
    Image image = loadImage(input_file);

    /////////////////////////////////////////////////////////////////////////

    // Kernel: gaussian 3x3
    Matrix filter = gaussian(3, 3, 1);
    runLucyRichardson(filter, image, output_file+"_gaussKernel3"+ ".png");

    // Kernel: gaussian 7x7
    filter = gaussian(7, 7, 1);
    runLucyRichardson(filter, image, output_file+"_gaussKernel7"+ ".png");

    /////////////////////////////////////////////////////////////////////////
    

    return 0;
}


void runLucyRichardson(const Matrix &kernel, const Image &blurry_image, const std::string &output_file)
{
    std::cout << "running lucy iterations..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    Image newImage = rlDeconv(blurry_image, kernel, NUM_ITERATIONS);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Execution time: " << 
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "microseconds" << std::endl;

    std::cout << "PSNR: " << psnr(newImage, blurry_image) << std::endl;

    saveImage(newImage, output_file);
    std::cout << "Image saved to: " << output_file << std::endl;
}


Image loadImage(const std::string &filename)
{
    png::image<png::rgb_pixel> image(filename);
    Image imageMatrix(3, Matrix(image.get_height(), Array(image.get_width())));

    int h, w;
    for (h = 0; h < image.get_height(); h++)
    {
        for (w = 0; w < image.get_width(); w++)
        {
            imageMatrix[0][h][w] = image[h][w].red;
            imageMatrix[1][h][w] = image[h][w].green;
            imageMatrix[2][h][w] = image[h][w].blue;
        }
    }

    return imageMatrix;
}

void saveImage(Image &image, const std::string &filename)
{
    assert(image.size() == 3);

    int height = image[0].size();
    int width = image[0][0].size();
    int x, y;

    png::image<png::rgb_pixel> imageFile(width, height);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            imageFile[y][x].red = image[0][y][x];
            imageFile[y][x].green = image[1][y][x];
            imageFile[y][x].blue = image[2][y][x];
        }
    }
    imageFile.write(filename);
}