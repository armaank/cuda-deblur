

#include <vector>
#include <iostream>

#include "gpuLucyRichardson.cu"
#include "../benchmarks/metrics.cpp" 
#include "../benchmarks/gputime.h"

#define NUM_ITERATIONS 5



using Array  = std::vector<double>;
using Matrix = std::vector<Array>;
using Image  = std::vector<Matrix>;


Image loadImage(const std::string &filename);
void  saveImage(Image &image, const std::string &filename);
void runLucyRichardson(const Matrix &kernel, const Image &blurry_image, const Image &target_image const std::string &output_file);
void runSimpleFilter(const Matrix &kernel, const Image &blurry_image, const Image &target_image const std::string &output_file);

Matrix gaussian(const int height, const int width, const double sigma);
Matrix sharpen(const int height, const int width);

double *image2ptr(const Image &input);
double *matrix2ptr(const Matrix &input);
Image ptr2image(const double *input, const int width, const int height);


int main(void)
{
    std::string input_file = argv[1];  // blurry image
    std::string output_file = argv[2]; // deblurred image
    std::string target_file = argv[3]; // target image 'ground truth';

    if (argc <= 3)
    {
        std::cerr << "error: specify input and output files" << std::endl;
        return -1;
    }
    std::cout << "Loading image from" << input_file << std::endl;
    Image image = loadImage(input_file);
    Image target_image = loadImage(target_file);

    /////////////////////////////////////////////////////////////////////////

    /* initalize gpu timers */
    GpuTimer gputime_lucy;
    GpuTimer gputime_gpu;

    // Kernel: gaussian 3x3
    Matrix filter = gaussian(3, 3, 1);
    runLucyRichardson(filter, image, output_file+"_gaussKernel3"+ ".png");

    // Kernel: gaussian 7x7
    filter = gaussian(7, 7, 1);
    runLucyRichardson(filter, image, output_file+"_gaussKernel7"+ ".png");

    filter = sharpen(3,3)
    runSimpleFilter(filter, image, output+file+"_sharpen3"+".png")

    std::cout << "Done!" << std::endl;
    
    /////////////////////////////////////////////////////////////////////////


    return 0;
}

void runSimpleFilter(const Matrix &filter, const Image &blurry_image, const Image &target_image const std::string &output_file)
{
    std::cout << "running simple filter..." << std::endl;

    cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

    // allocate device arrays
    // convolve filter and image
    // de-allocate device arrays
    // compute psnr
    // end

}
void runLucyRichardson(const Matrix &filter, const Image &blurry_image, const Image &target_image const std::string &output_file)
{
    std::cout << "running lucy iterations..." << std::endl;

    double *image_ptr  = image2ptr(image);

    Matrix filter_m(filt_length, Array(filt_width));
    for (int i = 0; i < filt_length; i++)
        for (int j = 0; j < filt_width; j++)
            filter_m[i][j] = filter[j][i];

    double *filter_ptr = matrix2ptr(filter);    
    double *filter_mirror_ptr = matrix2ptr(filter_m);
            
    cudaError_t err = cudaSuccess;  // Error code to check return values for CUDA calls

    gputime_gpu.start();

    // Allocate the device input vector f
    float *d_f = NULL;
    err = cudaMalloc((void **)&d_f, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector f (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector g
    float *d_g = NULL;
    err = cudaMalloc((void **)&d_g, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector g_m
    float *d_g_m = NULL;
    err = cudaMalloc((void **)&d_g_m, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_m (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector c
    float *d_c = NULL;
    err = cudaMalloc((void **)&d_c, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors f, g, and c in host memory to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_g, filter_ptr, filter_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_g_m, filter_mirror_ptr, filter_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_m from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_f, image_ptr, size, cudaMemcpyHostToDevice);
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
    float *d_tmp1 = NULL;
    err = cudaMalloc((void **)&d_tmp1, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector tmp1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_tmp2 = NULL;
    err = cudaMalloc((void **)&d_tmp2, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector tmp2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // TODO: time this loop, i guess?
    /* 
    maybe it's not a bad idea to use two timers. One that times the cpu and gpu lucy iterations, and 
    one that times the entire process for cpu and gpu, that way we know what the overhead is and factor 
    that into our analysis -armaan
    */
    gputime_lucy.start()
    for (int i=0; i<NUM_ITERATIONS; ++i)
    {
        updateUnderlyingImg(d_c, d_g, d_g_m, d_f, d_tmp1, d_tmp2, W, H);
    }
    gputime_lucy.stop()

    // Copy the device result vector in device memory to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_f, d_f, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector f from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_f);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector f (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_c);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_g);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_tmp1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector tmp1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_tmp2;
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector tmp2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    gputime_gpu.stop()    

    Image output = ptr2image(h_f, blurry_image[0][0].size, blurry_image[0].size);

    std::cout << "Total time Elapsed - GPU: " << gputime_gpu.elapsed_time << " ms" << std::endl;
    std::cout << "Lucy Iteration time Elapsed - GPU: " << gputime_lucy.elapsed_time << " ms" << std::endl; 

    std::cout << "PSNR: " << psnr(output, target_image) << std::endl;

    saveImage(output, output_file);
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


double *image2ptr(const Image& input)
{
    int width  = input[0][0].size;
    int height = input[0].size;

    double *ptr = new (std::nothrow) double(3*height*width);
    int idx = 0;
    for (int i = 0; i < height; ++i)
    {
        for (int j =0; j < width; ++j)
        {
            ptr[idx++] = input[0][i][j];
            ptr[idx++] = input[1][i][j];
            ptr[idx++] = input[2][i][j];
        }
    }
    return ptr;
}


double *matrix2ptr(const Matrix &input)
{
    int width  = input[0].size;
    int height = input.size;

    double *ptr = new (std::nothrow) double(height*width);
    int idx = 0;
    for (int i = 0; i < height; ++i)
        for (int j =0; j < width; ++j)
            ptr[idx++] = input[i][j];

    return ptr;
}


Image ptr2image(const double *input, const int width, const int height)
{
    Image output(3, Matrix(height, width));

    int idx = 0;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0 ; j < width; ++j)
        {
            output[0][i][j] = input[idx++];
            output[1][i][j] = input[idx++];
            output[2][i][j] = input[idx++];
        }
    }
    return output;
}


Matrix gaussian(const int height, const int width, const double sigma)
{
    Matrix kernel = createMatrix(height, width);
    double sum = 0.0;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            kernel[i][j] = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += kernel[i][j];
        }
    }

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            kernel[i][j] /= sum;

    return kernel;
}

Matrix sharpen(const int height, const int width)
{
    Matrix kernel = createMatrix(height, width);

    kernel[0][0] = 0;
    kernel[1][0] = -1;
    kernel[2][0] = 0;
    kernel[1][0] = -1;
    kernel[1][1] = 4;
    kernel[1][2] = -1;
    kernel[2][0] = 0;
    kernel[2][1] = -1;
    kernel[2][2] = 0;

    return kernel;
}