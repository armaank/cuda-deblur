#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert> // might not need this, todo: double check

#include "ops.hpp"

/* create an instance of a Matrix class */
Matrix createMatrix(const int height, const int width)
{
    return Matrix(height, Array(width, 0));
}

/* create a Matrix, initilized w/ Gaussian random vars as entries */
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

/* create a sharpening filter */
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

/* performs convolution between an Image and a Matrix kernel */
Image conv2D(const Image &image, const Matrix &filter)
{
    assert(image.size() == 3 && filter.size() != 0);

    int newImageHeight = image[0].size();
    int newImageWidth = image[0][0].size();

    Image newImage(3, createMatrix(newImageHeight, newImageWidth));

    for (int d = 0; d < 3; d++)
        for (int i = 0; i < newImageHeight; i++)
            for (int j = 0; j < newImageWidth; j++)
            {
                int w_max = std::min<int>(newImageWidth, filter[0].size() + j);
                int h_max = std::min<int>(newImageHeight, filter.size() + i);
                for (int h = i; h < h_max; h++)
                    for (int w = j; w < w_max; w++)
                        newImage[d][i][j] += filter[h - i][w - j] * image[d][h][w];
            }

    return newImage;
}

/* perform element-wise multiplication between two Matricies */
Matrix multiply(const Matrix &a, const Matrix &b)
{
    /* make sure that height and width match */
    assert(a.size() == b.size() && a[0].size() == b[0].size());

    Matrix result = createMatrix(a.size(), a[0].size());
    for (int i = 0; i < result.size(); i++)
        for (int j = 0; j < result[0].size(); j++)
            result[i][j] = a[i][j] * b[i][j];

    return result;
}

/* perform element-wise division between two Matricies */
Matrix divide(const Matrix &a, const Matrix &b)
{
    /* make sure that height and width match */
    assert(a.size() == b.size() && a[0].size() == b[0].size());

    Matrix result = createMatrix(a.size(), a[0].size());
    for (int i = 0; i < result.size(); i++)
        for (int j = 0; j < result[0].size(); j++)
        {
            result[i][j] = b[i][j] == 0 ? 1 : a[i][j] / b[i][j];
        }

    return result;
}

double *image2ptr(const Image& input)
{
    int width  = input[0][0].size();
    int height = input[0].size();

    double *ptr = new (std::nothrow) double[3*height*width];
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
    int width  = input[0].size();
    int height = input.size();

    double *ptr = new (std::nothrow) double[height*width];
    int idx = 0;
    for (int i = 0; i < height; ++i)
        for (int j =0; j < width; ++j)
            ptr[idx++] = input[i][j];

    return ptr;
}


Image ptr2image(const double *input, const int width, const int height)
{
    Image output(3, createMatrix(height, width) );

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
