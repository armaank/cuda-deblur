/* essential image processing operations */ 
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "ops.hpp"

/* create a matrix */
Matrix createMatrix(const int height, const int width)
{
	return Matrix(height, Array(width, 0));
}

/* create a matrix initilized w/ gaussian random variables */
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

/* convolve an image and a filter */
Image conv(const Image &image, const Matrix &filter)
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

/* element-wise matrix multiplicaiton */
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

/* for now, while I still don't know makefile, I'll include LR here too, but it really should be in a seperate file */ 


