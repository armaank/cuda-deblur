

#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "cpuLucyRichardson.hpp"

Matrix createMatrix(const int height, const int width)
{
    return Matrix(height, Array(width, 0));
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

Matrix divide(const Matrix &a, const Matrix &b)
{
    /* make sure that height and width match */
    assert(a.size() == b.size() && a[0].size() == b[0].size());

    Matrix result = createMatrix(a.size(), a[0].size());
    for (int i = 0; i < result.size(); i++)
        for (int j = 0; j < result[0].size(); j++)
        {
            result[i][j] = b[i][j] == 0 ? 999999999 : a[i][j] / b[i][j];
        }

    return result;
}

Image rlDeconv(const Image &image, const Matrix &filter, const int n_iter)
{
    Image im_deconv = image;
    Image rel_blur = image;

    int filt_length = filter.size();
    int filt_width = filter[0].size();

    /* compute and store mirrored psf */
    Matrix filter_m(filt_length, Array(filt_width));
    for (int i = 0; i < filt_length; i++)
        for (int j = 0; j < filt_width; j++)
            filter_m[i][j] = filter[j][i];

    /* perform lucy iterations */
    std::cout << "Iteration number: " << std::flush;
    for (int i = 0; i < n_iter; i++)
    {
        std::cout << i + 1 << ", " << std::flush;

        Image tmp1 = conv(im_deconv, filter); /* convolve target image by psf */

        for (int d = 0; d < 3; d++) /* element-wise division to compute relative blur */
            rel_blur[d] = divide(image[d], tmp1[d]);

        Image tmp2 = conv(rel_blur, filter_m); /* filter blur by psf */

        for (int d = 0; d < 3; d++) /* element-wise multiply to update deblurred image */
            im_deconv[d] = multiply(tmp2[d], im_deconv[d]);
    }
    std::cout << "\n";

    return im_deconv;
}