#include <iostream>
#include <cmath>
#include <limits>

#include "metrics.hpp"
#include "../utils/ops.hpp"

/* computes mean squared error between to images */
double _mse(const Image &im1, const Image &im2)
{
    double squared_err = 0.0;

    int height = im1[0].size();
    int width = im1[0][0].size();
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
        {
            squared_err += pow(im1[0][i][j] - im2[0][i][j], 2);
            squared_err += pow(im1[1][i][j] - im2[1][i][j], 2);
            squared_err += pow(im1[2][i][j] - im2[2][i][j], 2);
        }

    return squared_err / (3 * height * width);
}

/* computes peak signal-to-noise ratio */
double psnr(const Image &im1, const Image &im2)
{
    /* compute mse */
    double mse = _mse(im1, im2);

    double MAX = 255;
    /* from mse, compute psnr */
    double psnr = 10 * log10(pow(MAX, 2)/ mse);
    return psnr;
}

