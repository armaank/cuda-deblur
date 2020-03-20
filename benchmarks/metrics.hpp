/**
 * metrics.hpp - Header file for the metric functions.
 */

#pragma once

#include <iostream>
#include <math.h>

#include "../utils/ops.hpp"

#define MAX_PIXEL 255


double _mse(const Image& image1, const Image& image2);
double psnr(const Image& image1, const Image& image2);
