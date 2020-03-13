

#pragma once

#include <iostream>
#include <math.h>

#include "../cpu/cpuLucyRichardson.hpp"


double _mse(const Image& image1, const Image& image2);
double psnr(const Image& image1, const Image& image2);
