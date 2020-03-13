
#pragma once


#include <vector>

using Array  = std::vector<double>;
using Matrix = std::vector<Array>;
using Image  = std::vector<Matrix>;


Matrix createMatrix(const int height, const int width);
Matrix gaussian(const int height, const int width, const double sigma);
Image  conv(const Image &image, const Matrix &filter);
Matrix multiply(const Matrix &a, const Matrix &b);
Matrix divide(const Matrix &a, const Matrix &b);
Image  rlDeconv(const Image &image, const Matrix &filter, const int n_iter);