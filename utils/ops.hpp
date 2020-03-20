/**
 * ops.hpp - Header file for the commonly used functions and
 * objects by both the GPU and CPU implementations.
 */

#pragma once

#include <vector>


using Array = std::vector<double>;
using Matrix = std::vector<Array>;
using Image = std::vector<Matrix>;

Matrix  createMatrix(const unsigned height, const unsigned width);
Matrix  gaussian(const unsigned height, const unsigned width, const double sigma);
Matrix  sharpen(const unsigned height, const unsigned width);
Image   conv2D(const Image &image, const Matrix &filter);
Matrix  multiply(const Matrix &a, const Matrix &b);
Matrix  divide(const Matrix &a, const Matrix &b);
double *image2ptr(const Image &input);
double *matrix2ptr(const Matrix &input);
Image   ptr2image(const double *input, const unsigned width, const unsigned height);