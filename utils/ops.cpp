/**
 * ops.cpp - Implementation file for the commonly used functions and
 * objects by both the GPU and CPU implementations.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "ops.hpp"

/* create an instance of a Matrix class */
Matrix createMatrix(const unsigned height, const unsigned width)
{
	return Matrix(height, Array(width, 0));
}

/* create a Matrix, initilized w/ Gaussian random vars as entries */
Matrix gaussian(const unsigned height, const unsigned width, const double sigma)
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
Matrix sharpen(const unsigned height, const unsigned width)
{
	Matrix kernel = createMatrix(height, width);

	kernel[0][0] =  0;
	kernel[1][0] = -1;
	kernel[2][0] =  0;
	kernel[1][0] = -1;
	kernel[1][1] =  4;
	kernel[1][2] = -1;
	kernel[2][0] =  0;
	kernel[2][1] = -1;
	kernel[2][2] =  0;

	return kernel;
}

/* performs convolution between an Image and a Matrix kernel */
Image conv2D(const Image &image, const Matrix &filter)
{
	assert(image.size() == 3 && filter.size() != 0);

	unsigned i_height = image[0].size();
	unsigned i_width  = image[0][0].size();
	unsigned f_height = filter.size();
	unsigned f_width  = filter[0].size();
	
	Image newImage(3, createMatrix(i_height, i_width));

	for (int d = 0; d < 3; d++)     // iterate over r,b,g
		for (int i = 0; i < i_height; i++)
			for (int j = 0; j < i_width; j++)
			{
				int w_max = std::min<int>(i_width,  f_width+j);
				int h_max = std::min<int>(i_height, f_height+i);
				for (int h = i; h < h_max; h++)
					for (int w = j; w < w_max; w++)
						newImage[d][i][j] += filter[h-i][w-j] * image[d][h][w];
			}

	return newImage;
}

/* perform element-wise multiplication between two Matricies */
Matrix multiply(const Matrix &a, const Matrix &b)
{
	assert(a.size() == b.size() && a[0].size() == b[0].size());

	unsigned height = a.size();
	unsigned width  = a[0].size();
	
	Matrix result = createMatrix(height, width);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			result[i][j] = a[i][j] * b[i][j];

	return result;
}

/* perform element-wise division between two Matricies */
Matrix divide(const Matrix &a, const Matrix &b)
{
	assert(a.size() == b.size() && a[0].size() == b[0].size());

	unsigned height = a.size();
	unsigned width  = a[0].size();

	Matrix result = createMatrix(height, width);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			result[i][j] = b[i][j] == 0 ? 1 : a[i][j] / b[i][j];

	return result;
}

/* flattens an Image object into a 1D dynamically allocated pointer */
double *image2ptr(const Image& input)
{
	unsigned width  = input[0][0].size();
	unsigned height = input[0].size();

	double *ptr = new (std::nothrow) double[3*height*width];

	for (int i = 0, idx = 0; i < height; ++i)
		for (int j =0; j < width; ++j)
		{
			ptr[idx++] = input[0][i][j];
			ptr[idx++] = input[1][i][j];
			ptr[idx++] = input[2][i][j];
		}

	return ptr;
}

/*  flattens a Matrix object into a 1D dynamically allocated pointer */
double *matrix2ptr(const Matrix &input)
{
	unsigned width  = input[0].size();
	unsigned height = input.size();

	double *ptr = new (std::nothrow) double[height*width];

	for (int i = 0, idx = 0; i < height; ++i)
		for (int j =0; j < width; ++j)
			ptr[idx++] = input[i][j];

	return ptr;
}

/* converts a 1D pointer to an Image object with the specified height and width */
Image ptr2image(const double *input, const unsigned width, const unsigned height)
{
	Image output(3, createMatrix(height, width) );

	for (int i = 0, idx = 0; i < height; ++i)
		for (int j = 0 ; j < width; ++j)
		{
			output[0][i][j] = input[idx++];
			output[1][i][j] = input[idx++];
			output[2][i][j] = input[idx++];
		}

	return output;
}
