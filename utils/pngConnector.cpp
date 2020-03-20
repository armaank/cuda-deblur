/**
 * pngConnector.cpp - Implementation file for the functions 
 * that call the png++ wrapper.
 */

#include <vector>
#include <iostream>
#include <png++/png.hpp>

#include "ops.hpp"
#include "pngConnector.hpp"

/* loads a png image file and saves it as an Image object */
Image loadImage(const std::string &filename)
{
	png::image<png::rgb_pixel> in_image(filename);

	unsigned height = in_image.get_height();
	unsigned width  = in_image.get_width();

	Image imageMatrix(3, Matrix(height, Array(width)));

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			imageMatrix[0][i][j] = in_image[i][j].red;
			imageMatrix[1][i][j] = in_image[i][j].green;
			imageMatrix[2][i][j] = in_image[i][j].blue;
		}

	return imageMatrix;
}

/* saves an Image object as a png file with the inputted file name */
void saveImage(const Image &image, const std::string &filename)
{
	assert(image.size() == 3);

	unsigned height = image[0].size();
	unsigned width  = image[0][0].size();

	png::image<png::rgb_pixel> imageFile(width, height);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			imageFile[i][j].red   = image[0][i][j];
			imageFile[i][j].green = image[1][i][j];
			imageFile[i][j].blue  = image[2][i][j];
		}

	imageFile.write(filename);
}