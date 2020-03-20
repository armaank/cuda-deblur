#include <vector>
#include <iostream>

#include "cpuFunctions.cu"

#include "../utils/pngConnector.hpp"


int main(int argc, char **argv)
{
	if (argc <= 3)
	{
		std::cerr << "error: specify input and output files" << std::endl;
		return -1;
	}
	std::string input_file  = argv[1];  // blurry image
	std::string output_file = argv[2];  // deblurred image
	std::string target_file = argv[3];  // target image 'ground truth'

	std::cout << "Loading image from" << input_file << std::endl;
	Image image = loadImage(input_file);
	Image target_image = loadImage(target_file);

	double *image_ptr  = image2ptr(image);
	double *output_ptr = new (std::nothrow) double[3*image[0].size()*image[0][0].size()];


	/////////////////////////////////////////////////////////////////////////
	// preparing filters
	Matrix filter = gaussian(3, 3, 1);
	int filter_width  = filter[0].size();
	int filter_height = filter.size();
	Matrix filter_m = createMatrix(filter_height, filter_width);
	for (int i = 0; i < filter_height; i++)
		for (int j = 0; j < filter_width; j++)
			filter_m[i][j] = filter[j][i];

	Matrix s_filter     = sharpen(3,3);
	int s_filter_width  = s_filter[0].size();
	int s_filter_height = s_filter.size();

	// getting filter pointers
	double *filter_ptr        = matrix2ptr(filter);
	double *filter_mirror_ptr = matrix2ptr(filter_m);
	double *s_filter_ptr      = matrix2ptr(s_filter);

	deblurImage(filter_ptr, 
				filter_mirror_ptr,
				filter_width,
				filter_height, 
				image_ptr, 
				output_ptr,
				target_image,
				s_filter_ptr,
				s_filter_width,
				s_filter_height,
				output_file + "_gaussKernel3"+".png");

	std::cout << "Done!" << std::endl;
	/////////////////////////////////////////////////////////////////////////

	return 0;
}