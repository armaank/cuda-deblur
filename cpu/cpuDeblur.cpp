/**
 * cpuDeblur.cpp - Implementation file for the CPU deblurring functions.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "cpuDeblur.hpp"
#include "../utils/ops.hpp"


/* normal lucy-richardson deconvolution algorithm */
Image rlDeconv(const Image &image, const Matrix &filter, const int n_iter)
{
	Image im_deconv = image;
	Image rel_blur  = image;

	int f_length = filter.size();
	int f_width  = filter[0].size();

	/* compute and store mirrored psf */
	Matrix filter_m(f_length, Array(f_width));

	for (int i = 0; i < f_length; i++)
		for (int j = 0; j < f_width; j++)
			filter_m[i][j] = filter[j][i];

	/* perform lucy iterations */
	for (int i = 0; i < n_iter; i++)
	{
		Image tmp1 = conv2D(im_deconv, filter); /* convolve target image by psf */

		for (int d = 0; d < 3; d++)             /* element-wise division to compute relative blur */
			rel_blur[d] = divide(image[d], tmp1[d]);

		Image tmp2 = conv2D(rel_blur, filter_m); /* filter blur by psf */

		for (int d = 0; d < 3; d++)             /* element-wise multiply to update deblurred image */
			im_deconv[d] = multiply(tmp2[d], im_deconv[d]);
	}

	return im_deconv;
}

/* CPU implementation of our image deblurring algorithm */
Image cpuDeblur(const Image &image, const Matrix &filter, const unsigned n_iter)
{
	/* perform lucy-richardson deconvolution */
	Image im_deconv = rlDeconv(image, filter, n_iter);

	/* apply sharpening filter */
	Matrix s_filter = sharpen(3,3);
	Image im_deblur = conv2D(im_deconv, s_filter);

	return im_deblur;
}
