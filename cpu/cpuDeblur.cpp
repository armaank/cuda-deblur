/* modified lucy-richardson algorithm for image deblurring */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "ops.hpp"
#include "cpuDeblur.hpp"

/* normal lucy-richardson deconvolution algorithm */
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

        Image tmp1 = conv2D(im_deconv, filter); /* convolve target image by psf */

        for (int d = 0; d < 3; d++) /* element-wise division to compute relative blur */
            rel_blur[d] = divide(image[d], tmp1[d]);

        Image tmp2 = conv2D(rel_blur, filter_m); /* filter blur by psf */

        for (int d = 0; d < 3; d++) /* element-wise multiply to update deblurred image */
            im_deconv[d] = multiply(tmp2[d], im_deconv[d]);
    }
   //Matrix filter_sharp = sharpen(3,3);
   //im_deconv = conv(im_deconv, filter_sharp);
   std::cout << "\n";

    return im_deconv;
}

/* image deblurring algorithm */
Image cpuDeblur(const Image &image, const Matrix &filter)
{
	/* perform lucy-richardson deconvolution */
	const int n_iter = 1;
	Image im_deconv = rlDeconv(image, filter, n_iter);
	/* apply sharpening filter */
	Matrix s_filter = sharpen(3,3);
	Image im_deblur = conv2D(im_deconv, s_filter);
	
	return im_deblur;	
}
