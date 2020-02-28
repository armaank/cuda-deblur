#include <iostream>
#include <math.h>
#include "im_metrics.h"

/* maximum pixel value */ 
#define MAX 255



/* computes mean squared error between to images */
double _mse(unsigned char *im1, unsigned char *im2, int *imsize)
{
	double mse_temp = 0.0;
	for(int ii=0; ii < *imsize; ii++)
	{
		mse_temp += pow((im1[ii]-im2[ii], 2);
	}
	double mse = mse_temp / (double)*imsize;

	return mse;
}

/* computes peak signal-to-noise ratio */
double psnr(*unsigned char *im1, unsigned char *im2, int *imsize)
{	
	/* compute mse */
	mse = _mse(im1, im2, imsize)
	/* from mse, compute psnr */
	psnr = 10*log10((pow(MAX), 2)/ mse);

	return psnr;
}

