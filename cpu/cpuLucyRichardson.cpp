


#include "cpuLucyRichardson.h"


#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <random>


#define MAX_PIXEL 255


/* function to perform lucy richardson deconvolution on the cpu only */
int cpuLucyRichardson(const int W, const int H, const int num_iter, unsigned char *image_input, unsigned char *image_output)
{
    const unsigned img_len = 3*W*H;

    unsigned char *psf  = new(std::nothrow) unsigned char[img_len];
    unsigned char *tmp1 = new(std::nothrow) unsigned char[img_len];
    unsigned char *tmp2 = new(std::nothrow) unsigned char[img_len];
    if (!psf || !tmp1 || !tmp2)
    {
        std::cerr << "Error allocating memory for the point spread function or temporary variables." << std::endl;
        return -1;
    }

    /* initialize the psf and output image as a gaussian */
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(127, 100);
    for (int i=0; i < img_len; )
    {
        float sample = normal_dist(generator);
        if (sample < 0 || sample > MAX_PIXEL)
            continue;

        psf[i++] = static_cast<unsigned char>(std::round(sample)); 
    }
    
    for (int i=0; i < img_len; )
    {
        float sample = normal_dist(generator);
        if (sample < 0 || sample > MAX_PIXEL)
            continue;

        image_output[i++] = static_cast<unsigned char>(std::round(sample)); 
    }

    for (int i=0; i < num_iter; ++i)
        CpuLucyRichIteration(image_input, psf, image_output, tmp1, tmp2, W, H);
    
    return 0;
}


/*
 *   c - The blurred image.
 *   g - The PSF.
 *   f - The underlying image we are trying to recover.
 *   H - Height of the image.
 *   W - Width of the image.
 */
void CpuLucyRichIteration(const unsigned char *c, unsigned char *g, unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const int W, const int H)
{
    updatePSF(c, g, f, tmp1, tmp2, W, H);
    updateUnderlyingImg(c, g, f, tmp1, tmp2, W, H);
}

/**
  * updatePSF - Executes the 'blind iteration' in the Lucy-Richardson
  * algorithm. Updates the values of psf_k.
  */
void updatePSF(const unsigned char *c, unsigned char *g, const unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const int W, const int H)
{
    convolve(g, f, tmp1, W, H);

    elementWiseDivision(c, tmp1, tmp2, W, H);

    convolve(tmp2, f, tmp1, W, H);

    elementWiseMultiplication(tmp1, g, tmp2, W, H);

    memcpy(g, tmp2, 3*H*W*sizeof(unsigned char));
}

/**
  * updateUnderlyingImg - Executes a Lucy-Richardson iteration to get an updated
  * underlying image (updates the values of f).
  */
void updateUnderlyingImg(const unsigned char *c, const unsigned char *g, unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const int W, const int H)
{
    convolve(f, g, tmp1, W, H);

    elementWiseDivision(c, tmp1, tmp2, W, H);

    convolve(tmp2, g, tmp1, W, H);

    elementWiseMultiplication(tmp1, f, tmp2, W, H);

    memcpy(f, tmp2, 3*H*W*sizeof(unsigned char));
}

/**
 * convolve - Computes the discrete convolution C=A*B. 
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
void convolve(const unsigned char *A, const unsigned char *B, unsigned char *C, const int W, const int H)
{
    const int max_val = 3*H*W;

    // will not execute if start_idx<max_val
    for (int c_idx = 0; c_idx < max_val; ++c_idx)
    {
        int cur_val = 0;

        // get the single c_idx term in 2D terms
        int i = c_idx / (3*W);      // i refers to the rows (m)
        int j = c_idx - (3*W*i);    // j refers to the cols (n)

        for (int m = -H; m < H; ++m)
        {
            for (int n = -3*W+(j%3); n < 3*W; n+=3)
            {
                if (i-m>=0 && i-m<H && j-n>=0 && j-n<3*W)
                    cur_val += B[c_idx] * A[ (i - m)*3*W + (j-n) ];
            }
        }

        int tmp = B[c_idx];
        C[c_idx] = (cur_val > 255) ? 255 : cur_val;
    }
}

/** 
 * elementWiseDivision - Executes an elementwise division C = A/B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
void elementWiseDivision(const unsigned char *A, const unsigned char *B, unsigned char *C, const int W, const int H)
{
    const int max_val = 3*H*W;

    // will not execute if start_idx<max_val
    for (int c_idx = 0; c_idx < max_val; ++c_idx)
    {
        if (B[c_idx] == 0)  // not sure if this is technically correct, but for now..
            C[c_idx] = MAX_PIXEL;
        else
            C[c_idx] = A[c_idx] / B[c_idx];

    }
}

/** 
 * elementWiseMultiplication - Executes an elementwise multiplication C = A*B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
void elementWiseMultiplication(const unsigned char *A, const unsigned char *B, unsigned char *C, const int W, const int H)
{
    const int max_val = 3*H*W;

    // will not execute if start_idx<max_val
    for (int c_idx = 0; c_idx < max_val; ++c_idx)
    {
        int cur_val = int(A[c_idx]) * int(B[c_idx]);
        C[c_idx] = cur_val >  255 ? 255 : cur_val;
    }
}
