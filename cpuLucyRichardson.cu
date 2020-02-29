

/** 
 * LucyRichIteration - A single update of the point spread function (PSF) and
 * the underlying image. 
 * Input variables are based on: 
 * https://pdfs.semanticscholar.org/9e3f/a71e22caf358dbe873e9649f08c205d0c0c0.pdf.
 * They are as follows:
 *   c - The blurred image.
 *   g - The PSF.
 *   f - The underlying image we are trying to recover.
 *   H - Height of the image.
 *   W - Width of the image.
 */
 void CpuLucyRichIteration(const float *c, float *g, float *f, float *tmp1, float *tmp2, const uint W, const uint H)
 {
     updatePSF(c, g, f, tmp1, tmp2, W, H);
     updateUnderlyingImg(c, g, f, tmp1, tmp2, W, H);
 }
 

 /**
  * updatePSF - Executes the 'blind iteration' in the Lucy-Richardson
  * algorithm. Updates the values of psf_k.
  */
 void updatePSF(const float *c, float *g, const float *f, float *tmp1, float *tmp2, const uint W, const uint H)
 {
    // Launch the Vector Add CUDA Kernel
    convolve(g, f, tmp1, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convolve kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    elementWiseDivision(c, tmp1, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch elementWiseDivision kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    convolve(tmp2, f, tmp1, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convolve kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    elementWiseMultiplication(tmp1, g, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch elementWiseMultiplication kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    memcpy(g, tmp2, H*W*sizeof(float));
 }
 
 
 /**
  * updateUnderlyingImg - Executes a Lucy-Richardson iteration to get an updated
  * underlying image (updates the values of f).
  */
void updateUnderlyingImg(const float *c, const float *g, float *f, float *tmp1, float *tmp2, const uint W, const uint H)
{
    convolve(f, g, tmp1, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convolve kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    elementWiseDivision(c, tmp1, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch elementWiseDivision kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    convolve(tmp2, g, tmp1, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convolve kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    elementWiseMultiplication(tmp1, f, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch elementWiseMultiplication kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    memcpy(f, tmp2, H*W*sizeof(float));
}


/**
 * convolve - Computes the discrete convolution C=A*B. 
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
void convolve(const float *A, const float *B, float *C, const uint W, const uint H)
{
    const int max_val = H*W;

    // will not execute if start_idx<max_val
    for (int c_idx = 0; c_idx < max_val; c_idx += 1) 
    {
        C[c_idx] = 0;

        // get the single c_idx term in 2D terms
        int i = c_idx % W;
        int j = c_idx - W*i;

        for (int m=0; m<H; ++m)
        {
            for (int n=0; j<W; ++n)
            {
                int cur_idx = m*W + n;
                C[c_idx] += A[cur_idx] * B[ (i-m)*W+(j-n) ];
            }
        }
    }
}


/** 
 * elementWiseDivision - Executes an elementwise division C = A/B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
void elementWiseDivision(const float *A, const float *B, float *C, const uint W, const uint H)
{
    const int max_val = H*W;

    // will not execute if start_idx<max_val
    for (int c_idx = 0; c_idx < max_val; c_idx += 1) 
    {
        C[c_idx] = A[c_idx]/B[c_idx];
    }
}


/** 
 * elementWiseMultiplication - Executes an elementwise multiplication C = A*B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
void elementWiseMultiplication(const float *A, const float *B, float *C, const uint W, const uint H)
{
    const int max_val   = H*W;
 
    // will not execute if start_idx<max_val
    for (int c_idx = 0; c_idx < max_val; c_idx += 1) 
    {
        C[c_idx] = A[c_idx]*B[c_idx];
    }
}
 
