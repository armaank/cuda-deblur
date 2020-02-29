

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
 void GpuLucyRichIteration(const float *c, float *g, float *f, float *tmp1, float *tmp2, const uint W, const uint H)
 {
     KernelupdatePSF(c, g, f, tmp1, tmp2, W, H);
     KernelupdateUnderlyingImg(c, g, f, tmp1, tmp2, W, H);
 }
 

 /**
  * KernelupdatePSF - Executes the 'blind iteration' in the Lucy-Richardson
  * algorithm. Updates the values of psf_k.
  */
 void KernelupdatePSF(const float *c, float *g, const float *f, float *tmp1, float *tmp2, const uint W, const uint H)
 {
    int threadsPerBlock = 256;
    int blocksPerGrid =(H*W + threadsPerBlock - 1) / threadsPerBlock;
    KernelConvolve<<<blocksPerGrid, threadsPerBlock>>>(g, f, tmp1, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelConvolve kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    KernelElementWiseDivision<<<blocksPerGrid, threadsPerBlock>>>(c, tmp1, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelElementWiseDivision kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    KernelConvolve<<<blocksPerGrid, threadsPerBlock>>>(tmp2, f, tmp1, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelConvolve kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    KernelElementWiseMultiplication<<<blocksPerGrid, threadsPerBlock>>>(tmp1, g, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelElementWiseMultiplication kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    KernelInternalMemcpy<<<blocksPerGrid, threadsPerBlock>>>(g, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelInternalMemcpy kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
 
 
 /**
  * KernelupdateUnderlyingImg - Executes a Lucy-Richardson iteration to get an updated
  * underlying image (updates the values of f).
  */
void KernelupdateUnderlyingImg(const float *c, const float *g, float *f, float *tmp1, float *tmp2, const uint W, const uint H)
{
    int threadsPerBlock = 256;
    int blocksPerGrid =(H*W + threadsPerBlock - 1) / threadsPerBlock;
    KernelConvolve<<<blocksPerGrid, threadsPerBlock>>>(f, g, tmp1, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelConvolve kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    KernelElementWiseDivision<<<blocksPerGrid, threadsPerBlock>>>(c, tmp1, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelElementWiseDivision kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    KernelConvolve<<<blocksPerGrid, threadsPerBlock>>>(tmp2, g, tmp1, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelConvolve kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    KernelElementWiseMultiplication<<<blocksPerGrid, threadsPerBlock>>>(tmp1, f, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelElementWiseMultiplication kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    KernelInternalMemcpy<<<blocksPerGrid, threadsPerBlock>>>(f, tmp2, W, H);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch KernelInternalMemcpy kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Kernel Functions ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////


__global__
void KernelInternalMemcpy(float *to,const float *from, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = H*W;
    
    for (int idx = start_idx; c_idx < max_val; c_idx += stride) 
    {
        ti[idx] = from[idx];
    }
}


/**
 * KernelConvolve - Computes the discrete convolution C=A*B. 
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
__global__
void KernelConvolve(const float *A, const float *B, float *C, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = H*W;

    // will not execute if start_idx<max_val
    for (int c_idx=start_idx; c_idx<max_val; c_idx += stride) 
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
 * KernelElementWiseDivision - Executes an elementwise division C = A/B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
__global__
void KernelElementWiseDivision(const float *A, const float *B, float *C, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = H*W;

    // will not execute if start_idx<max_val
    for (int c_idx=start_idx; c_idx<max_val; c_idx += stride) 
    {
        C[c_idx] = A[c_idx]/B[c_idx];
    }
}

/** 
 * KernelElementWiseMultiplication - Executes an elementwise multiplication C = A*B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
__global__
void KernelElementWiseMultiplication(const float *A, const float *B, float *C, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = H*W;

    // will not execute if start_idx<max_val
    for(int c_idx=start_idx; c_idx<max_val; c_idx += stride) 
    {
        C[c_idx] = A[c_idx]*B[c_idx];
    }
}

