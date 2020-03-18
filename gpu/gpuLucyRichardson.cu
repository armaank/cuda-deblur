   
#include <iostream>


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Kernel Functions ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

__global__
void kernel_internalMemcpy(double *dest, const double *from, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = H*W;
    
    for (int idx = start_idx; idx < max_val; idx += stride) 
        dest[idx] = from[idx];
}


/**
 * KernelConvolve - Computes the discrete convolution C=A*B. 
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
__global__
void kernel_convolve(const double *A, const double *B, double *C, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = 3*H*W;

    // will not execute if start_idx<max_val
    for (int c_idx=start_idx; c_idx<max_val; c_idx += stride) 
    {
        C[c_idx] = 0;

        // get the single c_idx term in 2D terms
        int i = c_idx % (3*W);
        int j = c_idx / W;

        for (int m=0; m<H; ++m)
            for (int n=j%3; j<3*W; n+=3)
                C[c_idx] += A[m*3*W + n] * B[ (i-m)*3*W+(j-n) ];
    }
}


/** 
 * KernelElementWiseDivision - Executes an elementwise division C = A/B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
__global__
void kernel_elementWiseDivision(const double *A, const double *B, double *C, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = 3*H*W;

    // will not execute if start_idx<max_val
    for (int c_idx=start_idx; c_idx<max_val; c_idx += stride) 
    {
        C[c_idx] = B[c_idx]==0 ? 999999999 : A[c_idx]/B[c_idx];
    }
}

/** 
 * KernelElementWiseMultiplication - Executes an elementwise multiplication C = A*B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
__global__
void kernel_elementWiseMultiplication(const double *A, const double *B, double *C, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = 3*H*W;

    // will not execute if start_idx > max_val
    for(int c_idx=start_idx; c_idx<max_val; c_idx += stride) 
    {
        C[c_idx] = A[c_idx]*B[c_idx];
    }
}


/////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// CPU Functions ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

/**
  * updateUnderlyingImg - Executes a Lucy-Richardson iteration to get an updated
  * underlying image (updates the values of f).
  *   c - The blurred image.
  *   g - The PSF.
  *   f - The underlying image we are trying to recover.
  *   H - Height of the image.
  *   W - Width of the image.
  */
  void updateUnderlyingImg(const double *c, const double *g, const double *g_m, double *f, double *tmp1, double *tmp2, const uint W, const uint H)
  {
      cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

      int threadsPerBlock = 256;
      int blocksPerGrid =(H*W + threadsPerBlock - 1) / threadsPerBlock;
  
      kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(f, g, tmp1, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch KernelConvolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }
  
      kernel_elementWiseDivision<<<blocksPerGrid, threadsPerBlock>>>(c, tmp1, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch KernelElementWiseDivision kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }
  
      kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(tmp2, g_m, tmp1, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch KernelConvolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }
  
      kernel_elementWiseMultiplication<<<blocksPerGrid, threadsPerBlock>>>(tmp1, f, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to launch KernelElementWiseMultiplication kernel (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
  
      kernel_internalMemcpy<<<blocksPerGrid, threadsPerBlock>>>(f, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to launch KernelInternalMemcpy kernel (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }

  }