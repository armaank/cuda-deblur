   
#include <iostream>


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Kernel Functions ////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

__global__
void kernel_internalMemcpy(double *dest, const double *from, const uint W, const uint H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = 3*H*W;
    
    for (int idx = start_idx; idx < max_val; idx += stride) 
        dest[idx] = from[idx];
}


/**
 * KernelConvolve - Computes the discrete convolution C=Filter*Image. 
 * The dimensions of the image and C are both W x H, while the dimension 
 * of the filter is f_W x f_H.
 */
__global__
void kernel_convolve(const double *Filter, const double *Image, double *C, const uint W, const uint H, const uint f_W, const uint f_H)
{
    const int start_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride    = blockDim.x * gridDim.x;
    const int max_val   = 3*H*W;

    // will not execute if start_idx<max_val
    for (int c_idx = start_idx; c_idx < max_val; c_idx += stride) 
    {
        C[c_idx] = 0;

        // get the single c_idx term in 2D terms
        int i     = c_idx / (3*W);  // image and filter height index
        int j_tmp = c_idx % (3*W);  // image width index 
        int j     = j_tmp/3;        // filter width index
        int pixel = j_tmp%3;

        int n_max = (W < f_W+j) ? W : f_W+j;
        int m_max = (H < f_H+i) ? H : f_H+i;
        for (int m=i; m<m_max; ++m)
            for (int n=j; n<n_max; ++n)
                C[c_idx] += Filter[ (m-i)*f_W+(n-j) ] * Image[m*3*W + pixel+3*n];
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

    for (int idx=start_idx; idx < max_val; idx += stride) 
        C[idx] = B[idx]==0 ? 1 : A[idx]/B[idx];
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

    for(int idx = start_idx; idx < max_val; idx += stride) 
        C[idx] = A[idx]*B[idx];
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
/*
  void updateUnderlyingImg(const double *c, const double *g, const double *g_m, double *f, double *tmp1, double *tmp2, double *tmp3,
     const uint W, const uint H, const uint filter_W, const uint filter_H, const double *s, const uint s_filter_W, const uint s_filter_H)
  {
      cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

      int threadsPerBlock = 512;
      int blocksPerGrid = 10;

      kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(g, f, tmp1, W, H, filter_W, filter_H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }

      kernel_elementWiseDivision<<<blocksPerGrid, threadsPerBlock>>>(c, tmp1, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_elementWiseDivision kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }

      kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(g_m, tmp2, tmp1, W, H, filter_W, filter_H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }

	 kernel_elementWiseMultiplication<<<blocksPerGrid, threadsPerBlock>>>(tmp1, f, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_elementWiseMultiplication kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
          exit(EXIT_FAILURE);
      }


      kernel_internalMemcpy<<<blocksPerGrid, threadsPerBlock>>>(f, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_internalMemcpy kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
          exit(EXIT_FAILURE);
      }
  }



      kernel_elementWiseMultiplication<<<blocksPerGrid, threadsPerBlock>>>(tmp1, f, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_elementWiseMultiplication kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
          exit(EXIT_FAILURE);
      }

      kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(tmp2, s, tmp1, W, H, s_filter_W, s_filter_H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
          exit(EXIT_FAILURE);
      }
      kernel_internalMemcpy<<<blocksPerGrid, threadsPerBlock>>>(f, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_internalMemcpy kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
          exit(EXIT_FAILURE);
      }
  }
*/
 void updateUnderlyingImg(const double *c, const double *g, const double *g_m, double *f, double *tmp1, double *tmp2, double *tmp3,
     const uint W, const uint H, const uint filter_W, const uint filter_H, const double *s, const uint s_filter_W, const uint s_filter_H)
  {
      cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

      int threadsPerBlock = 512;
      int blocksPerGrid = 10;

      kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(g, f, tmp1, W, H, filter_W, filter_H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }

      kernel_elementWiseDivision<<<blocksPerGrid, threadsPerBlock>>>(c, tmp1, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_elementWiseDivision kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }

      kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(g_m, tmp2, tmp1, W, H, filter_W, filter_H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }

      kernel_elementWiseMultiplication<<<blocksPerGrid, threadsPerBlock>>>(tmp1, f, tmp2, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_elementWiseMultiplication kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
          exit(EXIT_FAILURE);
      }
     
      kernel_convolve<<<blocksPerGrid, threadsPerBlock>>>(s, tmp2, tmp3, W, H, s_filter_W, s_filter_H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_convolve kernel (error code " << cudaGetErrorString(err) << ")!\n";
          exit(EXIT_FAILURE);
      }


      kernel_internalMemcpy<<<blocksPerGrid, threadsPerBlock>>>(f, tmp3, W, H);
      err = cudaGetLastError();
      if (err != cudaSuccess)
      {
          std::cerr << "Failed to launch kernel_internalMemcpy kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
          exit(EXIT_FAILURE);
      }
  }


