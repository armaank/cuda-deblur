/**
 * kernelFunctions.cpp - Functions used by the device (the GPU)
 * in the (obviously) GPU implementaiton of our algorithm.
 */
 
 #include <iostream>


__global__
void kernel_internalMemcpy(double *dest, const double *from, const unsigned W, const unsigned H)
{
	const unsigned start_idx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned stride    = blockDim.x * gridDim.x;
	const unsigned max_val   = 3*H*W;

	for (int idx = start_idx; idx < max_val; idx += stride)
		dest[idx] = from[idx];
}


/**
 * KernelConvolve - Computes the discrete convolution C=Filter*Image.
 * The dimensions of the image and C are both W x H, while the dimension
 * of the filter is f_W x f_H.
 */
__global__
void kernel_convolve(const double *Filter, const double *Image, double *C, const unsigned W, 
					 const unsigned H, const unsigned f_W, const unsigned f_H)
{
	const unsigned start_idx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned stride    = blockDim.x * gridDim.x;
	const unsigned max_val   = 3*H*W;

	// will not execute if start_idx<max_val
	for (int c_idx = start_idx; c_idx < max_val; c_idx += stride)
	{
		C[c_idx] = 0;

		// get the single c_idx term in 2D terms
		unsigned i     = c_idx / (3*W);  // image and filter height index
		unsigned j_tmp = c_idx % (3*W);  // image width index
		unsigned j     = j_tmp/3;        // filter width index
		unsigned pixel = j_tmp%3;

		unsigned n_max = (W < f_W+j) ? W : f_W+j;
		unsigned m_max = (H < f_H+i) ? H : f_H+i;
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
void kernel_elementWiseDivision(const double *A, const double *B, double *C, const unsigned W, const unsigned H)
{
	const unsigned start_idx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned stride    = blockDim.x * gridDim.x;
	const unsigned max_val   = 3*H*W;

	for (int idx=start_idx; idx < max_val; idx += stride)
		C[idx] = (B[idx] == 0) ? 1 : A[idx]/B[idx];
}

/**
 * KernelElementWiseMultiplication - Executes an elementwise multiplication C = A*B.
 * The dimensions of A, B, and C are all assumed to be W x H.
 */
__global__
void kernel_elementWiseMultiplication(const double *A, const double *B, double *C, const unsigned W, const unsigned H)
{
	const unsigned start_idx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned stride    = blockDim.x * gridDim.x;
	const unsigned max_val   = 3*H*W;

	for(int idx = start_idx; idx < max_val; idx += stride)
		C[idx] = A[idx]*B[idx];
}
