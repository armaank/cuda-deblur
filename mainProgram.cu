

#include "gpuLucyRichardson.cu"
#include "cpuLucyRichardson.cu"


#define NUM_ITERATIONS 100


int main(void)
{
    const uint H = 50000;
    const uint W = 50000;
    const uint num_elements = H*W;
    const uint size = sizeof(float)*num_elements;

    /* TODO: initialize h_f, h_g, h_c and read in an image value for h_c. h_g should be a gaussian, 
             and h_f I guess could be garbage. maybe a gaussian also?  */


    ///////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////// CPU Run ////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    float *h_tmp1 = (float *)malloc(size);
    float *h_tmp2 = (float *)malloc(size);
    if (h_tmp1==NULL || h_tmp2==NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // TODO: Time the devonvolution in C
    for (int i=0; i<NUM_ITERATIONS; ++i)
    {   
        CpuLucyRichIteration(h_c, h_g, h_f, h_tmp1, h_tmp2, W, H);
    }

    // TODO: run some evaluation metric


    ///////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////// GPU Run ////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the device input vector f
    float *d_f = NULL;
    err = cudaMalloc((void **)&d_f, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector f (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector g
    float *d_g = NULL;
    err = cudaMalloc((void **)&d_g, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector c
    float *d_c = NULL;
    err = cudaMalloc((void **)&d_c, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors f, g, and c in host memory to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_f, h_f, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector f from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector c from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // allocate the temporary gpu memory
    float *d_tmp1 = NULL;
    err = cudaMalloc((void **)&d_tmp1, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector tmp1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_tmp2 = NULL;
    err = cudaMalloc((void **)&d_tmp2, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector tmp2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // TODO: time this loop, i guess?
    for (int i=0; i<NUM_ITERATIONS; ++i)
    {
        GpuLucyRichIteration(d_c, d_g, d_f, d_tmp1, d_tmp2, W, H);
    }


    // Copy the device result vector in device memory to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_f, d_f, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector f from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // TODO: run some evaluation metric


    // Free device global memory
    err = cudaFree(d_f);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector f (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_c);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_g);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_tmp1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector tmp1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_tmp2;
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector tmp2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // TODO: Free host memory

    printf("Done\n");
    return 0;

}