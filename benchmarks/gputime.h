/* Benchmarking algorithm performance via GPU time */

/* based off example here: https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc */
#pragma once 


struct GpuTimer
{
	cudaEvent_t start, stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void start()
	{
		cudaEventRecord(start, 0);
	}

	void stop()
	{
		cudaEventRecord(stop, 0);
	}

	float elapsed_time()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEvenElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

