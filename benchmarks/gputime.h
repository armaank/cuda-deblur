/* Benchmarking algorithm performance via GPU time */

/* based off example here: https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc */

struct gpuTimer
{
	cudaEvent_t start, stop;

	gpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~gpuTimer()
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

