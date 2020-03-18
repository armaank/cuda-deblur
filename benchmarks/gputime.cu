/* Benchmarking algorithm performance via GPU time */

/* based off example here: https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc */

struct GpuTimer
{
	cudaEvent_t start_val, stop_val;

	GpuTimer()
	{
		cudaEventCreate(&start_val);
		cudaEventCreate(&stop_val);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start_val);
		cudaEventDestroy(stop_val);
	}

	void start()
	{
		cudaEventRecord(start_val, 0);
	}

	void stop()
	{
		cudaEventRecord(stop_val, 0);
	}

	float elapsed_time()
	{
		float elapsed;
		cudaEventSynchronize(stop_val);
		cudaEventElapsedTime(&elapsed, start_val, stop_val);
		return elapsed;
	}
};

